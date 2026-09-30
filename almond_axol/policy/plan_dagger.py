"""Nonblocking DAgger backend for the custom policy interface.

The collector owns command arbitration, pacing, execution filtering and
recording. This backend owns observations and one serialized v2 RPC stream.
Takeover invalidates local work immediately, even while inference is blocked;
desktop resets run only after that call drains. Shadow predictions during
teleop are never accepted, dispatched, or reused on handback.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .plan_client import PlanPolicyClient
from .plan_protocol import Continuation, LastDispatched, PlanObservation, PlanSpec
from .plan_scheduler import (
    PlanRuntimeConfig,
    PlanScheduler,
    PlanSchedulingError,
    SensorTimingError,
    prepare_plan_images,
    validate_sensor_times,
)
from .protocol import CameraSpec


class PlanDaggerPolicy:
    """Run policy and shadow inference without blocking the control thread.

    ``connect`` negotiates declared features without reading or connecting
    hardware. ``reset`` activates policy inference; ``pause`` deactivates it.
    The single control owner must call ``note_dispatched`` after each returned
    action has been sent successfully, before calling ``act`` again. That
    acknowledgement refers to the original published row, before any generic
    execution filter, and does not claim a hardware acknowledgement.
    """

    asynchronous_observations = True

    def __init__(
        self,
        url: str,
        *,
        fps: int = 30,
        horizon: int = 30,
        config: PlanRuntimeConfig | None = None,
        shadow_inference: bool = True,
    ) -> None:
        self.config = config or PlanRuntimeConfig()
        self.config.validate(fps=fps, horizon=horizon)
        self.fps, self.horizon = fps, horizon
        self.shadow_inference = shadow_inference
        self._transport = PlanPolicyClient(
            url, reply_timeout=self.config.reply_timeout_s
        )
        self._ready = threading.Condition(threading.RLock())
        self._closed = False
        self._mode = "inactive"
        self._robot: Any = None
        self._spec: PlanSpec | None = None
        self._scheduler: PlanScheduler | None = None
        self._worker: threading.Thread | None = None
        self._fatal_error: BaseException | None = None
        self._last_dispatched: tuple[int, LastDispatched] | None = None
        self._reserved: tuple[int, LastDispatched] | None = None
        self._last_target: tuple[int, np.ndarray] | None = None
        self._reserved_target: np.ndarray | None = None
        self._next_shadow_at = 0.0
        self._instruction = ""

    def connect(self, robot: Any) -> None:
        """Validate the declared robot schema and connect, without sensor I/O."""
        if self._worker is not None or self._closed:
            raise RuntimeError("PlanDaggerPolicy instances connect only once")
        states: list[str] = []
        cameras: list[CameraSpec] = []
        for name, feature in robot.observation_features.items():
            if feature is float:
                states.append(name)
            elif isinstance(feature, tuple) and len(feature) == 3:
                shape = feature
                if self.config.output_width:
                    shape = (self.config.output_height, self.config.output_width, 3)
                cameras.append(CameraSpec(name, shape))
            else:
                raise ValueError(
                    f"Unsupported observation feature {name!r}: {feature!r}"
                )
        if any(feature is not float for feature in robot.action_features.values()):
            raise ValueError(
                "DAgger through the custom policy interface requires scalar float action features"
            )
        spec = PlanSpec(
            state_names=tuple(states),
            action_names=tuple(robot.action_features),
            cameras=tuple(cameras),
            fps=self.fps,
            actions_per_chunk=self.horizon,
            request_interval=self.config.request_interval,
            max_adoption_offset_steps=self.config.max_adoption_offset_steps,
            dispatch_feedback=True,
        )
        self._transport.connect(spec)
        self._spec = spec
        self._robot = robot
        self._scheduler = PlanScheduler(
            fps=self.fps,
            horizon=self.horizon,
            width=len(spec.action_names),
            config=self.config,
        )
        self._worker = threading.Thread(
            target=self._run, name="axol-dagger-plan", daemon=True
        )
        self._worker.start()

    def _change_mode(self, mode: str) -> None:
        with self._ready:
            if self._closed or self._scheduler is None:
                raise RuntimeError(
                    "Custom policy interface is not connected for DAgger"
                )
            self._scheduler.reset()
            self._mode = mode
            self._last_dispatched = self._reserved = None
            self._last_target = self._reserved_target = None
            self._next_shadow_at = 0.0
            self._ready.notify_all()

    def reset(self) -> None:
        """Start a fresh policy generation locally; never wait for the GPU."""
        self.check_error()
        self._change_mode("policy")

    def set_intervention(self, active: bool) -> None:
        """Invalidate queued and in-flight work on either ownership transition."""
        mode = "teleop" if active else "policy"
        with self._ready:
            if self._mode != mode:
                self._change_mode(mode)

    def pause(self) -> None:
        """Stop observation acquisition and invalidate policy work locally."""
        with self._ready:
            if not self._closed and self._scheduler is not None:
                self._change_mode("inactive")

    def set_instruction(self, text: str) -> None:
        """Retain the recording label; task selection belongs to the desktop."""
        self._instruction = text

    def check_error(self) -> None:
        """Surface background failures, including while teleop owns control."""
        with self._ready:
            if self._fatal_error is not None:
                raise RuntimeError("Remote DAgger policy failed") from self._fatal_error

    def act(self, observation: Any = None) -> dict[str, float] | None:
        """Reserve one unchanged row; fresh observations are worker-owned."""
        self.check_error()
        with self._ready:
            scheduler, spec = self._scheduler, self._spec
            if self._closed or self._mode != "policy" or scheduler is None:
                return None
            if self._reserved is not None:
                raise RuntimeError("note_dispatched() must follow a successful send")
            generation = scheduler.generation
            prediction_id = scheduler.prediction_id
            row = scheduler.next_tick - scheduler.origin_tick
            if scheduler.actions is not None and 0 <= row < len(scheduler.actions):
                self._check_step(scheduler.actions[row], generation)
            popped = scheduler.pop()
            if generation != scheduler.generation:
                self._last_dispatched = None
                self._last_target = None
            self._ready.notify_all()
            if popped is None:
                return None
            assert prediction_id is not None and spec is not None
            self._reserved = (generation, LastDispatched(prediction_id, row))
            self._reserved_target = popped[1]
            return dict(zip(spec.action_names, map(float, popped[1]), strict=True))

    def _check_step(self, target: np.ndarray, generation: int) -> None:
        """Bound consecutive published Cartesian targets within one ownership.

        The first policy command after takeover is anchored by the collector's
        downstream joint handover. A subsequent plan must not silently bypass
        the same Cartesian-step guard used by the ordinary v2 dispatcher.
        """
        previous = self._last_target
        if previous is None or previous[0] != generation:
            return
        assert self._spec is not None
        names = self._spec.action_names
        for side in ("left", "right"):
            keys = [f"{side}_ee.{axis}" for axis in ("x", "y", "z")]
            if all(key in names for key in keys):
                indices = [names.index(key) for key in keys]
                step = float(np.linalg.norm(target[indices] - previous[1][indices]))
                if step > self.config.max_cartesian_step_m:
                    raise PlanSchedulingError(
                        f"{side} Cartesian step {step:.4f}m exceeds limit"
                    )

    def note_dispatched(self) -> None:
        """Confirm the reserved prefilter row after successful local dispatch."""
        with self._ready:
            reserved, self._reserved = self._reserved, None
            target, self._reserved_target = self._reserved_target, None
            if (
                reserved is not None
                and not self._closed
                and self._mode == "policy"
                and self._scheduler is not None
                and reserved[0] == self._scheduler.generation
            ):
                self._last_dispatched = reserved
                if target is not None:
                    self._last_target = (reserved[0], target)
            self._ready.notify_all()

    def note_control_overrun(
        self, reason: str = "policy dispatch missed its scheduled tick"
    ) -> None:
        """Hold/replan after a deadline missed by the collector's pacing loop.

        Cached rows cannot be shifted into later execution slots. The control
        owner detects missed deadlines; this method invalidates those rows
        locally and uses the same configured recovery as a late model reply.
        """
        with self._ready:
            if (
                not self._closed
                and self._mode == "policy"
                and self._scheduler is not None
            ):
                self._last_dispatched = self._reserved = None
                self._last_target = self._reserved_target = None
                try:
                    self._scheduler.recover(reason)
                finally:
                    self._ready.notify_all()

    def _current(self, generation: int, mode: str) -> bool:
        return (
            not self._closed
            and self._scheduler is not None
            and self._scheduler.generation == generation
            and self._mode == mode
        )

    def _capture(self, generation: int, mode: str, startup: bool):
        # Import only when acquisition starts: connect is a hardware-free
        # schema preflight, and this module remains importable without LeRobot.
        from ..lerobot.robot.robot_axol import PolicyObservationNotReady

        deadline = time.perf_counter() + self.config.startup_observation_timeout_s
        while True:
            with self._ready:
                if not self._current(generation, mode):
                    return None
            try:
                raw, state_ns, camera_ns = (
                    self._robot.get_observation_with_sensor_timestamps()
                )
                now_ns = time.perf_counter_ns()
                assert self._spec is not None
                if set(camera_ns) != set(self._spec.camera_names):
                    raise ValueError(
                        "Camera timestamps do not match negotiated cameras"
                    )
                validate_sensor_times(state_ns, camera_ns, now_ns, self.config)
                return raw, state_ns, camera_ns, now_ns
            except (PolicyObservationNotReady, SensorTimingError) as exc:
                if not startup or (
                    isinstance(exc, SensorTimingError) and not exc.retriable
                ):
                    raise
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for fresh DAgger sensors"
                    ) from exc
                with self._ready:
                    if self._current(generation, mode):
                        self._ready.wait(timeout=min(0.01, remaining))

    def _run(self) -> None:
        wire_generation: int | None = None
        acquired_generation: int | None = None
        episode = 0
        try:
            while True:
                with self._ready:
                    scheduler, spec = self._scheduler, self._spec
                    assert scheduler is not None and spec is not None
                    if self._closed:
                        return
                    mode = self._mode
                    shadow_due = (
                        mode == "teleop"
                        and self.shadow_inference
                        and time.perf_counter() >= self._next_shadow_at
                    )
                    if not ((mode == "policy" and scheduler.request_due) or shadow_due):
                        self._ready.wait(timeout=1 / self.fps)
                        continue
                    generation = scheduler.generation
                # There is only one network owner. A generation boundary may
                # arrive during either call, and is checked after each one.
                if wire_generation != generation:
                    episode += 1
                    self._transport.reset(episode)
                    wire_generation = generation
                with self._ready:
                    if not self._current(generation, mode):
                        continue
                captured = self._capture(
                    generation, mode, startup=acquired_generation != generation
                )
                if captured is None:
                    continue
                raw, state_ns, camera_ns, now_ns = captured
                acquired_generation = generation
                with self._ready:
                    if not self._current(generation, mode) or not scheduler.request_due:
                        continue
                    pending = scheduler.begin_request(now_ns)
                    dispatched = self._last_dispatched
                    last_dispatched = (
                        dispatched[1]
                        if mode == "policy"
                        and dispatched is not None
                        and dispatched[0] == generation
                        else None
                    )
                    delay = (
                        scheduler.delay_steps if self.config.advertise_delay else None
                    )
                images = prepare_plan_images(
                    {name: raw[name] for name in spec.camera_names}, self.config
                )
                observation = PlanObservation(
                    request_id=pending.request_id,
                    state=np.asarray(
                        [raw[name] for name in spec.state_names], dtype=np.float32
                    ),
                    images=images,
                    state_sample_time_ns=state_ns,
                    image_capture_time_ns=camera_ns,
                    continuation=(
                        Continuation(pending.prediction_id, pending.from_row)
                        if mode == "policy" and pending.prediction_id is not None
                        else None
                    ),
                    delay_steps=delay,
                    last_dispatched=last_dispatched,
                )
                with self._ready:
                    if not self._current(generation, mode):
                        continue
                try:
                    validate_sensor_times(
                        state_ns, camera_ns, time.perf_counter_ns(), self.config
                    )
                except SensorTimingError as exc:
                    if not exc.retriable:
                        raise
                    with self._ready:
                        scheduler.cancel_unsent(pending.request_id)
                    continue
                with self._ready:
                    if not self._current(generation, mode):
                        continue
                    if mode == "teleop":
                        self._next_shadow_at = (
                            time.perf_counter()
                            + self.config.request_interval / self.fps
                        )
                reply = self._transport.infer(observation)
                with self._ready:
                    if not self._current(generation, mode):
                        continue
                    if mode == "policy":
                        scheduler.adopt(
                            reply.request_id,
                            reply.actions,
                            time.perf_counter_ns(),
                            reply.max_adoption_offset_steps,
                        )
                        if scheduler.generation != generation:
                            self._last_dispatched = None
                            self._last_target = None
                    else:
                        scheduler.cancel_unsent(pending.request_id)
                    self._ready.notify_all()
        except Exception as exc:
            with self._ready:
                if not self._closed:
                    self._fatal_error = exc
                    self._mode = "inactive"
                    if self._scheduler is not None:
                        self._scheduler.invalidate()
                    self._last_dispatched = self._reserved = None
                    self._last_target = self._reserved_target = None
                    self._ready.notify_all()

    def close(self) -> None:
        """Invalidate immediately and bound waiting for transport teardown."""
        with self._ready:
            if self._closed:
                return
            self._closed = True
            self._mode = "inactive"
            if self._scheduler is not None:
                self._scheduler.invalidate()
            self._last_dispatched = self._reserved = None
            self._last_target = self._reserved_target = None
            self._ready.notify_all()
        # WebSocket close can wait for the peer. Never make local control
        # shutdown depend on its close handshake or a wedged camera capture.
        closing = threading.Thread(target=self._transport.close, daemon=True)
        closing.start()
        closing.join(timeout=1.0)
        if self._worker is not None:
            self._worker.join(timeout=1.0)
