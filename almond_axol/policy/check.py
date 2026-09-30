"""Exercise a custom policy endpoint without a robot.

:func:`check_policy` connects to a running endpoint (e.g. one started with
:func:`~almond_axol.policy.serve`) and plays the robot's side of a session
on a virtual clock: the same :class:`~almond_axol.policy.plan_scheduler.
PlanScheduler` run-policy uses decides when to request, which rows of a reply
are adopted after a simulated ``delay_steps`` of latency, and when a late or
exhausted plan triggers recovery. The simulated arm tracks its commands
perfectly, so ``state`` is the last executed target wherever the state and
action layouts share a name.

The report says whether the endpoint keeps up and how smooth the executed
trajectory is — in particular how big the jumps are where one plan hands over
to the next, the thing a chunked policy most often gets wrong::

    from almond_axol.policy import check_policy

    report = check_policy("ws://127.0.0.1:8765", ticks=300, delay_steps=3)
    print(report.summary())

``axol policy.check`` is the same from the command line.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..constants import Joint
from .plan_client import PlanPolicyClient
from .plan_protocol import Continuation, LastDispatched, PlanObservation, PlanSpec
from .plan_scheduler import PlanRuntimeConfig, PlanScheduler, PlanSchedulingError
from .protocol import CameraSpec

_EE_AXES = ("x", "y", "z", "rx", "ry", "rz")


class _RecordingScheduler(PlanScheduler):
    """The robot's scheduler, recording every recovery it enters."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.recoveries: list[str] = []
        super().__init__(*args, **kwargs)

    def recover(self, reason: str) -> None:
        self.recoveries.append(reason)
        super().recover(reason)


def axol_action_names(
    *, cartesian: bool = False, gripper: bool = True
) -> tuple[str, ...]:
    """The robot's ordered action (and, by default, state) names.

    ``cartesian`` mirrors ``robot_config.observe_cartesian``; ``gripper=False``
    is the gripperless SKU.
    """
    names: list[str] = []
    for side in ("left", "right"):
        if cartesian:
            names += [f"{side}_ee.{axis}" for axis in _EE_AXES]
        else:
            names += [
                f"{side}_{joint.value}.pos"
                for joint in Joint
                if joint is not Joint.GRIPPER
            ]
        if gripper:
            names.append(f"{side}_gripper.pos")
    return tuple(names)


def default_spec(
    *,
    cartesian: bool = False,
    cameras: tuple[str, ...] = ("overhead",),
    image_shape: tuple[int, int] = (360, 640),
    fps: int = 30,
    actions_per_chunk: int = 50,
    request_interval: int = PlanRuntimeConfig.request_interval,
    max_adoption_offset_steps: int | None = PlanRuntimeConfig.max_adoption_offset_steps,
) -> PlanSpec:
    """A session contract like the one run-policy sends with default settings."""
    names = axol_action_names(cartesian=cartesian)
    height, width = image_shape
    return PlanSpec(
        state_names=names,
        action_names=names,
        cameras=tuple(CameraSpec(name, (height, width, 3)) for name in cameras),
        fps=fps,
        actions_per_chunk=actions_per_chunk,
        request_interval=request_interval,
        max_adoption_offset_steps=max_adoption_offset_steps,
        dispatch_feedback=True,
    )


@dataclass
class CheckReport:
    """What happened during :func:`check_policy`.

    Attributes:
        spec: The session contract used.
        targets: ``(ticks, D)`` executed targets (NaN rows: nothing executed,
            i.e. waiting for the first plan).
        plan_ids: The plan each executed tick came from (``None``: none).
        requests: Inference requests sent.
        adopted: Replies the scheduler adopted.
        recoveries: Recovery reasons (late reply, exhausted horizon), in order.
        latency_s: Wall-clock round trip per request.
    """

    spec: PlanSpec
    targets: np.ndarray
    plan_ids: list[str | None]
    requests: int = 0
    adopted: int = 0
    recoveries: list[str] = field(default_factory=list)
    latency_s: list[float] = field(default_factory=list)

    def _steps(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-dim max |Δtarget| within a plan, and across plan switches."""
        width = self.targets.shape[1]
        within = np.zeros(width)
        switch = np.zeros(width)
        for tick in range(1, len(self.targets)):
            a, b = self.targets[tick - 1], self.targets[tick]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            step = np.abs(b - a)
            if self.plan_ids[tick] == self.plan_ids[tick - 1]:
                within = np.maximum(within, step)
            else:
                switch = np.maximum(switch, step)
        return within, switch

    @property
    def max_step_within_plan(self) -> np.ndarray:
        return self._steps()[0]

    @property
    def max_step_at_switch(self) -> np.ndarray:
        return self._steps()[1]

    def summary(self) -> str:
        names = self.spec.action_names
        within, switch = self._steps()
        executed = int((~np.isnan(self.targets).any(axis=1)).sum())
        lines = [
            f"{len(self.targets)} ticks at {self.spec.fps} Hz: {executed} executed, "
            f"{self.requests} requests, {self.adopted} adopted, "
            f"{len(self.recoveries)} recoveries.",
        ]
        if self.latency_s:
            lat = np.asarray(self.latency_s) * 1000
            lines.append(
                f"Endpoint round trip: median {np.median(lat):.1f} ms, "
                f"max {lat.max():.1f} ms "
                f"({self.spec.request_interval} ticks = "
                f"{1000 * self.spec.request_interval / self.spec.fps:.0f} ms between "
                "requests)."
            )
        for reason in self.recoveries[:5]:
            lines.append(f"  recovery: {reason}")
        worst = np.argsort(switch)[::-1][:3]
        lines.append("Largest per-tick jumps where one plan hands over to the next:")
        for index in worst:
            lines.append(
                f"  {names[index]}: {switch[index]:.4f} at a switch vs "
                f"{within[index]:.4f} within a plan"
            )
        return "\n".join(lines)


def check_policy(
    url: str,
    *,
    spec: PlanSpec | None = None,
    ticks: int = 300,
    delay_steps: int = 2,
    initial_state: np.ndarray | None = None,
    images: Callable[[int], Mapping[str, np.ndarray]] | None = None,
    late_policy: str = "blocking_refresh",
) -> CheckReport:
    """Run one virtual-clock session against the endpoint at ``url``.

    Args:
        url: ``ws://host:port`` of the endpoint.
        spec: Session contract (default: :func:`default_spec`).
        ticks: Control ticks to simulate.
        delay_steps: Simulated latency: a reply is adopted this many ticks
            after its request (the real round trip is not counted, so the
            result doesn't depend on this machine's speed).
        initial_state: Starting state (default zeros).
        images: ``tick -> {camera: frame}``; default mid-grey frames.
        late_policy: ``"blocking_refresh"`` (run-policy's default) records
            recoveries and continues; ``"abort"`` stops at the first one.
    """
    import time

    spec = spec or default_spec()
    if ticks < 1 or delay_steps < 0:
        raise ValueError("ticks must be >= 1 and delay_steps >= 0")
    horizon = spec.actions_per_chunk
    scheduler = _RecordingScheduler(
        fps=spec.fps,
        horizon=horizon,
        width=len(spec.action_names),
        config=PlanRuntimeConfig(
            request_interval=spec.request_interval,
            max_adoption_offset_steps=(
                horizon - 1
                if spec.max_adoption_offset_steps is None
                else spec.max_adoption_offset_steps
            ),
            late_policy=late_policy,
        ),
    )
    state = (
        np.zeros(len(spec.state_names), np.float32)
        if initial_state is None
        else np.asarray(initial_state, np.float32).copy()
    )
    tracked = [
        (spec.state_names.index(name), index)
        for index, name in enumerate(spec.action_names)
        if name in spec.state_names
    ]
    if images is None:
        frames = {
            camera.name: np.full(camera.shape, 128, np.uint8) for camera in spec.cameras
        }

        def images(_tick: int) -> Mapping[str, np.ndarray]:
            return frames

    dt_ns = round(1e9 / spec.fps)
    report = CheckReport(
        spec=spec,
        targets=np.full((ticks, len(spec.action_names)), np.nan, np.float32),
        plan_ids=[None] * ticks,
    )
    in_flight: tuple[int, object] | None = None
    last_dispatched: tuple[int, str, int] | None = None
    episode = 1
    with PlanPolicyClient(url) as client:
        client.connect(spec)
        client.reset(episode)
        wire_generation = scheduler.generation
        for tick in range(ticks):
            now_ns = (tick + 1) * dt_ns
            if in_flight is None and scheduler.request_due:
                if scheduler.generation != wire_generation:
                    # The robot resets the endpoint whenever it discards a plan.
                    episode += 1
                    client.reset(episode)
                    wire_generation = scheduler.generation
                pending = scheduler.begin_request(now_ns)
                frames_now = dict(images(tick))
                request = PlanObservation(
                    request_id=pending.request_id,
                    state=state.copy(),
                    images=frames_now,
                    state_sample_time_ns=now_ns,
                    image_capture_time_ns=dict.fromkeys(frames_now, now_ns),
                    continuation=(
                        None
                        if pending.prediction_id is None
                        else Continuation(pending.prediction_id, pending.from_row)
                    ),
                    last_dispatched=(
                        LastDispatched(last_dispatched[1], last_dispatched[2])
                        if last_dispatched is not None
                        and last_dispatched[0] == scheduler.generation
                        else None
                    ),
                )
                started = time.perf_counter()
                reply = client.infer(request)
                report.latency_s.append(time.perf_counter() - started)
                report.requests += 1
                in_flight = (tick + delay_steps, reply)
            if in_flight is not None and tick >= in_flight[0]:
                reply = in_flight[1]
                in_flight = None
                try:
                    if scheduler.adopt(
                        reply.request_id,
                        reply.actions,
                        now_ns,
                        reply.max_adoption_offset_steps,
                    ):
                        report.adopted += 1
                except PlanSchedulingError:
                    break
            plan_id = scheduler.prediction_id
            generation = scheduler.generation
            try:
                popped = scheduler.pop()
            except PlanSchedulingError:
                break
            if popped is None:
                continue
            dispatched_tick, target = popped
            report.targets[tick] = target
            report.plan_ids[tick] = plan_id
            last_dispatched = (
                generation,
                plan_id,
                dispatched_tick - scheduler.origin_tick,
            )
            for state_index, action_index in tracked:
                state[state_index] = target[action_index]
    report.recoveries = scheduler.recoveries
    return report
