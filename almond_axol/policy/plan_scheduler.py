"""Unchanged-suffix scheduling for the custom policy interface.

This module has no robot, network or thread dependencies. A caller serializes
its methods with the dispatch lock; compression, inference and hardware I/O
must happen outside that lock. Row zero always belongs to the dispatch slot
saved by :meth:`PlanScheduler.begin_request`, never the reply's arrival slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PlanRuntimeConfig:
    """Custom policy interface options for ``run-policy --policy_type custom``.

    Zero output dimensions retain camera geometry. Resizing uses Pillow RGB
    interpolation before the mandatory lossless wire encoding. Late/exhausted
    plans either abort or enter explicit blocking-refresh mode: hold, obtain a
    fresh unconditioned prediction, execute ``request_interval`` rows, repeat.
    That recovery does not claim to reproduce another client's lateness rules.
    """

    request_interval: int = 5
    max_adoption_offset_steps: int = 6
    reply_timeout_s: float = 10.0
    late_policy: str = "blocking_refresh"
    output_width: int = 0
    output_height: int = 0
    interpolation: str = "bicubic"
    max_camera_age_s: float = 0.2
    max_state_age_s: float = 0.1
    max_camera_skew_s: float = 0.05
    # A new episode may wait for publishers to recover from startup work.
    # This bounds acquisition; it never increases a sensor's freshness limit.
    startup_observation_timeout_s: float = 3.0
    advertise_delay: bool = False
    max_cartesian_step_m: float = 0.05

    def validate(self, *, fps: int, horizon: int) -> None:
        for name, value, minimum in (
            ("fps", fps, 1),
            ("horizon", horizon, 1),
            ("request_interval", self.request_interval, 1),
            ("max_adoption_offset_steps", self.max_adoption_offset_steps, 0),
            ("output_width", self.output_width, 0),
            ("output_height", self.output_height, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.request_interval > horizon:
            raise ValueError("request_interval must not exceed the action horizon")
        if self.max_adoption_offset_steps >= horizon:
            raise ValueError("max_adoption_offset_steps must be inside the horizon")
        if bool(self.output_width) != bool(self.output_height):
            raise ValueError("output_width and output_height must both be set or zero")
        if max(self.output_width, self.output_height) > 8192:
            raise ValueError("output geometry exceeds the protocol dimension limit")
        if self.interpolation not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("unsupported image interpolation")
        if self.late_policy not in {"abort", "blocking_refresh"}:
            raise ValueError("late_policy must be abort or blocking_refresh")
        for name in (
            "reply_timeout_s",
            "max_camera_age_s",
            "max_state_age_s",
            "max_camera_skew_s",
            "startup_observation_timeout_s",
            "max_cartesian_step_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class PlanSchedulingError(RuntimeError):
    """The negotiated plan cannot safely continue."""


class SensorTimingError(PlanSchedulingError):
    """An observation cannot be used because its sensor times are unsuitable.

    ``retriable`` permits acquiring a *different* observation during a bounded
    startup wait. It never permits submitting the rejected observation or
    replacing its timestamps. Invalid/future timestamps are not retriable.
    """

    def __init__(
        self,
        message: str,
        *,
        retriable: bool,
        diagnostics: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class PendingPlan:
    request_id: str
    origin_tick: int
    generation: int
    snapshot_ns: int
    prediction_id: str | None
    from_row: int | None
    bootstrap: bool


class PlanScheduler:
    """One accepted prediction, one in-flight request, no future reservations."""

    def __init__(
        self,
        *,
        fps: int,
        horizon: int,
        width: int,
        config: PlanRuntimeConfig | None = None,
    ) -> None:
        self.config = config or PlanRuntimeConfig()
        self.config.validate(fps=fps, horizon=horizon)
        if width <= 0:
            raise ValueError("action width must be positive")
        self.fps, self.horizon, self.width = fps, horizon, width
        self.generation = 0
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        self.generation += 1
        self.next_tick = 0
        self.next_request_tick = 0
        self.pending: PendingPlan | None = None
        self.prediction_id: str | None = None
        self.actions: np.ndarray | None = None
        self.origin_tick = 0
        self.blocking = False
        self.delay_steps: int | None = None
        self.last_recovery: str | None = None

    def invalidate(self) -> None:
        """Local stop/hold: discard work immediately without waiting for I/O."""
        self.generation += 1
        self.pending = None
        self.prediction_id = None
        self.actions = None
        self.next_request_tick = self.next_tick

    def recover(self, reason: str) -> None:
        self.invalidate()
        if self.config.late_policy == "abort":
            raise PlanSchedulingError(reason)
        self.blocking = True
        self.last_recovery = reason

    @property
    def request_due(self) -> bool:
        return (
            self.pending is None
            and self.next_tick >= self.next_request_tick
            and (
                self.actions is None
                or self.next_tick - self.origin_tick < len(self.actions)
            )
        )

    def begin_request(self, now_ns: int) -> PendingPlan:
        if not self.request_due:
            raise PlanSchedulingError(
                "request is not due or another request is pending"
            )
        prediction_id, from_row = None, None
        if self.actions is not None:
            row = self.next_tick - self.origin_tick
            if 0 <= row < len(self.actions):
                prediction_id, from_row = self.prediction_id, row
        self._sequence += 1
        pending = PendingPlan(
            f"plan-{self._sequence}",
            self.next_tick,
            self.generation,
            now_ns,
            prediction_id,
            from_row,
            prediction_id is None,
        )
        self.pending = pending
        self.next_request_tick = self.next_tick + self.config.request_interval
        return pending

    def is_pending(self, request_id: str) -> bool:
        return (
            self.pending is not None
            and self.pending.request_id == request_id
            and self.pending.generation == self.generation
        )

    def cancel_unsent(self, request_id: str) -> None:
        """Drop stale unsent context without changing the accepted target plan."""
        if self.is_pending(request_id):
            self.pending = None
            self.next_request_tick = self.next_tick

    def adopt(
        self,
        request_id: str,
        actions: np.ndarray,
        now_ns: int,
        max_adoption_offset_steps: int | None = None,
    ) -> bool:
        if not self.is_pending(request_id):
            return False
        pending = self.pending
        assert pending is not None
        rows = np.asarray(actions, dtype=np.float32)
        if (
            rows.ndim != 2
            or rows.shape[1] != self.width
            or not 1 <= len(rows) <= self.horizon
            or not np.isfinite(rows).all()
        ):
            raise PlanSchedulingError("invalid returned action chunk")
        first_row = self.next_tick - pending.origin_tick
        bound = self.config.max_adoption_offset_steps
        if max_adoption_offset_steps is not None:
            if (
                type(max_adoption_offset_steps) is not int
                or max_adoption_offset_steps < 0
            ):
                raise PlanSchedulingError("invalid response adoption bound")
            bound = min(bound, max_adoption_offset_steps)
        # Bootstrap's cursor is held, so network delay cannot expire its row 0.
        if first_row >= len(rows) or (not pending.bootstrap and first_row > bound):
            self.recover(
                f"reply {request_id} arrived at row {first_row}, limit {bound}"
            )
            return False
        self.pending = None
        self.prediction_id = request_id
        self.origin_tick = pending.origin_tick
        # Own immutable copies: neither inference nor downstream filters may
        # mutate the accepted reference whose ID we advertise next time.
        self.actions = rows.copy()
        self.actions.flags.writeable = False
        # Adaptive hints use the protocol's bounded horizon-sized domain. A
        # cold bootstrap may wait far longer while the dispatch cursor holds;
        # saturate it instead of making the next observation unencodable.
        # ``horizon`` therefore means "at least a whole horizon of latency".
        self.delay_steps = min(
            self.horizon,
            max(
                first_row,
                math.ceil(
                    max(0, now_ns - pending.snapshot_ns) * self.fps / 1_000_000_000
                ),
            ),
        )
        return True

    def pop(self) -> tuple[int, np.ndarray] | None:
        if self.actions is None:
            return None
        row = self.next_tick - self.origin_tick
        limit = (
            min(len(self.actions), self.config.request_interval)
            if self.blocking
            else len(self.actions)
        )
        if row >= limit:
            if self.blocking:
                self.invalidate()
            else:
                self.recover("accepted action horizon exhausted")
            return None
        tick = self.next_tick
        target = self.actions[row].copy()
        self.next_tick += 1
        if self.blocking and row + 1 >= limit:
            self.invalidate()
        return tick, target


def prepare_plan_images(
    images: dict[str, np.ndarray], config: PlanRuntimeConfig
) -> dict[str, np.ndarray]:
    """Generic RGB preparation, bit-identical to Pillow's default RGB resize."""
    if not config.output_width:
        return {name: np.asarray(frame).copy() for name, frame in images.items()}
    from PIL import Image

    resample = getattr(Image.Resampling, config.interpolation.upper())
    return {
        name: np.asarray(
            Image.fromarray(frame).resize(
                (config.output_width, config.output_height), resample=resample
            )
        ).copy()
        for name, frame in images.items()
    }


def validate_sensor_times(
    state_ns: int, cameras_ns: dict[str, int], now_ns: int, config: PlanRuntimeConfig
) -> None:
    """Require actual, fresh sensor timestamps in the robot monotonic domain.

    Validate every timestamp before classifying age/skew as transient, so a
    stale state cannot hide an invalid camera timestamp during startup retry.
    """
    if type(now_ns) is not int or now_ns < 0:
        raise SensorTimingError(
            f"invalid current sensor-validation time: now_ns={now_ns!r}",
            retriable=False,
        )
    sensors = [
        ("state", state_ns, config.max_state_age_s),
        *((name, stamp, config.max_camera_age_s) for name, stamp in cameras_ns.items()),
    ]
    for name, stamp, _limit in sensors:
        if type(stamp) is not int or stamp < 0 or stamp > now_ns:
            raise SensorTimingError(
                f"invalid {name} sensor timestamp: stamp_ns={stamp!r}, "
                f"now_ns={now_ns} (expected an integer in [0, now_ns])",
                retriable=False,
            )
    skew_ns = max(cameras_ns.values()) - min(cameras_ns.values()) if cameras_ns else 0
    diagnostics = {
        "state_age_ms": (now_ns - state_ns) / 1_000_000,
        "camera_age_ms": {
            name: (now_ns - stamp) / 1_000_000 for name, stamp in cameras_ns.items()
        },
        "camera_skew_ms": skew_ns / 1_000_000,
        "max_state_age_ms": config.max_state_age_s * 1000,
        "max_camera_age_ms": config.max_camera_age_s * 1000,
        "max_camera_skew_ms": config.max_camera_skew_s * 1000,
    }
    camera_ages = ", ".join(
        f"{name}={age:.3f}" for name, age in diagnostics["camera_age_ms"].items()
    )
    summary = (
        f"state_age={diagnostics['state_age_ms']:.3f}ms "
        f"(limit {diagnostics['max_state_age_ms']:.3f}ms); "
        f"camera_ages=[{camera_ages}]ms "
        f"(limit {diagnostics['max_camera_age_ms']:.3f}ms); "
        f"camera_skew={diagnostics['camera_skew_ms']:.3f}ms "
        f"(limit {diagnostics['max_camera_skew_ms']:.3f}ms)"
    )
    for name, stamp, limit in sensors:
        if now_ns - stamp > limit * 1_000_000_000:
            raise SensorTimingError(
                f"{name} observation is stale; {summary}",
                retriable=True,
                diagnostics=diagnostics,
            )
    if skew_ns > config.max_camera_skew_s * 1_000_000_000:
        raise SensorTimingError(
            f"camera exposure skew exceeds configured limit; {summary}",
            retriable=True,
            diagnostics=diagnostics,
        )
