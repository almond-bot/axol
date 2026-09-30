"""Robot-side joint transition after IK; published policy plans stay immutable."""

from __future__ import annotations

import math

import numpy as np

from ..teleop.filter import TrapezoidalFilter


class JointHandover:
    """Blend from a held command, then converge under joint motion limits.

    The ramp starts at the first policy dispatch, not while inference is
    pending. Its first output is exactly the held command. Arm joints retain
    velocity/acceleration limits through the convergence tail; grippers blend
    only during the ramp. This is downstream execution shaping, not editing
    the plan advertised to the desktop.
    """

    def __init__(
        self,
        *,
        fps: int,
        duration_s: float,
        max_vel: float = 2.5,
        max_accel: float = 8.0,
    ) -> None:
        if fps <= 0 or not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("Invalid handover rate or duration")
        if any(not math.isfinite(v) or v <= 0 for v in (max_vel, max_accel)):
            raise ValueError("Handover motion limits must be finite and positive")
        self.steps = math.ceil(duration_s * fps)
        self.filter = TrapezoidalFilter(max_vel, max_accel, 1 / fps)
        self.anchor: dict[str, float] | None = None
        self.step = 0
        self.keys: tuple[str, ...] = ()

    def seed(self, action: dict[str, float]) -> None:
        if not action or not all(math.isfinite(v) for v in action.values()):
            raise ValueError("Handover needs a finite held joint command")
        self.anchor = dict(action) if self.steps else None
        self.keys = tuple(k for k in action if not k.endswith("gripper.pos"))
        self.filter.reset(
            seed=np.array([action[k] for k in self.keys], dtype=np.float32)
        )
        self.step = 0

    def apply(self, action: dict[str, float]) -> dict[str, float]:
        if self.anchor is None:
            return action
        if action.keys() != self.anchor.keys() or not all(
            math.isfinite(v) for v in action.values()
        ):
            raise ValueError(
                "Handover command schema changed or contains nonfinite values"
            )
        t = min(1.0, self.step / self.steps)
        alpha = t * t * (3.0 - 2.0 * t)
        blended = {k: v + alpha * (action[k] - v) for k, v in self.anchor.items()}
        if self.step == 0:
            self.step += 1
            return dict(self.anchor)
        limited = self.filter.update(np.array([blended[k] for k in self.keys]))
        blended.update(zip(self.keys, map(float, limited), strict=True))
        self.step += 1
        if t == 1.0 and all(abs(blended[k] - action[k]) < 1e-6 for k in self.keys):
            self.anchor = None
        return blended
