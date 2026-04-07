"""
Interpolation and filtering for VR teleoperation.
"""

from __future__ import annotations

import numpy as np


class AlphaSmoothFilter:
    """Exponential smoothing filter for joint angle arrays (radians).

    Applies an alpha-weighted moving average. Lower alpha = more lag but
    smoother output. Higher alpha = more responsive but noisier.

    Args:
        alpha: Blend factor in (0, 1]. ``1.0`` disables smoothing.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def update(self, new: np.ndarray | None) -> np.ndarray | None:
        """Apply one smoothing step. Returns ``new`` unchanged on first call."""
        if new is None:
            return None
        new = np.asarray(new, dtype=np.float32)
        if self._prev is None or len(self._prev) != len(new):
            self._prev = new.copy()
            return new.copy()
        out = self.alpha * new + (1.0 - self.alpha) * self._prev
        self._prev = out
        return out

    def reset(self, seed: np.ndarray | None = None) -> None:
        """Reset filter state, optionally seeding with a known starting value."""
        self._prev = (
            np.asarray(seed, dtype=np.float32).copy() if seed is not None else None
        )


class ResetInterpolator:
    """Steps through a pre-computed collision-aware trajectory one waypoint per call.

    Optionally ramps gripper values (normalized [0, 1]) from a start value to
    1.0 (open) over the same number of steps as the arm trajectory.
    """

    def __init__(self) -> None:
        self._trajectory: list[np.ndarray] | None = None
        self._traj_index: int = 0
        self._l_grip_start: float = 0.0
        self._r_grip_start: float = 0.0

    def set_trajectory(
        self,
        trajectory: list[np.ndarray],
        l_grip: float = 0.0,
        r_grip: float = 0.0,
    ) -> None:
        """Load a pre-computed trajectory and gripper start values."""
        self._trajectory = [np.array(q, dtype=np.float64) for q in trajectory]
        self._traj_index = 0
        self._l_grip_start = l_grip
        self._r_grip_start = r_grip

    def step(self) -> tuple[np.ndarray | None, float, float, bool]:
        """Advance one step.

        Returns ``(new_q_rad, l_grip, r_grip, done)`` where gripper values are
        smoothstepped from their start values to 1.0 over the trajectory length.
        """
        if self._trajectory is None or self._traj_index >= len(self._trajectory):
            self.clear()
            return None, 1.0, 1.0, True
        n = len(self._trajectory)
        alpha = (self._traj_index + 1) / n
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        l_grip = self._l_grip_start + smooth * (1.0 - self._l_grip_start)
        r_grip = self._r_grip_start + smooth * (1.0 - self._r_grip_start)
        q = self._trajectory[self._traj_index]
        self._traj_index += 1
        done = self._traj_index >= n
        if done:
            self._trajectory = None
            self._traj_index = 0
        return q, l_grip, r_grip, done

    def is_active(self) -> bool:
        """True if trajectory playback is in progress."""
        return self._trajectory is not None and self._traj_index < len(self._trajectory)

    def clear(self) -> None:
        """Cancel any active trajectory."""
        self._trajectory = None
        self._traj_index = 0
