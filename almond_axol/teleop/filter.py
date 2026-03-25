"""
Interpolation and filtering for VR teleoperation.

All interpolation is step-based (max rev per step), not time-based.
"""

from __future__ import annotations

import numpy as np

from ..shared import rad_to_rev, rev_to_rad

SMOOTH_ALPHA: float = 0.45
MAX_DELTA_REV_INTERPOLATION: float = 0.0001


class AlphaSmoothFilter:
    """Exponential smoothing filter for joint angle arrays (radians).

    Applies an alpha-weighted moving average. Lower alpha = more lag but
    smoother output. Higher alpha = more responsive but noisier.

    Args:
        alpha: Blend factor in (0, 1]. ``1.0`` disables smoothing.
    """

    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
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
    """Step-limited move toward a target joint configuration (radians).

    Supports two modes:

    - **Pre-computed trajectory** (via :meth:`set_trajectory`): steps through
      provided waypoints one per :meth:`step` call.
    - **Linear interpolation** (via :meth:`set`): moves toward target by at
      most ``max_step_rev`` revolutions per joint per step.
    """

    def __init__(self, max_step_rev: float = MAX_DELTA_REV_INTERPOLATION) -> None:
        self.max_step_rev = max_step_rev
        self._current: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._trajectory: list[np.ndarray] | None = None
        self._traj_index: int = 0

    def set(self, current_rad: np.ndarray, target_rad: np.ndarray) -> None:
        """Interpolate linearly from ``current_rad`` toward ``target_rad``."""
        self._trajectory = None
        self._traj_index = 0
        self._current = np.array(current_rad, dtype=np.float64)
        self._target = np.array(target_rad, dtype=np.float64)

    def set_trajectory(self, trajectory: list[np.ndarray]) -> None:
        """Use a pre-computed collision-aware trajectory instead of linear interpolation."""
        self._current = None
        self._target = None
        self._trajectory = [np.array(q, dtype=np.float64) for q in trajectory]
        self._traj_index = 0

    def step(self) -> tuple[np.ndarray | None, bool]:
        """Advance one step. Returns ``(new_q_rad, done)``."""
        if self._trajectory is not None:
            if self._traj_index >= len(self._trajectory):
                self.clear()
                return None, True
            q = self._trajectory[self._traj_index]
            self._traj_index += 1
            done = self._traj_index >= len(self._trajectory)
            if done:
                self._trajectory = None
                self._traj_index = 0
            return q, done

        if self._current is None or self._target is None:
            return None, True
        current = self._current
        target = self._target
        new_current = np.empty_like(current)
        done = True
        for i in range(len(current)):
            delta_rad = target[i] - current[i]
            delta_rev = rad_to_rev(delta_rad)
            step_rev = max(-self.max_step_rev, min(self.max_step_rev, delta_rev))
            new_current[i] = current[i] + rev_to_rad(step_rev)
            if abs(new_current[i] - target[i]) > 1e-6:
                done = False
        self._current = new_current
        if done:
            self._current = None
            self._target = None
        return new_current, done

    def is_active(self) -> bool:
        """True if interpolation is still in progress."""
        if self._trajectory is not None:
            return self._traj_index < len(self._trajectory)
        return self._current is not None and self._target is not None

    def clear(self) -> None:
        """Cancel any active interpolation immediately."""
        self._current = None
        self._target = None
        self._trajectory = None
        self._traj_index = 0
