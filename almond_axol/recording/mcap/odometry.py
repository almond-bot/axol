"""Jelly's base pose and velocity, dead-reckoned from its wheels.

Jelly is an x-drive: four omni wheels on the corners of a square, each
driving along a diagonal (see :data:`almond_axol.robot.jelly.WHEELS`). With
the wheel at body position ``p`` driving along unit direction ``d``, its rim
moves at ``d . (v + wz * z x p)``. For the four wheels (front left, front
right, back left, back right) that is::

    rim = A @ [vx, vy, wz]
    A = [[s, -s, -L],
         [s,  s,  L],
         [s,  s, -L],
         [s, -s,  L]],   s = 1/sqrt(2),  L = sqrt(2) * a

where ``a`` is each wheel's distance from the centre along x and along y.
The columns of ``A`` are orthogonal, so its least-squares inverse is closed
form, and the leftover ``(r1 + r2 - r3 - r4) / 4`` is how much the four
wheels disagree: zero when nothing slips.

The pose integrates each step's change in wheel angle (not the measured
speeds), so noise in the speed readings does not accumulate. Each step is
applied at the heading halfway through it. Heading comes from the wheels
alone, which drift when the rollers slip while the base turns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Rolling radius (m): the wheel axle's height above the floor in the Jelly
# URDF (axol_jelly.urdf: wheel joints at z = 0.06335 over a floor 0.0001 m
# below the base frame).
WHEEL_RADIUS_M = 0.0635
# Each wheel's distance from the base centre along x and along y (m), from the
# same URDF (wheel joints at +/-0.2158).
WHEEL_HALF_SPACING_M = 0.2158
# A wheel that seems to have turned faster than this between two samples is a
# bad reading, not motion (Jelly's wheels top out near 30 rad/s).
_MAX_WHEEL_SPEED_RAD_S = 60.0

_S = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class OdometrySample:
    """Base pose relative to where it started, and its body-frame velocity."""

    t: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    wz: float
    slip: float  # wheel disagreement, m/s at the rim


def body_twist(
    rim: Sequence[float], half_spacing: float = WHEEL_HALF_SPACING_M
) -> tuple[float, float, float, float]:
    """``(vx, vy, wz, slip)`` from the four rim speeds (or rim distances)."""
    r1, r2, r3, r4 = rim
    lever = math.sqrt(2.0) * half_spacing
    vx = 0.5 * _S * (r1 + r2 + r3 + r4)
    vy = 0.5 * _S * (-r1 + r2 + r3 - r4)
    wz = (-r1 + r2 - r3 + r4) / (4.0 * lever)
    slip = (r1 + r2 - r3 - r4) / 4.0
    return vx, vy, wz, slip


def wheel_rims(
    vx: float, vy: float, wz: float, half_spacing: float = WHEEL_HALF_SPACING_M
) -> tuple[float, float, float, float]:
    """Rim speeds for a body velocity (the forward map :func:`body_twist` inverts)."""
    lever = math.sqrt(2.0) * half_spacing
    return (
        _S * vx - _S * vy - lever * wz,
        _S * vx + _S * vy + lever * wz,
        _S * vx + _S * vy - lever * wz,
        _S * vx - _S * vy + lever * wz,
    )


class XDriveOdometry:
    """Integrates Jelly's wheel samples into a base pose.

    ``signs`` maps each wheel's motor id (1-4) to the factor that turns its
    measured angle into the body convention the mix uses (positive = rim
    moving with a forward component); it is
    :data:`almond_axol.robot.jelly.WHEEL_SIGNS`.
    """

    def __init__(
        self,
        signs: Mapping[int, float],
        *,
        radius: float = WHEEL_RADIUS_M,
        half_spacing: float = WHEEL_HALF_SPACING_M,
    ) -> None:
        self._signs = tuple(float(signs[i]) for i in (1, 2, 3, 4))
        self._radius = float(radius)
        self._half_spacing = float(half_spacing)
        self.rejected = 0
        self.reset()

    def reset(self) -> None:
        """Start again at the origin on the next sample."""
        self._last: tuple[float, tuple[float, ...]] | None = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def _rims(self, motor: Sequence[float]) -> list[float]:
        return [
            self._radius * s * float(m) for s, m in zip(self._signs, motor, strict=True)
        ]

    def command_twist(
        self, motor_speeds: Sequence[float]
    ) -> tuple[float, float, float]:
        """Body velocity ``(vx, vy, wz)`` that commanded motor speeds (rad/s) ask for."""
        vx, vy, wz, _ = body_twist(self._rims(motor_speeds), self._half_spacing)
        return vx, vy, wz

    def update(
        self,
        t: float,
        positions: Sequence[float],
        velocities: Sequence[float],
    ) -> OdometrySample:
        """Add one wheel sample (motor angles in rad, speeds in rad/s)."""
        angles = tuple(float(p) for p in positions)
        if self._last is not None:
            t_prev, prev = self._last
            dt = t - t_prev
            deltas = [a - b for a, b in zip(angles, prev, strict=True)]
            limit = _MAX_WHEEL_SPEED_RAD_S * max(dt, 0.02)
            if dt > 0 and all(abs(d) <= limit for d in deltas):
                dx, dy, dyaw, _ = body_twist(self._rims(deltas), self._half_spacing)
                heading = self.yaw + 0.5 * dyaw
                c, s = math.cos(heading), math.sin(heading)
                self.x += dx * c - dy * s
                self.y += dx * s + dy * c
                self.yaw = math.atan2(
                    math.sin(self.yaw + dyaw), math.cos(self.yaw + dyaw)
                )
            elif dt > 0:
                self.rejected += 1
        self._last = (t, angles)
        vx, vy, wz, slip = body_twist(self._rims(velocities), self._half_spacing)
        return OdometrySample(t, self.x, self.y, self.yaw, vx, vy, wz, slip)
