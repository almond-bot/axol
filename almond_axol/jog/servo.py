"""Cartesian velocity jogging of one gripper, like MoveIt Servo.

While a jog button is held the page streams a normalized twist for the arm:
translation along and rotation about x, y and z, in the torso frame or the
gripper's own. :class:`ArmJog` turns it into one joint command per control
tick:

- The twist ramps up to the held speed and back down on release at a fixed
  acceleration, like the lift.
- A target tip pose advances by the ramped twist every tick, and one
  :meth:`~almond_axol.kinematics.arm_ik.ArmKinematics.servo_step` moves the
  arm toward it. Each joint stays inside its limits and under
  ``joint_speed``, and the arm stops short of the body and the other arm.
- The target never leads the tip by more than ``lead_m`` / ``lead_rad``,
  and an arm held back by a limit runs at the speed it actually makes, so
  it stops at once and moves off the limit as soon as the command reverses.
- A dead command source (no command, see
  :meth:`~almond_axol.jog.session.JogSession.hold`) stops the arm in the
  same tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..kinematics.arm_ik import (
    ArmKinematics,
    Side,
    exp_rotation,
    rotation_vector,
)

Frame = Literal["torso", "gripper"]
FRAMES: tuple[Frame, Frame] = ("torso", "gripper")


@dataclass
class JogLimits:
    """Speeds and ramps for Cartesian jogging."""

    linear_speed: float = 0.06
    """Gripper speed (m/s) at a full command."""
    angular_speed: float = 0.5
    """Gripper turn rate (rad/s) at a full command."""
    ramp_s: float = 0.3
    """Time to reach full speed, and to stop from it."""
    joint_speed: float = 1.0
    """Fastest any joint may turn (rad/s)."""
    lead_m: float = 0.004
    """Furthest the jog target may run ahead of the tip (m)."""
    lead_rad: float = math.radians(1.5)
    """Largest angle the jog target may run ahead of the tip."""


def _approach(current: np.ndarray, goal: np.ndarray, step: float) -> np.ndarray:
    """Move ``current`` toward ``goal`` by at most ``step`` (vector length)."""
    delta = goal - current
    size = float(np.linalg.norm(delta))
    if size <= step:
        return goal.copy()
    return current + delta * (step / size)


class ArmJog:
    """Hold-to-jog state for one arm (see the module docstring)."""

    def __init__(
        self, kinematics: ArmKinematics, side: Side, limits: JogLimits
    ) -> None:
        self._kin = kinematics
        self.side: Side = side
        self.limits = limits
        self._velocity = np.zeros(6)  # torso-frame twist being run
        self._target: tuple[np.ndarray, np.ndarray] | None = None
        self.limited = False

    @property
    def moving(self) -> bool:
        return bool(np.any(self._velocity != 0.0))

    @property
    def velocity(self) -> np.ndarray:
        """The torso-frame twist being run: m/s, then rad/s."""
        return self._velocity.copy()

    def stop(self) -> None:
        """Stop now, with no ramp: a dead command source or a planned move."""
        self._velocity[:] = 0.0
        self._target = None
        self.limited = False

    def step(
        self,
        q: np.ndarray,
        command: np.ndarray | None,
        dt: float,
        *,
        frame: Frame = "torso",
        speed: float = 1.0,
    ) -> np.ndarray | None:
        """This tick's joint command, or None when the arm is not jogging.

        ``command`` is the held twist, each of ``(x, y, z, roll, pitch,
        yaw)`` in -1..1 (zeros: let go, so it ramps down), in ``frame``,
        scaled by ``speed`` (0..1). None is a dead command source: the arm
        stops this tick.
        """
        if command is None or dt <= 0.0:
            self.stop()
            return None
        limits = self.limits
        command = np.clip(np.asarray(command, dtype=np.float64), -1.0, 1.0)
        scale = min(max(float(speed), 0.0), 1.0)
        linear = command[:3] * limits.linear_speed * scale
        angular = command[3:] * limits.angular_speed * scale
        start = self._kin.tip(q, self.side)
        if frame == "gripper":
            linear, angular = start[1] @ linear, start[1] @ angular
        ramp = max(limits.ramp_s, 1e-3)
        self._velocity[:3] = _approach(
            self._velocity[:3], linear, limits.linear_speed / ramp * dt
        )
        self._velocity[3:] = _approach(
            self._velocity[3:], angular, limits.angular_speed / ramp * dt
        )
        if not self.moving:
            self._target = None
            self.limited = False
            return None
        target = self._target or start
        target = (
            target[0] + self._velocity[:3] * dt,
            exp_rotation(self._velocity[3:] * dt) @ target[1],
        )
        result = self._kin.servo_step(q, self.side, target, limits.joint_speed * dt)
        # Keep the target within reach of the tip, so an arm held back by a
        # limit or the body has no backlog to work off.
        tip = self._kin.tip(result.q, self.side)
        offset = target[0] - tip[0]
        turn = rotation_vector(target[1] @ tip[1].T)
        distance = float(np.linalg.norm(offset))
        angle = float(np.linalg.norm(turn))
        limited = False
        if distance > limits.lead_m:
            offset *= limits.lead_m / distance
            limited = True
        if angle > limits.lead_rad:
            turn *= limits.lead_rad / angle
            limited = True
        self._target = (tip[0] + offset, exp_rotation(turn) @ tip[1])
        if limited:
            # Held back: run at the speed the tip made, so a reversal starts
            # from there rather than ramping down a speed it never reached.
            made = np.concatenate(
                [
                    (tip[0] - start[0]) / dt,
                    rotation_vector(tip[1] @ start[1].T) / dt,
                ]
            )
            self._velocity = np.where(
                np.abs(made) < np.abs(self._velocity), made, self._velocity
            )
        self.limited = limited
        return result.q
