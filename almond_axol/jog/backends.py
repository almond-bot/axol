"""Jog backends: the simulator and the real robot behind one interface."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..robot.jelly import JellyConfig
from ..robot.lift import JOG_SPEED
from .sim import LIFT_FULL_TRAVEL_S, BaseSim, LiftSim, body_velocity

_logger = logging.getLogger(__name__)

# Robot versions with a powered lift and base.
MOBILE_MODELS = frozenset({"mobile"})


def _rest_pose() -> tuple[np.ndarray, np.ndarray]:
    from ..teleop.config import VRTeleopConfig

    rest = VRTeleopConfig()
    left = np.append(rest.rest_pose_left, 0.0).astype(np.float32)
    right = np.append(rest.rest_pose_right, 0.0).astype(np.float32)
    return left, right


def rest_q() -> np.ndarray:
    """The rest pose as a ``(14,)`` solver vector (left arm, then right)."""
    left, right = _rest_pose()
    return np.concatenate([left[:7], right[:7]]).astype(np.float32)


def _lift_dict(raised: float | None, velocity: float, moving: bool) -> dict[str, Any]:
    return {"raised": raised, "velocity": velocity, "moving": moving}


class SimBackend:
    """Kinematic arms plus the Jelly lift and base models.

    The arms track their command exactly, like :class:`~almond_axol.robot.sim.Sim`.
    The lift and base exist only for Axol on Jelly.
    """

    mode = "sim"
    has_arms = True

    def __init__(self, robot_model: str, jelly: JellyConfig | None = None) -> None:
        mobile = robot_model in MOBILE_MODELS
        self.has_lift = mobile
        self.has_base = mobile
        config = jelly or JellyConfig()
        self.lift = LiftSim(jog_speed=config.lift_speed)
        self.base = BaseSim(config)
        self._left, self._right = _rest_pose()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def arm_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self._left.copy(), self._right.copy()

    async def send_arms(self, left: np.ndarray, right: np.ndarray) -> None:
        self._left = np.asarray(left, dtype=np.float32).copy()
        self._right = np.asarray(right, dtype=np.float32).copy()

    def drive(self, command: tuple[float, float, float, int] | None, dt: float) -> None:
        if self.has_lift:
            # A lost source is the legs' jog deadman: they ramp to a stop.
            self.lift.step(0 if command is None else command[3], dt)
        if self.has_base:
            self.base.step(None if command is None else command[:3], dt)

    def lift_state(self) -> dict[str, Any] | None:
        if not self.has_lift:
            return None
        return _lift_dict(self.lift.raised, self.lift.velocity, self.lift.moving)

    def battery_state(self) -> dict[str, Any] | None:
        return None  # the simulated Jelly has no battery

    def base_state(self) -> dict[str, Any] | None:
        if not self.has_base:
            return None
        return {
            "x": self.base.x,
            "y": self.base.y,
            "yaw": self.base.yaw,
            "command": list(self.base.command),
            "velocity": list(self.base.velocity()),
            "moving": self.base.moving,
            "estimated": False,
        }


class HardwareBackend:
    """The arms through the realtime core, the lift and base through Jelly.

    The base has no odometry, so its pose is integrated from the wheel
    command for the 3D view and marked as estimated.
    """

    mode = "hardware"

    def __init__(
        self,
        axol: Any | None,
        jelly: Any | None,
    ) -> None:
        """
        Args:
            axol:  An unopened :class:`~almond_axol.robot.Axol`, or None to
                   leave the arms alone.
            jelly: An unenabled :class:`~almond_axol.robot.jelly.Jelly`, or
                   None when this robot has no Jelly.
        """
        self._axol = axol
        self._jelly = jelly
        self._axol_open = False
        self._jelly_open = False
        self._rest = _rest_pose()
        self._pose = [0.0, 0.0, 0.0]

    @property
    def has_arms(self) -> bool:
        axol = self._axol
        return (
            axol is not None
            and self._axol_open
            and (axol.left is not None or axol.right is not None)
        )

    @property
    def has_lift(self) -> bool:
        return self._jelly is not None and self._jelly_open and self._jelly.has_lift

    @property
    def has_base(self) -> bool:
        return self._jelly is not None and self._jelly_open and self._jelly.has_wheels

    async def start(self) -> None:
        if self._axol is not None:
            await self._axol.__aenter__()
            self._axol_open = True
        if self._jelly is not None:
            try:
                await self._jelly.enable()
            except BaseException:
                await self.stop()
                raise
            self._jelly_open = True

    async def stop(self) -> None:
        if self._jelly is not None and self._jelly_open:
            self._jelly_open = False
            try:
                await self._jelly.disable()
            except Exception:  # noqa: BLE001 - still release the arms
                _logger.warning("jog: Jelly shutdown failed", exc_info=True)
        if self._axol is not None and self._axol_open:
            self._axol_open = False
            await self._axol.__aexit__(None, None, None)

    async def arm_positions(self) -> tuple[np.ndarray, np.ndarray]:
        # The telemetry cache is the only safe read while the core owns the
        # bus; an absent arm reads as its rest pose and is never commanded.
        left, right = self._rest
        axol = self._axol
        if axol.left is not None:
            left = np.asarray(axol.left.positions, dtype=np.float32)
        if axol.right is not None:
            right = np.asarray(axol.right.positions, dtype=np.float32)
        return left.copy(), right.copy()

    async def send_arms(self, left: np.ndarray, right: np.ndarray) -> None:
        axol = self._axol
        await axol.motion_control(
            left=left if axol.left is not None else None,
            right=right if axol.right is not None else None,
        )

    def drive(self, command: tuple[float, float, float, int] | None, dt: float) -> None:
        jelly = self._jelly
        if jelly is None or not self._jelly_open:
            return
        if command is not None:
            # A dead source sends nothing: Jelly's own command timeout then
            # stops the wheels and the lift, as it does for a lost headset.
            vx, vy, wz, lift = command
            if not jelly.has_wheels:
                vx = vy = wz = 0.0
            jelly.set_command(vx, vy, wz, lift if jelly.has_lift else 0)
        if jelly.has_wheels and dt > 0.0:
            bx, by, bw = body_velocity(*jelly.body_cmd, jelly.config)
            x, y, yaw = self._pose
            c, s = np.cos(yaw), np.sin(yaw)
            self._pose = [
                x + float(bx * c - by * s) * dt,
                y + float(bx * s + by * c) * dt,
                float(np.remainder(yaw + bw * dt + np.pi, 2 * np.pi) - np.pi),
            ]

    def lift_state(self) -> dict[str, Any] | None:
        if not self.has_lift:
            return None
        status = self._jelly.lift_status
        velocity = 0.0
        moving = False
        if status is not None:
            # Full jog speed (JOG_SPEED counts/s) runs the travel in
            # LIFT_FULL_TRAVEL_S seconds.
            velocity = float(status.velocity) / JOG_SPEED / LIFT_FULL_TRAVEL_S
            moving = bool(status.moving)
        return _lift_dict(self._jelly.lift_raised, velocity, moving)

    def battery_state(self) -> dict[str, Any] | None:
        """The battery estimate the lift board's power frames give, while
        they are fresh (see :attr:`~almond_axol.robot.jelly.Jelly.battery`)."""
        jelly = self._jelly
        if jelly is None or not self._jelly_open:
            return None
        battery = jelly.battery
        if battery is None:
            return None
        return {
            "voltage": round(battery.voltage, 2),
            "percent": round(battery.percent, 1),
            "charging": battery.charging,
            "underLoad": battery.under_load,
            "ageSeconds": round(jelly.battery_age or 0.0, 1),
            "live": True,
        }

    def base_state(self) -> dict[str, Any] | None:
        if not self.has_base:
            return None
        jelly = self._jelly
        command = list(jelly.body_cmd)
        return {
            "x": self._pose[0],
            "y": self._pose[1],
            "yaw": self._pose[2],
            "command": command,
            "velocity": list(body_velocity(*command, jelly.config)),
            "moving": any(abs(v) >= 1e-3 for v in command),
            "estimated": True,
            "fault": bool(jelly.wheel_fault),
        }
