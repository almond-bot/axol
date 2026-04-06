from __future__ import annotations

import asyncio
import math

import numpy as np

from ..motor import CanBus, ControlMode, Joint, Motor, MotorGains, MotorStatus
from ..shared import CAN_LEFT, CAN_RIGHT
from .base import RobotBase
from .config import AxolConfig
from .control import Differentiator, compute_feedforward

_TAU = 2 * math.pi

# Per-joint position limits (rad).  shoulder_2 is asymmetric across arms.
SHOULDER_2_LEFT_LIMITS = (-0.25 * _TAU, 0.03 * _TAU)
SHOULDER_2_RIGHT_LIMITS = (-0.03 * _TAU, 0.25 * _TAU)
ELBOW_LEFT_LIMITS = (0.0, 0.42 * _TAU)
ELBOW_RIGHT_LIMITS = (-0.42 * _TAU, 0.0)

_LIMITS: dict[Joint, tuple[float, float]] = {
    Joint.SHOULDER_1: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.SHOULDER_3: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_1: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_2: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.WRIST_3: (-0.25 * _TAU, 0.25 * _TAU),
    Joint.GRIPPER: (-0.8037 * _TAU, 0.0),
}


def arm_limits(joint: Joint, is_left: bool) -> tuple[float, float]:
    """Return (min, max) position limits in radians for a joint on the given arm."""
    if joint == Joint.SHOULDER_2:
        return SHOULDER_2_LEFT_LIMITS if is_left else SHOULDER_2_RIGHT_LIMITS
    if joint == Joint.ELBOW:
        return ELBOW_LEFT_LIMITS if is_left else ELBOW_RIGHT_LIMITS
    return _LIMITS.get(joint, (-math.inf, math.inf))


class ArmController:
    """Controls one 7-DOF + gripper arm over a single CAN bus.

    Not instantiated directly — access via ``axol.left`` or ``axol.right``.
    """

    def __init__(self, bus: CanBus, config: AxolConfig, is_left: bool = True) -> None:
        self._config = config
        self._motors: dict[Joint, Motor] = {joint: Motor(bus, joint) for joint in Joint}
        self._limits_lo = np.array(
            [arm_limits(j, is_left)[0] for j in Joint], dtype=float
        )
        self._limits_hi = np.array(
            [arm_limits(j, is_left)[1] for j in Joint], dtype=float
        )
        self._differentiator = Differentiator(n=len(list(Joint)))

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float) -> None:
        """Start background telemetry polling on all motors at the given frequency.

        Args:
            hz: Poll frequency in Hz.
        """
        await asyncio.gather(*[m.start_telemetry(hz) for m in self._motors.values()])

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on all motors."""
        await asyncio.gather(*[m.stop_telemetry() for m in self._motors.values()])

    @property
    def positions(self) -> np.ndarray:
        """Latest cached joint positions (rad). Requires start_telemetry().

        Returns shape (8,) array in Joint enum order.
        """
        return np.array([m.position for m in self._motors.values()], dtype=np.float32)

    @property
    def torques(self) -> np.ndarray:
        """Latest cached joint torques (Nm / A). Requires start_telemetry().

        Returns shape (8,) array in Joint enum order.
        """
        return np.array([m.torque for m in self._motors.values()], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    async def enable(self) -> None:
        """Enable all motors and release brakes."""
        await asyncio.gather(*[m.enable() for m in self._motors.values()])

    async def disable(self) -> None:
        """Disable all motors and engage brakes."""
        await asyncio.gather(*[m.disable() for m in self._motors.values()])

    async def clear_errors(self) -> None:
        """Clear latched error flags on all motors."""
        await asyncio.gather(*[m.clear_errors() for m in self._motors.values()])

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors.

        Args:
            mode: Desired control mode.
        """
        await asyncio.gather(*[m.set_control_mode(mode) for m in self._motors.values()])

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> np.ndarray:
        """Return shaft position (rad) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_position() for j in joints])
        return np.array(values, dtype=np.float32)

    async def get_velocities(self) -> np.ndarray:
        """Return shaft velocity (rad/s) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_velocity() for j in joints])
        return np.array(values, dtype=np.float32)

    async def get_torques(self) -> np.ndarray:
        """Return torque estimate for every joint, fetched concurrently.

        Damiao: Nm. MyActuator: phase current in A.
        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(*[m.get_torque() for m in self._motors.values()])
        return np.array(values, dtype=np.float32)

    async def get_temperatures(self) -> np.ndarray:
        """Return motor temperature (°C) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(
            *[m.get_temperature() for m in self._motors.values()]
        )
        return np.array(values, dtype=np.float32)

    async def get_voltages(self) -> np.ndarray:
        """Return bus voltage (V) for every joint, fetched concurrently.

        Returns shape (8,) array in Joint enum order.
        """
        values = await asyncio.gather(*[m.get_voltage() for m in self._motors.values()])
        return np.array(values, dtype=np.float32)

    async def get_error_codes(self) -> list[MotorStatus]:
        """Return MotorStatus for every joint, fetched concurrently.

        Returns a list in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(
            *[self._motors[j].get_error_code() for j in joints]
        )
        return list(values)

    async def get_gains(self) -> list[MotorGains]:
        """Return PID gains for every joint, fetched concurrently.

        Returns a list in Joint enum order.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_gains() for j in joints])
        return list(values)

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(self, gains: dict[Joint, MotorGains]) -> None:
        """Write PID gains to the specified motors.

        Changes are persisted to non-volatile memory.
        """
        await asyncio.gather(*[self._motors[j].set_gains(g) for j, g in gains.items()])

    async def set_zero_position(self, joints: list[Joint]) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        Args:
            joints: List of joints to zero.
        """
        await asyncio.gather(*[self._motors[j].set_zero_position() for j in joints])

    async def set_acceleration(self, accelerations: dict[Joint, float]) -> None:
        """Set the acceleration ramp per joint. Deceleration matches acceleration.

        Args:
            accelerations: Mapping of joint → acceleration ramp (rad/s²).
                           Joints not in the dict are unchanged.
        """
        await asyncio.gather(
            *[self._motors[j].set_acceleration(a) for j, a in accelerations.items()]
        )

    async def set_position(self, positions: np.ndarray, max_speed: float) -> None:
        """Move joints to absolute positions using each motor's built-in controller.

        Positions are clipped to the arm's joint limits before being sent.
        The gripper value is normalized: 0.0 = closed, 1.0 = fully open.

        Args:
            positions: Shape (8,) array of target positions (rad) in Joint enum order,
                       except gripper which is [0, 1].
            max_speed: Maximum speed for all joints (rad/s).
        """
        positions = positions.copy()
        _lo, _hi = _LIMITS[Joint.GRIPPER]
        i = list(Joint).index(Joint.GRIPPER)
        positions[i] = _hi + positions[i] * (_lo - _hi)
        clipped = np.clip(positions, self._limits_lo, self._limits_hi)
        await asyncio.gather(
            *[
                self._motors[j].set_position(float(clipped[i]), max_speed)
                for i, j in enumerate(Joint)
            ]
        )

    async def set_velocity(self, velocities: np.ndarray) -> None:
        """Command target velocities using each motor's built-in speed controller.

        Args:
            velocities: Shape (8,) array of target velocities (rad/s) in Joint enum order.
        """
        await asyncio.gather(
            *[
                self._motors[j].set_velocity(float(velocities[i]))
                for i, j in enumerate(Joint)
            ]
        )

    async def motion_control(self, q: np.ndarray) -> None:
        """Send MIT impedance control to all joints concurrently.

        Desired positions are clipped to the arm's joint limits before being sent.
        Gains (kp, kd) and friction params are read from the AxolConfig. The
        feedforward torque is computed as gravity + friction; it is not specified
        directly in the config.

        Args:
            q: Shape (8,) array of desired positions (rad) in Joint enum order,
               except gripper which is [0, 1].
        """
        q = q.copy()
        _lo, _hi = _LIMITS[Joint.GRIPPER]
        i = list(Joint).index(Joint.GRIPPER)
        q[i] = _hi + q[i] * (_lo - _hi)
        clipped = np.clip(q, self._limits_lo, self._limits_hi)

        # Velocity feedforward via differentiation of commanded positions (rad/s).
        velocities = self._differentiator.differentiate(list(clipped))

        await asyncio.gather(
            *[
                self._motors[j].motion_control(
                    float(clipped[i]),
                    velocities[i],
                    getattr(self._config, j.value).kp,
                    getattr(self._config, j.value).kd,
                    compute_feedforward(
                        float(clipped[i]),
                        velocities[i],
                        getattr(self._config, j.value).ga,
                        getattr(self._config, j.value).gb,
                        getattr(self._config, j.value).fc,
                        getattr(self._config, j.value).k,
                        getattr(self._config, j.value).fv,
                        getattr(self._config, j.value).fo,
                    ),
                )
                for i, j in enumerate(Joint)
            ]
        )


class Axol(RobotBase):
    """Dual-arm Axol robot interface.

    Opens one CAN bus per arm and constructs all 16 motor drivers on entry.
    Use as an async context manager to ensure the buses are cleanly shut down.

        async with Axol() as axol:
            await axol.enable()
            await axol.start_telemetry(500)  # 500 Hz

            # control loop — instant, no await
            pos_l, pos_r = axol.left.positions, axol.right.positions

            await axol.motion_control(left=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]))

    Attributes:
        left:  ArmController for the left arm.
        right: ArmController for the right arm.

    Args:
        config:        Per-joint gains and friction parameters. Defaults to zero.
        left_channel:  SocketCAN interface name for the left arm.
        right_channel: SocketCAN interface name for the right arm.
    """

    def __init__(
        self,
        config: AxolConfig = AxolConfig(),
        left_channel: str = CAN_LEFT,
        right_channel: str = CAN_RIGHT,
    ) -> None:
        self._left_bus = CanBus(left_channel)
        self._right_bus = CanBus(right_channel)
        self.left = ArmController(self._left_bus, config, is_left=True)
        self.right = ArmController(self._right_bus, config, is_left=False)

    async def __aenter__(self) -> Axol:
        await asyncio.gather(self._left_bus.start(), self._right_bus.start())
        return self

    async def __aexit__(self, *_) -> None:
        await asyncio.gather(
            self.left.stop_telemetry(),
            self.right.stop_telemetry(),
            self._left_bus.close(),
            self._right_bus.close(),
        )

    # ------------------------------------------------------------------ #
    # Polling                                                              #
    # ------------------------------------------------------------------ #

    async def start_telemetry(self, hz: float) -> None:
        """Start background telemetry polling on both arms at the given frequency.

        Args:
            hz: Poll frequency in Hz.
        """
        await asyncio.gather(
            self.left.start_telemetry(hz), self.right.start_telemetry(hz)
        )

    async def stop_telemetry(self) -> None:
        """Stop the background telemetry polling loop on both arms."""
        await asyncio.gather(self.left.stop_telemetry(), self.right.stop_telemetry())

    # ------------------------------------------------------------------ #
    # Arm-wide commands                                                    #
    # ------------------------------------------------------------------ #

    async def enable(self) -> None:
        """Enable all motors on both arms."""
        await asyncio.gather(self.left.enable(), self.right.enable())

    async def disable(self) -> None:
        """Disable all motors on both arms."""
        await asyncio.gather(self.left.disable(), self.right.disable())

    async def clear_errors(self) -> None:
        """Clear latched error flags on both arms."""
        await asyncio.gather(self.left.clear_errors(), self.right.clear_errors())

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode on all motors on both arms.

        Args:
            mode: Desired control mode.
        """
        await asyncio.gather(
            self.left.set_control_mode(mode), self.right.set_control_mode(mode)
        )

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return joint positions (rad) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order.
        """
        return await asyncio.gather(
            self.left.get_positions(), self.right.get_positions()
        )

    async def get_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Return shaft velocity (rad/s) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order.
        """
        return await asyncio.gather(
            self.left.get_velocities(), self.right.get_velocities()
        )

    async def get_torques(self) -> tuple[np.ndarray, np.ndarray]:
        """Return torque estimates for both arms as (left, right).

        Each array is shape (8,) in Joint enum order.
        """
        return await asyncio.gather(self.left.get_torques(), self.right.get_torques())

    async def get_temperatures(self) -> tuple[np.ndarray, np.ndarray]:
        """Return motor temperatures (°C) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order.
        """
        return await asyncio.gather(
            self.left.get_temperatures(), self.right.get_temperatures()
        )

    async def get_voltages(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bus voltages (V) for both arms as (left, right).

        Each array is shape (8,) in Joint enum order.
        """
        return await asyncio.gather(self.left.get_voltages(), self.right.get_voltages())

    async def get_error_codes(
        self,
    ) -> tuple[list[MotorStatus], list[MotorStatus]]:
        """Return MotorStatus for both arms as (left, right).

        Each list is in Joint enum order.
        """
        return await asyncio.gather(
            self.left.get_error_codes(), self.right.get_error_codes()
        )

    async def get_gains(
        self,
    ) -> tuple[list[MotorGains], list[MotorGains]]:
        """Return PID gains for both arms as (left, right).

        Each list is in Joint enum order.
        """
        return await asyncio.gather(self.left.get_gains(), self.right.get_gains())

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_gains(
        self,
        left: dict[Joint, MotorGains] = {},
        right: dict[Joint, MotorGains] = {},
    ) -> None:
        """Write PID gains to the specified joints on both arms."""
        tasks = []
        if left:
            tasks.append(self.left.set_gains(left))
        if right:
            tasks.append(self.right.set_gains(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_zero_position(
        self,
        left: list[Joint] | None = None,
        right: list[Joint] | None = None,
    ) -> None:
        """Save the current shaft position as the encoder zero for the specified joints.

        Args:
            left:  Joints on the left arm to zero. ``None`` skips the arm.
            right: Joints on the right arm to zero. ``None`` skips the arm.
        """
        tasks = []
        if left is not None:
            tasks.append(self.left.set_zero_position(left))
        if right is not None:
            tasks.append(self.right.set_zero_position(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_acceleration(
        self,
        left: dict[Joint, float] = {},
        right: dict[Joint, float] = {},
    ) -> None:
        """Set per-joint acceleration ramps (rad/s²) on both arms.

        Args:
            left:  Joint → acceleration (rad/s²) for the left arm. ``None`` skips.
            right: Same for the right arm.
        """
        tasks = []
        if left:
            tasks.append(self.left.set_acceleration(left))
        if right:
            tasks.append(self.right.set_acceleration(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_positions(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
        max_speed: float = 0.0,
    ) -> None:
        """Command joint positions (rad) via the motor's built-in position controller.

        Args:
            left:      Shape (8,) array of target positions (rad) in Joint enum order.
                       ``None`` skips the arm.
            right:     Same for the right arm.
            max_speed: Maximum speed (rad/s). 0.0 uses the motor's default.
        """
        tasks = []
        if left is not None:
            tasks.append(self.left.set_position(left, max_speed))
        if right is not None:
            tasks.append(self.right.set_position(right, max_speed))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_velocity(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Command target velocities (rad/s) on both arms concurrently.

        Args:
            left:  Shape (8,) array of target velocities (rad/s). ``None`` skips the arm.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None:
            tasks.append(self.left.set_velocity(left))
        if right is not None:
            tasks.append(self.right.set_velocity(right))
        if tasks:
            await asyncio.gather(*tasks)

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Send MIT impedance control to both arms concurrently.

        Args:
            left:  Shape (8,) array of target positions (rad) for the left arm.
                   ``None`` skips the arm.
            right: Same for the right arm.
        """
        tasks = []
        if left is not None:
            tasks.append(self.left.motion_control(left))
        if right is not None:
            tasks.append(self.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)
