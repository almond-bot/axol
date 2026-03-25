from __future__ import annotations

import asyncio

from ..constants import CAN_LEFT, CAN_RIGHT
from ..motor import CanBus, Joint, JointValues, Motor, MotorGains, MotorStatus
from .base import RobotBase
from .config import AxolConfig, JointGains  # noqa: F401 (re-exported)


class ArmController:
    """Controls one 7-DOF + gripper arm over a single CAN bus.

    Not instantiated directly — access via ``axol.left`` or ``axol.right``.
    """

    def __init__(self, bus: CanBus, config: AxolConfig) -> None:
        self._config = config
        self._motors: dict[Joint, Motor] = {joint: Motor(bus, joint) for joint in Joint}

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
    def positions(self) -> JointValues:
        """Latest cached joint positions (rev). Requires start_telemetry()."""
        return {j: m.position for j, m in self._motors.items()}

    @property
    def torques(self) -> JointValues:
        """Latest cached joint torques (Nm / A). Requires start_telemetry()."""
        return {j: m.torque for j, m in self._motors.items()}

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

    async def set_zero_position(self) -> None:
        """Save the current shaft position as the encoder zero on all motors."""
        await asyncio.gather(*[m.set_zero_position() for m in self._motors.values()])

    async def set_acceleration(
        self, acceleration: float, deceleration: float | None = None
    ) -> None:
        """Set the acceleration ramp on all motors.

        Args:
            acceleration: Acceleration ramp (rev/s²)
            deceleration: Deceleration ramp (rev/s²). Defaults to acceleration.
        """
        await asyncio.gather(
            *[
                m.set_acceleration(acceleration, deceleration)
                for m in self._motors.values()
            ]
        )

    # ------------------------------------------------------------------ #
    # Getters                                                              #
    # ------------------------------------------------------------------ #

    async def get_positions(self) -> JointValues:
        """Return shaft position (rev) for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_position() for j in joints])
        return dict(zip(joints, values))

    async def get_velocities(self) -> JointValues:
        """Return shaft velocity (rev/s) for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_velocity() for j in joints])
        return dict(zip(joints, values))

    async def get_torques(self) -> JointValues:
        """Return torque estimate for every joint, fetched concurrently.

        Damiao: Nm. MyActuator: phase current in A.
        """
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_torque() for j in joints])
        return dict(zip(joints, values))

    async def get_temperatures(self) -> JointValues:
        """Return motor temperature (°C) for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(
            *[self._motors[j].get_temperature() for j in joints]
        )
        return dict(zip(joints, values))

    async def get_voltages(self) -> JointValues:
        """Return bus voltage (V) for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_voltage() for j in joints])
        return dict(zip(joints, values))

    async def get_error_codes(self) -> dict[Joint, MotorStatus]:
        """Return MotorStatus for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(
            *[self._motors[j].get_error_code() for j in joints]
        )
        return dict(zip(joints, values))

    async def get_gains(self) -> dict[Joint, MotorGains]:
        """Return PID gains for every joint, fetched concurrently."""
        joints = list(Joint)
        values = await asyncio.gather(*[self._motors[j].get_gains() for j in joints])
        return dict(zip(joints, values))

    # ------------------------------------------------------------------ #
    # Setters                                                              #
    # ------------------------------------------------------------------ #

    async def set_position(self, positions: JointValues, max_speed: float) -> None:
        """Move joints to absolute positions using each motor's built-in controller.

        Args:
            positions: Target positions (rev) keyed by joint.
            max_speed: Maximum speed for all commanded joints (rev/s).
        """
        await asyncio.gather(
            *[self._motors[j].set_position(p, max_speed) for j, p in positions.items()]
        )

    async def set_velocity(self, velocities: JointValues) -> None:
        """Command target velocities using each motor's built-in speed controller.

        Args:
            velocities: Target velocities (rev/s) keyed by joint.
        """
        await asyncio.gather(
            *[self._motors[j].set_velocity(v) for j, v in velocities.items()]
        )

    async def set_gains(self, gains: dict[Joint, MotorGains]) -> None:
        """Write PID gains to the specified motors.

        Changes are persisted to non-volatile memory.
        """
        await asyncio.gather(*[self._motors[j].set_gains(g) for j, g in gains.items()])

    async def motion_control(self, positions: JointValues) -> None:
        """Send MIT impedance control to the specified joints concurrently.

        Gains (kp, kd, t_ff) are read from the AxolConfig supplied at construction.

        Args:
            positions: Desired positions (rev) keyed by joint.
        """
        await asyncio.gather(
            *[
                self._motors[j].motion_control(
                    p,
                    0.0,
                    getattr(self._config, j.value).kp,
                    getattr(self._config, j.value).kd,
                    getattr(self._config, j.value).t_ff,
                )
                for j, p in positions.items()
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

            await axol.motion_control(left={Joint.ELBOW: 0.5})

    Attributes:
        left:  ArmController for the left arm.
        right: ArmController for the right arm.

    Args:
        config:        Per-joint gains. Defaults to the bundled config.json.
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
        self.left = ArmController(self._left_bus, config)
        self.right = ArmController(self._right_bus, config)

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

    async def enable(self) -> None:
        """Enable all motors on both arms."""
        await asyncio.gather(self.left.enable(), self.right.enable())

    async def disable(self) -> None:
        """Disable all motors on both arms."""
        await asyncio.gather(self.left.disable(), self.right.disable())

    async def clear_errors(self) -> None:
        """Clear latched error flags on both arms."""
        await asyncio.gather(self.left.clear_errors(), self.right.clear_errors())

    async def set_zero_position(self) -> None:
        """Save the current shaft position as the encoder zero on both arms."""
        await asyncio.gather(
            self.left.set_zero_position(), self.right.set_zero_position()
        )

    async def set_acceleration(
        self, acceleration: float, deceleration: float | None = None
    ) -> None:
        """Set the acceleration ramp on all motors on both arms."""
        await asyncio.gather(
            self.left.set_acceleration(acceleration, deceleration),
            self.right.set_acceleration(acceleration, deceleration),
        )

    async def get_positions(self) -> tuple[JointValues, JointValues]:
        """Return joint positions (rev) for both arms as (left, right)."""
        return await asyncio.gather(
            self.left.get_positions(), self.right.get_positions()
        )

    async def get_velocities(self) -> tuple[JointValues, JointValues]:
        """Return shaft velocity (rev/s) for both arms as (left, right)."""
        return await asyncio.gather(
            self.left.get_velocities(), self.right.get_velocities()
        )

    async def get_torques(self) -> tuple[JointValues, JointValues]:
        """Return torque estimates for both arms as (left, right)."""
        return await asyncio.gather(self.left.get_torques(), self.right.get_torques())

    async def get_temperatures(self) -> tuple[JointValues, JointValues]:
        """Return motor temperatures (°C) for both arms as (left, right)."""
        return await asyncio.gather(
            self.left.get_temperatures(), self.right.get_temperatures()
        )

    async def get_voltages(self) -> tuple[JointValues, JointValues]:
        """Return bus voltages (V) for both arms as (left, right)."""
        return await asyncio.gather(self.left.get_voltages(), self.right.get_voltages())

    async def get_error_codes(
        self,
    ) -> tuple[dict[Joint, MotorStatus], dict[Joint, MotorStatus]]:
        """Return MotorStatus for both arms as (left, right)."""
        return await asyncio.gather(
            self.left.get_error_codes(), self.right.get_error_codes()
        )

    async def get_gains(
        self,
    ) -> tuple[dict[Joint, MotorGains], dict[Joint, MotorGains]]:
        """Return PID gains for both arms as (left, right)."""
        return await asyncio.gather(self.left.get_gains(), self.right.get_gains())

    async def set_positions(
        self,
        left: JointValues | None = None,
        right: JointValues | None = None,
    ) -> None:
        """Command joint positions via MIT impedance control on both arms.

        Gains are taken from the ``AxolConfig`` provided at construction.

        Args:
            left:  Target positions (rev) for the left arm.  ``None`` skips the arm.
            right: Target positions (rev) for the right arm. ``None`` skips the arm.
        """
        await self.motion_control(left=left or {}, right=right or {})

    async def set_position(
        self,
        left: JointValues = {},
        right: JointValues = {},
        max_speed: float = 0.0,
    ) -> None:
        """Move joints to absolute positions using each motor's built-in controller."""
        tasks = []
        if left:
            tasks.append(self.left.set_position(left, max_speed))
        if right:
            tasks.append(self.right.set_position(right, max_speed))
        if tasks:
            await asyncio.gather(*tasks)

    async def set_velocity(
        self,
        left: JointValues = {},
        right: JointValues = {},
    ) -> None:
        """Command target velocities (rev/s) on both arms concurrently."""
        tasks = []
        if left:
            tasks.append(self.left.set_velocity(left))
        if right:
            tasks.append(self.right.set_velocity(right))
        if tasks:
            await asyncio.gather(*tasks)

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

    async def motion_control(
        self,
        left: JointValues = {},
        right: JointValues = {},
    ) -> None:
        """Send MIT impedance control to both arms concurrently.

        Args:
            left:  Target positions (rev) for the left arm.  Empty skips the arm.
            right: Target positions (rev) for the right arm. Empty skips the arm.
        """
        tasks = []
        if left:
            tasks.append(self.left.motion_control(left))
        if right:
            tasks.append(self.right.motion_control(right))
        if tasks:
            await asyncio.gather(*tasks)
