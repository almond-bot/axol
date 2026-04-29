"""Unified async motor interface and driver factory.

Provides :class:`Motor` (the public facade used throughout the codebase) and
:func:`make_driver` (selects the correct low-level :class:`MotorDriver` subclass
— :class:`DamiaoMotor` or :class:`MyActuatorMotor` — based on CAN ID or an
explicit type override).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from ..shared import Joint
from .bus import CanBus
from .damiao import DamiaoMotor
from .driver import MotorDriver
from .errors import MotorError
from .myactuator import MyActuatorMotor
from .types import ControlMode, MotorGains, MotorStatus


class _MotorType(Enum):
    MYACTUATOR = "myactuator"
    DAMIAO = "damiao"


@dataclass(frozen=True)
class _JointConfig:
    kind: _MotorType
    motor_id: int
    kt: float


_ID_TO_TYPE: dict[int, _MotorType] = {}  # populated after _JOINT_CONFIG is defined

_JOINT_CONFIG: dict[Joint, _JointConfig] = {
    Joint.SHOULDER_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x01, kt=2.4),
    Joint.SHOULDER_2: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x02, kt=2.4),
    Joint.SHOULDER_3: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x03, kt=2.1),
    Joint.ELBOW: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x04, kt=2.1),
    Joint.WRIST_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x05, kt=2.1),
    Joint.WRIST_2: _JointConfig(_MotorType.DAMIAO, motor_id=0x06, kt=0.945),
    Joint.WRIST_3: _JointConfig(_MotorType.DAMIAO, motor_id=0x07, kt=0.945),
    Joint.GRIPPER: _JointConfig(_MotorType.DAMIAO, motor_id=0x08, kt=0.945),
}


_ID_TO_TYPE = {cfg.motor_id: cfg.kind for cfg in _JOINT_CONFIG.values()}


def make_driver(
    bus: CanBus, motor_id: int, kt: float, motor_type: str | None = None
) -> MotorDriver:
    """Return the correct MotorDriver for *motor_id*.

    Args:
        kt:         Torque constant (Nm/A). Used by ``get_torque()`` to convert
                    raw current readings to Nm.
        motor_type: ``"myactuator"`` or ``"damiao"`` to override inference.
                    If ``None``, the type is inferred from *motor_id*.
    """
    if motor_type is not None:
        kind = _MotorType(motor_type)
    else:
        kind = _ID_TO_TYPE.get(motor_id)
        if kind is None:
            raise ValueError(
                f"Unknown motor ID {motor_id:#04x}. Known IDs: "
                + ", ".join(f"{i:#04x}" for i in sorted(_ID_TO_TYPE))
            )
    if kind == _MotorType.MYACTUATOR:
        return MyActuatorMotor(bus, motor_id, kt=kt)
    elif kind == _MotorType.DAMIAO:
        return DamiaoMotor(bus, motor_id, feedback_id=0x10 + motor_id, kt=kt)
    else:
        raise ValueError(f"Unknown motor type {kind}")


class Motor:
    """Unified async motor interface.

    Instantiate with a CanBus and a Joint; the correct underlying driver
    is selected automatically based on the joint.

        motor = Motor(bus, Joint.WRIST_2)
        await motor.enable()
        pos = await motor.get_position()  # radians
    """

    def __init__(self, bus: CanBus, joint: Joint, can_id: int | None = None) -> None:
        """Construct a Motor and select the correct underlying driver for the joint.

        Args:
            bus:    Shared CAN bus for this arm.
            joint:  The joint this motor drives; determines driver type and default CAN ID.
            can_id: Override the default CAN ID from the joint config table; useful for
                bench testing a motor before it is mounted to the arm.
        """
        self.joint = joint
        self.mode: ControlMode | None = None
        cfg = _JOINT_CONFIG[joint]
        motor_id = can_id if can_id is not None else cfg.motor_id
        self._driver = make_driver(bus, motor_id, kt=cfg.kt)
        self._position: float | None = None
        self._torque: float | None = None
        self._telemetry_task: asyncio.Task | None = None
        self._driver._on_feedback = lambda pos, torq: (
            setattr(self, "_position", pos) or setattr(self, "_torque", torq)
        )

    async def enable(self) -> None:
        """Enable the motor and release the brake."""
        await self._driver.enable()

    async def disable(self) -> None:
        """Disable the motor and engage the brake."""
        await self._driver.disable()

    async def clear_errors(self) -> None:
        """Clear any latched motor error flags."""
        await self._driver.clear_errors()

    async def set_zero_position(self) -> None:
        """Save the current shaft position as the encoder zero reference."""
        await self._driver.set_zero_position()

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the active control mode.

        Damiao: writes register 10 to match the requested mode.
        MyActuator: resets the motor (no persistent mode register; mode is
        determined per-command).

        Args:
            mode: Desired control mode.
        """
        await self._driver.set_control_mode(mode)
        self.mode = mode

    async def get_control_mode(self) -> ControlMode | None:
        """Return the active control mode read from hardware, or None if unsupported.

        Damiao: reads register 10 and returns the matching ControlMode.
        MyActuator: returns None — the mode is implicit in each command sent.
        """
        return await self._driver.get_control_mode()

    async def get_position(self) -> float:
        """Return current shaft position in radians."""
        if self._telemetry_task is not None:
            raise MotorError(
                f"Telemetry is active on {self.joint} — use motor.position or stop_telemetry() first"
            )
        return await self._driver.get_position()

    async def get_velocity(self) -> float:
        """Return current shaft velocity in radians per second."""
        return await self._driver.get_velocity()

    async def get_torque(self) -> float:
        """Return current torque estimate in Nm."""
        if self._telemetry_task is not None:
            raise MotorError(
                f"Telemetry is active on {self.joint} — use motor.torque or stop_telemetry() first"
            )
        return await self._driver.get_torque()

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        """Start the background polling loop at the given frequency.

        Args:
            hz:     Poll frequency in Hz.
            torque: If True, also fetch and cache torque (Nm) each cycle.
        """
        await self.stop_telemetry()
        self._telemetry_task = asyncio.create_task(
            self._telemetry_loop(hz, torque=torque)
        )

    async def stop_telemetry(self) -> None:
        """Stop the background polling loop."""
        if self._telemetry_task is not None:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
            self._telemetry_task = None

    async def _telemetry_loop(self, hz: float, *, torque: bool = False) -> None:
        interval = 1.0 / hz
        on_torque = (lambda t: setattr(self, "_torque", t)) if torque else None
        while True:
            start = asyncio.get_event_loop().time()
            try:
                await self._driver.get_telemetry(
                    on_position=lambda p: setattr(self, "_position", p),
                    on_torque=on_torque,
                )
            except MotorError:
                pass  # Dropped CAN frames are normal on physical buses; skip cycle
            elapsed = asyncio.get_event_loop().time() - start
            await asyncio.sleep(max(0.0, interval - elapsed))

    @property
    def position(self) -> float:
        """Latest cached shaft position (rad).

        Populated by start_telemetry() or set_impedance() responses.
        """
        if self._position is None:
            raise MotorError(
                f"No position data for {self.joint} — call start_telemetry() or send a set_impedance() command first"
            )
        return self._position

    @property
    def torque(self) -> float:
        """Latest cached torque estimate (Nm).

        Populated by start_telemetry(torque=True) or set_impedance() responses.
        """
        if self._torque is None:
            raise MotorError(
                f"No torque data for {self.joint} — call start_telemetry(torque=True) or send a set_impedance() command first"
            )
        return self._torque

    async def get_temperature(self) -> float:
        """Return motor temperature in degrees Celsius.

        Damiao: returns the higher of MOS and rotor temperatures.
        """
        return await self._driver.get_temperature()

    async def get_voltage(self) -> float:
        """Return bus voltage in Volts."""
        return await self._driver.get_voltage()

    async def get_error_code(self) -> MotorStatus:
        """Return the current motor status / error code."""
        return await self._driver.get_error_code()

    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        """Move to an absolute position using the motor's built-in position controller.

        Requires the motor to be in POSITION_VELOCITY mode (set via set_control_mode).

        Args:
            position:  Target shaft position (rad)
            max_speed: Maximum speed during the move (rad/s)
        """
        if self.mode != ControlMode.POSITION_VELOCITY:
            raise RuntimeError(
                f"{self.joint} is in mode {self.mode}, expected POSITION_VELOCITY. "
                f"Call set_control_mode(ControlMode.POSITION_VELOCITY) first."
            )
        await self._driver.set_position_velocity(position, max_speed)

    async def set_velocity(self, velocity: float) -> None:
        """Command a target velocity using the motor's built-in speed controller.

        Requires the motor to be in VELOCITY mode (set via set_control_mode).

        Args:
            velocity: Target shaft velocity (rad/s)
        """
        if self.mode != ControlMode.VELOCITY:
            raise RuntimeError(
                f"{self.joint} is in mode {self.mode}, expected VELOCITY. "
                f"Call set_control_mode(ControlMode.VELOCITY) first."
            )
        await self._driver.set_velocity(velocity)

    async def set_position_force(
        self, position: float, max_speed: float, max_torque: float
    ) -> None:
        """Move to a position with hard speed and torque limits.

        Only supported by Damiao motors. Raises MotorError on MyActuator.
        Requires the motor to be in POSITION_FORCE mode (set via set_control_mode).

        Args:
            position:   Target shaft position (rad)
            max_speed:  Maximum speed during the move (rad/s)
            max_torque: Maximum output torque (Nm)
        """
        if self.mode != ControlMode.POSITION_FORCE:
            raise RuntimeError(
                f"{self.joint} is in mode {self.mode}, expected POSITION_FORCE. "
                f"Call set_control_mode(ControlMode.POSITION_FORCE) first."
            )
        await self._driver.set_position_force(position, max_speed, max_torque)

    async def set_acceleration(
        self, acceleration: float, deceleration: float | None = None
    ) -> None:
        """Set the acceleration ramp for position and velocity control modes.

        Args:
            acceleration: Acceleration ramp (rad/s²)
            deceleration: Deceleration ramp (rad/s²). If None, matches acceleration.
                          Damiao stores acceleration and deceleration separately;
                          MyActuator applies the same value to both ramps.
        """
        await self._driver.set_acceleration(acceleration, deceleration)

    async def get_gains(self) -> MotorGains:
        """Read the stored PID gains for the speed and position control loops."""
        return await self._driver.get_gains()

    async def set_gains(self, gains: MotorGains) -> None:
        """Write PID gains for the speed and position control loops.

        Changes are persisted to non-volatile memory so they survive power cycles.

        Args:
            gains: Gain values to write. Damiao ignores current_kp / current_ki.
        """
        await self._driver.set_gains(gains)

    async def set_can_id(self, can_id: int) -> None:
        """Change the motor's CAN ID and persist it to flash.

        The driver updates its internal state immediately so subsequent commands
        use the new ID without re-instantiation.

        Damiao: also sets the feedback ID to can_id + 0x10.

        Args:
            can_id: New CAN ID for the motor.
        """
        await self._driver.set_can_id(can_id)

    async def set_can_baud_rate(self, baud_rate: int) -> None:
        """Change the motor's CAN baud rate and persist it to flash.

        The motor must be power-cycled for the new baud rate to take effect.

        Args:
            baud_rate: Baud rate in bps. Supported values:
                       MyActuator — 500_000, 1_000_000
                       Damiao     — 125_000, 200_000, 250_000, 500_000,
                                    1_000_000, 2_000_000, 2_500_000,
                                    3_200_000, 4_000_000, 5_000_000
        """
        await self._driver.set_can_baud_rate(baud_rate)

    async def set_impedance(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
    ) -> None:
        """Send an impedance control command.

        Requires the motor to be in IMPEDANCE mode (set via set_control_mode).

        Args:
            p_des: Desired position (rad)
            v_des: Desired velocity (rad/s)
            kp:    Position stiffness [0, 500]
            kd:    Velocity damping   [0, 5]
            t_ff:  Feedforward torque (Nm)
        """
        if self.mode != ControlMode.IMPEDANCE:
            raise RuntimeError(
                f"{self.joint} is in mode {self.mode}, expected IMPEDANCE. "
                f"Call set_control_mode(ControlMode.IMPEDANCE) first."
            )
        await self._driver.set_impedance(p_des, v_des, kp, kd, t_ff)
