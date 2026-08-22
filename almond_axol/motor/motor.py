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
from typing import Mapping

from ..constants import Joint
from .bus import CanBus
from .config import MotorParam
from .damiao import DamiaoMotor
from .driver import MotorDriver
from .errors import MotorError
from .myactuator import MyActuatorMotor
from .types import ControlMode, MotorGains, MotorStatus


class _MotorType(Enum):
    """Identifies which vendor protocol a joint's motor speaks.

    Values come from the drivers so the CLI, config snapshots, and this
    mapping cannot drift apart.
    """

    MYACTUATOR = MyActuatorMotor.MOTOR_TYPE
    DAMIAO = DamiaoMotor.MOTOR_TYPE


@dataclass(frozen=True)
class _JointConfig:
    """Per-joint motor configuration: vendor type, default CAN ID, and torque constant.

    ``kt`` is the MyActuator current→torque constant (Nm/A); Damiao joints leave
    it at the default because they report torque directly in feedback frames.
    """

    kind: _MotorType
    motor_id: int
    kt: float = 0.0


_ID_TO_TYPE: dict[int, _MotorType] = {}  # populated after _JOINT_CONFIG is defined

_JOINT_CONFIG: dict[Joint, _JointConfig] = {
    Joint.SHOULDER_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x01, kt=2),
    Joint.SHOULDER_2: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x02, kt=2),
    Joint.SHOULDER_3: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x03, kt=1.5),
    Joint.ELBOW: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x04, kt=1.5),
    Joint.WRIST_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x05, kt=1.5),
    Joint.WRIST_2: _JointConfig(_MotorType.DAMIAO, motor_id=0x06),
    Joint.WRIST_3: _JointConfig(_MotorType.DAMIAO, motor_id=0x07),
    Joint.GRIPPER: _JointConfig(_MotorType.DAMIAO, motor_id=0x08),
}


_ID_TO_TYPE = {cfg.motor_id: cfg.kind for cfg in _JOINT_CONFIG.values()}


def make_driver(
    bus: CanBus, motor_id: int, kt: float = 0.0, motor_type: str | None = None
) -> MotorDriver:
    """Return the correct MotorDriver for *motor_id*.

    Args:
        kt:         Torque constant (Nm/A). MyActuator only — used by
                    ``get_torque()`` to convert its raw current readings to Nm.
                    Ignored for Damiao, which reports torque directly, so it can
                    be omitted when building a Damiao driver.
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
        return DamiaoMotor(bus, motor_id, feedback_id=0x10 + motor_id)
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
        self._driver.set_feedback_callback(
            lambda pos, torq: (
                setattr(self, "_position", pos) or setattr(self, "_torque", torq)
            )
        )

    async def enable(self) -> None:
        """Enable the motor and release the brake."""
        await self._driver.enable()

    async def attach(self, mode: ControlMode) -> None:
        """Attach to an already-enabled motor without disturbing its torque state.

        The reconnect counterpart to :meth:`enable` + :meth:`set_control_mode`,
        for when a previous process left the motor enabled and holding (e.g.
        it died mid-session). Re-reads the state needed for correct command
        scaling and verifies the motor is enabled, fault-free, and — where the
        hardware exposes a mode register (Damiao) — already in ``mode``. No
        reset, brake, enable, or motion command is sent, so a motor holding
        position keeps holding it throughout. In contrast,
        :meth:`set_control_mode` reboots MyActuator motors, dropping torque
        for ~2 s.

        Args:
            mode: Control mode the motor is expected to already be in.

        Raises:
            MotorError: If the motor is unreachable, disabled, faulted, or in
                a different hardware control mode. Recover with a full
                :meth:`enable` bring-up.
        """
        await self._driver.attach()
        hw_mode = await self._driver.get_control_mode()
        if hw_mode is not None and hw_mode != mode:
            raise MotorError(
                f"{self.joint} is in {hw_mode.name} mode, expected {mode.name} "
                f"— attach is not possible; use enable() for a full bring-up"
            )
        self.mode = mode

    async def disable(self) -> None:
        """Disable the motor and engage the brake."""
        await self._driver.disable()

    async def is_holding(self) -> bool:
        """Return True if the motor is enabled and holding torque. Read-only.

        Damiao: feedback status is ENABLED — note an enabled motor that was
        never sent a command holds no torque but still reports True.
        MyActuator: the status-1 running byte is set and no fault is latched
        (on fleet firmware the byte is 1 only while actively executing
        commands).

        This is the per-motor probe behind the idempotent
        :meth:`Axol.enable`'s keep-holding-or-bring-up decision.
        """
        return await self._driver.is_holding()

    async def clear_errors(self) -> None:
        """Clear any latched motor error flags."""
        await self._driver.clear_errors()

    async def set_zero_position(self) -> None:
        """Save the current shaft position as the encoder zero reference.

        For arm joints this is calibrated at one of the joint's mechanical
        end stops, not at the rest position (see ``closer_end_stop``).
        """
        await self._driver.set_zero_position()

    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the active control mode.

        Damiao: writes register 10 to match the requested mode.
        MyActuator: resets the motor (no persistent mode register; mode is
        determined per-command).

        WARNING: the MyActuator reset drops torque for ~2 s — never switch
        modes while the motor is holding a load (the joint falls). Bring the
        arm to rest first, or use ``enable(hold=False)`` in flows that manage
        modes themselves.

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

    async def get_firmware_version(self) -> int | None:
        """Return the motor firmware version, or None if unsupported.

        MyActuator: the VersionDate (uint32, e.g. ``2026042402``) read via 0xB2.
        Damiao: None — not exposed by the protocol.
        """
        return await self._driver.get_firmware_version()

    async def get_model(self) -> str | None:
        """Return the motor model string, or None if unsupported.

        MyActuator: the model string read via 0xB5 (e.g. ``"X8S2V"``).
        Damiao: None — not exposed by the protocol.
        """
        return await self._driver.get_model()

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
    def kd_max(self) -> float:
        """Upper bound of the firmware's impedance ``kd`` range (Nm·s/rad).

        ``set_impedance`` clamps ``kd`` to this silently: 5 on Damiao and
        legacy MyActuator, 50 on MyActuator V4.4+ (detected on ``enable()`` /
        ``attach()``).
        """
        return self._driver.kd_max

    @property
    def telemetry_active(self) -> bool:
        """True while the background telemetry polling loop is running."""
        return self._telemetry_task is not None

    @property
    def has_position(self) -> bool:
        """True once a position has been cached by telemetry or a set_impedance() response."""
        return self._position is not None

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

    async def get_low_voltage_threshold(self) -> float:
        """Return the undervoltage protection threshold in Volts.

        MyActuator only; raises MotorError on Damiao.
        """
        return await self._driver.get_low_voltage_threshold()

    async def set_low_voltage_threshold(self, volts: float) -> None:
        """Set the undervoltage protection threshold (V) and persist it to ROM.

        MyActuator only; raises MotorError on Damiao. MyActuator motors already
        apply the project-wide threshold on ``enable()``.
        """
        await self._driver.set_low_voltage_threshold(volts)

    def resolve_config_param(self, name: str) -> MotorParam:
        """Look a configuration parameter up by name for this motor's family."""
        return type(self._driver).resolve_param(name)

    async def read_config(self, param: MotorParam) -> float:
        """Read one configuration parameter, in the unit its spec names."""
        return await self._driver.read_config(param)

    async def write_config(self, param: MotorParam, value: float) -> None:
        """Write one configuration parameter and persist it to flash/ROM."""
        await self._driver.write_config(param, value)

    async def dump_config(
        self, raw_range: range | None = None
    ) -> dict[MotorParam | int, float]:
        """Read every known configuration parameter."""
        return await self._driver.dump_config(raw_range)

    async def restore_config(
        self, values: Mapping[MotorParam, float], *, include_protected: bool = False
    ) -> list[MotorParam]:
        """Write a saved configuration back and return the parameters changed."""
        return await self._driver.restore_config(
            values, include_protected=include_protected
        )

    async def get_can_timeout(self) -> float:
        """Return the CAN loss-of-comms alarm time in milliseconds.

        Damiao only; raises MotorError on MyActuator.
        """
        return await self._driver.get_can_timeout()

    async def set_can_timeout(self, milliseconds: float) -> None:
        """Set the CAN loss-of-comms alarm time in ms and persist it.

        Damiao only; raises MotorError on MyActuator.
        """
        await self._driver.set_can_timeout(milliseconds)

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
            kd:    Velocity damping. The motor clamps this to its firmware's
                   range: [0, 5] on Damiao and legacy MyActuator, [0, 50] on
                   newer (V4.4+) MyActuator firmware. MyActuator detects this
                   on enable() and scales the command to match.
            t_ff:  Feedforward torque (Nm)
        """
        if self.mode != ControlMode.IMPEDANCE:
            raise RuntimeError(
                f"{self.joint} is in mode {self.mode}, expected IMPEDANCE. "
                f"Call set_control_mode(ControlMode.IMPEDANCE) first."
            )
        await self._driver.set_impedance(p_des, v_des, kp, kd, t_ff)
