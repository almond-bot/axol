"""Abstract base class defining the motor driver interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Mapping

from .config import Access, MotorParam, ParamSpec
from .errors import MotorError
from .types import ControlMode, MotorGains, MotorStatus


class MotorDriver(ABC):
    """Abstract base class for a single CAN motor driver.

    Concrete subclasses :class:`DamiaoMotor` and :class:`MyActuatorMotor`
    implement the protocol-specific CAN framing. All position values are in
    radians; all velocity values are in rad/s. Public methods defined here
    carry the authoritative docstrings inherited by both implementations.
    """

    # Push-based feedback callback, installed via ``set_feedback_callback``.
    # Subclasses initialize this to ``None`` in ``__init__`` and invoke it
    # from their CAN message handler when a feedback frame arrives.
    _on_feedback: Callable[[float, float], None] | None = None

    def set_feedback_callback(
        self, callback: Callable[[float, float], None] | None
    ) -> None:
        """Register a callback fired with ``(position, torque)`` on each feedback frame.

        This is the push-based counterpart to :meth:`get_telemetry`'s
        pull-based ``on_position`` / ``on_torque`` callbacks: drivers invoke
        it whenever an unsolicited or command-response feedback frame arrives
        (for example the response to :meth:`set_impedance`), letting callers
        cache the latest position/torque without an explicit telemetry
        round-trip. Pass ``None`` to clear the callback.

        Args:
            callback: Receives ``(position_rad, torque_nm)``, or ``None`` to
                disable feedback callbacks.
        """
        self._on_feedback = callback

    @abstractmethod
    async def enable(self) -> None:
        """Enable the motor and release the brake."""
        ...

    @abstractmethod
    async def attach(self) -> None:
        """Prepare to command an already-enabled motor without disturbing it.

        The reconnect counterpart to :meth:`enable`, for when a previous
        process left the motor enabled and holding (e.g. it died mid-session).
        Re-reads whatever state the driver needs for correct command scaling
        (limit registers, firmware capabilities) and verifies the motor is
        enabled and fault-free — but sends no reset, brake, enable, or motion
        command, so a motor that is holding position keeps holding it
        throughout.

        Raises MotorError if the motor is unreachable, disabled, or faulted:
        attaching is only valid while the motor is still up, and anything else
        needs a full :meth:`enable` bring-up.
        """
        ...

    @abstractmethod
    async def disable(self) -> None:
        """Disable the motor and engage the brake."""
        ...

    @abstractmethod
    async def is_holding(self) -> bool:
        """Return True if the motor is enabled and holding torque. Read-only.

        The state probe behind the idempotent ``Axol.enable()``: a holding
        motor must not be reset (it is attached to instead), while a motor
        reporting False holds nothing and can take the full bring-up.

        Damiao: feedback status is ENABLED. Note this cannot distinguish an
        actively-holding motor from one that was enabled but never commanded
        (which holds no torque) — such a motor conservatively reports True.
        MyActuator: the status-1 running byte is set and no fault is latched;
        on fleet firmware the running byte reads 1 only while the motor is
        actively executing commands, so this matches "holding" exactly.
        """
        ...

    @abstractmethod
    async def clear_errors(self) -> None:
        """Clear any latched motor error flags."""
        ...

    @abstractmethod
    async def set_zero_position(self) -> None:
        """Save the current shaft position as the encoder zero reference."""
        ...

    @abstractmethod
    async def get_position(self) -> float:
        """Return current shaft position in radians."""
        ...

    @abstractmethod
    async def get_velocity(self) -> float:
        """Return current shaft velocity in radians per second."""
        ...

    @abstractmethod
    async def get_torque(self) -> float:
        """Return current torque estimate in Nm."""
        ...

    @abstractmethod
    async def get_telemetry(
        self,
        on_position: Callable[[float], None],
        on_torque: Callable[[float], None] | None = None,
    ) -> None:
        """Fetch position and optionally torque, calling each callback as soon as its value arrives.

        Damiao: one feedback request — position always fetched, torque skipped if on_torque is None.
        MyActuator: position always fetched; torque request skipped if on_torque is None.
        """
        ...

    @abstractmethod
    async def get_temperature(self) -> float:
        """Return motor temperature in degrees Celsius.

        Damiao: returns the higher of MOS and rotor temperatures.
        """
        ...

    @abstractmethod
    async def get_voltage(self) -> float:
        """Return bus voltage in Volts."""
        ...

    @abstractmethod
    async def get_error_code(self) -> MotorStatus:
        """Return the current motor status / error code."""
        ...

    async def get_low_voltage_threshold(self) -> float:
        """Return the undervoltage protection threshold in Volts.

        Below this bus voltage the motor latches a level-2 undervoltage fault
        (:attr:`MotorStatus.UNDER_VOLTAGE`). Only supported by MyActuator
        motors; raises MotorError on Damiao.
        """
        raise MotorError(
            f"get_low_voltage_threshold is not supported by {type(self).__name__}"
        )

    async def set_low_voltage_threshold(self, volts: float) -> None:
        """Set the undervoltage protection threshold and persist it to ROM.

        Only supported by MyActuator motors; raises MotorError on Damiao.
        MyActuator applies this on every ``enable()`` — see
        ``_MA_LOW_VOLTAGE_THRESHOLD_V`` in ``myactuator.py``.

        Args:
            volts: Threshold in Volts. A negative value can never be reached,
                   which suppresses the undervoltage fault entirely.
        """
        raise MotorError(
            f"set_low_voltage_threshold is not supported by {type(self).__name__}"
        )

    #: Canonical name of this motor family, as accepted by ``make_driver``'s
    #: ``motor_type``. Config snapshots persist it and feed it back on restore,
    #: so it must be a declared identity rather than anything derived from the
    #: class name.
    MOTOR_TYPE: str = ""

    #: This driver's configuration parameter table, keyed by its own parameter
    #: enum. Empty for drivers with no configuration surface.
    PARAMS: Mapping[MotorParam, ParamSpec] = {}

    @classmethod
    def resolve_param(cls, name: str) -> MotorParam:
        """Look a parameter up by name in *this* driver's table.

        Names are unique across motor families, so naming a Damiao register
        while talking to a MyActuator motor is a mistake worth reporting rather
        than silently matching some same-valued index in the other table.
        """
        for param in cls.PARAMS:
            if param.name == name:
                return param
        raise MotorError(f"{cls.__name__} has no configuration parameter {name!r}")

    #: Bare indices a raw dump may sweep, for families whose table is still
    #: incomplete. ``None`` where every parameter is already known, so there is
    #: nothing a sweep could discover.
    PARAM_SWEEP_RANGE: range | None = None

    async def get_can_timeout(self) -> float:
        """Return the CAN loss-of-comms alarm time in milliseconds.

        Only supported by Damiao motors; raises MotorError on MyActuator.
        """
        raise MotorError(f"get_can_timeout is not supported by {type(self).__name__}")

    async def set_can_timeout(self, milliseconds: float) -> None:
        """Set the CAN loss-of-comms alarm time in ms and persist it.

        The motor faults with :attr:`MotorStatus.LOST_COMM` when it goes this
        long without a command; 0 disables the alarm. Only supported by Damiao
        motors; raises MotorError on MyActuator.
        """
        raise MotorError(f"set_can_timeout is not supported by {type(self).__name__}")

    async def _config_read(self, index: int) -> float:
        """Read one parameter by raw index, in the motor's own units."""
        raise MotorError(f"config access is not supported by {type(self).__name__}")

    async def _config_write(self, index: int, value: float) -> None:
        """Write one parameter by raw index without persisting it."""
        raise MotorError(f"config access is not supported by {type(self).__name__}")

    async def _config_commit(self) -> None:
        """Persist every deferred config write to flash/ROM."""
        raise MotorError(f"config access is not supported by {type(self).__name__}")

    async def read_config(self, param: MotorParam) -> float:
        """Read one configuration parameter, in the unit its spec names."""
        raw = await self._config_read(int(param))
        return raw * self.PARAMS[param].scale

    async def write_config(self, param: MotorParam, value: float) -> None:
        """Write one configuration parameter and persist it to flash/ROM.

        Refuses read-only parameters; protected ones are allowed here because
        naming a single parameter is already deliberate. Bulk restores gate
        them behind ``include_protected`` instead.
        """
        spec = self.PARAMS[param]
        if spec.access is Access.READ_ONLY:
            raise MotorError(f"{param.name} is read-only")
        await self._config_write(int(param), value / spec.scale)
        await self._config_commit()

    async def dump_config(
        self, raw_range: range | None = None
    ) -> dict[MotorParam | int, float]:
        """Read every known configuration parameter.

        Pass ``raw_range`` to sweep bare indices instead of the known table —
        the way to pin down parameters a vendor GUI exposes but whose indices
        are still unidentified. Indices the motor rejects are left out rather
        than reported as an error, since a sweep is expected to hit gaps.
        """
        if not self.PARAMS:
            raise MotorError(f"dump_config is not supported by {type(self).__name__}")
        if raw_range is not None:
            values: dict[MotorParam | int, float] = {}
            for index in raw_range:
                try:
                    values[index] = await self._config_read(index)
                except MotorError:
                    continue
            return values
        return {param: await self.read_config(param) for param in self.PARAMS}

    async def restore_config(
        self, values: Mapping[MotorParam, float], *, include_protected: bool = False
    ) -> list[MotorParam]:
        """Write ``values`` back to the motor and return the ones that changed.

        Parameters already holding the requested value are skipped so a restore
        onto a matching motor costs no flash writes, and the whole batch shares
        a single commit. Read-only parameters are always skipped, and protected
        ones unless ``include_protected`` is set — see :class:`Access`.
        """
        if not self.PARAMS:
            raise MotorError(
                f"restore_config is not supported by {type(self).__name__}"
            )
        written: list[MotorParam] = []
        for param, value in values.items():
            spec = self.PARAMS.get(param)
            if spec is None or spec.access is Access.READ_ONLY:
                continue
            if spec.access is Access.PROTECTED and not include_protected:
                continue
            if math.isclose(await self.read_config(param), value, rel_tol=1e-6):
                continue
            await self._config_write(int(param), value / spec.scale)
            written.append(param)
        if written:
            await self._config_commit()
        return written

    @abstractmethod
    async def set_position_velocity(self, position: float, max_speed: float) -> None:
        """Move to an absolute position using the motor's built-in position controller.

        Args:
            position:  Target shaft position (rad)
            max_speed: Maximum speed during the move (rad/s)
        """
        ...

    @abstractmethod
    async def set_velocity(self, velocity: float) -> None:
        """Command a target velocity using the motor's built-in speed controller.

        Args:
            velocity: Target shaft velocity (rad/s)
        """
        ...

    async def get_control_mode(self) -> ControlMode | None:
        """Return the active control mode read from hardware, or None if unsupported.

        Damiao: reads register 10 and returns the matching ControlMode.
        MyActuator: returns None — the mode is implicit in each command sent.
        """
        return None

    async def get_firmware_version(self) -> int | None:
        """Return the motor firmware version, or None if unsupported.

        MyActuator: the VersionDate (uint32, e.g. ``2026042402``) read via 0xB2;
        this value gates which MIT-command ranges the driver uses.
        Damiao: None — not exposed by the protocol.
        """
        return None

    async def get_model(self) -> str | None:
        """Return the motor model string, or None if unsupported.

        MyActuator: the model string read via 0xB5 (e.g. ``"X8S2V"``), whose
        series determines the motor's rated max torque.
        Damiao: None — not exposed by the protocol.
        """
        return None

    async def set_position_force(
        self, position: float, max_speed: float, max_torque: float
    ) -> None:
        """Move to a position with hard speed and torque limits.

        Only supported by Damiao motors. Raises MotorError on MyActuator.

        Args:
            position:   Target shaft position (rad)
            max_speed:  Maximum speed during the move (rad/s)
            max_torque: Maximum output torque (Nm)
        """
        raise MotorError(
            f"set_position_force is not supported by {type(self).__name__}"
        )

    @abstractmethod
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
        ...

    @abstractmethod
    async def get_gains(self) -> MotorGains:
        """Read the stored PID gains for the speed and position control loops."""
        ...

    @abstractmethod
    async def set_gains(self, gains: MotorGains) -> None:
        """Write PID gains for the speed and position control loops.

        Changes are persisted to non-volatile memory so they survive power cycles.

        Args:
            gains: Gain values to write. Damiao ignores current_kp / current_ki.
        """
        ...

    @abstractmethod
    async def set_impedance(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
    ) -> None:
        """Send an impedance control command.

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
        ...

    @abstractmethod
    async def set_control_mode(self, mode: ControlMode) -> None:
        """Set the active control mode.

        Damiao: writes register 10 to match the requested mode.
        MyActuator: has no persistent mode register — resets the motor instead
        so it comes back in a clean state; the next command determines the mode.

        Args:
            mode: Desired control mode.
        """
        ...

    @abstractmethod
    async def set_can_id(self, can_id: int) -> None:
        """Change the motor's CAN ID and persist it to flash.

        The change takes effect immediately on the motor; subsequent commands
        must use the new ID (the driver updates its internal state automatically).

        Damiao: also sets the feedback ID to can_id + 0x10.

        Args:
            can_id: New CAN ID for the motor.
        """
        ...
