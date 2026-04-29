"""Abstract base class defining the motor driver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .errors import MotorError
from .types import ControlMode, MotorGains, MotorStatus


class MotorDriver(ABC):
    """Abstract base class for a single CAN motor driver.

    Concrete subclasses :class:`DamiaoMotor` and :class:`MyActuatorMotor`
    implement the protocol-specific CAN framing. All position values are in
    radians; all velocity values are in rad/s. Public methods defined here
    carry the authoritative docstrings inherited by both implementations.
    """

    @abstractmethod
    async def enable(self) -> None:
        """Enable the motor and release the brake."""
        ...

    @abstractmethod
    async def disable(self) -> None:
        """Disable the motor and engage the brake."""
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
            kd:    Velocity damping   [0, 5]
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

    @abstractmethod
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
        ...
