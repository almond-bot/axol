from __future__ import annotations

from abc import ABC, abstractmethod

from .errors import MotorError
from .types import MotorGains, MotorStatus


class MotorDriver(ABC):
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
        """Return current shaft position in revolutions."""
        ...

    @abstractmethod
    async def get_velocity(self) -> float:
        """Return current shaft velocity in revolutions per second."""
        ...

    @abstractmethod
    async def get_torque(self) -> float:
        """Return current torque estimate.

        Damiao: estimated output torque in Nm.
        MyActuator: phase current in Amperes (multiply by motor Kt for Nm).
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
    async def set_position(self, position: float, max_speed: float) -> None:
        """Move to an absolute position using the motor's built-in position controller.

        Args:
            position:  Target shaft position (rev)
            max_speed: Maximum speed during the move (rev/s)
        """
        ...

    @abstractmethod
    async def set_velocity(self, velocity: float) -> None:
        """Command a target velocity using the motor's built-in speed controller.

        Args:
            velocity: Target shaft velocity (rev/s)
        """
        ...

    async def set_force_position(
        self, position: float, max_speed: float, max_current: float
    ) -> None:
        """Move to a position with hard speed and current limits.

        Only supported by Damiao motors. Raises MotorError on MyActuator.

        Args:
            position:    Target shaft position (rev)
            max_speed:   Maximum speed during the move (rev/s)
            max_current: Maximum phase current, normalized [0.0, 1.0]
        """
        raise MotorError(
            f"set_force_position is not supported by {type(self).__name__}"
        )

    @abstractmethod
    async def set_acceleration(
        self, acceleration: float, deceleration: float | None = None
    ) -> None:
        """Set the acceleration ramp for position and velocity control modes.

        Args:
            acceleration: Acceleration ramp (rev/s²)
            deceleration: Deceleration ramp (rev/s²). If None, matches acceleration.
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
    async def motion_control(
        self,
        p_des: float,
        v_des: float,
        kp: float,
        kd: float,
        t_ff: float,
    ) -> None:
        """Send an MIT-style impedance control command.

        Args:
            p_des: Desired position (rev)
            v_des: Desired velocity (rev/s)
            kp:    Position stiffness [0, 500]
            kd:    Velocity damping   [0, 5]
            t_ff:  Feedforward torque (Nm)
        """
        ...
