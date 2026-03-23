from __future__ import annotations

from abc import ABC, abstractmethod


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

        Damaio: estimated output torque in Nm.
        MyActuator: phase current in Amperes (multiply by motor Kt for Nm).
        """
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
