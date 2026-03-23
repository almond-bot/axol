from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bus import CanBus
from .damiao import DamiaoMotor
from .driver import MotorDriver
from .myactuator import MyActuatorMotor
from .types import MotorGains, MotorStatus


class Joint(Enum):
    SHOULDER_1 = "shoulder_1"
    SHOULDER_2 = "shoulder_2"
    SHOULDER_3 = "shoulder_3"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"
    GRIPPER = "gripper"


class _MotorType(Enum):
    MYACTUATOR = "myactuator"
    DAMIAO = "damiao"


@dataclass(frozen=True)
class _JointConfig:
    kind: _MotorType
    motor_id: int


_JOINT_CONFIG: dict[Joint, _JointConfig] = {
    Joint.SHOULDER_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x01),
    Joint.SHOULDER_2: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x02),
    Joint.SHOULDER_3: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x03),
    Joint.ELBOW: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x04),
    Joint.WRIST_1: _JointConfig(_MotorType.MYACTUATOR, motor_id=0x05),
    Joint.WRIST_2: _JointConfig(_MotorType.DAMIAO, motor_id=0x06),
    Joint.WRIST_3: _JointConfig(_MotorType.DAMIAO, motor_id=0x07),
    Joint.GRIPPER: _JointConfig(_MotorType.DAMIAO, motor_id=0x08),
}


class Motor:
    """
    Unified async motor interface.

    Instantiate with a CanBus and a Joint; the correct underlying driver
    is selected automatically based on the joint.

        motor = Motor(bus, Joint.WRIST_2)
        await motor.enable()
        pos = await motor.get_position()  # revolutions
    """

    def __init__(self, bus: CanBus, joint: Joint, can_id: int | None = None) -> None:
        self.joint = joint
        cfg = _JOINT_CONFIG[joint]
        motor_id = can_id if can_id is not None else cfg.motor_id
        self._driver: MotorDriver
        if cfg.kind == _MotorType.MYACTUATOR:
            self._driver = MyActuatorMotor(bus, motor_id)
        else:
            self._driver = DamiaoMotor(bus, motor_id, feedback_id=0x10 + motor_id)

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

    async def get_position(self) -> float:
        """Return current shaft position in revolutions."""
        return await self._driver.get_position()

    async def get_velocity(self) -> float:
        """Return current shaft velocity in revolutions per second."""
        return await self._driver.get_velocity()

    async def get_torque(self) -> float:
        """Return current torque estimate.

        Damiao: estimated output torque in Nm.
        MyActuator: phase current in Amperes (multiply by motor Kt for Nm).
        """
        return await self._driver.get_torque()

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

    async def set_position(self, position: float, max_speed: float) -> None:
        """Move to an absolute position using the motor's built-in position controller.

        Args:
            position:  Target shaft position (rev)
            max_speed: Maximum speed during the move (rev/s)
        """
        await self._driver.set_position(position, max_speed)

    async def set_velocity(self, velocity: float) -> None:
        """Command a target velocity using the motor's built-in speed controller.

        Args:
            velocity: Target shaft velocity (rev/s)
        """
        await self._driver.set_velocity(velocity)

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
        await self._driver.set_force_position(position, max_speed, max_current)

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
        await self._driver.motion_control(p_des, v_des, kp, kd, t_ff)
