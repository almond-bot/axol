from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bus import CanBus
from .damaio import DamaioMotor
from .driver import MotorDriver
from .myactuator import MyActuatorMotor


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
    DAMAIO = "damaio"


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
    Joint.WRIST_2: _JointConfig(_MotorType.DAMAIO, motor_id=0x06),
    Joint.WRIST_3: _JointConfig(_MotorType.DAMAIO, motor_id=0x07),
    Joint.GRIPPER: _JointConfig(_MotorType.DAMAIO, motor_id=0x08),
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

    def __init__(self, bus: CanBus, joint: Joint) -> None:
        self.joint = joint
        cfg = _JOINT_CONFIG[joint]
        self._driver: MotorDriver
        if cfg.kind == _MotorType.MYACTUATOR:
            self._driver = MyActuatorMotor(bus, cfg.motor_id)
        else:
            self._driver = DamaioMotor(
                bus, cfg.motor_id, feedback_id=0x10 + cfg.motor_id
            )

    async def enable(self) -> None:
        await self._driver.enable()

    async def disable(self) -> None:
        await self._driver.disable()

    async def clear_errors(self) -> None:
        await self._driver.clear_errors()

    async def get_position(self) -> float:
        """Return current shaft position in revolutions."""
        return await self._driver.get_position()
