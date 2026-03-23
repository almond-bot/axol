"""
almond_axol.motor

Async motor interface for the Almond Axol arm.

Public API
──────────
    CanBus      Shared async SocketCAN bus
    Motor       Unified motor interface (constructed from a Joint)
    Joint       Enum of all arm joints
    MotorError  Raised when a motor command fails or times out

Usage
─────
    async with CanBus("can_alm_axol_l") as bus:
        shoulder = Motor(bus, Joint.SHOULDER_1)
        wrist2   = Motor(bus, Joint.WRIST_2)

        await shoulder.enable()
        pos = await shoulder.get_position()  # revolutions
"""

from .errors import MotorError
from .bus import CanBus
from .motor import Joint, Motor

__all__ = ["CanBus", "Motor", "Joint", "MotorError"]
