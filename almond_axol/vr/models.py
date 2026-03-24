"""
Data models for VR teleoperation frames.

The VR headset sends JSON matching VRFrame over the WebSocket connection.
"""

from __future__ import annotations

from pydantic import BaseModel


class VRPose(BaseModel):
    """6-DOF pose from a VR controller.

    Position is in metres; orientation is a unit quaternion.
    """

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class VRFrame(BaseModel):
    """Single teleoperation frame sent by the VR headset.

    Attributes:
        left:    Left controller pose (position + orientation).
        right:   Right controller pose (position + orientation).
        l_grip:  Left gripper command — 0.0 = fully closed, 1.0 = fully open.
        r_grip:  Right gripper command — 0.0 = fully closed, 1.0 = fully open.
        l_lock:  Left deadman switch; only track left arm movement while True.
        r_lock:  Right deadman switch; only track right arm movement while True.
        reset:   Rising edge (False → True) triggers a reset to rest pose.
    """

    left: VRPose
    right: VRPose
    l_grip: float = 1.0
    r_grip: float = 1.0
    l_lock: bool = False
    r_lock: bool = False
    reset: bool = False
