"""
Data models for VR teleoperation frames.

The VR headset sends JSON matching VRFrame over the WebSocket connection.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class VRState(str, Enum):
    """Teleoperation session state for data collection.

    IDLE:      No active teleoperation or recording.
    TELEOP:    Actively teleoperating the arm.
    RECORDING: Teleoperating and recording a demonstration.
    """

    IDLE = "idle"
    TELEOP = "teleop"
    RECORDING = "recording"


class VRPosition(BaseModel):
    """3-DOF position in metres."""

    x: float
    y: float
    z: float


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
        l_ee:    Left end-effector pose (position + orientation).
        r_ee:    Right end-effector pose (position + orientation).
        l_elbow: Left elbow position.
        r_elbow: Right elbow position.
        l_grip:  Left gripper command — 0.0 = fully closed, 1.0 = fully open.
        r_grip:  Right gripper command — 0.0 = fully closed, 1.0 = fully open.
        l_lock:  Left deadman switch; only track left arm movement while True.
        r_lock:  Right deadman switch; only track right arm movement while True.
        reset:   Rising edge (False → True) triggers a reset to rest pose.
        state:   Current teleoperation session state (idle / teleop / recording).
    """

    l_ee: VRPose
    r_ee: VRPose
    l_elbow: VRPosition
    r_elbow: VRPosition
    l_grip: float = 1.0
    r_grip: float = 1.0
    l_lock: bool = False
    r_lock: bool = False
    reset: bool = False
    state: VRState = VRState.IDLE  # Current teleoperation session state.
