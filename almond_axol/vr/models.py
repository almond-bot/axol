"""
Data models for VR teleoperation frames.

The VR headset sends JSON matching VRFrame over the WebSocket connection.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class VRState(str, Enum):
    """Teleoperation session state for data collection.

    DATA_COLLECTION: Teleoperating and collecting data without recording.
    TELEOP:    Actively teleoperating the arm.
    RECORDING: Teleoperating and recording a demonstration.
    SAVING:    Episode saved — server is writing frames. Controls blocked.
    ERROR:     An unrecoverable server-side error occurred.
    """

    TELEOP = "teleop"
    DATA_COLLECTION = "data_collection"
    RECORDING = "recording"
    SAVING = "saving"
    ERROR = "error"


class VRPosition(BaseModel):
    """3-DOF position in metres."""

    x: float
    y: float
    z: float


class VRQuaternion(BaseModel):
    """Unit quaternion orientation."""

    x: float
    y: float
    z: float
    w: float


class VRPose(BaseModel):
    """6-DOF pose from a VR controller.

    Position is in metres; orientation is a unit quaternion.
    """

    position: VRPosition
    quaternion: VRQuaternion


class VRFrame(BaseModel):
    """Single teleoperation frame sent by the VR headset.

    Attributes:
        l_ee:    Left end-effector pose (position + orientation).
        r_ee:    Right end-effector pose (position + orientation).
        l_elbow: Left elbow position.
        r_elbow: Right elbow position.
        l_grip:  Left gripper command — 0.0 = fully closed, 1.0 = fully open.
        r_grip:  Right gripper command — 0.0 = fully closed, 1.0 = fully open.
        l_lock:  Left grip button state (True = pressed). VRTeleop uses rising
            edges of both buttons together to enable tracking, and a rising edge
            of either button alone to disable it.
        r_lock:  Right grip button state (True = pressed). See l_lock.
        reset:   Rising edge (False → True) triggers a reset to rest pose.
        state:   Current teleoperation session state (data_collection / teleop / recording).
        seq:     Monotonic sender-side frame sequence number, if provided.
        sent_at_ms: Sender-side ``performance.now()`` timestamp in milliseconds,
            useful only for comparing deltas from the same headset clock.
        client_dt_ms: Sender-side milliseconds since the previous transmitted
            frame, if provided.
        client_dropped_since_last: Number of pose frames intentionally skipped
            by the browser due to pose-channel backpressure since the previous
            transmitted frame.
        client_buffered_amount: Browser data-channel buffered bytes immediately
            before the transmitted frame was queued.
        client_send_copies: Number of redundant copies the browser attempted
            for this frame on the pose data channel.
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
    state: VRState = VRState.TELEOP
    seq: int | None = None
    sent_at_ms: float | None = None
    client_dt_ms: float | None = None
    client_dropped_since_last: int = 0
    client_buffered_amount: int | None = None
    client_send_copies: int = 1
