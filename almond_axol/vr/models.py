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
        l_lock:  Left grip button state (True = pressed). VRTeleop engages
            tracking on both buttons together (a rising edge, or a hold with
            ``hold_to_engage``); once engaged each button acts per-arm —
            toggling (or, in hold mode, holding) that arm between tracking
            and frozen. See :meth:`VRTeleopCore.update_engage`.
        r_lock:  Right grip button state (True = pressed). See l_lock.
        reset:   Rising edge (False → True) triggers a reset to rest pose.
        state:   Current teleoperation session state (data_collection / teleop / recording).
        t:       Client capture timestamp in milliseconds (``performance.now()``).
            Used by the server's pose interpolator to reconstruct the true motion
            cadence when frames arrive batched/jittered over the network. Optional:
            transports that don't stamp it (an older web build) fall back to
            "latest-wins" with no interpolation.
        seq:     Monotonically increasing frame counter set by the headset. The
            headset streams identical frames (same ``seq``) over both the USB
            and network transports; the server processes each logical frame
            exactly once, via whichever transport delivers it first. ``None``
            for senders that don't set it (then no cross-transport de-duplication).
        l_tracked: True while the left controller's position is optically
            tracked. ``False`` means the headset lost sight of the controller
            (occlusion, edge of camera FOV, very fast motion) and the reported
            position is inertially dead-reckoned (WebXR ``emulatedPosition``) —
            it drifts while coasting and snaps when tracking re-acquires, so
            the server's pose smoother excludes these frames and holds the last
            clean pose instead. Defaults to True so older web builds (which
            omit the field) keep the previous always-trusted behaviour.
        r_tracked: Same as ``l_tracked`` for the right controller.
        l_stick_x: Left thumbstick x, [-1, 1], right = +1. With ``l_stick_y``
            it drives Jelly's translation when one is configured
            (see :class:`almond_axol.robot.jelly.Jelly`). Neutral defaults keep
            older web builds (which omit the stick fields) fully compatible.
        l_stick_y: Left thumbstick y, [-1, 1], pushed forward = -1 (WebXR
            xr-standard convention, same as a gamepad).
        r_stick_x: Right thumbstick x, [-1, 1], right = +1. Drives Jelly's
            rotation.
        l_stick_click: Left thumbstick pressed in — lift down while held.
        r_stick_click: Right thumbstick pressed in — lift up while held.
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
    t: float | None = None
    seq: int | None = None
    l_tracked: bool = True
    r_tracked: bool = True
    l_stick_x: float = 0.0
    l_stick_y: float = 0.0
    r_stick_x: float = 0.0
    l_stick_click: bool = False
    r_stick_click: bool = False
