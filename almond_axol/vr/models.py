"""
Data models for VR teleoperation frames.

The VR headset sends JSON matching VRFrame over the WebSocket connection.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, FiniteFloat, model_validator


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


class VREpisodeOutcome(str, Enum):
    """Operator-supplied outcome attached to an episode-ending transition."""

    SUCCESS = "success"
    FAILURE = "failure"


class VRPosition(BaseModel):
    """3-DOF position in metres."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class VRQuaternion(BaseModel):
    """Unit quaternion orientation."""

    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
    w: FiniteFloat

    @model_validator(mode="after")
    def normalize(self) -> VRQuaternion:
        """Reject degenerate rotations and canonicalize valid wire input."""
        norm = math.hypot(self.x, self.y, self.z, self.w)
        if norm <= 1e-12:
            raise ValueError("quaternion norm must be nonzero")
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.w /= norm
        return self


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
        episode_outcome: Outcome attached to a RECORDING → DATA_COLLECTION
            transition. ``None`` keeps the legacy controller behavior (success
            unless ``reset`` requests re-recording); tracker trigger gestures
            set this explicitly so failure remains distinct from re-record.
        episode_end_t_host: Host ``time.perf_counter`` instant (seconds) at
            which a saved episode's data should end, sent alongside a
            successful ``episode_outcome``. A local tracker bridge sets it to
            the moment the operator *began* the end gesture (the first of the
            three trigger clicks) so the collector can trim the rows captured
            after it — otherwise the clicks, and the gripper commands they
            are, would close every saved take. ``None`` keeps every row (Quest
            and older clients).
        lock_release_id: Internal managed-tracker handshake. When set on a
            frame whose lock bits are both false, the teleop core echoes the
            identifier after it has consumed that release. This lets the
            bridge wait for a real low→high edge even while IK is blocked.
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
        t_host:  Estimated capture time of this frame's poses on the *host*
            clock (``time.perf_counter`` seconds), stamped server-side by the
            pose interpolator from ``t`` and its headset↔host clock-offset
            estimate. This is the timestamp dataset rows should be aligned to
            when the pose stream is the ground truth (Mantis recording): it names
            when the hand actually was at this pose, not when the frame was
            played out. ``None`` until the interpolator has seen the frame.
        pose_source_id: Logical producer identity shared by every transport
            carrying this frame. Quest sends the same id over USB, WebRTC,
            and network WebSocket so sequence de-duplication is global to the
            headset rather than per socket. Tracker bridges use one id across
            reconnects. Optional for compatibility with older clients.
        pose_source_kind: ``"webxr"`` for Quest or ``"tracker"`` for a local
            Lighthouse/Ultimate bridge. A managed Mantis server admits only
            its configured kind, allowing a Quest to display video/URDF state
            without its controller frames taking control.
        l_pose_profile: First (most preferred) WebXR input profile reported by
            the left Quest controller, such as ``"oculus-touch-v3"``. Used
            with ``l_pose_space`` to ensure a calibrated mount transform is
            never applied to another controller generation or local datum.
        r_pose_profile: Same for the right Quest controller.
        l_pose_space: ``"target-ray"`` for legacy relative Axol control;
            ``"grip"`` when absolute/Mantis ``l_ee`` came from WebXR
            ``gripSpace``. Absolute compatibility runtimes may report
            ``"target-ray"`` when they lack gripSpace, but production Quest
            collection requires the calibrated grip datum.
        r_pose_space: Same for the right Quest controller.
        l_tracked: True while the left controller's position is optically
            tracked. ``False`` means the headset lost sight of the controller
            (occlusion, edge of camera FOV, very fast motion) and the reported
            position is inertially dead-reckoned (WebXR ``emulatedPosition``) —
            it drifts while coasting and snaps when tracking re-acquires, so
            the server's pose smoother excludes these frames and holds the last
            clean pose instead. Defaults to True so older web builds (which
            omit the field) keep the previous always-trusted behaviour.
        r_tracked: Same as ``l_tracked`` for the right controller.
        l_trigger_live: True while the left Mantis trigger node is delivering
            fresh CAN frames. Managed tracker bridges set this False after a
            meaningful input dropout while holding the last safe grip command.
            Defaults to True for Quest and older clients that do not use a
            separate CAN trigger.
        r_trigger_live: Same as ``l_trigger_live`` for the right trigger.
        l_stick_x: Left thumbstick x, [-1, 1], right = +1. With ``l_stick_y``
            it drives the powered cart's translation when one is configured
            (see :class:`almond_axol.robot.cart.Cart`). Neutral defaults keep
            older web builds (which omit the stick fields) fully compatible.
        l_stick_y: Left thumbstick y, [-1, 1], pushed forward = -1 (WebXR
            xr-standard convention, same as a gamepad).
        r_stick_x: Right thumbstick x, [-1, 1], right = +1. Drives the cart's
            rotation.
        l_stick_click: Left thumbstick pressed in — lift down while held.
        r_stick_click: Right thumbstick pressed in — lift up while held.
    """

    l_ee: VRPose
    r_ee: VRPose
    l_elbow: VRPosition
    r_elbow: VRPosition
    l_grip: FiniteFloat = 1.0
    r_grip: FiniteFloat = 1.0
    l_lock: bool = False
    r_lock: bool = False
    reset: bool = False
    state: VRState = VRState.TELEOP
    episode_outcome: VREpisodeOutcome | None = None
    episode_end_t_host: FiniteFloat | None = None
    lock_release_id: int | None = None
    t: FiniteFloat | None = None
    seq: int | None = None
    t_host: FiniteFloat | None = None
    pose_source_id: str | None = None
    pose_source_kind: str | None = None
    l_pose_profile: str | None = None
    r_pose_profile: str | None = None
    l_pose_space: str | None = None
    r_pose_space: str | None = None
    l_tracked: bool = True
    r_tracked: bool = True
    l_trigger_live: bool = True
    r_trigger_live: bool = True
    l_stick_x: FiniteFloat = 0.0
    l_stick_y: FiniteFloat = 0.0
    r_stick_x: FiniteFloat = 0.0
    l_stick_click: bool = False
    r_stick_click: bool = False
