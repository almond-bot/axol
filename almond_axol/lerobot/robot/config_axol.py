"""Configuration dataclass for the Axol dual-arm robot as a LeRobot Robot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

from ...cli.config import register_literal
from ...robot.config import AxolConfig
from ...utils.shared import CAN_LEFT, CAN_RIGHT

# Camera capture backend. "gst" is the GPU-resident zed-gstreamer pipeline
# (low latency, shared with teleop); "sdk" is the ZED Python SDK; "auto"
# prefers gst when its stack is installed and falls back to the SDK.
# Registered with draccus so it decodes/validates on the CLI.
VideoBackend = register_literal(Literal["auto", "gst", "sdk"])


@RobotConfig.register_subclass("axol")
@dataclass
class AxolRobotConfig(RobotConfig):
    """Configuration for the Axol dual-arm robot as a LeRobot Robot.

    Args:
        cameras:          Camera configs keyed by name (e.g. "overhead",
                          "left_arm", "right_arm"). collect-data / run-policy
                          seed all three slots so each stays addressable, but
                          only the slots given a serial are actually used —
                          ``select_assigned_cameras`` drops the rest. On the
                          CLI the dict is one inline YAML/JSON value (e.g.
                          ``--robot_config.cameras "{overhead: {serial:
                          41234567}}"``).
        axol_config:      Per-joint gain config forwarded to the Axol hardware driver.
        telemetry_hz:     Background telemetry polling rate in Hz. Set to ``0``
                          (or below) to skip the poll loop entirely and rely on
                          ``motion_control`` command replies to keep the
                          position/torque cache fresh — matching ``axol teleop``.
                          Only safe when a ``motion_control`` loop runs every
                          step (e.g. ``collect-data``); otherwise the cache goes
                          stale between commands.
        observe_torques:  Include joint torques in observations. Default False.
        left_channel:     SocketCAN interface for the left arm.
        right_channel:    SocketCAN interface for the right arm.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    axol_config: AxolConfig = field(default_factory=AxolConfig)
    telemetry_hz: float = 120.0
    observe_torques: bool = False
    left_channel: str = CAN_LEFT
    right_channel: str = CAN_RIGHT
    video_backend: VideoBackend = "auto"

    def select_assigned_cameras(self, *, minimum: int = 1) -> None:
        """Drop unassigned camera slots (serial ``<= 0``), requiring ``minimum``.

        collect-data / run-policy seed all three slots (overhead, left_arm,
        right_arm) as placeholders so each is reachable as a dotted
        ``--robot_config.cameras.<slot>.serial`` override / control-panel
        field, but the operator only needs the cameras they actually have. This
        prunes the slots still at the unassigned sentinel serial (``0``) in
        place and raises if fewer than ``minimum`` real cameras remain.
        """
        assigned = {
            name: cfg
            for name, cfg in self.cameras.items()
            if int(getattr(cfg, "serial", 0) or 0) > 0
        }
        if len(assigned) < minimum:
            raise ValueError(
                f"At least {minimum} camera serial must be assigned; got "
                f"{len(assigned)}. Assign a ZED serial to a slot (overhead, "
                "left_arm, or right_arm) — on the CLI pass e.g. "
                '--robot_config.cameras "{overhead: {serial: 41234567}}", or '
                "use the control panel's Cameras dialog."
            )
        self.cameras = assigned

    def apply_detected_stereo(self, detected: set[int]) -> None:
        """Flag physically-stereo ZED X cameras the config left as mono.

        A ZED X is stereo hardware; opening it on the mono grab path
        (``zedxonesrc`` / a mono ``ZedCamera``) fails. Rather than make
        operators set ``stereo`` by hand, any assigned camera whose serial is in
        ``detected`` (see :func:`almond_axol.zed.stereo_serials`) is promoted
        here, so every capture path opens it identically — the collect-data
        relay *and* the in-process robot used by run-policy and the collect-data
        fallback. Without this, run-policy / the fallback would open a stereo
        wrist as mono and fail ``connect``.

        The eye policy follows the head/wrist convention: the overhead records
        both eyes (``overhead_left`` / ``overhead_right``), while a wrist records
        a single eye under its plain slot name, so a stereo wrist costs and
        records exactly like a mono one. An explicit ``stereo`` flag (set in the
        config) is left untouched so a deliberate eye choice still wins.
        """
        for name, cfg in self.cameras.items():
            if getattr(cfg, "stereo", False):
                continue
            if int(getattr(cfg, "serial", 0) or 0) in detected:
                cfg.stereo = True
                cfg.eyes = "both" if name == "overhead" else "left"

    def prepare_capture_cameras(self, detected: set[int], *, minimum: int = 1) -> None:
        """Finalize the camera set before any capture path opens it.

        Single entry point for the camera setup that **must be identical**
        between collect-data and run-policy: a policy only sees the observations
        it was trained on if both commands prune and flag cameras the same way.
        Prunes the unassigned slots (:meth:`select_assigned_cameras`) and then
        auto-promotes physically-stereo ZED X units
        (:meth:`apply_detected_stereo`) — order matters, so we only enumerate
        stereo over the slots that survived. ``detected`` comes from
        :func:`almond_axol.zed.stereo_serials`.
        """
        self.select_assigned_cameras(minimum=minimum)
        self.apply_detected_stereo(detected)

    def observation_cameras(self) -> dict[str, tuple[CameraConfig, str | None]]:
        """Effective observation cameras keyed by dataset/obs name.

        A mono camera ``X`` maps to ``X -> (cfg, None)``. A stereo camera
        (``ZedCameraConfig.stereo``) keyed by ``ZedCameraConfig.eyes``:

        - ``"both"`` -> ``X_left`` and ``X_right`` (the head camera).
        - ``"left"`` / ``"right"`` -> a single eye under the **plain** name
          ``X`` (the wrist policy), so one feed is indistinguishable from a mono
          camera downstream — matching the relay's ``eye_suffix=False`` export.

        Eyes of the same camera share the same config object (one decode). Used
        to build the camera set and the dataset observation features so both
        agree on the keys.
        """
        out: dict[str, tuple[CameraConfig, str | None]] = {}
        for name, cfg in self.cameras.items():
            # A camera set to stream-only (record=False) is opened to feed the
            # headset but kept out of the recorded dataset entirely.
            if not getattr(cfg, "record", True):
                continue
            if getattr(cfg, "stereo", False):
                eyes = getattr(cfg, "eyes", "both")
                if eyes != "both":
                    # Single-eye stereo records under the plain slot name.
                    out[name] = (cfg, eyes)
                    continue
                out[f"{name}_left"] = (cfg, "left")
                out[f"{name}_right"] = (cfg, "right")
            else:
                out[name] = (cfg, None)
        return out
