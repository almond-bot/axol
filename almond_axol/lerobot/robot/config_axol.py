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
                          "left_arm", "right_arm"). Each ``ZedCameraConfig``
                          requires the camera's serial number. On the CLI the
                          dict is one inline YAML/JSON value (e.g.
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

    def observation_cameras(self) -> dict[str, tuple[CameraConfig, str | None]]:
        """Effective observation cameras keyed by dataset/obs name.

        A mono camera ``X`` maps to ``X -> (cfg, None)``. A stereo camera
        (``ZedCameraConfig.stereo``) expands into one or both eyes depending on
        ``ZedCameraConfig.eyes``: ``"both"`` -> ``X_left`` and ``X_right``,
        ``"left"`` -> ``X_left`` only, ``"right"`` -> ``X_right`` only. Eyes of
        the same camera share the same config object (one decode). Used to build
        the camera set and the dataset observation features so both agree on the
        keys.
        """
        out: dict[str, tuple[CameraConfig, str | None]] = {}
        for name, cfg in self.cameras.items():
            if getattr(cfg, "stereo", False):
                eyes = getattr(cfg, "eyes", "both")
                if eyes in ("both", "left"):
                    out[f"{name}_left"] = (cfg, "left")
                if eyes in ("both", "right"):
                    out[f"{name}_right"] = (cfg, "right")
            else:
                out[name] = (cfg, None)
        return out
