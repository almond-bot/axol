"""Configuration dataclass for the Axol dual-arm robot as a LeRobot Robot."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

from ...robot.config import AxolConfig
from ...shared import CAN_LEFT, CAN_RIGHT


@RobotConfig.register_subclass("axol")
@dataclass
class AxolRobotConfig(RobotConfig):
    """Configuration for the Axol dual-arm robot as a LeRobot Robot.

    Args:
        cameras:          Camera configs keyed by name (e.g. "overhead", "left_arm", "right_arm").
        zed_host:         Shared IP of the ZED streamer. Applied to every
                          ``ZedCameraConfig`` camera that leaves its ``host``
                          unset (``None``), so all cameras share one host by
                          default; a camera with an explicit ``host`` keeps it.
        axol_config:      Per-joint gain config forwarded to the Axol hardware driver.
        telemetry_hz:     Background telemetry polling rate in Hz.
        observe_torques:  Include joint torques in observations. Default False.
        left_channel:     SocketCAN interface for the left arm.
        right_channel:    SocketCAN interface for the right arm.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    zed_host: str = "192.168.10.1"
    axol_config: AxolConfig = field(default_factory=AxolConfig)
    telemetry_hz: float = 120.0
    observe_torques: bool = False
    left_channel: str = CAN_LEFT
    right_channel: str = CAN_RIGHT

    def resolved_cameras(self) -> dict[str, CameraConfig]:
        """Return the camera configs with unset hosts filled from ``zed_host``.

        Resolved lazily (not in ``__post_init__``) so the shared host is
        applied to the *final* config after draccus has merged CLI/file
        overrides, rather than being baked into the default overlay.
        """
        for cam in self.cameras.values():
            if getattr(cam, "host", "") is None:
                cam.host = self.zed_host
        return self.cameras
