"""Configuration dataclass for the Axol dual-arm robot as a LeRobot Robot."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig

from ...robot.config import AxolConfig
from ...utils.shared import CAN_LEFT, CAN_RIGHT


@RobotConfig.register_subclass("axol")
@dataclass
class AxolRobotConfig(RobotConfig):
    """Configuration for the Axol dual-arm robot as a LeRobot Robot.

    Args:
        cameras:          Camera configs keyed by name (e.g. "overhead", "left_arm", "right_arm").
        zed_host:         Shared IP of the ZED streamer. Required — there is no
                          default, so it must be supplied (e.g.
                          ``--robot_config.zed_host 10.0.0.5``). Applied to every
                          ``ZedCameraConfig`` camera that leaves its ``host``
                          unset (``None``); a camera with an explicit ``host``
                          keeps it.
        axol_config:      Per-joint gain config forwarded to the Axol hardware driver.
        telemetry_hz:     Background telemetry polling rate in Hz.
        observe_torques:  Include joint torques in observations. Default False.
        left_channel:     SocketCAN interface for the left arm.
        right_channel:    SocketCAN interface for the right arm.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Required: no default. The CLI/serve config overlay (see
    # almond_axol.cli.config) strips ``required_input`` fields so draccus
    # forces the operator to supply a value instead of falling back to one.
    zed_host: str = field(kw_only=True, metadata={"required_input": True})
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

    def observation_cameras(self) -> dict[str, tuple[CameraConfig, str | None]]:
        """Effective observation cameras keyed by dataset/obs name.

        A mono camera ``X`` maps to ``X -> (cfg, None)``. A stereo camera
        (``ZedCameraConfig.stereo``) expands into two eyes,
        ``X_left -> (cfg, "left")`` and ``X_right -> (cfg, "right")``, sharing
        the same config object (one decode). Used to build the camera set and
        the dataset observation features so both agree on the keys.
        """
        out: dict[str, tuple[CameraConfig, str | None]] = {}
        for name, cfg in self.resolved_cameras().items():
            if getattr(cfg, "stereo", False):
                out[f"{name}_left"] = (cfg, "left")
                out[f"{name}_right"] = (cfg, "right")
            else:
                out[name] = (cfg, None)
        return out
