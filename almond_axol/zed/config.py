"""ZedConfig dataclass for the ZED-X One camera streamer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyzed.sl as sl


@dataclass
class ZedConfig:
    """Configuration for the ZED-X One camera streamer.

    At least one serial number must be provided.

    Valid resolutions for the ZED-X One UHD (IMX678 sensor):
        - ``sl.RESOLUTION.HD1200`` (1920×1200) — widest FOV, recommended
        - ``sl.RESOLUTION.HD1080`` (1920×1080)
        - ``sl.RESOLUTION.SVGA``   (960×600)  — ZED-X One GS only

    Attributes:
        overhead_serial:  Serial number of the overhead camera (optional).
        left_arm_serial:  Serial number of the left-arm camera (optional).
        right_arm_serial: Serial number of the right-arm camera (optional).
        overhead_port:    Streaming port for the overhead camera (default 30000).
        left_arm_port:    Streaming port for the left-arm camera (default 30002).
        right_arm_port:   Streaming port for the right-arm camera (default 30004).
        resolution:       Capture resolution for all cameras (default HD1080).
        fps:              Capture frame rate for all cameras (default 60).
        bitrate:          HEVC encoding bitrate in kbits/s (default 8000).
    """

    overhead_serial: int | None = None
    left_arm_serial: int | None = None
    right_arm_serial: int | None = None
    overhead_port: int = 30000
    left_arm_port: int = 30002
    right_arm_port: int = 30004
    resolution: sl.RESOLUTION = field(default_factory=lambda: sl.RESOLUTION.HD1080)
    fps: int = 60
    bitrate: int = 8000

    def __post_init__(self) -> None:
        """Validate that at least one camera serial number has been provided.

        Raises:
            ValueError: If all three serial fields are ``None``.
        """
        if (
            self.overhead_serial is None
            and self.left_arm_serial is None
            and self.right_arm_serial is None
        ):
            raise ValueError("At least one camera serial number must be provided.")
