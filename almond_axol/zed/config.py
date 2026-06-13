"""ZedConfig dataclass for the ZED-X One camera streamer."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyzed.sl as sl

# Default HEVC bitrate (kbits/s) per capture resolution at 60 fps, informed by
# Stereolabs' recommended streaming bitrates. Keys match ``sl.RESOLUTION``
# member names.
AUTO_BITRATE_KBPS: dict[str, int] = {
    "SVGA": 8000,
    "HD1080": 12500,
    "HD1200": 14000,
}

# A stereo ZED X streams both eyes side-by-side on one feed (double the
# pixels), so its auto bitrate gets a bump over the mono table above.
STEREO_BITRATE_FACTOR = 1.5


def auto_bitrate(resolution: sl.RESOLUTION, stereo: bool = False) -> int:
    """Recommended HEVC bitrate (kbits/s) for ``resolution``.

    Args:
        resolution: Capture resolution the camera streams at.
        stereo:     Whether the stream carries both eyes (side-by-side).
    """
    name = str(resolution).split(".")[-1]
    kbps = AUTO_BITRATE_KBPS.get(name, 8000)
    return int(kbps * STEREO_BITRATE_FACTOR) if stereo else kbps


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
        resolution:       Capture resolution for all cameras (default SVGA).
        fps:              Capture frame rate for all cameras (default 60).
        bitrate:          HEVC encoding bitrate in kbits/s. ``None`` (the
                          default) picks a recommended bitrate automatically
                          from the resolution (see ``AUTO_BITRATE_KBPS``),
                          with a bump for the stereo overhead.
        overhead_stereo:  Treat the overhead camera as a stereo ZED X
                          (``sl.Camera``) instead of a mono ZED-X One
                          (``sl.CameraOne``). The single stream then carries
                          both eyes; receivers retrieve LEFT/RIGHT. The wrist
                          cameras are always mono. Default False.
    """

    overhead_serial: int | None = None
    left_arm_serial: int | None = None
    right_arm_serial: int | None = None
    overhead_port: int = 30000
    left_arm_port: int = 30002
    right_arm_port: int = 30004
    resolution: sl.RESOLUTION = field(default_factory=lambda: sl.RESOLUTION.SVGA)
    fps: int = 60
    bitrate: int | None = None
    overhead_stereo: bool = False

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
