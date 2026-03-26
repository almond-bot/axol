from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for a ZED camera stream receiver.

    Connects to a single ZED stream produced by ZedStreamer on the sender machine.
    One instance per camera (overhead, left_arm, right_arm each get their own config).

    Resolution and FPS are auto-detected from the stream on connect.

    Args:
        host:       IP address of the ZedStreamer host (default localhost).
        port:       Streaming port matching the sender (default 30000).
        color_mode: Output color channel order (default RGB).
        warmup_s:   Seconds to read frames during connect() before returning.
    """

    host: str = "127.0.0.1"
    port: int = 30000
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
