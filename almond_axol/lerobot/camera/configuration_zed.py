from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for a ZED camera stream receiver.

    Connects to a single ZED stream produced by ZedStreamer on the sender machine.
    One instance per camera (overhead, left_arm, right_arm each get their own config).

    Args:
        host:       IP address of the ZedStreamer host (default localhost).
        port:       Streaming port matching the sender (default 30000).
        fps:        Expected stream FPS; must match the sender (default 60).
        width:      Expected frame width in pixels; must match the sender (default 1920 for HD1080).
        height:     Expected frame height in pixels; must match the sender (default 1080 for HD1080).
        color_mode: Output color channel order (default RGB).
        warmup_s:   Seconds to read frames during connect() before returning.
    """

    host: str = "127.0.0.1"
    port: int = 30000
    fps: int | None = 60
    width: int | None = 1920
    height: int | None = 1080
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
