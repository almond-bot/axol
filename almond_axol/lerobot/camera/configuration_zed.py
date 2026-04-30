from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for a ZED camera stream receiver.

    Connects to a single ZED stream produced by ZedStreamer on the sender machine.
    One instance per camera (overhead, left_arm, right_arm each get their own config).

    Resolution and FPS are always overridden from the live stream on connect(),
    so these defaults just need to match what the sender is configured to stream
    (axol zed.stream defaults: SVGA @ 60 fps).

    Args:
        host:       IP address of the ZedStreamer host (default 192.168.10.1).
        port:       Streaming port matching the sender (default 30000).
        fps:        Expected stream FPS (default 60, matches zed.stream default).
        width:      Expected frame width in pixels (default 960, SVGA).
        height:     Expected frame height in pixels (default 600, SVGA).
        color_mode: Output color channel order (default RGB).
        warmup_s:   Seconds to read frames during connect() before returning.
    """

    host: str = "192.168.10.1"
    port: int = 30000
    fps: int | None = 60
    width: int | None = 960
    height: int | None = 600
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
