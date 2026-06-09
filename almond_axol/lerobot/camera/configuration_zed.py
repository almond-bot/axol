"""Configuration dataclass for the ZED stream-receiver camera."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode

# Frame dimensions (width, height) for each sender resolution name
# (``axol zed.stream --resolution``). For a stereo ZED X these are per eye.
ZED_RESOLUTION_DIMS: dict[str, tuple[int, int]] = {
    "SVGA": (960, 600),
    "HD1080": (1920, 1080),
    "HD1200": (1920, 1200),
}


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
        host:       IP address of the ZedStreamer host. ``None`` (the default)
                    inherits the shared ``AxolRobotConfig.zed_host`` when the
                    camera is built as part of an Axol robot; set it explicitly
                    to point a single camera at a different sender.
        port:       Streaming port matching the sender (default 30000).
        fps:        Expected stream FPS (default 60, matches zed.stream default).
        width:      Expected frame width in pixels (default 960, SVGA).
        height:     Expected frame height in pixels (default 600, SVGA).
        color_mode: Output color channel order (default RGB).
        warmup_s:   Seconds to read frames during connect() before returning.
        stereo:     Treat the stream as a stereo ZED X carrying both eyes. The
                    robot expands a stereo camera ``X`` into two observation
                    keys ``X_left`` / ``X_right`` backed by a single decode.
                    Default False (mono ZED-X One).
    """

    host: str | None = None
    port: int = 30000
    fps: int | None = 60
    width: int | None = 960
    height: int | None = 600
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1
    stereo: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
