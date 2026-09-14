"""Configuration dataclass for the local ZED camera."""

from dataclasses import dataclass
from typing import Literal

from lerobot.cameras.configs import CameraConfig, ColorMode

from ...cli.config import register_literal

# Which eye(s) of a stereo ZED X to expose as observations. Registered with
# draccus so it decodes/validates on the CLI (see register_literal).
StereoEyes = register_literal(Literal["both", "left", "right"])

# Frame dimensions (width, height) for each ZED capture resolution name.
# For a stereo ZED X these are per eye.
ZED_RESOLUTION_DIMS: dict[str, tuple[int, int]] = {
    "SVGA": (960, 600),
    "HD1080": (1920, 1080),
    "HD1200": (1920, 1200),
}


def resolution_for_dims(width: int, height: int) -> str:
    """Resolution name for ``(width, height)`` frame dimensions.

    Raises:
        ValueError: If the dimensions match no supported ZED resolution.
    """
    for name, dims in ZED_RESOLUTION_DIMS.items():
        if dims == (width, height):
            return name
    raise ValueError(
        f"{width}x{height} matches no supported ZED resolution "
        f"({', '.join(f'{n} {w}x{h}' for n, (w, h) in ZED_RESOLUTION_DIMS.items())})"
    )


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for a locally connected ZED camera.

    Opens the GMSL-attached camera by serial number via the ZED SDK. One
    instance per camera (overhead, left_arm, right_arm each get their own
    config).

    Args:
        serial:     Serial number of the camera to open. Defaults to ``0``, an
                    "unassigned" sentinel — collect-data / run-policy seed all
                    three camera slots and then drop the ones left at ``0`` (see
                    ``AxolRobotConfig.select_assigned_cameras``), so an operator
                    only has to assign the cameras they actually have. A real
                    ZED serial is always positive.
        fps:        Capture frame rate (default 60). ``None`` adopts the
                    camera default on connect.
        width:      Frame width in pixels (default 960, SVGA). Together with
                    ``height`` it must name a supported ZED resolution (see
                    ``ZED_RESOLUTION_DIMS``). ``None`` adopts the camera
                    default on connect.
        height:     Frame height in pixels (default 600, SVGA).
        color_mode: Output color channel order (default RGB).
        warmup_s:   Seconds to read frames during connect() before returning.
        stereo:     Open the camera as a stereo ZED X. The robot expands a
                    stereo camera ``X`` into observation keys backed by a
                    single grab (see ``eyes``). Default False (mono ZED-X One).
        eyes:       Which eye(s) of a stereo camera to expose as **recorded**
                    observations (the dataset / what a policy is trained on).
                    ``"both"`` (default) yields ``X_left`` and ``X_right``;
                    ``"left"`` yields only ``X_left``; ``"right"`` yields only
                    ``X_right``. Ignored when ``stereo`` is False.
        stream_eyes: Which eye(s) of a stereo camera to **stream** to the VR
                    headset, independently of ``eyes``. ``None`` (default) means
                    follow ``eyes`` (the legacy coupled behaviour). Setting it
                    decouples the headset feed from the recording — e.g. stream
                    ``"both"`` for depth perception in teleop while recording
                    only ``"left"`` (or vice versa). Only the collect-data /
                    teleop relay honours this; run-policy streams no video.
                    Ignored when ``stereo`` is False.
        record:     Whether this camera is recorded to the dataset at all.
                    ``False`` drops it from the recorded observations
                    (``observation_cameras``) while still allowing it to stream.
                    Default True.
        stream:     Whether this camera is streamed to the headset at all.
                    ``False`` drops it from the relay's encoded feed while still
                    allowing it to be recorded. Default True.
    """

    # 0 is the "unassigned" sentinel. collect-data / run-policy seed all three
    # camera slots so each stays reachable as a dotted
    # ``--robot_config.cameras.<slot>.serial`` override / control-panel field,
    # then prune the slots left at 0 (see
    # ``AxolRobotConfig.select_assigned_cameras``). A real ZED serial is
    # positive, so 0 unambiguously marks a slot the operator didn't assign.
    serial: int = 0
    fps: int | None = 60
    width: int | None = 960
    height: int | None = 600
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1
    stereo: bool = False
    eyes: StereoEyes = "both"
    stream_eyes: StereoEyes | None = None
    record: bool = True
    stream: bool = True

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if self.eyes not in ("both", "left", "right"):
            raise ValueError(
                f"eyes must be one of 'both', 'left', 'right'; got {self.eyes!r}."
            )
        if self.stream_eyes is not None and self.stream_eyes not in (
            "both",
            "left",
            "right",
        ):
            raise ValueError(
                "stream_eyes must be None or one of 'both', 'left', 'right'; "
                f"got {self.stream_eyes!r}."
            )

    def streaming_eyes(self) -> StereoEyes:
        """Eye selection for the headset stream (falls back to ``eyes``)."""
        return self.stream_eyes or self.eyes

    def resolution_name(self) -> str | None:
        """Resolution name for the configured dims (``None`` = auto-detect)."""
        if self.width is None or self.height is None:
            return None
        return resolution_for_dims(self.width, self.height)
