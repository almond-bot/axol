"""Configuration dataclass for the local ZED camera."""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, ColorMode

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
        serial:     Serial number of the camera to open. Required — there is
                    no default, so it must be supplied (on the CLI, inside the
                    inline ``--robot_config.cameras`` dict value).
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
                    stereo camera ``X`` into two observation keys ``X_left``
                    / ``X_right`` backed by a single grab. Default False
                    (mono ZED-X One).
    """

    # Required: no default. The CLI/serve config overlay (see
    # almond_axol.cli.config) strips ``required_input`` fields so draccus
    # forces the operator to supply a value instead of falling back to one.
    serial: int = field(kw_only=True, metadata={"required_input": True})
    fps: int | None = 60
    width: int | None = 960
    height: int | None = 600
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1
    stereo: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)

    def resolution_name(self) -> str | None:
        """Resolution name for the configured dims (``None`` = auto-detect)."""
        if self.width is None or self.height is None:
            return None
        return resolution_for_dims(self.width, self.height)
