"""LeRobot ZED camera adapter: the local ZED cameras and their config."""

from .configuration_zed import ZedCameraConfig

__all__ = ["ZedCameraConfig"]

try:
    # pyzed ships with the ZED SDK (`axol zed.install`), not from PyPI, so it is
    # only present on provisioned robot machines. Keep the config importable
    # everywhere — it registers the "zed" camera type for CLI parsing (including
    # via the lerobot_robot_axol plugin) — and surface the missing SDK when a
    # camera is actually opened instead of at import time.
    from .camera_zed import ZedCamera, ZedStereoCamera

    __all__ += ["ZedCamera", "ZedStereoCamera"]
except ImportError:  # pragma: no cover - depends on the ZED SDK being provisioned
    pass
