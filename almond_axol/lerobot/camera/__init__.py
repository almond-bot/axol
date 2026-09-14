"""LeRobot ZED camera adapter: the local ZED cameras and their config.

The capture implementation is :mod:`almond_axol.video.zed_sdk` (no ``lerobot``
dependency); this package wraps it in LeRobot's ``Camera`` / ``CameraConfig``
contracts. ``pyzed`` ships with the ZED SDK (``axol zed.install``), not from
PyPI, so everything here imports without it — including on a development
machine — and the missing SDK surfaces when a camera is actually opened
(:func:`almond_axol.video.zed_sdk.require_sdk`).
"""

from .camera_zed import ZedCamera, ZedStereoCamera
from .configuration_zed import ZedCameraConfig

__all__ = ["ZedCamera", "ZedCameraConfig", "ZedStereoCamera"]
