"""
Local ZED camera for LeRobot.

Thin LeRobot adapters over the SDK cameras in :mod:`almond_axol.video.zed_sdk`
(which is where the grab thread, timestamps, and stereo handling live, free of
any ``lerobot`` import so teleop can use them from a base install). This module
adds what LeRobot's ``Camera`` contract needs — the abstract base class, and
LeRobot's ``DeviceNotConnectedError`` / ``DeviceAlreadyConnectedError`` on
misuse — so ``make_cameras_from_configs`` and the ``AxolRobot`` treat a ZED
like any other LeRobot camera.

ZedCamera opens a GMSL-attached ZED camera by serial number via the ZED SDK
and exposes it as a standard LeRobot Camera. One instance per camera —
instantiate three to cover overhead, left_arm, and right_arm.

Each grabbed frame carries two timestamps, both on ``time.perf_counter``:

* ``capture_perf_ts`` — when the sensor exposed the frame, derived from the
  SDK's ``TIME_REFERENCE.IMAGE`` wall-clock timestamp plus a per-frame
  wall→perf offset. Used by ``collect_data`` so dataset rows record the
  moment of capture, not the moment of decode.
* ``receive_perf_ts`` — when this process retrieved the frame.

The ZED X daemon only enumerates GMSL cameras when it starts, so a camera
plugged in after boot is invisible until the daemon restarts — see
``almond_axol.zed.restart_zed_daemon``.

Typical usage::

    from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig

    overhead  = ZedCamera(ZedCameraConfig(serial=41234567))
    left_arm  = ZedCamera(ZedCameraConfig(serial=41234568))
    right_arm = ZedCamera(ZedCameraConfig(serial=41234569))

    with overhead, left_arm, right_arm:
        frame = overhead.read()  # uint8 numpy array (600, 960, 3) RGB
"""

from __future__ import annotations

from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ...video.zed_sdk import StereoEyeView, ZedSdkCamera, ZedSdkStereoCamera
from .configuration_zed import ZedCameraConfig

__all__ = ["StereoEyeView", "ZedCamera", "ZedStereoCamera"]


class ZedCamera(ZedSdkCamera, Camera):
    """LeRobot camera that captures from a locally connected ZED camera.

    Opens the camera by serial number via the ZED SDK. A background thread
    continuously calls grab() and stores the latest frame so read() and
    async_read() never block on the sensor. See
    :class:`~almond_axol.video.zed_sdk.ZedSdkCamera` for the read API.

    Args:
        config: Serial, resolution, fps, color mode, and warmup duration.
    """

    not_connected_error = DeviceNotConnectedError
    already_connected_error = DeviceAlreadyConnectedError

    def __init__(self, config: ZedCameraConfig) -> None:
        # ZedSdkCamera.__init__ sets the fps/width/height attributes LeRobot's
        # Camera.__init__ would; skip the latter so the MRO stays single-rooted.
        ZedSdkCamera.__init__(self, config)
        self.config: ZedCameraConfig = config


class ZedStereoCamera(ZedSdkStereoCamera):
    """Local stereo ZED X camera with a single shared grab.

    See :class:`~almond_axol.video.zed_sdk.ZedSdkStereoCamera`; the eyes are
    exposed as ``left_view`` / ``right_view`` with the :class:`ZedCamera` read
    API, so :class:`~almond_axol.lerobot.robot.AxolRobot` treats each as an
    ordinary camera while the sensor is grabbed once.

    Args:
        config: Serial, resolution, fps, color mode, and warmup duration
            (``stereo`` set).
    """

    def __init__(self, config: ZedCameraConfig) -> None:
        super().__init__(config)
        self.config: ZedCameraConfig = config
