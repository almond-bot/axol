"""``almond_axol.video.zed_sdk``: the lerobot-free ZED SDK cameras.

Runs without the ZED SDK — ``pyzed.sl`` is replaced by a small in-memory fake
that produces frames from a background ``grab()`` — and checks:

* the module (and the teleop callers that use it) import without ``lerobot``;
* the SDK cameras open, time-stamp, and read frames as before;
* the LeRobot adapters in ``almond_axol.lerobot.camera`` remain LeRobot
  ``Camera`` instances raising LeRobot's device errors.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import unittest
from dataclasses import fields
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from almond_axol.video import zed_sdk

SUCCESS = "SUCCESS"


class _FakeMat:
    def __init__(self) -> None:
        self.data: np.ndarray | None = None

    def get_data(self) -> np.ndarray:
        assert self.data is not None
        return self.data


class _FakeZed:
    """One ``sl.CameraOne`` / ``sl.Camera``: opens, grabs, closes."""

    def __init__(self, sl: Any) -> None:
        self._sl = sl
        self.opened_with: Any = None
        self.closed = False
        self.grabs = 0

    def open(self, params: Any) -> str:
        self.opened_with = params
        return self._sl.open_result

    def get_camera_information(self) -> Any:
        return SimpleNamespace(
            serial_number=self._sl.live_serial,
            camera_configuration=SimpleNamespace(
                fps=self._sl.live_fps,
                resolution=SimpleNamespace(
                    width=self._sl.live_width, height=self._sl.live_height
                ),
            ),
        )

    def grab(self) -> str:
        time.sleep(0.002)
        self.grabs += 1
        return SUCCESS

    def retrieve_image(self, mat: _FakeMat, view: str = "MONO") -> None:
        shade = {"MONO": 10, "LEFT": 20, "RIGHT": 30}[view]
        mat.data = np.full(
            (self._sl.live_height, self._sl.live_width, 4), shade, dtype=np.uint8
        )
        # Alpha channel is dropped by the RGB/BGR convert; keep it distinct so
        # the tests can tell native BGRA from the converted frame.
        mat.data[..., 3] = 255

    def get_timestamp(self, _ref: str) -> Any:
        return SimpleNamespace(get_nanoseconds=lambda: time.time_ns() - 3_000_000)

    def close(self) -> None:
        self.closed = True


def _fake_sl(
    *, serial: int = 41, fps: int = 60, width: int = 960, height: int = 600
) -> Any:
    """A stand-in ``pyzed.sl`` reporting one camera with the given live params."""
    sl = SimpleNamespace(
        live_serial=serial,
        live_fps=fps,
        live_width=width,
        live_height=height,
        open_result=SUCCESS,
        created=[],
    )
    sl.ERROR_CODE = SimpleNamespace(SUCCESS=SUCCESS, CAMERA_NOT_DETECTED="NOT_DETECTED")
    sl.RESOLUTION = SimpleNamespace(SVGA="SVGA", HD1080="HD1080", HD1200="HD1200")
    sl.TIME_REFERENCE = SimpleNamespace(IMAGE="IMAGE")
    sl.VIEW = SimpleNamespace(LEFT="LEFT", RIGHT="RIGHT")
    sl.DEPTH_MODE = SimpleNamespace(NONE="NONE")
    sl.Mat = _FakeMat

    def _params() -> Any:
        return SimpleNamespace(set_from_serial_number=lambda s: None)

    sl.InitParametersOne = _params
    sl.InitParameters = _params

    def _camera() -> _FakeZed:
        cam = _FakeZed(sl)
        sl.created.append(cam)
        return cam

    sl.CameraOne = _camera
    sl.Camera = _camera
    return sl


def _quiet_latency_probe() -> Any:
    """Skip the 30-frame startup latency probe so connect() is fast."""
    return patch.object(
        zed_sdk.ZedSdkCamera, "_log_pipeline_latency", lambda self: None
    )


class ImportIsolationTests(unittest.TestCase):
    def test_sdk_module_imports_without_lerobot(self) -> None:
        code = (
            "import sys\n"
            "import almond_axol.video.zed_sdk\n"
            "import almond_axol.video.video_proc\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'lerobot')\n"
            "assert not bad, bad\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_lerobot_camera_config_shares_sdk_fields(self) -> None:
        from almond_axol.lerobot.camera import ZedCameraConfig

        sdk_fields = {f.name: f.default for f in fields(zed_sdk.ZedSdkCameraConfig)}
        lerobot_fields = {f.name: f.default for f in fields(ZedCameraConfig)}
        for name, default in sdk_fields.items():
            self.assertIn(name, lerobot_fields)
            if name == "color_mode":
                continue  # str vs. lerobot ColorMode enum with the same value
            self.assertEqual(lerobot_fields[name], default, name)


class ConfigTests(unittest.TestCase):
    def test_resolution_name_and_dims_round_trip(self) -> None:
        for name, (w, h) in zed_sdk.ZED_RESOLUTION_DIMS.items():
            cfg = zed_sdk.ZedSdkCameraConfig(serial=1, width=w, height=h)
            self.assertEqual(cfg.resolution_name(), name)
            self.assertEqual(zed_sdk.resolution_for_dims(w, h), name)
        self.assertIsNone(
            zed_sdk.ZedSdkCameraConfig(width=None, height=None).resolution_name()
        )
        with self.assertRaises(ValueError):
            zed_sdk.resolution_for_dims(1, 1)

    def test_color_mode_accepts_strings_and_lerobot_enum(self) -> None:
        from lerobot.cameras.configs import ColorMode

        self.assertEqual(zed_sdk.ZedSdkCameraConfig(color_mode="RGB").color_mode, "rgb")
        self.assertEqual(zed_sdk.ZedSdkCameraConfig(color_mode="bgr").color_mode, "bgr")
        with self.assertRaises(ValueError):
            zed_sdk.ZedSdkCameraConfig(color_mode="gray")
        self.assertTrue(zed_sdk.color_mode_is_rgb(ColorMode.RGB))
        self.assertFalse(zed_sdk.color_mode_is_rgb(ColorMode.BGR))
        self.assertTrue(zed_sdk.color_mode_is_rgb("rgb"))


class SdkUnavailableTests(unittest.TestCase):
    def test_connect_and_find_cameras_raise_clear_error(self) -> None:
        with patch.object(zed_sdk, "sl", None):
            self.assertFalse(zed_sdk.sdk_available())
            self.assertIsNotNone(zed_sdk.sdk_import_error())
            cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=1))
            with self.assertRaisesRegex(zed_sdk.ZedSdkUnavailableError, "zed.install"):
                cam.connect()
            with self.assertRaises(zed_sdk.ZedSdkUnavailableError):
                zed_sdk.ZedSdkCamera.find_cameras()
            stereo = zed_sdk.ZedSdkStereoCamera(
                zed_sdk.ZedSdkCameraConfig(serial=1, stereo=True)
            )
            with self.assertRaises(zed_sdk.ZedSdkUnavailableError):
                stereo.connect()

    def test_reads_before_connect_raise_not_connected(self) -> None:
        cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=1))
        for call in (cam.read, cam.async_read, cam.read_latest_with_ts):
            with self.assertRaises(zed_sdk.ZedNotConnectedError):
                call()
        with self.assertRaises(zed_sdk.ZedNotConnectedError):
            cam.read_at_or_after(0.0)
        with self.assertRaises(zed_sdk.ZedNotConnectedError):
            cam.disconnect()


class MonoCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sl = _fake_sl()
        patcher = patch.object(zed_sdk, "sl", self.sl)
        patcher.start()
        self.addCleanup(patcher.stop)
        probe = _quiet_latency_probe()
        probe.start()
        self.addCleanup(probe.stop)

    def test_connect_reads_timestamped_frames_and_disconnects(self) -> None:
        cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=41))
        cam.connect(warmup=False)
        self.addCleanup(lambda: cam.is_connected and cam.disconnect())
        self.assertTrue(cam.is_connected)
        self.assertEqual((cam.fps, cam.width, cam.height), (60, 960, 600))

        frame = cam.read()
        self.assertEqual(frame.shape, (600, 960, 3))
        self.assertEqual(int(frame[0, 0, 0]), 10)

        before = time.perf_counter()
        frame, cap_ts, recv_ts = cam.read_at_or_after(before, timeout_ms=1000)
        self.assertGreaterEqual(cap_ts, before)
        self.assertLess(cap_ts, recv_ts)  # exposure precedes receive
        self.assertAlmostEqual(recv_ts - cap_ts, 0.003, delta=0.02)

        bgra, _cap, _recv = cam.read_latest_bgra_with_ts()
        self.assertEqual(bgra.shape, (600, 960, 4))
        self.assertEqual(int(bgra[0, 0, 3]), 255)
        self.assertTrue(bgra.flags["C_CONTIGUOUS"])

        with self.assertRaises(zed_sdk.ZedAlreadyConnectedError):
            cam.connect()

        zed = self.sl.created[0]
        cam.disconnect()
        self.assertFalse(cam.is_connected)
        self.assertTrue(zed.closed)
        self.assertIsNone(cam.latest_frame)

    def test_bgr_color_mode_swaps_channels(self) -> None:
        cam = zed_sdk.ZedSdkCamera(
            zed_sdk.ZedSdkCameraConfig(serial=41, color_mode="bgr")
        )
        with cam:
            frame = cam.read()
        self.assertEqual(frame.shape, (600, 960, 3))
        self.assertFalse(cam.is_connected)

    def test_serial_mismatch_closes_and_raises(self) -> None:
        cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=99))
        with self.assertRaisesRegex(ConnectionError, "requested serial 99"):
            cam.connect(warmup=False)
        self.assertFalse(cam.is_connected)
        self.assertTrue(self.sl.created[0].closed)

    def test_live_parameter_mismatch_raises_runtime_error(self) -> None:
        self.sl.live_fps = 30
        cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=41, fps=60))
        with self.assertRaisesRegex(RuntimeError, "fps: expected 60, got 30"):
            cam.connect(warmup=False)
        self.assertFalse(cam.is_connected)
        self.assertTrue(self.sl.created[0].closed)

    def test_none_params_adopt_live_values(self) -> None:
        self.sl.live_fps, self.sl.live_width, self.sl.live_height = 30, 1920, 1200
        cam = zed_sdk.ZedSdkCamera(
            zed_sdk.ZedSdkCameraConfig(serial=41, fps=None, width=None, height=None)
        )
        cam.connect(warmup=False)
        self.addCleanup(cam.disconnect)
        self.assertEqual((cam.fps, cam.width, cam.height), (30, 1920, 1200))

    def test_open_failure_raises_connection_error(self) -> None:
        self.sl.open_result = self.sl.ERROR_CODE.CAMERA_NOT_DETECTED
        cam = zed_sdk.ZedSdkCamera(zed_sdk.ZedSdkCameraConfig(serial=41))
        with self.assertRaisesRegex(ConnectionError, "failed to open"):
            cam.connect(warmup=False)


class StereoCameraTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sl = _fake_sl()
        patcher = patch.object(zed_sdk, "sl", self.sl)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_one_grab_feeds_both_eyes(self) -> None:
        cam = zed_sdk.ZedSdkStereoCamera(
            zed_sdk.ZedSdkCameraConfig(serial=41, stereo=True)
        )
        cam.left_view.connect(warmup=False)
        self.addCleanup(lambda: cam.is_connected and cam.disconnect())
        self.assertTrue(cam.is_connected)
        self.assertEqual(len(self.sl.created), 1)
        self.assertEqual(self.sl.created[0].opened_with.depth_mode, "NONE")

        now = time.perf_counter()
        left, l_cap, _ = cam.left_view.read_at_or_after(now, timeout_ms=1000)
        right, r_cap, _ = cam.right_view.read_at_or_after(now, timeout_ms=1000)
        self.assertEqual(int(left[0, 0, 0]), 20)
        self.assertEqual(int(right[0, 0, 0]), 30)
        self.assertEqual(left.shape, (600, 960, 3))
        # Eyes read after the same target share a grab or a later one, never an
        # earlier one.
        self.assertGreaterEqual(r_cap, l_cap)

        bgra, _cap, _recv = cam.right_view.read_latest_bgra_with_ts()
        self.assertEqual(bgra.shape, (600, 960, 4))
        self.assertEqual((cam.left_view.fps, cam.left_view.width), (60, 960))

        cam.right_view.disconnect()
        self.assertFalse(cam.is_connected)
        self.assertTrue(self.sl.created[0].closed)
        with self.assertRaises(RuntimeError):
            cam.left_view.read_latest_with_ts()

    def test_per_eye_parameter_mismatch(self) -> None:
        self.sl.live_width = 1920
        cam = zed_sdk.ZedSdkStereoCamera(
            zed_sdk.ZedSdkCameraConfig(serial=41, stereo=True)
        )
        with self.assertRaisesRegex(RuntimeError, r"\(per eye\)"):
            cam.connect(warmup=False)
        self.assertTrue(self.sl.created[0].closed)


class LeRobotAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sl = _fake_sl()
        patcher = patch.object(zed_sdk, "sl", self.sl)
        patcher.start()
        self.addCleanup(patcher.stop)
        probe = _quiet_latency_probe()
        probe.start()
        self.addCleanup(probe.stop)

    def test_zed_camera_is_a_lerobot_camera_with_lerobot_errors(self) -> None:
        from lerobot.cameras.camera import Camera
        from lerobot.utils.errors import (
            DeviceAlreadyConnectedError,
            DeviceNotConnectedError,
        )

        from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig

        cam = ZedCamera(ZedCameraConfig(serial=41))
        self.assertIsInstance(cam, Camera)
        self.assertIsInstance(cam, zed_sdk.ZedSdkCamera)
        self.assertEqual(str(cam), "ZedCamera(serial=41)")
        with self.assertRaises(DeviceNotConnectedError):
            cam.read()
        with self.assertRaises(DeviceNotConnectedError):
            cam.disconnect()

        cam.connect(warmup=False)
        self.addCleanup(lambda: cam.is_connected and cam.disconnect())
        with self.assertRaises(DeviceAlreadyConnectedError):
            cam.connect()
        self.assertEqual(cam.read().shape, (600, 960, 3))
        self.assertEqual(cam.config.type, "zed")

    def test_make_cameras_from_configs_builds_zed_camera(self) -> None:
        from lerobot.cameras.utils import make_cameras_from_configs

        from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig

        cams = make_cameras_from_configs({"overhead": ZedCameraConfig(serial=41)})
        self.assertIsInstance(cams["overhead"], ZedCamera)

    def test_stereo_adapter_passes_lerobot_config_through(self) -> None:
        from lerobot.cameras.configs import ColorMode

        from almond_axol.lerobot.camera import ZedCameraConfig, ZedStereoCamera

        cam = ZedStereoCamera(
            ZedCameraConfig(serial=41, stereo=True, color_mode=ColorMode.BGR)
        )
        self.assertIsInstance(cam, zed_sdk.ZedSdkStereoCamera)
        self.assertEqual(str(cam), "ZedStereoCamera(serial=41)")
        self.assertEqual(cam.config.eyes, "both")
        cam.connect(warmup=False)
        self.addCleanup(cam.disconnect)
        frame, _cap, _recv = cam.left_view.read_at_or_after(0.0, timeout_ms=1000)
        self.assertEqual(frame.shape, (600, 960, 3))


class TeleopCallerTests(unittest.TestCase):
    """The teleop SDK fallbacks use the core directly and fail loudly without it."""

    def test_video_relay_sdk_fallback_logs_install_hint_without_sdk(self) -> None:
        from almond_axol.video import video_proc

        with (
            patch.object(zed_sdk, "sl", None),
            self.assertLogs(
                "almond_axol.video.video_proc", level=logging.ERROR
            ) as logs,
        ):
            cam = video_proc._open_sdk_camera("overhead", {"serial": 41})  # noqa: SLF001
        self.assertIsNone(cam)
        self.assertIn("zed.install", "\n".join(logs.output))

    def test_video_relay_sdk_fallback_opens_core_camera(self) -> None:
        from almond_axol.video import video_proc

        with patch.object(zed_sdk, "sl", _fake_sl()), _quiet_latency_probe():
            cam = video_proc._open_sdk_camera(  # noqa: SLF001
                "overhead", {"serial": 41, "resolution": "SVGA", "fps": 60}
            )
            self.assertIsInstance(cam, zed_sdk.ZedSdkCamera)
            cam.disconnect()

    def test_teleop_in_process_fallback_without_sdk(self) -> None:
        from almond_axol.cli import teleop

        cfg = SimpleNamespace(
            cameras={"overhead": 41}, resolution=None, camera_eyes=None
        )
        with (
            patch.object(zed_sdk, "sl", None),
            self.assertLogs("almond_axol.cli.teleop", level=logging.ERROR) as logs,
        ):
            cams = teleop._connect_zed_cameras(cfg, set())  # noqa: SLF001
        self.assertEqual(cams, [])
        self.assertIn("zed.install", "\n".join(logs.output))

    def test_teleop_in_process_fallback_opens_stereo_eyes(self) -> None:
        from almond_axol.cli import teleop

        cfg = SimpleNamespace(
            cameras={"overhead": 41}, resolution="SVGA", camera_eyes=None
        )
        with patch.object(zed_sdk, "sl", _fake_sl()):
            cams = teleop._connect_zed_cameras(cfg, {41})  # noqa: SLF001
            try:
                names = [name for name, _ in cams]
                self.assertEqual(names, ["overhead_left", "overhead_right"])
                self.assertIsInstance(cams[0][1], zed_sdk.StereoEyeView)
            finally:
                for _name, eye in cams:
                    eye.disconnect()


if __name__ == "__main__":
    unittest.main()
