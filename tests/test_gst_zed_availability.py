from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from almond_axol.video import gst_zed


def _fake_gst(factory: object) -> Mock:
    gst = Mock()
    gst.ElementFactory.find.return_value = factory
    return gst


class ZedGstElementAvailabilityTest(unittest.TestCase):
    """``zed_gst_available`` must answer the way ``parse_launch`` will behave."""

    def setUp(self) -> None:
        gst_zed._stale_plugin_warned.clear()  # noqa: SLF001

    def test_unregistered_element_is_unavailable(self) -> None:
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(None), None)
        ):
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001

    def test_registered_and_loadable_element_is_available(self) -> None:
        factory = Mock()
        factory.load.return_value = factory
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
        ):
            self.assertTrue(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001
        factory.load.assert_called_once_with()

    def test_registered_but_unloadable_plugin_is_unavailable_with_fix(self) -> None:
        # The registry cache still lists zedxonesrc after an in-place ZED SDK
        # upgrade broke the plugin's symbols; loading the feature fails just as
        # parse_launch would ("no element zedxonesrc"), so we must fall back.
        factory = Mock()
        factory.load.return_value = None
        with (
            patch.object(
                gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
            ),
            self.assertLogs(gst_zed._logger, level="WARNING") as logs,  # noqa: SLF001
        ):
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001
            # Probed once per camera and fps attempt; the guidance is said once.
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001

        self.assertEqual(len(logs.output), 1)
        self.assertIn("axol gst.build-zed", logs.output[0])
        self.assertIn("ZED SDK upgrade", logs.output[0])

    def test_loader_exception_is_treated_as_unavailable(self) -> None:
        factory = Mock()
        factory.load.side_effect = RuntimeError("dlopen failed")
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
        ):
            self.assertFalse(gst_zed._element_available("zedsrc"))  # noqa: SLF001

    def test_available_gate_requires_loadable_plugin(self) -> None:
        factory = Mock()
        factory.load.return_value = None
        with (
            patch.object(gst_zed, "_gi_available", return_value=True),
            patch.object(gst_zed, "hw_h264_available", return_value=True),
            patch.object(
                gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
            ),
        ):
            self.assertFalse(gst_zed.zed_gst_available())
            self.assertFalse(gst_zed.zed_stereo_gst_available())


class ZedGstDatasetValveTest(unittest.TestCase):
    """The blocking shmsink dataset branch must start gated shut.

    ``shmsink wait-for-connection=true`` blocks until the recorder attaches; an
    open valve in front of it pins the camera's NVMM capture buffers, starves
    Argus within ~15 s and the ZED SDK restarts nvargus-daemon under the live
    relay. The recorder opens the valve per episode via ``set_raw_enabled``.
    """

    def test_mono_shmsink_branch_starts_closed(self) -> None:
        cam = gst_zed.ZedGstCamera(
            1, "SVGA", 60, raw_socket_path="/tmp/axol-raw/left_arm.sock"
        )
        pipeline = cam._pipeline_str()  # noqa: SLF001
        self.assertIn("valve name=rawvalve drop=true !", pipeline)
        self.assertIn("gdppay ! shmsink", pipeline)
        self.assertIn("wait-for-connection=true", pipeline)

    def test_mono_in_process_raw_branch_stays_open(self) -> None:
        # inference / run-policy read RGBA off an appsink and never toggle the
        # valve, so the appsink path keeps its open default.
        cam = gst_zed.ZedGstCamera(1, "SVGA", 60, want_raw=True)
        pipeline = cam._pipeline_str()  # noqa: SLF001
        self.assertIn("valve name=rawvalve drop=false", pipeline)
        self.assertNotIn("shmsink", pipeline)

    def test_stereo_shmsink_branches_start_closed(self) -> None:
        cam = gst_zed.ZedGstStereoCamera(
            2,
            "SVGA",
            60,
            left_raw_socket_path="/tmp/axol-raw/l.sock",
            right_raw_socket_path="/tmp/axol-raw/r.sock",
        )
        pipeline = cam._pipeline_str()  # noqa: SLF001
        self.assertIn("valve name=rawvalve_l drop=true !", pipeline)
        self.assertIn("valve name=rawvalve_r drop=true !", pipeline)
        self.assertEqual(pipeline.count("wait-for-connection=true"), 2)

    def test_stereo_in_process_raw_branches_stay_open(self) -> None:
        cam = gst_zed.ZedGstStereoCamera(2, "SVGA", 60, want_raw=True)
        pipeline = cam._pipeline_str()  # noqa: SLF001
        self.assertIn("valve name=rawvalve_l drop=false", pipeline)
        self.assertIn("valve name=rawvalve_r drop=false", pipeline)

    def test_set_raw_enabled_opens_and_closes_the_gate(self) -> None:
        cam = gst_zed.ZedGstCamera(
            1, "SVGA", 60, raw_socket_path="/tmp/axol-raw/left_arm.sock"
        )
        cam._pipeline_str()  # noqa: SLF001 - sets the encoder bitrate
        valve = Mock()
        pipeline = Mock()
        pipeline.get_by_name.side_effect = lambda name: (
            valve if name == "rawvalve" else Mock()
        )
        cam._pipeline = pipeline  # noqa: SLF001
        cam.set_raw_enabled(True)
        cam.set_raw_enabled(False)
        self.assertEqual(
            [c.args for c in valve.set_property.call_args_list],
            [("drop", False), ("drop", True)],
        )

    def test_dataset_shmsink_is_named_for_the_guard(self) -> None:
        cam = gst_zed.ZedGstCamera(
            1, "SVGA", 60, raw_socket_path="/tmp/axol-raw/left_arm.sock"
        )
        self.assertIn("shmsink name=dssink ", cam._pipeline_str())  # noqa: SLF001
        stereo = gst_zed.ZedGstStereoCamera(
            2,
            "SVGA",
            60,
            left_raw_socket_path="/tmp/axol-raw/l.sock",
            right_raw_socket_path="/tmp/axol-raw/r.sock",
        )
        pipeline = stereo._pipeline_str()  # noqa: SLF001
        self.assertIn("shmsink name=dssink_l ", pipeline)
        self.assertIn("shmsink name=dssink_r ", pipeline)


class _FakeSink:
    """Just enough of shmsink to capture the guard's signal handlers."""

    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def connect(self, signal: str, handler: object) -> None:
        self.handlers[signal] = handler

    def attach(self, fd: int = 7) -> None:
        self.handlers["client-connected"](self, fd)

    def detach(self, fd: int = 7) -> None:
        self.handlers["client-disconnected"](self, fd)


def _guarded_pipeline() -> tuple[Mock, _FakeSink, Mock]:
    valve = Mock()
    sink = _FakeSink()
    pipeline = Mock()
    pipeline.get_by_name.side_effect = lambda name: {
        "rawvalve": valve,
        "dssink": sink,
    }.get(name, Mock())
    return valve, sink, pipeline


def _drop_history(valve: Mock) -> list[bool]:
    return [c.args[1] for c in valve.set_property.call_args_list if c.args[0] == "drop"]


class DatasetSinkGuardTest(unittest.TestCase):
    """The blocking dataset shmsink must never be fed while it has no reader.

    ``wait-for-connection=true`` blocks whenever the client count is zero, not
    only before the first attach -- so a recorder that dies (or detaches during
    teardown) while the valve is open would pin the camera exactly like the
    startup case. The guard makes the valve state a function of "wanted" AND
    "reader attached".
    """

    def test_open_is_deferred_until_a_reader_attaches(self) -> None:
        valve, sink, pipeline = _guarded_pipeline()
        guard = gst_zed._DatasetSinkGuard(pipeline, "rawvalve", "dssink", "cam")  # noqa: SLF001
        with self.assertLogs(gst_zed._logger, level="WARNING"):  # noqa: SLF001
            guard.set_wanted(True)
        self.assertEqual(_drop_history(valve), [True])  # still closed: no reader
        sink.attach()
        self.assertEqual(_drop_history(valve)[-1], False)

    def test_reader_loss_closes_an_open_branch(self) -> None:
        valve, sink, pipeline = _guarded_pipeline()
        guard = gst_zed._DatasetSinkGuard(pipeline, "rawvalve", "dssink", "cam")  # noqa: SLF001
        sink.attach()
        guard.set_wanted(True)
        self.assertEqual(_drop_history(valve)[-1], False)
        with self.assertLogs(gst_zed._logger, level="WARNING") as logs:  # noqa: SLF001
            sink.detach()
        self.assertEqual(_drop_history(valve)[-1], True)
        self.assertIn("recorder detached", logs.output[0])
        # A reattach (recorder restart) reopens without a new open request.
        sink.attach()
        self.assertEqual(_drop_history(valve)[-1], False)

    def test_reader_present_but_not_wanted_stays_closed(self) -> None:
        valve, sink, pipeline = _guarded_pipeline()
        guard = gst_zed._DatasetSinkGuard(pipeline, "rawvalve", "dssink", "cam")  # noqa: SLF001
        sink.attach()
        self.assertEqual(_drop_history(valve), [True])
        guard.set_wanted(True)
        guard.set_wanted(False)
        self.assertEqual(_drop_history(valve)[-2:], [False, True])
        self.assertEqual(guard.clients, 1)

    def test_camera_routes_set_raw_enabled_through_the_guard(self) -> None:
        cam = gst_zed.ZedGstCamera(
            1, "SVGA", 60, raw_socket_path="/tmp/axol-raw/left_arm.sock"
        )
        cam._pipeline_str()  # noqa: SLF001 - sets the encoder bitrate
        valve, sink, pipeline = _guarded_pipeline()
        cam._pipeline = pipeline  # noqa: SLF001
        cam._guard_dataset_sink("rawvalve", "dssink", "cam")  # noqa: SLF001
        with self.assertLogs(gst_zed._logger, level="WARNING"):  # noqa: SLF001
            cam.set_raw_enabled(True)
        self.assertEqual(_drop_history(valve), [True])
        sink.attach()
        self.assertEqual(_drop_history(valve)[-1], False)
        sink.detach()
        self.assertEqual(_drop_history(valve)[-1], True)
        cam.set_raw_enabled(False)
        sink.attach()
        # Not wanted any more, so the reattach does not reopen the branch.
        self.assertEqual(_drop_history(valve)[-1], True)


if __name__ == "__main__":
    unittest.main()
