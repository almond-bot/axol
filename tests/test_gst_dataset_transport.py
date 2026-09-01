from __future__ import annotations

import threading
import time
import types
import unittest
from unittest.mock import patch

from almond_axol.video.gst_zed import (
    _GstPipelineBase,
    ZedGstCamera,
    ZedGstStereoCamera,
)
from almond_axol.video.video_proc import _set_dataset_branches_enabled


_H264_OUTPUT = (
    "video/x-h264,stream-format=byte-stream,alignment=au ! "
    "queue leaky=downstream max-size-buffers=2 ! gdppay ! shmsink"
)


class _FakeGst:
    CLOCK_TIME_NONE = -1
    PadProbeType = types.SimpleNamespace(BUFFER=1)
    PadProbeReturn = types.SimpleNamespace(OK="ok", REMOVE="remove")


class _FakePad:
    def __init__(self, gst: _FakeGst) -> None:
        self._gst = gst
        self._next_probe = 1
        self._probes: dict[int, object] = {}

    def add_probe(self, _probe_type: int, callback: object) -> int:
        probe_id = self._next_probe
        self._next_probe += 1
        self._probes[probe_id] = callback
        return probe_id

    def remove_probe(self, probe_id: int) -> None:
        self._probes.pop(probe_id, None)

    def push(self, pts: int) -> None:
        buffer = types.SimpleNamespace(pts=pts)
        info = types.SimpleNamespace(get_buffer=lambda: buffer)
        for probe_id, callback in list(self._probes.items()):
            result = callback(self, info)
            if result == self._gst.PadProbeReturn.REMOVE:
                self._probes.pop(probe_id, None)


class _FakeValve:
    def __init__(self, gst: _FakeGst) -> None:
        self.drop = True
        self.sink = _FakePad(gst)

    def get_property(self, name: str) -> bool:
        assert name == "drop"
        return self.drop

    def set_property(self, name: str, value: bool) -> None:
        assert name == "drop"
        self.drop = bool(value)

    def get_static_pad(self, name: str) -> _FakePad | None:
        return self.sink if name == "sink" else None


class _FakePipeline:
    def __init__(self, elements: dict[str, object]) -> None:
        self._elements = elements

    def get_by_name(self, name: str) -> object | None:
        return self._elements.get(name)


def _fake_gate_base(
    *, offset_s: float = 90.0, valve_names: tuple[str, ...] = ("rawvalve",)
) -> tuple[_GstPipelineBase, dict[str, _FakeValve]]:
    gst = _FakeGst()
    valves = {name: _FakeValve(gst) for name in valve_names}
    base = _GstPipelineBase()
    base._gst = gst
    base._pipeline = _FakePipeline(valves)
    base._pts_perf_offset_s = offset_s
    return base, valves


class _FakeCoordinatedCamera:
    def __init__(
        self, name: str, calls: list[tuple], *, finish_error: str | None = None
    ) -> None:
        self.name = name
        self.calls = calls
        self.finish_error = finish_error

    def __repr__(self) -> str:
        return self.name

    def begin_raw_enable(self, target: float) -> None:
        self.calls.append(("begin", self.name, target))

    def finish_raw_enable(self, deadline: float) -> None:
        self.calls.append(("finish", self.name, deadline))
        if self.finish_error is not None:
            raise RuntimeError(self.finish_error)

    def abort_raw_enable(self) -> None:
        self.calls.append(("abort", self.name))


class GstDatasetTransportTest(unittest.TestCase):
    def assert_dataset_branch_is_backpressure_safe(
        self, pipeline: str, valve_name: str, encoder_name: str, socket_path: str
    ) -> None:
        start = pipeline.index(f"valve name={valve_name} drop=false")
        sink = f"shmsink socket-path={socket_path} wait-for-connection=true"
        end = pipeline.index(sink, start) + len(sink)
        branch = pipeline[start:end]

        self.assertIn(f"name={encoder_name}", branch)
        self.assertIn(_H264_OUTPUT, branch)
        self.assertIn(sink, branch)

    def test_mono_dataset_encoder_drains_before_recorder_connects(self) -> None:
        camera = ZedGstCamera(
            serial=1,
            resolution="SVGA",
            raw_socket_path="/tmp/mono-dataset.sock",
        )

        pipeline = camera._pipeline_str()

        self.assertIn("valve name=rawvalve drop=false", pipeline)
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve", "dsenc", "/tmp/mono-dataset.sock"
        )
        self.assertEqual(camera._raw_gates(), (("rawvalve", "dsenc"),))

    def test_both_stereo_dataset_encoders_drain_before_recorder_connects(
        self,
    ) -> None:
        camera = ZedGstStereoCamera(
            serial=2,
            resolution="SVGA",
            left_raw_socket_path="/tmp/left-dataset.sock",
            right_raw_socket_path="/tmp/right-dataset.sock",
        )

        pipeline = camera._pipeline_str()

        self.assertIn("valve name=rawvalve_l drop=false", pipeline)
        self.assertIn("valve name=rawvalve_r drop=false", pipeline)
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve_l", "dsenc_l", "/tmp/left-dataset.sock"
        )
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve_r", "dsenc_r", "/tmp/right-dataset.sock"
        )
        self.assertEqual(
            camera._raw_gates(),
            (("rawvalve_l", "dsenc_l"), ("rawvalve_r", "dsenc_r")),
        )


class GstDatasetEnableBarrierTest(unittest.TestCase):
    def test_valve_opens_on_first_exposure_at_or_after_shared_target(self) -> None:
        base, valves = _fake_gate_base(offset_s=90.0)
        valve = valves["rawvalve"]

        base._begin_dataset_enable((("rawvalve", "dsenc"),), 100.0)
        valve.sink.push(9_999_999_999)
        self.assertTrue(valve.drop)

        valve.sink.push(10_000_000_000)
        self.assertFalse(valve.drop)
        base._finish_dataset_enable(0.0)

    def test_failed_eye_recloses_sibling_which_already_opened(self) -> None:
        base, valves = _fake_gate_base(
            offset_s=90.0, valve_names=("rawvalve_l", "rawvalve_r")
        )
        gates = (("rawvalve_l", "dsenc_l"), ("rawvalve_r", "dsenc_r"))

        base._begin_dataset_enable(gates, 100.0)
        valves["rawvalve_l"].sink.push(10_000_000_000)
        self.assertFalse(valves["rawvalve_l"].drop)

        with self.assertRaisesRegex(RuntimeError, "dsenc_r.*timed out"):
            base._finish_dataset_enable(0.0)
        self.assertTrue(valves["rawvalve_l"].drop)
        self.assertTrue(valves["rawvalve_r"].drop)

    def test_missing_exposure_pts_fails_closed(self) -> None:
        base, valves = _fake_gate_base()
        valve = valves["rawvalve"]

        base._begin_dataset_enable((("rawvalve", "dsenc"),), 100.0)
        valve.sink.push(_FakeGst.CLOCK_TIME_NONE)

        with self.assertRaisesRegex(RuntimeError, "dsenc.*has no exposure PTS"):
            base._finish_dataset_enable(0.0)
        self.assertTrue(valve.drop)

    def test_abort_wakes_pending_finish_and_cannot_reopen(self) -> None:
        base, valves = _fake_gate_base()
        valve = valves["rawvalve"]
        gates = (("rawvalve", "dsenc"),)

        base._begin_dataset_enable(gates, 100.0)
        opened = base._dataset_enable[0][2]
        wait_entered = threading.Event()
        original_wait = opened.wait

        def tracked_wait(timeout: float | None = None) -> bool:
            wait_entered.set()
            return original_wait(timeout)

        opened.wait = tracked_wait  # type: ignore[method-assign]
        failures: list[BaseException] = []

        def finish() -> None:
            try:
                base._finish_dataset_enable(time.perf_counter() + 1.0)
            except BaseException as exc:
                failures.append(exc)

        waiter = threading.Thread(target=finish)
        waiter.start()
        self.assertTrue(wait_entered.wait(1.0))
        base._abort_dataset_enable(gates)
        waiter.join(1.0)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertRegex(str(failures[0]), "opening was cancelled")
        valve.sink.push(10_000_000_000)
        self.assertTrue(valve.drop)

    def test_relay_arms_every_camera_with_same_target_before_waiting(self) -> None:
        calls: list[tuple] = []
        cameras = [
            _FakeCoordinatedCamera("overhead", calls),
            _FakeCoordinatedCamera("left_arm", calls),
            _FakeCoordinatedCamera("right_arm", calls),
        ]

        with patch("almond_axol.video.video_proc.time.perf_counter", return_value=50.0):
            _set_dataset_branches_enabled(cameras, True)

        self.assertEqual([call[0] for call in calls], ["begin"] * 3 + ["finish"] * 3)
        targets = [call[2] for call in calls if call[0] == "begin"]
        self.assertEqual(targets, [50.1, 50.1, 50.1])

    def test_relay_failure_aborts_completed_peer_cameras(self) -> None:
        calls: list[tuple] = []
        cameras = [
            _FakeCoordinatedCamera("overhead", calls),
            _FakeCoordinatedCamera("left_arm", calls, finish_error="no exposure"),
        ]

        with (
            patch("almond_axol.video.video_proc.time.perf_counter", return_value=50.0),
            self.assertRaisesRegex(RuntimeError, "left_arm: no exposure"),
        ):
            _set_dataset_branches_enabled(cameras, True)

        self.assertIn(("abort", "overhead"), calls)
        self.assertIn(("abort", "left_arm"), calls)

    def test_relay_fails_closed_if_arming_misses_future_boundary(self) -> None:
        calls: list[tuple] = []
        cameras = [
            _FakeCoordinatedCamera("overhead", calls),
            _FakeCoordinatedCamera("left_arm", calls),
        ]

        with (
            patch(
                "almond_axol.video.video_proc.time.perf_counter",
                side_effect=(50.0, 50.11),
            ),
            self.assertRaisesRegex(RuntimeError, "missed the shared.*boundary"),
        ):
            _set_dataset_branches_enabled(cameras, True)

        self.assertNotIn("finish", [call[0] for call in calls])
        self.assertIn(("abort", "overhead"), calls)
        self.assertIn(("abort", "left_arm"), calls)


if __name__ == "__main__":
    unittest.main()
