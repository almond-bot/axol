from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.tracker import create_source
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.config import TrackerConfig
from almond_axol.tracker.survive import SurviveSource, _OutputClockMapper


class Recorder:
    def __init__(self):
        self.armed = False
        self.events = []
        self.faults = []

    def arm(self):
        self.armed = True

    def record(self, kind, **fields):
        self.events.append((kind, fields))

    def trigger(self, reason, context=None):
        if self.armed:
            self.faults.append((reason, context))

    def status(self):
        return {"state": "armed" if self.armed else "waiting"}


def ready_source():
    source = SurviveSource(expected_stations=1)
    source._diagnostics = recorder = Recorder()
    clock = _OutputClockMapper()

    def feed(line, now):
        source._consume_record(line, clock, 100 + now)

    feed('0 T20 CONFIG {"device_serial_number":"LHR-LEFT"}', 0)
    feed('0 T21 CONFIG {"device_serial_number":"LHR-RIGHT"}', 0)
    feed("0 0 LH_POSE 0 0 2 1 0 0 0 100", 0)
    for i in range(71):
        now = i * 0.02
        for alias in ("T20", "T21"):
            for sensor in range(5):
                feed(
                    f"{now} {alias} B 0 {sensor} {int((now + 0.001) * 48000000)} 0 .3",
                    now,
                )
            feed(f"{now} {alias} POSE 0 0 0 1 0 0 0", now)
        with patch(
            "almond_axol.tracker.survive.time.perf_counter", return_value=100 + now
        ):
            report = source.tracking_quality()
    assert report["status"] == "ready"
    assert recorder.armed
    return source, recorder, feed


def test_collection_enables_diagnostics_but_setup_does_not():
    config = TrackerConfig(backend="survive")
    assert create_source(config)._diagnostics_enabled
    assert not create_source(config, setup=True)._diagnostics_enabled
    assert not SurviveSource(
        force_calibrate=True, enable_diagnostics=True
    )._diagnostics_enabled


def test_pose_jump_triggers_before_next_bridge_poll_and_keeps_raw_evidence():
    source, recorder, feed = ready_source()
    feed("1.408 T20 POSE 0 0 0 0.8 0.6 0 0", 1.408)
    assert len(recorder.faults) == 1
    assert "pose discontinuity" in recorder.faults[0][0]
    raw = [data for kind, data in recorder.events if kind == "pose_receipt"][-1]
    assert raw["serial"] == "LHR-LEFT"
    assert raw["device"] == "T20"
    assert raw["quaternion_wxyz"] == [0.8, 0.6, 0, 0]
    assert raw["output_time"] == 1.408
    assert raw["receipt_time"] == 101.408
    assert raw["mapped_time"] == 101.408
    assert recorder.faults[0][1]["identities"]["T20"] == "LHR-LEFT"
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101.408):
        assert source.tracking_quality()["status"] == "error"


def test_no_further_pose_output_still_triggers_stale_capture_on_health_poll():
    source, recorder, _ = ready_source()
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101.7):
        report = source.tracking_quality()
    assert report["status"] == "error"
    assert recorder.faults


def test_pair_skew_capture_keeps_control_disengage_and_fires_once_per_edge():
    source, recorder, _ = ready_source()
    source.poses = Mock(
        return_value={
            "LHR-LEFT": TrackerPose(
                np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 101.325
            ),
            "LHR-RIGHT": TrackerPose(
                np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 101.4
            ),
        }
    )
    bridge = TrackerBridge(
        source,
        left="LHR-LEFT",
        right="LHR-RIGHT",
        controls=StopEventControls(threading.Event()),
    )
    with patch("almond_axol.tracker.bridge.time.perf_counter", return_value=101.4):
        frame = bridge.compose_frame()
        bridge.compose_frame()
    assert not frame["l_tracked"]
    assert len(recorder.faults) == 1
    reason, context = recorder.faults[0]
    assert "75 ms behind" in reason
    assert context["bindings"] == {"left": "LHR-LEFT", "right": "LHR-RIGHT"}
    assert context["sampleTimes"] == [101.325, 101.4]
    assert any(kind == "bridge_frame" for kind, _ in recorder.events)


def test_diagnostic_failure_does_not_interrupt_pose_delivery():
    source, recorder, feed = ready_source()
    recorder.record = Mock(side_effect=OSError("diagnostic queue unavailable"))
    feed("1.42 T20 POSE .001 0 0 1 0 0 0", 1.42)
    with patch("almond_axol.tracker.survive.time.perf_counter", return_value=101.42):
        report = source.tracking_quality()
        poses = source.poses()
    assert report["status"] == "ready"
    assert report["diagnostics"]["state"] == "error"
    assert poses["LHR-LEFT"].t == 101.42
    assert poses["LHR-LEFT"].tracking


def test_diagnostic_fd_is_separate_and_caller_environment_cannot_override_it(tmp_path):
    source = SurviveSource(enable_diagnostics=True)
    source._cli_executable = Path("/attested/survive-cli")
    recorder = Mock()
    recorder.start.return_value = 321
    source._stop.set()
    config = tmp_path / "config.json"
    config.write_text("{}")
    with (
        # Import the installer before mocking the shared subprocess module;
        # its package imports may themselves launch library discovery helpers.
        patch("almond_axol.cli.tracker_install._BUILD_REVISION", "diagnostic-test"),
        patch.object(source, "_prepare_native_config", return_value=config),
        patch(
            "almond_axol.tracker.survive.FaultDiagnosticRecorder", return_value=recorder
        ),
        patch(
            "almond_axol.tracker.survive.subprocess.Popen",
            return_value=SimpleNamespace(stdout=[], pid=12, poll=lambda: 0),
        ) as launch,
        patch.dict("os.environ", {"AXOL_SURVIVE_DIAGNOSTIC_FD": "999"}),
    ):
        source._run_cli_in_directory(tmp_path)
    assert launch.call_args.kwargs["pass_fds"] == (321,)
    assert launch.call_args.kwargs["env"]["AXOL_SURVIVE_DIAGNOSTIC_FD"] == "321"
    args = launch.call_args.args[0]
    assert args[args.index("--record-imu") + 1] == "0"
    recorder.child_started.assert_called_once()


def test_recorder_start_failure_leaves_tracking_available(tmp_path):
    source = SurviveSource(enable_diagnostics=True)
    config = tmp_path / "config.json"
    config.write_text("{}")
    with patch(
        "almond_axol.tracker.survive.FaultDiagnosticRecorder",
        side_effect=OSError("no descriptors"),
    ):
        assert source._start_diagnostics(config) is None
    assert source.tracking_quality()["diagnostics"]["state"] == "error"
