"""Saved takes end where the trigger x3 gesture *began*, not where it resolved.

The end gesture is three trigger clicks. Each click is a gripper command and
a hand twitch, so unless the take is cut before the first squeeze started,
every saved episode ends with ~1 s of clicking noise. This covers the whole
path: bridge (names the instant) → VRFrame → teleop event → recorder trim.
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np
from lerobot.teleoperators.utils import TeleopEvents

from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop
from almond_axol.recording import record_proc
from almond_axol.recording.record_proc import InProcessRecorder, trim_episode_after
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.vr.models import VREpisodeOutcome, VRFrame, VRState


class _Source:
    def poses(self) -> dict[str, TrackerPose]:
        now = time.perf_counter()
        return {
            side: TrackerPose(
                pos=np.array([x, 1.0, -0.4]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now,
                tracking=True,
            )
            for side, x in (("left", 0.2), ("right", -0.2))
        }


class _Trigger:
    def __init__(self, grip: float = 1.0) -> None:
        self.value = grip
        self.stale = False

    def grip(self) -> float:
        return self.value

    def is_stale(self) -> bool:
        return self.stale


class _Clock:
    """Deterministic perf_counter for the bridge."""

    def __init__(self, start: float) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, dt: float = 0.01) -> float:
        self.now += dt
        return self.now


def _recording_bridge() -> tuple[TrackerBridge, _Trigger, _Trigger, _Clock]:
    left = _Trigger()
    right = _Trigger()
    bridge = TrackerBridge(
        _Source(),
        left="left",
        right="right",
        controls=StopEventControls(threading.Event()),
        left_trigger=left,
        right_trigger=right,
        auto_engage=True,
        confirm_auto_engage=True,
    )
    bridge._state = VRState.DATA_COLLECTION
    clock = _Clock(500.0)
    with mock.patch("almond_axol.tracker.bridge.time.perf_counter", clock):
        left.value = right.value = 0.0
        bridge.compose_frame()
        clock.tick()
        left.value = right.value = 1.0
        bridge.compose_frame()
        bridge._handle_tracking_state(True)
        clock.tick()
        opened = bridge.compose_frame()
        if "lock_release_id" in opened:
            bridge._handle_lock_release(opened["lock_release_id"])
        clock.tick(1.0)  # some task time
        settled = bridge.compose_frame()
    assert settled["state"] == VRState.RECORDING.value
    return bridge, left, right, clock


class BridgeNamesTheCutTest(unittest.TestCase):
    def test_success_frame_carries_when_the_first_squeeze_began(self) -> None:
        bridge, left, right, clock = _recording_bridge()

        def frame() -> dict:
            with mock.patch("almond_axol.tracker.bridge.time.perf_counter", clock):
                return bridge.compose_frame()

        # Ordinary recording frames carry no cut.
        self.assertIsNone(frame()["episode_end_t_host"])

        # The trigger rests at 1.0 for a while; the last at-rest sample is the
        # instant the data should end at.
        clock.tick()
        last_at_rest = clock.now
        frame()
        # First click: a gradual squeeze (the analog trigger travels), then a
        # release. Only the crossing of 0.2 counts as the press, but the cut
        # must land where the travel started.
        for grip in (0.7, 0.4, 0.1):
            clock.tick()
            left.value = grip
            self.assertIsNone(frame()["episode_end_t_host"])
        clock.tick()
        left.value = 1.0
        frame()
        # Clicks two and three, quick and full.
        for _ in range(2):
            clock.tick(0.05)
            left.value = 0.0
            frame()
            clock.tick(0.05)
            left.value = 1.0
            frame()
        # Resolve after the inter-press timeout.
        clock.tick(1.0)
        saved = frame()
        self.assertEqual(saved["state"], VRState.DATA_COLLECTION.value)
        self.assertEqual(saved["episode_outcome"], VREpisodeOutcome.SUCCESS.value)
        self.assertEqual(saved["episode_end_t_host"], last_at_rest)
        self.assertLess(saved["episode_end_t_host"], clock.now - 1.0)

        # It survives the wire format and rides the outcome pulse, then clears.
        parsed = VRFrame.model_validate(saved)
        self.assertEqual(parsed.episode_end_t_host, last_at_rest)
        self.assertEqual(parsed.episode_outcome, VREpisodeOutcome.SUCCESS)
        pulse = frame()
        self.assertEqual(pulse["episode_end_t_host"], last_at_rest)
        for _ in range(12):
            frame()
        self.assertIsNone(frame()["episode_end_t_host"])
        self.assertIsNone(frame()["episode_outcome"])

    def test_two_handed_triple_cuts_at_the_earlier_hand(self) -> None:
        bridge, left, right, clock = _recording_bridge()

        def frame() -> dict:
            with mock.patch("almond_axol.tracker.bridge.time.perf_counter", clock):
                return bridge.compose_frame()

        clock.tick()
        frame()
        rest_before_right = clock.now
        # The right hand starts squeezing one sample before the left.
        clock.tick()
        right.value = 0.5
        frame()
        clock.tick()
        right.value = 0.0
        left.value = 0.0
        frame()
        clock.tick(0.05)
        left.value = right.value = 1.0
        frame()
        for _ in range(2):
            clock.tick(0.05)
            left.value = right.value = 0.0
            frame()
            clock.tick(0.05)
            left.value = right.value = 1.0
            frame()
        clock.tick(1.0)
        saved = frame()
        self.assertEqual(saved["episode_outcome"], VREpisodeOutcome.SUCCESS.value)
        self.assertEqual(saved["episode_end_t_host"], rest_before_right)

    def test_discard_carries_no_cut(self) -> None:
        bridge, left, right, clock = _recording_bridge()

        def frame() -> dict:
            with mock.patch("almond_axol.tracker.bridge.time.perf_counter", clock):
                return bridge.compose_frame()

        for _ in range(4):
            clock.tick(0.05)
            left.value = 0.0
            discarded = frame()
            clock.tick(0.05)
            left.value = 1.0
            frame()
        self.assertEqual(discarded["state"], VRState.DATA_COLLECTION.value)
        self.assertTrue(discarded["reset"])
        self.assertIsNone(discarded["episode_end_t_host"])


class TeleopEventCarriesTheCutTest(unittest.TestCase):
    def _teleop(self) -> AxolVRTeleop:
        teleop = object.__new__(AxolVRTeleop)
        teleop._rate_lock = threading.Lock()
        teleop._vr_frame_times = []
        teleop._core = mock.Mock()
        teleop._core.is_resetting = False
        teleop._cart = None
        teleop._prev_state = VRState.RECORDING
        teleop._event_lock = threading.Lock()
        teleop._rerecord_latch = False
        teleop._terminate_latch = False
        teleop._failure_latch = False
        teleop._episode_end_t_host = None
        teleop._start_recording_latch = False
        return teleop

    @staticmethod
    def _frame(**overrides) -> VRFrame:
        pose = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}
        pose["quaternion"] = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
        base = {
            "l_ee": pose,
            "r_ee": pose,
            "l_elbow": {"x": 0.0, "y": 0.0, "z": 0.0},
            "r_elbow": {"x": 0.0, "y": 0.0, "z": 0.0},
            "state": VRState.DATA_COLLECTION.value,
        }
        base.update(overrides)
        return VRFrame.model_validate(base)

    def test_success_end_exposes_cut_once(self) -> None:
        teleop = self._teleop()
        teleop._on_vr_frame(
            self._frame(
                episode_outcome=VREpisodeOutcome.SUCCESS.value,
                episode_end_t_host=1234.5,
            )
        )
        events = teleop.get_teleop_events()
        self.assertTrue(events[TeleopEvents.TERMINATE_EPISODE])
        self.assertTrue(events[TeleopEvents.SUCCESS])
        self.assertEqual(events["episode_end_t_host"], 1234.5)
        # Read-and-clear, like the other latches.
        self.assertIsNone(teleop.get_teleop_events()["episode_end_t_host"])

    def test_failure_and_discard_expose_no_cut(self) -> None:
        teleop = self._teleop()
        teleop._on_vr_frame(
            self._frame(
                episode_outcome=VREpisodeOutcome.FAILURE.value,
                episode_end_t_host=1234.5,
            )
        )
        events = teleop.get_teleop_events()
        self.assertTrue(events[TeleopEvents.TERMINATE_EPISODE])
        self.assertTrue(events[TeleopEvents.FAILURE])
        self.assertIsNone(events["episode_end_t_host"])

        teleop = self._teleop()
        teleop._on_vr_frame(self._frame(reset=True, episode_end_t_host=1234.5))
        events = teleop.get_teleop_events()
        self.assertTrue(events[TeleopEvents.RERECORD_EPISODE])
        self.assertIsNone(events["episode_end_t_host"])

    def test_quest_style_end_without_cut_keeps_every_row(self) -> None:
        teleop = self._teleop()
        teleop._on_vr_frame(self._frame())
        events = teleop.get_teleop_events()
        self.assertTrue(events[TeleopEvents.TERMINATE_EPISODE])
        self.assertIsNone(events["episode_end_t_host"])


def _fake_dataset(n_rows: int, fps: int = 30) -> mock.Mock:
    dataset = mock.Mock()
    dataset.fps = fps
    dataset.writer.episode_buffer = {
        "size": n_rows,
        "episode_index": 3,
        "task": ["t"] * n_rows,
        "frame_index": list(range(n_rows)),
        "timestamp": [i / fps for i in range(n_rows)],
        "observation.state": [np.array([float(i)]) for i in range(n_rows)],
        "observation.images.left_arm": [None] * n_rows,
    }
    return dataset


class TrimEpisodeAfterTest(unittest.TestCase):
    def test_rows_captured_after_the_cut_are_dropped_consistently(self) -> None:
        dataset = _fake_dataset(10)
        row_times = [100.0 + i * 0.1 for i in range(10)]

        removed = trim_episode_after(dataset, row_times, 100.55)

        self.assertEqual(removed, 4)
        buffer = dataset.writer.episode_buffer
        self.assertEqual(buffer["size"], 6)
        for key in ("task", "frame_index", "timestamp", "observation.state"):
            self.assertEqual(len(buffer[key]), 6, key)
        self.assertEqual(len(buffer["observation.images.left_arm"]), 6)
        self.assertEqual(buffer["episode_index"], 3)  # scalars untouched
        self.assertEqual(buffer["frame_index"], list(range(6)))
        self.assertEqual(row_times, [100.0 + i * 0.1 for i in range(6)])

    def test_cut_lands_before_the_first_late_row(self) -> None:
        # A row whose capture time is past the cut ends the kept prefix even
        # if a later row's time happens to be earlier (times are not resorted).
        dataset = _fake_dataset(5)
        row_times = [1.0, 2.0, 3.5, 3.0, 4.0]
        self.assertEqual(trim_episode_after(dataset, row_times, 3.2), 3)
        self.assertEqual(dataset.writer.episode_buffer["size"], 2)

    def test_cut_after_the_last_row_is_a_no_op(self) -> None:
        dataset = _fake_dataset(4)
        row_times = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(trim_episode_after(dataset, row_times, 4.0), 0)
        self.assertEqual(dataset.writer.episode_buffer["size"], 4)
        self.assertEqual(trim_episode_after(dataset, row_times, 9.0), 0)

    def test_cut_before_the_first_row_empties_the_take(self) -> None:
        dataset = _fake_dataset(3)
        row_times = [5.0, 6.0, 7.0]
        self.assertEqual(trim_episode_after(dataset, row_times, 4.0), 3)
        self.assertEqual(dataset.writer.episode_buffer["size"], 0)
        self.assertEqual(row_times, [])

    def test_empty_buffer_is_tolerated(self) -> None:
        dataset = mock.Mock()
        dataset.fps = 30
        dataset.writer.episode_buffer = None
        self.assertEqual(trim_episode_after(dataset, [1.0], 0.0), 0)


class InProcessRecorderTrimTest(unittest.TestCase):
    def _recorder(self, n_rows: int) -> InProcessRecorder:
        recorder = object.__new__(InProcessRecorder)
        recorder._dataset = _fake_dataset(n_rows)
        recorder._thread = None
        recorder._stop = None
        recorder._frames = {"n": n_rows}
        recorder._row_times = [200.0 + i * 0.1 for i in range(n_rows)]
        recorder._capture_error = {"v": None}
        recorder._fatal_error = None
        return recorder

    def test_stop_capture_trims_and_reports_the_kept_row_count(self) -> None:
        recorder = self._recorder(10)
        rows, error = recorder.stop_capture(trim_after=200.45)
        self.assertEqual((rows, error), (5, None))
        self.assertEqual(recorder._dataset.writer.episode_buffer["size"], 5)
        self.assertEqual(recorder.frame_count(), 5)

    def test_stop_capture_without_a_cut_keeps_every_row(self) -> None:
        recorder = self._recorder(10)
        self.assertEqual(recorder.stop_capture(), (10, None))
        self.assertEqual(recorder._dataset.writer.episode_buffer["size"], 10)


class RecorderProcessTrimProtocolTest(unittest.TestCase):
    def test_client_sends_the_cut_with_stop_capture(self) -> None:
        recorder = object.__new__(record_proc.DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._fatal_error = None
        recorder._closed = False
        recorder._conn = mock.Mock()
        recorder._conn.poll.return_value = True
        recorder._conn.recv.return_value = ("capture_stopped", 42, None)

        self.assertEqual(recorder.stop_capture(trim_after=77.5), (42, None))
        recorder._conn.send.assert_called_once_with(("stop_capture", 77.5))

        recorder._conn.send.reset_mock()
        recorder.stop_capture()
        recorder._conn.send.assert_called_once_with(("stop_capture", None))


if __name__ == "__main__":
    unittest.main()
