from __future__ import annotations

import multiprocessing
import threading
import time
import unittest
from unittest.mock import patch

from almond_axol.recording.record_proc import (
    DatasetRecorderProcess,
    InProcessRecorder,
    RecorderCaptureError,
    RecorderDatasetSaveError,
    _ENCODED_CONCEALMENT_WINDOW_S,
    _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW,
    _ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA,
    _align_independent_encoded_start,
    _concealable_encoded_gap_frames,
    _concealment_within_budget,
    _describe_snapshot_miss,
    run_encoded_capture_loop,
)


class SnapshotMissAttributionTest(unittest.TestCase):
    """The bracket-miss message must say which process fell behind."""

    def test_stale_newest_snapshot_blames_the_control_loop(self) -> None:
        # Exposure at t=10.000; the control loop last published at t=9.700 and
        # it is now t=10.050: the writer stalled, not the recorder.
        text = _describe_snapshot_miss(10.0, lambda: ({}, {}, 9.7, False), now=10.05)
        self.assertIn("control loop stopped publishing", text)
        self.assertIn("300 ms older than the exposure", text)
        self.assertIn("350 ms old now", text)
        self.assertNotIn("recorder fell behind", text)

    def test_exposure_behind_history_blames_the_recorder(self) -> None:
        text = _describe_snapshot_miss(10.0, lambda: ({}, {}, 14.5, False))
        self.assertIn("recorder fell behind", text)
        self.assertIn("4500 ms behind the newest snapshot", text)
        self.assertNotIn("control loop stopped", text)

    def test_no_snapshot_at_all(self) -> None:
        self.assertIn(
            "no robot-state snapshot", _describe_snapshot_miss(1.0, lambda: None)
        )

    def test_encoded_loop_error_carries_the_attribution(self) -> None:
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=10)
        errors: list[str] = []
        base = time.perf_counter() + 0.05
        cam = _EncodedCamera(
            [
                (b"\x00\x00\x00\x01\x65au", base + i / 60, base + i / 60)
                for i in range(6)
            ]
        )
        # The control loop's last publish predates every exposure.
        stale = ({"state": 1}, {"target": 2}, base - 0.3, False)
        with (
            patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=lambda _f, values, prefix: dict(values),
            ),
            patch("lerobot.utils.visualization_utils.log_rerun_data"),
            patch(
                "almond_axol.recording.record_proc._SNAPSHOT_BRACKET_TIMEOUT_S",
                0.01,
            ),
        ):
            run_encoded_capture_loop(
                cameras={"cam": cam},
                # Fresh enough to pass the episode-start gate, then never newer
                # than the exposures: the writer went quiet.
                read_snapshot=lambda: (
                    stale if cam.reads else ({}, {}, time.perf_counter(), False)
                ),
                read_snapshot_nearest=lambda _ts: None,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
            )
        self.assertEqual(len(errors), 1)
        self.assertIn("control loop stopped publishing", errors[0])


class _FakeDataset:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def clear_episode_buffer(self) -> None:
        self.events.append("clear")


class _FakeVerifier:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def suspend(self) -> None:
        self.events.append("verifier-suspend")

    def resume(self) -> None:
        self.events.append("verifier-resume")

    def close(self) -> None:
        self.events.append("verifier-close")


class _ReplyConnection:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.sent: list[object] = []

    def send(self, value: object) -> None:
        self.sent.append(value)

    def poll(self, _timeout: float) -> bool:
        return True

    def recv(self) -> object:
        return self.reply


class _EncodedCamera:
    frames_are_independent = True
    capture_fps = 60
    pending = 0

    def __init__(
        self,
        packets: list[tuple[bytes, float, float]],
        *,
        independent: bool = True,
        capture_fps: int = 60,
    ) -> None:
        self.packets = packets
        self.frames_are_independent = independent
        self.capture_fps = capture_fps
        self.reads = 0

    def begin_flush(self) -> None:
        pass

    def finish_flush(self) -> None:
        pass

    def read_next_au(self, timeout_ms: float) -> tuple[bytes, float, float]:
        del timeout_ms
        if not self.packets:
            raise TimeoutError
        self.reads += 1
        return self.packets.pop(0)


class _CaptureDataset:
    features: dict = {}

    def __init__(self, stop: threading.Event, stop_after: int = 1) -> None:
        self.stop = stop
        self.stop_after = stop_after
        self.rows: list[dict] = []

    def add_frame(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.stop_after:
            self.stop.set()


class DatasetRecorderCaptureErrorTest(unittest.TestCase):
    def _run_encoded(
        self,
        cameras: dict[str, _EncodedCamera],
        *,
        fps: int,
        rows: int,
    ) -> tuple[_CaptureDataset, list[str], list[dict], list[float]]:
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=rows)
        errors: list[str] = []
        repairs: list[dict] = []
        snapshot_times: list[float] = []

        def snapshot(ts: float) -> tuple[dict, dict, float, bool]:
            snapshot_times.append(ts)
            return {"state": 1}, {"target": 2}, ts, False

        def build_frame(_features: dict, values: dict, prefix: str) -> dict:
            return {f"{prefix}.{name}": value for name, value in values.items()}

        with (
            patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=build_frame,
            ),
            patch("lerobot.utils.visualization_utils.log_rerun_data"),
        ):
            run_encoded_capture_loop(
                cameras=cameras,
                read_snapshot=lambda: (
                    {"state": 1},
                    {"target": 2},
                    time.perf_counter(),
                    False,
                ),
                read_snapshot_nearest=snapshot,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=fps,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                repair_events=repairs,
                on_error=errors.append,
            )
        return dataset, errors, repairs, snapshot_times

    def test_bounded_gap_concealment_holds_future_au_and_state_grid(self) -> None:
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera_a": _EncodedCamera(
                [(b"a0", base, base), (b"a3", base + 3 * step, base)]
            ),
            "camera_b": _EncodedCamera(
                [
                    (b"b0", base, base),
                    (b"b1", base + step, base),
                    (b"b2", base + 2 * step, base),
                    (b"b3", base + 3 * step, base),
                ]
            ),
        }

        dataset, errors, repairs, snapshot_times = self._run_encoded(
            cameras, fps=60, rows=4
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [row["observation.camera_a"] for row in dataset.rows],
            [b"a0", b"a0", b"a0", b"a3"],
        )
        self.assertEqual(
            [row["observation.camera_b"] for row in dataset.rows],
            [b"b0", b"b1", b"b2", b"b3"],
        )
        self.assertEqual(cameras["camera_a"].reads, 2)
        self.assertEqual(cameras["camera_b"].reads, 4)
        self.assertEqual(len(snapshot_times), 4)
        for actual, expected in zip(
            snapshot_times, [base + i * step for i in range(4)]
        ):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["camera"], "camera_a")
        self.assertEqual(repairs[0]["frame_index"], 1)
        self.assertEqual(repairs[0]["missing_frames"], 2)

    def test_stop_mid_repair_reports_only_committed_synthetic_rows(self) -> None:
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera_a": _EncodedCamera(
                [(b"a0", base, base), (b"a3", base + 3 * step, base)]
            ),
            "camera_b": _EncodedCamera(
                [(b"b0", base, base), (b"b1", base + step, base)]
            ),
        }

        dataset, errors, repairs, _ = self._run_encoded(cameras, fps=60, rows=2)

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 2)
        self.assertEqual(repairs[0]["frame_index"], 1)
        self.assertEqual(repairs[0]["missing_frames"], 1)
        self.assertAlmostEqual(repairs[0]["concealed_ms"], 1000 / 60)

    def test_only_small_equal_rate_independent_grid_gaps_are_concealable(self) -> None:
        step = 1.0 / 60.0

        self.assertEqual(
            _concealable_encoded_gap_frames(
                previous_ts=1.0,
                capture_ts=1.0 + 3 * step - 0.0002,
                fps=60,
                capture_fps=60,
                frames_are_independent=True,
            ),
            2,
        )
        for kwargs in (
            {"capture_ts": 1.0 + 4 * step},
            {"capture_ts": 1.0 + 2.5 * step},
            {"capture_ts": 1.0 + 2 * step, "capture_fps": 120},
            {"capture_ts": 1.0 + 2 * step, "capture_fps": 30},
            {
                "capture_ts": 1.0 + 2 * step,
                "frames_are_independent": False,
            },
        ):
            args = dict(
                previous_ts=1.0,
                capture_ts=1.0 + 2 * step,
                fps=60,
                capture_fps=60,
                frames_are_independent=True,
            )
            args.update(kwargs)
            self.assertEqual(_concealable_encoded_gap_frames(**args), 0)

    def test_invalid_lower_capture_rate_cannot_enable_concealment(self) -> None:
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera": _EncodedCamera(
                [(b"a0", base, base), (b"a2", base + 2 * step, base)],
                capture_fps=30,
            )
        }

        dataset, errors, repairs, _ = self._run_encoded(cameras, fps=60, rows=3)

        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(repairs, [])
        self.assertRegex(errors[0], "dropped an encoded frame")

    def test_repeated_isolated_gaps_within_budget_are_concealed(self) -> None:
        # The ZED sources drop isolated frames in clusters around a record
        # start (this session: both overhead eyes at row 12, then again half a
        # second later). Each hole stays bounded, so the take survives.
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera": _EncodedCamera(
                [
                    (b"a0", base, base),
                    (b"a2", base + 2 * step, base),
                    (b"a4", base + 4 * step, base),
                ]
            )
        }

        dataset, errors, repairs, _ = self._run_encoded(cameras, fps=60, rows=5)

        self.assertEqual(errors, [])
        self.assertEqual(
            [row["observation.camera"] for row in dataset.rows],
            [b"a0", b"a0", b"a2", b"a2", b"a4"],
        )
        self.assertEqual([event["missing_frames"] for event in repairs], [1, 1])
        self.assertEqual([event["frame_index"] for event in repairs], [1, 3])

    def test_gap_burst_inside_window_is_fatal(self) -> None:
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        packets = [(b"a0", base, base)]
        # One event over the burst budget, back to back, each a single frame.
        for i in range(_ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + 1):
            ts = base + 2 * (i + 1) * step
            packets.append((f"a{2 * (i + 1)}".encode(), ts, base))
        cameras = {"camera": _EncodedCamera(packets)}

        dataset, errors, repairs, _ = self._run_encoded(
            cameras, fps=60, rows=len(packets) * 2
        )

        self.assertEqual(len(repairs), _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW)
        self.assertEqual(
            len(dataset.rows), 2 * _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + 1
        )
        self.assertRegex(errors[0], "dropped an encoded frame")

    def test_isolated_gaps_every_few_seconds_survive_a_long_take(self) -> None:
        # Today's overhead pattern: one exposure lost every 1-4 s for the whole
        # take. Spaced past the burst window and well under the fraction cap,
        # every hole is repaired.
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        spacing = 90  # frames between the lost exposures (1.5 s at 60 Hz)
        packets = []
        frame = 0
        for _ in range(8):
            for _ in range(spacing - 1):
                packets.append((f"a{frame}".encode(), base + frame * step, base))
                frame += 1
            frame += 1  # the lost exposure
        cameras = {"camera": _EncodedCamera(packets)}

        dataset, errors, repairs, _ = self._run_encoded(
            cameras, fps=60, rows=8 * spacing - 1
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(repairs), 7)
        self.assertEqual({event["missing_frames"] for event in repairs}, {1})

    def test_concealment_budget_rules(self) -> None:
        fps = 60
        window = int(_ENCODED_CONCEALMENT_WINDOW_S * fps)
        kwargs = dict(concealed_frames=0, missing=1, total_rows=1000, fps=fps)
        # Two earlier events inside the window leave room for a third only.
        self.assertTrue(
            _concealment_within_budget(event_rows=[10, 50], row=100, **kwargs)
        )
        self.assertFalse(
            _concealment_within_budget(event_rows=[10, 50, 90], row=100, **kwargs)
        )
        # Once the oldest event ages out of the window a new one fits again.
        self.assertTrue(
            _concealment_within_budget(
                event_rows=[10, 50, 90], row=10 + window, **kwargs
            )
        )
        # The frame allowance is the larger of the floor and the fraction.
        self.assertTrue(
            _concealment_within_budget(
                event_rows=[],
                row=10,
                concealed_frames=_ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA - 1,
                missing=1,
                total_rows=10,
                fps=fps,
            )
        )
        self.assertFalse(
            _concealment_within_budget(
                event_rows=[],
                row=10,
                concealed_frames=_ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA,
                missing=1,
                total_rows=10,
                fps=fps,
            )
        )
        self.assertTrue(
            _concealment_within_budget(
                event_rows=[],
                row=10_000,
                concealed_frames=_ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA,
                missing=2,
                total_rows=10_000,
                fps=fps,
            )
        )

    def test_row_zero_alignment_advances_only_lagging_all_intra_streams(
        self,
    ) -> None:
        packets = {
            "overhead_left": (b"o0", 10.0000, 1.0),
            "overhead_right": (b"o1", 10.0000, 1.0),
            "left_arm": (b"l", 10.0333, 1.0),
            "right_arm": (b"r0", 10.0000, 1.0),
        }
        queued = {
            "overhead_left": [
                (b"o2", 10.0167, 1.1),
                (b"o4", 10.0334, 1.2),
            ],
            "overhead_right": [
                (b"o3", 10.0167, 1.1),
                (b"o5", 10.0334, 1.2),
            ],
            "right_arm": [
                (b"r1", 10.0167, 1.1),
                (b"r2", 10.0334, 1.2),
            ],
        }

        def read_next(name: str) -> tuple[bytes, float, float] | None:
            values = queued.get(name, [])
            return values.pop(0) if values else None

        aligned, dropped = _align_independent_encoded_start(
            packets,
            read_next,
            fps=60,
            capture_fps=dict.fromkeys(packets, 60),
        )

        times = [packet[1] for packet in aligned.values()]
        self.assertLessEqual(max(times) - min(times), 0.025)
        self.assertEqual(
            dropped,
            {"overhead_left": 2, "overhead_right": 2, "right_arm": 2},
        )
        self.assertEqual(aligned["left_arm"][0], b"l")

    def test_row_zero_alignment_fails_if_lagging_stream_stalls(self) -> None:
        packets = {"overhead": (b"o", 1.0, 1.0), "wrist": (b"w", 1.1, 1.0)}

        with self.assertRaisesRegex(TimeoutError, "overhead.*did not catch up"):
            _align_independent_encoded_start(
                packets,
                lambda _name: None,
                fps=30,
                capture_fps=dict.fromkeys(packets, 30),
            )

    def test_row_zero_alignment_rejects_a_dropped_prefix_frame(self) -> None:
        packets = {"overhead": (b"o", 1.0, 1.0), "wrist": (b"w", 1.1, 1.0)}

        with self.assertRaisesRegex(RuntimeError, "dropped an encoded frame"):
            _align_independent_encoded_start(
                packets,
                lambda _name: (b"jump", 1.075, 1.1),
                fps=30,
                capture_fps=dict.fromkeys(packets, 60),
            )

    def test_encoded_capture_saves_the_aligned_access_units_verbatim(self) -> None:
        stop = threading.Event()
        dataset = _CaptureDataset(stop)
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "overhead_left": _EncodedCamera(
                [
                    (b"ol0", base, base),
                    (b"ol1", base + step, base),
                    (b"ol2", base + 2 * step, base),
                ]
            ),
            "overhead_right": _EncodedCamera(
                [
                    (b"or0", base, base),
                    (b"or1", base + step, base),
                    (b"or2", base + 2 * step, base),
                ]
            ),
            "left_arm": _EncodedCamera([(b"la2", base + 2 * step, base)]),
            "right_arm": _EncodedCamera(
                [
                    (b"ra0", base, base),
                    (b"ra1", base + step, base),
                    (b"ra2", base + 2 * step, base),
                ]
            ),
        }
        errors: list[str] = []

        def snapshot(ts: float) -> tuple[dict, dict, float, bool]:
            return {"state": 1}, {"target": 2}, ts, False

        def build_frame(_features: dict, values: dict, prefix: str) -> dict:
            return {f"{prefix}.{name}": value for name, value in values.items()}

        with (
            patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=build_frame,
            ),
            patch("lerobot.utils.visualization_utils.log_rerun_data"),
        ):
            run_encoded_capture_loop(
                cameras=cameras,
                read_snapshot=lambda: snapshot(time.perf_counter()),
                read_snapshot_nearest=lambda ts: snapshot(ts),
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 1)
        row = dataset.rows[0]
        self.assertEqual(row["observation.overhead_left"], b"ol2")
        self.assertEqual(row["observation.overhead_right"], b"or2")
        self.assertEqual(row["observation.left_arm"], b"la2")
        self.assertEqual(row["observation.right_arm"], b"ra2")

    def test_in_process_close_discards_normal_unsaved_episode_before_finalize(
        self,
    ) -> None:
        events: list[str] = []
        recorder = InProcessRecorder.__new__(InProcessRecorder)
        recorder._thread = None
        recorder._stop = None
        recorder._capture_error = None
        recorder._dataset = _FakeDataset(events)
        recorder._config = {}
        recorder._episodes_recorded = 0
        recorder._verifier = _FakeVerifier(events)

        def finalize(*_args: object) -> None:
            events.append("finalize")

        with patch(
            "almond_axol.recording.record_proc._finalize_dataset",
            side_effect=finalize,
        ):
            recorder.close()

        self.assertEqual(
            events, ["verifier-resume", "clear", "finalize", "verifier-close"]
        )

    def test_in_process_finish_capture_error_clears_then_raises_typed(self) -> None:
        events: list[str] = []
        recorder = InProcessRecorder.__new__(InProcessRecorder)
        recorder._thread = None
        recorder._stop = None
        recorder._capture_error = "camera alignment failed"
        recorder._dataset = _FakeDataset(events)
        recorder._frames = {"n": 4}
        recorder._verifier = _FakeVerifier(events)

        with self.assertRaisesRegex(RecorderCaptureError, "alignment failed"):
            recorder.finish_episode()

        # Capture stopped: the verifier is released before the buffer is cleared.
        self.assertEqual(events, ["verifier-resume", "clear"])

    def test_in_process_finish_stop_failure_stays_fatal(self) -> None:
        recorder = InProcessRecorder.__new__(InProcessRecorder)
        with patch.object(
            recorder,
            "_stop_capture",
            side_effect=RuntimeError("capture thread did not stop"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                recorder.finish_episode()

        self.assertNotIsInstance(raised.exception, RecorderCaptureError)

    def test_in_process_save_dropped_frame_is_typed_precommit_rejection(self) -> None:
        events: list[str] = []
        recorder = InProcessRecorder.__new__(InProcessRecorder)
        recorder._thread = None
        recorder._stop = None
        recorder._capture_error = None
        recorder._dataset = _FakeDataset(events)
        recorder._verifier = _FakeVerifier(events)

        with (
            patch(
                "almond_axol.lerobot.nvenc_encoder.dropped_frames",
                return_value=2,
            ),
            self.assertRaisesRegex(RecorderCaptureError, "2 video frame"),
        ):
            recorder.save_episode()

        self.assertEqual(events, ["verifier-resume", "clear"])

    def test_process_finish_uses_post_join_capture_error_reply(self) -> None:
        conn = _ReplyConnection(("finished", 7, "camera alignment failed"))
        recorder = DatasetRecorderProcess.__new__(DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._conn = conn
        recorder._capture_error = None

        with self.assertRaisesRegex(RecorderCaptureError, "alignment failed"):
            recorder.finish_episode()

        self.assertEqual(conn.sent, [("finish_episode",)])
        self.assertEqual(recorder._capture_error, "camera alignment failed")

    def test_process_finish_non_capture_error_stays_fatal(self) -> None:
        conn = _ReplyConnection(("error", "capture thread did not stop"))
        recorder = DatasetRecorderProcess.__new__(DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._conn = conn

        with self.assertRaises(RuntimeError) as raised:
            recorder.finish_episode()

        self.assertNotIsInstance(raised.exception, RecorderCaptureError)

    def test_process_save_distinguishes_capture_rejection_from_commit_failure(
        self,
    ) -> None:
        recorder = DatasetRecorderProcess.__new__(DatasetRecorderProcess)
        recorder._lock = threading.Lock()
        recorder._capture_error = None

        recorder._conn = _ReplyConnection(("capture_error", "2 dropped frames"))
        with self.assertRaises(RecorderCaptureError):
            recorder.save_episode()

        recorder._conn = _ReplyConnection(("error", "mux prepare failed"))
        with self.assertRaises(RuntimeError) as prepare_failure:
            recorder.save_episode()
        self.assertNotIsInstance(prepare_failure.exception, RecorderCaptureError)

        recorder._conn = _ReplyConnection(("fatal", "parquet commit failed"))
        with self.assertRaises(RecorderDatasetSaveError):
            recorder.save_episode()

    def test_capture_error_uses_separate_nonblocking_channel(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        recv_conn, send_conn = ctx.Pipe(duplex=False)
        recorder = DatasetRecorderProcess.__new__(DatasetRecorderProcess)
        recorder._error_conn = recv_conn
        recorder._capture_error = None
        try:
            self.assertIsNone(recorder.poll_capture_error())

            send_conn.send("camera alignment failed")

            self.assertEqual(recorder.poll_capture_error(), "camera alignment failed")
            # The first failure stays visible after the pipe has been drained.
            self.assertEqual(recorder.poll_capture_error(), "camera alignment failed")
        finally:
            send_conn.close()
            recv_conn.close()


if __name__ == "__main__":
    unittest.main()
