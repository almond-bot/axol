from __future__ import annotations

import multiprocessing
import threading
import time
import unittest
from collections.abc import Callable
from unittest.mock import patch

from almond_axol.recording.record_proc import (
    DatasetRecorderProcess,
    InProcessRecorder,
    RecorderCaptureError,
    RecorderDatasetSaveError,
    _ENCODED_CONCEALMENT_WINDOW_S,
    _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW,
    _ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA,
    _STATE_LOSS_FATAL_S,
    _RowStatePairer,
    _align_independent_encoded_start,
    _classify_snapshot_miss,
    _concealment_within_budget,
    _describe_snapshot_miss,
    _missing_cadence_slots,
    format_capture_quality,
    run_capture_loop,
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

    def test_only_a_control_loop_silent_for_a_second_is_fatal(self) -> None:
        # A writer that paused briefly (newest snapshot 300 ms old) is a
        # per-row defect: attributed to the control loop, but not fatal.
        detail, fatal, latest = _classify_snapshot_miss(
            10.0, lambda: ({}, {}, 9.7, False), now=10.05
        )
        self.assertIn("control loop stopped publishing", detail)
        self.assertFalse(fatal)
        self.assertEqual(latest[2], 9.7)
        # Silent for the fatal threshold: the robot is moving unobserved.
        detail, fatal, _ = _classify_snapshot_miss(
            10.0, lambda: ({}, {}, 9.7, False), now=9.7 + _STATE_LOSS_FATAL_S
        )
        self.assertTrue(fatal)
        self.assertIn("no robot state has been published for", detail)
        # A live writer the recorder fell behind is never fatal.
        detail, fatal, _ = _classify_snapshot_miss(
            10.0, lambda: ({}, {}, 14.5, False), now=14.6
        )
        self.assertIn("recorder fell behind", detail)
        self.assertFalse(fatal)
        # A transiently unreadable ring (raced copy) is not fatal either.
        detail, fatal, latest = _classify_snapshot_miss(1.0, lambda: None)
        self.assertFalse(fatal)
        self.assertIsNone(latest)

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
        # The control loop's last publish predates every exposure by more
        # than the fatal silence threshold.
        stale = ({"state": 1}, {"target": 2}, base - 1.5 * _STATE_LOSS_FATAL_S, False)
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
        self.assertIn("no robot state has been published for", errors[0])

    def test_brief_control_loop_pause_drops_rows_and_continues(self) -> None:
        """A writer pause shorter than the fatal threshold costs rows only.

        The pause covers the first exposures, so there is no previous row's
        state to hold and the rows are dropped (see the held-state tests for
        the mid-take case).
        """
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=2)
        errors: list[str] = []
        quality: dict[str, int] = {}
        base = time.perf_counter() + 0.05
        step = 1.0 / 60.0
        cam = _EncodedCamera(
            [
                (b"\x00\x00\x00\x01\x65au", base + i * step, base + i * step)
                for i in range(4)
            ]
        )
        # The writer publishes a fresh snapshot at the episode start, then goes
        # quiet: its newest snapshot is 300 ms behind the first two exposures
        # (well past the pairing tolerance, well short of the fatal silence),
        # after which every exposure brackets normally.
        pause_until = base + 2 * step
        quiet = ({"state": 1}, {"target": 2}, base - 0.3, False)

        def read_latest() -> tuple[dict, dict, float, bool]:
            if cam.reads:
                return quiet
            return {"state": 1}, {"target": 2}, time.perf_counter(), False

        def read_nearest(target_ts: float) -> tuple[dict, dict, float, bool] | None:
            if target_ts < pause_until - step / 2:
                return None
            return {"state": 1}, {"target": 2}, target_ts, False

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
                read_snapshot=read_latest,
                read_snapshot_nearest=read_nearest,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
                quality=quality,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 2)
        self.assertEqual(quality, {"rows_dropped_state_miss": 2})
        self.assertIn("rows_dropped_state_miss=2", format_capture_quality(quality))

    def test_mid_take_state_miss_holds_the_previous_state_and_keeps_the_row(
        self,
    ) -> None:
        """A lost bracket mid-take costs neither the take nor the timeline.

        LeRobot stamps rows ``frame_index / fps``, so a dropped row silently
        shortens the episode by a frame. The previous row's state is at most a
        frame or two old — within the skew accepted on any ordinary row — so
        it is held instead and the row is kept.
        """
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=4)
        errors: list[str] = []
        quality: dict[str, int] = {}
        base = time.perf_counter() + 0.05
        step = 1.0 / 60.0
        cam = _EncodedCamera(
            [
                (b"\x00\x00\x00\x01\x65au", base + i * step, base + i * step)
                for i in range(4)
            ]
        )
        missing = base + 2 * step  # exposure 2's bracket is gone

        def read_latest() -> tuple[dict, dict, float, bool]:
            return {"state": "latest"}, {"target": 0}, time.perf_counter(), False

        def read_nearest(target_ts: float) -> tuple[dict, dict, float, bool] | None:
            if abs(target_ts - missing) < step / 2:
                return None
            index = round((target_ts - base) / step)
            return {"state": f"s{index}"}, {"target": index}, target_ts, False

        with (
            patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=lambda _f, values, prefix: {
                    f"{prefix}.{k}": v for k, v in values.items()
                },
            ),
            patch("lerobot.utils.visualization_utils.log_rerun_data"),
            patch(
                "almond_axol.recording.record_proc._SNAPSHOT_BRACKET_TIMEOUT_S",
                0.01,
            ),
        ):
            run_encoded_capture_loop(
                cameras={"cam": cam},
                read_snapshot=read_latest,
                read_snapshot_nearest=read_nearest,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
                quality=quality,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 4)
        self.assertEqual(
            [row["observation.state"] for row in dataset.rows],
            ["s0", "s1", "s1", "s3"],
        )
        self.assertEqual(quality, {"rows_with_held_state": 1})

    def test_recorder_backlog_drops_the_row_instead_of_the_episode(self) -> None:
        """A live writer that simply aged one exposure out must not reset.

        Regression for a customer session that lost episode 18 to exactly
        this: the writer never stopped (0 late/missed CAN ticks throughout),
        but a scheduling hiccup on the recorder side made one exposure age out
        of the retained history. That must drop one dataset row and keep
        capturing — not raise, discard the take, and send the operator
        through a return-to-rest/re-record cycle.
        """
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=2)
        errors: list[str] = []
        base = time.perf_counter() + 0.05
        step = 1.0 / 60.0
        cam = _EncodedCamera(
            [
                (b"\x00\x00\x00\x01\x65au", base + i * step, base + i * step)
                for i in range(3)
            ]
        )

        # A fixed timestamp well ahead of every camera exposure below: the
        # writer is alive and always ahead of whatever exposure the recorder
        # is currently pairing, i.e. a recorder backlog, not a control-process
        # stall.
        writer_latest_ts = base + 10.0

        def read_latest() -> tuple[dict, dict, float, bool]:
            return {"state": 1}, {"target": 2}, writer_latest_ts, False

        def read_nearest(target_ts: float) -> tuple[dict, dict, float, bool] | None:
            # Only the very first exposure aged out of the retained history;
            # every later one brackets fine.
            if target_ts < base + step / 2:
                return None
            return {"state": 1}, {"target": 2}, target_ts, False

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
                read_snapshot=read_latest,
                read_snapshot_nearest=read_nearest,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=60,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 2)


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
        dataset, errors, repairs, snapshot_times, _quality = self._run_encoded_q(
            cameras, fps=fps, rows=rows
        )
        return dataset, errors, repairs, snapshot_times

    def _run_encoded_q(
        self,
        cameras: dict[str, _EncodedCamera],
        *,
        fps: int,
        rows: int,
    ) -> tuple[_CaptureDataset, list[str], list[dict], list[float], dict[str, int]]:
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=rows)
        errors: list[str] = []
        repairs: list[dict] = []
        snapshot_times: list[float] = []
        quality: dict[str, int] = {}

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
                quality=quality,
            )
        return dataset, errors, repairs, snapshot_times, quality

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

    def test_missing_cadence_slots_measures_whole_periods_only(self) -> None:
        step = 1.0 / 60.0

        def slots(capture_ts: float, capture_fps: int = 60) -> int:
            return _missing_cadence_slots(
                previous_ts=1.0, capture_ts=capture_ts, fps=60, capture_fps=capture_fps
            )

        # Normal step and sub-half-period jitter: nothing missing.
        self.assertEqual(slots(1.0 + step), 0)
        self.assertEqual(slots(1.0 + 1.4 * step), 0)
        # Whole missing periods, rounded to the nearest grid slot (any size:
        # the caller bounds the repair, this helper only measures).
        self.assertEqual(slots(1.0 + 2 * step), 1)
        self.assertEqual(slots(1.0 + 3 * step - 0.0002), 2)
        self.assertEqual(slots(1.0 + 2.5 * step), 2)  # a half period rounds up
        self.assertEqual(slots(1.0 + 30 * step), 29)
        # A 120 -> 60 decimation legitimately spans two source intervals.
        self.assertEqual(slots(1.0 + 2 * (1 / 120), capture_fps=120), 0)
        self.assertEqual(slots(1.0 + 4 * (1 / 120), capture_fps=120), 1)
        # Non-advancing or non-finite timestamps are not a gap.
        self.assertEqual(slots(1.0), 0)
        self.assertEqual(slots(float("nan")), 0)

    def test_hole_longer_than_the_camera_loss_limit_is_fatal(self) -> None:
        base = time.perf_counter() + 0.1
        cameras = {
            "camera": _EncodedCamera(
                [(b"a0", base, base), (b"late", base + 1.5, base)],
            )
        }

        dataset, errors, repairs, _ = self._run_encoded(cameras, fps=60, rows=3)

        self.assertEqual(len(dataset.rows), 1)
        self.assertEqual(repairs, [])
        self.assertRegex(errors[0], "delivered no exposure for 1.50s")

    def test_duplicate_au_is_skipped_and_the_next_one_taken(self) -> None:
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera": _EncodedCamera(
                [
                    (b"a0", base, base),
                    (b"a0-again", base, base),  # duplicated exposure
                    (b"a1", base + step, base),
                    (b"a2", base + 2 * step, base),
                ]
            )
        }

        dataset, errors, repairs, _, quality = self._run_encoded_q(
            cameras, fps=60, rows=3
        )

        self.assertEqual(errors, [])
        self.assertEqual(
            [row["observation.camera"] for row in dataset.rows], [b"a0", b"a1", b"a2"]
        )
        self.assertEqual(repairs, [])
        self.assertEqual(quality, {"camera.unusable_aus_skipped": 1})

    def test_lagging_camera_is_advanced_to_its_peers(self) -> None:
        """A camera with a surplus AU rejoins the row instead of ending it."""
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            # camera_a delivers a burst of three surplus exposures after row
            # zero, so it soon trails camera_b by more than the alignment
            # limit (1.5 frames). Each such row skips camera_a's stale AU(s)
            # until it is back within the limit.
            "camera_a": _EncodedCamera(
                [
                    (b"a0", base, base),
                    (b"x1", base + 0.2 * step, base),
                    (b"x2", base + 0.4 * step, base),
                    (b"x3", base + 0.6 * step, base),
                    (b"a1", base + step, base),
                    (b"a2", base + 2 * step, base),
                    (b"a3", base + 3 * step, base),
                ]
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

        dataset, errors, _, _, quality = self._run_encoded_q(cameras, fps=60, rows=4)

        self.assertEqual(errors, [])
        # Row 1 accepts x1 (0.8 frames behind, within the limit); row 2 skips
        # x2 for x3; row 3 skips a1 for a2. camera_b is never touched.
        self.assertEqual(
            [row["observation.camera_a"] for row in dataset.rows],
            [b"a0", b"x1", b"x3", b"a2"],
        )
        self.assertEqual(
            [row["observation.camera_b"] for row in dataset.rows],
            [b"b0", b"b1", b"b2", b"b3"],
        )
        self.assertEqual(quality.get("camera_a.exposures_skipped_to_realign"), 2)
        self.assertNotIn("rows_dropped_camera_skew", quality)

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

    def test_gap_burst_over_budget_is_still_repaired_and_flagged(self) -> None:
        """Past the rate budget the take survives; the source is flagged.

        The budget used to end the episode. Under the fail-open policy the
        hole is repaired the same way, the repair event carries
        ``within_budget=False``, and the quality counters record the
        over-budget event so the unhealthy camera is found in the log.
        """
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        packets = [(b"a0", base, base)]
        # One event over the burst budget, back to back, each a single frame.
        for i in range(_ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + 1):
            ts = base + 2 * (i + 1) * step
            packets.append((f"a{2 * (i + 1)}".encode(), ts, base))
        n_packets = len(packets)
        cameras = {"camera": _EncodedCamera(packets)}

        dataset, errors, repairs, _, quality = self._run_encoded_q(
            cameras, fps=60, rows=2 * n_packets - 1
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(repairs), _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + 1)
        self.assertEqual(len(dataset.rows), 2 * n_packets - 1)
        self.assertEqual(
            [event["within_budget"] for event in repairs],
            [True] * _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + [False],
        )
        self.assertEqual(quality["camera.concealment_over_budget_events"], 1)
        self.assertEqual(
            quality["camera.concealed_frames"],
            _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW + 1,
        )

    def test_long_hole_within_the_loss_limit_is_repaired(self) -> None:
        # A 20-frame (333 ms) hole used to be fatal (holes were capped at two
        # frames). It is now repaired: the prior IDR fills every missing slot.
        base = time.perf_counter() + 0.1
        step = 1.0 / 60.0
        cameras = {
            "camera": _EncodedCamera(
                [(b"a0", base, base), (b"a21", base + 21 * step, base)]
            )
        }

        dataset, errors, repairs, _, quality = self._run_encoded_q(
            cameras, fps=60, rows=22
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 22)
        self.assertEqual(
            [row["observation.camera"] for row in dataset.rows],
            [b"a0"] * 21 + [b"a21"],
        )
        self.assertEqual([event["missing_frames"] for event in repairs], [20])
        self.assertEqual(quality["camera.concealed_frames"], 20)

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

    def test_row_zero_alignment_tolerates_a_dropped_prefix_frame(self) -> None:
        # Nothing before row zero is recorded, so a hole (or a duplicate) in
        # the discarded prefix must not end the take before it starts.
        packets = {"overhead": (b"o", 1.0, 1.0), "wrist": (b"w", 1.1, 1.0)}
        queued = [(b"jump", 1.075, 1.1), (b"dup", 1.075, 1.15), (b"o1", 1.1, 1.2)]

        aligned, dropped = _align_independent_encoded_start(
            packets,
            lambda _name: queued.pop(0),
            fps=30,
            capture_fps=dict.fromkeys(packets, 60),
        )

        self.assertEqual(aligned["overhead"][0], b"o1")
        self.assertEqual(aligned["wrist"][0], b"w")
        self.assertEqual(dropped, {"overhead": 3})

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

        self.assertEqual(conn.sent, [("finish_episode", None)])
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


class _RawCamera:
    """Scripted ``read_at_or_after`` camera for the raw (tick-paced) loop.

    ``script`` holds one entry per call: a ``(frame, capture_ts)`` tuple, or
    ``TimeoutError`` to simulate no fresh frame arriving within the timeout.
    Once the script is exhausted the camera keeps delivering fresh frames at
    the requested tick time so the dataset can reach its row target.
    """

    def __init__(self, script: list[object]) -> None:
        self.script = list(script)
        self.calls = 0

    def read_at_or_after(
        self, target_perf_ts: float, timeout_ms: float
    ) -> tuple[object, float, float]:
        del timeout_ms
        self.calls += 1
        if self.script:
            entry = self.script.pop(0)
            if entry is TimeoutError:
                raise TimeoutError("no frame")
            frame, cap_ts = entry  # type: ignore[misc]
            return frame, cap_ts, cap_ts
        return b"fresh", target_perf_ts, target_perf_ts


class RowStatePairerTest(unittest.TestCase):
    def _snap(self, ts: float, tag: str) -> tuple[dict, dict, float, bool]:
        return {"state": tag}, {"target": tag}, ts, False

    def test_holds_previous_state_only_while_within_tolerance(self) -> None:
        quality: dict[str, int] = {}
        pairer = _RowStatePairer(label="exposure", can_drop=True, quality=quality)
        step = 1.0 / 60.0

        fresh = self._snap(10.0, "fresh")
        self.assertEqual(pairer.resolve(10.0, fresh, None), (fresh, 0.0))

        # Bracket gone one and two frames later: hold the fresh state.
        snap, skew = pairer.resolve(10.0 + step, None, "aged out")
        self.assertIs(snap, fresh)
        self.assertAlmostEqual(skew, step)
        snap, _ = pairer.resolve(10.0 + 2 * step, None, "aged out")
        self.assertIs(snap, fresh)
        # A distant candidate is no better than a missing one.
        snap, _ = pairer.resolve(10.0 + 2.5 * step, self._snap(10.4, "far"), None)
        self.assertIs(snap, fresh)
        # Past the tolerance the held state is as stale as any other: drop.
        snap, skew = pairer.resolve(10.0 + 4 * step, None, "aged out")
        self.assertIsNone(snap)
        self.assertEqual(skew, float("inf"))
        snap, _ = pairer.resolve(10.0 + 4 * step, self._snap(10.4, "far"), None)
        self.assertIsNone(snap)

        self.assertEqual(
            quality,
            {
                "rows_with_held_state": 3,
                "rows_dropped_state_miss": 1,
                "rows_dropped_state_skew": 1,
            },
        )
        self.assertEqual(pairer.misses, 5)

    def test_no_previous_row_drops_or_keeps_by_transport(self) -> None:
        droppable = _RowStatePairer(label="exposure", can_drop=True, quality={})
        self.assertEqual(
            droppable.resolve(1.0, None, "nothing yet"), (None, float("inf"))
        )
        far = self._snap(1.3, "far")
        snap, skew = droppable.resolve(1.0, far, None)
        self.assertIsNone(snap)
        self.assertAlmostEqual(skew, 0.3)

        predictive = _RowStatePairer(label="exposure", can_drop=False, quality={})
        # Nothing to pair with at all still drops; a distant candidate is kept.
        self.assertEqual(
            predictive.resolve(1.0, None, "nothing yet"), (None, float("inf"))
        )
        snap, skew = predictive.resolve(1.0, far, None)
        self.assertIs(snap, far)
        self.assertAlmostEqual(skew, 0.3)


class RawCaptureLoopFailOpenTest(unittest.TestCase):
    def _run_raw(
        self,
        cameras: dict[str, _RawCamera],
        *,
        rows: int,
        fps: int = 60,
        bracket_missing: Callable[[float], bool] = lambda _ts: False,
        latest_ts: Callable[[], float] = time.perf_counter,
    ) -> tuple[_CaptureDataset, list[str], dict[str, int]]:
        stop = threading.Event()
        dataset = _CaptureDataset(stop, stop_after=rows)
        errors: list[str] = []
        quality: dict[str, int] = {}

        def read_latest() -> tuple[dict, dict, float, bool]:
            return {"state": "latest"}, {"target": 2}, latest_ts(), False

        def read_nearest(target_ts: float) -> tuple[dict, dict, float, bool] | None:
            if bracket_missing(target_ts):
                return None
            return {"state": target_ts}, {"target": 2}, target_ts, False

        with (
            patch(
                "lerobot.utils.feature_utils.build_dataset_frame",
                side_effect=lambda _f, values, prefix: dict(values),
            ),
            patch("lerobot.utils.visualization_utils.log_rerun_data"),
        ):
            run_capture_loop(
                cameras=cameras,
                read_snapshot=read_latest,
                read_snapshot_nearest=read_nearest,
                dataset=dataset,
                robot_obs_proc=lambda obs: obs,
                fps=fps,
                task="test",
                rerun_ip=None,
                stop_event=stop,
                on_error=errors.append,
                quality=quality,
            )
        return dataset, errors, quality

    def test_one_missing_frame_skips_the_tick_not_the_take(self) -> None:
        cam = _RawCamera([TimeoutError])

        dataset, errors, quality = self._run_raw({"cam": cam}, rows=3)

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 3)
        self.assertEqual(quality, {"cam.ticks_without_frame": 1})

    def test_stale_and_invalid_frames_skip_their_ticks(self) -> None:
        # The scripted exposures sit before the loop's tick clock so the
        # camera's later (tick-timed) frames still advance past them.
        first = time.perf_counter() - 1.0
        cam = _RawCamera(
            [
                (b"f0", first),
                (b"f0-again", first),  # repeated exposure
                (b"bad", float("nan")),  # unusable timestamp
            ]
        )

        dataset, errors, quality = self._run_raw({"cam": cam}, rows=3)

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 3)
        self.assertEqual(
            quality,
            {"cam.ticks_with_stale_frame": 1, "cam.ticks_with_invalid_timestamp": 1},
        )

    def test_mid_take_state_miss_holds_the_previous_state(self) -> None:
        # Exposures are the tick times themselves (see _RawCamera); the second
        # tick's bracket is gone and the writer's newest snapshot is 300 ms
        # stale (a brief pause). The row keeps the first tick's state instead
        # of being skipped, so the timeline stays true.
        seen: list[float] = []

        def bracket_missing(target_ts: float) -> bool:
            if not seen or target_ts > seen[-1]:
                seen.append(target_ts)
            return len(seen) == 2 and target_ts == seen[1]

        def latest_ts() -> float:
            # Fresh for the episode-start gate, then quiet.
            return time.perf_counter() - (0.3 if seen else 0.0)

        with patch(
            "almond_axol.recording.record_proc._SNAPSHOT_BRACKET_TIMEOUT_S", 0.01
        ):
            dataset, errors, quality = self._run_raw(
                {"cam": _RawCamera([])},
                rows=3,
                bracket_missing=bracket_missing,
                latest_ts=latest_ts,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(dataset.rows), 3)
        states = [row["state"] for row in dataset.rows]
        self.assertEqual(states[0], states[1])
        self.assertGreater(states[2], states[1])
        self.assertEqual(quality, {"rows_with_held_state": 1})

    def test_camera_silent_for_the_loss_limit_ends_the_take(self) -> None:
        cam = _RawCamera([TimeoutError] * 4)

        with patch("almond_axol.recording.record_proc._CAMERA_LOSS_FATAL_S", 0.0):
            dataset, errors, quality = self._run_raw({"cam": cam}, rows=3)

        self.assertEqual(dataset.rows, [])
        self.assertEqual(len(errors), 1)
        self.assertRegex(errors[0], "produced no fresh frame for .*s \\(limit 0.0s")
        self.assertEqual(quality, {})


class CaptureQualitySummaryTest(unittest.TestCase):
    def test_format_is_sorted_and_names_a_clean_take(self) -> None:
        self.assertEqual(
            format_capture_quality({}), "clean (no mitigated capture defects)"
        )
        self.assertEqual(
            format_capture_quality(
                {"rows_dropped_state_miss": 2, "cam.concealed_frames": 1}
            ),
            "cam.concealed_frames=1, rows_dropped_state_miss=2",
        )


if __name__ == "__main__":
    unittest.main()
