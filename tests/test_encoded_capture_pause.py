"""``record_event`` on the relay-encoded capture loop (DAgger's frozen gap).

While paused every arriving AU is discarded and no row is appended; on
resume the loop forgets its cross-row continuity (so the gap is neither a
concealable hole nor a cadence drift) and re-runs the row-zero alignment, so
the first rows after the gap are as synchronized as an episode's first.
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from unittest.mock import patch

from almond_axol.recording.record_proc import run_encoded_capture_loop

_STEP = 1.0 / 60.0


class _FeedCamera:
    """An encoded reader double fed from the test thread."""

    frames_are_independent = True
    capture_fps = 60
    pending = 0

    def __init__(self) -> None:
        self._queue: deque[tuple[bytes, float, float]] = deque()
        self._cond = threading.Condition()
        self.reads = 0

    def begin_flush(self) -> None:
        pass

    def finish_flush(self) -> None:
        pass

    def feed(self, *packets: tuple[bytes, float, float]) -> None:
        with self._cond:
            self._queue.extend(packets)
            self._cond.notify_all()

    def read_next_au(self, timeout_ms: float) -> tuple[bytes, float, float]:
        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._cond:
            while not self._queue:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError
                self._cond.wait(remaining)
            self.reads += 1
            return self._queue.popleft()


class _RowsDataset:
    features: dict = {}

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.appended = threading.Condition()

    def add_frame(self, row: dict) -> None:
        with self.appended:
            self.rows.append(row)
            self.appended.notify_all()

    def wait_rows(self, n: int, timeout_s: float = 5.0) -> None:
        deadline = time.perf_counter() + timeout_s
        with self.appended:
            while len(self.rows) < n:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise AssertionError(f"only {len(self.rows)} of {n} rows arrived")
                self.appended.wait(remaining)


def _packets(
    prefix: str, base: float, indices: range
) -> list[tuple[bytes, float, float]]:
    return [
        (f"{prefix}{i}".encode(), base + i * _STEP, base + i * _STEP) for i in indices
    ]


class EncodedCapturePauseTest(unittest.TestCase):
    def _start_loop(
        self, cameras: dict[str, _FeedCamera]
    ) -> tuple[_RowsDataset, threading.Event, threading.Event, list, list, dict]:
        stop = threading.Event()
        record = threading.Event()
        record.set()
        dataset = _RowsDataset()
        errors: list[str] = []
        repairs: list[dict] = []
        quality: dict[str, int] = {}

        def build_frame(_features: dict, values: dict, prefix: str) -> dict:
            return {f"{prefix}.{name}": value for name, value in values.items()}

        def run() -> None:
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
                    read_snapshot_nearest=lambda ts: (
                        {"state": 1},
                        {"target": 2},
                        ts,
                        False,
                    ),
                    dataset=dataset,
                    robot_obs_proc=lambda obs: obs,
                    fps=60,
                    task="test",
                    rerun_ip=None,
                    stop_event=stop,
                    repair_events=repairs,
                    on_error=errors.append,
                    quality=quality,
                    record_event=record,
                )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        self.addCleanup(stop.set)
        return dataset, stop, record, errors, repairs, quality

    def test_paused_exposures_are_discarded_and_resume_splices_rows(self) -> None:
        cam_a, cam_b = _FeedCamera(), _FeedCamera()
        dataset, stop, record, errors, repairs, quality = self._start_loop(
            {"a": cam_a, "b": cam_b}
        )
        base = time.perf_counter() + 0.05
        cam_a.feed(*_packets("a", base, range(0, 3)))
        cam_b.feed(*_packets("b", base, range(0, 3)))
        dataset.wait_rows(3)

        # Freeze: a long gap's worth of exposures arrives while paused. None
        # of them may become rows, and the loop must keep draining them.
        record.clear()
        time.sleep(0.05)  # let the loop observe the pause
        gap = range(3, 3 + 120)  # 2 s of exposures — far past the concealable gap
        cam_a.feed(*_packets("a", base, gap))
        cam_b.feed(*_packets("b", base, gap))
        deadline = time.perf_counter() + 2.0
        while (cam_a.reads < 3 + 120 or cam_b.reads < 3 + 120) and (
            time.perf_counter() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual((cam_a.reads, cam_b.reads), (123, 123), "gap AUs not drained")
        self.assertEqual(len(dataset.rows), 3)

        # Resume: the next exposures are this segment's row zero. (AUs still
        # queued at the resume instant were exposed during the gap and are
        # dropped too, so give the loop a poll interval to observe the resume
        # before the live exposures arrive.)
        record.set()
        time.sleep(0.05)
        after = range(3 + 120, 3 + 120 + 3)
        cam_a.feed(*_packets("a", base, after))
        cam_b.feed(*_packets("b", base, after))
        dataset.wait_rows(6)
        stop.set()

        self.assertEqual(errors, [])
        self.assertEqual(repairs, [], "the gap must not be concealed as a hole")
        self.assertEqual(quality, {})
        self.assertEqual(
            [row["observation.a"] for row in dataset.rows],
            [b"a0", b"a1", b"a2", b"a123", b"a124", b"a125"],
        )
        self.assertEqual(
            [row["observation.b"] for row in dataset.rows],
            [b"b0", b"b1", b"b2", b"b123", b"b124", b"b125"],
        )

    def test_resume_realigns_row_zero_across_cameras(self) -> None:
        """A camera that got ahead during the gap is re-aligned on resume."""
        cam_a, cam_b = _FeedCamera(), _FeedCamera()
        dataset, stop, record, errors, repairs, quality = self._start_loop(
            {"a": cam_a, "b": cam_b}
        )
        base = time.perf_counter() + 0.05
        cam_a.feed(*_packets("a", base, range(0, 2)))
        cam_b.feed(*_packets("b", base, range(0, 2)))
        dataset.wait_rows(2)

        record.clear()
        time.sleep(0.05)
        # The relay keeps delivering through the freeze; the loop must see the
        # pause between two of these reads rather than sit in a blocked read.
        cam_a.feed(*_packets("a", base, range(2, 6)))
        cam_b.feed(*_packets("b", base, range(2, 6)))
        deadline = time.perf_counter() + 2.0
        while (cam_a.reads < 6 or cam_b.reads < 6) and time.perf_counter() < deadline:
            time.sleep(0.005)
        self.assertEqual((cam_a.reads, cam_b.reads), (6, 6))
        record.set()
        time.sleep(0.05)
        # Camera a's first post-gap AUs are two exposures older than b's
        # (past the alignment limit); only the aligned pair may form the
        # segment's first row.
        cam_a.feed(*_packets("a", base, range(9, 13)))
        cam_b.feed(*_packets("b", base, range(11, 13)))
        dataset.wait_rows(4)
        stop.set()

        self.assertEqual(errors, [])
        self.assertEqual(repairs, [])
        self.assertEqual(
            [row["observation.a"] for row in dataset.rows],
            [b"a0", b"a1", b"a11", b"a12"],
        )
        self.assertEqual(
            [row["observation.b"] for row in dataset.rows],
            [b"b0", b"b1", b"b11", b"b12"],
        )
        self.assertNotIn("rows_dropped_camera_skew", quality)

    def test_stop_while_paused_returns_promptly(self) -> None:
        cam = _FeedCamera()
        dataset, stop, record, errors, _repairs, _quality = self._start_loop({"a": cam})
        base = time.perf_counter() + 0.05
        cam.feed(*_packets("a", base, range(0, 1)))
        dataset.wait_rows(1)
        record.clear()
        time.sleep(0.05)
        started = time.perf_counter()
        stop.set()
        # The cleanup join asserts the thread ended; here just bound the wait.
        while time.perf_counter() - started < 1.0 and len(errors) == 0:
            time.sleep(0.01)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
