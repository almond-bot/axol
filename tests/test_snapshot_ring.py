from __future__ import annotations

import multiprocessing
import threading
import time
import unittest
from unittest.mock import patch

from almond_axol.video import shm_frames
from almond_axol.video.shm_frames import SnapshotReader, SnapshotWriter

_OBS = ["l.j1", "l.j2", "r.j1"]
_ACT = ["l.j1", "r.j1"]


def _obs(i: int) -> dict:
    return {k: float(i) + 0.1 * n for n, k in enumerate(_OBS)}


def _act(i: int) -> dict:
    return {k: -float(i) - 0.1 * n for n, k in enumerate(_ACT)}


class SnapshotRingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = multiprocessing.get_context("spawn").Lock()
        self.writer = SnapshotWriter(_OBS, _ACT, self.lock)
        self.reader = SnapshotReader(self.writer.name, _OBS, _ACT, self.lock)

    def tearDown(self) -> None:
        self.reader.close()
        self.writer.close()

    def test_empty_ring_reads_none(self) -> None:
        self.assertIsNone(self.reader.read_latest())
        self.assertIsNone(self.reader.read_nearest(1.0))

    def test_round_trip_and_nearest(self) -> None:
        for i in range(1, 6):
            self.assertTrue(self.writer.write(_obs(i), _act(i), float(i), i == 3))
        joint_obs, action, ts, intervention = self.reader.read_latest()
        self.assertEqual(ts, 5.0)
        self.assertEqual(joint_obs, _obs(5))
        self.assertEqual(action, _act(5))
        self.assertFalse(intervention)

        _, _, ts, intervention = self.reader.read_nearest(3.2)
        self.assertEqual(ts, 3.0)
        self.assertTrue(intervention)
        # Exact tie prefers the later sample.
        self.assertEqual(self.reader.read_nearest(3.5)[2], 4.0)
        # Outside the retained window is a miss, never a clamp.
        self.assertIsNone(self.reader.read_nearest(0.5))
        self.assertIsNone(self.reader.read_nearest(5.5))

    def test_nearest_after_wrap(self) -> None:
        n = shm_frames._SNAP_RING_CAPACITY + 40
        for i in range(1, n + 1):
            self.writer.write(_obs(i), _act(i), float(i))
        oldest = n - shm_frames._SNAP_RING_CAPACITY + 1
        self.assertIsNone(self.reader.read_nearest(oldest - 1.0))
        self.assertEqual(self.reader.read_nearest(float(oldest))[2], float(oldest))
        self.assertEqual(self.reader.read_nearest(n - 0.4)[2], float(n))

    def test_transient_hold_skips_sample_without_raising(self) -> None:
        self.writer.write(_obs(1), _act(1), 1.0)
        self.lock.acquire()
        try:
            with patch.object(shm_frames, "_SNAP_LOCK_WAIT_S", 0.001):
                t0 = time.perf_counter()
                self.assertFalse(self.writer.write(_obs(2), _act(2), 2.0))
                self.assertLess(time.perf_counter() - t0, 0.05)
                self.assertFalse(self.writer.write(_obs(3), _act(3), 3.0))
        finally:
            self.lock.release()
        with self.assertLogs(shm_frames._logger, level="WARNING") as logs:
            self.assertTrue(self.writer.write(_obs(4), _act(4), 4.0))
        self.assertIn("skipped 2 control snapshot(s)", logs.output[0])
        self.assertEqual(self.writer.skipped, 2)
        # The gap is invisible to the reader beyond a farther nearest sample.
        self.assertEqual(self.reader.read_nearest(2.4)[2], 1.0)
        self.assertEqual(self.reader.read_nearest(2.6)[2], 4.0)

    def test_abandoned_lock_still_raises(self) -> None:
        self.lock.acquire()
        try:
            with (
                patch.object(shm_frames, "_SNAP_LOCK_WAIT_S", 0.001),
                patch.object(shm_frames, "_SNAP_LOCK_ABANDONED_S", 0.02),
            ):
                deadline = time.perf_counter() + 2.0
                with self.assertRaises(RuntimeError):
                    while time.perf_counter() < deadline:
                        self.writer.write(_obs(1), _act(1), 1.0)
        finally:
            self.lock.release()

    def test_reader_hold_is_bounded_under_gil_contention(self) -> None:
        """The lock hold must not stretch with Python-level work in the reader."""
        for i in range(1, 300):
            self.writer.write(_obs(i), _act(i), float(i))
        stop = threading.Event()

        def spin() -> None:
            while not stop.is_set():
                sum(range(2000))

        hogs = [threading.Thread(target=spin, daemon=True) for _ in range(3)]
        for t in hogs:
            t.start()
        try:
            skipped_before = self.writer.skipped
            for i in range(300, 600):
                self.reader.read_nearest(float(i) - 0.5)
                self.writer.write(_obs(i), _act(i), float(i))
            self.assertEqual(self.writer.skipped, skipped_before)
        finally:
            stop.set()
            for t in hogs:
                t.join()


if __name__ == "__main__":
    unittest.main()
