from __future__ import annotations

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
        self.writer = SnapshotWriter(_OBS, _ACT)
        self.reader = SnapshotReader(self.writer.name, _OBS, _ACT)

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

    def _fill_ring(self) -> int:
        n = shm_frames._SNAP_RING_CAPACITY
        for i in range(1, n + 1):
            self.writer.write(_obs(i), _act(i), float(i))
        return n

    def test_raced_copy_is_retaken(self) -> None:
        """A write landing mid-copy invalidates that copy; the next one is used."""
        n = self._fill_ring()
        real_copy = self.reader._copy_data
        calls = 0

        def racing_copy():
            nonlocal calls
            calls += 1
            data = real_copy()
            if calls == 1:
                # Reuse the oldest slot while the reader is "between" its two
                # bulk copies: the metadata comparison must reject this copy.
                self.writer.write(_obs(n + 1), _act(n + 1), float(n + 1))
            return data

        with patch.object(self.reader, "_copy_data", racing_copy):
            joint_obs, action, ts, _ = self.reader.read_nearest(300.2)
        self.assertEqual(calls, 2)
        self.assertEqual(ts, 300.0)
        self.assertEqual(joint_obs, _obs(300))
        self.assertEqual(action, _act(300))
        # The retaken copy already includes the racing write.
        self.assertEqual(self.reader.read_latest()[2], float(n + 1))

    def test_persistently_raced_copy_is_a_transient_miss(self) -> None:
        n = self._fill_ring()
        real_copy = self.reader._copy_data
        calls = 0

        def always_racing_copy():
            nonlocal calls
            calls += 1
            data = real_copy()
            self.writer.write(_obs(n + calls), _act(n + calls), float(n + calls))
            return data

        with patch.object(self.reader, "_copy_data", always_racing_copy):
            self.assertIsNone(self.reader.read_nearest(300.0))
        self.assertEqual(calls, shm_frames._SNAP_READ_ATTEMPTS)
        # Nothing is poisoned: an unraced read succeeds right after.
        self.assertEqual(self.reader.read_nearest(300.0)[2], 300.0)

    def test_in_flight_oldest_slots_are_given_up_not_fatal(self) -> None:
        n = self._fill_ring()
        meta = self.writer._slot_meta
        # Generation 1 lives in slot 0. An odd seq is a write in progress.
        meta["seq"][0] += 1
        self.assertEqual(self.reader.read_nearest(2.0)[2], 2.0)
        self.assertEqual(self.reader.read_nearest(300.4)[2], 300.0)
        self.assertEqual(self.reader.read_latest()[2], float(n))
        # The abandoned generation is outside the window, never clamped onto.
        self.assertIsNone(self.reader.read_nearest(1.0))
        # Only a bounded run of oldest slots is skipped; past that the copy is
        # a miss (the caller retries), not a wrong answer.
        for slot in range(1, shm_frames._SNAP_OLDEST_SKIP):
            meta["seq"][slot] += 1
        self.assertIsNone(self.reader.read_nearest(300.4))
        meta["seq"][shm_frames._SNAP_OLDEST_SKIP - 1] += 1
        self.assertEqual(self.reader.read_nearest(300.4)[2], 300.0)

    def test_writer_never_waits_on_a_stalled_reader(self) -> None:
        """A reader descheduled mid-copy must cost the control loop nothing."""
        n = self._fill_ring()
        real_copy = self.reader._copy_data
        in_copy = threading.Event()
        release = threading.Event()
        first = [True]

        def stalled_copy():
            data = real_copy()
            if first[0]:
                first[0] = False
                in_copy.set()
                release.wait(5.0)
            return data

        result: list = []
        with patch.object(self.reader, "_copy_data", stalled_copy):
            t = threading.Thread(
                target=lambda: result.append(self.reader.read_nearest(300.0))
            )
            t.start()
            self.assertTrue(in_copy.wait(5.0))
            t0 = time.perf_counter()
            for i in range(n + 1, n + 4):
                self.assertTrue(self.writer.write(_obs(i), _act(i), float(i)))
            self.assertLess(time.perf_counter() - t0, 0.05)
            release.set()
            t.join(5.0)
        self.assertFalse(t.is_alive())
        # The stalled copy was raced by those writes and retaken.
        self.assertEqual(result[0][2], 300.0)
        self.assertEqual(result[0][0], _obs(300))

    def test_concurrent_reads_are_never_torn(self) -> None:
        """Whatever the interleaving, a returned record is one whole write."""
        stop = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            i = 1
            while not stop.is_set():
                self.writer.write(_obs(i), _act(i), float(i), i % 7 == 0)
                i += 1

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            # Count reads, not wall time: the writer thread contends for the
            # GIL, so throughput varies wildly with host load.
            deadline = time.perf_counter() + 5.0
            reads = 0
            while reads < 200 and time.perf_counter() < deadline:
                latest = self.reader.read_latest()
                if latest is None:
                    continue
                joint_obs, action, ts, intervention = latest
                i = int(ts)
                if joint_obs != _obs(i) or action != _act(i):
                    errors.append(f"torn record at ts={ts}")
                    break
                if intervention != (i % 7 == 0):
                    errors.append(f"wrong intervention flag at ts={ts}")
                    break
                nearest = self.reader.read_nearest(ts - 3.5)
                if nearest is not None and nearest[0] != _obs(int(nearest[2])):
                    errors.append(f"torn nearest record at ts={nearest[2]}")
                    break
                reads += 1
        finally:
            stop.set()
            t.join(5.0)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(reads, 200)


if __name__ == "__main__":
    unittest.main()
