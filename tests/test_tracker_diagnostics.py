from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from almond_axol.tracker.diagnostics import FaultDiagnosticRecorder


class FaultDiagnosticRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def recorder(self, **options: Any) -> tuple[FaultDiagnosticRecorder, int]:
        settings = {
            "directory": Path(self.temporary.name),
            "pre_seconds": 1.0,
            "post_seconds": 0.06,
            "cooldown_seconds": 0.0,
            **options,
        }
        recorder = FaultDiagnosticRecorder(**settings)
        child_fd = os.dup(recorder.start())
        recorder.child_started()
        self.addCleanup(recorder.close)
        self.addCleanup(os.close, child_fd)
        return recorder, child_fd

    def wait_for(self, predicate: Any, message: str = "condition") -> None:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail(f"timed out waiting for {message}")

    def send(self, fd: int, packet: bytes) -> None:
        deadline = time.monotonic() + 2.0
        remaining = packet
        while remaining:
            try:
                written = os.write(fd, remaining)
                remaining = remaining[written:]
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.fail("diagnostic pipe did not drain")
                time.sleep(0.001)

    def native(self, seq: int, *, drops: int = 0) -> bytes:
        return (
            json.dumps(
                {
                    "v": 1,
                    "seq": seq,
                    "t": time.monotonic(),
                    "kind": "imu",
                    "device": "WM0",
                    "drops": drops,
                    "device_tick32": 1000 + seq,
                    "gyro": [1, 2, 3],
                }
            ).encode()
            + b"\n"
        )

    def read_capture(
        self, recorder: FaultDiagnosticRecorder
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        self.wait_for(lambda: recorder.status()["latest_path"], "saved capture")
        directory = Path(recorder.status()["latest_path"])
        self.assertFalse(directory.name.endswith("partial"))
        summary = json.loads((directory / "summary.json").read_text())
        with gzip.open(directory / "events.jsonl.gz", "rt") as source:
            events = [json.loads(line) for line in source]
        return summary, events

    def test_real_pipe_preserves_native_and_python_pre_and_post_fault(self) -> None:
        recorder, fd = self.recorder(metadata={"source_id": "test-source"})
        self.assertFalse(recorder.trigger("startup is not a fault"))
        self.assertFalse(os.get_blocking(fd))
        self.send(fd, self.native(1))
        recorder.record("pose_receipt", serial="LHR-example", position=[1, 2, 3])
        self.wait_for(lambda: recorder.status()["counters"]["python_events"] == 1)
        recorder.arm()
        self.assertTrue(recorder.trigger("sample skew", {"skew_seconds": 0.072}))
        self.assertFalse(recorder.trigger("same unresolved fault"))
        self.send(fd, self.native(2))
        recorder.record("recovered", ready=True)
        summary, events = self.read_capture(recorder)
        self.assertEqual(summary["reason"], "sample skew")
        self.assertEqual(summary["context"]["skew_seconds"], 0.072)
        self.assertEqual(summary["session"]["source_id"], "test-source")
        self.assertEqual(summary["stop_reason"], "complete")
        self.assertEqual(summary["event_count"], 4)
        native = [event for event in events if event["origin"] == "native"]
        self.assertEqual([event["seq"] for event in native], [1, 2])
        self.assertEqual(native[0]["device_tick32"], 1001)
        self.assertIn("received_monotonic", native[0])
        self.assertEqual(
            {event["kind"] for event in events}, {"imu", "pose_receipt", "recovered"}
        )

    def test_trigger_finishes_without_any_post_fault_input(self) -> None:
        recorder, _ = self.recorder()
        recorder.arm()
        self.assertTrue(recorder.trigger("all input vanished"))
        summary, events = self.read_capture(recorder)
        self.assertEqual(events, [])
        self.assertEqual(summary["stop_reason"], "complete")
        self.assertGreaterEqual(summary["finished_monotonic"], summary["end_monotonic"])

    def test_burst_capacity_limits_and_native_drop_counters_are_explicit(self) -> None:
        recorder, fd = self.recorder(max_events=12, max_bytes=5000)
        for seq in range(1, 301):
            self.send(fd, self.native(seq, drops=7 if seq > 250 else 0))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 300)
        recorder.arm()
        self.assertTrue(recorder.trigger("burst"))
        summary, events = self.read_capture(recorder)
        self.assertLessEqual(len(events), 12)
        self.assertLessEqual(summary["bytes"], 5000)
        self.assertTrue(summary["pre_window_capacity_truncated"])
        self.assertGreater(summary["counters_at_finish"]["ring_capacity_evictions"], 0)
        self.assertEqual(summary["counters_at_finish"]["native_reported_drops"], 7)
        self.assertEqual(events[-1]["seq"], 300)

    def test_invalid_and_oversized_packets_do_not_break_following_frames(self) -> None:
        recorder, fd = self.recorder(max_packet_bytes=512)
        self.send(fd, b"not-json\n[]\n{}\n" + b"x" * 1000)
        self.send(fd, b"x" * 1000 + b"\n" + self.native(1))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 1)
        counters = recorder.status()["counters"]
        self.assertEqual(counters["native_invalid_packets"], 3)
        self.assertEqual(counters["native_oversized_packets"], 1)
        recorder.arm()
        recorder.trigger("valid event after damaged input")
        _, events = self.read_capture(recorder)
        self.assertEqual(len(events), 1)

    def test_native_fragments_and_sequence_loss_are_retained(self) -> None:
        recorder, fd = self.recorder()
        packet = self.native(4, drops=3)
        self.send(fd, packet[:20])
        self.send(fd, packet[20:])
        self.send(fd, self.native(8, drops=6))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 2)
        counters = recorder.status()["counters"]
        self.assertEqual(counters["native_sequence_gaps"], 3)
        self.assertEqual(counters["native_reported_drops"], 6)

    def test_post_window_pressure_preserves_recent_events_and_reports_truncation(
        self,
    ) -> None:
        recorder, fd = self.recorder(max_events=3, post_seconds=0.2)
        self.send(fd, self.native(1))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 1)
        recorder.arm()
        recorder.trigger("before burst")
        self.wait_for(lambda: recorder._capture is not None)
        for seq in range(2, 10):
            self.send(fd, self.native(seq))
        summary, events = self.read_capture(recorder)
        self.assertEqual([event["seq"] for event in events], [7, 8, 9])
        self.assertEqual(summary["truncated_events"], 6)
        self.assertEqual(summary["counters_at_finish"]["capture_capacity_evictions"], 6)

    def test_pre_window_excludes_expired_events(self) -> None:
        recorder, fd = self.recorder(pre_seconds=0.03)
        self.send(fd, self.native(1))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 1)
        time.sleep(0.06)
        self.send(fd, self.native(2))
        self.wait_for(lambda: recorder.status()["counters"]["native_packets"] == 2)
        recorder.arm()
        recorder.trigger("new event")
        _, events = self.read_capture(recorder)
        self.assertEqual([event["seq"] for event in events], [2])

    def test_shutdown_flushes_partial_post_window_and_is_idempotent(self) -> None:
        recorder, fd = self.recorder(post_seconds=30)
        self.send(fd, self.native(1))
        recorder.arm()
        self.assertTrue(recorder.trigger("shutting down"))
        recorder.close()
        recorder.close()
        summary, events = self.read_capture(recorder)
        self.assertEqual(summary["stop_reason"], "shutdown")
        self.assertEqual(len(events), 1)
        self.assertEqual(recorder.status()["state"], "closed")
        recorder.record("ignored_after_close")
        self.assertFalse(recorder.trigger("ignored_after_close"))

    def test_save_failure_is_reported_without_raising_on_tracking_caller(self) -> None:
        blocked_path = Path(self.temporary.name) / "file-not-directory"
        blocked_path.write_text("existing file")
        recorder, _ = self.recorder(directory=blocked_path)
        recorder.arm()
        self.assertTrue(recorder.trigger("filesystem failure"))
        self.wait_for(lambda: recorder.status()["error"], "error status")
        self.assertEqual(recorder.status()["counters"]["capture_write_failures"], 1)
        self.assertIsNone(recorder.status()["latest_path"])
        recorder.close()

    def test_capture_count_and_cooldown_bound_disk_growth(self) -> None:
        recorder, _ = self.recorder(max_captures=1)
        recorder.arm()
        self.assertTrue(recorder.trigger("first"))
        self.read_capture(recorder)
        self.assertFalse(recorder.trigger("second"))
        self.assertEqual(len(list(Path(self.temporary.name).glob("fault-*"))), 1)
        another, _ = self.recorder(cooldown_seconds=100)
        another.arm()
        self.assertTrue(another.trigger("first"))
        self.read_capture(another)
        self.assertFalse(another.trigger("too soon"))

    def test_python_queue_and_payloads_are_bounded_before_worker_serialization(
        self,
    ) -> None:
        recorder = FaultDiagnosticRecorder(queue_size=2)
        # A paused worker must not make producers wait or grow the queue.
        recorder._accepting = True
        recorder.record("one", value=[1, 2])
        recorder.record("two")
        for _ in range(100):
            recorder.record("overflow")
        recorder.record("too_large", value="x" * 10_000_000)
        recorder.record("not_json", value=object())
        self.assertEqual(recorder._events.qsize(), 2)
        counters = recorder.status()["counters"]
        self.assertEqual(counters["python_queue_drops"], 100)
        self.assertEqual(counters["python_invalid_events"], 2)
        recorder.close()

    def test_python_events_copy_mutable_values_before_enqueue(self) -> None:
        recorder, _ = self.recorder()
        values = [1, 2, 3]
        recorder.record("pose_receipt", position=values)
        values[0] = 999
        recorder.arm()
        recorder.trigger("snapshot")
        _, events = self.read_capture(recorder)
        self.assertEqual(events[0]["position"], [1, 2, 3])

    def test_thread_start_failure_releases_pipe_and_close_remains_safe(self) -> None:
        recorder = FaultDiagnosticRecorder(directory=self.temporary.name)
        with patch("threading.Thread.start", side_effect=RuntimeError("no threads")):
            with self.assertRaisesRegex(RuntimeError, "no threads"):
                recorder.start()
        self.assertIsNone(recorder._read_fd)
        self.assertIsNone(recorder._write_fd)
        self.assertEqual(recorder.status()["state"], "error")
        self.assertFalse(recorder._accepting)
        recorder.close()

    def test_collector_failure_disables_new_events_and_exposes_error(self) -> None:
        recorder = FaultDiagnosticRecorder(directory=self.temporary.name)
        with patch.object(recorder, "_poll_native", side_effect=OSError("read failed")):
            recorder.start()
            self.wait_for(lambda: recorder.status()["state"] == "error")
            recorder.arm()
            self.assertFalse(recorder.trigger("cannot collect"))
            self.assertFalse(recorder._accepting)
            recorder.record("ignored")
            self.assertEqual(recorder._events.qsize(), 0)
            recorder.close()


if __name__ == "__main__":
    unittest.main()
