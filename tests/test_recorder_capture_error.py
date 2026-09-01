from __future__ import annotations

import multiprocessing
import unittest
from unittest.mock import patch

from almond_axol.recording.record_proc import (
    DatasetRecorderProcess,
    InProcessRecorder,
)


class _FakeDataset:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def clear_episode_buffer(self) -> None:
        self.events.append("clear")


class _FakeVerifier:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("verifier-close")


class DatasetRecorderCaptureErrorTest(unittest.TestCase):
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

        self.assertEqual(events, ["clear", "finalize", "verifier-close"])

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
