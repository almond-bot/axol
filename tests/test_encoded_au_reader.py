from __future__ import annotations

import types
import unittest
from collections.abc import Iterable
from unittest.mock import patch

from almond_axol.video.shm_frames import EncodedAuReader

_IDR = b"\x00\x00\x00\x01\x65\x88"
_P_FRAME = b"\x00\x00\x00\x01\x41\x9a"


class _FakeBuffer:
    def __init__(self, au: bytes, pts: int, *, discont: bool = False) -> None:
        self._au = au
        self.pts = pts
        self._discont = discont

    def has_flags(self, _flags: int) -> bool:
        return self._discont

    def map(self, _flags: int) -> tuple[bool, types.SimpleNamespace]:
        return True, types.SimpleNamespace(data=self._au)

    def unmap(self, _mapinfo: types.SimpleNamespace) -> None:
        pass


class _FakeSample:
    def __init__(self, buffer: _FakeBuffer) -> None:
        self._buffer = buffer

    def get_buffer(self) -> _FakeBuffer:
        return self._buffer


class _FakePipeline:
    def get_bus(self) -> None:
        return None


class _FakeGst:
    SECOND = 1_000_000_000
    CLOCK_TIME_NONE = -1
    BufferFlags = types.SimpleNamespace(DISCONT=1)
    MapFlags = types.SimpleNamespace(READ=1)
    MessageType = types.SimpleNamespace(ERROR=1, EOS=2)

    def parse_launch(self, _pipeline: str) -> _FakePipeline:
        return _FakePipeline()


class _FakeSink:
    def __init__(self, reader: EncodedAuReader, buffers: Iterable[_FakeBuffer]) -> None:
        self._reader = reader
        self._samples = iter(_FakeSample(buffer) for buffer in buffers)

    def emit(self, _signal: str, _timeout: int) -> _FakeSample | None:
        sample = next(self._samples, None)
        if sample is None:
            self._reader._stop.set()
        return sample


def _reader() -> EncodedAuReader:
    gst = _FakeGst()
    with patch("almond_axol.video.gst_zed._require_gst", return_value=(gst, None)):
        return EncodedAuReader(
            "/tmp/test-encoded-au.sock",
            width=960,
            height=600,
            fps=60,
            name="test-camera",
            pts_perf_offset_s=0.0,
        )


def _pull(reader: EncodedAuReader, buffers: Iterable[_FakeBuffer]) -> None:
    reader._stop.clear()
    reader._sink = _FakeSink(reader, buffers)
    reader._pull_loop()


class EncodedAuReaderDiscontinuityTest(unittest.TestCase):
    def test_startup_discontinuities_are_validated_and_drained(self) -> None:
        reader = _reader()

        _pull(
            reader,
            [
                _FakeBuffer(_IDR, 1_000_000_000),
                _FakeBuffer(_IDR, 1_016_000_000, discont=True),
                _FakeBuffer(_IDR, 1_032_000_000, discont=True),
            ],
        )

        self.assertTrue(reader._first_sample.is_set())
        self.assertEqual(reader._latest_capture_perf, 1.032)
        self.assertEqual(reader.pending, 0)
        self.assertEqual(reader._delivered, 0)
        self.assertFalse(reader._seen_first_au)
        self.assertIsNone(reader._error)

    def test_episode_discontinuity_after_first_idr_remains_fatal(self) -> None:
        reader = _reader()
        reader._flush_requested.set()
        reader._complete_flush()

        _pull(
            reader,
            [
                _FakeBuffer(_IDR, 1_016_000_000, discont=True),
                _FakeBuffer(_IDR, 1_032_000_000),
            ],
        )

        self.assertEqual(reader.pending, 2)
        self.assertEqual(reader._delivered, 2)
        self.assertIsNone(reader._error)

        _pull(
            reader,
            [_FakeBuffer(_IDR, 1_048_000_000, discont=True)],
        )

        self.assertEqual(reader.pending, 0)
        with self.assertRaisesRegex(
            RuntimeError, "encoded-AU discontinuity.*near frame 2"
        ):
            reader.read_next_au(timeout_ms=0)

    def test_predictive_frame_violates_all_intra_contract(self) -> None:
        reader = _reader()

        _pull(reader, [_FakeBuffer(_P_FRAME, 1_000_000_000)])

        self.assertFalse(reader._first_sample.is_set())
        self.assertEqual(reader.pending, 0)
        self.assertIn("configured all-intra", reader._permanent_error or "")

    def test_invalid_startup_timestamp_remains_fatal(self) -> None:
        reader = _reader()

        _pull(reader, [_FakeBuffer(_IDR, _FakeGst.CLOCK_TIME_NONE)])

        self.assertFalse(reader._first_sample.is_set())
        self.assertEqual(reader.pending, 0)
        self.assertIn("has no PTS", reader._permanent_error or "")


if __name__ == "__main__":
    unittest.main()
