from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from almond_axol.lerobot.h264_mux_encoder import _CameraH264Muxer

_IDR = b"\x00\x00\x00\x01\x65\x88"


class _FakeBuffer:
    @classmethod
    def new_wrapped(cls, payload: bytes) -> "_FakeBuffer":
        instance = cls()
        instance.payload = payload
        instance.pts = None
        instance.dts = None
        instance.duration = None
        return instance


class _FakeSource:
    def __init__(self, level_bytes: int = 0) -> None:
        self.pushed = 0
        self.buffers: list[_FakeBuffer] = []
        self.level_bytes = level_bytes

    def emit(self, signal: str, buffer: _FakeBuffer) -> str:
        assert signal == "push-buffer"
        self.pushed += 1
        self.buffers.append(buffer)
        return "ok"

    def get_property(self, name: str) -> int:
        assert name == "current-level-bytes"
        return self.level_bytes


def _bare_muxer(src: _FakeSource) -> _CameraH264Muxer:
    muxer = _CameraH264Muxer.__new__(_CameraH264Muxer)
    muxer._gst = types.SimpleNamespace(
        SECOND=1_000_000_000,
        Buffer=_FakeBuffer,
        FlowReturn=types.SimpleNamespace(OK="ok"),
    )
    muxer.video_path = Path("camera.mp4")
    muxer._dur = muxer._gst.SECOND // 60
    muxer._count = 0
    muxer._stats_stride = 15
    muxer._src = src
    muxer._stats_worker = None
    muxer._peak_queued = 0
    muxer._last_pressure_log = 0.0
    return muxer


class _FakeStatsWorker:
    def __init__(self) -> None:
        self.samples: list[bytes] = []

    def feed(self, au: bytes) -> None:
        self.samples.append(au)


class _FakeFrame:
    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakePacket:
    def __init__(self, index: int) -> None:
        self.dts = index
        self.is_keyframe = True
        self.decode_calls = 0

    def decode(self) -> list[_FakeFrame]:
        self.decode_calls += 1
        return [_FakeFrame()]


class _FakeContainer:
    def __init__(self, packets: list[_FakePacket]) -> None:
        self.packets = packets
        self.streams = types.SimpleNamespace(video=[types.SimpleNamespace()])

    def __enter__(self) -> "_FakeContainer":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def demux(self, _stream: object) -> list[_FakePacket]:
        return self.packets


class H264MuxStatsSamplingTest(unittest.TestCase):
    def test_all_idr_stream_keeps_stats_decode_at_four_hz(self) -> None:
        muxer = _bare_muxer(_FakeSource())
        muxer._stats_worker = _FakeStatsWorker()

        for _ in range(60):
            muxer.feed(_IDR)

        self.assertEqual(muxer._src.pushed, 60)
        self.assertEqual(muxer._count, 60)
        self.assertEqual(len(muxer._stats_worker.samples), 4)

    def test_repeated_idr_still_advances_mux_timeline(self) -> None:
        muxer = _bare_muxer(_FakeSource())

        muxer.feed(_IDR)
        muxer.feed(_IDR)

        self.assertEqual(
            [buffer.pts for buffer in muxer._src.buffers],
            [0, muxer._dur],
        )
        self.assertEqual(
            [buffer.duration for buffer in muxer._src.buffers],
            [muxer._dur, muxer._dur],
        )

    def test_write_pressure_warns_once_per_period_and_tracks_peak(self) -> None:
        src = _FakeSource(level_bytes=0)
        muxer = _bare_muxer(src)
        with self.assertNoLogs("almond_axol.lerobot.h264_mux_encoder", "WARNING"):
            muxer.feed(_IDR)
        self.assertEqual(muxer._peak_queued, 0)

        # The queue climbs past the warning threshold: one line, then silence
        # for the rate-limit period even though every push sees the pressure.
        src.level_bytes = 6 * 1024 * 1024
        with self.assertLogs("almond_axol.lerobot.h264_mux_encoder", "WARNING") as cap:
            for _ in range(10):
                muxer.feed(_IDR)
        self.assertEqual(len(cap.records), 1)
        self.assertIn("storage is not keeping up", cap.output[0])
        self.assertIn("6.0 MiB", cap.output[0])
        self.assertEqual(muxer._peak_queued, 6 * 1024 * 1024)
        self.assertEqual(muxer.queued_bytes(), 6 * 1024 * 1024)

        # Once the period has elapsed the next pressured push warns again.
        muxer._last_pressure_log = 0.0
        with self.assertLogs("almond_axol.lerobot.h264_mux_encoder", "WARNING") as cap:
            muxer.feed(_IDR)
        self.assertEqual(len(cap.records), 1)

    def test_fallback_decodes_only_the_stats_stride(self) -> None:
        packets = [_FakePacket(index) for index in range(60)]
        muxer = _CameraH264Muxer.__new__(_CameraH264Muxer)
        muxer._want_stats = True
        muxer.video_path = Path(__file__)
        muxer._stats_stride = 15

        with patch("av.open", return_value=_FakeContainer(packets)):
            result = muxer._compute_stats_from_file()

        self.assertIsNotNone(result)
        self.assertEqual(sum(packet.decode_calls for packet in packets), 4)


if __name__ == "__main__":
    unittest.main()
