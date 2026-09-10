"""End-to-end finalize check for the mux-only recorder against real GStreamer.

Regression for the "finalized with n-1 packets (n advertised)" save failure:
without a framerate on the appsrc caps, h264parse drops the per-buffer
duration and mp4mux writes the final sample with duration 0, so the track /
edit list ends one frame early and ffmpeg never yields the last frame.

Runs only where PyGObject + h264parse/mp4mux + openh264enc + PyAV are
installed; skipped elsewhere. openh264enc is the fixture encoder on purpose:
like nvv4l2h264enc it writes an SPS without VUI timing, so h264parse has no
in-band source for the frame duration and the bug reproduces; x264enc's SPS
carries timing info and masks it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from almond_axol.lerobot.h264_mux_encoder import _CameraH264Muxer
from almond_axol.video.gst_zed import _element_available

_FPS = 60


def _software_encoded_aus(count: int) -> list[bytes]:
    """``count`` all-IDR Annex-B access units from openh264enc (no VUI timing)."""
    from almond_axol.video.gst_zed import _require_gst

    Gst, _ = _require_gst()
    pipeline = Gst.parse_launch(
        f"videotestsrc num-buffers={count} pattern=ball "
        f"! video/x-raw,width=64,height=64,framerate={_FPS}/1 "
        "! openh264enc gop-size=1 bitrate=64000 "
        "! video/x-h264,stream-format=byte-stream,alignment=au "
        "! appsink name=out sync=false"
    )
    sink = pipeline.get_by_name("out")
    pipeline.set_state(Gst.State.PLAYING)
    aus: list[bytes] = []
    try:
        while True:
            sample = sink.emit("pull-sample")
            if sample is None:
                break
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            assert ok
            try:
                aus.append(bytes(info.data))
            finally:
                buf.unmap(info)
    finally:
        pipeline.set_state(Gst.State.NULL)
    return aus


def _demuxed(path: Path) -> tuple[int, int, int]:
    """(advertised sample count, real demuxed packets, last packet pts ticks)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packets = [
            packet
            for packet in container.demux(stream)
            if packet.pts is not None and packet.dts is not None
        ]
        return int(stream.frames or 0), len(packets), int(packets[-1].pts)


@unittest.skipUnless(
    all(
        _element_available(e) for e in ("appsrc", "h264parse", "mp4mux", "openh264enc")
    ),
    "needs GStreamer with h264parse/mp4mux/openh264enc",
)
class H264MuxFinalizeTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest("needs PyAV")

    def test_every_fed_access_unit_is_a_demuxable_sample(self) -> None:
        for count in (1, 3, 100):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "observation.images.cam_streaming.mp4"
                muxer = _CameraH264Muxer(path, _FPS, want_stats=False)
                for au in _software_encoded_aus(count):
                    muxer.feed(au)
                out_path, stats = muxer.finish()

                self.assertEqual(out_path, path)
                self.assertIsNone(stats)
                advertised, demuxed, last_pts = _demuxed(path)
                self.assertEqual(advertised, count)
                self.assertEqual(demuxed, count)
                # mp4 timescale is fps*1000, so one frame == 1000 ticks and
                # the last frame sits exactly at (count-1)/fps: the final
                # sample carries a full frame duration, not zero.
                self.assertEqual(last_pts, (count - 1) * 1000)
