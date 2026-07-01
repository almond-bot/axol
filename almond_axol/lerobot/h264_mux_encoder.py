"""Mux pre-encoded H.264 into LeRobot dataset video (no re-encode).

This is the recorder side of the "encode once, in the relay" data path. Where
:mod:`almond_axol.lerobot.nvenc_encoder` receives *raw* frames and runs a full
NVENC encode per camera in the recorder subprocess, this encoder receives the
relay's **already-encoded** H.264 access units (one per frame) and only muxes
them into an mp4 — no colorspace convert, no NVENC, no raw frame copy on the
recorder at all.

Why
---
During ``collect-data`` the relay downscales each camera to the dataset
resolution on its VIC and, in the old path, shipped the *raw* NV12 frames to the
recorder over shared memory (~51 MB/s per camera) where a per-camera NVENC
subprocess re-encoded them. That raw copy plus the recorder-side re-encode was
the bulk of the recording CPU (relay's shm memcpy core + the recorder's three
encoder cores), and with the headset stream encoding the *same* cameras at full
resolution it saturated every core. Moving the dataset encode into the relay
(one extra NVENC branch on the GPU, already holding the frame) and shipping the
~1 MB/s compressed stream instead removes both costs: the recorder just parses
and muxes, which is nearly free.

Alignment contract
------------------
LeRobot decodes dataset video **by timestamp** with a razor-thin tolerance
(``tolerance_s`` defaults to ``1e-4`` s), and ``add_frame`` stamps row *i* at
``timestamp = i / fps``. So the mp4 must be perfectly constant-fps: frame *i* at
exactly ``i / fps``, with at least as many frames as dataset rows. Shared memory
does not carry buffer PTS across the ``shmsink``/``shmsrc`` boundary, so this
encoder assigns the PTS itself: the *k*-th access unit fed for a camera is muxed
at ``pts = k / fps`` (``dts`` and ``duration`` to match). The caller
(:func:`~almond_axol.recording.record_proc.run_encoded_capture_loop`) guarantees
exactly one ``feed_frame`` per camera per dataset row (re-feeding the previous
AU on a per-camera stall), so frame-count == row-count by construction.

Because the frames arrive pre-encoded, the first muxed AU of an episode must be
an IDR or the mp4 is undecodable from frame 0; the relay forces a keyframe when
it opens the dataset branch (see :meth:`ZedGstCamera.set_raw_enabled`) and the
:class:`~almond_axol.video.shm_frames.EncodedAuReader` drops any leading
non-IDR AUs, so the first fed AU is always a keyframe.

Image stats
-----------
LeRobot folds a sampled subset of frames into running image-normalization stats.
Here the recorder no longer has raw frames, so — best-effort — a *second* tee
branch decodes the same stream on NVDEC (hardware, ~free CPU) and a sampled
subset of the decoded RGBA frames feeds the same ``RunningQuantileStats`` the
raw path uses. If the decoder is unavailable the encoder still records correctly
and returns ``None`` stats (video normalization can be recomputed offline).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..video.hw_video import hw_h264_available

_logger = logging.getLogger(__name__)

# Target rate (Hz) at which each camera folds a decoded frame into running image
# stats — same rationale as the raw encoder's ``_STATS_SAMPLE_HZ`` (keep the
# GIL-bound quantile update off the per-frame hot path).
_STATS_SAMPLE_HZ = 10

# How long finish() waits for a muxer pipeline to flush EOS (write the moov atom)
# before giving up on that camera's mp4.
_EOS_TIMEOUT_S = 30.0


def hw_mux_encoder_available() -> bool:
    """True when the gst stack needed to mux pre-encoded H.264 is usable.

    Only the parser + muxer are strictly required (the encode happens in the
    relay); those ship with the same GStreamer install the NVENC path needs, so
    gate on the same hardware-H.264 probe.
    """
    return hw_h264_available()


def _decoder_element() -> str | None:
    """Pick a hardware H.264 decoder for the (best-effort) stats branch.

    Prefer NVDEC (``nvv4l2decoder``); it decodes on a dedicated block so the
    stats branch costs the recorder's cores almost nothing. Never fall back to a
    software decoder here: decoding every frame in libav would burn the very CPU
    this whole path exists to save, so if NVDEC is absent we simply record
    without inline stats.
    """
    from ..video.gst_zed import _element_available

    if _element_available("nvv4l2decoder"):
        return "nvv4l2decoder"
    return None


class _CameraH264Muxer:
    """One camera's ``appsrc -> h264parse -> mp4mux -> filesink`` pipeline.

    ``feed`` pushes one pre-encoded access unit and assigns it the next
    constant-fps PTS. A second (leaky, best-effort) tee branch decodes the stream
    on NVDEC so a sampled subset of frames can feed running image stats. All work
    is on gst's own threads / hardware blocks; ``feed`` only wraps the bytes and
    pushes, so the recorder does no per-frame encode or copy.
    """

    def __init__(self, video_path: Path, fps: int, want_stats: bool = True) -> None:
        from ..video.gst_zed import _require_gst

        self._gst, _ = _require_gst()
        self.video_path = video_path
        self._fps = fps
        self._dur = self._gst.SECOND // fps
        self._count = 0
        self._dropped = 0
        self._error: str | None = None
        self._last_au: bytes | None = None

        # Running image stats from the decoded sample (mirrors the raw encoder).
        self._stats: Any = None
        self._downsample: Any = None
        self._stats_samples = 0
        self._stats_stride = max(1, round(fps / _STATS_SAMPLE_HZ))
        if want_stats:
            try:
                from lerobot.datasets.compute_stats import (
                    RunningQuantileStats,
                    auto_downsample_height_width,
                )

                self._stats = RunningQuantileStats()
                self._downsample = auto_downsample_height_width
            except Exception as exc:  # noqa: BLE001 - no lerobot -> no stats
                _logger.warning(
                    "video stats unavailable for %s, recording without them: %s",
                    video_path.name,
                    exc,
                )

        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._pipeline, self._src, self._stats_sink = self._build(want_stats)
        self._pipeline.set_state(self._gst.State.PLAYING)
        _logger.info(
            "H264 mux pipeline started: %s @ %dfps (stats=%s)",
            video_path.name,
            fps,
            self._stats_sink is not None,
        )

    def _build(self, want_stats: bool) -> tuple[Any, Any, Any]:
        """Build the muxer pipeline; add the decode/stats tee branch if possible."""
        Gst = self._gst
        decoder = (
            _decoder_element() if (want_stats and self._stats is not None) else None
        )
        mux = f"mp4mux ! filesink location={self.video_path} sync=false"
        if decoder is not None:
            # tee: one non-leaky branch muxes every AU (integrity), one leaky
            # branch decodes for sampled stats (drops are fine — stats only need a
            # sample, and a stalled stats pull must never back-pressure the mux).
            desc = (
                "appsrc name=src is-live=false format=time do-timestamp=false "
                "! tee name=t "
                "t. ! queue max-size-buffers=8 ! h264parse ! mp4mux "
                f"! filesink location={self.video_path} sync=false "
                "t. ! queue leaky=downstream max-size-buffers=4 ! h264parse "
                f"! {decoder} ! nvvidconv ! video/x-raw,format=RGBA "
                "! appsink name=stats emit-signals=false max-buffers=2 drop=true "
                "sync=false wait-on-eos=false"
            )
        else:
            desc = (
                "appsrc name=src is-live=false format=time do-timestamp=false "
                f"! h264parse ! {mux}"
            )
        try:
            pipeline = Gst.parse_launch(desc)
        except Exception:  # noqa: BLE001 - fall back to a mux-only pipeline
            self._stats = None
            pipeline = Gst.parse_launch(
                "appsrc name=src is-live=false format=time do-timestamp=false "
                f"! h264parse ! {mux}"
            )
        src = pipeline.get_by_name("src")
        src.set_property(
            "caps",
            Gst.Caps.from_string("video/x-h264,stream-format=byte-stream,alignment=au"),
        )
        # Bound appsrc so a wedged pipeline surfaces as back-pressure, not
        # unbounded memory; block so we never silently drop a (dependency-bearing)
        # encoded frame.
        src.set_property("max-bytes", 8 * 1024 * 1024)
        src.set_property("block", True)
        stats_sink = pipeline.get_by_name("stats")
        return pipeline, src, stats_sink

    @property
    def dropped(self) -> int:
        return self._dropped

    def feed(self, au: bytes) -> None:
        """Mux one access unit at the next constant-fps PTS; sample it for stats."""
        Gst = self._gst
        buf = Gst.Buffer.new_wrapped(au)
        pts = self._count * self._dur
        buf.pts = pts
        buf.dts = pts
        buf.duration = self._dur
        ret = self._src.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 10 == 0:
                _logger.warning(
                    "H264 muxer push returned %s for %s (dropped %d)",
                    ret,
                    self.video_path.name,
                    self._dropped,
                )
        self._last_au = au
        if self._stats_sink is not None and self._count % self._stats_stride == 0:
            self._sample_stats()
        self._count += 1

    def feed_repeat(self) -> None:
        """Re-mux the previous AU (a duplicate frame) to keep counts aligned.

        The frame-driven capture loop calls this when a camera has no fresh AU
        for a row, so every camera stays at the same frame index as the dataset
        rows (the encoded analog of the raw path's "reuse last frame"). A repeated
        AU decodes to the same image, so the mp4 stays valid.
        """
        if self._last_au is not None:
            self.feed(self._last_au)

    def _sample_stats(self) -> None:
        """Pull the latest decoded frame (if any) and fold it into image stats."""
        try:
            import numpy as np

            sample = self._stats_sink.emit("try-pull-sample", 0)
            if sample is None:
                return
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w = caps.get_value("width")
            h = caps.get_value("height")
            ok, mapinfo = buf.map(self._gst.MapFlags.READ)
            if not ok:
                return
            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                if arr.size < w * h * 4:
                    return
                rgb = arr[: w * h * 4].reshape(h, w, 4)[:, :, :3]
                ds = self._downsample(np.ascontiguousarray(rgb).transpose(2, 0, 1))
                self._stats.update(ds.transpose(1, 2, 0).reshape(-1, ds.shape[0]))
                self._stats_samples += 1
            finally:
                buf.unmap(mapinfo)
        except Exception as exc:  # noqa: BLE001 - never kill mux over stats
            _logger.debug("stats sample failed for %s: %s", self.video_path.name, exc)

    def finish(self) -> tuple[Path, dict | None]:
        """EOS the pipeline (flush the moov), then return the path + stats."""
        stats = self._teardown(finalize=True)
        return self.video_path, stats

    def cancel(self) -> None:
        self._teardown(finalize=False)

    def _teardown(self, finalize: bool) -> dict | None:
        Gst = self._gst
        if self._pipeline is None:
            return None
        try:
            if finalize:
                self._src.emit("end-of-stream")
                bus = self._pipeline.get_bus()
                msg = bus.timed_pop_filtered(
                    int(_EOS_TIMEOUT_S * Gst.SECOND),
                    Gst.MessageType.EOS | Gst.MessageType.ERROR,
                )
                if msg is None:
                    _logger.error(
                        "H264 muxer for %s did not EOS in %.0fs",
                        self.video_path.name,
                        _EOS_TIMEOUT_S,
                    )
                elif msg.type == Gst.MessageType.ERROR:
                    err, _ = msg.parse_error()
                    self._error = str(err)
                    _logger.error(
                        "H264 muxer for %s errored: %s", self.video_path.name, err
                    )
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        if not finalize:
            return None
        return (
            self._stats.get_statistics()
            if self._stats is not None and self._stats_samples >= 2
            else None
        )


class H264MuxStreamingEncoder:
    """LeRobot ``StreamingVideoEncoder``-compatible muxer for pre-encoded H.264.

    Drop-in for :class:`~almond_axol.lerobot.nvenc_encoder.NvencStreamingEncoder`
    on the encoded transport: implements ``start_episode`` / ``feed_frame`` /
    ``finish_episode`` / ``cancel_episode`` / ``close``. ``feed_frame`` takes a
    pre-encoded access unit (``bytes``) rather than a raw frame array.
    """

    # Inline stats decode the stream on NVDEC for image-normalization stats, but
    # nvv4l2decoder does not reliably propagate EOS on Jetson (it stalls the mp4
    # finalize), so it is off by default: the dataset records correctly with
    # ``None`` video stats, which LeRobot accepts and which can be recomputed
    # offline. Flip to True only once the decoder-EOS drain is sorted.
    def __init__(self, fps: int, want_stats: bool = False) -> None:
        self.fps = fps
        self._want_stats = want_stats
        self._cams: dict[str, _CameraH264Muxer] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        if self._episode_active:
            self.cancel_episode()
        temp_dir = Path(temp_dir)
        self._cams = {}
        for video_key in video_keys:
            ep_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = ep_dir / f"{video_key.replace('/', '_')}_streaming.mp4"
            self._cams[video_key] = _CameraH264Muxer(
                video_path, self.fps, self._want_stats
            )
        self._episode_active = True

    def feed_frame(self, video_key: str, au: bytes) -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self._cams[video_key].feed(au)

    def feed_repeat(self, video_key: str) -> None:
        """Re-mux ``video_key``'s previous AU (per-camera stall; keep counts aligned)."""
        if self._episode_active and video_key in self._cams:
            self._cams[video_key].feed_repeat()

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        results: dict[str, tuple[Path, dict | None]] = {}
        dropped: dict[str, int] = {}
        for video_key, cam in self._cams.items():
            results[video_key] = cam.finish()
            if cam.dropped:
                dropped[video_key] = cam.dropped
        if dropped:
            _logger.warning(
                "episode dropped muxer frames (%s) — recorded video may be "
                "misaligned with actions; consider re-recording.",
                ", ".join(f"{k}={v}" for k, v in dropped.items()),
            )
        self._cams = {}
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:
        if not self._episode_active:
            return
        for cam in self._cams.values():
            cam.cancel()
            if cam.video_path.parent.exists():
                shutil.rmtree(str(cam.video_path.parent), ignore_errors=True)
        self._cams = {}
        self._episode_active = False

    def close(self) -> None:
        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True
