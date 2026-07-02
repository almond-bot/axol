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
AU on a per-camera stall), so frame-count == row-count by construction. The mp4
this muxer writes carries small per-frame PTS rounding (its timescale cannot
represent ``1/fps`` exactly); the concat step re-stamps every frame onto an
exact constant-fps grid so LeRobot's razor-thin per-row timestamp lookup holds
(see :func:`~almond_axol.recording.record_proc._concatenate_video_files_rebased`).

Because the frames arrive pre-encoded, the first muxed AU of an episode must be
an IDR or the mp4 is undecodable from frame 0; the relay forces a keyframe when
it opens the dataset branch (see :meth:`ZedGstCamera.set_raw_enabled`) and the
:class:`~almond_axol.video.shm_frames.EncodedAuReader` drops any leading
non-IDR AUs, so the first fed AU is always a keyframe.

Image stats
-----------
LeRobot folds a sampled subset of frames into running image-normalization stats.
Here the recorder no longer has raw frames, so — once the episode's mp4 is
finalized — :meth:`_CameraH264Muxer.finish` decodes a sampled subset of that
*already-written* file and feeds the same ``RunningQuantileStats`` the raw path
uses. Decoding a finished file (rather than a live tee branch during muxing)
sidesteps the Jetson ``nvv4l2decoder`` EOS-drain that used to stall the mp4
finalize, and it runs between episodes (recording paused) so it never competes
with the live capture loop. If decoding or lerobot is unavailable the encoder
still records correctly and returns ``None`` stats (recomputable offline).
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


class _CameraH264Muxer:
    """One camera's ``appsrc -> h264parse -> mp4mux -> filesink`` pipeline.

    ``feed`` pushes one pre-encoded access unit and assigns it the next
    constant-fps PTS; all work is on gst's own threads / hardware blocks, so the
    recorder does no per-frame encode or copy. Image-normalization stats are
    computed after the fact in :meth:`finish` by decoding a sampled subset of the
    finalized mp4 (see the module "Image stats" note).
    """

    def __init__(self, video_path: Path, fps: int, want_stats: bool = True) -> None:
        from ..video.gst_zed import _require_gst

        self._gst, _ = _require_gst()
        self.video_path = video_path
        self._fps = fps
        self._dur = self._gst.SECOND // fps
        # mp4 media timescale: a multiple of fps so each sample's duration
        # (SECOND // fps ns) converts to an *exact* integer tick and cumulative
        # PTS lands precisely on k / fps. mp4mux's default (10000 for 60 fps)
        # can't represent 1/fps, leaving ~0.05 ms of per-frame rounding that,
        # compounded across the concat, breaks LeRobot's 1e-4 s per-row lookup.
        self._mux_timescale = fps * 1000
        self._count = 0
        self._error: str | None = None
        self._last_au: bytes | None = None

        # Image stats are computed post-finalize (see finish()); sample every
        # ``_stats_stride``-th decoded frame to hit ~_STATS_SAMPLE_HZ.
        self._want_stats = want_stats
        self._stats_stride = max(1, round(fps / _STATS_SAMPLE_HZ))

        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._pipeline, self._src = self._build()
        self._pipeline.set_state(self._gst.State.PLAYING)
        _logger.info(
            "H264 mux pipeline started: %s @ %dfps (stats=%s)",
            video_path.name,
            fps,
            self._want_stats,
        )

    def _build(self) -> tuple[Any, Any]:
        """Build the ``appsrc -> h264parse -> mp4mux -> filesink`` pipeline."""
        Gst = self._gst
        mux = (
            f"mp4mux trak-timescale={self._mux_timescale} "
            f"! filesink location={self.video_path} sync=false"
        )
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
        return pipeline, src

    def feed(self, au: bytes) -> None:
        """Mux one access unit at the next constant-fps PTS.

        Raises on a rejected push: the appsrc is blocking, so a non-OK flow
        return means the pipeline is flushing or errored — the mp4 can no
        longer contain one sample per fed AU, and continuing would silently
        desync the video from the dataset rows. Aborting the episode (the
        capture loop surfaces the error) is the only honest outcome.
        """
        Gst = self._gst
        buf = Gst.Buffer.new_wrapped(au)
        pts = self._count * self._dur
        buf.pts = pts
        buf.dts = pts
        buf.duration = self._dur
        ret = self._src.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            raise RuntimeError(
                f"H264 muxer push returned {ret} for {self.video_path.name} at "
                f"frame {self._count} — pipeline is flushing/errored, aborting "
                "the episode"
            )
        self._last_au = au
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

    def _compute_stats_from_file(self) -> dict | None:
        """Decode a sampled subset of the finalized mp4 for image-norm stats.

        Runs once per episode after the file is written (recording paused between
        episodes), so it never competes with the live capture loop. Folds every
        ``_stats_stride``-th decoded frame into the same ``RunningQuantileStats``
        the raw encoder uses, matching that path's downsample + (H*W, C) layout so
        LeRobot consumes the result identically. Best-effort: any failure (no
        lerobot, no PyAV, an unreadable file) logs and returns ``None`` rather
        than aborting the episode save.
        """
        if not self._want_stats or not self.video_path.exists():
            return None
        try:
            import av
            import numpy as np
            from lerobot.datasets.compute_stats import (
                RunningQuantileStats,
                auto_downsample_height_width,
            )
        except Exception as exc:  # noqa: BLE001 - deps missing -> no stats
            _logger.warning(
                "video stats unavailable for %s, recording without them: %s",
                self.video_path.name,
                exc,
            )
            return None

        stats = RunningQuantileStats()
        samples = 0
        # The yuv420p -> rgb24 conversion has no SIMD path in this ffmpeg build,
        # so libswscale logs a WARNING per scaler context; silence ffmpeg below
        # ERROR for the decode so it doesn't flood the recorder console.
        prev_level = av.logging.get_level()
        av.logging.set_level(av.logging.ERROR)
        try:
            with av.open(str(self.video_path)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for i, frame in enumerate(container.decode(stream)):
                    if i % self._stats_stride:
                        continue
                    rgb = frame.to_ndarray(format="rgb24")  # H, W, C
                    ds = auto_downsample_height_width(
                        np.ascontiguousarray(rgb).transpose(2, 0, 1)  # -> C, H, W
                    )
                    stats.update(ds.transpose(1, 2, 0).reshape(-1, ds.shape[0]))
                    samples += 1
        except Exception as exc:  # noqa: BLE001 - never fail the save over stats
            _logger.warning(
                "post-finalize stats decode failed for %s: %s",
                self.video_path.name,
                exc,
            )
            return None
        finally:
            av.logging.set_level(prev_level)
        if samples < 2:
            return None
        return stats.get_statistics()

    def finish(self) -> tuple[Path, dict | None]:
        """EOS the pipeline (flush the moov), then return the path + stats.

        Feeds exactly one muxed frame per fed AU (== one dataset row); the
        constant-fps PTS grid that keeps the mp4 aligned with the rows is
        re-asserted by the concat step (see
        :func:`~almond_axol.recording.record_proc._concatenate_video_files_rebased`),
        so no trailing guard frame is added here — an extra frame would shift the
        index-based alignment of every later episode in the concatenated file.
        Image-normalization stats are then decoded from the just-written file.

        Raises if the pipeline errored or never delivered EOS: the mp4 may be
        truncated (moov missing or short), and returning it as a success would
        let LeRobot append/move a corrupt segment into the dataset.
        """
        self._teardown(finalize=True)
        if self._error:
            raise RuntimeError(
                f"H264 muxer for {self.video_path.name} failed to finalize: "
                f"{self._error}"
            )
        return self.video_path, self._compute_stats_from_file()

    def cancel(self) -> None:
        self._teardown(finalize=False)

    def _teardown(self, finalize: bool) -> None:
        Gst = self._gst
        if self._pipeline is None:
            return
        try:
            if finalize:
                self._src.emit("end-of-stream")
                bus = self._pipeline.get_bus()
                msg = bus.timed_pop_filtered(
                    int(_EOS_TIMEOUT_S * Gst.SECOND),
                    Gst.MessageType.EOS | Gst.MessageType.ERROR,
                )
                if msg is None:
                    # A wedged muxer that never flushes the moov atom: the file
                    # on disk is not a complete mp4. Record it so finish()
                    # raises instead of handing the truncated file to LeRobot.
                    self._error = f"no EOS within {_EOS_TIMEOUT_S:.0f}s"
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


class H264MuxStreamingEncoder:
    """LeRobot ``StreamingVideoEncoder``-compatible muxer for pre-encoded H.264.

    Drop-in for :class:`~almond_axol.lerobot.nvenc_encoder.NvencStreamingEncoder`
    on the encoded transport: implements ``start_episode`` / ``feed_frame`` /
    ``finish_episode`` / ``cancel_episode`` / ``close``. ``feed_frame`` takes a
    pre-encoded access unit (``bytes``) rather than a raw frame array.
    """

    # Image-normalization stats are decoded from each finalized episode mp4 (see
    # _CameraH264Muxer._compute_stats_from_file), which happens between episodes
    # with recording paused, so it costs nothing on the live loop and avoids the
    # Jetson nvv4l2decoder EOS-drain that an inline decode branch used to hit. On
    # (best-effort) failure the dataset still records with ``None`` video stats,
    # which LeRobot accepts and which can be recomputed offline.
    def __init__(self, fps: int, want_stats: bool = True) -> None:
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
        """Finalize every camera's mp4; raises if any muxer failed to finalize.

        A raise leaves ``_episode_active`` set; the next ``start_episode``
        cancels (and cleans up) the half-finished episode first.
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        results: dict[str, tuple[Path, dict | None]] = {}
        for video_key, cam in self._cams.items():
            results[video_key] = cam.finish()
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
