"""Dataset recording, off the control loop.

``collect-data``'s control loop runs at 120 Hz on the robot's event loop. Writing
the LeRobot dataset — capturing camera frames, assembling rows, ``add_frame``,
NVENC encoding, ``save_episode`` — is heavy per-frame Python work. Running it as
threads *inside* the control process makes it share one GIL with the control
loop, so even with spare CPU cores the loop stutters during recording (the
remaining jitter after the SVGA downscale + stats-off-hot-path fixes).

This module moves all of that into a dedicated **recorder subprocess**
(:class:`DatasetRecorderProcess`) so the control process only writes a tiny
joint/action snapshot per tick (via
:class:`~almond_axol.video.shm_frames.SnapshotWriter`) and sends episode-lifecycle
commands. The recorder pulls each camera's raw frames from the relay — via a gst
``shmsrc`` consumer (the relay's ``shmsink`` exports frames in C, so the relay
does no Python per frame and its WebRTC send keeps its GIL), or, when gst's shm
plugin is absent, via a :class:`RawFrameReader` over a shared-memory block the
relay's Python pull loop fills. Either way the recorder owns the
``LeRobotDataset`` end to end.

When the video relay is unavailable (no gst stack — a degraded, non-Jetson path),
:class:`InProcessRecorder` keeps the old behavior: dataset + capture thread in
the control process. Both expose the same interface, so the control loop is
single-path; only the construction differs.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import multiprocessing.connection
import os
import platform
import shutil
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

_logger = logging.getLogger("almond_axol.recording.record_proc")

# Per-stream encoder thread count (three cameras each spawn their own encoder
# thread); a small inner libx264 pool leaves cores for the control loop.
_ENCODER_THREADS = 2
# Keyframe interval for the software encoder. LeRobot hardcodes GOP 2 (a keyframe
# almost every frame), which tanks Tegra encode throughput; 30 (0.5 s at 60 fps)
# is plenty fine-grained for timestamp-tolerant dataset decode.
_ENCODER_GOP = 30

# How long the recorder subprocess may take to open cameras' shm + the dataset.
_READY_TIMEOUT_S = 60.0
# How long a save_episode (encoder flush + parquet write + post-episode stats)
# may take.
_SAVE_TIMEOUT_S = 180.0
# How long a lightweight episode command (pause/resume/frame_count) may take —
# these only flip an event / read a counter in the recorder's command thread.
_CMD_TIMEOUT_S = 10.0

# --- Encoded (relay-side H.264) capture-loop tuning ---
# How long the first row waits for each camera's first access unit (relay valve
# open + forced IDR + shmsrc spin-up). If a camera produces nothing in this
# window its dataset branch never came up, so the episode is aborted.
_ENCODED_START_TIMEOUT_S = 15.0
# Row-wide budget shared by all cameras: how long one row may wait for fresh AUs
# before duplicating a stalled camera's previous one. Deliberately generous —
# the blocking read paces the loop to the slowest camera and stays
# frame-accurate, so a brief hiccup is absorbed by *waiting*, not by inserting a
# duplicate; only a genuine multi-frame stall duplicates (and a duplicate AU
# decodes to the same image, keeping the mp4 valid and frame-count == row-count).
# Shared (not per-camera) so serial reads can't compound the wait, and a camera
# whose AU is already queued still advances after the deadline (see read_au).
_ENCODED_ROW_TIMEOUT_S = 1.0
# How often the blocking AU read wakes to re-check stop_event.
_ENCODED_POLL_MS = 100


# ---------------------------------------------------------------------------
# Encoder selection (applied in whichever process owns the dataset)
# ---------------------------------------------------------------------------


def default_vcodec() -> str:
    """Pick a video codec that can actually open on this machine.

    LeRobot's "auto" prefers ``h264_nvenc``, but on Jetson/Tegra (aarch64) there
    is no desktop ``libnvidia-encode`` to back it, so it fails to open mid
    episode. Default to CPU "h264" on aarch64 and let "auto" pick the HW encoder
    elsewhere.
    """
    return "h264" if platform.machine() == "aarch64" else "auto"


def _tune_software_encoder() -> None:
    """Patch LeRobot's libx264 options: ``preset=veryfast g=30`` (idempotent)."""
    import lerobot.datasets.video_utils as _vu

    if getattr(_vu, "_axol_encoder_tuned", False):
        return
    _orig_get_codec_options = _vu._get_codec_options

    def _tuned(
        vcodec: str, g: int | None = 2, crf: int | None = 30, preset=None
    ) -> dict:
        options = _orig_get_codec_options(vcodec, g=g, crf=crf, preset=preset)
        if vcodec in ("h264", "hevc"):
            options["preset"] = "veryfast"
            options["g"] = str(_ENCODER_GOP)
        return options

    _vu._get_codec_options = _tuned
    _vu._axol_encoder_tuned = True
    _logger.info(
        "tuned software video encoder: preset=veryfast g=%d threads=%d",
        _ENCODER_GOP,
        _ENCODER_THREADS,
    )


# How many leading packets of the first input the fps probe reads. The chunk
# file is read in full by the remux pass anyway; reading it a second time just
# to infer fps doubled the concat's I/O, which dominates save_episode once the
# accumulating chunk grows (it is rewritten on *every* save).
_CONCAT_PROBE_PACKETS = 121


class _BFramesDetected(Exception):
    """Raised mid-remux when an input turns out to have B-frames."""


def _concat_probe_constant_fps(input_video_paths: list) -> "Fraction | None":
    """Return the frame rate to re-stamp onto, or ``None`` for shift-rebase.

    Cheap checks only: every input must be a single video stream (no
    audio/subtitle — stream *metadata*, no demux), and the rate is inferred
    from the first input's leading packets' median PTS delta (our muxer's mp4
    timescale rounds ``1/fps``, so no single delta is exact, but the median
    is), falling back to the stream's advertised ``average_rate``. The
    no-B-frames requirement (every packet ``pts == dts``, so demux order is
    display order) is *not* pre-scanned here — the remux pass verifies it on
    every packet as it copies and falls back if violated — so the whole
    chunk is read once per concat, not twice.
    """
    import statistics

    import av

    fps: Fraction | None = None
    for input_path in input_video_paths:
        with av.open(str(input_path), mode="r") as src:
            videos = [s for s in src.streams if s.type == "video"]
            if len(videos) != 1 or any(
                s.type in ("audio", "subtitle") for s in src.streams
            ):
                return None
            if fps is not None:
                continue
            vstream = videos[0]
            secs: list[float] = []
            tb = vstream.time_base
            for packet in src.demux(vstream):
                if packet.pts is None or packet.dts is None:
                    continue  # demux flushing packet
                if packet.pts != packet.dts:
                    return None  # B-frames present -> can't reindex by demux order
                secs.append(float(packet.pts * tb))
                if len(secs) >= _CONCAT_PROBE_PACKETS:
                    break
            if len(secs) >= 2:
                secs.sort()
                med = statistics.median(
                    secs[i + 1] - secs[i] for i in range(len(secs) - 1)
                )
                if med <= 0:
                    return None
                fps = Fraction(round(1.0 / med), 1)
            elif vstream.average_rate:
                fps = Fraction(vstream.average_rate)
            else:
                return None
    return fps


def _concat_constant_fps(
    input_video_paths: list, output_video_path: "Path", fps: "Fraction"
) -> None:
    """Concatenate segments, stamping every frame onto an exact ``k / fps`` grid.

    Ignores the source PTS entirely and assigns frame *k* (demux order, global
    across segments) ``pts = dts = k`` in a timebase whose unit is ``1 / fps`` —
    concretely ``time_base = 1 / (fps * 1000)`` with a per-frame step of 1000, so
    every frame lands on ``k / fps`` with zero rounding error. Bitstream packets
    are copied verbatim (no re-encode); only container timing is rewritten.

    Each input's demuxed packet count is checked against its container's sample
    count (moov ``stsz``): a gst-muxed segment can demux short of the samples it
    advertises (the trailing sample of an mp4mux file is sometimes not
    surfaced), which would leave the chunk one frame short of its dataset rows
    and silently shift every later episode's timestamp lookup. A short input is
    padded back to its advertised count by re-muxing its last packet (a
    duplicated frame decodes to the same image, so alignment and decodability
    hold) with a loud log.

    The temp file lives next to the output (rename, never a cross-fs copy) and
    is muxed without ``faststart``: this runs once per camera on *every*
    ``save_episode``, rewriting the whole accumulating chunk file, and the
    faststart pass reads + rewrites that file *again* just to front-load the
    moov atom — pure overhead for a file that is appended to on the next save
    and only ever read locally (seekable either way).
    """
    import tempfile

    import av

    step = 1000
    time_base = Fraction(fps.denominator, fps.numerator * step)

    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=output_video_path.parent
    ) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name
    try:
        with av.open(tmp_output_video_path, mode="w") as dst:
            out_stream = None
            frame_idx = 0
            for input_path in input_video_paths:
                with av.open(str(input_path), mode="r") as src:
                    in_stream = next(s for s in src.streams if s.type == "video")
                    if out_stream is None:
                        out_stream = dst.add_stream_from_template(
                            template=in_stream, opaque=True
                        )
                        out_stream.time_base = time_base
                    expected = in_stream.frames or 0
                    demuxed = 0
                    last_payload: bytes | None = None
                    for packet in src.demux(in_stream):
                        if packet.dts is None:  # demux flushing packet
                            continue
                        if packet.pts != packet.dts:
                            # B-frames: demux order is not display order, so
                            # index-based re-stamping would scramble frames.
                            # The probe only samples leading packets; this is
                            # the full-stream check.
                            raise _BFramesDetected(str(input_path))
                        last_payload = bytes(packet)
                        packet.pts = frame_idx * step
                        packet.dts = frame_idx * step
                        packet.duration = step
                        packet.stream = out_stream
                        dst.mux(packet)
                        frame_idx += 1
                        demuxed += 1
                    if expected > demuxed and last_payload is not None:
                        _logger.error(
                            "concat: %s demuxed %d of %d advertised samples; "
                            "padding %d duplicate trailing frame(s) to keep "
                            "frame-count == row-count",
                            Path(str(input_path)).name,
                            demuxed,
                            expected,
                            expected - demuxed,
                        )
                        for _ in range(expected - demuxed):
                            pad = av.Packet(last_payload)
                            pad.pts = frame_idx * step
                            pad.dts = frame_idx * step
                            pad.duration = step
                            pad.stream = out_stream
                            dst.mux(pad)
                            frame_idx += 1
        shutil.move(tmp_output_video_path, str(output_video_path))
    except Exception:
        Path(tmp_output_video_path).unlink(missing_ok=True)
        raise


def _concat_shift_rebased(input_video_paths: list, output_video_path: "Path") -> None:
    """Stream-copy concat that shifts each segment past the previous one.

    Fallback for inputs that :func:`_concat_constant_fps` can't re-stamp (B-frames
    or extra streams): open each input independently and shift every packet so each
    segment starts exactly where the previous one ended (per output stream, in the
    output stream's time_base). Within a segment timestamps are already monotonic,
    so the concatenated stream is monotonic by construction, and PTS-vs-DTS spacing
    is preserved so B-frame reordering survives.
    """
    import tempfile

    import av

    # Same-dir temp + no faststart, for the same reasons as _concat_constant_fps.
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False, dir=output_video_path.parent
    ) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    try:
        with av.open(tmp_output_video_path, mode="w") as dst:
            out_streams: dict[int, object] = {}  # input stream index -> output stream
            offsets: dict[object, int] = {}  # output stream -> next start dts (out tb)

            for file_idx, input_path in enumerate(input_video_paths):
                with av.open(str(input_path), mode="r") as src:
                    seg_start: dict[object, int] = {}
                    seg_end: dict[object, int] = {}
                    for in_stream in src.streams:
                        if in_stream.type not in ("video", "audio", "subtitle"):
                            continue
                        if file_idx == 0:
                            out_stream = dst.add_stream_from_template(
                                template=in_stream, opaque=True
                            )
                            out_stream.time_base = in_stream.time_base
                            out_streams[in_stream.index] = out_stream
                            offsets[out_stream] = 0

                    for packet in src.demux():
                        if packet.dts is None:  # demux flushing packet
                            continue
                        out_stream = out_streams.get(packet.stream.index)
                        if out_stream is None:
                            continue
                        ratio = Fraction(packet.stream.time_base) / Fraction(
                            out_stream.time_base
                        )
                        dts = int(round(packet.dts * ratio))
                        pts = (
                            None
                            if packet.pts is None
                            else int(round(packet.pts * ratio))
                        )
                        dur = int(round((packet.duration or 0) * ratio))

                        if out_stream not in seg_start:
                            seg_start[out_stream] = dts
                        shift = offsets[out_stream] - seg_start[out_stream]

                        packet.dts = dts + shift
                        packet.pts = None if pts is None else pts + shift
                        packet.duration = dur
                        packet.stream = out_stream

                        end = packet.dts + dur
                        if end > seg_end.get(out_stream, end - 1):
                            seg_end[out_stream] = end
                        dst.mux(packet)

                    for out_stream, end in seg_end.items():
                        offsets[out_stream] = end

        shutil.move(tmp_output_video_path, str(output_video_path))
    except Exception:
        Path(tmp_output_video_path).unlink(missing_ok=True)
        raise


def _concatenate_video_files_rebased(
    input_video_paths: list,
    output_video_path: "Path | str",
    overwrite: bool = True,
    compatibility_check: bool = False,
) -> None:
    """Concatenate per-episode video segments (drop-in for LeRobot's
    ``concatenate_video_files``).

    LeRobot appends each new episode's video to the running per-key chunk file
    (``save_episode`` on episode index >= 1). Its stock implementation feeds both
    segments through PyAV's ``concat`` demuxer and copies packets verbatim, trusting
    the demuxer to offset the later segment past the first. With our mp4 segments
    that offset isn't applied, so at the segment boundary the muxer's DTS jumps
    backwards (e.g. ``734200 >= 367200``) and libav aborts with ``non monotonically
    increasing dts to muxer``, losing the episode.

    Beyond just making the boundary monotonic, the concatenated video must be a
    perfect constant-fps grid: LeRobot indexes dataset rows to video *by timestamp*
    (row *i* -> ``i / fps``) with a razor-thin ``tolerance_s`` (``1e-4`` s), so a
    frame whose PTS is even ~0.15 ms off its ideal ``i / fps`` fails to load. Our
    per-episode mp4 muxer stamps frames in a timescale that can't represent
    ``1 / fps`` exactly (e.g. mp4 timescale 10000 for 60 fps), so ~0.2 ms of PTS
    rounding jitter accumulates and a large fraction of rows would violate the
    tolerance. So for the common case — our recorder's single-stream, B-frame-free
    (IPPP) segments — re-stamp every frame onto an exact ``k / fps`` grid
    (:func:`_concat_constant_fps`); this requires exactly one packet per dataset
    row (guaranteed upstream by the one-AU-per-row capture loop and the non-VCL AU
    filter — no guard/duplicate frames, which would shift the index-based
    alignment of every later episode). Anything we can't safely reindex (B-frames,
    extra streams) falls back to a plain monotonic shift (:func:`_concat_shift_rebased`).
    """
    output_video_path = Path(output_video_path)
    if output_video_path.exists() and not overwrite:
        _logger.warning(
            "Video file already exists: %s. Skipping concatenation.", output_video_path
        )
        return
    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fps = _concat_probe_constant_fps(input_video_paths)
    if fps is not None:
        try:
            _concat_constant_fps(input_video_paths, output_video_path, fps)
        except _BFramesDetected as exc:
            _logger.warning(
                "concat: B-frames found mid-stream in %s; "
                "falling back to shift-rebase — per-row timestamp tolerance "
                "may suffer.",
                exc,
            )
            _concat_shift_rebased(input_video_paths, output_video_path)
    else:
        _logger.warning(
            "concat: constant-fps re-stamp unavailable (B-frames or extra streams); "
            "falling back to shift-rebase — per-row timestamp tolerance may suffer."
        )
        _concat_shift_rebased(input_video_paths, output_video_path)
    _logger.info(
        "concat re-stamp -> %s (%.0f MB) in %.2fs",
        output_video_path.name,
        output_video_path.stat().st_size / 1e6 if output_video_path.exists() else 0,
        time.perf_counter() - t0,
    )

    if not output_video_path.exists():
        raise OSError(
            f"Video concatenation did not work. File not found: {output_video_path}."
        )


def _video_duration_exact(video_path: "Path | str") -> float:
    """Frame-count-exact video duration (drop-in for LeRobot's
    ``get_video_duration_in_s``).

    LeRobot stamps each episode's ``videos/<key>/from_timestamp`` /
    ``to_timestamp`` from the *segment's* container duration, and the reader
    later locates row *i*'s frame at ``from_timestamp + i / fps``. The stock
    implementation trusts ``stream.duration`` — which on a gst ``mp4mux``
    segment reads ``(N-1)/fps`` for N samples (the trailing sample's duration
    is not reflected in the track header). That one-frame shortfall accumulates
    through the chunk file: episode *k* appended to a file inherits a
    ``from_timestamp`` short by ``k/fps``, so its rows silently resolve to
    frames up to *k* ticks stale (no error — the exact-fps grid always has *a*
    frame within tolerance). Deriving the duration from the packets themselves
    (count x per-frame duration, using the advertised sample count when demux
    comes up short — see the concat pad guard) makes ``to - from`` exactly
    ``rows / fps`` for every episode.
    """
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        demuxed = 0
        last = None
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            demuxed += 1
            last = packet
        if last is None or tb is None:
            return 0.0
        n = max(demuxed, stream.frames or 0)
        if last.duration:
            return float(n * last.duration * tb)
        if stream.average_rate:
            return float(n / stream.average_rate)
        return float((stream.duration or 0) * tb)


class _RemuxOnMoveShutil:
    """``shutil`` stand-in for ``dataset_writer`` that re-muxes moved mp4 segments.

    LeRobot writes a fresh per-key video file (episode 0, and whenever a new
    episode would push the current file past ``video_files_size_in_mb``) by a plain
    ``shutil.move`` of the gst-muxed episode segment — the append path, by
    contrast, goes through our re-stamping :func:`_concatenate_video_files_rebased`.
    The gst muxer leaves the segment's *final* frame undecodable (mp4mux writes N
    samples but the decoder only emits N-1), so an episode that lands as its own
    file has its last dataset row fail to load. Routing the move through the same
    single-input av re-mux rewrites a decodable, exact-fps-grid file; every other
    attribute delegates to the real ``shutil`` so the module is otherwise
    unchanged. (Leaving the source behind is fine — the writer removes the temp
    dir right after.)
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def move(self, src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if str(dst).endswith(".mp4"):
            _concatenate_video_files_rebased([src], dst)
            return dst
        return self._real.move(src, dst, *args, **kwargs)


def _patch_video_concat() -> None:
    """Route every dataset video write through the re-stamping av mux.

    Idempotent. Patches ``concatenate_video_files`` where it's *called*
    (``dataset_writer``, which does ``from .video_utils import
    concatenate_video_files``) and at its definition, so the episode-append path
    re-bases + re-stamps. Also swaps ``dataset_writer``'s ``shutil`` for a shim
    that re-muxes the "move segment to a fresh file" path (episode 0 / size
    rollover) through the same mux — otherwise those files keep the gst muxer's
    undecodable final frame and their episode's last row won't load. Also
    replaces the writer's ``get_video_duration_in_s`` with the frame-count-exact
    :func:`_video_duration_exact` so per-episode from/to timestamps span exactly
    ``rows / fps`` (the stock stream-duration read is one frame short on gst
    segments, silently shifting every later episode's frame lookups).
    """
    import lerobot.datasets.dataset_writer as _dw
    import lerobot.datasets.video_utils as _vu

    if getattr(_vu, "_axol_concat_rebased", False):
        return
    _vu.concatenate_video_files = _concatenate_video_files_rebased
    _dw.concatenate_video_files = _concatenate_video_files_rebased
    _dw.shutil = _RemuxOnMoveShutil(_dw.shutil)
    _dw.get_video_duration_in_s = _video_duration_exact
    _vu._axol_concat_rebased = True
    _logger.info(
        "patched LeRobot video writes: re-stamp concat + re-mux moved segments "
        "+ frame-exact durations"
    )


def _patch_embed_images_skip() -> None:
    """Skip the per-save ``embed_images`` map when no image columns exist.

    ``_save_episode_data`` runs every episode row through
    ``embed_images``'s ``Dataset.map(embed_table_storage)`` before the parquet
    write. With video features (our only camera dtype) excluded from the hf
    schema there is nothing to embed, yet the map still copies every row
    through Python at ~1.6k rows/s — ~1.5 s of pure overhead per save for a
    one-minute episode. Delegate to the real ``embed_images`` only when the
    dataset actually has an ``Image`` column. Idempotent.
    """
    import datasets as hf_datasets
    import lerobot.datasets.dataset_writer as _dw

    if getattr(_dw, "_axol_embed_images_skip", False):
        return
    _orig = _dw.embed_images

    def _embed_if_needed(dataset):  # type: ignore[no-untyped-def]
        if any(isinstance(f, hf_datasets.Image) for f in dataset.features.values()):
            return _orig(dataset)
        return dataset

    _dw.embed_images = _embed_if_needed
    _dw._axol_embed_images_skip = True
    _logger.info("skipping no-op embed_images map on episode save")


def _patch_frame_validation() -> None:
    """Let ``add_frame`` accept packed NV12 video frames from the relay.

    LeRobot's ``validate_frame`` requires each video feature's value to be a 3-D
    ``(H, W, C)`` / ``(C, H, W)`` array (or PIL image). The NVENC encoder is fed
    the relay's packed **NV12** buffers — a 2-D ``(H*3//2, W)`` uint8 array — which
    would otherwise be rejected before ``feed_frame`` ever sees them. Relax only
    the image/video shape check, and only for an array whose shape is exactly the
    NV12 layout of the declared ``(H, W, C)`` feature (so a genuinely malformed
    frame is still caught); everything else (feature presence, state/action dtype
    and shape, the RGB fallback path) is unchanged. Idempotent.
    """
    import lerobot.datasets.feature_utils as _fu
    import numpy as np

    if getattr(_fu, "_axol_nv12_validation", False):
        return
    _orig = _fu.validate_feature_image_or_video

    def _lenient(name, expected_shape, value):  # type: ignore[no-untyped-def]
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and value.dtype == np.uint8
            and len(expected_shape) == 3
            and value.shape == (expected_shape[0] * 3 // 2, expected_shape[1])
        ):
            return ""  # packed NV12 — shape is correct for the feature by construction
        return _orig(name, expected_shape, value)

    _fu.validate_feature_image_or_video = _lenient
    _fu._axol_nv12_validation = True
    _logger.info("relaxed LeRobot frame validation to accept packed NV12 video frames")


def _patch_frame_validation_encoded() -> None:
    """Let ``add_frame`` accept a pre-encoded H.264 access unit as a video value.

    On the encoded (``gstshm-h264``) transport the recorder never holds a raw
    frame — the relay already encoded it and the recorder only muxes the bytes.
    The capture loop injects the AU (``bytes``) as the video feature's value so
    LeRobot's ``feed_frame`` receives it verbatim, but ``validate_frame`` would
    reject a non-array video value first. Accept ``bytes``/``bytearray`` for
    image/video features (everything else — presence, state/action dtype+shape,
    the array/PIL path — is unchanged). Idempotent.
    """
    import lerobot.datasets.feature_utils as _fu

    if getattr(_fu, "_axol_au_validation", False):
        return
    _orig = _fu.validate_feature_image_or_video

    def _lenient_bytes(name, expected_shape, value):  # type: ignore[no-untyped-def]
        if isinstance(value, (bytes, bytearray)):
            return ""  # a pre-encoded access unit — muxed as-is, not shape-checked
        return _orig(name, expected_shape, value)

    _fu.validate_feature_image_or_video = _lenient_bytes
    _fu._axol_au_validation = True
    _logger.info(
        "relaxed LeRobot frame validation to accept pre-encoded H.264 access units"
    )


def install_dataset_encoder() -> bool:
    """Prefer the Jetson NVENC encoder for dataset video; else tune libx264.

    Module-level monkeypatch of ``LeRobotDataset._build_streaming_encoder`` — must
    be applied in whatever process creates the dataset (the recorder subprocess,
    or the control process for the in-process fallback). Returns True when NVENC
    is in use.

    The NVENC encoder runs in **VBR with a peak cap** (see
    ``hw_video.dataset_vbr_bitrate``): NVENC targets the average bitrate, so every
    camera's dataset video stays bounded and uniformly sized — a noisy sensor is
    compressed down to the target instead of ballooning the dataset and fragmenting
    it into many video files.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from ..lerobot.nvenc_encoder import (
        NvencStreamingEncoder,
        hw_dataset_encoder_available,
    )

    # Episode-append concat re-bases timestamps regardless of which encoder writes
    # the segments, so patch it on every path (NVENC and the libx264 fallback).
    _patch_video_concat()
    _patch_embed_images_skip()

    if getattr(LeRobotDataset, "_axol_nvenc_installed", False):
        return True

    if not hw_dataset_encoder_available():
        _tune_software_encoder()
        return False

    # The relay ships NV12 to the recorder, which the NVENC encoder feeds straight
    # through; teach LeRobot's frame validation to accept that packed layout.
    _patch_frame_validation()

    def _build_nvenc(fps, vcodec, encoder_queue_maxsize, encoder_threads):
        # Ignore LeRobot's shallow default (30); NvencStreamingEncoder uses its own
        # deeper feed queue to ride out gst pipeline spin-up. See _FEED_QUEUE_MAXSIZE.
        return NvencStreamingEncoder(fps=fps)

    LeRobotDataset._build_streaming_encoder = staticmethod(_build_nvenc)
    LeRobotDataset._axol_nvenc_installed = True
    _logger.info("using Jetson NVENC hardware video encoder for dataset recording")
    return True


def install_encoded_dataset_encoder() -> bool:
    """Install the mux-only encoder for the relay-encoded (gstshm-h264) transport.

    On this path the relay already H.264-encoded each dataset frame, so the
    recorder must *not* re-encode: swap ``_build_streaming_encoder`` for one that
    returns :class:`~almond_axol.lerobot.h264_mux_encoder.H264MuxStreamingEncoder`
    (``appsrc -> h264parse -> mp4mux``, constant-fps PTS), teach frame validation
    to accept the AU ``bytes``, and keep the timestamp-rebasing concat (episode
    append still stitches per-key mp4 segments). Raises if the gst mux stack is
    missing — the relay wouldn't have chosen this transport without it, so a miss
    here is a real misconfiguration rather than something to silently downgrade.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from ..lerobot.h264_mux_encoder import (
        H264MuxStreamingEncoder,
        hw_mux_encoder_available,
    )

    _patch_video_concat()
    _patch_embed_images_skip()

    if getattr(LeRobotDataset, "_axol_h264mux_installed", False):
        return True

    if not hw_mux_encoder_available():
        raise RuntimeError(
            "encoded (gstshm-h264) transport selected but the GStreamer H.264 "
            "mux stack is unavailable in the recorder"
        )

    _patch_frame_validation_encoded()

    def _build_mux(fps, vcodec, encoder_queue_maxsize, encoder_threads):
        return H264MuxStreamingEncoder(fps=fps)

    LeRobotDataset._build_streaming_encoder = staticmethod(_build_mux)
    LeRobotDataset._axol_h264mux_installed = True
    _logger.info(
        "using relay-encoded H.264 mux (no recorder re-encode) for dataset video"
    )
    return True


# ---------------------------------------------------------------------------
# Joint/action snapshot (in-process publisher, mirrors the cross-process one)
# ---------------------------------------------------------------------------


class _SnapshotPublisher:
    """Single-slot in-process publisher (the no-relay fallback's snapshot sink).

    The control loop calls :meth:`write` every tick; the capture thread reads the
    latest via :meth:`read_latest`. Returns ``None`` before the first write. The
    method names mirror :class:`~almond_axol.video.shm_frames.SnapshotReader` so the
    capture loop is identical in both paths.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: tuple[dict, dict, float] | None = None

    def write(self, joint_obs: dict, action: dict, ts: float) -> None:
        with self._lock:
            self._latest = (joint_obs, action, ts)

    def read_latest(self) -> tuple[dict, dict, float] | None:
        with self._lock:
            return self._latest


# ---------------------------------------------------------------------------
# Capture loop (shared by both recorders, runs on its own thread)
# ---------------------------------------------------------------------------


def _obs_for_rerun(obs: dict[str, Any], cam_keys: Any) -> dict[str, Any]:
    """Copy ``obs`` with any packed-NV12 camera frames converted to RGB.

    The gstshm path delivers camera frames as 2-D ``(H*3//2, W)`` NV12 (fed
    straight to NVENC), which rerun can't display; convert just those for the
    (opt-in, debug-only) rerun log. Other transports already deliver RGB, so this
    is a no-op there. Only touched when ``rerun_ip`` is set.
    """
    import numpy as np

    out = dict(obs)
    for key in cam_keys:
        val = out.get(key)
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.dtype == np.uint8:
            try:
                import cv2

                h = (val.shape[0] * 2) // 3
                out[key] = cv2.cvtColor(
                    val.reshape(h * 3 // 2, val.shape[1]), cv2.COLOR_YUV2RGB_NV12
                )
            except Exception:  # noqa: BLE001 - rerun is best-effort/debug-only
                pass
    return out


def run_capture_loop(
    *,
    cameras: dict[str, Any],
    read_snapshot: Callable[[], tuple[dict, dict, float] | None],
    dataset: "LeRobotDataset",
    robot_obs_proc: Callable[[Any], Any],
    fps: int,
    task: str,
    rerun_ip: str | None,
    stop_event: threading.Event,
    record_event: "threading.Event | None" = None,
    frame_counter: "dict[str, int] | None" = None,
    on_error: Callable[[str], None] | None = None,
) -> None:
    """Capture dataset rows at ``fps`` Hz until ``stop_event`` is set.

    Each tick sleeps until ``T_n = recording_start + n/fps``, waits for a frame
    with ``capture_perf_ts >= T_n`` from every camera, pulls the latest
    joint+action snapshot, and appends one dataset row. A camera read timeout
    reuses the previous frame for that camera (or skips the tick if none yet).
    Any fatal error is reported via ``on_error`` instead of dying silently.

    ``record_event`` (optional) gates mid-episode capture: while cleared the
    loop idles without appending rows, and on the next set it re-anchors its
    tick clock to "now" so the dataset's index-based timestamps stay
    contiguous across the gap — the saved episode plays straight through it.
    Used by DAgger-style flows that must not record while the robot is frozen
    between the policy and an operator takeover. ``None`` records
    unconditionally (the pre-existing behaviour).

    ``frame_counter`` (optional) is a mutable ``{"n": int}`` incremented after
    every appended row, so the owner can convert instants into dataset time
    (``n / fps``) — e.g. to annotate intervention spans.
    """
    try:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        # Wait for the first snapshot (the control loop publishes every tick).
        first_deadline = time.perf_counter() + 5.0
        while read_snapshot() is None:
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                _logger.warning("capture loop saw no snapshot within 5s; exiting.")
                return
        if stop_event.is_set():
            return

        frame_interval = 1.0 / fps
        timeout_ms = int(2 * frame_interval * 1000 + 200)
        recording_start: float | None = None
        last_frames: dict[str, tuple[Any, float, float]] = {}
        tick = 0

        tick_cost_sum = 0.0
        reuse_count = 0
        skip_count = 0
        frames_added = 0
        ticks_window = 0
        cap_last_log = time.perf_counter()

        while not stop_event.is_set():
            if record_event is not None and not record_event.is_set():
                # Paused: idle without capturing and drop the anchor so the
                # tick clock re-anchors on resume (no timestamp gap).
                recording_start = None
                if stop_event.wait(timeout=0.02):
                    return
                continue
            if recording_start is None:
                # First tick, or first tick after a resume: anchor so the
                # current tick's target is "now" and the cadence continues.
                recording_start = time.perf_counter() - tick * frame_interval

            now = time.perf_counter()
            if now - cap_last_log >= 1.0:
                dt = now - cap_last_log
                _logger.debug(
                    "capture: %.1f fps  tick=%.1fms  added=%d reused=%d skipped=%d",
                    ticks_window / dt,
                    1e3 * tick_cost_sum / ticks_window if ticks_window else 0.0,
                    frames_added,
                    reuse_count,
                    skip_count,
                )
                tick_cost_sum = 0.0
                reuse_count = 0
                skip_count = 0
                frames_added = 0
                ticks_window = 0
                cap_last_log = now

            target_perf_ts = recording_start + tick * frame_interval
            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and stop_event.wait(timeout=wait_s):
                return

            body_t0 = time.perf_counter()
            frames: dict[str, tuple[Any, float, float]] = {}
            skip_tick = False
            for cam_key, cam in cameras.items():
                try:
                    frame, cap_ts, recv_ts = cam.read_at_or_after(
                        target_perf_ts, timeout_ms=timeout_ms
                    )
                except (TimeoutError, RuntimeError) as exc:
                    cached = last_frames.get(cam_key)
                    if cached is None:
                        _logger.debug(
                            "Capture tick %d: %s read failed (%s) and no cached "
                            "frame; skipping tick.",
                            tick,
                            cam_key,
                            exc,
                        )
                        skip_tick = True
                        break
                    reuse_count += 1
                    frame, cap_ts, recv_ts = cached
                frames[cam_key] = (frame, cap_ts, recv_ts)
                last_frames[cam_key] = (frame, cap_ts, recv_ts)

            if skip_tick:
                skip_count += 1
                tick += 1
                continue

            snap = read_snapshot()
            if snap is None:
                tick += 1
                continue
            joint_obs, action, _snap_ts = snap

            obs: dict[str, Any] = dict(joint_obs)
            for cam_key, (frame, _cap_ts, _recv_ts) in frames.items():
                obs[cam_key] = frame
            obs_processed = robot_obs_proc(obs)

            obs_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            if stop_event.is_set():
                return
            dataset.add_frame({**obs_frame, **act_frame, "task": task})
            if frame_counter is not None:
                frame_counter["n"] += 1
            frames_added += 1
            tick_cost_sum += time.perf_counter() - body_t0
            ticks_window += 1

            if rerun_ip:
                log_rerun_data(
                    observation=_obs_for_rerun(obs_processed, frames.keys()),
                    action=action,
                )

            tick += 1
    except Exception as exc:  # noqa: BLE001 - surface instead of dying silently
        _logger.error("capture loop failed: %s", exc)
        if on_error is not None:
            on_error(str(exc))


def run_encoded_capture_loop(
    *,
    cameras: dict[str, Any],
    read_snapshot: Callable[[], tuple[dict, dict, float] | None],
    dataset: "LeRobotDataset",
    robot_obs_proc: Callable[[Any], Any],
    fps: int,
    task: str,
    rerun_ip: str | None,
    stop_event: threading.Event,
    frame_counter: "dict[str, int] | None" = None,
    on_error: Callable[[str], None] | None = None,
) -> None:
    """Frame-driven capture for the relay-encoded (gstshm-h264) transport.

    ``frame_counter`` mirrors :func:`run_capture_loop`'s (a mutable
    ``{"n": int}`` incremented per appended row). There is no ``record_event``
    on this path: an encoded stream cannot gate mid-episode — every dropped
    access unit is referenced by later P-frames.

    Unlike :func:`run_capture_loop` (real-time paced, *selecting* the camera
    frame nearest each tick and dropping the rest), an encoded stream cannot drop
    frames — every P-frame depends on its predecessor — so this loop is driven by
    the **arrival** of access units: it consumes exactly one AU per camera per
    dataset row and pairs it with the latest joint/action snapshot. The blocking
    per-camera read naturally paces the loop to the camera cadence and keeps the
    cameras mutually frame-aligned; a genuine per-camera stall (no fresh AU within
    :data:`_ENCODED_ROW_TIMEOUT_S`) re-muxes that camera's previous AU so every
    mp4 keeps frame-count == row-count (the encoded analog of "reuse last frame").

    The muxer assigns each AU a constant-fps PTS (``k / fps``), so the mp4
    timeline is exact regardless of arrival jitter; ``recv_ts`` is used only for
    the (best-effort) snapshot pairing. The first delivered AU per camera is
    always an IDR (:meth:`EncodedAuReader.flush` re-arms keyframe-wait), so each
    episode's mp4 is decodable from frame 0.
    """
    try:
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        # Wait for the first snapshot (the control loop publishes every tick).
        # Keep it: rows whose seqlock read later misses reuse the last good one.
        first_deadline = time.perf_counter() + 5.0
        last_snap = read_snapshot()
        while last_snap is None:
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                _logger.warning(
                    "encoded capture loop saw no snapshot within 5s; exiting."
                )
                return
            last_snap = read_snapshot()
        if stop_event.is_set():
            return

        # Arm each reader: drop stragglers from the previous episode and require
        # the next delivered AU to be a keyframe.
        for cam in cameras.values():
            cam.flush()

        def read_au(cam: Any, deadline: float) -> bytes | None:
            """Pop the next AU by ``deadline``, waking every poll for stop_event.

            Once the deadline has passed, still makes one non-blocking attempt:
            a camera whose AU is already queued must advance even when an
            earlier camera consumed the whole row budget (repeating it would
            leave the queued AU to a later row and skew that camera's timeline).
            """
            while not stop_event.is_set():
                remaining_ms = (deadline - time.perf_counter()) * 1000.0
                try:
                    au, _recv = cam.read_next_au(
                        timeout_ms=min(_ENCODED_POLL_MS, max(remaining_ms, 0.0))
                    )
                    return au
                except TimeoutError:
                    if remaining_ms <= 0:
                        return None
            return None

        last_au: dict[str, bytes] = {}
        primed = False
        rows_added = 0
        repeats = 0
        max_pending = 0
        last_log = time.perf_counter()

        while not stop_event.is_set():
            budget = _ENCODED_START_TIMEOUT_S if not primed else _ENCODED_ROW_TIMEOUT_S
            # One shared deadline for the whole row: with per-camera budgets the
            # serial reads compound (a stalled first camera would hand every
            # later camera an extra full budget of implicit wait), and the
            # repeat-vs-advance decision would be made against a different
            # clock per camera. A row-wide deadline keeps the cameras on the
            # same clock; read_au's post-deadline non-blocking attempt still
            # advances any camera whose AU is already queued.
            row_deadline = time.perf_counter() + budget
            aus: dict[str, bytes] = {}
            missing_first = False
            for cam_key, cam in cameras.items():
                au = read_au(cam, row_deadline)
                if au is not None:
                    aus[cam_key] = au
                    last_au[cam_key] = au
                elif cam_key in last_au:
                    aus[cam_key] = last_au[cam_key]
                    repeats += 1
                else:
                    missing_first = True
                    break
                pending = cam.pending
                if pending > max_pending:
                    max_pending = pending

            if stop_event.is_set():
                return
            if missing_first:
                # A camera produced no encoded frame within the startup budget —
                # its relay dataset branch never came up. Abort rather than record
                # a dataset with a missing/short video for that camera.
                raise RuntimeError(
                    f"camera produced no encoded frames within {budget:.0f}s"
                )
            primed = True

            snap = read_snapshot()
            if snap is None:
                # Seqlock retry miss (writer mid-update). Reuse the previous
                # tick's snapshot rather than skipping the row: the AUs are
                # already dequeued, and discarding them would punch a hole in
                # each camera's H.264 stream (later P-frames reference the
                # dropped picture) while later rows kept advancing.
                snap = last_snap
            last_snap = snap
            joint_obs, action, _snap_ts = snap

            # Process joint obs alone, then inject the AU bytes as the video
            # values: build_dataset_frame copies video values verbatim, so each
            # AU reaches feed_frame unmodified (the obs processor never sees, and
            # so never mangles, the encoded bytes).
            obs_processed = robot_obs_proc(dict(joint_obs))
            for cam_key, au in aus.items():
                obs_processed[cam_key] = au

            obs_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            if stop_event.is_set():
                return
            dataset.add_frame({**obs_frame, **act_frame, "task": task})
            if frame_counter is not None:
                frame_counter["n"] += 1
            rows_added += 1

            if rerun_ip:
                # No decoded frames on this path; log joints/action only.
                log_rerun_data(
                    observation={
                        k: v for k, v in obs_processed.items() if k not in aus
                    },
                    action=action,
                )

            now = time.perf_counter()
            if now - last_log >= 1.0:
                dt = now - last_log
                _logger.debug(
                    "encoded capture: %.1f fps  rows(win)=%d repeats=%d backlog=%d",
                    rows_added / dt,
                    rows_added,
                    repeats,
                    max_pending,
                )
                rows_added = 0
                repeats = 0
                max_pending = 0
                last_log = now
    except Exception as exc:  # noqa: BLE001 - surface instead of dying silently
        _logger.error("encoded capture loop failed: %s", exc)
        if on_error is not None:
            on_error(str(exc))


# ---------------------------------------------------------------------------
# Dataset open + finalize (shared by both recorders)
# ---------------------------------------------------------------------------


def _open_dataset(config: dict) -> "LeRobotDataset":
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if config["is_complete"]:
        return LeRobotDataset.resume(
            repo_id=config["repo_id"],
            root=config["dataset_root"],
            image_writer_threads=4,
            streaming_encoding=True,
            encoder_threads=_ENCODER_THREADS,
            vcodec=config["vcodec"],
        )
    return LeRobotDataset.create(
        repo_id=config["repo_id"],
        fps=config["fps"],
        root=config["root"],
        features=config["features"],
        robot_type=config["robot_type"],
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_threads=_ENCODER_THREADS,
        vcodec=config["vcodec"],
    )


def _verify_videos_decodable(
    dataset_root: "Path | str", since: float | None = None
) -> list[tuple[str, int, int]]:
    """Safety net: assert every recorded video frame is actually decodable.

    The exact-``k/fps`` timestamp grid means LeRobot's save-time validation only
    checks *packet* timestamps — so if an upstream stall drops a keyframe, the
    orphaned frames are muxed (packet present, correct PTS) yet fail to *decode*,
    producing a dataset that validates on save but raises ``FrameTimestampError``
    at train time. This re-decodes video files end to end and, for any file
    whose decoded-frame count is short of its packet count, logs a loud error
    naming the file (so the operator re-records the affected episode). Best-effort:
    a probe failure never breaks finalize. Returns the list of bad files.

    ``since`` (wall-clock epoch) limits the scan to files modified after it —
    i.e. the files this session actually wrote (an append rewrites the whole
    chunk file, so its mtime is fresh). Without the filter the verify decodes
    the *entire* dataset on every shutdown, which grows with total dataset
    size and makes Ctrl+C appear hung on a large resumed dataset (earlier
    sessions' files were already verified when they were written).
    """
    import concurrent.futures

    import av

    root = Path(dataset_root)
    videos_root = root / "videos"
    if not videos_root.exists():
        return []
    mp4s = sorted(videos_root.glob("observation.images.*/chunk-*/file-*.mp4"))
    if since is not None:
        mp4s = [p for p in mp4s if p.stat().st_mtime >= since]
    if not mp4s:
        return []
    _logger.info(
        "video integrity: verifying %d file(s)%s (%.0f MB)",
        len(mp4s),
        " written this session" if since is not None else "",
        sum(p.stat().st_size for p in mp4s) / 1e6,
    )

    def _probe(mp4: "Path") -> tuple[int, int] | None:
        with av.open(str(mp4)) as container:
            packets = sum(1 for p in container.demux(video=0) if p.pts is not None)
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            decoded = sum(1 for _ in container.decode(stream))
        return packets, decoded

    bad: list[tuple[str, int, int]] = []
    # Decode files in parallel (PyAV releases the GIL), so shutdown waits for
    # the slowest file rather than the sum of all cameras.
    workers = min(len(mp4s), os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe, mp4): mp4 for mp4 in mp4s}
        for fut, mp4 in futures.items():
            try:
                packets, decoded = fut.result()
            except Exception as exc:  # noqa: BLE001 - probe must never break finalize
                _logger.warning("video integrity: could not verify %s: %s", mp4, exc)
                continue
            if decoded != packets:
                rel = str(mp4.relative_to(root))
                bad.append((rel, packets, decoded))
                _logger.error(
                    "video integrity: %s has %d frames but only %d decode (%d "
                    "undecodable) — an upstream drop cost a keyframe; those dataset "
                    "rows will fail to load, re-record the affected episode(s)",
                    rel,
                    packets,
                    decoded,
                    packets - decoded,
                )
    if not bad:
        _logger.info("video integrity: all recorded frames decodable")
    return bad


def _finalize_dataset(
    dataset: "LeRobotDataset",
    config: dict,
    episodes_recorded: int,
    session_start: float | None = None,
) -> None:
    from lerobot.utils.utils import log_say

    dataset.finalize()
    if episodes_recorded > 0:
        with contextlib.suppress(Exception):
            _verify_videos_decodable(config["dataset_root"], since=session_start)
    if config["push_to_hub"] and episodes_recorded > 0:
        dataset.push_to_hub()
    dataset_root = Path(config["dataset_root"])
    if not config["is_complete"] and episodes_recorded == 0 and dataset_root.exists():
        try:
            shutil.rmtree(dataset_root)
            log_say(f"No episodes saved — removed empty dataset at {dataset_root}.")
        except OSError as exc:
            _logger.warning(
                "Failed to remove empty dataset at %s: %s", dataset_root, exc
            )


# ---------------------------------------------------------------------------
# In-process recorder (no-relay fallback)
# ---------------------------------------------------------------------------


class InProcessRecorder:
    """Dataset + capture thread in the control process (no-relay fallback).

    Used only when the gst video relay is unavailable. Owns the dataset, reads
    the robot's own (SDK) cameras, and runs the capture loop on a thread here.
    """

    def __init__(self, config: dict, robot: Any, robot_obs_proc: Callable) -> None:
        install_dataset_encoder()
        self._config = config
        self._robot = robot
        self._robot_obs_proc = robot_obs_proc
        self._dataset = _open_dataset(config)
        self._publisher = _SnapshotPublisher()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        # Mid-episode capture gate + row counter; same semantics as
        # DatasetRecorderProcess.pause_episode/resume_episode/frame_count.
        self._record = threading.Event()
        self._frames: dict[str, int] = {"n": 0}
        self._episodes_recorded = 0
        self._session_start = time.time()

    def publish(self, joint_obs: dict, action: dict, ts: float) -> None:
        self._publisher.write(joint_obs, action, ts)

    def episode_count(self) -> int:
        return self._dataset.num_episodes

    def start_episode(self, task: str) -> None:
        self._stop_capture()  # defensive: never overlap two capture threads
        self._dataset.clear_episode_buffer()
        self._record.set()
        self._frames["n"] = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=run_capture_loop,
            kwargs=dict(
                cameras=self._robot.cameras,
                read_snapshot=self._publisher.read_latest,
                dataset=self._dataset,
                robot_obs_proc=self._robot_obs_proc,
                fps=self._config["fps"],
                task=task,
                rerun_ip=self._config["rerun_ip"],
                stop_event=self._stop,
                record_event=self._record,
                frame_counter=self._frames,
            ),
            name="axol-capture",
            daemon=True,
        )
        self._thread.start()

    def pause_episode(self) -> int:
        """Stop capturing mid-episode; returns rows so far. Idempotent."""
        self._record.clear()
        return self._frames["n"]

    def resume_episode(self) -> int:
        """Resume a paused episode (the capture clock re-anchors). Idempotent."""
        self._record.set()
        return self._frames["n"]

    def frame_count(self) -> int:
        """Rows captured in the current episode (dataset time = n / fps)."""
        return self._frames["n"]

    def _stop_capture(self) -> None:
        if self._thread is not None and self._stop is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def save_episode(self) -> None:
        self._stop_capture()
        self._dataset.save_episode()
        self._episodes_recorded += 1

    def cancel_episode(self) -> None:
        self._stop_capture()
        self._dataset.clear_episode_buffer()

    def close(self) -> None:
        self._stop_capture()
        _finalize_dataset(
            self._dataset,
            self._config,
            self._episodes_recorded,
            session_start=self._session_start,
        )


# ---------------------------------------------------------------------------
# Recorder subprocess (relay path)
# ---------------------------------------------------------------------------


def _recorder_main(
    conn: multiprocessing.connection.Connection,
    raw_cond: Any,
    config: dict,
) -> None:
    """Recorder subprocess entry: own the dataset, capture from shared memory."""
    logging.basicConfig(level=config["log_level"])
    # Finalize's decode-verify scans only files written after this (a resumed
    # dataset's earlier files were verified by the sessions that wrote them).
    session_start = time.time()

    # Keep the recorder (+ its NVENC gst children, which inherit this) off the
    # control loop's cores; fall back to a positive nice where affinity isn't
    # available so it still never preempts the control loop / IK.
    from ..utils import affinity

    if not affinity.pin_background():
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass

    from lerobot.processor import make_default_processors

    from ..video.shm_frames import (
        EncodedAuReader,
        GstShmFrameReader,
        RawFrameReader,
        SnapshotReader,
    )

    # The relay uses one transport for all dataset sources: "gstshm-h264" ships
    # already-encoded H.264 (recorder only muxes), the others ship raw frames
    # (recorder encodes). Pick the matching encoder + capture loop from that.
    raw_meta = config["raw_meta"]
    encoded_mode = bool(raw_meta) and all(
        m["transport"] == "gstshm-h264" for m in raw_meta.values()
    )
    if encoded_mode:
        install_encoded_dataset_encoder()
    else:
        install_dataset_encoder()
    _, _, robot_obs_proc = make_default_processors()

    # Build a per-source frame reader matching the relay's chosen transport.
    # gstshm-h264: an EncodedAuReader (shmsrc → h264parse → appsink) pulling
    # pre-encoded access units in order. gstshm: a shmsrc → appsink consumer
    # pulling raw frames on THIS process's GIL (so the relay's send is never
    # starved). pyshm: the older RawFrameReader over a shared-memory block the
    # relay's Python pull loop fills. Started here (before "ready") and torn down
    # in the finally; the relay's rawvalve gates episode on/off, so the consumers
    # can run continuously and just idle when the valve is closed.
    cameras: dict[str, Any] = {}
    for source, meta in raw_meta.items():
        if meta["transport"] == "gstshm-h264":
            cam = EncodedAuReader(
                meta["socket_path"],
                meta["width"],
                meta["height"],
                meta["fps"],
                name=source,
            )
            cam.connect()
            cameras[source] = cam
        elif meta["transport"] == "gstshm":
            cam = GstShmFrameReader(
                meta["socket_path"],
                meta["caps"],
                meta["width"],
                meta["height"],
                meta["fps"],
                meta["latency_s"],
            )
            cam.connect()
            cameras[source] = cam
        else:
            cameras[source] = RawFrameReader(
                meta["shm_name"],
                meta["width"],
                meta["height"],
                meta["fps"],
                raw_cond,
            )
    snap_reader = SnapshotReader(
        config["snapshot_shm_name"], config["obs_keys"], config["action_keys"]
    )

    if config["rerun_ip"]:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(
            session_name="axol_record", ip=config["rerun_ip"], port=config["rerun_port"]
        )

    dataset = _open_dataset(config)
    conn.send(("ready", dataset.num_episodes))

    thread: threading.Thread | None = None
    stop: threading.Event | None = None
    capture_error: dict[str, str | None] = {"v": None}
    episodes_recorded = 0
    # Mid-episode capture gate + row counter (see run_capture_loop). The gate
    # is only supported on the raw transports: pausing the encoded
    # (gstshm-h264) stream mid-episode would drop access units that later
    # P-frames reference, corrupting the mp4.
    record_event = threading.Event()
    frame_counter: dict[str, int] = {"n": 0}

    def stop_capture() -> None:
        nonlocal thread
        if thread is not None and stop is not None:
            stop.set()
            thread.join(timeout=10.0)
            thread = None

    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if msg is None or msg[0] == "shutdown":
                break
            kind = msg[0]
            if kind == "start_episode":
                task = msg[1]
                stop_capture()  # defensive: never overlap two capture threads
                dataset.clear_episode_buffer()
                capture_error["v"] = None
                record_event.set()
                frame_counter["n"] = 0
                stop = threading.Event()
                loop_kwargs = dict(
                    cameras=cameras,
                    read_snapshot=snap_reader.read_latest,
                    dataset=dataset,
                    robot_obs_proc=robot_obs_proc,
                    fps=config["fps"],
                    task=task,
                    rerun_ip=config["rerun_ip"],
                    stop_event=stop,
                    on_error=lambda m: capture_error.__setitem__("v", m),
                )
                loop_kwargs["frame_counter"] = frame_counter
                if not encoded_mode:
                    loop_kwargs["record_event"] = record_event
                thread = threading.Thread(
                    target=(
                        run_encoded_capture_loop if encoded_mode else run_capture_loop
                    ),
                    kwargs=loop_kwargs,
                    name="axol-capture",
                    daemon=True,
                )
                thread.start()
            elif kind == "pause_episode":
                if encoded_mode:
                    conn.send(
                        (
                            "error",
                            "pause_episode requires a raw transport; the "
                            "encoded (gstshm-h264) transport can't gate "
                            "mid-episode.",
                        )
                    )
                else:
                    record_event.clear()
                    conn.send(("paused", frame_counter["n"]))
            elif kind == "resume_episode":
                if encoded_mode:
                    conn.send(
                        (
                            "error",
                            "resume_episode requires a raw transport; the "
                            "encoded (gstshm-h264) transport can't gate "
                            "mid-episode.",
                        )
                    )
                else:
                    record_event.set()
                    conn.send(("resumed", frame_counter["n"]))
            elif kind == "frame_count":
                conn.send(("frame_count", frame_counter["n"]))
            elif kind == "save_episode":
                stop_capture()
                if capture_error["v"] is not None:
                    conn.send(("error", capture_error["v"]))
                else:
                    try:
                        t_save = time.perf_counter()
                        dataset.save_episode()
                        _logger.info(
                            "save_episode took %.1fs", time.perf_counter() - t_save
                        )
                        episodes_recorded += 1
                        conn.send(("saved", dataset.num_episodes))
                    except Exception as exc:  # noqa: BLE001 - report to control proc
                        _logger.error("recorder save_episode failed: %s", exc)
                        conn.send(("error", str(exc)))
            elif kind == "cancel_episode":
                stop_capture()
                dataset.clear_episode_buffer()
                conn.send(("cancelled",))
    finally:
        stop_capture()
        with contextlib.suppress(Exception):
            _finalize_dataset(
                dataset, config, episodes_recorded, session_start=session_start
            )
        for cam in cameras.values():
            with contextlib.suppress(Exception):
                cam.close()
        with contextlib.suppress(Exception):
            snap_reader.close()


class DatasetRecorderProcess:
    """Parent-side handle for the recorder subprocess.

    Creates the cross-process :class:`SnapshotWriter`, spawns the recorder, and
    exposes the same interface as :class:`InProcessRecorder`. ``publish`` is the
    only hot-path call (one ~40-float shm write per control tick); the episode
    commands are rare and run on the main thread between episodes.
    """

    def __init__(
        self,
        *,
        raw_cond: Any,
        raw_meta: dict[str, dict],
        obs_keys: list[str],
        action_keys: list[str],
        config: dict,
    ) -> None:
        from ..video.shm_frames import SnapshotWriter

        self._snap = SnapshotWriter(obs_keys, action_keys)
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        full_config = {
            **config,
            "raw_meta": raw_meta,
            "obs_keys": obs_keys,
            "action_keys": action_keys,
            "snapshot_shm_name": self._snap.name,
        }
        self._proc = ctx.Process(
            target=_recorder_main,
            args=(child_conn, raw_cond, full_config),
            daemon=True,
            name="dataset-recorder",
        )
        self._proc.start()
        child_conn.close()
        self._lock = threading.Lock()
        self._episode_count = 0

        if self._conn.poll(_READY_TIMEOUT_S):
            msg = self._conn.recv()
            if isinstance(msg, tuple) and msg[0] == "ready":
                self._episode_count = int(msg[1])
            else:
                raise RuntimeError(f"recorder sent unexpected ready message: {msg!r}")
        else:
            raise RuntimeError("recorder subprocess did not become ready in time")

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def publish(self, joint_obs: dict, action: dict, ts: float) -> None:
        self._snap.write(joint_obs, action, ts)

    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str) -> None:
        with self._lock:
            self._conn.send(("start_episode", task))

    def _episode_gate(self, command: str, expect: str) -> int:
        """Send a pause/resume/frame-count command; return the row count.

        The reply's count may lag the capture thread by one in-flight row
        (the gate is checked at tick boundaries) — a ±1-frame slop that is
        negligible for annotation spans.
        """
        with self._lock:
            self._conn.send((command,))
            if not self._conn.poll(_CMD_TIMEOUT_S):
                raise RuntimeError(f"recorder did not answer {command} in time")
            msg = self._conn.recv()
        if msg[0] == expect:
            return int(msg[1])
        raise RuntimeError(f"recorder {command} failed: {msg[1]}")

    def pause_episode(self) -> int:
        """Stop capturing mid-episode (rows + clock gate); returns rows so far.

        Raw transports only — the encoded (gstshm-h264) transport can't gate
        mid-episode (raises). On resume the capture clock re-anchors, so the
        episode's index-based timestamps stay contiguous across the gap.
        Idempotent.
        """
        return self._episode_gate("pause_episode", "paused")

    def resume_episode(self) -> int:
        """Resume a paused episode; returns rows captured so far. Idempotent."""
        return self._episode_gate("resume_episode", "resumed")

    def frame_count(self) -> int:
        """Rows captured in the current episode (dataset time = n / fps)."""
        return self._episode_gate("frame_count", "frame_count")

    def save_episode(self) -> None:
        with self._lock:
            self._conn.send(("save_episode",))
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not finish save_episode in time")
            msg = self._conn.recv()
        if msg[0] == "saved":
            self._episode_count = int(msg[1])
        elif msg[0] == "error":
            raise RuntimeError(f"recorder save_episode failed: {msg[1]}")

    def cancel_episode(self) -> None:
        with self._lock:
            self._conn.send(("cancel_episode",))
            if self._conn.poll(_SAVE_TIMEOUT_S):
                self._conn.recv()  # ("cancelled",)

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.send(("shutdown",))
        except (OSError, ValueError):
            pass
        self._proc.join(timeout=_SAVE_TIMEOUT_S)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5.0)
        with contextlib.suppress(Exception):
            self._conn.close()
        with contextlib.suppress(Exception):
            self._snap.close()
