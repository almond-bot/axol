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
    from lerobot.configs.video import RGBEncoderConfig
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
# A capture read normally wakes at least every second.  Ten seconds leaves
# generous room for a slow decoder without allowing one wedged reader to block
# an episode command forever.  Exceeding this bound is an error: callers must
# never save/clear a buffer or start a replacement thread while the old writer
# can still mutate it.
_CAPTURE_STOP_TIMEOUT_S = 10.0

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


def _stop_capture_thread(
    thread: threading.Thread | None,
    stop: threading.Event | None,
    *,
    timeout: float = _CAPTURE_STOP_TIMEOUT_S,
) -> threading.Thread | None:
    """Stop one capture thread, returning ``None`` only after proven exit.

    The returned value is intentionally assignment-friendly: the recorder may
    write ``thread = _stop_capture_thread(thread, stop)``.  If the join bound is
    exceeded this raises before that assignment, so the live thread reference
    is retained and every destructive episode operation can fail closed until
    a later retry proves it has exited.
    """
    if thread is None:
        return None
    if stop is None:
        raise RuntimeError("capture thread exists without its stop event")
    stop.set()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise RuntimeError(
            f"capture thread did not stop within {timeout:g}s; refusing to "
            "save, clear, or replace its live episode buffer"
        )
    return None


def _shutdown_process(
    process: Any,
    *,
    graceful_timeout: float,
) -> tuple[bool, bool, list[tuple[str, BaseException]]]:
    """Stop a child process while completing every independent shutdown step.

    Returns ``(still_alive, forced, failures)``.  A failed join, liveness
    probe, or terminate call does not prevent a later kill attempt: teardown
    code uses this precisely when process ownership is already uncertain.
    """
    failures: list[tuple[str, BaseException]] = []
    forced = False

    def join(label: str, timeout: float) -> None:
        try:
            process.join(timeout=timeout)
        except BaseException as error:
            failures.append((label, error))

    def is_alive(label: str) -> bool:
        try:
            return bool(process.is_alive())
        except BaseException as error:
            failures.append((label, error))
            # Failure to prove exit must be treated as live so the next,
            # stronger shutdown action is still attempted.
            return True

    join("graceful join", graceful_timeout)
    alive = is_alive("post-join liveness check")
    if alive:
        forced = True
        try:
            process.terminate()
        except BaseException as error:
            failures.append(("terminate", error))
        join("post-terminate join", 5.0)
        alive = is_alive("post-terminate liveness check")
    if alive:
        try:
            process.kill()
        except BaseException as error:
            failures.append(("kill", error))
        join("post-kill join", 5.0)
        alive = is_alive("post-kill liveness check")
    return alive, forced, failures


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


def make_rgb_encoder(vcodec: str) -> "RGBEncoderConfig":
    """Build the dataset's RGB encoder config for the chosen codec.

    Replaces the old ``_get_codec_options`` monkeypatch: since LeRobot moved
    all codec tuning onto encoder-config objects, the tuning now rides the
    ``RGBEncoderConfig`` handed to ``LeRobotDataset.create/resume``, which
    LeRobot forwards to every encode path (streaming encoder, batch workers,
    video-info metadata). For the CPU codecs we keep the tuned options —
    ``preset=veryfast`` and a ``g=`` :data:`_ENCODER_GOP` GOP instead of
    LeRobot's near-every-frame keyframe default — so the software fallback
    stays realtime on Tegra; other codecs keep LeRobot's defaults.
    """
    from lerobot.configs.video import RGBEncoderConfig

    cfg = RGBEncoderConfig(vcodec=vcodec)  # resolves "auto" to a concrete codec
    if cfg.vcodec in ("h264", "hevc"):
        # bf=0 disables B-frames: x264/x265 presets otherwise emit reordered
        # (pts != dts) packets whose 2-frame lead-in defeats the exact-k/fps
        # re-stamp in _concat_constant_fps — the first chunk frame would sit
        # ~2/fps after 0 and every episode-0 timestamp lookup would miss.
        cfg = RGBEncoderConfig(
            vcodec=cfg.vcodec,
            preset="veryfast",
            g=_ENCODER_GOP,
            extra_options={"bf": 0},
        )
        _logger.info(
            "tuned software video encoder: preset=veryfast g=%d bf=0 threads=%d",
            _ENCODER_GOP,
            _ENCODER_THREADS,
        )
    return cfg


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

    # PyAV reopens its output path after NamedTemporaryFile closes. Keep that
    # path outside the operator-owned dataset tree, then publish the completed
    # bytes through descriptor-relative no-follow I/O.
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
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
                        # The stamped values are in the k/fps grid time base, not
                        # the source's; declare it so mux()'s rescale (from
                        # packet.time_base to the muxer's actual track timescale)
                        # starts from the right unit. Without this a source whose
                        # tb differs from the grid (e.g. the accumulated chunk
                        # after the mp4 muxer picked its own timescale) lands at
                        # the wrong instants.
                        packet.time_base = time_base
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
                            pad.time_base = time_base
                            pad.stream = out_stream
                            dst.mux(pad)
                            frame_idx += 1
        from ..utils.state_files import secure_atomic_copy_file

        secure_atomic_copy_file(tmp_output_video_path, output_video_path)
        Path(tmp_output_video_path).unlink()
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
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    try:
        with av.open(tmp_output_video_path, mode="w") as dst:
            out_streams: dict[int, object] = {}  # input stream index -> output stream
            offsets: dict[object, int] = {}  # output stream -> next start dts (out tb)
            # Time base the shifted values are computed in, captured at stream
            # creation: once the header is written the muxer may report its own
            # track timescale from out_stream.time_base, so it can't be re-read.
            out_tbs: dict[object, Fraction] = {}

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
                            out_tbs[out_stream] = Fraction(in_stream.time_base)

                    for packet in src.demux():
                        if packet.dts is None:  # demux flushing packet
                            continue
                        out_stream = out_streams.get(packet.stream.index)
                        if out_stream is None:
                            continue
                        out_tb = out_tbs[out_stream]
                        ratio = Fraction(packet.stream.time_base) / out_tb
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
                        # Declare the unit the shifted values are in so mux()'s
                        # rescale to the actual track timescale starts from it.
                        packet.time_base = out_tb
                        packet.stream = out_stream

                        end = packet.dts + dur
                        if end > seg_end.get(out_stream, end - 1):
                            seg_end[out_stream] = end
                        dst.mux(packet)

                    for out_stream, end in seg_end.items():
                        offsets[out_stream] = end

        from ..utils.state_files import secure_atomic_copy_file

        secure_atomic_copy_file(tmp_output_video_path, output_video_path)
        Path(tmp_output_video_path).unlink()
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
        # No NVENC: the stock streaming encoder runs with the tuned
        # RGBEncoderConfig built in _open_dataset (see make_rgb_encoder).
        return False

    # The relay ships NV12 to the recorder, which the NVENC encoder feeds straight
    # through; teach LeRobot's frame validation to accept that packed layout.
    _patch_frame_validation()

    def _build_nvenc(
        fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
    ):
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

    def _build_mux(
        fps, rgb_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
    ):
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
    """In-process snapshot ring (the no-relay fallback's snapshot sink).

    The control loop calls :meth:`write` every tick; the capture thread reads
    via :meth:`read_latest` or :meth:`read_nearest`. Returns ``None`` before
    the first write. The method names mirror
    :class:`~almond_axol.video.shm_frames.SnapshotReader` so the capture loop
    is identical in both paths.
    """

    def __init__(self, maxlen: int = 64) -> None:
        import collections

        self._lock = threading.Lock()
        self._ring: collections.deque[tuple[dict, dict, float, bool]] = (
            collections.deque(maxlen=maxlen)
        )

    def write(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        with self._lock:
            self._ring.append((joint_obs, action, ts, intervention))

    def read_latest(self) -> tuple[dict, dict, float, bool] | None:
        with self._lock:
            return self._ring[-1] if self._ring else None

    def read_nearest(self, target_ts: float) -> tuple[dict, dict, float, bool] | None:
        """Snapshot whose timestamp is nearest ``target_ts`` (see SnapshotReader)."""
        with self._lock:
            if not self._ring:
                return None
            return min(self._ring, key=lambda s: abs(s[2] - target_ts))


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
    read_snapshot: Callable[[], tuple[dict, dict, float, bool] | None],
    read_snapshot_nearest: (
        Callable[[float], tuple[dict, dict, float, bool] | None] | None
    ) = None,
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
    with ``capture_perf_ts >= T_n`` from every camera, pulls the joint+action
    snapshot, and appends one dataset row. A camera read timeout reuses the
    previous frame for that camera (or skips the tick if none yet). Any fatal
    error is reported via ``on_error`` instead of dying silently.

    ``read_snapshot_nearest`` (optional) enables nearest-timestamp pose↔image
    pairing: the row's snapshot is the one whose capture time is closest to
    the row's freshest camera exposure time (both on the system-wide
    ``perf_counter`` timeline), instead of whichever was newest when the loop
    got around to reading (latest-wins). ``None`` keeps latest-wins.

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

    When the dataset declares an ``intervention`` feature (LeRobot's native
    DAgger annotation: a per-frame bool, see ``lerobot.rollout``'s DAgger
    strategy), each row is tagged from the snapshot's intervention flag — the
    publisher (the control loop) marks the ticks where a human was driving.
    """
    try:
        import numpy as np
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        tag_intervention = "intervention" in dataset.features

        # Wait for the first snapshot *published after this episode started*.
        # The snapshot slot is a single latest-wins register that persists
        # across episodes, so at episode start it can still hold the previous
        # episode's final snapshot (the control loop may not have published
        # yet); pairing fresh camera frames with it would write a stale pose
        # into the episode's opening rows. Snapshot timestamps are
        # perf_counter values from the publishing process — comparable here
        # because CLOCK_MONOTONIC is system-wide (the camera-frame pairing
        # below already relies on this).
        episode_start = time.perf_counter()
        first_deadline = episode_start + 5.0
        while True:
            snap = read_snapshot()
            if snap is not None and snap[2] >= episode_start:
                break
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                # A capture thread that dies here records *zero* rows for the
                # episode; report it like any other capture failure so the
                # operator hears about it at save instead of discovering an
                # empty episode later.
                msg = (
                    "capture loop saw no fresh snapshot within 5s of episode "
                    "start — the control loop isn't publishing; no dataset "
                    "rows will be recorded for this episode"
                )
                _logger.error(msg)
                if on_error is not None:
                    on_error(msg)
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

            # Pair the row with the snapshot captured nearest the freshest
            # camera exposure time (Mantis pose↔image alignment); latest-wins
            # when no nearest reader was provided.
            row_cap_ts = max(cap for (_f, cap, _r) in frames.values())
            snap = (
                read_snapshot_nearest(row_cap_ts)
                if read_snapshot_nearest is not None
                else read_snapshot()
            )
            if snap is None:
                tick += 1
                continue
            joint_obs, action, snap_ts, intervention = snap

            obs: dict[str, Any] = dict(joint_obs)
            for cam_key, (frame, _cap_ts, _recv_ts) in frames.items():
                obs[cam_key] = frame
            obs_processed = robot_obs_proc(obs)
            # Residual pose↔image skew for this row: freshest camera exposure
            # time minus the snapshot's capture time (both on the system-wide
            # perf_counter timeline). Only recorded when the dataset declares
            # an ``observation.pose_lag`` feature (Mantis mode) — otherwise
            # build_dataset_frame ignores the value.
            obs_processed["pose_lag"] = row_cap_ts - snap_ts

            obs_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            if stop_event.is_set():
                return
            row = {**obs_frame, **act_frame, "task": task}
            if tag_intervention:
                row["intervention"] = np.array([intervention], dtype=bool)
            dataset.add_frame(row)
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
    read_snapshot: Callable[[], tuple[dict, dict, float, bool] | None],
    read_snapshot_nearest: (
        Callable[[float], tuple[dict, dict, float, bool] | None] | None
    ) = None,
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
    dataset row and pairs it with the nearest joint/action snapshot using the
    relay-compensated capture-time estimate. The blocking
    per-camera read naturally paces the loop to the camera cadence and keeps the
    cameras mutually frame-aligned; a genuine per-camera stall (no fresh AU within
    :data:`_ENCODED_ROW_TIMEOUT_S`) re-muxes that camera's previous AU so every
    mp4 keeps frame-count == row-count (the encoded analog of "reuse last frame").

    The muxer assigns each AU a constant-fps PTS (``k / fps``), so the mp4
    timeline is exact regardless of arrival jitter; the reader's estimated
    capture timestamp is used only for snapshot pairing. The first delivered AU per camera is
    always an IDR (:meth:`EncodedAuReader.flush` re-arms keyframe-wait), so each
    episode's mp4 is decodable from frame 0.
    """
    try:
        import numpy as np
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        tag_intervention = "intervention" in dataset.features

        # Wait for the first snapshot *published after this episode started* —
        # the slot persists across episodes, so a stale previous-episode
        # snapshot must not seed the opening rows (see run_capture_loop).
        # Keep it: rows whose seqlock read later misses reuse the last good one.
        episode_start = time.perf_counter()
        first_deadline = episode_start + 5.0
        last_snap = read_snapshot()
        while last_snap is None or last_snap[2] < episode_start:
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                # Same as run_capture_loop: dying silently here yields a
                # zero-row episode nobody hears about until much later.
                msg = (
                    "encoded capture loop saw no fresh snapshot within 5s of "
                    "episode start — the control loop isn't publishing; no "
                    "dataset rows will be recorded for this episode"
                )
                _logger.error(msg)
                if on_error is not None:
                    on_error(msg)
                return
            last_snap = read_snapshot()
        if stop_event.is_set():
            return

        # Arm each reader: drop stragglers from the previous episode and require
        # the next delivered AU to be a keyframe.
        for cam in cameras.values():
            cam.flush()

        def read_au(cam: Any, deadline: float) -> tuple[bytes, float] | None:
            """Pop the next ``(au, recv_ts)`` by ``deadline``, waking every poll.

            Once the deadline has passed, still makes one non-blocking attempt:
            a camera whose AU is already queued must advance even when an
            earlier camera consumed the whole row budget (repeating it would
            leave the queued AU to a later row and skew that camera's timeline).
            """
            while not stop_event.is_set():
                remaining_ms = (deadline - time.perf_counter()) * 1000.0
                try:
                    return cam.read_next_au(
                        timeout_ms=min(_ENCODED_POLL_MS, max(remaining_ms, 0.0))
                    )
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
            row_capture_ts = 0.0
            missing_first = False
            for cam_key, cam in cameras.items():
                popped = read_au(cam, row_deadline)
                if popped is not None:
                    au, capture_ts = popped
                    aus[cam_key] = au
                    last_au[cam_key] = au
                    if capture_ts > row_capture_ts:
                        row_capture_ts = capture_ts
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

            # Pair the row with the snapshot captured nearest the latest camera
            # capture estimate. Each reader subtracts the relay-measured camera
            # + dataset-encode latency from AU receipt time; latest-wins when no
            # nearest reader was provided. A full-stall row (every camera
            # re-muxed its previous AU) has no fresh timestamp — fall back to
            # "now".
            pair_ts = row_capture_ts if row_capture_ts else time.perf_counter()
            snap = (
                read_snapshot_nearest(pair_ts)
                if read_snapshot_nearest is not None
                else read_snapshot()
            )
            if snap is None:
                # Seqlock retry miss (writer mid-update). Reuse the previous
                # tick's snapshot rather than skipping the row: the AUs are
                # already dequeued, and discarding them would punch a hole in
                # each camera's H.264 stream (later P-frames reference the
                # dropped picture) while later rows kept advancing.
                snap = last_snap
            last_snap = snap
            joint_obs, action, snap_ts, intervention = snap

            # Process joint obs alone, then inject the AU bytes as the video
            # values: build_dataset_frame copies video values verbatim, so each
            # AU reaches feed_frame unmodified (the obs processor never sees, and
            # so never mangles, the encoded bytes).
            obs_processed = robot_obs_proc(dict(joint_obs))
            # Residual pose↔image skew (see run_capture_loop). Encoded AUs do
            # not retain their sensor PTS, so pair_ts is the reader's compensated
            # capture-time estimate rather than an exact exposure timestamp.
            # Only recorded when the dataset declares observation.pose_lag.
            obs_processed["pose_lag"] = pair_ts - snap_ts
            for cam_key, au in aus.items():
                obs_processed[cam_key] = au

            obs_frame = build_dataset_frame(
                dataset.features, obs_processed, prefix=OBS_STR
            )
            act_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            if stop_event.is_set():
                return
            row = {**obs_frame, **act_frame, "task": task}
            if tag_intervention:
                row["intervention"] = np.array([intervention], dtype=bool)
            dataset.add_frame(row)
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

    from ..utils.state_files import (
        confine_service_dataset_path,
        privileged_service_active,
    )

    if privileged_service_active():
        # Every current caller validates before hardware startup. Repeat the
        # boundary at the final third-party write site so a future caller—or a
        # separately spawned recorder with altered config—cannot bypass it.
        dataset_root = confine_service_dataset_path(
            Path(config["dataset_root"]),
            label="recorder dataset root",
        )
        config["dataset_root"] = str(dataset_root)
        config["root"] = str(dataset_root)

    rgb_encoder = make_rgb_encoder(config["vcodec"])
    if config["is_complete"]:
        # Defense in depth: callers validate before opening hardware, then the
        # recorder rechecks immediately before LeRobot.resume. This closes the
        # gap where changed metadata (or a future unchecked caller) could make
        # resume silently retain a schema different from ``config['features']``.
        from .datasets import require_dataset_resume_schema

        require_dataset_resume_schema(
            Path(config["dataset_root"]),
            config["features"],
            fps=int(config["fps"]),
            allowed_extra_features=frozenset(config.get("allowed_resume_features", ())),
        )
        return LeRobotDataset.resume(
            repo_id=config["repo_id"],
            root=config["dataset_root"],
            image_writer_threads=4,
            streaming_encoding=True,
            encoder_threads=_ENCODER_THREADS,
            rgb_encoder=rgb_encoder,
        )
    dataset = LeRobotDataset.create(
        repo_id=config["repo_id"],
        fps=config["fps"],
        root=config["root"],
        features=config["features"],
        robot_type=config["robot_type"],
        use_videos=True,
        image_writer_threads=4,
        streaming_encoding=True,
        encoder_threads=_ENCODER_THREADS,
        rgb_encoder=rgb_encoder,
    )
    # LeRobot's codebase_version describes the dataset format, not the Axol
    # URDF/world frame.  Record our pose-frame provenance on fresh Cartesian
    # datasets so future migrations can distinguish them without guessing.
    action_names = (config["features"].get("action") or {}).get("names") or []
    if any("_ee." in name for name in action_names):
        from .cartesian_frame import write_cartesian_frame_marker

        write_cartesian_frame_marker(config["dataset_root"])
    return dataset


def _maybe_smooth_episode(dataset: "LeRobotDataset", config: dict) -> None:
    """Zero-phase low-pass the buffered episode's EE pose track before save.

    Active only when the recorder config carries a positive ``smooth_ee_hz``
    cutoff — collect-data sets it for Mantis sessions, where the pose track comes
    from the VR tracker and carries broadband measurement noise on the order
    of the per-frame motion. On-robot sessions (encoder FK) leave it unset.
    See :func:`almond_axol.mantis.smoothing.smooth_episode_ee_poses`.
    """
    cutoff_hz = float(config.get("smooth_ee_hz") or 0.0)
    if cutoff_hz <= 0.0:
        return
    from ..mantis.smoothing import smooth_episode_ee_poses

    smooth_episode_ee_poses(
        dataset.writer.episode_buffer, dataset.meta.features, config["fps"], cutoff_hz
    )


def make_episode_durable(dataset: "LeRobotDataset") -> dict[str, Any]:
    """Flush the just-saved episode to disk so a kill can no longer lose it.

    LeRobot keeps two parquet writers open across the whole session — the data
    rows, and ``meta/episodes`` (which buffers ten episodes per row group) —
    and only ``finalize()`` writes their footers, without which a parquet file
    cannot be read at all. Every episode saved so far therefore used to ride
    on a clean shutdown: a recorder killed mid-session (or an operator killing
    a shutdown that looked hung) lost them all, leaving the dataset
    crash-inconsistent. Closing both writers right after each ``save_episode``
    puts the footers on disk immediately, and re-arming the writers' rotation
    state — the same state a fresh ``resume`` starts from — makes the *next*
    episode open new data/meta/video files instead of appending. At every
    point between saves the dataset on disk is complete, readable, and
    resumable, so shutdown has nothing left to flush and a crash costs at most
    the episode currently being written.

    The rotation gives each episode its own parquet/mp4 files (LeRobot's
    format addresses them per episode by chunk/file index, so per-episode
    files are just the smallest legal packing). That also retires the old
    append path's video concat — which rewrote the whole accumulating chunk
    file on *every* save — so saves get cheaper as the session grows, not
    more expensive.

    Returns the just-saved episode's metadata row (scalar values), which names
    its data/video files.
    """
    meta = dataset.meta
    # latest_episode is the row _save_episode_metadata just buffered, with
    # every value wrapped in a single-element list; unwrap to the scalar form
    # meta.episodes rows use.
    last = {
        k: (v[0] if isinstance(v, list) else v) for k, v in meta.latest_episode.items()
    }
    dataset.writer.close_writer()  # data parquet footer
    try:
        meta._close_writer()  # flush the metadata buffer + its footer
    except Exception:
        # Don't leave the metadata writer half-open: its file would stay
        # footerless (unreadable). Closing writes the footer over the row
        # groups that did land; a row still in the metadata buffer stays there
        # and flushes with a later save or finalize (each row targets its own
        # rotated file, so a late flush can't truncate earlier episodes), and
        # if the process dies first the missing row is exactly the torn tail
        # the resume repair truncates.
        if getattr(meta, "_pq_writer", None) is not None:
            with contextlib.suppress(Exception):
                meta._pq_writer.close()
            meta._pq_writer = None
        raise
    finally:
        # The data writer is closed at this point no matter what happened
        # above, so rotation MUST be re-armed even on failure — otherwise the
        # next save_episode would see stale rotation state and reopen (i.e.
        # truncate) the just-finished data parquet. The writers'
        # rotate-on-resume branches read the previous episode from
        # meta.episodes[-1]; hand them the row we already hold instead of
        # re-loading every metadata parquet from disk on each save.
        meta.episodes = [last]
        meta.latest_episode = None
        dataset.writer._latest_episode = None
    return last


def _episode_video_paths(dataset_root: "Path", episode_row: dict[str, Any]) -> list:
    """The mp4 files an episode's metadata row references, one per camera."""
    paths = []
    for key in episode_row:
        if not (key.startswith("videos/") and key.endswith("/chunk_index")):
            continue
        video_key = key[len("videos/") : -len("/chunk_index")]
        paths.append(
            Path(dataset_root)
            / "videos"
            / video_key
            / f"chunk-{int(episode_row[key]):03d}"
            / f"file-{int(episode_row[f'videos/{video_key}/file_index']):03d}.mp4"
        )
    return paths


class _EpisodeVideoVerifier:
    """Decode-verifies each saved episode's videos on a background thread.

    The exact-``k/fps`` timestamp grid means LeRobot's save-time validation
    only checks *packet* timestamps — so if an upstream stall drops a
    keyframe, the orphaned frames are muxed (packet present, correct PTS) yet
    fail to *decode*, producing a dataset that validates on save but raises
    ``FrameTimestampError`` at train time. This re-decodes each episode's
    video files end to end right after the episode saves and logs a loud
    error naming any bad file, so the operator can re-record that episode on
    the spot. It used to run over the whole session's files at shutdown
    instead, which grew with session length and made Ctrl+C appear hung — the
    very thing that tempts an operator into the kill that corrupts the
    dataset. Off the save path (background thread) so the SAVING window stays
    short; per-episode files are written-once, so each is verified exactly
    once, and by shutdown the backlog is at most the last episode.
    """

    def __init__(self, dataset_root: "Path | str") -> None:
        import queue

        self._root = Path(dataset_root)
        self._queue: "queue.Queue[list | None]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="axol-video-verify", daemon=True
        )
        self._thread.start()

    def submit(self, episode_row: dict[str, Any]) -> None:
        paths = _episode_video_paths(self._root, episode_row)
        if paths:
            self._queue.put(paths)

    def close(self, timeout: float = 60.0) -> None:
        """Finish the pending verifies (bounded — the backlog is ~1 episode)."""
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            _logger.warning(
                "video integrity: verifier still running after %.0fs; leaving "
                "the last episode's videos unverified",
                timeout,
            )

    def _run(self) -> None:
        while True:
            paths = self._queue.get()
            if paths is None:
                return
            for mp4 in paths:
                try:
                    packets, decoded = self._probe(mp4)
                except Exception as exc:  # noqa: BLE001 - verify is best-effort
                    _logger.warning(
                        "video integrity: could not verify %s: %s", mp4, exc
                    )
                    continue
                if decoded != packets:
                    _logger.error(
                        "video integrity: %s has %d frames but only %d decode "
                        "(%d undecodable) — an upstream drop cost a keyframe; "
                        "those dataset rows will fail to load, re-record the "
                        "affected episode",
                        mp4.relative_to(self._root),
                        packets,
                        decoded,
                        packets - decoded,
                    )
                else:
                    _logger.info(
                        "video integrity: %s fully decodable (%d frames)",
                        mp4.relative_to(self._root),
                        decoded,
                    )

    @staticmethod
    def _probe(mp4: "Path") -> tuple[int, int]:
        import av

        with av.open(str(mp4)) as container:
            packets = sum(1 for p in container.demux(video=0) if p.pts is not None)
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            decoded = sum(1 for _ in container.decode(stream))
        return packets, decoded


def _close_dataset_writers(dataset: "LeRobotDataset") -> None:
    """Run ``dataset.finalize()``, guaranteeing the parquet footers get written.

    A parquet file is only readable once its writer is closed — that is what
    appends the row-group index and the trailing ``PAR1`` magic. With
    :func:`make_episode_durable` closing both parquet writers after every save,
    finalize normally has nothing left to flush and returns immediately — but
    if a durability flush failed mid-session the writers fall back to staying
    open, and ``DatasetWriter.finalize()`` closes them only as steps 3 and 4,
    behind the image writer drain (step 1) and the video encoder flush
    (step 2). Anything raising in those first two steps leaves the files
    footerless, which loses every episode still buffered in them: pyarrow
    can't locate a single row group without the footer, so ``load_episodes``
    fails and the dataset can no longer be read, resumed, or shipped.

    Video and image teardown failures are real but recoverable (at worst one
    episode's video is bad); losing the metadata is not. So close the writers
    even when ``finalize()`` blows up, and re-raise afterwards so the caller
    still reports the original failure.
    """
    try:
        dataset.finalize()
    except Exception:
        _logger.exception(
            "dataset.finalize() failed — closing the parquet writers directly so "
            "the episodes recorded this session stay readable"
        )
        writer = getattr(dataset, "writer", None)
        if writer is not None:
            with contextlib.suppress(Exception):
                writer.close_writer()
        with contextlib.suppress(Exception):
            dataset.meta.finalize()
        raise


def _verify_episodes_readable(dataset_root: "Path | str") -> None:
    """Log loudly if ``meta/episodes`` can't be read back after finalize.

    The failure mode this guards against (a parquet file left without its
    footer) is invisible until something downstream tries to load the dataset —
    by which point the session is over and the operator has moved on. Reading
    the metadata back here names the bad file while the log still has the
    context that produced it.     Best-effort: never raises — it runs from a ``finally``, where an escaping
    exception would mask the failure that got us here.
    """
    try:
        from lerobot.datasets.io_utils import load_episodes

        load_episodes(Path(dataset_root))
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        _logger.error(
            "meta/episodes at %s is unreadable after finalize (%s) — the episode "
            "metadata parquet is corrupt, so the dataset cannot be resumed or "
            "shipped as-is",
            dataset_root,
            exc,
        )


def _finalize_dataset(
    dataset: "LeRobotDataset",
    config: dict,
    episodes_recorded: int,
) -> None:
    """Close out the dataset at session end.

    With :func:`make_episode_durable` flushing after every save (and the
    per-episode video verify running as episodes save), this is normally
    instant: close already-closed writers, read back the episode metadata as
    a cheap sanity check, optionally push, wipe an empty fresh dataset. A
    slow shutdown is exactly what tempted operators into killing the process
    mid-finalize — the main way datasets got corrupted — so nothing
    session-length-proportional is allowed here.
    """
    from lerobot.utils.utils import log_say

    try:
        _close_dataset_writers(dataset)
    finally:
        # In a finally because the recovery path re-raises, and that is the
        # case this check exists for: the writers have been closed by then
        # either way, so this is where we find out whether the salvage worked.
        if episodes_recorded > 0:
            _verify_episodes_readable(config["dataset_root"])
    if config["push_to_hub"] and episodes_recorded > 0:
        dataset.push_to_hub()
    dataset_root = Path(config["dataset_root"])
    if not config["is_complete"] and episodes_recorded == 0 and dataset_root.exists():
        from ..utils.state_files import privileged_service_active

        if privileged_service_active():
            _logger.warning(
                "Keeping the empty dataset at %s because the hosted service "
                "will not recursively delete operator-owned paths",
                dataset_root,
            )
        else:
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
        self._verifier = _EpisodeVideoVerifier(config["dataset_root"])
        self._publisher = _SnapshotPublisher()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        # Mid-episode capture gate + row counter; same semantics as
        # DatasetRecorderProcess.pause_episode/resume_episode/frame_count.
        self._record = threading.Event()
        self._frames: dict[str, int] = {"n": 0}
        # Fatal capture-thread failure for the current episode, surfaced at
        # save_episode — same contract as the recorder subprocess's
        # ``capture_error`` (an episode whose capture died must not save).
        self._capture_error: dict[str, str | None] = {"v": None}
        self._episodes_recorded = 0

    def publish(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        self._publisher.write(joint_obs, action, ts, intervention)

    def episode_count(self) -> int:
        return self._dataset.num_episodes

    def start_episode(self, task: str) -> None:
        from ..lerobot.nvenc_encoder import reset_dropped_frames

        self._stop_capture()  # defensive: never overlap two capture threads
        self._dataset.clear_episode_buffer()
        self._record.set()
        self._frames["n"] = 0
        self._capture_error["v"] = None
        reset_dropped_frames()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=run_capture_loop,
            kwargs=dict(
                cameras=self._robot.cameras,
                read_snapshot=self._publisher.read_latest,
                read_snapshot_nearest=self._publisher.read_nearest,
                dataset=self._dataset,
                robot_obs_proc=self._robot_obs_proc,
                fps=self._config["fps"],
                task=task,
                rerun_ip=self._config["rerun_ip"],
                stop_event=self._stop,
                record_event=self._record,
                frame_counter=self._frames,
                on_error=lambda m: self._capture_error.__setitem__("v", m),
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
        # Match the subprocess recorder's fail-closed lifecycle: never block
        # forever on a wedged SDK camera read, and never forget the exact
        # writer until a bounded retry proves it exited.
        self._thread = _stop_capture_thread(self._thread, self._stop)

    def stop_capture(self) -> tuple[int, str | None]:
        """Stop the capture thread WITHOUT saving or clearing the buffer.

        Call the moment an episode terminates — before any post-episode robot
        motion (gripper valve close, return-to-rest) — so the capture thread
        can't keep appending rows that pair a frozen snapshot with
        reused/stale camera frames (the "junk tail" every episode would
        otherwise carry). ``save_episode`` / ``cancel_episode`` remain valid
        afterwards and operate on the buffer as frozen here. Returns the exact
        number of buffered rows and any fatal capture error so callers can
        reject a bad take before asking LeRobot to save it. Idempotent.
        """
        self._stop_capture()
        return self._frames["n"], self._capture_error["v"]

    def save_episode(self) -> None:
        from ..lerobot.nvenc_encoder import dropped_frames

        self._stop_capture()
        if self._capture_error["v"] is not None:
            # Mirrors the recorder subprocess: a dead capture thread means the
            # buffered episode is empty or truncated — surface it instead of
            # silently saving.
            self._dataset.clear_episode_buffer()
            raise RuntimeError(f"episode capture failed: {self._capture_error['v']}")
        n_dropped = dropped_frames()
        if n_dropped:
            # Mirrors the recorder subprocess: a dropped frame's row still
            # exists, so the video is misaligned — never save it silently.
            self._dataset.clear_episode_buffer()
            raise RuntimeError(
                f"{n_dropped} video frame(s) were dropped by the encoder "
                "(feed queue overflow) — the episode's video is misaligned "
                "with its rows; episode discarded."
            )
        _maybe_smooth_episode(self._dataset, self._config)
        self._dataset.save_episode()
        # Flush the episode to disk so a kill from here on can't lose it (see
        # make_episode_durable). Best-effort: on failure the episode is still
        # saved and its remaining rows reach disk at the next save or finalize
        # (make_episode_durable leaves the writers consistent either way).
        try:
            self._verifier.submit(make_episode_durable(self._dataset))
        except Exception:  # noqa: BLE001 - durability is best-effort
            _logger.exception(
                "could not fully flush the saved episode to disk; it completes "
                "at the next save or finalize — do not kill this process"
            )
        self._episodes_recorded += 1

    def cancel_episode(self) -> None:
        self._stop_capture()
        self._dataset.clear_episode_buffer()

    def close(self) -> None:
        primary_error: BaseException | None = None
        capture_stopped = False
        try:
            self._stop_capture()
        except BaseException as error:
            primary_error = error
        else:
            capture_stopped = True

        if capture_stopped:
            try:
                _finalize_dataset(self._dataset, self._config, self._episodes_recorded)
            except BaseException as error:
                primary_error = error
        elif primary_error is not None:
            primary_error.add_note(
                "dataset finalization was skipped because capture-thread exit "
                "could not be proved"
            )

        try:
            self._verifier.close()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
            else:
                primary_error.add_note(
                    "additional recorder video verifier close failure: "
                    f"{type(error).__name__}: {error}"
                )
        if primary_error is not None:
            raise primary_error


# ---------------------------------------------------------------------------
# Recorder subprocess (relay path)
# ---------------------------------------------------------------------------


def _cleanup_recorder_session(
    *,
    stop_capture: Callable[[], None],
    dataset: Any,
    config: dict,
    episodes_recorded: int,
    verifier: Any,
    cameras: dict[str, Any],
    snap_reader: Any,
) -> None:
    """Finish a recorder session, preserving errors after complete cleanup.

    A capture thread owns the mutable episode buffer, so dataset finalization is
    permitted only after its exit is proven. If its normal stop bound expires,
    close every input reader to unblock a wedged read and retry the join before
    deciding whether finalization is safe. All independent resources are then
    closed even when stop/finalize fails; the first failure is re-raised after
    later failures have been attached as notes, making the child exit non-zero
    for :class:`DatasetRecorderProcess.close` to propagate.
    """
    primary_error: BaseException | None = None

    def remember(label: str, error: BaseException) -> None:
        nonlocal primary_error
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"additional recorder {label} failure: {type(error).__name__}: {error}"
            )

    def close_readers() -> None:
        for name, camera in cameras.items():
            try:
                camera.close()
            except BaseException as error:
                remember(f"camera {name} close", error)
        try:
            snap_reader.close()
        except BaseException as error:
            remember("snapshot reader close", error)

    capture_stopped = False
    try:
        stop_capture()
    except BaseException as error:
        _logger.exception("recorder capture thread did not stop cleanly")
        remember("capture stop", error)
        # Camera/snapshot readers are the only blocking inputs used by the
        # capture loop. Closing them is the best chance to wake a stuck read;
        # then retry the exact retained thread rather than losing ownership.
        close_readers()
        try:
            stop_capture()
        except BaseException as retry_error:
            _logger.exception("recorder capture thread still alive after reader close")
            remember("capture stop retry", retry_error)
        else:
            capture_stopped = True
    else:
        capture_stopped = True

    if capture_stopped:
        try:
            _finalize_dataset(dataset, config, episodes_recorded)
        except BaseException as error:
            _logger.exception("recorder failed to finalize the dataset")
            remember("dataset finalize", error)
    else:
        # Finalizing concurrently with a live dataset writer can corrupt the
        # buffer/files. The retained stop error is already primary; make the
        # deliberate skip explicit in its diagnostics.
        assert primary_error is not None
        primary_error.add_note(
            "dataset finalization was skipped because the capture thread is "
            "still alive and may still own the episode buffer"
        )

    try:
        # Give the verifier its bounded chance to finish the last completed
        # episode even when another teardown step failed.
        verifier.close()
    except BaseException as error:
        remember("video verifier close", error)

    # Idempotent reader closes complete the normal path and retry any close that
    # was attempted early to wake a wedged capture thread.
    close_readers()

    if primary_error is not None:
        raise primary_error


def _recorder_main(
    conn: multiprocessing.connection.Connection,
    raw_cond: Any,
    config: dict,
) -> None:
    """Recorder subprocess entry: own the dataset, capture from shared memory."""
    logging.basicConfig(level=config["log_level"])

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
                latency_s=meta.get("latency_s", 0.0),
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
    verifier = _EpisodeVideoVerifier(config["dataset_root"])
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
        # Assignment happens only after _stop_capture_thread has proved exit.
        # On timeout it raises and leaves ``thread`` pointing at the live
        # writer, which makes every destructive command below fail closed.
        thread = _stop_capture_thread(thread, stop)

    session_error: BaseException | None = None
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
                try:
                    stop_capture()  # defensive: never overlap capture threads
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                    continue
                dataset.clear_episode_buffer()
                capture_error["v"] = None
                record_event.set()
                frame_counter["n"] = 0
                from ..lerobot.nvenc_encoder import reset_dropped_frames

                reset_dropped_frames()
                stop = threading.Event()
                loop_kwargs = dict(
                    cameras=cameras,
                    read_snapshot=snap_reader.read_latest,
                    read_snapshot_nearest=snap_reader.read_nearest,
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
                conn.send(("started",))
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
            elif kind == "stop_capture":
                # Freeze the episode: stop the capture thread but keep the
                # buffered rows (and any capture_error) intact, so the caller
                # can run post-episode robot motion (valve close, rest move)
                # without junk rows being appended, then decide save/cancel.
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                else:
                    conn.send(
                        ("capture_stopped", frame_counter["n"], capture_error["v"])
                    )
            elif kind == "save_episode":
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                    continue
                from ..lerobot.nvenc_encoder import dropped_frames

                n_dropped = dropped_frames()
                if capture_error["v"] is not None:
                    conn.send(("error", capture_error["v"]))
                elif n_dropped:
                    # A dropped frame was never encoded but its row exists, so
                    # the episode's video is misaligned with its rows (and
                    # with the other cameras) — refuse to save silently
                    # corrupted data. The caller should discard and re-record.
                    dataset.clear_episode_buffer()
                    conn.send(
                        (
                            "error",
                            f"{n_dropped} video frame(s) were dropped by the "
                            "encoder (feed queue overflow) — the episode's "
                            "video is misaligned with its rows; episode "
                            "discarded. Check recorder-core load.",
                        )
                    )
                else:
                    try:
                        t_save = time.perf_counter()
                        _maybe_smooth_episode(dataset, config)
                        dataset.save_episode()
                        # Flush the episode to disk *before* acknowledging the
                        # save, so "saved" means "survives a kill". Best-effort:
                        # if the flush itself fails the episode is still saved
                        # in memory and its remaining rows reach disk at the
                        # next save or finalize (make_episode_durable leaves
                        # the writers in a consistent state either way), so
                        # don't fail the session over it — but say so loudly.
                        try:
                            episode_row = make_episode_durable(dataset)
                        except Exception:  # noqa: BLE001 - durability is best-effort
                            episode_row = None
                            _logger.exception(
                                "could not fully flush the saved episode to "
                                "disk; it completes at the next save or "
                                "finalize — do not kill this process"
                            )
                        _logger.info(
                            "save_episode took %.1fs", time.perf_counter() - t_save
                        )
                        if episode_row is not None:
                            verifier.submit(episode_row)
                        episodes_recorded += 1
                        conn.send(("saved", dataset.num_episodes))
                    except Exception as exc:  # noqa: BLE001 - report to control proc
                        _logger.error("recorder save_episode failed: %s", exc)
                        conn.send(("error", str(exc)))
            elif kind == "cancel_episode":
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                    continue
                dataset.clear_episode_buffer()
                conn.send(("cancelled",))
    except BaseException as error:
        session_error = error
        raise
    finally:
        try:
            _cleanup_recorder_session(
                stop_capture=stop_capture,
                dataset=dataset,
                config=config,
                episodes_recorded=episodes_recorded,
                verifier=verifier,
                cameras=cameras,
                snap_reader=snap_reader,
            )
        except BaseException as cleanup_error:
            if session_error is None:
                raise
            session_error.add_note(
                "additional recorder session cleanup failure: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


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

        snap = SnapshotWriter(obs_keys, action_keys)
        conn: Any | None = None
        child_conn: Any | None = None
        proc: Any | None = None
        started = False
        try:
            ctx = multiprocessing.get_context("spawn")
            conn, child_conn = ctx.Pipe()
            full_config = {
                **config,
                "raw_meta": raw_meta,
                "obs_keys": obs_keys,
                "action_keys": action_keys,
                "snapshot_shm_name": snap.name,
            }
            proc = ctx.Process(
                target=_recorder_main,
                args=(child_conn, raw_cond, full_config),
                daemon=True,
                name="dataset-recorder",
            )
            try:
                proc.start()
            except BaseException:
                # multiprocessing normally publishes ``pid`` only once the
                # child exists. If start failed after that boundary, treat it
                # as started so the constructor still reaps it.
                try:
                    started = proc.pid is not None
                except BaseException:
                    started = True
                raise
            else:
                started = True
            child_conn.close()
            if conn.poll(_READY_TIMEOUT_S):
                msg = conn.recv()
                if not (isinstance(msg, tuple) and msg[0] == "ready"):
                    raise RuntimeError(
                        f"recorder sent unexpected ready message: {msg!r}"
                    )
                episode_count = int(msg[1])
            else:
                raise RuntimeError("recorder subprocess did not become ready in time")
        except BaseException as setup_error:
            cleanup_failures: list[tuple[str, BaseException]] = []

            # A child that never completed the ready handshake cannot be
            # adopted safely. Stop it before unlinking the snapshot shm; the
            # process may already have attached to that block or opened the
            # dataset near the end of initialization.
            if started and proc is not None:
                process_alive, _, process_failures = _shutdown_process(
                    proc, graceful_timeout=0
                )
                cleanup_failures.extend(
                    (f"subprocess {label}", error) for label, error in process_failures
                )
                if process_alive:
                    cleanup_failures.append(
                        (
                            "subprocess",
                            RuntimeError(
                                "recorder subprocess remained alive after "
                                "terminate/kill"
                            ),
                        )
                    )
            local_cleanups: list[tuple[str, Callable[[], None]]] = []
            if child_conn is not None:
                local_cleanups.append(("child pipe", child_conn.close))
            if conn is not None:
                local_cleanups.append(("parent pipe", conn.close))
            local_cleanups.append(("snapshot shared memory", snap.close))
            for label, cleanup in local_cleanups:
                try:
                    cleanup()
                except BaseException as error:
                    cleanup_failures.append((label, error))
            for label, error in cleanup_failures:
                setup_error.add_note(
                    f"recorder constructor {label} cleanup failed: "
                    f"{type(error).__name__}: {error}"
                )
            raise

        assert conn is not None
        assert proc is not None
        self._snap = snap
        self._conn = conn
        self._proc = proc
        self._lock = threading.Lock()
        self._episode_count = episode_count
        self._closed = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def publish(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        self._snap.write(joint_obs, action, ts, intervention)

    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str) -> None:
        with self._lock:
            self._conn.send(("start_episode", task))
            if not self._conn.poll(_CMD_TIMEOUT_S):
                raise RuntimeError("recorder did not answer start_episode in time")
            msg = self._conn.recv()
        if msg[0] != "started":
            detail = msg[1] if len(msg) > 1 else repr(msg)
            raise RuntimeError(f"recorder start_episode failed: {detail}")

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

    def stop_capture(self) -> tuple[int, str | None]:
        """Stop the capture thread WITHOUT saving or clearing the buffer.

        Call the moment an episode terminates — before any post-episode robot
        motion (gripper valve close, return-to-rest) — so the recorder's
        capture thread can't keep appending rows that pair a frozen snapshot
        with reused/re-muxed camera frames (the "junk tail"). The buffered
        rows and any pending capture error survive; ``save_episode`` /
        ``cancel_episode`` remain valid afterwards. Returns the exact buffered
        row count and any fatal capture error reported after the capture thread
        stops, so callers can reject a bad take before asking LeRobot to save
        it. Idempotent.

        Uses the save timeout rather than the lightweight command timeout:
        the reply waits for the capture thread to join, which can take up to
        its 10 s join budget on a wedged camera read.
        """
        with self._lock:
            self._conn.send(("stop_capture",))
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not answer stop_capture in time")
            msg = self._conn.recv()
        if msg[0] != "capture_stopped":
            raise RuntimeError(f"recorder stop_capture failed: {msg!r}")
        return int(msg[1]), msg[2]

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
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not answer cancel_episode in time")
            msg = self._conn.recv()
        if msg[0] != "cancelled":
            detail = msg[1] if len(msg) > 1 else repr(msg)
            raise RuntimeError(f"recorder cancel_episode failed: {detail}")

    def close(self) -> None:
        if self._closed:
            return
        primary_error: BaseException | None = None

        def remember(label: str, error: BaseException) -> None:
            nonlocal primary_error
            if primary_error is None:
                primary_error = error
            else:
                primary_error.add_note(
                    f"additional recorder parent {label} failure: "
                    f"{type(error).__name__}: {error}"
                )

        try:
            with self._lock:
                self._conn.send(("shutdown",))
        except (OSError, ValueError):
            # An already-dead child or closed pipe is diagnosed from exitcode
            # after join; still finish every local cleanup.
            pass
        except BaseException as error:
            remember("shutdown request", error)

        forced = False
        process_alive, forced, process_failures = _shutdown_process(
            self._proc, graceful_timeout=_SAVE_TIMEOUT_S
        )
        for label, error in process_failures:
            remember(f"subprocess {label}", error)

        if process_alive:
            remember(
                "subprocess shutdown",
                RuntimeError(
                    "recorder subprocess remained alive after shutdown, terminate, "
                    "and kill; dataset ownership is uncertain"
                ),
            )
        elif forced:
            remember(
                "subprocess shutdown",
                RuntimeError(
                    f"recorder did not shut down within {_SAVE_TIMEOUT_S:.0f}s and "
                    "was forcibly terminated; dataset finalization is unverified"
                ),
            )
        elif self._proc.exitcode:
            remember(
                "subprocess exit",
                RuntimeError(
                    f"recorder subprocess exited with {self._proc.exitcode}; "
                    "dataset finalization failed or the recorder crashed"
                ),
            )

        conn_closed = False
        try:
            self._conn.close()
        except BaseException as error:
            remember("pipe close", error)
        else:
            conn_closed = True
        snap_closed = False
        try:
            self._snap.close()
        except BaseException as error:
            remember("snapshot shared memory close", error)
        else:
            snap_closed = True

        self._closed = not process_alive and conn_closed and snap_closed
        if primary_error is not None:
            raise primary_error
