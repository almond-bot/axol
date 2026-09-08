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
commands. Normally the recorder pulls GDP-wrapped H.264 from the relay via
``shmsrc ! gdpdepay``: the relay exports both compressed access units and their
sensor-exposure PTS entirely in GStreamer C threads, so its WebRTC send keeps the
GIL. If that stack is absent, a :class:`RawFrameReader` carries raw frames from
the relay's Python pull loop instead. In both cases a timestamped snapshot ring
lets the recorder select the joint/action sample nearest each camera exposure,
and the recorder owns the ``LeRobotDataset`` end to end.

When the video relay is unavailable (no gst stack — a degraded, non-Jetson path),
:class:`InProcessRecorder` keeps the old behavior: dataset + capture thread in
the control process. Both expose the same interface, so the control loop is
single-path; only the construction differs.
"""

from __future__ import annotations

import contextlib
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import platform
import shutil
import stat
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Callable

from ..utils.state_files import UnsafeStatePathError

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
_CAPTURE_STOP_TIMEOUT_S = 10.0

# --- Encoded (relay-side H.264) capture-loop tuning ---
# How long the first row waits for each camera's first access unit (shared valve
# open + all-intra NVENC + shmsrc delivery). If a camera produces nothing in
# this window its dataset branch never came up, so the episode is aborted.
_ENCODED_START_TIMEOUT_S = 15.0
# Row-wide budget shared by all cameras. A timeout aborts the episode rather
# than fabricating an exposure. Shared (not per-camera) so serial reads cannot
# compound the wait.
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


# A relay/source hiccup occasionally omits one or two all-intra AUs even while
# the rest of the camera/control stack remains healthy.  Holding the last IDR
# for those missing cadence slots is bounded corruption (at most 33 ms at
# 60 Hz) and, unlike accepting the future AU early, does not shift that camera
# against every subsequent robot-state/dataset row.  The ZED sources have been
# observed dropping one exposure every one to four seconds during a
# teleoperated take (both stereo eyes together, so the loss is in the source,
# not the relay's queues or transport), and a fixed handful of events per
# episode aborted every take within seconds and homed the arms mid-teleop.
# The budget below is therefore a rate: a burst of events inside a short
# window, or concealed frames above a small fraction of the episode, marks the
# stream as unhealthy.  Under the fail-open policy (see _CAMERA_LOSS_FATAL_S)
# that no longer ends the take — the hole is still repaired — but the event is
# logged at ERROR and counted so the source gets looked at.  Holes of any
# length up to _ENCODED_MAX_CONCEALED_GAP_S are repaired the same way; only a
# hole longer than that ends the episode.
_ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW = 3
_ENCODED_CONCEALMENT_WINDOW_S = 2.0
_ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA = 6
_ENCODED_MAX_CONCEALED_FRACTION = 0.02
# A just-arrived exposure can precede the newest control snapshot by one control
# tick. Briefly poll the state ring for the upper bracket; never clamp outside
# its retained range.
_SNAPSHOT_BRACKET_TIMEOUT_S = 0.100

# Match the cross-process snapshot ring's retention for the no-relay fallback.
# At the 120 Hz control rate this keeps a little over four seconds of state,
# comfortably covering camera delivery/encode latency and short scheduler stalls.
_SNAPSHOT_HISTORY_SIZE = 512
# Nearest control samples should normally be within half a 120 Hz tick. Leave
# headroom for lower configured control rates, but surface clock/history failures
# before they silently contaminate a whole episode.
_STATE_ALIGNMENT_WARN_S = 0.050

# Fail-open recording policy. A take is a scarce, operator-driven artefact; a
# single imperfect row is not. Both capture loops therefore *mitigate* every
# per-row defect they can (repeat the prior IDR into a bounded hole, discard a
# stale AU and take the next one, hold the previous row's state when the exact
# bracket is gone, and as a last resort drop the row for every camera at once
# so the streams stay index-aligned) and record it in the
# episode's capture-quality counters (logged at the end of the take and, for
# the worst cases, per event) so it can be tracked down later — without ending
# the episode. Only a *drastic* loss ends a take: a camera that delivers no
# usable exposure for this long, or a control loop that publishes no state for
# this long. Those match the "silent motor" threshold the realtime core uses
# to go limp — past that point the rows would describe a robot nobody is
# observing.
_CAMERA_LOSS_FATAL_S = 1.0
_STATE_LOSS_FATAL_S = 1.0
# Concealment past the historical per-camera budget is still applied (the
# alternative is discarding the take), but the log escalates from WARNING to
# ERROR once per camera per episode so an unhealthy source stands out.
_ENCODED_MAX_CONCEALED_GAP_S = _CAMERA_LOSS_FATAL_S

# A capture row that goes this long without a heartbeat is a stall worth a
# stack trace: 30 frame periods at 60 fps, well under the 2 s encoded-AU
# backlog limit and the 1 s row deadline that would otherwise end the episode
# without saying what the capture thread was doing.
_CAPTURE_ROW_STALL_S = 0.5


class EpisodeDurabilityError(Exception):
    """A saved episode could not be made crash-durable.

    This is deliberately not a ``RuntimeError``: collection loops may recover
    from ordinary per-episode validation failures by discarding the take and
    continuing, but a failed parquet durability flush leaves dataset ownership
    uncertain until orderly finalization. The recorder is terminal after this
    error and may only be closed.
    """


class RecorderDatasetSaveError(RuntimeError):
    """A dataset save failed after its irreversible commit phase began.

    The recorder/session must stop: LeRobot may already have appended parquet
    rows or advanced writer indices, and clearing the in-memory episode buffer
    cannot make a subsequent save safe.
    """


class RecorderCaptureError(RuntimeError):
    """An episode was rejected before commit because capture lost integrity.

    The capture thread has stopped and the in-memory episode buffer has been
    cleared before this is raised, so the caller may safely retry the same
    episode.  Recorder lifecycle, IPC, mux preparation, and dataset-writer
    failures deliberately use other exception paths and remain session-fatal.
    """


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

    Each input's demuxed packet count must match its container's nonzero sample
    count (moov ``stsz``). A mismatch is an integrity failure: duplicating a
    compressed packet would fabricate an exposure and only make a desynchronized
    episode look structurally valid.

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
                    for packet in src.demux(in_stream):
                        if packet.dts is None:  # demux flushing packet
                            continue
                        if packet.pts != packet.dts:
                            # B-frames: demux order is not display order, so
                            # index-based re-stamping would scramble frames.
                            # The probe only samples leading packets; this is
                            # the full-stream check.
                            raise _BFramesDetected(str(input_path))
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
                    if expected and expected != demuxed:
                        raise RuntimeError(
                            f"concat input {Path(str(input_path)).name} demuxed "
                            f"{demuxed} of {expected} advertised samples; refusing "
                            "to fabricate a replacement exposure"
                        )
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
    (count x per-frame duration, after requiring any advertised sample count to
    agree with demux) makes ``to - from`` exactly
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
        advertised = int(stream.frames or 0)
        if advertised and advertised != demuxed:
            raise RuntimeError(
                f"{Path(str(video_path)).name} demuxed {demuxed} of "
                f"{advertised} advertised video samples"
            )
        n = demuxed
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


def _prepare_streaming_episode(dataset: Any) -> None:
    """Finalize fallible mux work before LeRobot commits episode row data.

    Other streaming encoders need no special handling. The relay-H264 adapter
    exposes ``prepare_finish_episode`` because LeRobot 0.6 otherwise writes its
    parquet rows before asking the muxer to EOS its mp4.
    """
    writer = getattr(dataset, "writer", None)
    encoder = getattr(writer, "_streaming_encoder", None)
    prepare = getattr(encoder, "prepare_finish_episode", None)
    if prepare is not None:
        prepare()


# ---------------------------------------------------------------------------
# Joint/action snapshot (in-process publisher, mirrors the cross-process one)
# ---------------------------------------------------------------------------


class _SnapshotPublisher:
    """Timestamped in-process history for the no-relay fallback.

    The control loop calls :meth:`write` every tick; the capture thread reads the
    nearest state for a camera exposure via :meth:`read_nearest`. The bounded
    history mirrors :class:`~almond_axol.video.shm_frames.SnapshotReader`, while
    :meth:`read_latest` preserves the episode-start freshness check. Both return
    ``None`` before the first write.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: deque[tuple[dict, dict, float, bool]] = deque(
            maxlen=_SNAPSHOT_HISTORY_SIZE
        )

    def write(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        # The shared-memory publisher serializes values immediately. Copy here
        # too so retaining history never observes a caller mutating/reusing a
        # dict on a later control tick.
        snap = (dict(joint_obs), dict(action), ts, intervention)
        with self._lock:
            self._history.append(snap)

    def read_latest(self) -> tuple[dict, dict, float, bool] | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def read_nearest(self, target_ts: float) -> tuple[dict, dict, float, bool] | None:
        """Return the retained snapshot closest to ``target_ts``.

        The ring is tiny (512 references), so the linear scan is cheaper and
        simpler than maintaining a second timestamp index. Exact ties prefer the
        later control sample, matching the cross-process reader.
        """
        with self._lock:
            if not self._history:
                return None
            if target_ts < self._history[0][2] or target_ts > self._history[-1][2]:
                return None
            return min(
                self._history,
                key=lambda snap: (abs(snap[2] - target_ts), -snap[2]),
            )


# ---------------------------------------------------------------------------
# Capture loop (shared by both recorders, runs on its own thread)
# ---------------------------------------------------------------------------


def _snapshot_nearest(
    target_ts: float,
    read_latest: Callable[[], tuple[dict, dict, float, bool] | None],
    read_nearest: (Callable[[float], tuple[dict, dict, float, bool] | None] | None),
) -> tuple[dict, dict, float, bool] | None:
    """Read state nearest a camera exposure, with a compatibility fallback."""
    if read_nearest is not None:
        # Preserve a transient/out-of-range miss so the caller can wait for a
        # future bracket or reject an exposure that fell out of history.
        return read_nearest(target_ts)
    return read_latest()


def _wait_snapshot_nearest(
    target_ts: float,
    read_latest: Callable[[], tuple[dict, dict, float, bool] | None],
    read_nearest: (Callable[[float], tuple[dict, dict, float, bool] | None] | None),
    stop_event: threading.Event,
) -> tuple[dict, dict, float, bool] | None:
    """Wait briefly for a state sample that brackets ``target_ts``."""
    deadline = time.perf_counter() + _SNAPSHOT_BRACKET_TIMEOUT_S
    while not stop_event.is_set():
        snap = _snapshot_nearest(target_ts, read_latest, read_nearest)
        if snap is not None:
            return snap
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        stop_event.wait(min(0.002, remaining))
    return None


def _describe_snapshot_miss(
    target_ts: float,
    read_latest: Callable[[], tuple[dict, dict, float, bool] | None],
    now: float | None = None,
) -> str:
    """Say *why* no robot-state snapshot brackets ``target_ts``.

    The bracket wait fails for two unrelated reasons that used to share one
    message. Comparing the exposure against the newest published snapshot
    tells them apart:

    * the newest snapshot is *older* than the exposure — the control loop
      stopped publishing state (a stall in the control process, not the
      recorder), and no amount of recorder history could have helped;
    * the newest snapshot is *newer* than the exposure — the exposure aged out
      of the retained history, i.e. the recorder itself fell behind.
    """
    latest = read_latest()
    if latest is None:
        return "no robot-state snapshot has been published yet"
    latest_ts = latest[2]
    if latest_ts < target_ts:
        age_ms = (target_ts - latest_ts) * 1e3
        since_ms = ((now if now is not None else time.perf_counter()) - latest_ts) * 1e3
        return (
            "the control loop stopped publishing robot state: newest snapshot is "
            f"{age_ms:.0f} ms older than the exposure ({since_ms:.0f} ms old now) "
            "(control-process stall, not a recorder backlog)"
        )
    behind_ms = (latest_ts - target_ts) * 1e3
    return (
        f"the exposure is {behind_ms:.0f} ms behind the newest snapshot and has "
        "aged out of the retained state history (recorder fell behind)"
    )


def _classify_snapshot_miss(
    target_ts: float,
    read_latest: Callable[[], tuple[dict, dict, float, bool] | None],
    now: float | None = None,
) -> tuple[str, bool, tuple[dict, dict, float, bool] | None]:
    """Describe a bracket miss and decide whether it ends the take.

    Returns ``(detail, fatal, latest)`` from a *single* ``read_latest`` so the
    logged attribution and the decision can never disagree (the writer may
    publish between two reads).

    Only one miss is drastic enough to end an episode: the control loop has
    published nothing for :data:`_STATE_LOSS_FATAL_S` — the robot is moving
    (or not) with nobody recording its state, and every later row would pair a
    fresh image with the same stale pose. Everything else is a per-row defect:
    the exposure aged out of the ring while the recorder was descheduled, the
    ring copy raced the writer for every retry, or the writer paused briefly
    and has since resumed. Those cost one row (or one row's exact pairing),
    not the take.
    """
    latest = read_latest()
    detail = _describe_snapshot_miss(target_ts, lambda: latest, now)
    if latest is None:
        # Nothing readable right now. Mid-episode this is a transient (a raced
        # ring copy); the startup wait already proved the writer publishes.
        return detail, False, None
    reference = now if now is not None else time.perf_counter()
    silent_for = reference - latest[2]
    fatal = silent_for >= _STATE_LOSS_FATAL_S
    if fatal:
        detail += (
            f"; no robot state has been published for {silent_for:.2f}s "
            f"(limit {_STATE_LOSS_FATAL_S:.1f}s)"
        )
    return detail, fatal, latest


class _RowStatePairer:
    """Choose the robot state for one dataset row under the fail-open policy.

    The normal case is the snapshot nearest the row's camera exposure, within
    :data:`_STATE_ALIGNMENT_WARN_S`. When that is unavailable (the exact
    bracket aged out of the ring, the ring copy raced the writer, the writer
    paused briefly) the fallbacks are, in order:

    1. **Hold the previous row's state** if it is still within the same
       tolerance of this exposure. That is at most one or two frames stale —
       no worse than a pairing the loop accepts on any ordinary row — and, like
       repeating the prior IDR for a lost exposure, it keeps the row so the
       dataset's ``frame_index / fps`` timeline stays true to wall time.
    2. **Drop the row** (every camera together, so mp4s and parquet stay
       index-aligned) when nothing within tolerance exists and the transport
       allows skipping an AU.
    3. **Keep the row with the degraded pairing** on a predictive stream that
       cannot skip an AU.

    Each outcome is counted in ``quality`` and logged with the row's reason so
    it can be tracked down later.
    """

    def __init__(
        self, *, label: str, can_drop: bool, quality: "dict[str, int] | None"
    ) -> None:
        self._label = label
        self._can_drop = can_drop
        self._quality = quality
        self._last_fresh: tuple[dict, dict, float, bool] | None = None
        self._held_run = 0
        self.misses = 0

    def resolve(
        self,
        row_capture_ts: float,
        candidate: tuple[dict, dict, float, bool] | None,
        miss_detail: str | None,
    ) -> tuple[tuple[dict, dict, float, bool] | None, float]:
        """Return ``(snapshot, skew_s)``; ``snapshot`` is ``None`` to drop the row.

        ``candidate`` is the best snapshot available (the nearest bracket, or
        the newest published when no bracket exists); ``miss_detail`` carries
        :func:`_classify_snapshot_miss`'s attribution when the bracket was
        missing.
        """
        skew = abs(candidate[2] - row_capture_ts) if candidate is not None else math.inf
        if skew <= _STATE_ALIGNMENT_WARN_S:
            self._last_fresh = candidate
            self._held_run = 0
            return candidate, skew
        self.misses += 1
        if miss_detail is not None:
            reason = (
                f"no retained robot-state snapshot brackets {self._label} "
                f"{row_capture_ts:.6f}: {miss_detail}"
            )
        else:
            reason = (
                f"nearest robot state is {skew * 1e3:.1f}ms from the {self._label} "
                f"(limit {_STATE_ALIGNMENT_WARN_S * 1e3:.1f}ms)"
            )
        if self._last_fresh is not None:
            held_skew = abs(self._last_fresh[2] - row_capture_ts)
            if held_skew <= _STATE_ALIGNMENT_WARN_S:
                self._held_run += 1
                _note_quality(self._quality, "rows_with_held_state")
                _logger.warning(
                    "holding the previous row's robot state for this dataset row "
                    "(%.1fms from the %s, %d consecutive): %s (%d state pairing "
                    "miss(es) this episode)",
                    held_skew * 1e3,
                    self._label,
                    self._held_run,
                    reason,
                    self.misses,
                )
                return self._last_fresh, held_skew
        if self._can_drop or candidate is None:
            key = (
                "rows_dropped_state_miss"
                if miss_detail is not None
                else "rows_dropped_state_skew"
            )
            _note_quality(self._quality, key)
            _logger.warning(
                "dropping one dataset row: %s (%d state pairing miss(es) this episode)",
                reason,
                self.misses,
            )
            return None, skew
        _note_quality(self._quality, "rows_kept_with_state_skew")
        _logger.warning(
            "keeping dataset row with a %.1fms state skew (a predictive stream "
            "cannot skip the row): %s (%d state pairing miss(es) this episode)",
            skew * 1e3,
            reason,
            self.misses,
        )
        return candidate, skew


def _note_quality(quality: "dict[str, int] | None", key: str, n: int = 1) -> int:
    """Bump one capture-quality counter; returns the new value (0 if unused)."""
    if quality is None:
        return 0
    value = quality.get(key, 0) + n
    quality[key] = value
    return value


def format_capture_quality(quality: "dict[str, int]") -> str:
    """One-line summary of an episode's mitigated capture defects."""
    if not quality:
        return "clean (no mitigated capture defects)"
    return ", ".join(f"{key}={value}" for key, value in sorted(quality.items()))


def _log_capture_quality(
    quality: "dict[str, int]",
    rows: int,
    cameras: "dict[str, Any]",
    capture_error: str | None,
) -> None:
    """Summarize a finished take's mitigated defects in one log line.

    Folds in each encoded reader's transport-side losses (upstream frames the
    relay dropped before the recorder saw them) so the summary covers the
    whole path. INFO for a clean take, WARNING when anything was mitigated —
    the operator did not see any of it, so this line is how a take that needs
    a second look gets found later.
    """
    for cam_key, cam in cameras.items():
        lost = getattr(cam, "lost_exposures", 0)
        if isinstance(lost, int) and lost > 0:
            quality[f"{cam_key}.transport_lost_exposures"] = lost
    summary = format_capture_quality(quality)
    if capture_error is not None:
        _logger.warning(
            "capture quality: %d row(s) before the take ended on %s; %s",
            rows,
            capture_error,
            summary,
        )
    elif quality:
        _logger.warning("capture quality: %d row(s); %s", rows, summary)
    else:
        _logger.info("capture quality: %d row(s); %s", rows, summary)


def _camera_alignment_limit(fps: int) -> float:
    """Maximum allowed exposure spread within a multi-camera dataset row."""
    return max(0.010, 1.5 / fps) if fps > 0 else 0.050


def _missing_cadence_slots(
    *,
    previous_ts: float,
    capture_ts: float,
    fps: int,
    capture_fps: int,
) -> int:
    """Number of dataset cadence slots skipped between two consecutive AUs.

    Zero for a normal step (including the alternating source-frame spacings a
    non-divisor decimation such as 60 -> 50 legitimately produces: its largest
    planned spacing plus a quarter source interval is allowed). Otherwise the
    gap is rounded to the nearest whole number of dataset periods — source PTS
    jitter is far below half a period, so a real loss is never mistaken for
    jitter or vice versa — and the count of slots with no exposure is
    returned. Callers decide how to repair the hole; this helper only measures
    it.
    """
    if fps <= 0 or not math.isfinite(previous_ts) or not math.isfinite(capture_ts):
        return 0
    delta = capture_ts - previous_ts
    if delta <= 0:
        return 0
    if capture_fps > fps:
        planned_steps = (capture_fps + fps - 1) // fps
        gap_limit = (planned_steps + 0.25) / capture_fps
        if delta <= gap_limit:
            return 0
    periods = int(math.floor(delta * fps + 0.5))
    return max(0, periods - 1)


def _encoded_cadence_deviation(
    *,
    first_ts: float,
    capture_ts: float,
    intervals: int,
    fps: int,
    capture_fps: int,
) -> str | None:
    """Describe a long-run cadence drift off the dataset grid, else ``None``.

    videorate preserves the selected source PTS. Its phase may differ from an
    ideal dataset-rate grid by roughly one source interval, but sustained
    capture-rate output (for example after reopening a stale downstream
    videorate) must not be silently stretched onto the dataset timeline. The
    caller logs the deviation and re-anchors the check; it is not fatal.
    """
    elapsed = capture_ts - first_ts
    cadence_slack = 1.5 / capture_fps
    minimum_elapsed = intervals / fps - cadence_slack
    maximum_elapsed = intervals / fps + cadence_slack
    if minimum_elapsed <= elapsed <= maximum_elapsed:
        return None
    direction = "too fast" if elapsed < minimum_elapsed else "too slow"
    return (
        f"{direction}: {intervals + 1} frames span {elapsed * 1e3:.2f}ms at "
        f"requested {fps}fps (allowed {max(0.0, minimum_elapsed) * 1e3:.2f}–"
        f"{maximum_elapsed * 1e3:.2f}ms)"
    )


def _concealment_within_budget(
    *,
    event_rows: list[int],
    row: int,
    concealed_frames: int,
    missing: int,
    total_rows: int,
    fps: int,
) -> bool:
    """Decide whether one more repair keeps this camera inside its rate budget.

    ``event_rows`` holds the dataset rows of this camera's earlier repairs in
    this episode. A burst (too many events inside a short window) or a stream
    whose concealed frames exceed a small fraction of the episode so far is
    unhealthy and must fail closed; an isolated hiccup every few seconds is
    repaired and audited.
    """
    window_rows = max(1, int(round(_ENCODED_CONCEALMENT_WINDOW_S * fps)))
    recent = sum(1 for prior in event_rows if row - prior < window_rows)
    if recent >= _ENCODED_MAX_CONCEALMENT_EVENTS_PER_WINDOW:
        return False
    allowance = max(
        _ENCODED_MIN_CONCEALED_FRAMES_PER_CAMERA,
        int(_ENCODED_MAX_CONCEALED_FRACTION * total_rows),
    )
    return concealed_frames + missing <= allowance


def _align_independent_encoded_start(
    packets: dict[str, tuple[bytes, float, float]],
    read_next: Callable[[str], tuple[bytes, float, float] | None],
    *,
    fps: int,
    capture_fps: dict[str, int],
) -> tuple[dict[str, tuple[bytes, float, float]], dict[str, int]]:
    """Advance lagging all-intra streams to one fixed row-zero boundary.

    The shared raw-valve target is the primary synchronization barrier. A
    dataset input queue can nevertheless shed a leading exposure, and NVENC can
    deliver a stale in-flight tail after a prior close. Because the dataset
    stream is explicitly all-intra, advancing a lagging camera here is safe: the
    retained AU is still a self-contained frame-zero. The newest initial
    exposure is a fixed boundary; moving it on overshoot could chase forever
    when unsynchronized sensors have different phases. Predictive streams must
    never call this helper.
    """
    dropped: dict[str, int] = {}
    if len(packets) <= 1:
        return packets, dropped
    boundary = max(packet[1] for packet in packets.values())
    for name in packets:
        while packets[name][1] < boundary:
            previous_ts = packets[name][1]
            packet = read_next(name)
            if packet is None:
                raise TimeoutError(
                    f"camera {name!r} did not catch up to the row-zero exposure window"
                )
            if not math.isfinite(packet[1]) or packet[1] <= previous_ts:
                # Unusable prefix AU (no/duplicate timestamp): nothing before
                # row zero is recorded, so just keep advancing.
                dropped[name] = dropped.get(name, 0) + 1
                continue
            missing = _missing_cadence_slots(
                previous_ts=previous_ts,
                capture_ts=packet[1],
                fps=fps,
                capture_fps=capture_fps[name],
            )
            if missing:
                # A hole in the discarded prefix costs nothing; note it so a
                # camera that is already losing exposures at start is visible.
                _logger.info(
                    "camera %r skipped %d exposure(s) before row zero (PTS gap "
                    "%.2fms); nothing recorded yet, continuing alignment",
                    name,
                    missing,
                    1e3 * (packet[1] - previous_ts),
                )
            packets[name] = packet
            dropped[name] = dropped.get(name, 0) + 1
    return packets, dropped


def _warn_timestamp_skew(
    label: str, state_skew_s: float, camera_skew_s: float, fps: int
) -> bool:
    """Warn once when an episode's physical timestamp alignment is implausible."""
    camera_limit_s = _camera_alignment_limit(fps)
    if state_skew_s <= _STATE_ALIGNMENT_WARN_S and camera_skew_s <= camera_limit_s:
        return False
    _logger.warning(
        "%s timestamp alignment degraded: nearest state skew %.1fms "
        "(limit %.1fms), fresh-camera spread %.1fms (limit %.1fms). "
        "Check the patched ZED source, PTS transport, and camera health.",
        label,
        1e3 * state_skew_s,
        1e3 * _STATE_ALIGNMENT_WARN_S,
        1e3 * camera_skew_s,
        1e3 * camera_limit_s,
    )
    return True


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
    heartbeat: Callable[[], None] | None = None,
    row_times: "list[float] | None" = None,
    quality: "dict[str, int] | None" = None,
) -> None:
    """Capture dataset rows at ``fps`` Hz until ``stop_event`` is set.

    ``row_times`` (optional) receives one capture-time append per appended row
    (the row's camera exposure time on the ``perf_counter`` timeline) so the
    recorder can cut a take at an operator gesture (see
    :func:`trim_episode_after`).

    ``heartbeat`` (optional) is called once per tick from this thread so a
    :class:`~almond_axol.utils.stall_diag.StallWatchdog` can attribute a
    stalled row (see :func:`run_encoded_capture_loop`).

    Each tick sleeps until ``T_n = recording_start + n/fps`` and waits for a
    frame with ``capture_perf_ts >= T_n`` from every camera. The row is paired
    with the joint/action snapshot nearest the median camera exposure, rather
    than whichever control tick happens to be latest after camera delivery.
    Every row requires one fresh frame per camera. Under the fail-open policy
    (see :data:`_CAMERA_LOSS_FATAL_S`) a tick whose frames are missing, stale,
    or unsynchronized is *skipped* — no image and no row is written for any
    camera, so the dataset stays self-consistent — and a tick without an exact
    state bracket holds the previous row's state while that is within
    tolerance (:class:`_RowStatePairer`), else is skipped too. Everything is
    counted in ``quality``. Only a camera that produces no fresh frame
    for :data:`_CAMERA_LOSS_FATAL_S`, a transport failure, or a control loop
    silent for :data:`_STATE_LOSS_FATAL_S` ends the episode (reported via
    ``on_error``).

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
        # The snapshot history persists across episodes, so its newest record
        # can still be the previous episode's final control tick (the control
        # loop may not have published again yet); pairing fresh camera frames
        # with it would write a stale pose into the opening rows. Timestamps are
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
                raise RuntimeError(
                    "capture loop saw no fresh robot-state snapshot within 5s"
                )
        if stop_event.is_set():
            return

        frame_interval = 1.0 / fps
        timeout_ms = int(2 * frame_interval * 1000 + 200)
        recording_start: float | None = None
        last_capture_ts: dict[str, float] = {}
        # When each camera last produced a usable fresh frame (perf_counter):
        # a camera silent for _CAMERA_LOSS_FATAL_S ends the take; anything
        # shorter costs the ticks it missed.
        last_fresh_frame_at: dict[str, float] = dict.fromkeys(
            cameras, time.perf_counter()
        )
        tick = 0

        tick_cost_sum = 0.0
        frames_added = 0
        ticks_window = 0
        snapshot_skew_sum = 0.0
        snapshot_skew_max = 0.0
        camera_skew_max = 0.0
        skew_warning_issued = False
        # The raw path writes each row's images itself, so skipping a tick is
        # always alignment-safe.
        state_pairer = _RowStatePairer(
            label="raw camera exposure", can_drop=True, quality=quality
        )
        cap_last_log = time.perf_counter()

        while not stop_event.is_set():
            if heartbeat is not None:
                heartbeat()
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
                    "capture: %.1f fps  tick=%.1fms  added=%d "
                    "state-skew=%.2fms avg/%.2fms max "
                    "camera-skew=%.2fms max",
                    ticks_window / dt,
                    1e3 * tick_cost_sum / ticks_window if ticks_window else 0.0,
                    frames_added,
                    1e3 * snapshot_skew_sum / frames_added if frames_added else 0.0,
                    1e3 * snapshot_skew_max,
                    1e3 * camera_skew_max,
                )
                if not skew_warning_issued:
                    skew_warning_issued = _warn_timestamp_skew(
                        "raw capture", snapshot_skew_max, camera_skew_max, fps
                    )
                tick_cost_sum = 0.0
                frames_added = 0
                ticks_window = 0
                snapshot_skew_sum = 0.0
                snapshot_skew_max = 0.0
                camera_skew_max = 0.0
                cap_last_log = now

            target_perf_ts = recording_start + tick * frame_interval
            wait_s = target_perf_ts - time.perf_counter()
            if wait_s > 0 and stop_event.wait(timeout=wait_s):
                return

            body_t0 = time.perf_counter()
            frames: dict[str, tuple[Any, float, float]] = {}
            capture_ts: list[float] = []
            # Set when this tick cannot become a row. The tick is skipped for
            # every camera at once (no image, no row), which keeps the dataset
            # self-consistent; the reason is logged and counted.
            skip_tick: tuple[str, str] | None = None
            for cam_key, cam in cameras.items():
                try:
                    frame, cap_ts, recv_ts = cam.read_at_or_after(
                        target_perf_ts, timeout_ms=timeout_ms
                    )
                except TimeoutError as exc:
                    silent_for = time.perf_counter() - last_fresh_frame_at[cam_key]
                    if silent_for >= _CAMERA_LOSS_FATAL_S:
                        raise RuntimeError(
                            f"camera {cam_key!r} produced no fresh frame for "
                            f"{silent_for:.2f}s (limit {_CAMERA_LOSS_FATAL_S:.1f}s, "
                            f"last miss at tick {tick}: {exc}); episode discarded"
                        ) from exc
                    skip_tick = (
                        f"{cam_key}.ticks_without_frame",
                        f"camera {cam_key!r} produced no fresh frame for tick "
                        f"{tick} ({exc}; silent for {silent_for * 1e3:.0f}ms)",
                    )
                    break
                except RuntimeError as exc:
                    # The frame transport itself failed (not a late frame).
                    raise RuntimeError(
                        f"camera {cam_key!r} frame transport failed at tick "
                        f"{tick}: {exc}; episode discarded"
                    ) from exc
                if not np.isfinite(cap_ts):
                    skip_tick = (
                        f"{cam_key}.ticks_with_invalid_timestamp",
                        f"camera {cam_key!r} produced an invalid capture "
                        f"timestamp at tick {tick}",
                    )
                    break
                previous = last_capture_ts.get(cam_key)
                if previous is not None and cap_ts <= previous:
                    skip_tick = (
                        f"{cam_key}.ticks_with_stale_frame",
                        f"camera {cam_key!r} repeated a stale frame at tick {tick} "
                        f"(PTS delta {(cap_ts - previous) * 1e3:.2f}ms)",
                    )
                    break
                last_fresh_frame_at[cam_key] = time.perf_counter()
                frames[cam_key] = (frame, cap_ts, recv_ts)
                capture_ts.append(cap_ts)
                last_capture_ts[cam_key] = cap_ts

            if skip_tick is None and capture_ts:
                camera_skew = max(capture_ts) - min(capture_ts)
                camera_skew_max = max(camera_skew_max, camera_skew)
                camera_limit = _camera_alignment_limit(fps)
                if camera_skew > camera_limit:
                    skip_tick = (
                        "ticks_with_camera_skew",
                        "raw camera exposures are not synchronized "
                        f"(spread {camera_skew * 1e3:.1f}ms, limit "
                        f"{camera_limit * 1e3:.1f}ms) at tick {tick}",
                    )
            if skip_tick is not None:
                key, reason = skip_tick
                skipped = _note_quality(quality, key)
                _logger.warning(
                    "skipping one dataset tick: %s (%d such tick(s) this episode)",
                    reason,
                    skipped,
                )
                tick += 1
                continue

            if capture_ts:
                row_capture_ts = float(median(capture_ts))
            else:
                # Joint-only datasets have no exposure clock; retain their
                # original scheduled-tick association.
                row_capture_ts = target_perf_ts
            snap = _wait_snapshot_nearest(
                row_capture_ts,
                read_snapshot,
                read_snapshot_nearest,
                stop_event,
            )
            miss_detail: str | None = None
            if snap is None:
                if stop_event.is_set():
                    return
                miss_detail, fatal, snap = _classify_snapshot_miss(
                    row_capture_ts, read_snapshot
                )
                if fatal:
                    raise RuntimeError(
                        "no retained robot-state snapshot brackets raw camera "
                        f"exposure {row_capture_ts:.6f}: {miss_detail}; episode "
                        "discarded"
                    )
            # A missing or distant bracket is a transient: hold the previous
            # row's state, or skip this one tick — never the take. See
            # _RowStatePairer.
            snap, snapshot_skew = state_pairer.resolve(
                row_capture_ts, snap, miss_detail
            )
            if snap is None:
                tick += 1
                continue
            joint_obs, action, _snap_ts, intervention = snap
            snapshot_skew_sum += snapshot_skew
            snapshot_skew_max = max(snapshot_skew_max, snapshot_skew)

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
            row = {**obs_frame, **act_frame, "task": task}
            if tag_intervention:
                row["intervention"] = np.array([intervention], dtype=bool)
            dataset.add_frame(row)
            if frame_counter is not None:
                frame_counter["n"] += 1
            if row_times is not None:
                row_times.append(row_capture_ts)
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
    repair_events: "list[dict[str, Any]] | None" = None,
    on_error: Callable[[str], None] | None = None,
    on_armed: Callable[[], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    row_times: "list[float] | None" = None,
    quality: "dict[str, int] | None" = None,
) -> None:
    """Frame-driven capture for the relay-encoded (gstshm-h264) transport.

    ``heartbeat`` (optional) is called once per row from this thread so a
    :class:`~almond_axol.utils.stall_diag.StallWatchdog` can name what the
    loop was doing if a row ever takes far longer than a frame period.

    ``frame_counter`` and ``row_times`` mirror :func:`run_capture_loop`'s (a
    mutable ``{"n": int}`` incremented per appended row, and one capture-time
    append per row). There is no ``record_event``
    on this path: capture rows remain continuous within an episode even though
    each all-intra AU is independently decodable.

    Unlike :func:`run_capture_loop` (real-time paced, *selecting* the camera
    frame nearest each tick), this loop is driven by the **arrival** of access
    units: after a one-time row-zero alignment it consumes exactly one AU per
    camera per dataset row. GDP preserves each AU's sensor-exposure PTS; the row
    is paired with the joint/action snapshot nearest the median camera exposure,
    independent of encoder, shm, and scheduler latency. The blocking per-camera
    read naturally paces the loop to the dataset cadence.

    Fail-open policy (see :data:`_CAMERA_LOSS_FATAL_S`): per-row defects are
    mitigated and counted in ``quality``, never fatal. A hole in an all-intra
    camera's exposures is concealed by repeating the prior IDR on the missing
    cadence slots while the future AU is held (preserving every later
    image/camera/state alignment; audited through ``repair_events``); past the
    per-camera rate budget the repair still happens but is logged at ERROR. A
    stale or duplicate AU is discarded and the next one taken. A camera that
    lags its peers by a frame is advanced to them. A row whose exact
    robot-state bracket is gone holds the previous row's state while that is
    still within the alignment tolerance (:class:`_RowStatePairer`), keeping
    the ``frame_index / fps`` timeline true. A row whose exposures still do
    not line up, or with no state within tolerance, is dropped for *every*
    camera at once (the mp4s and the parquet stay index-aligned) — or, on a
    predictive stream that cannot skip an AU, kept with the degraded pairing.
    What does end the take: a camera with no usable exposure
    for :data:`_CAMERA_LOSS_FATAL_S`, a hole that long, a control loop silent
    for :data:`_STATE_LOSS_FATAL_S`, or a transport/encoder failure.

    The muxer assigns each AU a constant-fps PTS (``k / fps``), so the mp4
    timeline is exact regardless of arrival jitter; its physical exposure PTS is
    used only for state/action pairing and integrity checks. The first delivered
    AU per camera is always an IDR (:meth:`EncodedAuReader.flush` re-arms
    keyframe-wait), so each episode's mp4 is decodable from frame 0.
    """
    try:
        import numpy as np
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.visualization_utils import log_rerun_data

        tag_intervention = "intervention" in dataset.features

        # Flush before the relay valve opens. Arming the cutoff first guarantees
        # that a newly admitted all-intra AU survives into row zero instead of
        # being cleared by a racing flush.
        split_flush = [
            cam
            for cam in cameras.values()
            if hasattr(cam, "begin_flush") and hasattr(cam, "finish_flush")
        ]
        for cam in split_flush:
            cam.begin_flush()
        for cam in cameras.values():
            if cam not in split_flush:
                cam.flush()
        for cam in split_flush:
            cam.finish_flush()
        if on_armed is not None:
            on_armed()

        # Wait for the first snapshot *published after this episode started* —
        # the history persists across episodes, so a stale previous-episode
        # snapshot must not seed the opening rows (see run_capture_loop).
        episode_start = time.perf_counter()
        first_deadline = episode_start + 5.0
        last_snap = read_snapshot()
        while last_snap is None or last_snap[2] < episode_start:
            if stop_event.wait(0.02):
                return
            if time.perf_counter() > first_deadline:
                raise RuntimeError(
                    "encoded capture saw no fresh robot-state snapshot within 5s"
                )
            last_snap = read_snapshot()
        if stop_event.is_set():
            return

        def read_au(cam: Any, deadline: float) -> tuple[bytes, float, float] | None:
            """Pop the next AU by ``deadline``, waking every poll for stop_event.

            Once the deadline has passed, still makes one non-blocking attempt:
            a camera whose AU is already queued must advance even when an
            earlier camera consumed the whole shared row budget.
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

        previous_capture_ts: dict[str, float] = {}
        previous_packets: dict[str, tuple[bytes, float, float]] = {}
        first_capture_ts: dict[str, float] = {}
        capture_intervals: dict[str, int] = {}

        def read_usable_au(
            cam_key: str, cam: Any, deadline: float
        ) -> tuple[bytes, float, float] | None:
            """Pop AUs until one carries a finite exposure newer than the last.

            An AU whose PTS is not finite, or did not advance past the previous
            exposure (a duplicated or reordered frame), cannot be placed on the
            row grid. It is discarded and counted, and the next AU is taken;
            ``deadline`` bounds the retries, so a camera that only produces
            unusable AUs still ends the take through the row timeout.
            """
            previous = previous_capture_ts.get(cam_key)
            while True:
                packet = read_au(cam, deadline)
                if packet is None:
                    return None
                cap_ts = packet[1]
                if np.isfinite(cap_ts) and (previous is None or cap_ts > previous):
                    return packet
                skipped = _note_quality(quality, f"{cam_key}.unusable_aus_skipped")
                if not np.isfinite(cap_ts):
                    why = "is not a finite timestamp"
                else:
                    why = (
                        "did not advance past the previous exposure "
                        f"({(cap_ts - previous) * 1e3:+.2f}ms)"
                    )
                _logger.warning(
                    "camera %r: discarding an AU whose exposure %s; taking the "
                    "next one (%d unusable AU(s) skipped this episode)",
                    cam_key,
                    why,
                    skipped,
                )

        def reanchor_cadence(cam_key: str) -> None:
            """Restart the long-run cadence check after an out-of-band skip."""
            first_capture_ts.pop(cam_key, None)
            capture_intervals.pop(cam_key, None)

        # Dropping a whole row (every camera's AU together) keeps the mp4s and
        # the parquet index-aligned only if no stream needs that AU to decode
        # the next one. Predictive streams instead keep the row and degrade
        # the state pairing.
        row_drop_is_safe = all(
            bool(getattr(cam, "frames_are_independent", False))
            for cam in cameras.values()
        )
        # A detected future AU stays here until every missing logical cadence
        # slot has emitted the previous independently-decodable IDR.  Reading
        # peers continues normally, so the existing cross-camera validator
        # proves every repaired row is still synchronized.
        held_packets: dict[
            str, tuple[tuple[bytes, float, float], int, dict[str, Any]]
        ] = {}
        concealed_frames: dict[str, int] = {}
        concealment_rows: dict[str, list[int]] = {}
        primed = False
        rows_added = 0
        total_rows = 0
        max_pending = 0
        snapshot_skew_sum = 0.0
        snapshot_skew_max = 0.0
        camera_skew_max = 0.0
        state_pairer = _RowStatePairer(
            label="camera exposure", can_drop=row_drop_is_safe, quality=quality
        )
        last_log = time.perf_counter()

        while not stop_event.is_set():
            if heartbeat is not None:
                heartbeat()
            budget = _ENCODED_START_TIMEOUT_S if not primed else _ENCODED_ROW_TIMEOUT_S
            # One shared deadline for the whole row: with per-camera budgets the
            # serial reads compound (a stalled first camera would hand every
            # later camera an extra full budget of implicit wait). A row-wide
            # deadline keeps all cameras on the same clock; read_au's final
            # non-blocking attempt still accepts an AU already queued.
            row_deadline = time.perf_counter() + budget
            packets: dict[str, tuple[bytes, float, float]] = {}
            synthetic_repairs: dict[str, dict[str, Any]] = {}
            for cam_key, cam in cameras.items():
                held = held_packets.get(cam_key)
                if held is not None:
                    future, remaining, event = held
                    if remaining > 0:
                        previous = previous_packets[cam_key]
                        packet = (
                            previous[0],
                            previous[1] + 1.0 / fps,
                            future[2],
                        )
                        held_packets[cam_key] = (future, remaining - 1, event)
                        synthetic_repairs[cam_key] = event
                    else:
                        packet = future
                        del held_packets[cam_key]
                else:
                    packet = read_usable_au(cam_key, cam, row_deadline)
                if packet is None:
                    if stop_event.is_set():
                        return
                    phase = "startup" if not primed else "recording"
                    raise RuntimeError(
                        f"camera {cam_key!r} produced no fresh encoded frame "
                        f"within {budget:.1f}s during {phase}; episode discarded"
                    )
                packets[cam_key] = packet
                pending = cam.pending
                if pending > max_pending:
                    max_pending = pending

            if stop_event.is_set():
                return

            # Trust but verify the raw-valve barrier using the timestamps that
            # actually reached the recorder. A bounded leaky input queue or a
            # stale NVENC tail can make one first AU later/earlier even though
            # every valve crossed the same target. All-intra makes it safe to
            # advance only the lagging streams until their first retained
            # exposures form one valid camera cluster.
            if not primed and len(packets) > 1:
                start_times = [packet[1] for packet in packets.values()]
                camera_limit = _camera_alignment_limit(fps)
                if max(start_times) - min(start_times) > camera_limit:
                    non_independent = [
                        name
                        for name, cam in cameras.items()
                        if not bool(getattr(cam, "frames_are_independent", False))
                    ]
                    if non_independent:
                        raise RuntimeError(
                            "cannot align encoded row zero by dropping predictive "
                            "frames from " + ", ".join(sorted(non_independent))
                        )
                    try:
                        packets, startup_drops = _align_independent_encoded_start(
                            packets,
                            lambda name: read_au(cameras[name], row_deadline),
                            fps=fps,
                            capture_fps={
                                name: max(
                                    fps,
                                    int(getattr(camera, "capture_fps", fps)),
                                )
                                for name, camera in cameras.items()
                            },
                        )
                    except TimeoutError as exc:
                        raise RuntimeError(f"{exc}; episode discarded") from exc
                    if startup_drops:
                        _logger.info(
                            "aligned encoded row zero by discarding independent "
                            "startup AUs: %s",
                            ", ".join(
                                f"{name}={count}"
                                for name, count in sorted(startup_drops.items())
                            ),
                        )
                    for cam in cameras.values():
                        max_pending = max(max_pending, cam.pending)

            # Do not accept a future AU early: mux re-stamping would shift this
            # camera against every later row.  For the one explicitly bounded
            # all-IDR case, hold it and fill the missing logical exposure slots
            # from the prior IDR.  This decision belongs here (with all cameras
            # visible), not inside an individual EncodedAuReader.
            if primed:
                for cam_key, packet in tuple(packets.items()):
                    if cam_key in held_packets:
                        continue  # already emitting a planned synthetic slot
                    previous = previous_packets[cam_key]
                    camera = cameras[cam_key]
                    capture_fps = int(getattr(camera, "capture_fps", fps))
                    missing = _missing_cadence_slots(
                        previous_ts=previous[1],
                        capture_ts=packet[1],
                        fps=fps,
                        capture_fps=capture_fps,
                    )
                    if missing == 0:
                        continue
                    gap_s = packet[1] - previous[1]
                    if gap_s > _ENCODED_MAX_CONCEALED_GAP_S:
                        # Drastic: the camera produced nothing for as long as
                        # the row timeout tolerates; the rows in between would
                        # be a frozen image over a moving robot.
                        raise RuntimeError(
                            f"camera {cam_key!r} delivered no exposure for "
                            f"{gap_s:.2f}s (limit "
                            f"{_ENCODED_MAX_CONCEALED_GAP_S:.1f}s): capture PTS "
                            f"jumped {gap_s * 1e3:.1f}ms at dataset row "
                            f"{total_rows}; episode discarded"
                        )
                    if not bool(getattr(camera, "frames_are_independent", False)):
                        # A predictive AU can neither be repeated nor skipped
                        # without breaking the decode chain: accept the hole
                        # and say so. This camera now leads the row grid.
                        unrepaired = _note_quality(
                            quality, f"{cam_key}.unrepaired_missing_frames", missing
                        )
                        _logger.error(
                            "camera %r lost %d exposure(s) at dataset row %d "
                            "(PTS gap %.2fms) on a predictive stream that cannot "
                            "be repaired; its video now leads the row grid by "
                            "%d frame(s) for the rest of the episode",
                            cam_key,
                            missing,
                            total_rows,
                            1e3 * gap_s,
                            unrepaired,
                        )
                        continue
                    already_concealed = concealed_frames.get(cam_key, 0)
                    prior_rows = concealment_rows.setdefault(cam_key, [])
                    within_budget = _concealment_within_budget(
                        event_rows=prior_rows,
                        row=total_rows,
                        concealed_frames=already_concealed,
                        missing=missing,
                        total_rows=total_rows,
                        fps=fps,
                    )
                    event: dict[str, Any] = {
                        "camera": cam_key,
                        "frame_index": total_rows,
                        # Updated only after each synthetic dataset row commits.
                        # This keeps a save that lands midway through a two-row
                        # repair honest, and records nothing if add_frame fails.
                        "missing_frames": 0,
                        "gap_ms": 1e3 * gap_s,
                        "concealed_ms": 0.0,
                        "method": "repeat_previous_idr",
                        "within_budget": within_budget,
                    }
                    held_packets[cam_key] = (packet, missing - 1, event)
                    packets[cam_key] = (
                        previous[0],
                        previous[1] + 1.0 / fps,
                        packet[2],
                    )
                    synthetic_repairs[cam_key] = event
                    concealed_frames[cam_key] = already_concealed + missing
                    prior_rows.append(total_rows)
                    _note_quality(quality, f"{cam_key}.concealed_frames", missing)
                    if within_budget:
                        _logger.warning(
                            "camera %r omitted %d all-intra frame(s) at dataset "
                            "row %d (PTS gap %.2fms); repeating the prior IDR for "
                            "%.2fms and holding the future AU to preserve camera/"
                            "state alignment",
                            cam_key,
                            missing,
                            total_rows,
                            event["gap_ms"],
                            1e3 * missing / fps,
                        )
                    else:
                        # Past the rate budget the source is unhealthy. The
                        # fail-open policy still repairs the hole rather than
                        # discarding the take, but says so at ERROR (once per
                        # camera per episode, then WARNING) so it gets fixed.
                        over = _note_quality(
                            quality, f"{cam_key}.concealment_over_budget_events"
                        )
                        _logger.log(
                            logging.ERROR if over == 1 else logging.WARNING,
                            "camera %r is losing exposures faster than a rare "
                            "hiccup: %d more frame(s) omitted at dataset row %d "
                            "(PTS gap %.2fms, %d concealed so far this episode, "
                            "%d over-budget event(s)); repeating the prior IDR "
                            "anyway under the fail-open policy — check the "
                            "camera source",
                            cam_key,
                            missing,
                            total_rows,
                            event["gap_ms"],
                            already_concealed + missing,
                            over,
                        )

            camera_limit = _camera_alignment_limit(fps)
            if primed and row_drop_is_safe and len(packets) > 1:
                # A camera whose exposure trails its peers by more than the
                # alignment limit has an extra AU in its stream (a source clock
                # running fast, or a startup phase the row-zero alignment did
                # not see). On an all-intra stream the surplus AU can simply be
                # skipped: advance the camera until it rejoins the row.
                newest = max(packet[1] for packet in packets.values())
                for cam_key, packet in tuple(packets.items()):
                    if cam_key in held_packets:
                        continue  # a planned synthetic slot; never advance it
                    while packet[1] < newest - camera_limit:
                        candidate = read_au(cameras[cam_key], row_deadline)
                        if candidate is None:
                            break  # nothing queued: the skew check decides
                        if not np.isfinite(candidate[1]) or candidate[1] <= packet[1]:
                            _note_quality(quality, f"{cam_key}.unusable_aus_skipped")
                            continue
                        skipped = _note_quality(
                            quality, f"{cam_key}.exposures_skipped_to_realign"
                        )
                        _logger.warning(
                            "camera %r exposure trails the row by %.2fms at "
                            "dataset row %d; skipping it and taking the next AU "
                            "(%d skipped to realign this episode)",
                            cam_key,
                            1e3 * (newest - packet[1]),
                            total_rows,
                            skipped,
                        )
                        packet = candidate
                        packets[cam_key] = packet
                        reanchor_cadence(cam_key)

            aus = {name: packet[0] for name, packet in packets.items()}
            capture_ts = {name: packet[1] for name, packet in packets.items()}
            primed = True

            for cam_key, cap_ts in capture_ts.items():
                previous = previous_capture_ts.get(cam_key)
                capture_fps = max(
                    fps, int(getattr(cameras[cam_key], "capture_fps", fps))
                )
                first_ts = first_capture_ts.get(cam_key)
                if previous is not None and first_ts is not None:
                    intervals = capture_intervals[cam_key] + 1
                    deviation = _encoded_cadence_deviation(
                        first_ts=first_ts,
                        capture_ts=cap_ts,
                        intervals=intervals,
                        fps=fps,
                        capture_fps=capture_fps,
                    )
                    if deviation is None:
                        capture_intervals[cam_key] = intervals
                    else:
                        # The long-run cadence drifted off the dataset grid
                        # (e.g. a decimated source briefly delivering at its
                        # capture rate). Not fatal: log, count, and re-anchor
                        # so the check keeps watching from here.
                        reanchors = _note_quality(
                            quality, f"{cam_key}.cadence_reanchors"
                        )
                        _logger.warning(
                            "camera %r encoded cadence is %s at dataset row %d; "
                            "re-anchoring the cadence check (%d re-anchor(s) "
                            "this episode)",
                            cam_key,
                            deviation,
                            total_rows,
                            reanchors,
                        )
                        first_capture_ts[cam_key] = cap_ts
                        capture_intervals[cam_key] = 0
                else:
                    first_capture_ts[cam_key] = cap_ts
                    capture_intervals[cam_key] = 0
                previous_capture_ts[cam_key] = cap_ts
                previous_packets[cam_key] = packets[cam_key]

            exposure_times = list(capture_ts.values())
            row_capture_ts = float(median(exposure_times))
            camera_skew = (
                max(exposure_times) - min(exposure_times)
                if len(exposure_times) > 1
                else 0.0
            )
            camera_skew_max = max(camera_skew_max, camera_skew)
            if camera_skew > camera_limit:
                detail = ", ".join(
                    f"{name}={1e3 * (ts - row_capture_ts):+.1f}ms"
                    for name, ts in sorted(capture_ts.items())
                )
                if row_drop_is_safe:
                    # The exposures still do not form one row after the
                    # realignment above. Pairing them with a single state
                    # would be wrong for at least one camera; dropping every
                    # camera's AU together costs one row and keeps the mp4s
                    # and parquet index-aligned.
                    dropped = _note_quality(quality, "rows_dropped_camera_skew")
                    _logger.warning(
                        "dropping one dataset row: camera exposures are not "
                        "synchronized (spread %.1fms, limit %.1fms; %s; %d "
                        "row(s) dropped for skew this episode)",
                        camera_skew * 1e3,
                        camera_limit * 1e3,
                        detail,
                        dropped,
                    )
                    continue
                # A predictive stream needs this AU muxed; keep the row and
                # accept the skew (the state pairs with the median exposure).
                accepted = _note_quality(quality, "rows_kept_with_camera_skew")
                _logger.error(
                    "camera exposures are not synchronized (spread %.1fms, "
                    "limit %.1fms; %s) and a predictive stream cannot skip the "
                    "row; keeping it (%d skewed row(s) kept this episode)",
                    camera_skew * 1e3,
                    camera_limit * 1e3,
                    detail,
                    accepted,
                )

            snap = _wait_snapshot_nearest(
                row_capture_ts,
                read_snapshot,
                read_snapshot_nearest,
                stop_event,
            )
            miss_detail: str | None = None
            if snap is None:
                if stop_event.is_set():
                    return
                miss_detail, fatal, snap = _classify_snapshot_miss(
                    row_capture_ts, read_snapshot
                )
                if fatal:
                    raise RuntimeError(
                        "no retained robot-state snapshot brackets camera "
                        f"exposure {row_capture_ts:.6f}: {miss_detail}; episode "
                        "discarded"
                    )
            # A missing or distant bracket is a transient (the exposure aged
            # out of the ring while the recorder was descheduled, a raced ring
            # copy, a brief writer pause): hold the previous row's state, or
            # drop this one row — never the take. See _RowStatePairer.
            snap, snapshot_skew = state_pairer.resolve(
                row_capture_ts, snap, miss_detail
            )
            if snap is None:
                continue
            joint_obs, action, _snap_ts, intervention = snap
            snapshot_skew_sum += snapshot_skew
            snapshot_skew_max = max(snapshot_skew_max, snapshot_skew)

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
            row = {**obs_frame, **act_frame, "task": task}
            if tag_intervention:
                row["intervention"] = np.array([intervention], dtype=bool)
            dataset.add_frame(row)
            for event in synthetic_repairs.values():
                if event["missing_frames"] == 0 and repair_events is not None:
                    repair_events.append(event)
                event["missing_frames"] += 1
                event["concealed_ms"] = 1e3 * event["missing_frames"] / fps
            if frame_counter is not None:
                frame_counter["n"] += 1
            if row_times is not None:
                row_times.append(row_capture_ts)
            rows_added += 1
            total_rows += 1

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
                    "encoded capture: %.1f fps  rows(win)=%d backlog=%d "
                    "state-skew=%.2fms avg/%.2fms max "
                    "camera-skew=%.2fms max",
                    rows_added / dt,
                    rows_added,
                    max_pending,
                    1e3 * snapshot_skew_sum / rows_added if rows_added else 0.0,
                    1e3 * snapshot_skew_max,
                    1e3 * camera_skew_max,
                )
                rows_added = 0
                max_pending = 0
                snapshot_skew_sum = 0.0
                snapshot_skew_max = 0.0
                camera_skew_max = 0.0
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


def trim_episode_after(
    dataset: "LeRobotDataset", row_times: list[float], cutoff: float
) -> int:
    """Drop the buffered rows captured after ``cutoff``; return how many.

    Used when the operator ends a take with the trigger x3 gesture: the
    bridge reports when the first click *began* (host ``perf_counter``), and
    every row captured from that instant on shows the clicks — three gripper
    close/open commands and the hand motion that goes with them — rather than
    the task. ``row_times`` is the per-row capture-time list the capture loop
    filled (see ``run_capture_loop``); the cut lands before the first row whose
    capture time is past ``cutoff``, so a late row can never survive on the
    strength of an earlier neighbour.

    Only the row buffer is cut. The streamed video already holds the trailing
    frames, and that is fine: LeRobot indexes each row's frame at
    ``from_timestamp + k / fps`` and takes the episode's video span from the
    file's real duration, so the extra pictures are simply never referenced,
    and the next episode's span still starts where this file actually ends.

    Call only after the capture thread has been stopped (the buffer must be
    quiescent). Mutates ``row_times`` to match.
    """
    total = len(row_times)
    keep = next((i for i, t in enumerate(row_times) if t > cutoff), total)
    if keep >= total:
        return 0
    buffer = dataset.writer.episode_buffer
    if buffer is None:
        return 0
    for values in buffer.values():
        if isinstance(values, list):
            del values[keep:]
    buffer["size"] = keep
    del row_times[keep:]
    removed = total - keep
    _logger.info(
        "trimmed %d row(s) captured after the end gesture began (%.2f s at the "
        "end of the take); %d row(s) kept",
        removed,
        removed / max(1, dataset.fps),
        keep,
    )
    return removed


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


_DURABILITY_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
)
_DURABILITY_FILE_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK


def _dataset_relative_file(path: str | Path, *, label: str) -> Path:
    """Validate one metadata-derived path before descriptor-relative use."""
    relative = Path(path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".."} for part in relative.parts)
    ):
        raise UnsafeStatePathError(
            f"refusing unsafe {label} path from dataset metadata: {path!s}"
        )
    return relative


def _episode_durability_files(
    meta: Any, episode_row: dict[str, Any]
) -> tuple[Path, list[Path]]:
    """Resolve every file that makes the just-saved episode readable."""
    # The dataset root is *our* configured location, and operators routinely
    # reach it through a link (``~/datasets -> /mnt/ssd/datasets``, a linked
    # ``/home``). The barrier walks the root's ancestors with O_NOFOLLOW, so a
    # lexical path would hit ELOOP on that link and every save would fail as
    # "not durable". Resolve the root once here; the no-follow discipline then
    # applies to everything *inside* the dataset, which is what the metadata
    # (untrusted-ish, rewritten each save) can influence.
    root = Path(os.path.realpath(Path(meta.root).expanduser()))
    if root == Path(root.anchor):
        raise UnsafeStatePathError("dataset root may not be a filesystem root")

    def relative(path: str | Path, label: str) -> Path:
        return _dataset_relative_file(path, label=label)

    data_chunk = int(episode_row["data/chunk_index"])
    data_file = int(episode_row["data/file_index"])
    meta_chunk = int(episode_row["meta/episodes/chunk_index"])
    meta_file = int(episode_row["meta/episodes/file_index"])
    files = [
        relative(
            meta.data_path.format(chunk_index=data_chunk, file_index=data_file),
            "episode data",
        ),
        relative(
            f"meta/episodes/chunk-{meta_chunk:03d}/file-{meta_file:03d}.parquet",
            "episode metadata",
        ),
    ]

    video_template = getattr(meta, "video_path", None)
    for key in episode_row:
        if not (key.startswith("videos/") and key.endswith("/chunk_index")):
            continue
        if not video_template:
            raise UnsafeStatePathError(
                "episode metadata names videos but the dataset has no video path"
            )
        video_key = key[len("videos/") : -len("/chunk_index")]
        chunk_index = int(episode_row[key])
        file_index = int(episode_row[f"videos/{video_key}/file_index"])
        files.append(
            relative(
                video_template.format(
                    video_key=video_key,
                    chunk_index=chunk_index,
                    file_index=file_index,
                ),
                f"episode video {video_key!r}",
            )
        )

    # save_episode always rewrites info + aggregate stats. tasks.parquet is
    # part of the episode's referential integrity whenever it names a task;
    # syncing it every save is harmless and also covers a newly-added task.
    files.extend(
        [
            relative("meta/info.json", "dataset info"),
            relative("meta/stats.json", "dataset stats"),
        ]
    )
    if episode_row.get("tasks"):
        files.append(relative("meta/tasks.parquet", "dataset tasks"))

    # A malformed metadata template must not make one path masquerade as two
    # required pieces. Keep stable ordering for deterministic barriers/tests.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path in seen:
            raise UnsafeStatePathError(
                f"dataset metadata aliases multiple required files at {path}"
            )
        seen.add(path)
        unique.append(path)
    return root, unique


def _fsync_episode_files(meta: Any, episode_row: dict[str, Any]) -> None:
    """Power-loss barrier for an episode, without following dataset links.

    All payload and metadata files are fsynced first. Their containing
    directories (plus the lexical ancestors that may have been created with a
    fresh dataset) are then fsynced deepest-first, which persists every file
    and directory entry before a save acknowledgement can escape.
    """
    root, files = _episode_durability_files(meta, episode_row)
    directory_fds: dict[tuple[str, ...], int] = {}
    directory_entries: list[
        tuple[tuple[str, ...], str, tuple[str, ...], tuple[int, int]]
    ] = []
    file_entries: list[tuple[tuple[str, ...], str, tuple[int, int]]] = []

    try:
        root_fd = os.open(root.anchor, _DURABILITY_DIRECTORY_FLAGS)
        directory_fds[()] = root_fd
        current: tuple[str, ...] = ()
        for component in root.parts[1:]:
            child_key = (*current, component)
            child_fd = os.open(
                component,
                _DURABILITY_DIRECTORY_FLAGS,
                dir_fd=directory_fds[current],
            )
            child_stat = os.fstat(child_fd)
            directory_fds[child_key] = child_fd
            directory_entries.append(
                (
                    current,
                    component,
                    child_key,
                    (child_stat.st_dev, child_stat.st_ino),
                )
            )
            current = child_key
        root_key = current

        for relative in files:
            parent_key = root_key
            for component in relative.parent.parts:
                child_key = (*parent_key, component)
                if child_key not in directory_fds:
                    child_fd = os.open(
                        component,
                        _DURABILITY_DIRECTORY_FLAGS,
                        dir_fd=directory_fds[parent_key],
                    )
                    child_stat = os.fstat(child_fd)
                    directory_fds[child_key] = child_fd
                    directory_entries.append(
                        (
                            parent_key,
                            component,
                            child_key,
                            (child_stat.st_dev, child_stat.st_ino),
                        )
                    )
                parent_key = child_key

            name = relative.name
            entry = os.stat(
                name,
                dir_fd=directory_fds[parent_key],
                follow_symlinks=False,
            )
            if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
                raise UnsafeStatePathError(
                    f"refusing non-regular or hard-linked dataset file: {relative}"
                )
            descriptor = os.open(
                name,
                _DURABILITY_FILE_FLAGS,
                dir_fd=directory_fds[parent_key],
            )
            try:
                opened = os.fstat(descriptor)
                identity = (opened.st_dev, opened.st_ino)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or identity != (entry.st_dev, entry.st_ino)
                ):
                    raise UnsafeStatePathError(
                        f"dataset file changed while opening it: {relative}"
                    )
                os.fsync(descriptor)
                file_entries.append((parent_key, name, identity))
            finally:
                os.close(descriptor)

        # File contents must reach stable storage before their names. Fsync all
        # traversed directories deepest-first; ``/`` itself is not affected.
        for key in sorted(directory_fds, key=len, reverse=True):
            if key:
                os.fsync(directory_fds[key])

        # Detect substitutions during the barrier while every ancestor is
        # still pinned. A symlink or different inode can never be acknowledged.
        for parent_key, name, identity in file_entries:
            current_entry = os.stat(
                name,
                dir_fd=directory_fds[parent_key],
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(current_entry.st_mode)
                or current_entry.st_nlink != 1
                or (current_entry.st_dev, current_entry.st_ino) != identity
            ):
                raise UnsafeStatePathError(
                    f"dataset file changed during durability barrier: {name}"
                )
        for parent_key, name, child_key, identity in directory_entries:
            current_entry = os.stat(
                name,
                dir_fd=directory_fds[parent_key],
                follow_symlinks=False,
            )
            opened_directory = os.fstat(directory_fds[child_key])
            if (
                not stat.S_ISDIR(current_entry.st_mode)
                or (current_entry.st_dev, current_entry.st_ino) != identity
                or (opened_directory.st_dev, opened_directory.st_ino) != identity
            ):
                raise UnsafeStatePathError(
                    f"dataset directory changed during durability barrier: {name}"
                )
    finally:
        for descriptor in reversed(tuple(directory_fds.values())):
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _discard_episode_buffer(dataset: Any) -> str | None:
    """Drop the buffered episode; return an error string instead of raising.

    ``clear_episode_buffer`` also cancels the streaming encoder's in-flight
    episode, and the H.264 muxer's ``cancel_episode`` raises when a camera's
    pipeline refuses to stop or its staging cannot be removed. In the recorder
    subprocess that exception would escape the command loop and take the whole
    recorder (and session) down over a discard of data nobody wanted -- so
    log it and hand the caller a message to attach to its reply instead. The
    encoder keeps the cameras it could not cancel and retries on the next
    ``start_episode``.
    """
    try:
        dataset.clear_episode_buffer()
    except Exception as error:  # noqa: BLE001 - keep the recorder alive
        _logger.exception("discarding the buffered episode failed")
        return f"episode discard failed: {type(error).__name__}: {error}"
    return None


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
        # Closing a writer only reaches the kernel's page cache. Explicitly
        # persist every file that makes this episode readable, then all of the
        # directory entries that name them, before returning a saved ack.
        _fsync_episode_files(meta, last)
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
    once.

    The decode only runs **between** episodes. A software re-decode of four
    60 fps files costs more CPU time than the episode took to record (~100
    core-seconds for a three-minute take), all of it on the two background
    cores the recorder's capture thread and the relay's CFS encoder-feed
    threads share. Measured 2026-09-03: with the verifier decoding episode 1
    throughout episode 2, ``left_arm``'s NVENC input queue overran four times
    in six seconds and the take was discarded — while episode 1, recorded with
    the verifier idle, lost nothing. ``nice 19`` does not fix that (it only
    yields to CFS peers, and the load also contends for memory bandwidth and
    the GIL), so the owner calls :meth:`suspend` when an episode starts and
    :meth:`resume` once its capture thread has stopped; the decode picks up
    where it left off, and :meth:`close` lifts the gate before its bounded
    wait so shutdown never blocks on it.
    """

    def __init__(self, dataset_root: "Path | str") -> None:
        import queue

        self._root = Path(dataset_root)
        self._queue: "queue.Queue[list | None]" = queue.Queue()
        # Set while no episode is recording; the decode loop waits on it
        # between frames, so a running verify parks within one frame of
        # suspend() and continues from the same file on resume().
        self._idle = threading.Event()
        self._idle.set()
        self._pending_files = 0
        self._pending_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="axol-video-verify", daemon=True
        )
        self._thread.start()

    def submit(self, episode_row: dict[str, Any]) -> None:
        paths = _episode_video_paths(self._root, episode_row)
        if paths:
            with self._pending_lock:
                self._pending_files += len(paths)
            self._queue.put(paths)

    @property
    def pending_files(self) -> int:
        """Video files submitted but not yet verified (a backlog indicator)."""
        with self._pending_lock:
            return self._pending_files

    def suspend(self) -> None:
        """Park the decode (call when an episode starts recording)."""
        self._idle.clear()

    def resume(self) -> None:
        """Let the decode continue (call once the capture thread has stopped)."""
        self._idle.set()

    def close(self, timeout: float = 60.0) -> None:
        """Finish the pending verifies (bounded so shutdown can't hang on them).

        Lifts the recording gate first: by the time the owner closes, nothing
        records anymore, and a suspended verifier would otherwise just sit out
        the timeout.
        """
        self.resume()
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            _logger.warning(
                "video integrity: verifier still running after %.0fs; leaving "
                "%d video file(s) of the last episode(s) unverified",
                timeout,
                self.pending_files,
            )

    def _run(self) -> None:
        # Even between episodes this thread (and the libav workers it spawns,
        # which inherit its priority) is CPU-bound on the background cores
        # while a save/return-to-rest is under way; yield to everything else
        # there.
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
        except (AttributeError, OSError):
            pass
        while True:
            paths = self._queue.get()
            if paths is None:
                return
            for mp4 in paths:
                try:
                    packets, decoded = self._probe(mp4, self._idle)
                except Exception as exc:  # noqa: BLE001 - verify is best-effort
                    _logger.warning(
                        "video integrity: could not verify %s: %s", mp4, exc
                    )
                    continue
                finally:
                    with self._pending_lock:
                        self._pending_files -= 1
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
    def _probe(
        mp4: "Path", proceed: "threading.Event | None" = None
    ) -> tuple[int, int]:
        """``(muxed packets, decodable frames)`` of ``mp4``.

        ``proceed`` (when given) is waited on before every decoded frame, so a
        cleared event parks the decode — including libav's worker threads,
        which stall as soon as this pump stops feeding them — until it is set
        again.
        """
        import av

        with av.open(str(mp4)) as container:
            packets = sum(1 for p in container.demux(video=0) if p.pts is not None)
        decoded = 0
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for _ in container.decode(stream):
                decoded += 1
                if proceed is not None and not proceed.is_set():
                    proceed.wait()
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
        # Per-row capture times for the current episode (see trim_episode_after).
        self._row_times: list[float] = []
        # Mitigated per-row defects of the current take (see _CAMERA_LOSS_FATAL_S).
        self._quality: dict[str, int] = {}
        self._capture_error: str | None = None
        self._episodes_recorded = 0
        self._fatal_error: EpisodeDurabilityError | None = None

    def _require_usable(self) -> None:
        fatal_error = getattr(self, "_fatal_error", None)
        if fatal_error is not None:
            raise fatal_error

    def publish(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        self._require_usable()
        self._publisher.write(joint_obs, action, ts, intervention)

    def episode_count(self) -> int:
        return self._dataset.num_episodes

    def start_episode(self, task: str) -> None:
        from ..lerobot.nvenc_encoder import reset_dropped_frames

        self._require_usable()
        self._stop_capture()  # defensive: never overlap two capture threads
        self._dataset.clear_episode_buffer()
        self._record.set()
        self._frames["n"] = 0
        self._row_times.clear()
        self._quality.clear()
        self._capture_error = None
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
                on_error=lambda message: setattr(self, "_capture_error", message),
                row_times=self._row_times,
                quality=self._quality,
            ),
            name="axol-capture",
            daemon=True,
        )
        # Park the previous episode's video verify for the whole take (see
        # _EpisodeVideoVerifier); _stop_capture() resumes it.
        self._verifier.suspend()
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
        # forever on a wedged camera read, and never forget the exact writer
        # until a bounded retry proves it exited (the raise happens before the
        # assignment, so ``_thread`` keeps pointing at the live writer).
        was_capturing = self._thread is not None
        self._thread = _stop_capture_thread(self._thread, self._stop)
        if was_capturing:
            _log_capture_quality(
                self._quality,
                self._frames["n"],
                dict(getattr(self._robot, "cameras", None) or {}),
                self._capture_error,
            )
        # Nothing records now: let the previous episode's verify continue.
        self._verifier.resume()

    def finish_episode(self, trim_after: float | None = None) -> int:
        """Freeze capture without saving and return the exact buffered rows.

        ``trim_after`` (host ``perf_counter`` seconds) additionally drops the
        rows captured after that instant — the operator's end gesture — see
        :func:`trim_episode_after`. The returned row count is post-trim.
        Raises :class:`RecorderCaptureError` (after discarding the buffer) when
        the capture thread failed mid-take.
        """
        self._require_usable()
        self._stop_capture()
        if self._capture_error is not None:
            self._dataset.clear_episode_buffer()
            raise RecorderCaptureError(
                f"recorder capture failed: {self._capture_error}; episode discarded"
            )
        if trim_after is not None:
            trim_episode_after(self._dataset, self._row_times, trim_after)
            self._frames["n"] = len(self._row_times)
        return self._frames["n"]

    def stop_capture(self, trim_after: float | None = None) -> tuple[int, str | None]:
        """Non-raising :meth:`finish_episode`: ``(rows, capture_error)``.

        Session flows that decide on a take *before* asking LeRobot to save it
        (Mantis) consume the capture verdict as a value; a failed capture
        yields ``(0, message)`` with the buffer already discarded.
        """
        try:
            return self.finish_episode(trim_after), None
        except RecorderCaptureError as exc:
            return 0, str(exc)

    def poll_capture_error(self) -> str | None:
        """Return an episode-local capture failure without blocking."""
        return self._capture_error

    def save_episode(self) -> None:
        from ..lerobot.nvenc_encoder import dropped_frames

        self._require_usable()
        self._stop_capture()
        if self._capture_error is not None:
            self._dataset.clear_episode_buffer()
            raise RecorderCaptureError(
                f"recorder capture failed: {self._capture_error}; episode discarded"
            )
        n_dropped = dropped_frames()
        if n_dropped:
            # Mirrors the recorder subprocess: a dropped frame's row still
            # exists, so the video is misaligned — never save it silently.
            self._dataset.clear_episode_buffer()
            raise RecorderCaptureError(
                f"{n_dropped} video frame(s) were dropped by the encoder "
                "(feed queue overflow) — the episode's video is misaligned "
                "with its rows; episode discarded."
            )
        try:
            _maybe_smooth_episode(self._dataset, self._config)
            _prepare_streaming_episode(self._dataset)
        except Exception:
            self._dataset.clear_episode_buffer()
            raise
        self._dataset.save_episode()
        # The write succeeded, so shutdown must preserve/finalize this episode
        # even if the durability step below fails. This is an internal cleanup
        # count, not a success acknowledgement to the caller.
        self._episodes_recorded += 1
        # Flush the episode to disk *before* returning, so "saved" means
        # "survives a kill" (see make_episode_durable). A failure is terminal:
        # later saves could reuse uncertain writer state, so the recorder
        # refuses further work and may only be closed.
        try:
            episode_row = make_episode_durable(self._dataset)
        except Exception as error:
            fatal_error = EpisodeDurabilityError(
                "episode was written but could not be made crash-durable; "
                "recording cannot continue safely"
            )
            self._fatal_error = fatal_error
            _logger.exception(
                "episode durability flush failed; terminating recording so "
                "orderly finalization can recover the dataset"
            )
            raise fatal_error from error
        self._verifier.submit(episode_row)

    def cancel_episode(self) -> None:
        self._require_usable()
        self._stop_capture()
        self._dataset.clear_episode_buffer()
        self._capture_error = None

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
                # close() is also the Ctrl+C / panel-Stop escape hatch. The
                # caller may never have reached finish/cancel, so always tear
                # down a live streaming encoder and discard uncommitted rows
                # before dataset finalization. This is idempotent after
                # save_episode(), which creates a fresh buffer.
                self._dataset.clear_episode_buffer()
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


def _rollback_recorder_startup(
    *,
    cameras: dict[str, Any],
    snap_reader: Any | None,
    dataset: Any | None,
    verifier: Any | None,
    config: dict,
    startup_error: BaseException,
) -> None:
    """Release every resource acquired before the recorder can report ready."""

    def attempt(label: str, close: Callable[[], None]) -> None:
        try:
            close()
        except BaseException as error:
            startup_error.add_note(
                f"recorder startup cleanup failed for {label}: "
                f"{type(error).__name__}: {error}"
            )

    # Detach shmsrc consumers first so a failed recorder launch cannot leave
    # the relay blocked waiting on a reader that will never consume another AU.
    for name, camera in cameras.items():
        attempt(f"camera {name}", camera.close)
    if snap_reader is not None:
        attempt("snapshot reader", snap_reader.close)
    if verifier is not None:
        attempt("video verifier", verifier.close)
    if dataset is not None:
        attempt(
            "dataset finalization",
            lambda: _finalize_dataset(dataset, config, episodes_recorded=0),
        )


def _cleanup_recorder_session(
    *,
    stop_capture: Callable[[], None],
    dataset: Any,
    config: dict,
    episodes_recorded: int,
    verifier: Any,
    cameras: dict[str, Any],
    snap_reader: Any,
    before_finalize: Callable[[], None] | None = None,
) -> None:
    """Finish a recorder session, preserving errors after complete cleanup.

    A capture thread owns the mutable episode buffer, so dataset finalization is
    permitted only after its exit is proven. If its normal stop bound expires,
    close every input reader to unblock a wedged read and retry the join before
    deciding whether finalization is safe. ``before_finalize`` (discarding the
    unsaved episode buffer) runs only once capture has stopped; its failure
    also withholds finalization. All independent resources are then closed
    even when stop/finalize fails; the first failure is re-raised after later
    failures have been attached as notes, making the child exit non-zero for
    :class:`DatasetRecorderProcess.close` to propagate.
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

    finalize_ok = capture_stopped
    if capture_stopped and before_finalize is not None:
        try:
            before_finalize()
        except BaseException as error:
            _logger.exception(
                "recorder failed to discard its unsaved episode; refusing "
                "dataset finalization"
            )
            remember("episode discard", error)
            finalize_ok = False

    if finalize_ok:
        try:
            _finalize_dataset(dataset, config, episodes_recorded)
        except BaseException as error:
            _logger.exception("recorder failed to finalize the dataset")
            remember("dataset finalize", error)
    else:
        # Finalizing concurrently with a live dataset writer (or over an
        # undiscarded partial episode) can corrupt the buffer/files. The
        # retained error is already primary; make the deliberate skip explicit.
        assert primary_error is not None
        primary_error.add_note(
            "dataset finalization was skipped because the capture thread may "
            "still own the episode buffer or the buffer could not be discarded"
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
    error_conn: multiprocessing.connection.Connection,
    raw_cond: Any,
    config: dict,
) -> None:
    """Recorder subprocess entry: own the dataset, capture from shared memory."""
    logging.basicConfig(level=config["log_level"])

    # Keep the recorder (+ its NVENC gst children, which inherit this) off the
    # control loop's cores; fall back to a positive nice where affinity isn't
    # available so it still never preempts the control loop / IK.
    from ..utils import affinity
    from ..utils.stall_diag import GcHold, StallWatchdog, install_gc_pause_logger

    if not affinity.pin_background():
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass

    # Stall attribution. A capture row that stops for seconds ends the episode
    # ("encoded-AU backlog exceeded") without saying what the thread was doing;
    # the watchdog logs its stack + kernel state while it is still stuck, and
    # the gc hook names a stop-the-world collection if that is what paused it.
    # Cyclic GC is held for the whole take (the per-row dicts/arrays are freed
    # by refcount anyway) and swept between episodes, as the relay and
    # run-policy already do for the same measured reason.
    uninstall_gc_log = install_gc_pause_logger(_logger)
    gc_hold = GcHold("recorder take", _logger)
    watchdog: StallWatchdog | None = None

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
    transports = {str(meta["transport"]) for meta in raw_meta.values()}
    # Capture is either access-unit-driven (every H.264 exposure becomes a row)
    # or raw-frame-driven (one independently selectable image per tick). Combining
    # those contracts in one episode would require two pacing models, so reject an
    # unsupported mixed construction explicitly instead of failing later on a
    # missing reader API.
    if "gstshm-h264" in transports and len(transports) != 1:
        raise RuntimeError(
            "recorder camera transports must be uniformly gstshm-h264 or raw; "
            f"got {sorted(transports)}"
        )
    encoded_mode = transports == {"gstshm-h264"}
    if encoded_mode:
        install_encoded_dataset_encoder()
    else:
        install_dataset_encoder()
    _, _, robot_obs_proc = make_default_processors()

    # Build a per-source frame reader matching the relay's chosen transport.
    # gstshm-h264: an EncodedAuReader (shmsrc → gdpdepay → h264parse → appsink)
    # pre-encoded access units in order. gstshm: a shmsrc → appsink consumer
    # pulling raw frames on THIS process's GIL (so the relay's send is never
    # starved). pyshm: the older RawFrameReader over a shared-memory block the
    # relay's Python pull loop fills. Started here (before "ready") and torn down
    # in the finally; the relay's rawvalve gates episode on/off, so the consumers
    # can run continuously and just idle when the valve is closed.
    cameras: dict[str, Any] = {}
    snap_reader: Any | None = None
    dataset: Any | None = None
    verifier: Any | None = None
    try:
        for source, meta in raw_meta.items():
            if meta["transport"] == "gstshm-h264":
                cam = EncodedAuReader(
                    meta["socket_path"],
                    meta["width"],
                    meta["height"],
                    meta["fps"],
                    name=source,
                    pts_perf_offset_s=meta["pts_perf_offset_s"],
                    capture_fps=meta.get("capture_fps", meta["fps"]),
                )
                # Retain ownership before the fallible connect so the outer
                # rollback retries its own constructor-level cleanup if needed.
                cameras[source] = cam
                cam.connect()
            elif meta["transport"] == "gstshm":
                cam = GstShmFrameReader(
                    meta["socket_path"],
                    meta["caps"],
                    meta["width"],
                    meta["height"],
                    meta["fps"],
                    meta["latency_s"],
                )
                cameras[source] = cam
                cam.connect()
            else:
                cameras[source] = RawFrameReader(
                    meta["shm_name"],
                    meta["width"],
                    meta["height"],
                    meta["fps"],
                    raw_cond,
                )
        snap_reader = SnapshotReader(
            config["snapshot_shm_name"],
            config["obs_keys"],
            config["action_keys"],
        )

        if config["rerun_ip"]:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(
                session_name="axol_record",
                ip=config["rerun_ip"],
                port=config["rerun_port"],
            )

        dataset = _open_dataset(config)
        verifier = _EpisodeVideoVerifier(config["dataset_root"])
        conn.send(("ready", dataset.num_episodes))
    except BaseException as startup_error:
        _rollback_recorder_startup(
            cameras=cameras,
            snap_reader=snap_reader,
            dataset=dataset,
            verifier=verifier,
            config=config,
            startup_error=startup_error,
        )
        uninstall_gc_log()
        raise

    # Narrow Optional types after the transactional setup above. Every later
    # path is protected by _cleanup_recorder_session's complete cleanup pass.
    assert snap_reader is not None
    assert dataset is not None
    assert verifier is not None

    thread: threading.Thread | None = None
    stop: threading.Event | None = None
    capture_error: dict[str, str | None] = {"v": None}
    capture_repairs: list[dict[str, Any]] = []
    # Mitigated per-row defects of the current take (see _CAMERA_LOSS_FATAL_S);
    # summarized once when capture stops so a take that needed repairs is easy
    # to find in the session log.
    capture_quality: dict[str, int] = {}
    episodes_recorded = 0
    save_poisoned = False
    # Mid-episode capture gate + row counter (see run_capture_loop). The gate
    # is only supported on the raw transports: pausing the encoded
    # (gstshm-h264) stream mid-episode would drop access units that later
    # P-frames reference, corrupting the mp4.
    record_event = threading.Event()
    frame_counter: dict[str, int] = {"n": 0}
    # Per-row capture times for the current episode (see trim_episode_after).
    row_times: list[float] = []

    def report_capture_error(message: str) -> None:
        """Publish the first capture failure without corrupting command replies."""
        if capture_error["v"] is not None:
            return
        capture_error["v"] = message
        with contextlib.suppress(OSError, EOFError, ValueError):
            error_conn.send(message)

    def stop_capture() -> None:
        nonlocal thread, watchdog
        if watchdog is not None:
            watchdog.stop()
            watchdog = None
        # Assignment happens only after _stop_capture_thread has proved exit.
        # On timeout it raises and leaves ``thread`` pointing at the live
        # writer, which makes every destructive command below fail closed.
        was_capturing = thread is not None
        thread = _stop_capture_thread(thread, stop)
        if was_capturing:
            _log_capture_quality(
                capture_quality, frame_counter["n"], cameras, capture_error["v"]
            )
        # The take is over: sweep the garbage deferred during it now, while
        # nothing time-critical runs in this process.
        gc_hold.end()
        # Nothing records now: let the previous episode's verify continue.
        verifier.resume()

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
                capture_repairs.clear()
                capture_quality.clear()
                for cam in cameras.values():
                    reset_lost = getattr(cam, "reset_lost_exposures", None)
                    if callable(reset_lost):
                        reset_lost()
                record_event.set()
                frame_counter["n"] = 0
                row_times.clear()
                from ..lerobot.nvenc_encoder import reset_dropped_frames

                reset_dropped_frames()
                stop = threading.Event()
                gc_hold.begin()
                watchdog = StallWatchdog(
                    "recorder capture row", _CAPTURE_ROW_STALL_S, logger=_logger
                )
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
                    on_error=report_capture_error,
                    heartbeat=watchdog.beat,
                    row_times=row_times,
                    quality=capture_quality,
                )
                loop_kwargs["frame_counter"] = frame_counter
                if not encoded_mode:
                    loop_kwargs["record_event"] = record_event
                armed = threading.Event()
                if encoded_mode:
                    loop_kwargs["on_armed"] = armed.set
                    loop_kwargs["repair_events"] = capture_repairs
                thread = threading.Thread(
                    target=(
                        run_encoded_capture_loop if encoded_mode else run_capture_loop
                    ),
                    kwargs=loop_kwargs,
                    name="axol-capture",
                    daemon=True,
                )
                # Park the previous episode's video verify for the whole take
                # (see _EpisodeVideoVerifier); stop_capture() resumes it.
                verifier.suspend()
                thread.start()
                if encoded_mode and not armed.wait(2.0):
                    stop.set()
                    thread.join(timeout=2.0)
                    if not thread.is_alive():
                        gc_hold.end()
                        verifier.resume()
                    detail = capture_error["v"] or "encoded readers did not arm"
                    conn.send(("error", detail))
                    continue
                # Arm the watchdog only once the readers are flushed and
                # armed: the pre-arm flush handshake legitimately blocks.
                watchdog.start()
                conn.send(("started",))
            elif kind == "finish_episode":
                trim_after = msg[1] if len(msg) > 1 else None
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                else:
                    finished_capture_error = capture_error["v"]
                    if finished_capture_error is not None:
                        discard_error = _discard_episode_buffer(dataset)
                        if discard_error is not None:
                            finished_capture_error += f" ({discard_error})"
                    elif trim_after is not None:
                        # The take ended on the operator's end gesture: drop
                        # the rows captured after its first click began.
                        trim_episode_after(dataset, row_times, trim_after)
                        frame_counter["n"] = len(row_times)
                    # Carry the post-join result on the command channel.  The
                    # notification pipe makes live polling prompt, but cannot
                    # be the finish authority: its delivery may race this
                    # reply even though the capture thread has already exited.
                    conn.send(("finished", frame_counter["n"], finished_capture_error))
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
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                    continue
                from ..lerobot.nvenc_encoder import dropped_frames

                n_dropped = dropped_frames()
                if capture_error["v"] is not None:
                    dataset.clear_episode_buffer()
                    conn.send(("capture_error", capture_error["v"]))
                elif n_dropped:
                    # A dropped frame was never encoded but its row exists, so
                    # the episode's video is misaligned with its rows (and
                    # with the other cameras) — refuse to save silently
                    # corrupted data. The caller should discard and re-record.
                    dataset.clear_episode_buffer()
                    conn.send(
                        (
                            "capture_error",
                            f"{n_dropped} video frame(s) were dropped by the "
                            "encoder (feed queue overflow) — the episode's "
                            "video is misaligned with its rows; episode "
                            "discarded. Check recorder-core load.",
                        )
                    )
                else:
                    t_save = time.perf_counter()
                    try:
                        # LeRobot 0.6 writes parquet rows before it finalizes
                        # streaming video. Finalize/cache all supported mux
                        # results first so an EOS/count failure cannot leave
                        # orphan row data.
                        _maybe_smooth_episode(dataset, config)
                        _prepare_streaming_episode(dataset)
                    except Exception as exc:  # safe: row commit has not begun
                        _logger.error("recorder video prepare failed: %s", exc)
                        with contextlib.suppress(Exception):
                            dataset.clear_episode_buffer()
                        conn.send(("error", str(exc)))
                        continue
                    try:
                        dataset.save_episode()
                        # The dataset write happened even if the durability
                        # flush below fails. Count it for shutdown recovery so
                        # finalization verifies/preserves the episode instead
                        # of treating a fresh dataset as empty.
                        episodes_recorded += 1
                        # Flush the episode to disk *before* acknowledging the
                        # save, so "saved" means "survives a kill". A failure
                        # is terminal: later saves could reuse uncertain writer
                        # state, so report a distinct fatal reply and unwind
                        # through the recorder's orderly finalizer.
                        try:
                            episode_row = make_episode_durable(dataset)
                        except Exception as error:
                            fatal_error = EpisodeDurabilityError(
                                "episode was written but could not be made "
                                "crash-durable; recording cannot continue safely"
                            )
                            _logger.exception(
                                "episode durability flush failed; terminating "
                                "the recorder so orderly finalization can "
                                "recover the dataset"
                            )
                            save_poisoned = True
                            conn.send(("fatal_durability", str(fatal_error)))
                            raise fatal_error from error
                        _logger.info(
                            "save_episode took %.1fs", time.perf_counter() - t_save
                        )
                        verifier.submit(episode_row)
                        if capture_repairs:
                            _logger.warning(
                                "saved episode %d with camera-gap repair(s): %s",
                                dataset.num_episodes - 1,
                                ", ".join(
                                    f"{event['camera']}@row "
                                    f"{event['frame_index']}="
                                    f"{event['missing_frames']} frame(s)"
                                    + (
                                        ""
                                        if event.get("within_budget", True)
                                        else " (over budget)"
                                    )
                                    for event in capture_repairs
                                ),
                            )
                        capture_repairs.clear()
                        conn.send(("saved", dataset.num_episodes))
                    except EpisodeDurabilityError:
                        # Unwind through the finalizer with a non-zero exit so
                        # the parent's close() also refuses to trust the
                        # dataset, not just the poisoned episode handle.
                        raise
                    except Exception as exc:  # irreversible writer state possible
                        _logger.exception(
                            "recorder dataset save failed after commit began; "
                            "the session cannot safely continue"
                        )
                        save_poisoned = True
                        conn.send(("fatal", str(exc)))
                        break
            elif kind == "cancel_episode":
                try:
                    stop_capture()
                except RuntimeError as exc:
                    conn.send(("error", str(exc)))
                    continue
                capture_repairs.clear()
                discard_error = _discard_episode_buffer(dataset)
                if discard_error is not None:
                    conn.send(("error", discard_error))
                    continue
                conn.send(("cancelled",))
    finally:
        # Never close/finalize the dataset while its capture thread may still
        # be inside add_frame/appsrc. _cleanup_recorder_session bounds the
        # join, closes the readers to wake a wedged read, retries once, and
        # skips finalization (raising) if the writer still cannot be proved
        # gone; the parent's close() then reports a non-zero child exit.
        finalize_config = config
        if save_poisoned:
            # Preserve a fresh first-save failure for inspection/repair and
            # never upload a dataset whose writer indices may be partial.
            finalize_config = {**config, "is_complete": True, "push_to_hub": False}
        try:
            _cleanup_recorder_session(
                stop_capture=stop_capture,
                dataset=dataset,
                config=finalize_config,
                episodes_recorded=episodes_recorded,
                verifier=verifier,
                cameras=cameras,
                snap_reader=snap_reader,
                # EOF, shutdown, and KeyboardInterrupt may all bypass the
                # explicit cancel command. Always discard any uncommitted rows
                # and cancel a live streaming encoder before finalizing; after
                # a successful save this is an idempotent clear of the
                # newly-created empty episode buffer.
                before_finalize=dataset.clear_episode_buffer,
            )
        finally:
            uninstall_gc_log()
            with contextlib.suppress(Exception):
                error_conn.close()


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

        # Lock-free SPSC ring: the control loop's write never waits on the
        # recorder, however long the recorder is descheduled (see
        # shm_frames._SNAP_READ_ATTEMPTS for the history behind that).
        snap = SnapshotWriter(obs_keys, action_keys)
        conn: Any | None = None
        child_conn: Any | None = None
        error_conn: Any | None = None
        child_error_conn: Any | None = None
        proc: Any | None = None
        started = False
        try:
            ctx = multiprocessing.get_context("spawn")
            conn, child_conn = ctx.Pipe()
            # Capture runs on a child thread while command/reply messages use
            # ``conn``. Keep failures on a dedicated one-way pipe so the hot
            # control loop can poll them without stealing an expected command
            # reply.
            error_conn, child_error_conn = ctx.Pipe(duplex=False)
            full_config = {
                **config,
                "raw_meta": raw_meta,
                "obs_keys": obs_keys,
                "action_keys": action_keys,
                "snapshot_shm_name": snap.name,
            }
            proc = ctx.Process(
                target=_recorder_main,
                args=(child_conn, child_error_conn, raw_cond, full_config),
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
            child_error_conn.close()
            if conn.poll(_READY_TIMEOUT_S):
                try:
                    msg = conn.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(
                        "recorder subprocess exited during startup "
                        f"(exit code {proc.exitcode})"
                    ) from exc
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
            if child_error_conn is not None:
                local_cleanups.append(("child error pipe", child_error_conn.close))
            if conn is not None:
                local_cleanups.append(("parent pipe", conn.close))
            if error_conn is not None:
                local_cleanups.append(("parent error pipe", error_conn.close))
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
        assert error_conn is not None
        assert proc is not None
        self._snap = snap
        self._conn = conn
        self._error_conn = error_conn
        self._proc = proc
        self._lock = threading.Lock()
        self._episode_count = episode_count
        self._capture_error: str | None = None
        self._closed = False
        self._fatal_error: EpisodeDurabilityError | None = None

    def _require_usable(self) -> None:
        fatal_error = getattr(self, "_fatal_error", None)
        if fatal_error is not None:
            raise fatal_error

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def publish(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        self._require_usable()
        self._snap.write(joint_obs, action, ts, intervention)

    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, task: str) -> None:
        # Discard a failure already consumed by the previous episode. The child
        # clears its matching local value before it starts this capture thread.
        try:
            while self._error_conn.poll():
                self._error_conn.recv()
        except (EOFError, OSError, ValueError):
            pass
        self._capture_error = None
        self._require_usable()
        with self._lock:
            self._conn.send(("start_episode", task))
            if not self._conn.poll(_CMD_TIMEOUT_S):
                raise RuntimeError("recorder did not start the episode in time")
            msg = self._conn.recv()
        if msg[0] != "started":
            detail = msg[1] if len(msg) > 1 else repr(msg)
            raise RuntimeError(f"recorder start_episode failed: {detail}")

    def finish_episode(self, trim_after: float | None = None) -> int:
        """Freeze capture and return the exact buffered row count.

        Call the moment an episode terminates — before any post-episode robot
        motion (gripper valve close, return-to-rest) — so the recorder's
        capture thread can't keep appending rows that pair a frozen snapshot
        with later camera frames. ``save_episode`` / ``cancel_episode`` remain
        valid afterwards and operate on the buffer as frozen here.

        ``trim_after`` (host ``perf_counter`` seconds) additionally drops the
        rows captured after that instant — the operator's end gesture — see
        :func:`trim_episode_after`. The returned row count is post-trim.
        Raises :class:`RecorderCaptureError` when the capture thread failed
        mid-take (the recorder has already discarded the buffer).
        """
        self._require_usable()
        with self._lock:
            self._conn.send(("finish_episode", trim_after))
            if not self._conn.poll(_CMD_TIMEOUT_S + 1.0):
                raise RuntimeError("recorder did not stop episode capture in time")
            msg = self._conn.recv()
        if not isinstance(msg, tuple) or not msg or msg[0] != "finished":
            detail = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else repr(msg)
            raise RuntimeError(f"recorder finish_episode failed: {detail}")
        if len(msg) != 3:
            raise RuntimeError(
                f"recorder finish_episode sent unexpected reply: {msg!r}"
            )
        capture_error = msg[2]
        if capture_error is not None:
            self._capture_error = str(capture_error)
            raise RecorderCaptureError(
                f"recorder capture failed: {capture_error}; episode discarded"
            )
        return int(msg[1])

    def stop_capture(self, trim_after: float | None = None) -> tuple[int, str | None]:
        """Non-raising :meth:`finish_episode`: ``(rows, capture_error)``.

        Session flows that decide on a take *before* asking LeRobot to save it
        (Mantis) consume the capture verdict as a value; a failed capture
        yields ``(0, message)`` with the buffer already discarded.
        """
        try:
            return self.finish_episode(trim_after), None
        except RecorderCaptureError as exc:
            return 0, str(exc)

    def poll_capture_error(self) -> str | None:
        """Return the first capture-thread failure for this episode, if any."""
        try:
            while self._error_conn.poll():
                self._capture_error = str(self._error_conn.recv())
        except (EOFError, OSError, ValueError):
            pass
        return self._capture_error

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
        self._require_usable()
        with self._lock:
            self._conn.send(("save_episode",))
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not finish save_episode in time")
            msg = self._conn.recv()
            if isinstance(msg, tuple) and msg and msg[0] == "fatal_durability":
                # The episode is on disk but not provably durable; the child
                # is unwinding through its finalizer. Only close() may follow.
                fatal_error = EpisodeDurabilityError(str(msg[1]))
                self._fatal_error = fatal_error
                raise fatal_error
        if not isinstance(msg, tuple) or not msg:
            raise RuntimeError(f"recorder save_episode sent unexpected reply: {msg!r}")
        if msg[0] == "saved":
            self._episode_count = int(msg[1])
        elif msg[0] == "capture_error":
            self._capture_error = str(msg[1])
            raise RecorderCaptureError(
                f"recorder capture failed: {msg[1]}; episode discarded"
            )
        elif msg[0] == "error":
            raise RuntimeError(f"recorder save_episode failed: {msg[1]}")
        elif msg[0] == "fatal":
            raise RecorderDatasetSaveError(
                "recorder dataset save failed after commit began; stop the "
                f"session and inspect the dataset before resuming: {msg[1]}"
            )
        else:
            raise RuntimeError(f"recorder save_episode sent unexpected reply: {msg!r}")

    def cancel_episode(self) -> None:
        self._require_usable()
        with self._lock:
            self._conn.send(("cancel_episode",))
            if not self._conn.poll(_SAVE_TIMEOUT_S):
                raise RuntimeError("recorder did not cancel the episode in time")
            msg = self._conn.recv()
        if msg[0] != "cancelled":
            detail = msg[1] if len(msg) > 1 else repr(msg)
            raise RuntimeError(f"recorder cancel_episode failed: {detail}")

    def close(self) -> None:
        """Shut the recorder down; raise if the dataset cannot be trusted.

        A child that had to be terminated (SIGTERM: it dies where it stands,
        without running the finally that finalizes the dataset), or one that
        died on its own (a crash, or the OOM killer) never finalized either.
        Until this raised, join() returning instantly on an already-dead
        process was indistinguishable from a clean shutdown, and the session
        went on to validate and upload an unreadable dataset. Local cleanup
        still runs to completion; ``_closed`` records whether it all succeeded
        so a retry re-attempts only what did not.
        """
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
                    "was forcibly terminated; dataset finalization is unverified "
                    "and its parquet files are likely unreadable"
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

        local_ok = True
        for label, closer in (
            ("pipe close", self._conn.close),
            ("error pipe close", self._error_conn.close),
            ("snapshot shared memory close", self._snap.close),
        ):
            try:
                closer()
            except BaseException as error:
                local_ok = False
                remember(label, error)

        self._closed = not process_alive and local_ok
        if primary_error is not None:
            raise primary_error
