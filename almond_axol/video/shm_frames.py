"""Shared-memory transport for raw camera frames across the relay boundary.

``collect-data`` needs the ZED cameras' raw frames in the **control** process
(to write the dataset), but running the camera grab + NVENC encode + aiortc
WebRTC in that process starves the teleop/IK loops (see
:mod:`almond_axol.video.video_proc`). The relay subprocess therefore owns the
cameras and does all the heavy work; this module ships the raw frames it produces
back to the recorder process through shared memory — NV12 on the gst-native
``shmsink``/``shmsrc`` transport (:class:`GstShmFrameReader`), or RGB on the
``multiprocessing`` fallback (:class:`RawFrameReader`) — so the recorder only ever
copies a frame out of shared memory at the 60 Hz capture rate while recording,
never on the hot control path.

Layout (one :class:`SharedMemory` block per camera source):

    [ meta: seq, slot, cap_ts, recv_ts ][ buffer 0 ][ buffer 1 ]

The two frame buffers are double-buffered: the writer always fills the buffer
the reader isn't pointed at, then publishes the new ``slot`` + timestamps under
a shared :class:`multiprocessing.Condition` and notifies. A reader copies out of
the published slot *outside* the lock; double-buffering guarantees the writer
won't reuse that slot for a full extra frame (~16 ms at 60 fps), far longer than
a ~1 ms 6 MB copy, and a post-copy sequence recheck retries on the rare overlap.

Timestamps are ``time.perf_counter`` seconds. On Linux that is
``CLOCK_MONOTONIC``, which shares an origin across processes, so a ``cap_ts``
stamped in the relay subprocess stays directly comparable to the joint-sample
timestamps taken in the control process — preserving the image/joint alignment
the dataset relies on.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Meta header: a single structured record at the front of each block. Padded to
# 64 bytes so the frame buffers start cache-line aligned.
_META_DTYPE = np.dtype(
    [("seq", "<i8"), ("slot", "<i8"), ("cap_ts", "<f8"), ("recv_ts", "<f8")]
)
_HEADER_BYTES = 64

# Frames are RGB (3 channels): the VIC delivers RGBA, the writer drops alpha so
# only what the dataset stores crosses the boundary.
_CHANNELS = 3

# Snapshot channel header: a single int64 sequence counter, padded to 16 bytes so
# the float64 payload that follows stays 8-byte aligned.
_SNAP_META_DTYPE = np.dtype([("seq", "<i8")])
_SNAP_HEADER_BYTES = 16

# A healthy local camera/encode/shared-memory path is measured in milliseconds.
# Treat anything above this generous ceiling as corrupt metadata: subtracting an
# arbitrary large value can select the oldest entry in the finite snapshot ring.
_MAX_CAPTURE_LATENCY_S = 5.0
# A pipeline/perf-clock co-sample can differ by a few microseconds. Keep a
# generous bound for scheduler jitter, but never let a recovered timestamp make
# dataset pairing meaningfully future-dated.
_MAX_CAPTURE_FUTURE_S = 0.05
_GST_CLOCK_TIME_NONE = (1 << 64) - 1
# The snapshot ring holds about 0.5 s at the 120 Hz control rate. Abort an
# encoded take before its oldest undelivered AU can age out of that history;
# silently dropping an AU is not an option because later H.264 pictures may
# reference it. This also puts a hard ceiling on memory if row processing
# stalls while the GStreamer pull thread keeps draining shmsrc.
_MAX_ENCODED_AU_BACKLOG_S = 0.25
_GST_READER_STOP_TIMEOUT_S = 2.0


def _set_gst_state_checked(pipeline: Any, gst: Any, state: Any, *, label: str) -> None:
    """Change state and turn GStreamer's non-exception failure into an error."""
    result = pipeline.set_state(state)
    state_change_return = getattr(gst, "StateChangeReturn", None)
    failure = (
        getattr(state_change_return, "FAILURE", None)
        if state_change_return is not None
        else None
    )
    if failure is not None and result == failure:
        raise RuntimeError(f"{label} GStreamer pipeline rejected state {state!s}")


def _disconnect_gst_pull_reader(reader: Any, *, label: str) -> None:
    """Stop one appsink pull owner without losing a live thread/pipeline.

    Setting the pipeline to NULL first is the cancellation mechanism for a
    native ``try-pull-sample`` call. Ownership fields are cleared only after
    both that transition and thread exit are proved; retaining them lets the
    recorder's second cleanup pass retry an uncertain teardown.
    """
    reader._stop.set()
    pipeline = reader._pipeline
    thread = reader._thread
    primary_error: BaseException | None = None

    def remember(error: BaseException) -> None:
        nonlocal primary_error
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"additional {label} teardown failure: {type(error).__name__}: {error}"
            )

    if pipeline is not None:
        try:
            _set_gst_state_checked(
                pipeline,
                reader._gst,
                reader._gst.State.NULL,
                label=label,
            )
        except BaseException as error:
            remember(error)

    if thread is not None:
        try:
            if thread.is_alive():
                thread.join(timeout=_GST_READER_STOP_TIMEOUT_S)
            thread_alive = thread.is_alive()
        except BaseException as error:
            remember(error)
        else:
            if thread_alive:
                remember(
                    RuntimeError(
                        f"{label} pull thread did not stop within "
                        f"{_GST_READER_STOP_TIMEOUT_S:g}s; reader ownership "
                        "remains uncertain"
                    )
                )
            else:
                reader._thread = None

    if primary_error is None:
        reader._pipeline = None
        reader._sink = None
        return
    raise primary_error


def _capture_perf_from_receive(recv_perf: float, latency_s: object) -> float:
    """Estimate capture time from receipt time and measured pipeline latency.

    Invalid, non-finite, or negative latency metadata is treated as unavailable,
    preserving the historical receipt-time fallback instead of producing a
    future, NaN, or otherwise unusable snapshot lookup timestamp.
    """
    if isinstance(latency_s, bool):
        return recv_perf
    try:
        latency = float(latency_s)
    except (TypeError, ValueError):
        return recv_perf
    if (
        not math.isfinite(latency)
        or latency < 0.0
        or latency > _MAX_CAPTURE_LATENCY_S
        or latency > recv_perf
    ):
        return recv_perf
    return recv_perf - latency


def _capture_perf_from_gst_pts(
    recv_perf: float,
    pts_ns: object,
    pts_origin_perf: object,
    fallback_latency_s: object,
) -> float:
    """Map a GDP-preserved sensor PTS onto the shared ``perf_counter`` clock.

    ``pts_ns`` is running time in the camera relay's GStreamer pipeline;
    ``pts_origin_perf`` is that pipeline's running-time zero co-sampled on the
    Linux monotonic/perf-counter timeline. Corrupt, absent, implausibly stale,
    or future metadata falls back to the prior receipt-minus-latency estimate.
    """
    fallback = _capture_perf_from_receive(recv_perf, fallback_latency_s)
    if isinstance(pts_ns, bool) or isinstance(pts_origin_perf, bool):
        return fallback
    try:
        pts = float(pts_ns)
        origin = float(pts_origin_perf)
    except (TypeError, ValueError):
        return fallback
    if (
        not math.isfinite(pts)
        or not math.isfinite(origin)
        or pts < 0.0
        or pts >= _GST_CLOCK_TIME_NONE
        or origin < 0.0
    ):
        return fallback
    capture_perf = origin + pts / 1e9
    if (
        not math.isfinite(capture_perf)
        or capture_perf < 0.0
        or capture_perf > recv_perf + _MAX_CAPTURE_FUTURE_S
        or recv_perf - capture_perf > _MAX_CAPTURE_LATENCY_S
    ):
        return fallback
    # A tiny positive clock-sampling error is harmless, but never ask the joint
    # snapshot ring for a frame that has not yet been received.
    return min(capture_perf, recv_perf)


def _block_size(width: int, height: int) -> int:
    return _HEADER_BYTES + 2 * width * height * _CHANNELS


class RawFrameWriter:
    """Relay-subprocess side: publish raw RGB frames into shared memory.

    One per camera source (a mono camera or one eye of a stereo pair). Created
    with :meth:`create`, which allocates the backing block; the auto-generated
    :attr:`name` is sent to the control process so it can attach a
    :class:`RawFrameReader`.
    """

    def __init__(self, shm: Any, width: int, height: int, cond: Any) -> None:
        self._shm = shm
        self.name = shm.name
        self.width = width
        self.height = height
        self._cond = cond
        self._meta = np.ndarray((1,), dtype=_META_DTYPE, buffer=shm.buf)
        self._bufs = _frame_views(shm.buf, width, height)
        self._next_slot = 0
        self._meta["seq"][0] = 0
        self._meta["slot"][0] = 0

    @classmethod
    def create(cls, width: int, height: int, cond: Any) -> "RawFrameWriter":
        shm = shared_memory.SharedMemory(create=True, size=_block_size(width, height))
        return cls(shm, width, height, cond)

    def publish(self, rgba: "NDArray[Any]", cap_ts: float, recv_ts: float) -> None:
        """Copy one frame's RGB into the idle buffer and commit it.

        ``rgba`` is an ``(H, W, 4)`` view over the GStreamer buffer (valid only
        for this call); the ``[:, :, :3]`` copy into shared memory drops alpha.
        """
        slot = self._next_slot
        np.copyto(self._bufs[slot], rgba[:, :, :_CHANNELS])
        with self._cond:
            self._meta["slot"][0] = slot
            self._meta["cap_ts"][0] = cap_ts
            self._meta["recv_ts"][0] = recv_ts
            self._meta["seq"][0] += 1
            self._cond.notify_all()
        self._next_slot = 1 - slot

    def close(self) -> None:
        # Drop numpy views into the buffer before releasing it.
        self._meta = None  # type: ignore[assignment]
        self._bufs = None  # type: ignore[assignment]
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class RawFrameReader:
    """Control-process side: a camera-shaped view over a writer's frames.

    Implements the slice of the ``ZedCamera`` interface the dataset capture
    thread and ``AxolRobot`` use — ``read_at_or_after`` / ``read_latest`` /
    ``read_latest_with_ts`` plus ``width`` / ``height`` / ``fps`` / ``connect``
    / ``disconnect`` / ``is_connected`` — so it drops straight into
    ``robot.cameras`` with no other changes.
    """

    def __init__(self, name: str, width: int, height: int, fps: int, cond: Any) -> None:
        self._shm = shared_memory.SharedMemory(name=name)
        self.width = width
        self.height = height
        self.fps = fps
        self._cond = cond
        self._meta = np.ndarray((1,), dtype=_META_DTYPE, buffer=self._shm.buf)
        self._bufs = _frame_views(self._shm.buf, width, height)

    @property
    def is_connected(self) -> bool:
        return self._shm is not None

    def connect(self, warmup: bool = True) -> None:
        """No-op: the relay subprocess owns and opens the camera."""

    def _copy_slot(self, slot: int) -> "NDArray[Any]":
        return np.array(self._bufs[slot], dtype=np.uint8)

    def read_at_or_after(
        self, target: float, timeout_ms: float = 500
    ) -> tuple["NDArray[Any]", float, float]:
        """Block until a frame with ``cap_ts >= target`` is available; copy it."""
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            with self._cond:
                while True:
                    seq = int(self._meta["seq"][0])
                    cap = float(self._meta["cap_ts"][0])
                    if seq > 0 and cap >= target:
                        slot = int(self._meta["slot"][0])
                        recv = float(self._meta["recv_ts"][0])
                        break
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"shared-memory camera timed out waiting for a frame "
                            f"at capture_perf_ts >= {target:.6f} after "
                            f"{timeout_ms:.1f}ms."
                        )
                    self._cond.wait(remaining)
            frame = self._copy_slot(slot)
            # Double-buffer reuse only happens two frames later; if the writer
            # lapped us mid-copy (seq advanced by >=2), the copy may be torn —
            # retry against the new latest frame.
            if int(self._meta["seq"][0]) - seq < 2:
                return frame, cap, recv

    def read_latest_with_ts(self) -> tuple["NDArray[Any]", float, float]:
        while True:
            with self._cond:
                seq = int(self._meta["seq"][0])
                if seq == 0:
                    raise RuntimeError("shared-memory camera has no frames yet.")
                slot = int(self._meta["slot"][0])
                cap = float(self._meta["cap_ts"][0])
                recv = float(self._meta["recv_ts"][0])
            frame = self._copy_slot(slot)
            if int(self._meta["seq"][0]) - seq < 2:
                return frame, cap, recv

    def read_latest(self, max_age_ms: int = 500) -> "NDArray[Any]":
        frame, _cap, recv = self.read_latest_with_ts()
        age_ms = (time.perf_counter() - recv) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"latest shared-memory frame is {age_ms:.0f}ms old (> {max_age_ms})."
            )
        return frame

    def read(self) -> "NDArray[Any]":
        return self.read_at_or_after(0.0, timeout_ms=10000)[0]

    def disconnect(self) -> None:
        self._meta = None  # type: ignore[assignment]
        self._bufs = None  # type: ignore[assignment]
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._shm = None  # type: ignore[assignment]

    # ZedCamera-compatible alias.
    close = disconnect


class GstShmFrameReader:
    """Recorder-side raw-frame source backed by a gst ``shmsrc`` → ``appsink``.

    The relay's raw branch writes NV12 frames to shared memory with gst's native
    (C) ``shmsink`` — no Python pull loop in the relay, so its interpreter is free
    for the latency-critical aiortc send (running the pull loop *in the relay* is
    what halved the send during recording and made the live feed laggy/grainy).
    This reader runs the matching ``shmsrc`` consumer in the **recorder** process,
    where the per-frame Python work lands on the recorder's own GIL and can't
    starve the relay's send. It exposes the same ``read_at_or_after`` /
    ``read_latest`` / ``connect`` / ``close`` slice of the camera interface as
    :class:`RawFrameReader`, so the capture loop and ``AxolRobot`` are unchanged.

    Frames are returned as **packed NV12**: a ``(height * 3 // 2, width)`` uint8
    array (the Y plane's ``height`` rows followed by the interleaved UV plane's
    ``height // 2`` rows). The recorder's NVENC encoder consumes this directly, so
    no colorspace convert or channel copy runs on the recorder's GIL per frame
    (the NV12→RGB conversion is done only on the sampled subset of frames folded
    into image stats). The VIC may emit rows padded to a stride wider than
    ``width``; :meth:`_pull_loop` de-pads to a packed buffer so the encoder's
    ``rawvideoparse format=nv12`` (which assumes ``stride == width``) is always
    fed a correct layout.

    Shared memory carries no buffer PTS, so each frame is stamped
    ``recv_perf - latency_s`` (``latency_s`` a relay-reported pipeline-latency
    scalar) on the shared ``perf_counter`` clock — an approximation of the
    per-frame :meth:`~almond_axol.video.gst_zed._GstPipelineBase._cap_perf_from_pts`
    compensation. A small constant bias only shifts all images uniformly vs the
    joint samples, within the capture loop's frame tolerance.
    """

    def __init__(
        self,
        socket_path: str,
        caps: str,
        width: int,
        height: int,
        fps: int,
        latency_s: float,
    ) -> None:
        from .gst_zed import _require_gst

        self._gst, _ = _require_gst()
        self.width = width
        self.height = height
        self.fps = fps
        self._latency_s = latency_s
        # Packed NV12 rows for one frame: Y (height) + interleaved UV (height/2).
        self._nv12_rows = height * 3 // 2
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._frame: NDArray[Any] | None = None
        self._cap_ts: float | None = None
        self._recv_ts: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: Any = None
        self._pipeline = self._gst.parse_launch(
            f"shmsrc socket-path={socket_path} is-live=true do-timestamp=true "
            f"! {caps} ! appsink name=raw emit-signals=false max-buffers=2 "
            "drop=true sync=false"
        )

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self, warmup: bool = True) -> None:
        """Start the shmsrc pipeline + pull thread (relay owns the camera)."""
        del warmup
        if self._pipeline is None:
            raise RuntimeError("shmsrc raw-frame reader has already been closed")
        if self._thread is not None:
            raise RuntimeError("shmsrc raw-frame reader is already connected")
        self._stop.clear()
        try:
            self._sink = self._pipeline.get_by_name("raw")
            if self._sink is None:
                raise RuntimeError("shmsrc raw-frame pipeline has no appsink 'raw'")
            _set_gst_state_checked(
                self._pipeline,
                self._gst,
                self._gst.State.PLAYING,
                label="shmsrc raw-frame reader",
            )
            self._thread = threading.Thread(
                target=self._pull_loop, name="recorder-shmsrc", daemon=True
            )
            self._thread.start()
        except BaseException as error:
            try:
                self.disconnect()
            except BaseException as cleanup_error:
                error.add_note(
                    "shmsrc raw-frame reader startup cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def _pull_loop(self) -> None:
        Gst = self._gst
        w = self.width
        rows = self._nv12_rows
        while not self._stop.is_set():
            sample = self._sink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                continue  # valve closed (not recording) or starting up — idle
            recv_perf = time.perf_counter()
            buf = sample.get_buffer()
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                nv12 = self._pack_nv12(
                    np.frombuffer(mapinfo.data, dtype=np.uint8), w, rows
                )
            finally:
                buf.unmap(mapinfo)
            if nv12 is None:
                continue
            with self._lock:
                self._frame = nv12
                self._cap_ts = recv_perf - self._latency_s
                self._recv_ts = recv_perf
            self._new_frame.set()

    @staticmethod
    def _pack_nv12(arr: "NDArray[Any]", w: int, rows: int) -> "NDArray[Any] | None":
        """Copy the mapped buffer into a packed ``(rows, w)`` NV12 array.

        The VIC may pad each row to a stride wider than ``w`` (the buffer is then
        ``stride * rows`` bytes, with both planes sharing the stride). Slice each
        row back to ``w`` so the encoder's packed ``rawvideoparse`` reads correct
        Y/UV planes; when unpadded (``stride == w``) this is a plain copy.
        """
        if arr.size < w * rows:
            return None
        stride = arr.size // rows
        if stride == w:
            return arr[: w * rows].reshape(rows, w).copy()
        return np.ascontiguousarray(arr[: stride * rows].reshape(rows, stride)[:, :w])

    def read_at_or_after(
        self, target: float, timeout_ms: float = 500
    ) -> tuple["NDArray[Any]", float, float]:
        """Block until a frame with ``cap_ts >= target`` is available; return it."""
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            self._new_frame.clear()
            with self._lock:
                frame, cap, recv = self._frame, self._cap_ts, self._recv_ts
            if (
                frame is not None
                and cap is not None
                and recv is not None
                and cap >= target
            ):
                return frame, cap, recv
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"shmsrc camera timed out waiting for a frame at "
                    f"capture_perf_ts >= {target:.6f} after {timeout_ms:.1f}ms."
                )
            self._new_frame.wait(timeout=remaining)

    def read_latest_with_ts(self) -> tuple["NDArray[Any]", float, float]:
        with self._lock:
            frame, cap, recv = self._frame, self._cap_ts, self._recv_ts
        if frame is None or cap is None or recv is None:
            raise RuntimeError("shmsrc camera has not captured any frames yet.")
        return frame, cap, recv

    def read_latest(self, max_age_ms: int = 500) -> "NDArray[Any]":
        frame, _cap, recv = self.read_latest_with_ts()
        age_ms = (time.perf_counter() - recv) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"latest shmsrc frame is {age_ms:.0f}ms old (> {max_age_ms})."
            )
        return frame

    def read(self) -> "NDArray[Any]":
        return self.read_at_or_after(0.0, timeout_ms=10000)[0]

    def disconnect(self) -> None:
        _disconnect_gst_pull_reader(self, label="shmsrc raw-frame reader")

    # camera-compatible alias.
    close = disconnect


def _au_has_coded_slice(au: bytes) -> bool:
    """True if the Annex-B access unit contains a VCL (coded-picture) NAL.

    Integrity guard for the one-AU-per-row contract: the relay's encoder could
    emit an access unit carrying only non-VCL NALs (access-unit delimiter / SPS /
    PPS / SEI / end-of-sequence) with no coded slice — e.g. a boundary AU when
    the dataset valve closes. Such an AU decodes to *no* picture, so muxing it as
    a dataset frame would occupy a PTS slot without yielding a retrievable frame
    and desync frame-count from row-count. Delivering only AUs with a coded slice
    keeps them aligned; if a coded picture is then missing, the capture loop
    aborts the take rather than replaying another AU. VCL NAL types are 1-5
    (non-IDR .. IDR).

    Note: this only guards the one-AU-per-row count. The separate per-row
    timestamp precision the dataset needs (frame *k* within LeRobot's tolerance
    of ``k / fps``) is handled by the constant-fps re-stamp in the concat step
    (:func:`~almond_axol.recording.record_proc._concatenate_video_files_rebased`).
    """
    i, n = 0, len(au)
    while i + 3 < n:
        if au[i] == 0 and au[i + 1] == 0:
            if au[i + 2] == 1:
                if 1 <= (au[i + 3] & 0x1F) <= 5:
                    return True
                i += 4
                continue
            if au[i + 2] == 0 and i + 4 < n and au[i + 3] == 1:
                if 1 <= (au[i + 4] & 0x1F) <= 5:
                    return True
                i += 5
                continue
        i += 1
    return False


class EncodedAuReader:
    """Recorder-side source of the relay's pre-encoded H.264 access units.

    The relay's dataset branch encodes each camera to H.264 on the GPU and writes
    the access units to shared memory with gst's native (C) ``shmsink`` — no
    Python and no raw frame copy on the relay, and ~1 MB/s across the boundary
    instead of the ~51 MB/s the old raw NV12 path cost. This reader runs the
    matching ``shmsrc`` consumer in the **recorder** process and hands the AUs to
    :class:`~almond_axol.lerobot.h264_mux_encoder.H264MuxStreamingEncoder`, which
    just muxes them (no re-encode).

    Unlike the raw :class:`GstShmFrameReader` (which serves ``read_at_or_after`` —
    *selecting* the frame nearest a target time and dropping the rest), an encoded
    stream cannot drop frames: every P-frame depends on its predecessors. So this
    reader delivers **every** AU strictly **in order** via :meth:`read_next_au`,
    and the capture loop is frame-driven (one AU consumed per dataset row). A
    dedicated pull thread drains the (non-leaky) appsink into an in-process queue
    so a momentarily slow consumer grows the queue rather than dropping AUs and
    corrupting the stream.

    Each episode's mp4 must start on a keyframe (a leading P-frame is
    undecodable), so after :meth:`flush` the reader drops AUs until the next IDR.
    The relay can't force a keyframe on demand (the ``nvv4l2h264enc`` ``force-IDR``
    signal segfaults and force-key-unit events are ignored on L4T), so the dataset
    encoder runs a short ``idrinterval``; the episode's rows simply begin at the
    first IDR after the valve opens (a sub-``idrinterval`` start delay, no
    misalignment — video and joints both start there). GDP restores each AU's
    sensor PTS after ``shmsrc``. The relay also supplies the corresponding
    pipeline-running-time origin on the shared ``perf_counter`` clock, so this
    reader can pair the image with the nearest joint snapshot at exposure time
    rather than Python receipt time. Missing or implausible PTS/origin metadata
    falls back to ``recv_perf - latency_s`` (and ultimately receipt time). The
    mp4's own timeline is the constant-fps PTS the muxer assigns, independent of
    this pairing timestamp.
    """

    def __init__(
        self,
        socket_path: str,
        width: int,
        height: int,
        fps: int,
        name: str | None = None,
        latency_s: float = 0.0,
        pts_origin_perf: float | None = None,
    ) -> None:
        from .gst_zed import _DATASET_IDR_INTERVAL_S, _require_gst

        self._gst, _ = _require_gst()
        self.width = width
        self.height = height
        self.fps = fps
        self._name = name or socket_path
        self._latency_s = latency_s
        self._pts_origin_perf = pts_origin_perf
        self._queue: deque[tuple[bytes, float]] = deque()
        self._cond = threading.Condition()
        # Parked until the capture loop arms it with ``flush()``: the relay's
        # valve opens (and its first IDR lands) before the recorder has its
        # muxers up and its loop draining, so anything that arrives while no
        # consumer exists is dropped silently instead of being counted as a
        # backlog fault. ``disarm()`` parks it again when the loop exits.
        self._armed = False
        self._await_keyframe = True
        self._stream_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: Any = None
        # Keyframe-cadence integrity guard. The relay forces an IDR every
        # ``_DATASET_IDR_INTERVAL_S`` (see gst_zed), so a run of more than
        # ~1.5x that many frames without a keyframe means a *keyframe was lost
        # upstream* — and every frame muxed until the next IDR references that
        # missing reference, so those dataset rows won't decode. Latch a stream
        # error that ``read_next_au`` raises into the capture loop; the whole take
        # must be discarded because already-delivered orphaned pictures cannot be
        # repaired. ``_delivered`` is the running AU index for diagnostics.
        self._expected_gop = max(1, round(fps * _DATASET_IDR_INTERVAL_S))
        self._gop_warn_at = self._expected_gop + max(2, self._expected_gop // 2)
        self._max_pending_aus = max(2, round(fps * _MAX_ENCODED_AU_BACKLOG_S))
        self._since_keyframe = 0
        self._delivered = 0
        self._seen_first_au = False
        self._discont_count = 0
        # GDP restores both the producer caps and GstBuffer timing/flags after
        # shmsrc. h264parse re-derives the dimensions from the SPS as before.
        # drop=false: never discard an AU (it would break H.264 decode); the pull
        # thread keeps the appsink drained so it rarely back-pressures shmsrc.
        self._pipeline = self._gst.parse_launch(
            f"shmsrc socket-path={socket_path} is-live=true do-timestamp=false "
            "! application/x-gdp ! gdpdepay ! h264parse "
            "! appsink name=au emit-signals=false max-buffers=60 drop=false sync=false"
        )

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @property
    def pending(self) -> int:
        """Queued (undelivered) access units — a backlog/consumer-lag indicator."""
        with self._cond:
            return len(self._queue)

    def connect(self, warmup: bool = True) -> None:
        """Start the shmsrc pipeline + pull thread (relay owns the camera)."""
        del warmup
        if self._pipeline is None:
            raise RuntimeError("encoded-AU reader has already been closed")
        if self._thread is not None:
            raise RuntimeError("encoded-AU reader is already connected")
        self._stop.clear()
        try:
            self._sink = self._pipeline.get_by_name("au")
            if self._sink is None:
                raise RuntimeError("encoded-AU pipeline has no appsink 'au'")
            _set_gst_state_checked(
                self._pipeline,
                self._gst,
                self._gst.State.PLAYING,
                label="encoded-AU reader",
            )
            self._thread = threading.Thread(
                target=self._pull_loop, name="recorder-au-shmsrc", daemon=True
            )
            self._thread.start()
        except BaseException as error:
            try:
                self.disconnect()
            except BaseException as cleanup_error:
                error.add_note(
                    "encoded-AU reader startup cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def flush(self) -> None:
        """Arm the reader for an episode: drop stragglers, wait for an IDR.

        Call from the capture loop right before it starts draining. Between
        episodes the reader is parked (see :meth:`disarm`) and the relay's
        valve is shut; on the next episode the valve opens and the encoder's
        short ``idrinterval`` yields a keyframe within a fraction of a second.
        Clearing here discards anything that slipped in before the loop was
        ready so the episode's first delivered AU is a fresh IDR, and from this
        point on every accepted AU must be consumed (the backlog guard applies).
        """
        with self._cond:
            self._reset_sequence_locked()
            self._armed = True

    def disarm(self) -> None:
        """Park the reader: drop what is queued and ignore AUs until :meth:`flush`.

        The capture loop calls this when it exits (save, discard, or failure).
        The relay closes its valve moments later, but AUs keep arriving until
        it does -- and nobody is draining -- so without parking, the backlog
        guard would latch a spurious "episode aborted" against a take that has
        already ended (or, at construction, one that has not started yet).
        """
        with self._cond:
            self._reset_sequence_locked()
            self._armed = False

    def _reset_sequence_locked(self) -> None:
        self._queue.clear()
        self._await_keyframe = True
        self._since_keyframe = 0
        self._seen_first_au = False
        self._discont_count = 0
        self._stream_error = None
        self._cond.notify_all()

    def _fail_stream_locked(self, message: str) -> None:
        """Latch one episode-fatal bitstream error while ``_cond`` is held."""
        if self._stream_error is None:
            self._stream_error = message
            self._queue.clear()
            _logger.error(message)
        self._cond.notify_all()

    def _accept_access_unit(
        self,
        au: bytes,
        capture_perf: float,
        *,
        is_keyframe: bool,
        discont: bool,
    ) -> None:
        """Validate and queue one coded picture from the pull thread."""
        with self._cond:
            if not self._armed or self._stream_error is not None:
                return
            if self._await_keyframe:
                if not is_keyframe:
                    return  # wait for the episode's first IDR
                self._await_keyframe = False
                self._since_keyframe = 0
            elif is_keyframe:
                self._since_keyframe = 0
            else:
                self._since_keyframe += 1
                if self._since_keyframe >= self._gop_warn_at:
                    self._fail_stream_locked(
                        f"encoded-AU keyframe gap on {self._name} near frame "
                        f"{self._delivered}: {self._since_keyframe} frames since "
                        f"the last keyframe (the relay emits one every "
                        f"~{self._expected_gop}); episode aborted because the "
                        "H.264 reference chain is no longer trustworthy"
                    )
                    return

            # The first AU after every flush may legitimately carry DISCONT at
            # the valve/segment boundary. A later DISCONT is *not* proof that a
            # coded picture was lost: the relay's ``queue leaky=downstream``
            # sits upstream of the dataset encoder, so a raw frame it sheds
            # under load marks the next buffer DISCONT while the H.264
            # reference chain (built after that point) stays intact, and
            # shmsrc itself never drops bytes. Aborting the take here threw
            # away good episodes; the keyframe-cadence guard above is what
            # catches a genuinely broken bitstream. Log it (rate-limited) and
            # keep the picture.
            if discont and self._seen_first_au:
                self._discont_count += 1
                if self._discont_count in (1, 10) or self._discont_count % 100 == 0:
                    _logger.warning(
                        "encoded-AU discontinuity on %s near frame %d (%d this "
                        "episode); upstream frame shed, bitstream still valid",
                        self._name,
                        self._delivered,
                        self._discont_count,
                    )

            if len(self._queue) >= self._max_pending_aus:
                self._fail_stream_locked(
                    f"encoded-AU backlog on {self._name} exceeded "
                    f"{self._max_pending_aus} pending pictures near frame "
                    f"{self._delivered}; episode aborted before stale image/pose "
                    "pairing or unbounded memory growth"
                )
                return

            self._seen_first_au = True
            self._queue.append((au, capture_perf))
            self._delivered += 1
            self._cond.notify()

    def _pull_loop(self) -> None:
        Gst = self._gst
        while not self._stop.is_set():
            sample = self._sink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                continue  # valve shut (not recording) or starting up — idle
            recv_perf = time.perf_counter()
            buf = sample.get_buffer()
            capture_perf = _capture_perf_from_gst_pts(
                recv_perf,
                buf.pts,
                self._pts_origin_perf,
                self._latency_s,
            )
            is_keyframe = not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
            discont = buf.has_flags(Gst.BufferFlags.DISCONT)
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                au = bytes(mapinfo.data)
            finally:
                buf.unmap(mapinfo)
            # Drop non-VCL boundary AUs (no coded picture, e.g. the trailing AU
            # emitted when the valve closes): muxing one would make the video one
            # frame short of the dataset rows. See _au_has_coded_slice.
            if not _au_has_coded_slice(au):
                continue
            self._accept_access_unit(
                au,
                capture_perf,
                is_keyframe=is_keyframe,
                discont=discont,
            )

    def read_next_au(self, timeout_ms: float = 500) -> tuple[bytes, float]:
        """Pop the next access unit in order; block up to ``timeout_ms``.

        Returns ``(au_bytes, capture_ts)``. ``capture_ts`` is the GDP-preserved
        sensor PTS mapped to ``perf_counter``; invalid metadata falls back to
        receipt time minus the relay-reported pipeline latency (or receipt time
        when that is unavailable). Raises :class:`TimeoutError` if no AU arrives
        in time; the caller aborts the episode because skipping or replaying an
        H.264 picture would break the decoder reference chain.
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._cond:
            while not self._queue and self._stream_error is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"encoded-AU reader timed out after {timeout_ms:.1f}ms."
                    )
                self._cond.wait(remaining)
            if self._stream_error is not None:
                raise RuntimeError(self._stream_error)
            return self._queue.popleft()

    def disconnect(self) -> None:
        _disconnect_gst_pull_reader(self, label="encoded-AU reader")

    # camera-compatible alias.
    close = disconnect


# Snapshot ring depth: ~0.5 s of history at the 120 Hz control rate — plenty
# for the recorder's nearest-timestamp pose↔image pairing (the match target is
# within a frame interval or two of "now").
_SNAP_RING_SLOTS = 64


class SnapshotWriter:
    """Control-process side: publish joint/action snapshots into a shm ring.

    A lock-free seqlock ring over one shared-memory block: ``_SNAP_RING_SLOTS``
    slots of float64s
    (``[ts, *joint_obs_vals, *action_vals, intervention]`` in fixed key order),
    each guarded by its own seq counter, plus a global committed-write count.
    The intervention value is 0.0/1.0 for DAgger collection. The control loop
    calls :meth:`write` every tick; the recorder reads either the newest
    snapshot or the one nearest a camera frame's capture time. Single-writer /
    single-reader.
    """

    def __init__(self, obs_keys: list[str], action_keys: list[str]) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        n = len(self._obs_keys) + len(self._action_keys)
        self._slot_len = 2 + n
        header_bytes = _SNAP_HEADER_BYTES + 8 * _SNAP_RING_SLOTS
        self._shm = shared_memory.SharedMemory(
            create=True, size=header_bytes + 8 * _SNAP_RING_SLOTS * self._slot_len
        )
        self.name = self._shm.name
        self._meta = np.ndarray((1,), dtype=_SNAP_META_DTYPE, buffer=self._shm.buf)
        self._slot_seq = np.ndarray(
            (_SNAP_RING_SLOTS,),
            dtype="<i8",
            buffer=self._shm.buf,
            offset=_SNAP_HEADER_BYTES,
        )
        self._data = np.ndarray(
            (_SNAP_RING_SLOTS, self._slot_len),
            dtype="<f8",
            buffer=self._shm.buf,
            offset=header_bytes,
        )
        self._meta["seq"][0] = 0  # committed-write count
        self._slot_seq[:] = 0

    def write(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        """Pack one snapshot into the next ring slot (per-slot seqlock)."""
        c = int(self._meta["seq"][0])
        slot = c % _SNAP_RING_SLOTS
        self._slot_seq[slot] += 1  # odd: write in progress
        d = self._data[slot]
        d[0] = ts
        i = 1
        for k in self._obs_keys:
            d[i] = joint_obs[k]
            i += 1
        for k in self._action_keys:
            d[i] = action[k]
            i += 1
        d[i] = 1.0 if intervention else 0.0
        self._slot_seq[slot] += 1  # even: committed
        self._meta["seq"][0] = c + 1

    def close(self) -> None:
        self._meta = None  # type: ignore[assignment]
        self._slot_seq = None  # type: ignore[assignment]
        self._data = None  # type: ignore[assignment]
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class SnapshotReader:
    """Recorder-subprocess side: read joint/action snapshots from the shm ring.

    Attaches to a :class:`SnapshotWriter`'s block by name and reconstructs the
    ``(joint_obs, action, ts, intervention)`` using the same key order. Returns
    ``None`` before the first write (mirroring the in-process publisher).
    """

    def __init__(self, name: str, obs_keys: list[str], action_keys: list[str]) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        n = len(self._obs_keys) + len(self._action_keys)
        self._slot_len = 2 + n
        header_bytes = _SNAP_HEADER_BYTES + 8 * _SNAP_RING_SLOTS
        self._shm = shared_memory.SharedMemory(name=name)
        self._meta = np.ndarray((1,), dtype=_SNAP_META_DTYPE, buffer=self._shm.buf)
        self._slot_seq = np.ndarray(
            (_SNAP_RING_SLOTS,),
            dtype="<i8",
            buffer=self._shm.buf,
            offset=_SNAP_HEADER_BYTES,
        )
        self._data = np.ndarray(
            (_SNAP_RING_SLOTS, self._slot_len),
            dtype="<f8",
            buffer=self._shm.buf,
            offset=header_bytes,
        )

    def _read_slot(self, slot: int) -> np.ndarray | None:
        """Copy one slot consistently (per-slot seqlock), or ``None`` on miss."""
        for _ in range(8):
            s1 = int(self._slot_seq[slot])
            if s1 == 0:
                return None  # never written
            if s1 & 1:
                continue  # writer mid-write
            snap = np.array(self._data[slot], dtype="<f8")
            if int(self._slot_seq[slot]) == s1:
                return snap
        return None

    def _to_dicts(self, snap: np.ndarray) -> tuple[dict, dict, float, bool]:
        ts = float(snap[0])
        vals = snap[1:]
        no = len(self._obs_keys)
        joint_obs = {k: float(vals[i]) for i, k in enumerate(self._obs_keys)}
        action = {k: float(vals[no + i]) for i, k in enumerate(self._action_keys)}
        intervention = bool(snap[-1] >= 0.5)
        return joint_obs, action, ts, intervention

    def read_latest(self) -> tuple[dict, dict, float, bool] | None:
        count = int(self._meta["seq"][0])
        if count == 0:
            return None
        snap = self._read_slot((count - 1) % _SNAP_RING_SLOTS)
        return self._to_dicts(snap) if snap is not None else None

    def read_nearest(self, target_ts: float) -> tuple[dict, dict, float, bool] | None:
        """Return the buffered snapshot whose timestamp is nearest ``target_ts``.

        Pairs a camera frame's capture time with the pose/action snapshot
        captured closest to it (both on the system-wide ``perf_counter``
        timeline) instead of whatever happened to be newest — the residual
        skew is recorded per row as ``pose_lag``. Falls back over any slot the
        writer is concurrently updating; returns ``None`` before the first
        write.
        """
        count = int(self._meta["seq"][0])
        if count == 0:
            return None
        best: np.ndarray | None = None
        best_err = float("inf")
        for back in range(min(count, _SNAP_RING_SLOTS)):
            slot = (count - 1 - back) % _SNAP_RING_SLOTS
            snap = self._read_slot(slot)
            if snap is None:
                continue
            err = abs(float(snap[0]) - target_ts)
            if err < best_err:
                best, best_err = snap, err
            elif best is not None and float(snap[0]) < target_ts:
                # Timestamps decrease as we walk back; once past the target
                # and no longer improving, stop.
                break
        return self._to_dicts(best) if best is not None else None

    def close(self) -> None:
        self._meta = None  # type: ignore[assignment]
        self._slot_seq = None  # type: ignore[assignment]
        self._data = None  # type: ignore[assignment]
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._shm = None  # type: ignore[assignment]


def _frame_views(buf: Any, width: int, height: int) -> list["NDArray[Any]"]:
    """Two ``(H, W, 3)`` uint8 views over the double buffer after the header."""
    frame_bytes = width * height * _CHANNELS
    views = []
    for i in range(2):
        offset = _HEADER_BYTES + i * frame_bytes
        views.append(
            np.ndarray(
                (height, width, _CHANNELS),
                dtype=np.uint8,
                buffer=buf,
                offset=offset,
            )
        )
    return views
