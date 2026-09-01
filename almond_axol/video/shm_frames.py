"""Camera and state transport across the relay/recorder boundary.

``collect-data`` needs camera samples and matching robot state in its recorder,
but running camera grab + NVENC encode + aiortc WebRTC in the control process
starves the teleop/IK loops (see :mod:`almond_axol.video.video_proc`). The relay
subprocess therefore owns the cameras and does all the heavy work. The primary
path carries GDP-wrapped H.264 access units and their sensor PTS from ``shmsink``
to :class:`EncodedAuReader`; legacy/fallback paths carry raw NV12 or RGB frames.
A separate bounded snapshot ring carries the control loop's timestamped
joint/action history, so the recorder can choose the state nearest each exposure
without touching the hot control path.

Raw-frame fallback layout (one :class:`SharedMemory` block per camera source):

    [ meta: seq, slot, cap_ts, recv_ts, slot0_seq, slot1_seq ]
    [ buffer 0 ][ buffer 1 ]

The two frame buffers are double-buffered: the writer always fills the buffer
the reader isn't pointed at, then publishes the new ``slot`` + timestamps under
a shared :class:`multiprocessing.Condition` and notifies. A reader copies out of
the published slot *outside* the lock. The per-slot odd/even sequence marks reuse
before the writer touches pixels and is checked again after the copy, so even a
reader delayed across multiple camera frames detects and retries a torn copy.

Timestamps are ``time.perf_counter`` seconds. On Linux that is
``CLOCK_MONOTONIC``, which shares an origin across processes, so a ``cap_ts``
stamped in the relay subprocess stays directly comparable to the joint-sample
timestamps taken in the control process — preserving the image/joint alignment
the dataset relies on.
"""

from __future__ import annotations

import logging
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
    [
        ("seq", "<i8"),
        ("slot", "<i8"),
        ("cap_ts", "<f8"),
        ("recv_ts", "<f8"),
        ("slot0_seq", "<u8"),
        ("slot1_seq", "<u8"),
    ]
)
_HEADER_BYTES = 64

# Frames are RGB (3 channels): the VIC delivers RGBA, the writer drops alpha so
# only what the dataset stores crosses the boundary.
_CHANNELS = 3

# Snapshot history retains a little over four seconds at the 120 Hz control rate.
# Camera AUs normally arrive only a few frames late, but this leaves generous
# headroom for encoder / recorder startup and short scheduling stalls while
# keeping the block tiny (roughly 130 KiB for the usual 28-value snapshot).
_SNAP_RING_CAPACITY = 512

# Header publication is separate from each slot's seqlock. A writer commits a
# slot first, then advances ``published``; readers use ``generation`` to reject a
# slot that wrapped while they were inspecting it. All fields are naturally
# aligned, fixed-width values so the same layout is valid in both processes.
_SNAP_HEADER_DTYPE = np.dtype(
    [("published", "<u8"), ("capacity", "<u4"), ("width", "<u4")]
)
_SNAP_HEADER_BYTES = 64
_SNAP_SLOT_META_DTYPE = np.dtype([("seq", "<u8"), ("generation", "<u8")])


def _snapshot_block_size(width: int, capacity: int = _SNAP_RING_CAPACITY) -> int:
    return (
        _SNAP_HEADER_BYTES
        + capacity * _SNAP_SLOT_META_DTYPE.itemsize
        + capacity * width * np.dtype("<f8").itemsize
    )


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
        self._meta["slot0_seq"][0] = 0
        self._meta["slot1_seq"][0] = 0

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
        slot_seq_field = "slot0_seq" if slot == 0 else "slot1_seq"
        # Mark this physical slot busy before touching its pixels. Readers copy
        # outside the shared lock; the odd/even slot generation lets them
        # detect reuse that has started but is not published yet.
        with self._cond:
            slot_seq = int(self._meta[slot_seq_field][0])
            if slot_seq & 1:
                slot_seq += 1
            self._meta[slot_seq_field][0] = slot_seq + 1
        np.copyto(self._bufs[slot], rgba[:, :, :_CHANNELS])
        with self._cond:
            self._meta["slot"][0] = slot
            self._meta["cap_ts"][0] = cap_ts
            self._meta["recv_ts"][0] = recv_ts
            self._meta[slot_seq_field][0] = slot_seq + 2
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
                        slot_seq_field = "slot0_seq" if slot == 0 else "slot1_seq"
                        slot_seq = int(self._meta[slot_seq_field][0])
                        if slot_seq & 1:
                            continue
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
            # Reacquire for a formal ARM memory barrier before checking whether
            # the writer lapped this out-of-lock copy and reused its slot.
            with self._cond:
                slot_stable = int(self._meta[slot_seq_field][0]) == slot_seq
            if slot_stable:
                return frame, cap, recv

    def read_latest_with_ts(self) -> tuple["NDArray[Any]", float, float]:
        while True:
            with self._cond:
                seq = int(self._meta["seq"][0])
                if seq == 0:
                    raise RuntimeError("shared-memory camera has no frames yet.")
                slot = int(self._meta["slot"][0])
                slot_seq_field = "slot0_seq" if slot == 0 else "slot1_seq"
                slot_seq = int(self._meta[slot_seq_field][0])
                if slot_seq & 1:
                    continue
                cap = float(self._meta["cap_ts"][0])
                recv = float(self._meta["recv_ts"][0])
            frame = self._copy_slot(slot)
            with self._cond:
                slot_stable = int(self._meta[slot_seq_field][0]) == slot_seq
            if slot_stable:
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
        self._sink = self._pipeline.get_by_name("raw")
        self._pipeline.set_state(self._gst.State.PLAYING)
        self._thread = threading.Thread(
            target=self._pull_loop, name="recorder-shmsrc", daemon=True
        )
        self._thread.start()

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
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._gst.State.NULL)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._pipeline = None  # type: ignore[assignment]
        self._sink = None

    # camera-compatible alias.
    close = disconnect


def _au_has_nal_type(au: bytes, wanted: range | tuple[int, ...]) -> bool:
    """Return whether an Annex-B access unit contains a requested NAL type."""
    i, n = 0, len(au)
    while i + 3 < n:
        if au[i] == 0 and au[i + 1] == 0:
            if au[i + 2] == 1:
                if (au[i + 3] & 0x1F) in wanted:
                    return True
                i += 4
                continue
            if au[i + 2] == 0 and i + 4 < n and au[i + 3] == 1:
                if (au[i + 4] & 0x1F) in wanted:
                    return True
                i += 5
                continue
        i += 1
    return False


def _au_has_coded_slice(au: bytes) -> bool:
    """True if the Annex-B access unit contains a VCL (coded-picture) NAL.

    Integrity guard for the one-AU-per-row contract: the relay's encoder could
    emit an access unit carrying only non-VCL NALs (access-unit delimiter / SPS /
    PPS / SEI / end-of-sequence) with no coded slice — e.g. a boundary AU when
    the dataset valve closes. Such an AU decodes to *no* picture, so muxing it as
    a dataset frame would occupy a PTS slot without yielding a retrievable frame
    and desync frame-count from row-count. Delivering only AUs with a coded slice
    keeps them aligned; a starved coded stream aborts the episode because a
    predictive AU cannot safely be duplicated. VCL NAL types are 1-5 (non-IDR
    .. IDR).

    Note: this only guards the one-AU-per-row count. The separate per-row
    timestamp precision the dataset needs (frame *k* within LeRobot's tolerance
    of ``k / fps``) is handled by the constant-fps re-stamp in the concat step
    (:func:`~almond_axol.recording.record_proc._concatenate_video_files_rebased`).
    """
    return _au_has_nal_type(au, range(1, 6))


def _au_is_idr(au: bytes) -> bool:
    """True only for an H.264 IDR AU, not merely a non-delta I-picture."""
    return _au_has_nal_type(au, (5,))


class EncodedAuReader:
    """Recorder-side source of the relay's pre-encoded H.264 access units.

    The relay's dataset branch encodes each camera to H.264 on the GPU, wraps the
    access units (including PTS/flags/caps) with ``gdppay``, and writes them to
    shared memory with gst's native (C) ``shmsink`` — no Python and no raw frame
    copy on the relay, and ~1 MB/s across the boundary instead of the ~51 MB/s
    the old raw NV12 path cost. This reader runs the matching ``shmsrc !
    gdpdepay`` consumer in the **recorder** process and hands the AUs to
    :class:`~almond_axol.lerobot.h264_mux_encoder.H264MuxStreamingEncoder`, which
    just muxes them (no re-encode).

    Unlike the raw :class:`GstShmFrameReader` (which serves ``read_at_or_after`` —
    *selecting* the frame nearest a target time and dropping the rest), an encoded
    stream cannot drop frames: every P-frame depends on its predecessors. So this
    reader delivers **every** AU strictly **in order** via :meth:`read_next_au`,
    and the capture loop is frame-driven (one AU consumed per dataset row). A
    dedicated pull thread drains the (non-leaky) appsink. Before the first
    episode flush it discards validated startup AUs; afterwards it fills an
    in-process queue so a momentarily slow consumer grows the queue rather than
    dropping AUs and corrupting the stream.

    Each episode's mp4 must start on a keyframe (a leading P-frame is
    undecodable), so after :meth:`flush` the reader drops AUs until the next IDR.
    The relay can't force a keyframe on demand (the ``nvv4l2h264enc`` ``force-IDR``
    signal segfaults and force-key-unit events are ignored on L4T), so the dataset
    encoder runs a short ``idrinterval``; the episode's rows simply begin at the
    first IDR after the valve opens. GDP restores the original sensor-exposure
    PTS after shm; ``pts_perf_offset_s`` maps that pipeline running-time onto the
    system-wide ``perf_counter`` clock used by joint/action snapshots. The mp4's
    own timeline remains the constant-fps PTS the muxer assigns, independent of
    this physical capture timestamp.
    """

    def __init__(
        self,
        socket_path: str,
        width: int,
        height: int,
        fps: int,
        name: str | None = None,
        *,
        pts_perf_offset_s: float,
        capture_fps: int | None = None,
    ) -> None:
        from .gst_zed import _DATASET_IDR_INTERVAL_S, _require_gst

        self._gst, _ = _require_gst()
        self.width = width
        self.height = height
        self.fps = fps
        self.capture_fps = fps if capture_fps is None else int(capture_fps)
        if self.capture_fps < self.fps or self.fps <= 0:
            raise ValueError(
                f"capture fps ({self.capture_fps}) must be at least dataset "
                f"fps ({self.fps})"
            )
        self._name = name or socket_path
        self._pts_perf_offset_s = float(pts_perf_offset_s)
        if not np.isfinite(self._pts_perf_offset_s):
            raise ValueError("PTS/perf_counter offset must be finite")
        self._queue: deque[tuple[bytes, float, float]] = deque()
        # Never let a capture-loop failure turn into unbounded compressed-frame
        # growth while the operator continues moving before ending the take.
        # Overflow is fatal for the episode because dropping one predictive AU
        # invalidates every dependent P-frame until the next IDR.
        self._queue_limit = max(60, 2 * fps)
        self._cond = threading.Condition()
        # Episode-start drain handshake. While the relay valve is closed the
        # pull worker discards until appsink has been quiet for one full poll,
        # then atomically arms the next actual IDR. This prevents an unobserved
        # old IDR in the GDP/shm tail from becoming the next episode's row zero.
        self._flush_requested = threading.Event()
        self._flush_complete = threading.Event()
        self._await_keyframe = True
        self._error: str | None = None
        self._permanent_error: str | None = None
        self._minimum_capture_perf = float("-inf")
        self._latest_capture_perf = float("-inf")
        self._episode_cutoff_active = False
        self._first_sample = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: Any = None
        # Keyframe-cadence integrity guard. The relay emits an IDR every
        # ``_DATASET_IDR_INTERVAL_S`` (see gst_zed), so a run of more than
        # ~1.5x that many frames without a keyframe means a *keyframe was lost
        # upstream* — and every frame muxed until the next IDR references that
        # missing reference, so those dataset rows won't decode. We can't undo it
        # here (the orphaned frames are already delivered), but logging it loudly
        # turns a silent, train-time-only corruption into a diagnosable signal
        # (which camera, which frame). ``_delivered`` is the running AU index for
        # the log; ``_gap_warnings`` counts detections for the disconnect summary.
        self._expected_gop = max(1, round(fps * _DATASET_IDR_INTERVAL_S))
        self._gop_warn_at = self._expected_gop + max(2, self._expected_gop // 2)
        self._since_keyframe = 0
        self._delivered = 0
        self._seen_first_au = False
        # gdpdepay restores the sender's serialized caps and buffer metadata.
        # Keep a fixed H.264 filter as an integrity check; h264parse re-derives
        # dimensions from the SPS and preserves the exposure PTS.
        # drop=false: never discard an AU (it would break H.264 decode); the pull
        # thread keeps the appsink drained so it rarely back-pressures shmsrc.
        caps = (
            f"video/x-h264,stream-format=byte-stream,alignment=au,"
            f"width={width},height={height},framerate={fps}/1"
        )
        self._pipeline = self._gst.parse_launch(
            f"shmsrc socket-path={socket_path} is-live=true do-timestamp=false "
            f"! gdpdepay ! {caps} ! h264parse "
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
        """Start shmsrc and verify that GDP delivers at least one coded AU."""
        self._sink = self._pipeline.get_by_name("au")
        self._pipeline.set_state(self._gst.State.PLAYING)
        self._thread = threading.Thread(
            target=self._pull_loop, name="recorder-au-shmsrc", daemon=True
        )
        self._thread.start()
        deadline = time.perf_counter() + 10.0
        while not self._first_sample.wait(0.05):
            with self._cond:
                error = self._permanent_error or self._error
            if error is not None:
                self.disconnect()
                raise RuntimeError(error)
            if time.perf_counter() >= deadline:
                self.disconnect()
                raise TimeoutError(
                    f"encoded-AU reader {self._name} received no GDP/H.264 "
                    "sample within 10s"
                )
        with self._cond:
            error = self._permanent_error or self._error
        if error is not None:
            self.disconnect()
            raise RuntimeError(error)

    def begin_flush(self) -> None:
        """Begin discarding transport tail while the relay valve is closed."""
        with self._cond:
            self._flush_complete.clear()
            self._flush_requested.set()
            self._queue.clear()
            self._cond.notify_all()

    def finish_flush(self, timeout_s: float = 2.0) -> None:
        """Wait until the pull worker observes a full quiet appsink interval."""
        if not self._flush_complete.wait(timeout_s):
            raise TimeoutError(
                f"encoded-AU reader {self._name} did not drain before episode start"
            )
        with self._cond:
            error = self._permanent_error or self._error
        if error is not None:
            raise RuntimeError(error)

    def flush(self) -> None:
        """Drain old AUs and re-arm keyframe-wait (call at episode start).

        ``_minimum_capture_perf`` is the newest sensor PTS this reader had
        already observed, not the host time at which ``flush`` runs. Exposure
        necessarily predates delivery, so a wall-time cutoff would reject the
        freshly reopened IDR and delay row zero by a whole GOP. Queue clearing,
        strictly newer PTS, and the actual-IDR requirement together reject the
        previous episode's tail without confusing transport latency for age.
        """
        self.begin_flush()
        self.finish_flush()

    def _complete_flush(self) -> None:
        """Commit a requested flush after appsink has gone quiet."""
        with self._cond:
            if not self._flush_requested.is_set():
                return
            self._queue.clear()
            self._await_keyframe = True
            self._since_keyframe = 0
            self._seen_first_au = False
            self._minimum_capture_perf = self._latest_capture_perf
            self._episode_cutoff_active = True
            # Transport errors are permanent. Episode-local continuity errors
            # may recover after the relay closes immediately before an IDR.
            self._error = self._permanent_error
            self._flush_requested.clear()
            self._flush_complete.set()
            self._cond.notify_all()

    def _fail(self, message: str, *, permanent: bool = False) -> None:
        """Wake consumers with a fatal integrity error; keep draining gst."""
        with self._cond:
            if permanent:
                self._permanent_error = message
            if self._error is None:
                self._error = message
            self._queue.clear()
            self._cond.notify_all()

    def _check_bus_error(self) -> None:
        bus = self._pipeline.get_bus() if self._pipeline is not None else None
        if bus is None:
            return
        msg = bus.pop_filtered(self._gst.MessageType.ERROR | self._gst.MessageType.EOS)
        if msg is None:
            return
        if msg.type == self._gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            detail = f": {debug}" if debug else ""
            reason = f"{err}{detail}"
        else:
            reason = "unexpected end of stream"
        self._fail(
            f"encoded-AU GStreamer pipeline failed on {self._name}: {reason}",
            permanent=True,
        )

    def _pull_loop(self) -> None:
        Gst = self._gst
        while not self._stop.is_set():
            flush_was_requested = self._flush_requested.is_set()
            sample = self._sink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                self._check_bus_error()
                # The whole quiet interval must begin after begin_flush(). A
                # request arriving near the end of an already-running empty
                # pull cannot certify that no delayed tail follows it.
                if flush_was_requested and self._flush_requested.is_set():
                    self._complete_flush()
                continue  # valve shut (not recording) or starting up — idle
            recv_perf = time.perf_counter()
            buf = sample.get_buffer()
            discont = buf.has_flags(Gst.BufferFlags.DISCONT)
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                self._fail(
                    f"encoded AU on {self._name} could not be mapped; a "
                    "dependency-bearing frame was lost"
                )
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
            is_idr = _au_is_idr(au)
            if buf.pts == Gst.CLOCK_TIME_NONE:
                self._fail(
                    f"encoded AU on {self._name} has no PTS; exact "
                    "image/state alignment is unavailable",
                    permanent=True,
                )
                continue
            # gdppay/gdpdepay normalizes a missing input PTS to zero. Zero is a
            # legitimate value only for the pipeline's very first frame; once
            # flush() establishes an episode boundary, seeing it means the
            # sender lost a timestamp. Silently filtering it as an old frame
            # would also leave subsequent P-frames without a reference.
            if buf.pts == 0 and self._episode_cutoff_active:
                self._fail(
                    f"encoded AU on {self._name} reset to PTS 0; exact "
                    "image/state alignment is unavailable"
                )
                continue
            capture_perf = buf.pts / 1e9 + self._pts_perf_offset_s
            if not np.isfinite(capture_perf):
                self._fail(
                    f"encoded AU on {self._name} has invalid PTS",
                    permanent=True,
                )
                continue
            # Publish readiness only after the timestamp has passed every
            # transport-integrity check.  connect() polls errors separately;
            # setting this event before _fail() creates a race where a malformed
            # first AU can make startup appear successful.
            self._first_sample.set()
            with self._cond:
                prior_latest = self._latest_capture_perf
                self._latest_capture_perf = max(prior_latest, capture_perf)
                if (
                    self._flush_requested.is_set()
                    or capture_perf <= self._minimum_capture_perf
                    or self._error is not None
                ):
                    continue
                # Before the first episode boundary, the relay's output queue
                # may intentionally shed AUs while shmsink waits for this late
                # reader. GStreamer marks the resumed stream DISCONT, and there
                # may be more than one such buffer. These startup AUs are never
                # recorded: the initial closed-valve flush below establishes a
                # strictly newer PTS cutoff and re-arms the first-IDR gate. Drain
                # them here so their expected discontinuities and volume cannot
                # fail connect() or overflow the bounded episode queue. Transport
                # and timestamp validation above remains fail-closed.
                if not self._episode_cutoff_active:
                    continue
                if self._await_keyframe:
                    if not is_idr:
                        continue  # wait for the episode's first IDR
                    self._await_keyframe = False
                    self._since_keyframe = 0
                elif is_idr:
                    self._since_keyframe = 0
                else:
                    self._since_keyframe += 1
                    if self._since_keyframe == self._gop_warn_at:
                        self._fail_keyframe_gap()
                        continue
                # A shmsrc DISCONT after the first AU means an upstream buffer was
                # dropped between the relay and here — the following frames can lose
                # their reference. Surface it (the first AU legitimately carries it).
                if discont and self._seen_first_au:
                    self._fail(
                        f"encoded-AU discontinuity on {self._name} near frame "
                        f"{self._delivered}; an upstream frame was dropped"
                    )
                    continue
                if len(self._queue) >= self._queue_limit:
                    self._error = (
                        f"encoded-AU backlog on {self._name} exceeded "
                        f"{self._queue_limit} frames; capture stopped draining "
                        "the predictive stream"
                    )
                    self._queue.clear()
                    self._cond.notify_all()
                    continue
                self._seen_first_au = True
                self._queue.append((au, capture_perf, recv_perf))
                self._delivered += 1
                self._cond.notify()

    def _fail_keyframe_gap(self) -> None:
        """Reject a stream that lost a periodic IDR upstream."""
        self._fail(
            f"encoded-AU keyframe gap on {self._name} near frame "
            f"{self._delivered}: {self._since_keyframe} frames since the last "
            f"keyframe (expected about {self._expected_gop})"
        )

    def read_next_au(self, timeout_ms: float = 500) -> tuple[bytes, float, float]:
        """Pop the next access unit in order; block up to ``timeout_ms``.

        Returns ``(au_bytes, capture_perf_ts, recv_perf_ts)``. The capture time
        is the sender's sensor-exposure PTS mapped onto ``perf_counter``; receive
        time is retained for latency diagnostics. Raises :class:`RuntimeError`
        on any transport/PTS/reference discontinuity and :class:`TimeoutError`
        if no fresh AU arrives in time. Predictive AUs are never duplicated.
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._cond:
            while not self._queue:
                if self._error is not None:
                    raise RuntimeError(self._error)
                if self._stop.is_set():
                    raise RuntimeError(f"encoded-AU reader {self._name} is closed")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"encoded-AU reader timed out after {timeout_ms:.1f}ms."
                    )
                self._cond.wait(remaining)
            return self._queue.popleft()

    def disconnect(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._gst.State.NULL)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._pipeline = None
        self._sink = None

    # camera-compatible alias.
    close = disconnect


class SnapshotWriter:
    """Control-process side: publish timestamped joint/action snapshots.

    A bounded SPSC ring stores float64 records in the form
    ``[ts, *joint_obs_vals, *action_vals, intervention]`` (fixed key order;
    ``intervention`` is 0.0/1.0). Each slot has its own seqlock and logical
    generation. A process-shared lock supplies the acquire/release memory
    ordering required on Tegra/ARM; the slot seqlock remains a defensive layout
    check. The 512-slot history spans >4 seconds at the 120 Hz control rate, and
    its critical section is only one small fixed-width record.
    """

    def __init__(
        self, obs_keys: list[str], action_keys: list[str], process_lock: Any
    ) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        self._lock = process_lock
        self._width = 2 + len(self._obs_keys) + len(self._action_keys)
        self._capacity = _SNAP_RING_CAPACITY
        self._shm = shared_memory.SharedMemory(
            create=True, size=_snapshot_block_size(self._width, self._capacity)
        )
        self.name = self._shm.name
        self._header = np.ndarray((1,), dtype=_SNAP_HEADER_DTYPE, buffer=self._shm.buf)
        self._slot_meta = np.ndarray(
            (self._capacity,),
            dtype=_SNAP_SLOT_META_DTYPE,
            buffer=self._shm.buf,
            offset=_SNAP_HEADER_BYTES,
        )
        data_offset = (
            _SNAP_HEADER_BYTES + self._capacity * _SNAP_SLOT_META_DTYPE.itemsize
        )
        self._data = np.ndarray(
            (self._capacity, self._width),
            dtype="<f8",
            buffer=self._shm.buf,
            offset=data_offset,
        )
        self._header["published"][0] = 0
        self._header["capacity"][0] = self._capacity
        self._header["width"][0] = self._width
        self._slot_meta.fill(0)
        self._next_generation = 1

    def write(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool = False
    ) -> None:
        """Pack and commit one snapshot under the shared memory-ordering lock."""
        # SemLock is not owner-death robust. If the recorder is SIGKILLed while
        # reading, never let the 120 Hz robot-control thread block forever on
        # its abandoned lock; surface a fatal recorder failure instead.
        if not self._lock.acquire(timeout=0.050):
            raise RuntimeError(
                "recorder snapshot lock was not released within 50ms; "
                "the recorder subprocess may have exited"
            )
        try:
            self._write_locked(joint_obs, action, ts, intervention)
        finally:
            self._lock.release()

    def _write_locked(
        self, joint_obs: dict, action: dict, ts: float, intervention: bool
    ) -> None:
        """Pack one record; caller holds :attr:`_lock`."""
        generation = self._next_generation
        slot = (generation - 1) % self._capacity
        # seq is odd while this slot is being replaced and even once committed.
        # It never resets on ring wrap, preventing an ABA match during a read.
        seq = int(self._slot_meta["seq"][slot])
        if seq & 1:
            # A prior write can only be abandoned by an exception while packing
            # caller data. Keep that generation unreadable, but recover the slot
            # for a later successful write without ever publishing partial data.
            seq += 1
        self._slot_meta["seq"][slot] = seq + 1
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
        self._slot_meta["generation"][slot] = generation
        self._slot_meta["seq"][slot] = seq + 2
        # Publish only after the complete slot is visible. Readers that observed
        # the prior high-water mark can still safely read that older generation.
        self._header["published"][0] = generation
        self._next_generation += 1

    def close(self) -> None:
        self._header = None  # type: ignore[assignment]
        self._slot_meta = None  # type: ignore[assignment]
        self._data = None  # type: ignore[assignment]
        if self._shm is None:
            return
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        self._shm = None  # type: ignore[assignment]


class SnapshotReader:
    """Recorder-subprocess side: query timestamped joint/action history.

    Attaches to a :class:`SnapshotWriter`'s block by name and reconstructs the
    ``(joint_obs, action, ts, intervention)`` tuple using the same key order.
    :meth:`read_latest` preserves the original single-slot API, while
    :meth:`read_nearest` selects the committed record nearest a camera capture
    timestamp. Both return ``None`` before the first publication or on the rare
    occasion that a continuously wrapping writer prevents a coherent read.
    """

    def __init__(
        self,
        name: str,
        obs_keys: list[str],
        action_keys: list[str],
        process_lock: Any,
    ) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        self._lock = process_lock
        expected_width = 2 + len(self._obs_keys) + len(self._action_keys)
        self._shm = shared_memory.SharedMemory(name=name)
        self._header = np.ndarray((1,), dtype=_SNAP_HEADER_DTYPE, buffer=self._shm.buf)
        self._capacity = int(self._header["capacity"][0])
        self._width = int(self._header["width"][0])
        if self._capacity <= 0 or self._width != expected_width:
            self._header = None  # type: ignore[assignment]
            self._shm.close()
            self._shm = None  # type: ignore[assignment]
            raise ValueError(
                "snapshot shared-memory layout does not match the supplied keys "
                f"(capacity={self._capacity}, width={self._width}, "
                f"expected_width={expected_width})."
            )
        expected_size = _snapshot_block_size(self._width, self._capacity)
        if self._shm.size < expected_size:
            actual_size = self._shm.size
            self._header = None  # type: ignore[assignment]
            self._shm.close()
            self._shm = None  # type: ignore[assignment]
            raise ValueError(
                "snapshot shared-memory block is smaller than its declared layout "
                f"({actual_size} < {expected_size})."
            )
        self._slot_meta = np.ndarray(
            (self._capacity,),
            dtype=_SNAP_SLOT_META_DTYPE,
            buffer=self._shm.buf,
            offset=_SNAP_HEADER_BYTES,
        )
        data_offset = (
            _SNAP_HEADER_BYTES + self._capacity * _SNAP_SLOT_META_DTYPE.itemsize
        )
        self._data = np.ndarray(
            (self._capacity, self._width),
            dtype="<f8",
            buffer=self._shm.buf,
            offset=data_offset,
        )

    def read_latest(self) -> tuple[dict, dict, float, bool] | None:
        """Return the newest committed snapshot, preserving the original API."""
        with self._lock:
            return self._read_latest_locked()

    def _read_latest_locked(self) -> tuple[dict, dict, float, bool] | None:
        """Read the newest snapshot; caller holds :attr:`_lock`."""
        for _ in range(8):
            generation = int(self._header["published"][0])
            if generation == 0:
                return None  # no snapshot published yet
            snap = self._read_generation(generation)
            if snap is not None:
                return self._unpack(snap)
        return None

    def read_nearest(self, target_ts: float) -> tuple[dict, dict, float, bool] | None:
        """Return the committed snapshot closest to ``target_ts``.

        Snapshot timestamps are monotonic ``perf_counter`` values, so a binary
        search finds the bracketing generations in O(log capacity). Slot
        generation + seqlock validation detects any overwrite during the search;
        in that case the search restarts against the writer's new window.
        """
        with self._lock:
            return self._read_nearest_locked(target_ts)

    def _read_nearest_locked(
        self, target_ts: float
    ) -> tuple[dict, dict, float, bool] | None:
        """Find the nearest snapshot; caller holds :attr:`_lock`."""
        for _ in range(8):
            newest = int(self._header["published"][0])
            if newest == 0:
                return None
            oldest = max(1, newest - self._capacity + 1)
            oldest_ts = self._read_generation_ts(oldest)
            newest_ts = self._read_generation_ts(newest)
            if oldest_ts is None or newest_ts is None:
                continue
            # Never silently clamp an exposure outside retained state history.
            # A target just newer than ``newest`` can be retried by the caller;
            # one older than ``oldest`` means the recorder fell irrecoverably
            # behind and the episode must not be saved as synchronized.
            if target_ts < oldest_ts or target_ts > newest_ts:
                return None
            lo = oldest
            hi = newest
            first_at_or_after = newest + 1
            retry = False
            while lo <= hi:
                mid = (lo + hi) // 2
                ts = self._read_generation_ts(mid)
                if ts is None:
                    retry = True
                    break
                if ts >= target_ts:
                    first_at_or_after = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            if retry:
                continue

            candidates: list[tuple[float, "NDArray[Any]"]] = []
            if first_at_or_after <= newest:
                snap = self._read_generation(first_at_or_after)
                if snap is None:
                    continue
                candidates.append((abs(float(snap[0]) - target_ts), snap))
            before = first_at_or_after - 1
            if before >= oldest:
                snap = self._read_generation(before)
                if snap is None:
                    continue
                candidates.append((abs(float(snap[0]) - target_ts), snap))
            if not candidates:
                continue
            # On an exact tie, prefer the later sample. This avoids pairing an
            # image with a needlessly older state at a half-tick boundary.
            _distance, nearest = min(
                candidates, key=lambda item: (item[0], -float(item[1][0]))
            )
            return self._unpack(nearest)
        return None

    def _read_generation_ts(self, generation: int) -> float | None:
        """Read only a slot timestamp, validating it against concurrent wrap."""
        slot = (generation - 1) % self._capacity
        for _ in range(4):
            seq1 = int(self._slot_meta["seq"][slot])
            if seq1 == 0 or seq1 & 1:
                continue
            if int(self._slot_meta["generation"][slot]) != generation:
                return None
            ts = float(self._data[slot, 0])
            if (
                int(self._slot_meta["seq"][slot]) == seq1
                and int(self._slot_meta["generation"][slot]) == generation
            ):
                return ts
        return None

    def _read_generation(self, generation: int) -> "NDArray[Any] | None":
        """Copy one coherent logical generation from its physical ring slot."""
        slot = (generation - 1) % self._capacity
        for _ in range(4):
            seq1 = int(self._slot_meta["seq"][slot])
            if seq1 == 0 or seq1 & 1:
                continue
            if int(self._slot_meta["generation"][slot]) != generation:
                return None
            snap = np.array(self._data[slot], dtype="<f8")
            if (
                int(self._slot_meta["seq"][slot]) == seq1
                and int(self._slot_meta["generation"][slot]) == generation
            ):
                return snap
        return None

    def _unpack(self, snap: "NDArray[Any]") -> tuple[dict, dict, float, bool]:
        ts = float(snap[0])
        vals = snap[1:]
        no = len(self._obs_keys)
        joint_obs = {k: float(vals[i]) for i, k in enumerate(self._obs_keys)}
        action = {k: float(vals[no + i]) for i, k in enumerate(self._action_keys)}
        intervention = bool(snap[-1] >= 0.5)
        return joint_obs, action, ts, intervention

    def close(self) -> None:
        self._header = None  # type: ignore[assignment]
        self._slot_meta = None  # type: ignore[assignment]
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
