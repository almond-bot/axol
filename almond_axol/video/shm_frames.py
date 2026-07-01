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

import threading
import time
from collections import deque
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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


def _au_has_coded_slice(au: bytes) -> bool:
    """True if the Annex-B access unit contains a VCL (coded-picture) NAL.

    Integrity guard for the one-AU-per-row contract: the relay's encoder could
    emit an access unit carrying only non-VCL NALs (access-unit delimiter / SPS /
    PPS / SEI / end-of-sequence) with no coded slice — e.g. a boundary AU when
    the dataset valve closes. Such an AU decodes to *no* picture, so muxing it as
    a dataset frame would occupy a PTS slot without yielding a retrievable frame
    and desync frame-count from row-count. Delivering only AUs with a coded slice
    keeps them aligned (the capture loop re-muxes the previous real frame for a
    starved row instead). VCL NAL types are 1-5 (non-IDR .. IDR).

    Note: this is *not* the fix for the observed last-row failure — that was a
    mux/decode tail artifact (the final muxed sample is not timestamp-retrievable
    even for an all-VCL stream), handled by the trailing guard frame in
    :meth:`~almond_axol.lerobot.h264_mux_encoder._CameraH264Muxer.finish`.
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
    misalignment — video and joints both start there). ``recv_ts`` is
    ``perf_counter`` at pull time
    (shared-clock across processes), used only to pair the frame with the nearest
    joint snapshot — the mp4's own timeline is the constant-fps PTS the muxer
    assigns, independent of ``recv_ts``.
    """

    def __init__(self, socket_path: str, width: int, height: int, fps: int) -> None:
        from .gst_zed import _require_gst

        self._gst, _ = _require_gst()
        self.width = width
        self.height = height
        self.fps = fps
        self._queue: deque[tuple[bytes, float]] = deque()
        self._cond = threading.Condition()
        self._await_keyframe = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sink: Any = None
        # shmsrc carries no caps, so the downstream capsfilter must be fully fixed
        # (width/height/framerate as well as the h264 layout) or it won't link;
        # h264parse re-derives the real dimensions from the SPS regardless.
        # drop=false: never discard an AU (it would break H.264 decode); the pull
        # thread keeps the appsink drained so it rarely back-pressures shmsrc.
        caps = (
            f"video/x-h264,stream-format=byte-stream,alignment=au,"
            f"width={width},height={height},framerate={fps}/1"
        )
        self._pipeline = self._gst.parse_launch(
            f"shmsrc socket-path={socket_path} is-live=true do-timestamp=false "
            f"! {caps} ! h264parse "
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
        self._sink = self._pipeline.get_by_name("au")
        self._pipeline.set_state(self._gst.State.PLAYING)
        self._thread = threading.Thread(
            target=self._pull_loop, name="recorder-au-shmsrc", daemon=True
        )
        self._thread.start()

    def flush(self) -> None:
        """Drop any queued AUs and re-arm keyframe-wait (call at episode start).

        Between episodes the relay's valve is shut so nothing is produced; on the
        next episode it opens the valve and forces an IDR. Clearing here discards
        any stragglers so the episode's first delivered AU is that fresh IDR.
        """
        with self._cond:
            self._queue.clear()
            self._await_keyframe = True

    def _pull_loop(self) -> None:
        Gst = self._gst
        while not self._stop.is_set():
            sample = self._sink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                continue  # valve shut (not recording) or starting up — idle
            recv_perf = time.perf_counter()
            buf = sample.get_buffer()
            is_keyframe = not buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
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
            with self._cond:
                if self._await_keyframe:
                    if not is_keyframe:
                        continue  # wait for the episode's first IDR
                    self._await_keyframe = False
                self._queue.append((au, recv_perf))
                self._cond.notify()

    def read_next_au(self, timeout_ms: float = 500) -> tuple[bytes, float]:
        """Pop the next access unit in order; block up to ``timeout_ms``.

        Returns ``(au_bytes, recv_ts)``. Raises :class:`TimeoutError` if no AU
        arrives in time (the caller re-muxes the previous AU to keep frame counts
        aligned across cameras).
        """
        deadline = time.perf_counter() + timeout_ms / 1000.0
        with self._cond:
            while not self._queue:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        f"encoded-AU reader timed out after {timeout_ms:.1f}ms."
                    )
                self._cond.wait(remaining)
            return self._queue.popleft()

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
            self._pipeline = None
        self._sink = None

    # camera-compatible alias.
    close = disconnect


class SnapshotWriter:
    """Control-process side: publish the latest joint/action snapshot.

    A lock-free single-slot seqlock over one shared-memory block of float64s
    (``[ts, *joint_obs_vals, *action_vals]`` in fixed ``obs_keys`` / ``action_keys``
    order). The control loop calls :meth:`write` every tick; the cost is a handful
    of dict lookups + an aligned float store — no pickle, no lock, no blocking, so
    it stays off the hot path. The recorder subprocess reads it via
    :class:`SnapshotReader`. Single-writer / single-reader.
    """

    def __init__(self, obs_keys: list[str], action_keys: list[str]) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        n = len(self._obs_keys) + len(self._action_keys)
        self._shm = shared_memory.SharedMemory(
            create=True, size=_SNAP_HEADER_BYTES + 8 * (1 + n)
        )
        self.name = self._shm.name
        self._meta = np.ndarray((1,), dtype=_SNAP_META_DTYPE, buffer=self._shm.buf)
        self._data = np.ndarray(
            (1 + n,), dtype="<f8", buffer=self._shm.buf, offset=_SNAP_HEADER_BYTES
        )
        self._meta["seq"][0] = 0

    def write(self, joint_obs: dict, action: dict, ts: float) -> None:
        """Pack one snapshot into the slot (seq odd while writing, even when done)."""
        self._meta["seq"][0] += 1  # odd: write in progress
        d = self._data
        d[0] = ts
        i = 1
        for k in self._obs_keys:
            d[i] = joint_obs[k]
            i += 1
        for k in self._action_keys:
            d[i] = action[k]
            i += 1
        self._meta["seq"][0] += 1  # even: committed

    def close(self) -> None:
        self._meta = None  # type: ignore[assignment]
        self._data = None  # type: ignore[assignment]
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class SnapshotReader:
    """Recorder-subprocess side: read the latest joint/action snapshot.

    Attaches to a :class:`SnapshotWriter`'s block by name and reconstructs the
    ``(joint_obs, action, ts)`` dicts using the same key order. Returns ``None``
    before the first write (mirroring the in-process ``_SnapshotPublisher.latest``
    contract), so the caller can skip a tick just as it did before.
    """

    def __init__(self, name: str, obs_keys: list[str], action_keys: list[str]) -> None:
        self._obs_keys = list(obs_keys)
        self._action_keys = list(action_keys)
        n = len(self._obs_keys) + len(self._action_keys)
        self._shm = shared_memory.SharedMemory(name=name)
        self._meta = np.ndarray((1,), dtype=_SNAP_META_DTYPE, buffer=self._shm.buf)
        self._data = np.ndarray(
            (1 + n,), dtype="<f8", buffer=self._shm.buf, offset=_SNAP_HEADER_BYTES
        )

    def read_latest(self) -> tuple[dict, dict, float] | None:
        # SPSC seqlock: retry while the writer is mid-write (odd seq) or laps us
        # (seq changed across the copy). Writes are sub-microsecond so this
        # converges on the first try in practice; bail to None on the rare miss.
        for _ in range(8):
            s1 = int(self._meta["seq"][0])
            if s1 == 0:
                return None  # no snapshot published yet
            if s1 & 1:
                continue  # writer mid-write
            snap = np.array(self._data, dtype="<f8")
            if int(self._meta["seq"][0]) != s1:
                continue  # writer lapped us mid-copy
            ts = float(snap[0])
            vals = snap[1:]
            no = len(self._obs_keys)
            joint_obs = {k: float(vals[i]) for i, k in enumerate(self._obs_keys)}
            action = {k: float(vals[no + i]) for i, k in enumerate(self._action_keys)}
            return joint_obs, action, ts
        return None

    def close(self) -> None:
        self._meta = None  # type: ignore[assignment]
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
