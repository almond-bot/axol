"""Shared-memory transport for raw camera frames across the relay boundary.

``collect-data`` needs the ZED cameras' raw frames in the **control** process
(to write the dataset), but running the camera grab + NVENC encode + aiortc
WebRTC in that process starves the teleop/IK loops (see
:mod:`almond_axol.vr.video_proc`). The relay subprocess therefore owns the
cameras and does all the heavy work; this module ships the raw RGB frames it
produces back to the control process through ``multiprocessing`` shared memory,
so the control process only ever copies a frame out of shared memory at the
60 Hz capture rate while recording — never on the hot control path.

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

import time
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
