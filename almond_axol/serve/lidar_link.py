"""Live lidar scans for the control panel.

:class:`LidarHub` reads the lidar while at least one browser watches it
(``/api/lidar/ws``), turns each frame into the robot's base frame (floor at
z = 0, +x forward), thins it and fans it out as a compact binary message.
It stops reading a few seconds after the last viewer leaves, unless another
program on this computer is reading the lidar through it (see
:mod:`almond_axol.lidar.stream`, which also lets this hub read a copy while
``nav.*`` or ``lidar.view`` has the lidar).

Messages on the socket:

- binary, one per lidar frame (~10 per second)::

    b"LDR1" | seq u32 | n u32 | xyz int16[n*3] (mm) | reflectivity u8[n]

  little-endian, points in the base frame.

- JSON, about once a second::

    {"type": "status", "state": "starting" | "streaming" | "error",
     "error": str | None, "shared": bool, "pointsPerSecond": float,
     "mount": {"x": .., "y": .., "z": .., "roll": .., "pitch": .., "yaw": ..}}
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..lidar.config import LidarConfig, config_path
from ..lidar.stream import LidarFrame, LidarStream

_logger = logging.getLogger(__name__)

MAGIC = b"LDR1"
_HEADER = struct.Struct("<4sII")
DEFAULT_POINTS = 12_000
MAX_POINTS = 30_000
MIN_POINTS = 1_000
# Coordinates must fit int16 millimetres.
_LIMIT_M = 32.0
# Seconds to keep reading after the last viewer leaves, so a page reload
# doesn't restart the stream.
LINGER_S = 3.0
RETRY_S = 3.0
STATUS_PERIOD_S = 1.0


def encode_frame(
    seq: int, xyz_base: np.ndarray, reflectivity: np.ndarray, max_points: int
) -> bytes:
    """Pack one base-frame frame into the binary message, thinned to ``max_points``."""
    keep = np.all(np.abs(xyz_base) < _LIMIT_M, axis=1)
    xyz = xyz_base[keep]
    refl = reflectivity[keep]
    n = len(xyz)
    if n > max_points:
        idx = (np.arange(max_points) * (n / max_points)).astype(np.int64)
        xyz, refl = xyz[idx], refl[idx]
        n = max_points
    mm = np.round(xyz * 1000.0).astype("<i2")
    return (
        _HEADER.pack(MAGIC, seq & 0xFFFFFFFF, n)
        + mm.tobytes()
        + refl.astype(np.uint8).tobytes()
    )


def decode_frame(buf: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_frame` (for tests and tools)."""
    magic, seq, n = _HEADER.unpack_from(buf)
    if magic != MAGIC:
        raise ValueError("not a lidar frame")
    off = _HEADER.size
    xyz = np.frombuffer(buf, "<i2", n * 3, off).reshape(n, 3).astype(np.float32) / 1000
    refl = np.frombuffer(buf, np.uint8, n, off + n * 6)
    return seq, xyz, refl


def lidar_configured(path: Any = None) -> bool:
    """Whether ``axol lidar.setup`` has run on this computer."""
    p = path or config_path()
    if not p.exists():
        return False
    try:
        cfg = LidarConfig.load(p)
    except (OSError, ValueError):
        return False
    return bool(cfg.lidar_ip and cfg.host_ip)


@dataclasses.dataclass(eq=False)
class LidarSubscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[Any]
    max_points: int


def _offer(queue: asyncio.Queue[Any], message: Any) -> None:
    """Queue a message, dropping the oldest one for a slow viewer."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(message)


class LidarHub:
    """Reads the lidar on demand and fans frames out to viewers."""

    def __init__(
        self,
        *,
        load_config: Callable[[], LidarConfig] = LidarConfig.load,
        stream_factory: Callable[..., LidarStream] = LidarStream,
        linger: float = LINGER_S,
        retry: float = RETRY_S,
    ) -> None:
        self._load_config = load_config
        self._stream_factory = stream_factory
        self._linger = linger
        self._retry = retry
        self._lock = threading.Lock()
        self._subs: list[LidarSubscriber] = []
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._closed = False
        self._state = "starting"
        self._error: str | None = None
        self._stream: LidarStream | None = None
        self._mount = np.eye(4)
        self._mount_json: dict[str, float] = {}
        self._seq = 0

    # -- viewers -----------------------------------------------------------

    def subscribe(self, max_points: int = DEFAULT_POINTS) -> LidarSubscriber:
        sub = LidarSubscriber(
            asyncio.get_running_loop(),
            asyncio.Queue(maxsize=4),
            int(min(max(max_points, MIN_POINTS), MAX_POINTS)),
        )
        with self._lock:
            self._subs.append(sub)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._supervise, name="lidar-hub", daemon=True
                )
                self._thread.start()
        _offer(sub.queue, self.status())
        self._wake.set()
        return sub

    def unsubscribe(self, sub: LidarSubscriber) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        self._wake.set()

    def status(self) -> dict[str, Any]:
        stream = self._stream
        pps = 0.0
        if stream is not None and self._state == "streaming":
            pps = round(stream.stats().points_per_s)
        return {
            "type": "status",
            "state": self._state,
            "error": self._error,
            "shared": bool(stream is not None and stream.shared),
            "pointsPerSecond": pps,
            "mount": self._mount_json,
        }

    def shutdown(self) -> None:
        self._closed = True
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(3.0)

    # -- worker ------------------------------------------------------------

    def _broadcast(self, message: Any) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            try:
                sub.loop.call_soon_threadsafe(_offer, sub.queue, message)
            except RuntimeError:
                pass  # the viewer's loop has closed

    def _set_state(self, state: str, error: str | None = None) -> None:
        changed = (state, error) != (self._state, self._error)
        self._state, self._error = state, error
        if changed:
            self._broadcast(self.status())

    def _on_frame(self, frame: LidarFrame) -> None:
        with self._lock:
            if not self._subs:
                return
            cap = max(s.max_points for s in self._subs)
        rot, trans = self._mount[:3, :3], self._mount[:3, 3]
        xyz = frame.xyz @ rot.T.astype(np.float32) + trans.astype(np.float32)
        self._seq += 1
        self._broadcast(encode_frame(self._seq, xyz, frame.reflectivity, cap))

    def _wanted(self) -> bool:
        with self._lock:
            if self._subs:
                return True
        stream = self._stream
        return stream is not None and stream.tap_readers() > 0

    def _supervise(self) -> None:
        idle_since: float | None = None
        next_try = 0.0
        last_status = 0.0
        try:
            while not self._closed:
                now = time.monotonic()
                if self._wanted():
                    idle_since = None
                elif idle_since is None:
                    idle_since = now
                elif now - idle_since >= self._linger:
                    break
                if self._stream is None and now >= next_try and idle_since is None:
                    self._open()
                    if self._stream is None:
                        next_try = time.monotonic() + self._retry
                stream = self._stream
                if stream is not None:
                    if stream.error is not None:
                        self._close_stream()
                        self._set_state(
                            "error", f"Lidar receive failed: {stream.error}"
                        )
                        next_try = time.monotonic() + self._retry
                    else:
                        stats = stream.stats()
                        age = stats.last_packet_age
                        if age is not None and age < 1.0:
                            self._set_state("streaming")
                        elif age is not None or now - self._opened_at > 3.0:
                            self._set_state(
                                "error",
                                "No data from the lidar. Check its power and cable.",
                            )
                if now - last_status >= STATUS_PERIOD_S:
                    last_status = now
                    self._broadcast(self.status())
                self._wake.wait(0.25)
                self._wake.clear()
        finally:
            self._close_stream()
            self._state, self._error = "starting", None

    def _open(self) -> None:
        self._set_state("starting")
        try:
            config = self._load_config()
            if not (config.lidar_ip and config.host_ip):
                raise RuntimeError("The lidar isn't set up. Run `axol lidar.setup`.")
            self._mount = config.mount.matrix()
            self._mount_json = {
                k: round(float(v), 4)
                for k, v in dataclasses.asdict(config.mount).items()
            }
            stream = self._stream_factory(config, on_frame=self._on_frame, queue_size=1)
            stream.start()
        except Exception as exc:  # noqa: BLE001 - surfaced to the viewer
            _logger.info("lidar view: %s", exc)
            self._set_state("error", _friendly(exc))
            return
        self._stream = stream
        self._opened_at = time.monotonic()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                _logger.exception("stopping the lidar stream failed")

    _opened_at = 0.0


def _friendly(exc: BaseException) -> str:
    text = str(exc)
    if isinstance(exc, TimeoutError) or "timed out" in text.lower():
        return "The lidar didn't answer. Check its power and cable."
    return text or type(exc).__name__
