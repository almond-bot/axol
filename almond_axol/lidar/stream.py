"""Receive the lidar's point and IMU streams and cut them into frames.

:class:`LidarStream` owns two UDP sockets and one background thread. Per
packet the thread only reads the header timestamp (to know where a frame
ends) and keeps the raw bytes; when a frame's time window closes, its packets
are decoded together in one vectorised pass. At the Mid-360's ~2100 packets/s
that keeps the receive loop to a few percent of a core.

Timestamps: the lidar stamps every packet with its own clock (power-on time
unless PTP or GPS sync is set up). The stream maps that onto this host's
:func:`time.monotonic` clock by tracking the smallest seen ``arrival -
sensor`` offset, which is the network delay floor. Every frame and IMU sample
carries host-monotonic times, so they line up with Jelly's wheel samples and
the rest of Axol.
"""

from __future__ import annotations

import collections
import logging
import select
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from . import livox
from .config import LidarConfig

_logger = logging.getLogger(__name__)

# Socket receive buffer. The point stream is ~2.9 MB/s, so 8 MB rides out a
# second of the receive thread being descheduled. The kernel caps the request
# at net.core.rmem_max, which ``axol lidar.setup`` raises.
_RCVBUF = 8 * 1024 * 1024

# How long the clock-offset estimator remembers a minimum before letting it
# rise again (the two clocks drift by a few ppm, so the floor moves slowly).
_OFFSET_WINDOW_S = 10.0

_TS = struct.Struct("<Q")  # packet timestamp (ns) at offset 28


@dataclass
class LidarFrame:
    """One frame of points in the lidar frame.

    Attributes:
        stamp:      Host :func:`time.monotonic` time of the first point (s).
        sensor_ns:  Lidar-clock time of the first point (ns).
        xyz:        ``(n, 3)`` float32 points in metres.
        reflectivity: ``(n,)`` uint8, 0-255.
        tag:        ``(n,)`` uint8 Livox confidence tag.
        times:      ``(n,)`` float32 seconds of each point after ``stamp``,
                    for motion compensation.
        duration:   Seconds from the first to the last point.
        seq:        Frame counter since the stream started.
    """

    stamp: float
    sensor_ns: int
    xyz: np.ndarray
    reflectivity: np.ndarray
    tag: np.ndarray
    times: np.ndarray
    duration: float
    seq: int

    def __len__(self) -> int:
        return len(self.xyz)


@dataclass(frozen=True)
class ImuSample:
    """One IMU reading: gyro in rad/s and accel in m/s², lidar frame."""

    stamp: float
    gyro: np.ndarray
    accel: np.ndarray


@dataclass(frozen=True)
class StreamStats:
    packets_per_s: float
    points_per_s: float
    imu_per_s: float
    frames_per_s: float
    bad_packets: int
    dropped_frames: int
    last_packet_age: float | None


class _ClockMap:
    """Sensor-ns to host-monotonic-s mapping from the arrival-delay floor."""

    def __init__(self) -> None:
        self._best = None  # (offset_ns, window start)
        self._prev = None

    def update(self, sensor_ns: int, arrival_ns: int) -> None:
        offset = arrival_ns - sensor_ns
        now = arrival_ns
        if self._best is None or now - self._best[1] > _OFFSET_WINDOW_S * 1e9:
            self._prev = self._best
            self._best = (offset, now)
        elif offset < self._best[0]:
            self._best = (offset, self._best[1])

    @property
    def ready(self) -> bool:
        return self._best is not None

    def offset_ns(self) -> int:
        if self._best is None:
            return 0
        if self._prev is None:
            return self._best[0]
        return min(self._best[0], self._prev[0])

    def to_host(self, sensor_ns: int) -> float:
        return (sensor_ns + self.offset_ns()) * 1e-9


class FrameAssembler:
    """Collect raw point packets and emit time-sliced :class:`LidarFrame` s.

    Separate from the sockets so it can be fed recorded or simulated packets.
    """

    def __init__(
        self,
        period: float = 0.1,
        *,
        min_range: float = 0.0,
        max_range: float = float("inf"),
        clock: _ClockMap | None = None,
    ) -> None:
        self.period_ns = int(period * 1e9)
        self.min_range = min_range
        self.max_range = max_range
        self.clock = clock or _ClockMap()
        self._packets: list[bytes] = []
        self._start_ns: int | None = None
        self._last_ns: int = 0
        self._seq = 0
        self.bad_packets = 0

    def add(self, buf: bytes, arrival_ns: int | None = None) -> LidarFrame | None:
        """Add one raw point packet; returns a frame when this packet closes one."""
        if len(buf) < livox.PACKET_HEADER_SIZE or buf[10] not in (1, 2):
            self.bad_packets += 1
            return None
        interval, ts = buf[3] | (buf[4] << 8), _TS.unpack_from(buf, 28)[0]
        if arrival_ns is not None:
            self.clock.update(ts + interval * 100, arrival_ns)
        out = None
        if self._start_ns is None:
            self._start_ns = ts
        elif ts - self._start_ns >= self.period_ns or ts < self._last_ns:
            # A step backwards is a clock jump (sync just locked): start over.
            out = self.flush()
            self._start_ns = ts
        self._packets.append(buf)
        self._last_ns = ts
        return out

    def flush(self) -> LidarFrame | None:
        packets, self._packets = self._packets, []
        start = self._start_ns
        self._start_ns = None
        if not packets or start is None:
            return None
        frame = decode_packets(packets, start, self.clock, self._seq)
        self._seq += 1
        if frame is None:
            self.bad_packets += len(packets)
            return None
        rng = np.linalg.norm(frame.xyz, axis=1)
        keep = (rng >= max(self.min_range, 1e-3)) & (rng <= self.max_range)
        if not keep.all():
            frame.xyz = frame.xyz[keep]
            frame.reflectivity = frame.reflectivity[keep]
            frame.tag = frame.tag[keep]
            frame.times = frame.times[keep]
        return frame


def decode_packets(
    packets: list[bytes], start_ns: int, clock: _ClockMap, seq: int = 0
) -> LidarFrame | None:
    """Decode a list of same-type point packets into one frame (vectorised)."""
    data_type = packets[0][10]
    dtype, scale = (
        (livox.POINT_DTYPE_32, 1e-3) if data_type == 1 else (livox.POINT_DTYPE_16, 1e-2)
    )
    size = dtype.itemsize
    hdr = livox.PACKET_HEADER_SIZE
    counts = np.empty(len(packets), dtype=np.int64)
    stamps = np.empty(len(packets), dtype=np.int64)
    steps = np.empty(len(packets), dtype=np.float64)
    chunks = []
    for i, p in enumerate(packets):
        n = p[5] | (p[6] << 8)
        if p[10] != data_type or len(p) < hdr + n * size:
            return None
        counts[i] = n
        interval, ts = p[3] | (p[4] << 8), _TS.unpack_from(p, 28)[0]
        stamps[i] = ts
        steps[i] = interval * 100.0 / max(n - 1, 1)
        chunks.append(p[hdr : hdr + n * size])
    raw = np.frombuffer(b"".join(chunks), dtype=dtype)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32) * scale
    idx = np.arange(len(raw)) - np.repeat(np.cumsum(counts) - counts, counts)
    times_ns = np.repeat(stamps - start_ns, counts) + idx * np.repeat(steps, counts)
    zero = (raw["x"] == 0) & (raw["y"] == 0) & (raw["z"] == 0)
    keep = ~zero
    times = (times_ns * 1e-9).astype(np.float32)
    end = float(times.max()) if len(times) else 0.0
    return LidarFrame(
        stamp=clock.to_host(start_ns),
        sensor_ns=int(start_ns),
        xyz=np.ascontiguousarray(xyz[keep]),
        reflectivity=raw["reflectivity"][keep].copy(),
        tag=raw["tag"][keep].copy(),
        times=times[keep],
        duration=end,
        seq=seq,
    )


class LidarStream:
    """Background receiver for one lidar's point and IMU streams.

    Typical use::

        with LidarStream(LidarConfig.load()) as lidar:
            for frame in lidar.frames():
                print(len(frame), "points")

    ``configure=True`` (the default) points the lidar's streams at
    ``config.host_ip`` and starts sampling, so the lidar's stored settings
    don't matter. ``on_frame`` is called on the receive thread; keep it
    short, or read :meth:`frames` / :meth:`latest` from another thread.
    """

    def __init__(
        self,
        config: LidarConfig | None = None,
        *,
        configure: bool = True,
        on_frame: Callable[[LidarFrame], None] | None = None,
        on_imu: Callable[[ImuSample], None] | None = None,
        imu_history: int = 2000,
        queue_size: int = 4,
    ) -> None:
        self.config = config or LidarConfig.load()
        self.configure = configure
        self._on_frame = on_frame
        self._on_imu = on_imu
        self._clock = _ClockMap()
        self._assembler = FrameAssembler(
            self.config.frame_period,
            min_range=self.config.min_range,
            max_range=self.config.max_range,
            clock=self._clock,
        )
        self._imu: collections.deque[ImuSample] = collections.deque(maxlen=imu_history)
        self._queue: collections.deque[LidarFrame] = collections.deque(
            maxlen=queue_size
        )
        self._cond = threading.Condition()
        self._latest: LidarFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socks: list[socket.socket] = []
        self.status: livox.LidarStatus | None = None
        self.error: BaseException | None = None
        self.dropped_frames = 0
        self._counts = collections.deque(maxlen=64)  # (t, packets, points, imu, frames)
        self._n = [0, 0, 0, 0]
        self._last_packet: float | None = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> LidarStream:
        cfg = self.config
        if not cfg.lidar_ip or not cfg.host_ip:
            raise RuntimeError(
                "The lidar's address isn't set up. Run `axol lidar.setup` first."
            )
        bind_ip = cfg.host_ip
        points = self._bind(bind_ip, cfg.point_port)
        imu = self._bind(bind_ip, cfg.imu_port)
        self._socks = [points, imu]
        if self.configure:
            with livox.LivoxClient(cfg.lidar_ip, cfg.host_ip) as client:
                client.configure_streams(
                    cfg.host_ip, point_port=cfg.point_port, imu_port=cfg.imu_port
                )
                self.status = client.status()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(points, imu), name="lidar-rx", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None
        for s in self._socks:
            s.close()
        self._socks = []
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> LidarStream:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @staticmethod
    def _bind(ip: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RCVBUF)
        except OSError:
            pass
        try:
            s.bind((ip, port))
        except OSError as exc:
            s.close()
            raise RuntimeError(
                f"Can't listen on {ip}:{port} ({exc.strerror}). Check that "
                f"{ip} is assigned to the lidar's port (`axol lidar.setup`) and "
                "that no other lidar program is running."
            ) from exc
        return s

    # -- consumers -----------------------------------------------------

    def latest(self) -> LidarFrame | None:
        """The newest frame, or None before the first one."""
        return self._latest

    def get(self, timeout: float | None = None) -> LidarFrame | None:
        """Pop the oldest queued frame, waiting up to ``timeout`` s."""
        with self._cond:
            if not self._queue:
                self._cond.wait(timeout)
            return self._queue.popleft() if self._queue else None

    def frames(self, timeout: float = 2.0) -> Iterator[LidarFrame]:
        """Yield frames as they arrive; raises if the lidar goes silent."""
        while not self._stop.is_set():
            frame = self.get(timeout)
            if frame is None:
                if self._stop.is_set():
                    return
                if self.error is not None:
                    raise RuntimeError(f"Lidar receive failed: {self.error}")
                raise TimeoutError(
                    f"No lidar frames for {timeout:.0f} s. Check the lidar's power "
                    "and cable, then run `axol lidar.info`."
                )
            yield frame

    def imu_between(self, t0: float, t1: float) -> list[ImuSample]:
        """IMU samples with ``t0 <= stamp < t1`` (host-monotonic seconds)."""
        return [s for s in list(self._imu) if t0 <= s.stamp < t1]

    def imu_recent(self, n: int = 200) -> list[ImuSample]:
        return list(self._imu)[-n:]

    def stats(self) -> StreamStats:
        counts = list(self._counts)
        now = time.monotonic()
        if len(counts) >= 2:
            t0, *a = counts[0]
            t1, *b = counts[-1]
            dt = max(t1 - t0, 1e-6)
            rates = [(y - x) / dt for x, y in zip(a, b, strict=True)]
        else:
            rates = [0.0, 0.0, 0.0, 0.0]
        return StreamStats(
            packets_per_s=rates[0],
            points_per_s=rates[1],
            imu_per_s=rates[2],
            frames_per_s=rates[3],
            bad_packets=self._assembler.bad_packets,
            dropped_frames=self.dropped_frames,
            last_packet_age=None
            if self._last_packet is None
            else now - self._last_packet,
        )

    # -- receive thread ------------------------------------------------

    def _emit(self, frame: LidarFrame) -> None:
        self._latest = frame
        self._n[3] += 1
        with self._cond:
            if len(self._queue) == self._queue.maxlen:
                self.dropped_frames += 1
            self._queue.append(frame)
            self._cond.notify_all()
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:  # noqa: BLE001 - a consumer bug must not kill rx
                _logger.exception("lidar frame callback failed")

    def _run(self, points: socket.socket, imu: socket.socket) -> None:
        try:
            last_tick = 0.0
            while not self._stop.is_set():
                ready, _, _ = select.select([points, imu], [], [], 0.2)
                now_ns = time.monotonic_ns()
                if points in ready:
                    for _ in range(64):
                        try:
                            buf = points.recv(2048, socket.MSG_DONTWAIT)
                        except BlockingIOError:
                            break
                        self._n[0] += 1
                        self._n[1] += buf[5] | (buf[6] << 8) if len(buf) > 6 else 0
                        frame = self._assembler.add(buf, now_ns)
                        if frame is not None:
                            self._emit(frame)
                    self._last_packet = now_ns * 1e-9
                if imu in ready:
                    for _ in range(16):
                        try:
                            buf = imu.recv(2048, socket.MSG_DONTWAIT)
                        except BlockingIOError:
                            break
                        try:
                            pkt = livox.parse_imu_packet(buf)
                        except livox.LivoxError:
                            continue
                        self._n[2] += 1
                        if not self._clock.ready:
                            continue
                        sample = ImuSample(
                            self._clock.to_host(pkt.timestamp_ns), pkt.gyro, pkt.accel
                        )
                        self._imu.append(sample)
                        if self._on_imu is not None:
                            try:
                                self._on_imu(sample)
                            except Exception:  # noqa: BLE001
                                _logger.exception("lidar IMU callback failed")
                t = now_ns * 1e-9
                if t - last_tick >= 0.25:
                    last_tick = t
                    self._counts.append((t, *self._n))
        except Exception as exc:  # noqa: BLE001 - surfaced through frames()
            self.error = exc
            _logger.exception("lidar receive thread stopped")
            with self._cond:
                self._cond.notify_all()
