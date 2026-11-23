"""IMU samples from a ZED camera, received by the MCAP recorder.

The ZED GStreamer plugin, built with Axol's IMU patch
(``almond_axol/cli/gst/patches/zed-gstreamer-imu-socket.patch``), sends
every IMU sample of a stereo camera to a Unix datagram socket named by its
``imu-socket`` property. This module binds that socket in the recorder
process and keeps the samples until the capture loop writes them.

Each datagram is one packed, little-endian sample (:data:`PACKET`): the magic
``AXIM``, a format version, the SDK's acquisition time in Unix nanoseconds,
acceleration (m/s², gravity included), angular velocity (deg/s) and the IMU
temperature (°C, NaN when unknown). The camera never waits for this reader:
a sample sent while nothing is bound, or while the queue is full, is lost.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

PACKET = struct.Struct("<4sIQ3f3ff")
MAGIC = b"AXIM"
VERSION = 1
# Samples older than this when they arrive are stale (the SDK can hand out an
# old sample right after the camera opens).
_STALE_S = 2.0
# Kept between polls: 20 s at the IMU's 200 Hz.
_MAX_SAMPLES = 4000
_RECV_BUFFER = 1 << 18


@dataclass(frozen=True)
class ImuSample:
    """One IMU sample, its time moved onto ``perf_counter``."""

    perf: float
    acceleration: tuple[float, float, float]  # m/s^2, gravity included
    angular_velocity: tuple[float, float, float]  # rad/s
    temperature: float


def parse_packet(data: bytes, wall_minus_perf: float) -> ImuSample | None:
    """One datagram as a sample, or ``None`` when it isn't one."""
    if len(data) != PACKET.size:
        return None
    magic, version, ts_ns, ax, ay, az, gx, gy, gz, temp = PACKET.unpack(data)
    if magic != MAGIC or version != VERSION or ts_ns == 0:
        return None
    values = (ax, ay, az, gx, gy, gz)
    if not all(math.isfinite(v) for v in values):
        return None
    return ImuSample(
        perf=ts_ns / 1e9 - wall_minus_perf,
        acceleration=(ax, ay, az),
        angular_velocity=tuple(math.radians(v) for v in (gx, gy, gz)),  # type: ignore[arg-type]
        temperature=temp,
    )


class ImuReceiver:
    """Binds ``path`` and collects samples on a thread of its own."""

    def __init__(self, path: str) -> None:
        self.path = path
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # a socket left by an earlier session
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER)
            self._sock.bind(path)
            self._sock.settimeout(0.2)
        except BaseException:
            self._sock.close()
            raise
        self._samples: deque[ImuSample] = deque(maxlen=_MAX_SAMPLES)
        self.received = 0
        self.rejected = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mcap-imu", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._sock.recv(PACKET.size + 64)
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    _logger.exception("IMU socket %s failed", self.path)
                return
            wall_minus_perf = time.time() - time.perf_counter()
            sample = parse_packet(data, wall_minus_perf)
            if sample is None or time.perf_counter() - sample.perf > _STALE_S:
                self.rejected += 1
                continue
            self.received += 1
            self._samples.append(sample)

    def drain(self) -> list[ImuSample]:
        """Every sample received since the last call, oldest first."""
        out = []
        while True:
            try:
                out.append(self._samples.popleft())
            except IndexError:
                return out

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._sock.close()
        self._thread.join(timeout=1.0)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)
