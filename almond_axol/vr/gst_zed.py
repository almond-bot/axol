"""GPU-resident ZED camera relay for the headset (Jetson + zed-gstreamer).

The SDK relay path (``ZedCamera`` -> numpy -> pipe -> NVENC, see
``hw_video``) costs ~26 ms per frame on a Zed Box because every frame is
copied to host memory, into Python, and back over a pipe. Stereolabs'
``zedxonesrc`` GStreamer element grabs frames straight into NVMM (GPU)
memory as NV12, so the whole grab -> convert -> NVENC chain stays on the
GPU; measured grab-to-encoded-AU latency is ~4.5 ms. Python only ever sees
the *encoded* H.264 byte stream, which the WebRTC track forwards as
pre-encoded packets (aiortc's ``encoder.pack`` path).

One :class:`ZedXOneGstStream` per camera owns the ``gst-launch-1.0``
subprocess and broadcasts encoded access units to any number of
subscriber queues (one per WebRTC track). The camera is exclusively owned
by the pipeline — the ZED SDK cannot open it at the same time, which is
why teleop (relay-only) uses this path while collect-data and inference
keep the SDK.

Requires the zed-gstreamer plugins (https://github.com/stereolabs/zed-gstreamer)
and the Jetson ``nvv4l2h264enc`` element; :func:`zed_gst_available` gates use.
"""

from __future__ import annotations

import functools
import logging
import queue
import shutil
import subprocess
import threading

from .hw_video import _bitrate_for, hw_h264_available, read_annexb_aus

_logger = logging.getLogger(__name__)

# zedxonesrc camera-resolution enum values (GstZedXOneSrcResol), keyed by the
# resolution names used in ZedCameraConfig / ZED_RESOLUTION_DIMS.
_RESOLUTION_ENUM: dict[str, int] = {
    "SVGA": 0,
    "HD1080": 1,
    "HD1200": 2,
}

_RESOLUTION_DIMS: dict[str, tuple[int, int]] = {
    "SVGA": (960, 600),
    "HD1080": (1920, 1080),
    "HD1200": (1920, 1200),
}

# Per-subscriber AU queue depth. A healthy consumer pops every AU as it
# arrives; the bound only matters for a stalled consumer, where the oldest
# AUs are dropped so a backlog can never become latency.
_SUBSCRIBER_QUEUE_DEPTH = 4

# How long the camera may take to open and produce its first access unit
# (the daemon handshake plus sensor start is a few seconds).
_READY_TIMEOUT_S = 15.0

# Cap on auto-exposure time (µs). Exposure happens before the frame exists,
# so it is pure glass-to-glass latency: the SDK default lets auto-exposure
# run to 66.7 ms (4 frame intervals at 60 fps!) in dim light. 8 ms keeps
# capture latency bounded; auto analog/digital gain stays enabled and
# compensates the brightness (at the cost of some noise in dim scenes).
_MAX_AUTO_EXPOSURE_US = 8000


@functools.cache
def zed_gst_available() -> bool:
    """True when the zed-gstreamer ``zedxonesrc`` element and NVENC exist."""
    if not hw_h264_available():
        return False
    inspect = shutil.which("gst-inspect-1.0")
    if not inspect:
        return False
    try:
        ok = (
            subprocess.run(
                [inspect, "zedxonesrc"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - inspection failure → no native path
        return False
    if ok:
        _logger.info("zed-gstreamer native camera pipeline (zedxonesrc) available")
    return ok


def _gst_argv(serial: int, resolution: str, fps: int, bitrate: int) -> list[str]:
    """``gst-launch-1.0`` argv: camera grab -> NVENC -> H.264 AUs on stdout.

    ``stream-type=1`` selects the zero-copy NV12 output, which is already
    ``memory:NVMM`` (GPU) — the encoder consumes it directly, with no
    ``nvvidconv`` or host round trip. ``insert-aud=true`` delimits access
    units for the reader; ``idrinterval`` bounds keyframe spacing to 1 s
    (``gst-launch`` offers no runtime keyframe control, so that is also the
    worst-case PLI recovery and late-joiner startup delay).
    """
    return [
        "gst-launch-1.0",
        "-q",
        "zedxonesrc",
        f"camera-sn={serial}",
        f"camera-resolution={_RESOLUTION_ENUM[resolution]}",
        f"camera-fps={fps}",
        "stream-type=1",
        f"ctrl-auto-exposure-range-max={_MAX_AUTO_EXPOSURE_US}",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "nvv4l2h264enc",
        "control-rate=1",
        f"bitrate={bitrate}",
        "preset-level=1",
        "insert-sps-pps=true",
        "insert-aud=true",
        f"idrinterval={fps}",
        "maxperf-enable=true",
        "!",
        "video/x-h264,stream-format=byte-stream",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


class ZedXOneGstStream:
    """One camera's GPU-resident grab->encode pipeline with AU fan-out.

    WebRTC tracks call :meth:`subscribe` for a queue of encoded access
    units (each a ``list[bytes]`` of NALs, Annex-B start codes stripped).
    The object satisfies the duck type ``WebRTCManager`` recognizes as a
    pre-encoded source (``subscribe`` / ``unsubscribe`` / ``alive``).
    """

    def __init__(self, serial: int, resolution: str = "HD1200", fps: int = 60) -> None:
        if resolution not in _RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported ZED X One resolution {resolution!r} "
                f"(expected one of {', '.join(_RESOLUTION_ENUM)})"
            )
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        width, height = _RESOLUTION_DIMS[resolution]
        bitrate = _bitrate_for(width, height, fps)
        self._subscribers: list[queue.Queue[list[bytes]]] = []
        self._lock = threading.Lock()
        self._first_au = threading.Event()
        self._stderr_tail: list[str] = []
        self._proc = subprocess.Popen(
            _gst_argv(serial, resolution, fps, bitrate),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        _logger.info(
            "zed gst pipeline started: sn=%d %s@%dfps %.1f Mbps (pid %d)",
            serial,
            resolution,
            fps,
            bitrate / 1e6,
            self._proc.pid,
        )
        threading.Thread(
            target=self._read_loop, name=f"zed-gst-{serial}", daemon=True
        ).start()
        threading.Thread(
            target=self._stderr_loop, name=f"zed-gst-{serial}-err", daemon=True
        ).start()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def wait_ready(self, timeout: float = _READY_TIMEOUT_S) -> bool:
        """Block until the first encoded AU arrives (camera is streaming).

        Polls in small steps so a pipeline that exits (camera absent or
        already in use) fails fast instead of sitting out the timeout.
        """
        waited = 0.0
        while waited < timeout:
            if self._first_au.wait(0.25):
                return True
            waited += 0.25
            if not self.alive:
                break
        if self._stderr_tail:
            _logger.warning(
                "zed gst pipeline sn=%d not ready: %s",
                self.serial,
                " | ".join(self._stderr_tail[-3:]),
            )
        return False

    def subscribe(self) -> queue.Queue[list[bytes]]:
        """Register a consumer; returns its queue of encoded AUs."""
        q: queue.Queue[list[bytes]] = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_DEPTH)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[list[bytes]]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)  # reap — no zombies
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

    # ZedCamera-compatible alias so teleop's cleanup loop can stay generic.
    disconnect = close

    def __del__(self) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _broadcast(self, au: list[bytes]) -> None:
        self._first_au.set()
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            while True:
                try:
                    q.put_nowait(au)
                    break
                except queue.Full:
                    # Stalled consumer: drop its oldest AU so the queue
                    # always holds the freshest video.
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        read_annexb_aus(self._proc.stdout.fileno(), self._broadcast, lambda: self.alive)

    def _stderr_loop(self) -> None:
        """Keep the last few stderr lines for open-failure diagnostics."""
        assert self._proc.stderr is not None
        try:
            for raw in self._proc.stderr:
                line = raw.decode(errors="replace").strip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-10]
                    _logger.debug("zed gst sn=%d: %s", self.serial, line)
        except Exception:  # noqa: BLE001 - fd closed during teardown
            pass
