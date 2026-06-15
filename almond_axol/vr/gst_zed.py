"""GPU-resident ZED camera relay for the headset (Jetson + zed-gstreamer).

The SDK relay path (``ZedCamera`` -> numpy -> pipe -> NVENC, see
``hw_video``) costs ~26 ms per frame on a Zed Box because every frame is
copied to host memory, into Python, and back over a pipe. Stereolabs'
``zedxonesrc`` GStreamer element grabs frames straight into NVMM (GPU)
memory as NV12, so the whole grab -> convert -> NVENC chain stays on the
GPU; measured grab-to-encoded-AU latency is ~4.5 ms. Python only ever sees
the *encoded* H.264 byte stream, which the WebRTC track forwards as
pre-encoded packets (aiortc's ``encoder.pack`` path).

One :class:`ZedXOneGstStream` per mono camera owns the ``gst-launch-1.0``
subprocess and broadcasts encoded access units to any number of
subscriber queues (one per WebRTC track). A stereo ZED X uses
:class:`ZedXStereoGstStream`: ``zedsrc`` delivers both eyes side-by-side
in one zero-copy NV12 buffer, which a ``tee`` splits into two GPU crops,
each encoded to its own ``overhead_left`` / ``overhead_right`` channel.
The camera is exclusively owned by the pipeline — the ZED SDK cannot open
it at the same time, which is why teleop (relay-only) uses this path while
collect-data and inference keep the SDK.

Requires the zed-gstreamer plugins (https://github.com/stereolabs/zed-gstreamer)
and the Jetson ``nvv4l2h264enc`` element; :func:`zed_gst_available` (mono)
and :func:`zed_stereo_gst_available` (stereo) gate use.
"""

from __future__ import annotations

import functools
import logging
import os
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable

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

# zedsrc camera-resolution enum values (GstZedSrcRes) for the stereo ZED X.
# Only the GMSL2 60-fps-capable modes are exposed; SVGA is ZED-X-One-only.
_STEREO_RESOLUTION_ENUM: dict[str, int] = {
    "HD1080": 1,
    "HD1200": 2,
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


def _element_available(element: str) -> bool:
    """True when ``gst-inspect-1.0`` finds the named GStreamer element."""
    inspect = shutil.which("gst-inspect-1.0")
    if not inspect:
        return False
    try:
        return (
            subprocess.run(
                [inspect, element],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - inspection failure → treat as absent
        return False


@functools.cache
def zed_gst_available() -> bool:
    """True when the zed-gstreamer ``zedxonesrc`` element and NVENC exist."""
    if not hw_h264_available():
        return False
    ok = _element_available("zedxonesrc")
    if ok:
        _logger.info("zed-gstreamer native camera pipeline (zedxonesrc) available")
    return ok


@functools.cache
def zed_stereo_gst_available() -> bool:
    """True when the stereo ``zedsrc`` element and NVENC are both present."""
    if not hw_h264_available():
        return False
    ok = _element_available("zedsrc")
    if ok:
        _logger.info("zed-gstreamer native stereo pipeline (zedsrc) available")
    return ok


class _AUChannel:
    """Fan-out of one H.264 stream's access units to subscriber queues.

    Satisfies the pre-encoded source duck type ``WebRTCManager`` /
    ``PrecodedVideoTrack`` expect (``subscribe`` / ``unsubscribe`` /
    ``alive``). A mono camera has one channel; a stereo camera has one per
    eye, each fed by its own encoder branch.
    """

    def __init__(self, alive: Callable[[], bool]) -> None:
        self._alive = alive
        self._subscribers: list[queue.Queue[list[bytes]]] = []
        self._lock = threading.Lock()
        self.first_au = threading.Event()

    @property
    def alive(self) -> bool:
        return self._alive()

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

    def broadcast(self, au: list[bytes]) -> None:
        """Push one access unit to every subscriber (drop-oldest if full)."""
        self.first_au.set()
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
        self._channel = _AUChannel(lambda: self.alive)
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
        if _wait_first_au((self._channel,), lambda: self.alive, timeout):
            return True
        if self._stderr_tail:
            _logger.warning(
                "zed gst pipeline sn=%d not ready: %s",
                self.serial,
                " | ".join(self._stderr_tail[-3:]),
            )
        return False

    def subscribe(self) -> queue.Queue[list[bytes]]:
        """Register a consumer; returns its queue of encoded AUs."""
        return self._channel.subscribe()

    def unsubscribe(self, q: queue.Queue[list[bytes]]) -> None:
        self._channel.unsubscribe(q)

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

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        read_annexb_aus(
            self._proc.stdout.fileno(), self._channel.broadcast, lambda: self.alive
        )

    def _stderr_loop(self) -> None:
        """Keep the last few stderr lines for open-failure diagnostics."""
        _drain_stderr(self._proc, self._stderr_tail, self.serial)


def _wait_first_au(
    channels: tuple[_AUChannel, ...],
    alive: Callable[[], bool],
    timeout: float,
) -> bool:
    """Block until every channel has produced its first AU (or pipeline dies).

    Polls in small steps so a pipeline that exits (camera absent or already
    in use) fails fast instead of sitting out the whole timeout.
    """
    waited = 0.0
    while waited < timeout:
        if all(ch.first_au.is_set() for ch in channels):
            return True
        # Wait on the first not-yet-ready channel in a short slice.
        pending = next(ch for ch in channels if not ch.first_au.is_set())
        if pending.first_au.wait(0.25):
            continue
        waited += 0.25
        if not alive():
            break
    return all(ch.first_au.is_set() for ch in channels)


def _drain_stderr(proc: subprocess.Popen, tail: list[str], serial: int) -> None:
    """Keep the last few stderr lines of ``proc`` for failure diagnostics."""
    if proc.stderr is None:
        return
    try:
        for raw in proc.stderr:
            line = raw.decode(errors="replace").strip()
            if line:
                tail.append(line)
                del tail[:-10]
                _logger.debug("zed gst sn=%d: %s", serial, line)
    except Exception:  # noqa: BLE001 - fd closed during teardown
        pass


def _h264_enc_branch(bitrate: int, fps: int, out_fd: int) -> list[str]:
    """One ``crop-output -> NVENC -> fdsink`` branch (tokens after the crop)."""
    return [
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
        f"fd={out_fd}",
        "sync=false",
    ]


def _stereo_gst_argv(
    serial: int,
    resolution: str,
    fps: int,
    bitrate: int,
    fd_left: int,
    fd_right: int,
) -> list[str]:
    """``gst-launch-1.0`` argv: stereo grab -> per-eye NVENC -> two fds.

    ``zedsrc stream-type=7`` delivers both eyes side-by-side in one
    zero-copy NV12 NVMM buffer (``2W x H``). A ``tee`` feeds two branches;
    each crops its half on the VIC (``nvvidconv`` ``left/right/top/bottom``
    define the kept rectangle, with the output caps width pinning the crop
    — both are required or nvvidconv passes the full frame through) and
    encodes it with NVENC to its own fd. Everything stays on the GPU; the
    eyes are encoded independently so the headset renders each per-lens,
    matching the SDK ``overhead_left`` / ``overhead_right`` contract.
    """
    eye_w, eye_h = _RESOLUTION_DIMS[resolution]
    full_w = eye_w * 2
    eye_caps = (
        f"video/x-raw(memory:NVMM),format=NV12,"
        f"width={eye_w},height={eye_h},pixel-aspect-ratio=1/1"
    )
    return [
        "gst-launch-1.0",
        "-q",
        "zedsrc",
        f"camera-sn={serial}",
        f"camera-resolution={_STEREO_RESOLUTION_ENUM[resolution]}",
        f"camera-fps={fps}",
        "stream-type=7",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "tee",
        "name=t",
        # Left eye: keep columns [0, eye_w).
        "t.",
        "!",
        "queue",
        "!",
        "nvvidconv",
        "left=0",
        f"right={eye_w}",
        "top=0",
        f"bottom={eye_h}",
        "!",
        eye_caps,
        "!",
        *_h264_enc_branch(bitrate, fps, fd_left),
        # Right eye: keep columns [eye_w, 2*eye_w).
        "t.",
        "!",
        "queue",
        "!",
        "nvvidconv",
        f"left={eye_w}",
        f"right={full_w}",
        "top=0",
        f"bottom={eye_h}",
        "!",
        eye_caps,
        "!",
        *_h264_enc_branch(bitrate, fps, fd_right),
    ]


class ZedXStereoGstStream:
    """GPU-resident stereo ZED X grab pipeline with one channel per eye.

    Exposes :attr:`left_view` and :attr:`right_view` — each an
    :class:`_AUChannel` satisfying the pre-encoded source duck type
    (``subscribe`` / ``unsubscribe`` / ``alive``) — so the relay can
    register them as ``overhead_left`` / ``overhead_right``, exactly like
    the SDK ``ZedStereoCamera`` exposes ``left_view`` / ``right_view``.

    The camera is opened once (``zedsrc``); the two eyes are cropped and
    encoded from its single side-by-side buffer, written to two pipes.
    """

    def __init__(self, serial: int, resolution: str = "HD1200", fps: int = 60) -> None:
        if resolution not in _STEREO_RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported stereo ZED X resolution {resolution!r} "
                f"(expected one of {', '.join(_STEREO_RESOLUTION_ENUM)})"
            )
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        eye_w, eye_h = _RESOLUTION_DIMS[resolution]
        bitrate = _bitrate_for(eye_w, eye_h, fps)

        self.left_view = _AUChannel(lambda: self.alive)
        self.right_view = _AUChannel(lambda: self.alive)
        self._stderr_tail: list[str] = []

        # One pipe per eye: the child's fdsink writes the encoded byte
        # stream to the write end (kept open in the child via pass_fds,
        # same fd number), and a reader thread frames it from the read end.
        r_left, w_left = os.pipe()
        r_right, w_right = os.pipe()
        self._read_fds = [r_left, r_right]
        try:
            self._proc = subprocess.Popen(
                _stereo_gst_argv(serial, resolution, fps, bitrate, w_left, w_right),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(w_left, w_right),
                bufsize=0,
            )
        finally:
            # The parent never writes; close its copies so EOF reaches the
            # readers if the child exits.
            os.close(w_left)
            os.close(w_right)
        _logger.info(
            "zed stereo gst pipeline started: sn=%d %s@%dfps %.1f Mbps/eye (pid %d)",
            serial,
            resolution,
            fps,
            bitrate / 1e6,
            self._proc.pid,
        )
        for fd, channel, eye in (
            (r_left, self.left_view, "left"),
            (r_right, self.right_view, "right"),
        ):
            threading.Thread(
                target=read_annexb_aus,
                args=(fd, channel.broadcast, lambda: self.alive),
                name=f"zed-gst-{serial}-{eye}",
                daemon=True,
            ).start()
        threading.Thread(
            target=self._stderr_loop, name=f"zed-gst-{serial}-err", daemon=True
        ).start()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def wait_ready(self, timeout: float = _READY_TIMEOUT_S) -> bool:
        """Block until both eyes have produced their first AU."""
        if _wait_first_au(
            (self.left_view, self.right_view), lambda: self.alive, timeout
        ):
            return True
        if self._stderr_tail:
            _logger.warning(
                "zed stereo gst pipeline sn=%d not ready: %s",
                self.serial,
                " | ".join(self._stderr_tail[-3:]),
            )
        return False

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)  # reap — no zombies
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        for fd in self._read_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        if self._proc.stderr is not None:
            try:
                self._proc.stderr.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    # ZedCamera-compatible alias so teleop's cleanup loop can stay generic.
    disconnect = close

    def __del__(self) -> None:
        self.close()

    def _stderr_loop(self) -> None:
        _drain_stderr(self._proc, self._stderr_tail, self.serial)
