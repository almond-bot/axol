"""GPU-resident ZED camera pipeline (zed-gstreamer) for teleop + collection.

This is the single, fast camera path the project unifies on. The Stereolabs
``zedxonesrc`` / ``zedsrc`` GStreamer elements grab frames straight into NVMM
(GPU) memory; an in-process GStreamer pipeline then tees that one zero-copy
buffer to two consumers:

* **encoded branch** — ``nvv4l2h264enc`` (NVENC) -> ``appsink``. The whole
  grab -> encode chain stays on the GPU (~4.5 ms), and Python only ever sees
  the encoded H.264 access units, which the WebRTC relay forwards as
  pre-encoded packets (aiortc ``encoder.pack``). This is the headset view for
  both teleop and data collection.
* **raw branch** — ``nvvidconv`` -> RGBA ``appsink`` -> numpy. Only built when
  raw frames are needed (data collection's dataset, policy inference). Each
  frame carries a ``capture_perf_ts`` derived from the buffer PTS. We run a
  patched ``zedxonesrc``/``zedsrc`` (``do-timestamp=false``) that stamps the
  PTS at the true sensor-exposure instant (``TIME_REFERENCE::IMAGE``) instead
  of host-receive time; :meth:`_cap_perf_from_pts` maps that running-time onto
  ``time.perf_counter``, so dataset rows align image capture with the joint
  sample on the same exposure clock as the SDK ``ZedCamera`` path — without the
  SDK's host round trip. (The stock plugin stamps host-receive time, which lags
  exposure by the camera delivery latency; see ``scripts``/zed-gstreamer patch.)

:class:`ZedGstCamera` (mono ``zedxonesrc``) and :class:`ZedGstStereoCamera`
(stereo ``zedsrc``, per-eye crop) are drop-in replacements for
``ZedCamera`` / ``ZedStereoCamera``: they expose ``connect`` / ``disconnect``,
``read_at_or_after`` / ``read_latest`` / ``read`` (raw, for collection +
inference) **and** ``subscribe`` / ``unsubscribe`` / ``alive`` (encoded AUs,
for the WebRTC relay). The camera is exclusively owned by this pipeline — the
ZED SDK cannot open it at the same time.

Requires the zed-gstreamer plugins (``zedxonesrc`` / ``zedsrc``), the Jetson
``nvv4l2h264enc`` element, and PyGObject (installed into the axol env by
``axol gst.install``). :func:`zed_gst_available` / :func:`zed_stereo_gst_available`
gate use; without them callers fall back to the SDK ``ZedCamera``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from .hw_video import _bitrate_for, hw_h264_available

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# zedxonesrc camera-resolution enum (GstZedXOneSrcResol), keyed by the names
# used in ZedCameraConfig / ZED_RESOLUTION_DIMS.
_RESOLUTION_ENUM: dict[str, int] = {"SVGA": 0, "HD1080": 1, "HD1200": 2}
_RESOLUTION_DIMS: dict[str, tuple[int, int]] = {
    "SVGA": (960, 600),
    "HD1080": (1920, 1080),
    "HD1200": (1920, 1200),
}
# zedsrc camera-resolution enum (GstZedSrcRes) for the stereo ZED X. Only the
# GMSL2 60-fps modes are exposed (SVGA is ZED-X-One-only).
_STEREO_RESOLUTION_ENUM: dict[str, int] = {"HD1080": 1, "HD1200": 2}

# Per-subscriber AU queue depth. A healthy consumer pops every AU as it
# arrives; the bound only matters for a stalled consumer, where the oldest
# AUs are dropped so a backlog can never become latency.
_SUBSCRIBER_QUEUE_DEPTH = 4

# How long the pipeline may take to open the camera and deliver its first
# sample (the daemon handshake plus sensor start is a few seconds).
_READY_TIMEOUT_S = 15.0

# Cap on auto-exposure time (µs). Exposure happens before the frame exists, so
# it is pure glass-to-glass latency: the SDK default lets auto-exposure run to
# 66.7 ms (4 frame intervals at 60 fps) in dim light. 8 ms keeps capture
# latency bounded; auto gain stays enabled and compensates the brightness.
_MAX_AUTO_EXPOSURE_US = 8000

# All branch queues leak downstream (drop the oldest buffer when full) so a
# momentarily slow consumer can never let a backlog accumulate into latency —
# the pipeline always favours the freshest frame. max-size-buffers=2 keeps the
# decouple shallow.
_QUEUE = "queue leaky=downstream max-size-buffers=2"

_gst_init_lock = threading.Lock()
_gst_inited = False


def _set_typelib_path() -> None:
    """Ensure PyGObject finds the system GObject-introspection typelibs.

    PyGObject installed into the axol venv loads typelibs from
    ``GI_TYPELIB_PATH``; the GStreamer ones live with the system packages.
    Prepend the standard multiarch + ``/usr/lib`` locations if they are not
    already on the path so the venv interpreter can import ``Gst`` / ``GstApp``.
    """
    candidates = [
        "/usr/lib/aarch64-linux-gnu/girepository-1.0",
        "/usr/lib/x86_64-linux-gnu/girepository-1.0",
        "/usr/lib/girepository-1.0",
    ]
    existing = os.environ.get("GI_TYPELIB_PATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for path in candidates:
        if os.path.isdir(path) and path not in parts:
            parts.append(path)
    if parts:
        os.environ["GI_TYPELIB_PATH"] = os.pathsep.join(parts)


def _require_gst() -> tuple[Any, Any]:
    """Import and initialise GStreamer (PyGObject). Raises if unavailable."""
    global _gst_inited
    _set_typelib_path()
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp  # noqa: F401 - GstApp registers AppSink

    with _gst_init_lock:
        if not _gst_inited:
            Gst.init(None)
            _gst_inited = True
    return Gst, GstApp


def _gi_available() -> bool:
    try:
        _require_gst()
        return True
    except Exception:  # noqa: BLE001 - missing PyGObject / typelibs
        return False


def zed_gst_available() -> bool:
    """True when PyGObject, NVENC, and the mono ``zedxonesrc`` element exist."""
    if not _gi_available() or not hw_h264_available():
        return False
    ok = _element_available("zedxonesrc")
    if ok:
        _logger.info("zed-gstreamer mono pipeline (zedxonesrc) available")
    return ok


def zed_stereo_gst_available() -> bool:
    """True when PyGObject, NVENC, and the stereo ``zedsrc`` element exist."""
    if not _gi_available() or not hw_h264_available():
        return False
    ok = _element_available("zedsrc")
    if ok:
        _logger.info("zed-gstreamer stereo pipeline (zedsrc) available")
    return ok


def _element_available(element: str) -> bool:
    """True when GStreamer can find ``element`` in its registry."""
    try:
        Gst, _ = _require_gst()
    except Exception:  # noqa: BLE001 - no PyGObject
        return False
    return Gst.ElementFactory.find(element) is not None


def _split_nals(data: bytes) -> list[bytes]:
    """Split one Annex-B access unit into NALs (start codes stripped)."""
    nals: list[bytes] = []
    i = data.find(b"\x00\x00\x01")
    while i != -1:
        start = i + 3
        nxt = data.find(b"\x00\x00\x01", start)
        if nxt == -1:
            nals.append(data[start:])
            break
        end = nxt - 1 if data[nxt - 1] == 0 else nxt
        nals.append(data[start:end])
        i = nxt
    return [n for n in nals if n]


class _AUChannel:
    """Fan-out of one H.264 stream's access units to subscriber queues.

    Satisfies the pre-encoded source duck type the WebRTC relay expects
    (``subscribe`` / ``unsubscribe`` / ``alive``). A mono camera has one
    channel; a stereo camera has one per eye.
    """

    def __init__(self, alive: Any) -> None:
        self._alive = alive
        self._subscribers: list[queue.Queue[list[bytes]]] = []
        self._lock = threading.Lock()
        self.first_au = threading.Event()

    @property
    def alive(self) -> bool:
        return bool(self._alive())

    def subscribe(self) -> queue.Queue[list[bytes]]:
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
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass


class _RawBuffer:
    """Latest raw RGBA frame plus its capture timestamps, for dataset/inference.

    Mirrors the ``ZedCamera`` frame store so ``read_at_or_after`` /
    ``read_latest`` behave identically. Frames are kept RGBA (the VIC's
    ``nvvidconv`` output); ``read_*`` return RGB (``[:, :, :3]``).
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self.new_frame = threading.Event()
        self._rgba: NDArray[Any] | None = None
        self._cap_ts: float | None = None
        self._recv_ts: float | None = None

    def set(self, rgba: NDArray[Any], cap_ts: float, recv_ts: float) -> None:
        with self._lock:
            self._rgba = rgba
            self._cap_ts = cap_ts
            self._recv_ts = recv_ts
        self.new_frame.set()

    def _rgb(self, rgba: NDArray[Any]) -> NDArray[Any]:
        import numpy as np

        return np.ascontiguousarray(rgba[:, :, :3])

    def read_at_or_after(
        self, target: float, timeout_ms: float = 500
    ) -> tuple[NDArray[Any], float, float]:
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            self.new_frame.clear()
            with self._lock:
                rgba, cap, recv = self._rgba, self._cap_ts, self._recv_ts
            if (
                rgba is not None
                and cap is not None
                and recv is not None
                and cap >= target
            ):
                return self._rgb(rgba), cap, recv
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"gst camera timed out waiting for frame at "
                    f"capture_perf_ts >= {target:.6f} after {timeout_ms:.1f}ms."
                )
            self.new_frame.wait(timeout=remaining)

    def read_latest_with_ts(self) -> tuple[NDArray[Any], float, float]:
        with self._lock:
            rgba, cap, recv = self._rgba, self._cap_ts, self._recv_ts
        if rgba is None or cap is None or recv is None:
            raise RuntimeError("gst camera has not captured any frames yet.")
        return self._rgb(rgba), cap, recv

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        frame, _cap, recv = self.read_latest_with_ts()
        age_ms = (time.perf_counter() - recv) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"latest gst frame is {age_ms:.0f}ms old (> {max_age_ms})."
            )
        return frame


class _GstStreamConsumer:
    """Consumer-facing API shared by the mono camera and each stereo eye.

    Backed by an :class:`_AUChannel` (encoded, for WebRTC) and optionally a
    :class:`_RawBuffer` (raw, for dataset/inference). Subclasses populate
    ``_enc``, ``_raw``, ``_alive_fn``, and the ``width`` / ``height`` / ``fps``
    fields.
    """

    _enc: _AUChannel | None
    _raw: _RawBuffer | None
    _alive_fn: Any
    width: int
    height: int
    fps: int

    # -- encoded (WebRTC PrecodedVideoTrack) --------------------------------
    @property
    def alive(self) -> bool:
        return bool(self._alive_fn())

    def subscribe(self) -> queue.Queue[list[bytes]]:
        if self._enc is None:
            raise RuntimeError("this gst stream has no encoded branch")
        return self._enc.subscribe()

    def unsubscribe(self, q: queue.Queue[list[bytes]]) -> None:
        if self._enc is not None:
            self._enc.unsubscribe(q)

    # -- raw (collect-data / inference) -------------------------------------
    def read_at_or_after(
        self, target_capture_perf_ts: float, timeout_ms: float = 500
    ) -> tuple[NDArray[Any], float, float]:
        if self._raw is None:
            raise RuntimeError("this gst stream has no raw branch (built without raw)")
        return self._raw.read_at_or_after(target_capture_perf_ts, timeout_ms)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self._raw is None:
            raise RuntimeError("this gst stream has no raw branch (built without raw)")
        return self._raw.read_latest(max_age_ms)

    def read_latest_with_ts(self) -> tuple[NDArray[Any], float, float]:
        if self._raw is None:
            raise RuntimeError("this gst stream has no raw branch (built without raw)")
        return self._raw.read_latest_with_ts()

    def read(self) -> NDArray[Any]:
        return self.read_at_or_after(0.0, timeout_ms=10000)[0]


def _enc_appsink(name: str) -> str:
    return f"appsink name={name} emit-signals=false max-buffers=6 drop=false sync=false"


def _raw_appsink(name: str) -> str:
    return f"appsink name={name} emit-signals=false max-buffers=2 drop=true sync=false"


def _enc_branch(bitrate: int, fps: int) -> str:
    return (
        f"nvv4l2h264enc control-rate=1 bitrate={bitrate} preset-level=1 "
        f"insert-sps-pps=true insert-aud=true idrinterval={fps} maxperf-enable=true "
        "! video/x-h264,stream-format=byte-stream"
    )


class _GstPipelineBase:
    """Common pipeline lifecycle: build, pull threads, ready-wait, teardown."""

    def __init__(self) -> None:
        self._gst: Any = None
        self._pipeline: Any = None
        self._clock: Any = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    @property
    def alive(self) -> bool:
        if self._pipeline is None:
            return False
        _, state, _ = self._pipeline.get_state(0)
        return state != self._gst.State.NULL and not self._stop.is_set()

    @property
    def is_connected(self) -> bool:
        """ZedCamera-compatible: the pipeline is built and not torn down."""
        return self._pipeline is not None and not self._stop.is_set()

    def _cap_perf_from_pts(self, pts: int, recv_perf: float) -> float:
        """Map a buffer running-time PTS onto ``time.perf_counter`` seconds.

        With our patched ``zedxonesrc``/``zedsrc`` (and ``do-timestamp=false``),
        ``pts`` is the pipeline running-time of the true sensor-exposure instant
        (``TIME_REFERENCE::IMAGE``), not host-receive time. The remaining
        ``(clock_now_running - pts)`` is the glass-to-pull latency, a duration,
        so subtracting it from the receive ``perf_counter`` yields the
        sensor-capture timestamp on the ``perf_counter`` timeline -- parity with
        the ZED SDK ``ZedCamera`` path, so image frames align with joint
        samples.
        """
        if pts == self._gst.CLOCK_TIME_NONE or self._clock is None:
            return recv_perf
        running_now = self._clock.get_time() - self._pipeline.get_base_time()
        latency_s = max(0, running_now - pts) / 1e9
        return recv_perf - latency_s

    def _start_pull(self, name: str, sink_name: str, handler: Any) -> None:
        sink = self._pipeline.get_by_name(sink_name)
        thread = threading.Thread(
            target=self._pull_loop, args=(sink, handler), name=name, daemon=True
        )
        thread.start()
        self._threads.append(thread)

    def _pull_loop(self, sink: Any, handler: Any) -> None:
        Gst = self._gst
        while not self._stop.is_set():
            sample = sink.emit("try-pull-sample", Gst.SECOND // 2)
            if sample is None:
                continue
            recv_perf = time.perf_counter()
            buf = sample.get_buffer()
            try:
                handler(buf, recv_perf)
            except Exception as exc:  # noqa: BLE001 - never kill the pull thread
                _logger.debug("gst pull handler error: %s", exc)

    def _make_au_handler(self, channel: _AUChannel) -> Any:
        def handle(buf: Any, _recv_perf: float) -> None:
            ok, mapinfo = buf.map(self._gst.MapFlags.READ)
            if not ok:
                return
            try:
                nals = _split_nals(bytes(mapinfo.data))
            finally:
                buf.unmap(mapinfo)
            if nals:
                channel.broadcast(nals)

        return handle

    def _make_raw_handler(self, raw: _RawBuffer) -> Any:
        import numpy as np

        h, w = raw.height, raw.width

        def handle(buf: Any, recv_perf: float) -> None:
            ok, mapinfo = buf.map(self._gst.MapFlags.READ)
            if not ok:
                return
            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                # nvvidconv may pad rows; trust the negotiated WxHx4 size.
                if arr.size < w * h * 4:
                    return
                rgba = arr[: w * h * 4].reshape(h, w, 4).copy()
            finally:
                buf.unmap(mapinfo)
            raw.set(rgba, self._cap_perf_from_pts(buf.pts, recv_perf), recv_perf)

        return handle

    def _launch(self, pipeline_str: str) -> None:
        Gst, _ = _require_gst()
        self._gst = Gst
        _logger.info("gst zed pipeline: %s", pipeline_str)
        self._pipeline = Gst.parse_launch(pipeline_str)

    def _play_and_wait(self, channels: tuple[_AUChannel, ...]) -> bool:
        Gst = self._gst
        self._pipeline.set_state(Gst.State.PLAYING)
        self._pipeline.get_state(Gst.SECOND * 5)
        self._clock = self._pipeline.get_pipeline_clock()
        # Ready when every encoded channel has produced its first AU (or, if
        # there are no encoded channels, give the raw branch a moment).
        deadline = time.perf_counter() + _READY_TIMEOUT_S
        if not channels:
            time.sleep(0.5)
            return self.alive
        while time.perf_counter() < deadline:
            if all(ch.first_au.is_set() for ch in channels):
                return True
            if not self.alive:
                break
            time.sleep(0.05)
        return all(ch.first_au.is_set() for ch in channels)

    def disconnect(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None

    # ZedCamera-compatible alias.
    close = disconnect

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


class ZedGstCamera(_GstPipelineBase, _GstStreamConsumer):
    """Mono ZED X One camera via ``zedxonesrc`` (encoded + optional raw).

    Drop-in for ``ZedCamera`` (``read_at_or_after`` / ``read_latest`` / ``read``)
    that also serves the WebRTC relay (``subscribe`` / ``alive``). Build with
    ``want_raw=False`` for the teleop relay (encoded only, lowest cost) and
    ``want_raw=True`` for data collection / inference (adds the numpy branch).
    """

    def __init__(
        self,
        serial: int,
        resolution: str = "HD1200",
        fps: int = 60,
        *,
        want_encoded: bool = True,
        want_raw: bool = False,
    ) -> None:
        _GstPipelineBase.__init__(self)
        if resolution not in _RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported ZED X One resolution {resolution!r} "
                f"(expected one of {', '.join(_RESOLUTION_ENUM)})"
            )
        if not (want_encoded or want_raw):
            raise ValueError("ZedGstCamera needs at least one of encoded/raw")
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        self.width, self.height = _RESOLUTION_DIMS[resolution]
        self._want_encoded = want_encoded
        self._want_raw = want_raw
        self._enc = _AUChannel(lambda: self.alive) if want_encoded else None
        self._raw = _RawBuffer(self.width, self.height) if want_raw else None
        self._alive_fn = lambda: self.alive

    def __repr__(self) -> str:
        return f"ZedGstCamera(serial={self.serial})"

    def _pipeline_str(self) -> str:
        bitrate = _bitrate_for(self.width, self.height, self.fps)
        src = (
            f"zedxonesrc camera-sn={self.serial} "
            f"camera-resolution={_RESOLUTION_ENUM[self.resolution]} "
            f"camera-fps={self.fps} stream-type=1 do-timestamp=false "
            f"ctrl-auto-exposure-range-max={_MAX_AUTO_EXPOSURE_US} "
            "! video/x-raw(memory:NVMM),format=NV12"
        )
        enc = f"{_QUEUE} ! {_enc_branch(bitrate, self.fps)} ! {_enc_appsink('enc')}"
        raw = f"{_QUEUE} ! nvvidconv ! video/x-raw,format=RGBA ! {_raw_appsink('raw')}"
        if self._want_encoded and self._want_raw:
            return f"{src} ! tee name=t  t. ! {enc}  t. ! {raw}"
        if self._want_encoded:
            return f"{src} ! {enc}"
        return f"{src} ! {raw}"

    def connect(self, warmup: bool = True) -> None:
        """Open the camera, start the pipeline, and block until it streams."""
        self._launch(self._pipeline_str())
        if self._enc is not None:
            self._start_pull(
                f"zedgst-{self.serial}-enc", "enc", self._make_au_handler(self._enc)
            )
        if self._raw is not None:
            self._start_pull(
                f"zedgst-{self.serial}-raw", "raw", self._make_raw_handler(self._raw)
            )
        channels = (self._enc,) if self._enc is not None else ()
        if not self._play_and_wait(channels):
            self.disconnect()
            raise RuntimeError(
                f"ZedGstCamera(serial={self.serial}) did not start streaming "
                f"within {_READY_TIMEOUT_S:.0f}s (camera absent or in use?)."
            )
        _logger.info(
            "ZedGstCamera connected (sn=%d %dx%d @ %dfps, encoded=%s raw=%s).",
            self.serial,
            self.width,
            self.height,
            self.fps,
            self._want_encoded,
            self._want_raw,
        )


class _GstEye(_GstStreamConsumer):
    """One eye of a stereo gst pipeline, presented as a camera.

    ``connect`` / ``disconnect`` / ``is_connected`` defer to the shared parent
    so the stereo camera is opened and closed exactly once regardless of
    iteration order (matches ``_StereoEyeView``).
    """

    def __init__(
        self,
        parent: "ZedGstStereoCamera",
        enc: _AUChannel | None,
        raw: _RawBuffer | None,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._parent = parent
        self._enc = enc
        self._raw = raw
        self._alive_fn = lambda: parent.alive
        self.width = width
        self.height = height
        self.fps = fps

    @property
    def is_connected(self) -> bool:
        return self._parent.is_connected

    def connect(self, warmup: bool = True) -> None:
        if not self._parent.is_connected:
            self._parent.connect(warmup=warmup)

    def disconnect(self) -> None:
        if self._parent.is_connected:
            self._parent.disconnect()


class ZedGstStereoCamera(_GstPipelineBase):
    """Stereo ZED X via ``zedsrc``: one grab, two cropped per-eye streams.

    Exposes :attr:`left_view` / :attr:`right_view` (each a
    :class:`_GstStreamConsumer`), matching ``ZedStereoCamera`` so the rest of
    the pipeline treats the two eyes as ordinary cameras.
    """

    def __init__(
        self,
        serial: int,
        resolution: str = "HD1200",
        fps: int = 60,
        *,
        want_encoded: bool = True,
        want_raw: bool = False,
    ) -> None:
        _GstPipelineBase.__init__(self)
        if resolution not in _STEREO_RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported stereo ZED X resolution {resolution!r} "
                f"(expected one of {', '.join(_STEREO_RESOLUTION_ENUM)})"
            )
        if not (want_encoded or want_raw):
            raise ValueError("ZedGstStereoCamera needs at least one of encoded/raw")
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        self.width, self.height = _RESOLUTION_DIMS[resolution]
        self._want_encoded = want_encoded
        self._want_raw = want_raw

        def eye() -> tuple[_AUChannel | None, _RawBuffer | None, _GstEye]:
            enc = _AUChannel(lambda: self.alive) if want_encoded else None
            raw = _RawBuffer(self.width, self.height) if want_raw else None
            view = _GstEye(self, enc, raw, self.width, self.height, self.fps)
            return enc, raw, view

        self._left_enc, self._left_raw, self.left_view = eye()
        self._right_enc, self._right_raw, self.right_view = eye()

    def __repr__(self) -> str:
        return f"ZedGstStereoCamera(serial={self.serial})"

    def _eye_branch(self, side: str, sink_suffix: str) -> str:
        """One eye: crop its half on the VIC, then encode and/or raw appsink."""
        eye_w, eye_h = self.width, self.height
        left = 0 if side == "left" else eye_w
        right = eye_w if side == "left" else eye_w * 2
        bitrate = _bitrate_for(eye_w, eye_h, self.fps)
        caps = (
            f"video/x-raw(memory:NVMM),format=NV12,width={eye_w},height={eye_h},"
            "pixel-aspect-ratio=1/1"
        )
        crop = f"{_QUEUE} ! nvvidconv left={left} right={right} top=0 bottom={eye_h} ! {caps}"
        if self._want_encoded and self._want_raw:
            enc = (
                f"{_QUEUE} ! {_enc_branch(bitrate, self.fps)} ! "
                f"{_enc_appsink('enc_' + sink_suffix)}"
            )
            raw = (
                f"{_QUEUE} ! nvvidconv ! video/x-raw,format=RGBA ! "
                f"{_raw_appsink('raw_' + sink_suffix)}"
            )
            return f"{crop} ! tee name=t{sink_suffix}  t{sink_suffix}. ! {enc}  t{sink_suffix}. ! {raw}"
        if self._want_encoded:
            return f"{crop} ! {_enc_branch(bitrate, self.fps)} ! {_enc_appsink('enc_' + sink_suffix)}"
        return f"{crop} ! nvvidconv ! video/x-raw,format=RGBA ! {_raw_appsink('raw_' + sink_suffix)}"

    def _pipeline_str(self) -> str:
        src = (
            f"zedsrc camera-sn={self.serial} "
            f"camera-resolution={_STEREO_RESOLUTION_ENUM[self.resolution]} "
            f"camera-fps={self.fps} stream-type=7 do-timestamp=false "
            "! video/x-raw(memory:NVMM),format=NV12 ! tee name=split"
        )
        return (
            f"{src}  split. ! {self._eye_branch('left', 'l')}  "
            f"split. ! {self._eye_branch('right', 'r')}"
        )

    def connect(self, warmup: bool = True) -> None:
        self._launch(self._pipeline_str())
        for enc, raw, suffix in (
            (self._left_enc, self._left_raw, "l"),
            (self._right_enc, self._right_raw, "r"),
        ):
            if enc is not None:
                self._start_pull(
                    f"zedgst-{self.serial}-enc{suffix}",
                    f"enc_{suffix}",
                    self._make_au_handler(enc),
                )
            if raw is not None:
                self._start_pull(
                    f"zedgst-{self.serial}-raw{suffix}",
                    f"raw_{suffix}",
                    self._make_raw_handler(raw),
                )
        channels = tuple(c for c in (self._left_enc, self._right_enc) if c is not None)
        if not self._play_and_wait(channels):
            self.disconnect()
            raise RuntimeError(
                f"ZedGstStereoCamera(serial={self.serial}) did not start "
                f"streaming within {_READY_TIMEOUT_S:.0f}s."
            )
        _logger.info(
            "ZedGstStereoCamera connected (sn=%d %dx%d/eye @ %dfps).",
            self.serial,
            self.width,
            self.height,
            self.fps,
        )
