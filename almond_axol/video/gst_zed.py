"""GPU-resident ZED camera pipeline (zed-gstreamer) for teleop + collection.

This is the single, fast camera path the project unifies on. The Stereolabs
``zedxonesrc`` / ``zedsrc`` GStreamer elements grab frames straight into NVMM
(GPU) memory; an in-process GStreamer pipeline then tees that one zero-copy
buffer to two consumers:

* **encoded branch** — ``videorate`` fixes headset delivery at 30 fps, then
  ``nvv4l2h264enc`` (NVENC) -> ``appsink``. The whole grab -> encode chain
  stays on the GPU (~4.5 ms), and Python only ever sees the encoded H.264
  access units, which the WebRTC relay forwards as pre-encoded packets
  (aiortc ``encoder.pack``). This is the headset view for both teleop and data
  collection; camera capture remains independent for recording and policies.
* **dataset branch** — only built when the dataset / a policy needs the frames.
  For the recorder it is a second GPU encode -> ``gdppay`` -> ``shmsink``
  carrying H.264 AUs *and their original PTS* (``_dataset_enc_shmsink``): the
  recorder just muxes them, so no raw copy or re-encode crosses the boundary.
  For in-process consumers (inference, or the pyshm fallback) it is instead
  ``nvvidconv`` -> RGBA ``appsink`` -> numpy. Each RGBA frame carries a
  ``capture_perf_ts`` derived from the buffer PTS. We run a
  patched ``zedxonesrc``/``zedsrc`` (``do-timestamp=false``) that stamps the
  PTS at the true sensor-exposure instant (``TIME_REFERENCE::IMAGE``) instead
  of host-receive time; :meth:`_cap_perf_from_pts` maps that running-time onto
  ``time.perf_counter``, so dataset rows align image capture with the joint
  sample on the same exposure clock as the SDK ``ZedCamera`` path — without the
  SDK's host round trip. (The stock plugin stamps host-receive time, which lags
  exposure by the camera delivery latency; the sensor-timestamp patch
  ``axol gst.build-zed`` applies is what enables ``TIME_REFERENCE::IMAGE``.)

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
import math
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from .constants import HEADSET_STREAM_FPS
from .hw_video import _bitrate_for, dataset_vbr_bitrate, hw_h264_available

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
# zedsrc camera-resolution enum (GstZedSrcRes) for the stereo ZED X. Note the
# enum values differ from zedxonesrc's: SVGA is 4 here but 0 there (HD1080 and
# HD1200 happen to share 1/2 in both).
_STEREO_RESOLUTION_ENUM: dict[str, int] = {"SVGA": 4, "HD1080": 1, "HD1200": 2}

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


def _verified_sensor_timestamps_available(element: str) -> bool:
    """True when runtime resolves ``gst.build-zed``'s stamped plugin binary."""
    try:
        from ..cli.gst.build_zed import sensor_timestamp_patch_installed

        Gst, _ = _require_gst()
        factory = Gst.ElementFactory.find(element)
        plugin = None if factory is None else factory.get_plugin()
        filename = None if plugin is None else plugin.get_filename()
        return filename is not None and sensor_timestamp_patch_installed(
            element, filename
        )
    except Exception:  # noqa: BLE001 - an unverifiable build is not exact
        return False


def zed_gst_available(*, require_sensor_timestamps: bool = False) -> bool:
    """True when PyGObject, NVENC, and the mono ``zedxonesrc`` element exist."""
    if not _gi_available() or not hw_h264_available():
        return False
    ok = _element_available("zedxonesrc")
    if not ok:
        return False
    if require_sensor_timestamps and not _verified_sensor_timestamps_available(
        "zedxonesrc"
    ):
        _logger.warning(
            "zedxonesrc is installed but its sensor-timestamp patch cannot be "
            "verified; run `axol gst.build-zed` before collection/policy"
        )
        return False
    _logger.info("zed-gstreamer mono pipeline (zedxonesrc) available")
    return True


def zed_stereo_gst_available(*, require_sensor_timestamps: bool = False) -> bool:
    """True when PyGObject, NVENC, and the stereo ``zedsrc`` element exist."""
    if not _gi_available() or not hw_h264_available():
        return False
    ok = _element_available("zedsrc")
    if not ok:
        return False
    if require_sensor_timestamps and not _verified_sensor_timestamps_available(
        "zedsrc"
    ):
        _logger.warning(
            "zedsrc is installed but its sensor-timestamp patch cannot be "
            "verified; run `axol gst.build-zed` before collection/policy"
        )
        return False
    _logger.info("zed-gstreamer stereo pipeline (zedsrc) available")
    return True


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


def _raw_shmsink(socket_path: str) -> str:
    """Write raw NV12 frames to shared memory via gst's native (C) ``shmsink``.

    Legacy raw transport (kept for reference / the NV12
    :class:`~almond_axol.video.shm_frames.GstShmFrameReader`). The encoded
    dataset path (:func:`_dataset_enc_shmsink`) supersedes it: shipping H.264
    instead of raw NV12 cuts the boundary bandwidth ~50x and removes the
    recorder's re-encode. ``wait-for-connection=false`` so the relay never
    blocks when the recorder isn't attached yet.
    """
    return (
        f"shmsink socket-path={socket_path} wait-for-connection=false "
        "sync=false async=false"
    )


# Dataset AUs are all-intra. The Jetson encoder cannot be force-keyframed on
# demand on this L4T (the ``force-IDR`` signal segfaults and force-key-unit
# events are ignored), and its input/VIC pipeline retains a variable number of
# frames after an upstream valve closes. With a predictive GOP that hidden tail
# advances each encoder by a different amount, so the first recorder-visible
# IDRs can describe exposures several camera ticks apart even though the raw
# valves reopen on one shared timestamp. Making every dataset frame an IDR
# removes that unobservable phase state and also lets the recorder discard a
# leading independently-decodable AU when forming its row-zero camera cluster.
# Both ``iframeinterval`` and ``idrinterval`` must be one: nvv4l2h264enc exposes
# them as separate READY-only controls.
_DATASET_GOP_FRAMES = 1
# Before the recorder connects, tiny leaky queues around the dataset encoder are
# intentional: the upstream branch queue keeps the camera tee fresh while the
# valve is closed, and the output queue keeps NVENC returning its NVMM surfaces
# while shmsink parks GDP's one-shot header. During an episode a short input
# queue *after* a forced VIC copy absorbs NVENC startup jitter without retaining
# a large pool of camera-owned/crop surfaces; the compressed output keeps a
# longer cushion for recorder scheduling jitter. Both remain bounded and leaky
# so a dead recorder cannot back-pressure Argus or the independent headset path.
_DATASET_STARTUP_QUEUE_BUFFERS = 2
_DATASET_ACTIVE_INPUT_QUEUE_MIN_BUFFERS = 15
_DATASET_ACTIVE_INPUT_QUEUE_SECONDS = 0.5
_DATASET_ACTIVE_OUTPUT_QUEUE_MIN_BUFFERS = 60
_DATASET_ACTIVE_OUTPUT_QUEUE_SECONDS = 2
# The VIC pool must exceed the downstream queue capacity so NVENC can retain its
# input surfaces while a new buffer still reaches the queue and triggers its
# downstream-leaky policy instead of blocking back into the camera tee. Eight
# spare surfaces is a thin, undocumented allowance for NVENC retention across
# four simultaneous all-intra 60 Hz encoders. Thirty-two provides bounded
# headroom (about 28 MiB beyond the queue itself per SVGA branch) and keeps
# backpressure behind the copied-surface boundary.
_DATASET_VIC_INFLIGHT_SURFACES = 32
# Opening every camera with a sequence of host-side property writes can straddle
# multiple source frames. Arm a timestamp probe on every branch first, then let
# each one open itself on its first exposure at/after one shared future instant.
# 100 ms is ample time to install the handful of probes without making a record
# press feel sluggish; the relay caller performs this wait away from control.
_DATASET_ENABLE_LEAD_S = 0.100
# The shared gate still has a bounded acknowledgement wait. A timeout means a
# source stopped producing exposure timestamps, not ordinary scheduler jitter.
_DATASET_GATE_TIMEOUT_S = 2.0


def _dataset_rate_limit(capture_fps: int, dataset_fps: int) -> str:
    """Decimate the encoded dataset branch without synthesizing frames."""
    if capture_fps <= 0 or dataset_fps <= 0:
        raise ValueError("camera and dataset fps must be positive")
    if dataset_fps > capture_fps:
        raise ValueError(
            f"dataset fps ({dataset_fps}) cannot exceed camera capture fps "
            f"({capture_fps})"
        )
    if dataset_fps == capture_fps:
        return ""
    return (
        f"videorate drop-only=true max-rate={dataset_fps} ! "
        "video/x-raw(memory:NVMM),format=NV12,"
        f"framerate={dataset_fps}/1 ! "
    )


def _dataset_input_queue(name: str) -> str:
    """Named bounded queue between the dataset VIC copy and NVENC."""
    return (
        f"queue name={name}_inq leaky=downstream "
        f"max-size-buffers={_DATASET_STARTUP_QUEUE_BUFFERS} "
        "max-size-bytes=0 max-size-time=0"
    )


def _dataset_source_queue(name: str) -> str:
    """Shallow named queue before the dataset valve/VIC ownership boundary."""
    return (
        f"queue name={name}_srcq leaky=downstream "
        f"max-size-buffers={_DATASET_STARTUP_QUEUE_BUFFERS} "
        "max-size-bytes=0 max-size-time=0"
    )


def _dataset_active_input_buffers(dataset_fps: int) -> int:
    """Short converted-surface cushion for an active dataset encoder."""
    return max(
        _DATASET_ACTIVE_INPUT_QUEUE_MIN_BUFFERS,
        math.ceil(_DATASET_ACTIVE_INPUT_QUEUE_SECONDS * max(1, dataset_fps)),
    )


def _dataset_enc_shmsink(
    socket_path: str,
    w: int,
    h: int,
    dataset_fps: int,
    name: str,
) -> str:
    """Encode on the GPU and ship timestamped H.264 AUs over ``shmsink``.

    This replaces the raw-NV12 shmsink on the relay's dataset branch: the relay
    already holds the frame in NVMM, so a second NVENC branch (a GPU block, ~free
    on the CPU) hands the recorder a compressed stream (a fraction of the ~51 MB/s
    raw copy) — and the recorder only *muxes* it (see
    :class:`~almond_axol.lerobot.h264_mux_encoder.H264MuxStreamingEncoder`) rather
    than re-encoding. ``nvvidconv`` must output NVMM for ``nvv4l2h264enc``; the
    AU-aligned byte-stream is wrapped in GStreamer Data Protocol by ``gdppay``.
    Unlike bare ``shmsink``, GDP serializes each buffer's PTS (and its caps), so
    the recorder can recover the sensor-exposure timestamp instead of using the
    much later receive time. The whole path remains in GStreamer's C threads.

    ``wait-for-connection=true`` is essential. ``gdppay`` emits its stream/caps
    header only once; if shmsink discarded that header before the recorder's
    late ``gdpdepay`` connection, the consumer could never parse the stream.
    Waiting parks this branch's first GDP packet until the recorder attaches.
    A leaky queue after the encoder keeps NVENC draining (and releases its NVMM
    input surfaces) while that packet is parked; the queue must stay before
    ``gdppay`` so it can never discard GDP's one-shot stream/caps header. The
    short named queue between a forced VIC copy and NVENC decouples encoder
    startup stalls from camera capture and the independent headset branch
    without retaining the source camera's buffers. Both named queues grow only
    while an episode is active.
    The rate limiter deliberately stays upstream of the valve: it must continue
    tracking the source timeline between episodes, otherwise reopening a stale
    ``videorate`` can briefly pass capture-rate frames into a lower-rate dataset.
    The valve is born open so startup reaches shmsink and parks the GDP header
    before the parent later closes the branch between episodes.

    Encoding runs in VBR with a peak cap so the recorded dataset stays bounded
    and uniformly sized across cameras even when one sensor is very noisy (see
    ``dataset_vbr_bitrate``).
    """
    target, peak = dataset_vbr_bitrate(w, h, dataset_fps)
    input_buffers = _dataset_active_input_buffers(dataset_fps)
    converter_buffers = input_buffers + _DATASET_VIC_INFLIGHT_SURFACES
    return (
        f"nvvidconv output-buffers={converter_buffers} ! "
        f"video/x-raw(memory:NVMM),format=I420,width={w},height={h} "
        f"! {_dataset_input_queue(name)} "
        f"! nvv4l2h264enc name={name} control-rate=0 "
        f"bitrate={target} peak-bitrate={peak} preset-level=1 "
        f"insert-sps-pps=true insert-aud=true "
        f"iframeinterval={_DATASET_GOP_FRAMES} "
        f"idrinterval={_DATASET_GOP_FRAMES} maxperf-enable=true "
        "! video/x-h264,stream-format=byte-stream,alignment=au "
        f"! queue name={name}_outq leaky=downstream "
        f"max-size-buffers={_DATASET_STARTUP_QUEUE_BUFFERS} "
        "max-size-bytes=0 max-size-time=0 ! gdppay "
        f"! shmsink name={name}_shmsink socket-path={socket_path} "
        "wait-for-connection=true "
        "sync=false async=false"
    )


def _enc_branch(bitrate: int, fps: int, name: str = "venc") -> str:
    return (
        f"nvv4l2h264enc name={name} control-rate=1 bitrate={bitrate} preset-level=1 "
        f"insert-sps-pps=true insert-aud=true idrinterval={fps} maxperf-enable=true "
        "! video/x-h264,stream-format=byte-stream"
    )


def _stream_rate_limit(capture_fps: int, stream_fps: int) -> str:
    """GStreamer prefix that fixes only the encoded branch at headset rate.

    Faster capture is decimated without duplication. If a separately chosen
    recording/policy rate is below the headset rate, ``videorate`` repeats the
    latest frame so WebRTC still has an immutable 30 fps cadence. It changes
    only timestamps/buffer selection; the NVMM frame remains GPU-resident.
    Dataset/policy branches tee off before this element and therefore continue
    at ``capture_fps``.
    """
    if stream_fps == capture_fps:
        return ""
    drop_only = "drop-only=true " if capture_fps > stream_fps else ""
    return (
        f"videorate {drop_only}max-rate={stream_fps} ! "
        "video/x-raw(memory:NVMM),format=NV12,"
        f"framerate={stream_fps}/1 ! "
    )


class _GstPipelineBase:
    """Common pipeline lifecycle: build, pull threads, ready-wait, teardown."""

    def __init__(self) -> None:
        self._gst: Any = None
        self._pipeline: Any = None
        self._clock: Any = None
        # Constant affine mapping for this PLAYING pipeline:
        # capture_perf_seconds = buffer_pts_ns / 1e9 + offset. ``perf_counter``
        # is system-wide on our supported platforms, so the recorder process can
        # apply the same offset to PTS recovered by gdpdepay.
        self._pts_perf_offset_s: float | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        # Kept as a list for cancellation compatibility with an in-flight close.
        # Dataset encoders are all-intra now, so current closes are immediate and
        # never populate it.
        self._dataset_disable: list[tuple[Any, int, threading.Event, Any, str]] = []
        # One ``(sink_pad, probe_id, opened, cancelled, lock, status, valve,
        # label)`` per dataset branch waiting for a common exposure boundary.
        # ``lock`` serializes an opening callback with fail-close so a callback
        # can never reopen a valve after an aborted multi-camera transaction.
        self._dataset_enable: list[
            tuple[
                Any,
                int,
                threading.Event,
                threading.Event,
                threading.Lock,
                dict[str, str | None],
                Any,
                str,
            ]
        ] = []
        self._dataset_enabled = False
        # Named gst queue overruns are the missing stage-level evidence when a
        # later reader sees only a PTS hole. Values are reset at each episode;
        # callbacks log only active/opening dataset branches.
        self._dataset_queue_overruns: dict[str, int] = {}

    def _cancel_dataset_disable(self) -> None:
        """Remove any outstanding legacy dataset-close probes (best effort)."""
        pending, self._dataset_disable = self._dataset_disable, []
        for pad, probe_id, event, _valve, _label in pending:
            if event.is_set():
                continue  # a REMOVE return already detached this probe
            try:
                pad.remove_probe(probe_id)
            except Exception:  # noqa: BLE001 - pipeline may be tearing down
                pass

    def _cancel_dataset_enable(self) -> None:
        """Cancel a pending common-frame open and leave every valve closed."""
        pending, self._dataset_enable = self._dataset_enable, []
        for pad, probe_id, opened, cancelled, lock, status, valve, label in pending:
            # The callback checks ``cancelled`` and changes the valve under this
            # same lock. If it won the race, fail-close runs second and closes
            # the valve; if cancellation won, the callback cannot reopen it.
            with lock:
                cancelled.set()
                if status["error"] is None:
                    status["error"] = f"dataset source {label!r} opening was cancelled"
                valve.set_property("drop", True)
                # Wake a concurrent finish waiter so abort/disconnect never
                # leaves it sleeping until the full gate timeout.
                opened.set()
            try:
                pad.remove_probe(probe_id)
            except Exception:  # noqa: BLE001 - callback/pipeline may be exiting
                pass

    def _abort_dataset_enable(self, gates: tuple[tuple[str, str | None], ...]) -> None:
        """Fail a multi-camera open closed, including branches already opened."""
        self._cancel_dataset_enable()
        self._dataset_enabled = False
        if self._pipeline is None:
            return
        for valve_name, _encoder_name in gates:
            valve = self._pipeline.get_by_name(valve_name)
            if valve is not None:
                valve.set_property("drop", True)
        # Shrink only after every input valve is closed. Reducing a live input
        # queue first could itself discard an exposure from the episode we are
        # trying to fail closed.
        self._set_dataset_queue_depths(gates, active=False)

    def _set_dataset_queue_depths(
        self,
        gates: tuple[tuple[str, str | None], ...],
        *,
        active: bool,
    ) -> None:
        """Select bounded startup or active-episode encoder queue headroom.

        Startup must remain aggressively leaky: the ordinary pre-valve branch
        queue keeps the camera tee current while capture is disabled, and GDP's
        header is parked behind the post-encoder output queue at
        ``shmsink wait-for-connection=true`` before the recorder exists. During
        an episode the reader is connected and flushed, so retain a short queue
        of copied NVMM frames before NVENC and two seconds of compressed AUs
        after it. Both queues stay bounded and downstream-leaky; exhausting
        either still surfaces as a PTS/DISCONT capture failure.
        """
        if self._pipeline is None:
            return
        dataset_fps = max(1, int(getattr(self, "dataset_fps", 1)))
        active_input_buffers = _dataset_active_input_buffers(dataset_fps)
        active_output_buffers = max(
            _DATASET_ACTIVE_OUTPUT_QUEUE_MIN_BUFFERS,
            _DATASET_ACTIVE_OUTPUT_QUEUE_SECONDS * dataset_fps,
        )
        for _valve_name, encoder_name in gates:
            if encoder_name is None:
                continue
            queue_depths = (
                (
                    f"{encoder_name}_inq",
                    active_input_buffers,
                    "input",
                ),
                (
                    f"{encoder_name}_outq",
                    active_output_buffers,
                    "output",
                ),
            )
            for queue_name, active_depth, side in queue_depths:
                queue = self._pipeline.get_by_name(queue_name)
                if queue is None:
                    raise RuntimeError(
                        f"missing dataset encoder {side} queue {queue_name}"
                    )
                queue.set_property("leaky", 2)  # downstream
                queue.set_property(
                    "max-size-buffers",
                    active_depth if active else _DATASET_STARTUP_QUEUE_BUFFERS,
                )
                # Explicitly disable the queue defaults (10 MB / 1 s);
                # otherwise either can undercut the frame-count bound.
                queue.set_property("max-size-bytes", 0)
                queue.set_property("max-size-time", 0)

    def _attach_dataset_queue_diagnostics(
        self, gates: tuple[tuple[str, str | None], ...]
    ) -> None:
        """Log the exact relay queue that sheds an active-episode frame.

        Queue callbacks are event-only (no per-buffer Python probe), so they add
        no steady-state GIL work to the relay. Counts are rate-limited to the
        first and powers of two in case a failed recorder leaves a branch
        overflowing until teardown.
        """
        if self._pipeline is None:
            return
        queue_valves: dict[str, Any] = {}
        for valve_name, encoder_name in gates:
            if encoder_name is None:
                continue
            valve = self._pipeline.get_by_name(valve_name)
            if valve is None:
                _logger.debug("dataset queue diagnostic missing valve %s", valve_name)
                continue
            for queue_name in (
                f"{encoder_name}_srcq",
                f"{encoder_name}_inq",
                f"{encoder_name}_outq",
            ):
                queue_valves[queue_name] = valve
            if encoder_name.startswith("dsenc_"):
                queue_valves[f"eye_{encoder_name.rsplit('_', 1)[1]}_cropq"] = valve
        owner = repr(self)
        for queue_name, valve in sorted(queue_valves.items()):
            queue = self._pipeline.get_by_name(queue_name)
            if queue is None:
                _logger.debug("dataset queue diagnostic missing %s", queue_name)
                continue
            self._dataset_queue_overruns.setdefault(queue_name, 0)

            def on_overrun(
                element: Any,
                *,
                label: str = queue_name,
                branch_valve: Any = valve,
                source: str = owner,
            ) -> None:
                # Source/crop queues are live before an episode. An overrun
                # counts only after this branch's valve admits its first
                # dataset exposure; one stereo eye can open before its sibling.
                try:
                    if bool(branch_valve.get_property("drop")):
                        return
                except Exception:  # noqa: BLE001 - diagnostics only
                    return
                count = self._dataset_queue_overruns.get(label, 0) + 1
                self._dataset_queue_overruns[label] = count
                if count != 1 and count & (count - 1):
                    return
                try:
                    level = int(element.get_property("current-level-buffers"))
                    limit = int(element.get_property("max-size-buffers"))
                except Exception:  # noqa: BLE001 - diagnostics only
                    level = limit = -1
                _logger.warning(
                    "dataset relay %s queue %s overrun #%d "
                    "(level=%d, limit=%d); an encoded exposure may be lost",
                    source,
                    label,
                    count,
                    level,
                    limit,
                )

            try:
                queue.connect("overrun", on_overrun)
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                _logger.debug(
                    "could not attach overrun diagnostic to %s: %s",
                    queue_name,
                    exc,
                )

    def _begin_dataset_enable(
        self,
        gates: tuple[tuple[str, str | None], ...],
        target_perf_s: float,
    ) -> None:
        """Arm every branch to open on its first exposure at/after one instant.

        Dataset valves receive sensor-exposure PTS even while dropping buffers.
        Installing probes on all cameras before ``target_perf_s`` avoids the
        one/two-frame phase error caused by sequential property writes. Each
        camera has a different pipeline running-time origin, so the shared
        system-wide ``perf_counter`` target is converted with that pipeline's
        calibrated PTS offset.
        """
        self._cancel_dataset_disable()
        if self._dataset_enabled:
            # Relay commands are idempotent. Re-arming an already-open branch
            # would close it until the next future boundary, dropping H.264 AUs
            # from the live episode for no semantic state change.
            return
        self._cancel_dataset_enable()
        if self._pipeline is None:
            return
        if not math.isfinite(target_perf_s):
            raise ValueError("dataset enable target must be finite")
        if self._pts_perf_offset_s is None:
            raise RuntimeError(
                "camera pipeline PTS/perf_counter mapping is unavailable"
            )

        # Round upward: a sub-nanosecond conversion error must not admit an
        # exposure infinitesimally before the shared boundary.
        target_pts = math.ceil((target_perf_s - self._pts_perf_offset_s) * 1e9)
        armed: list[
            tuple[
                Any,
                int,
                threading.Event,
                threading.Event,
                threading.Lock,
                dict[str, str | None],
                Any,
                str,
            ]
        ] = []
        try:
            for queue_name in self._dataset_queue_overruns:
                self._dataset_queue_overruns[queue_name] = 0
            # Expand both sides before any valve can admit the shared-boundary
            # exposure. A partial property-update failure is caught below;
            # abort closes all valves before restoring startup depths.
            self._set_dataset_queue_depths(gates, active=True)
            for valve_name, encoder_name in gates:
                valve = self._pipeline.get_by_name(valve_name)
                if valve is None:
                    raise RuntimeError(f"missing dataset valve {valve_name!r}")
                # A common-frame open is valid only from the known closed state.
                # Closing here also makes repeated/partially failed requests
                # deterministic instead of leaking an already-open frame.
                valve.set_property("drop", True)
                pad = valve.get_static_pad("sink")
                if pad is None:
                    raise RuntimeError(f"missing dataset valve pad {valve_name!r}")
                opened = threading.Event()
                cancelled = threading.Event()
                transition_lock = threading.Lock()
                status: dict[str, str | None] = {"error": None}
                label = encoder_name or valve_name

                def open_at_common_exposure(
                    _pad: Any,
                    info: Any,
                    *,
                    branch_valve: Any = valve,
                    branch_opened: threading.Event = opened,
                    branch_cancelled: threading.Event = cancelled,
                    branch_lock: threading.Lock = transition_lock,
                    branch_status: dict[str, str | None] = status,
                    branch_label: str = label,
                    threshold_pts: int = target_pts,
                ) -> Any:
                    buf = info.get_buffer()
                    if buf is None:
                        return self._gst.PadProbeReturn.OK
                    if buf.pts == self._gst.CLOCK_TIME_NONE:
                        with branch_lock:
                            if not branch_cancelled.is_set():
                                branch_status["error"] = (
                                    f"dataset source {branch_label!r} has no exposure PTS"
                                )
                                branch_opened.set()
                        return self._gst.PadProbeReturn.REMOVE
                    if int(buf.pts) < threshold_pts:
                        return self._gst.PadProbeReturn.OK
                    with branch_lock:
                        if branch_cancelled.is_set():
                            return self._gst.PadProbeReturn.REMOVE
                        # This is a sink-pad probe: changing ``drop`` before the
                        # valve handles the current buffer admits this exposure,
                        # not merely the following one.
                        branch_valve.set_property("drop", False)
                        branch_opened.set()
                    return self._gst.PadProbeReturn.REMOVE

                probe_id = pad.add_probe(
                    self._gst.PadProbeType.BUFFER, open_at_common_exposure
                )
                if not probe_id:
                    raise RuntimeError(
                        f"could not install dataset enable probe on {valve_name!r}"
                    )
                armed.append(
                    (
                        pad,
                        probe_id,
                        opened,
                        cancelled,
                        transition_lock,
                        status,
                        valve,
                        label,
                    )
                )
        except Exception:
            self._dataset_enable = armed
            self._abort_dataset_enable(gates)
            raise
        self._dataset_enable = armed

    def _finish_dataset_enable(self, deadline: float) -> None:
        """Wait for all common-frame probes, failing the whole camera closed."""
        # Keep the list published while waiting so a concurrent abort/disconnect
        # can mark every transition cancelled and close its valve. Swapping it
        # out here would let a later streaming callback reopen after fail-close.
        pending = self._dataset_enable
        failures: list[str] = []
        for (
            _pad,
            _probe_id,
            opened,
            cancelled,
            _lock,
            status,
            _valve,
            label,
        ) in pending:
            remaining = max(0.0, deadline - time.perf_counter())
            if not opened.wait(remaining) and not opened.is_set():
                failures.append(f"{label}: timed out waiting for exposure boundary")
                continue
            error = status["error"]
            if error is not None:
                failures.append(error)
            elif cancelled.is_set():
                failures.append(f"{label}: exposure-boundary opening was cancelled")
        if failures:
            # Cancel only if this is still the transaction we waited on. A
            # concurrent abort already closed it; a later begin owns a new list
            # which this failed waiter must not touch.
            if self._dataset_enable is pending:
                self._cancel_dataset_enable()
            raise RuntimeError("; ".join(failures))
        if self._dataset_enable is pending:
            self._dataset_enable = []
        self._dataset_enabled = True

    def _begin_dataset_disable(self, gates: tuple[tuple[str, str | None], ...]) -> None:
        """Close every dataset input valve immediately.

        Every encoded AU is an IDR, so there is no predictive GOP phase to park.
        NVENC may still emit a small in-flight tail after the close; the reader's
        quiet flush drains it before the next shared-timestamp open.
        """
        self._cancel_dataset_enable()
        self._cancel_dataset_disable()
        if self._pipeline is None:
            return
        try:
            for valve_name, _encoder_name in gates:
                valve = self._pipeline.get_by_name(valve_name)
                if valve is None:
                    raise RuntimeError(f"missing dataset valve {valve_name!r}")
                valve.set_property("drop", True)
        except Exception:
            self._dataset_enabled = False
            for valve_name, _encoder_name in gates:
                valve = self._pipeline.get_by_name(valve_name)
                if valve is not None:
                    valve.set_property("drop", True)
            self._set_dataset_queue_depths(gates, active=False)
            raise
        self._dataset_disable = []

    def _finish_dataset_disable(self, deadline: float) -> None:
        """Complete an immediate all-intra dataset close."""
        del deadline
        self._dataset_disable = []
        self._dataset_enabled = False

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

    @property
    def pts_perf_offset_s(self) -> float:
        """Map this pipeline's running-time PTS seconds to ``perf_counter``.

        The mapping is calibrated once after the pipeline reaches PLAYING and
        remains valid for its lifetime (the pipeline is never paused/rebased).
        It is safe to send to another process because Python's ``perf_counter``
        is a system-wide monotonic clock on supported platforms.
        """
        if self._pts_perf_offset_s is None:
            raise RuntimeError("gst pipeline PTS/perf_counter mapping is unavailable")
        return self._pts_perf_offset_s

    def _calibrate_pts_perf_offset(self) -> float | None:
        """Return ``perf_counter_s - pipeline_running_time_s`` with low jitter.

        Reading the two clocks is not atomic. Take several bracketed samples and
        use the one with the shortest ``perf_counter`` bracket; its midpoint
        bounds the cross-clock sampling error to half that (normally a few us).
        The ZED source stamps exposure as pipeline running time, hence applying
        this one offset recovers exposure time without any receive-time bias.
        """
        if self._clock is None or self._pipeline is None:
            return None
        base_ns = self._pipeline.get_base_time()
        if base_ns == self._gst.CLOCK_TIME_NONE:
            return None
        best: tuple[int, float] | None = None
        for _ in range(9):
            perf_before_ns = time.perf_counter_ns()
            clock_ns = self._clock.get_time()
            perf_after_ns = time.perf_counter_ns()
            if clock_ns == self._gst.CLOCK_TIME_NONE:
                continue
            bracket_ns = perf_after_ns - perf_before_ns
            perf_mid_ns = (perf_before_ns + perf_after_ns) // 2
            running_ns = clock_ns - base_ns
            offset_s = (perf_mid_ns - running_ns) / 1e9
            if best is None or bracket_ns < best[0]:
                best = (bracket_ns, offset_s)
        return None if best is None else best[1]

    def _cap_perf_from_pts(self, pts: int, recv_perf: float) -> float:
        """Map a buffer running-time PTS onto ``time.perf_counter`` seconds.

        With our patched ``zedxonesrc``/``zedsrc`` (and ``do-timestamp=false``),
        ``pts`` is the pipeline running-time of the true sensor-exposure instant
        (``TIME_REFERENCE::IMAGE``), not host-receive time. The offset calibrated
        when the pipeline entered PLAYING maps that running-time directly onto
        the system-wide ``perf_counter`` timeline -- parity with the ZED SDK
        ``ZedCamera`` path, so image frames align with joint samples without
        receive/encoder/scheduler latency in the result.
        """
        if pts == self._gst.CLOCK_TIME_NONE:
            raise RuntimeError("raw camera buffer has no sensor PTS")
        if self._pts_perf_offset_s is None:
            raise RuntimeError("gst pipeline PTS/perf_counter mapping is unavailable")
        return pts / 1e9 + self._pts_perf_offset_s

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

    def _make_au_handler(self, channel: _AUChannel, label: str = "enc") -> Any:
        # Rolling per-second encoder-output health: the headset H.264 stream's
        # actual fps + bitrate + bytes/frame. If these hold steady while the feed
        # looks grainy, the encoder is fine and the loss is downstream (transport);
        # if fps/bytes collapse, NVENC itself isn't keeping up.
        stat = {"frames": 0, "bytes": 0, "last": time.perf_counter()}

        def handle(buf: Any, _recv_perf: float) -> None:
            ok, mapinfo = buf.map(self._gst.MapFlags.READ)
            if not ok:
                return
            try:
                size = mapinfo.size
                nals = _split_nals(bytes(mapinfo.data))
            finally:
                buf.unmap(mapinfo)
            stat["frames"] += 1
            stat["bytes"] += size
            now = time.perf_counter()
            dt = now - stat["last"]
            if dt >= 1.0:
                _logger.debug(
                    "relay-enc %s: %.1f fps  %.0f kbps  %.1f KB/frame",
                    label,
                    stat["frames"] / dt,
                    8e-3 * stat["bytes"] / dt,
                    stat["bytes"] / stat["frames"] / 1024 if stat["frames"] else 0.0,
                )
                stat["frames"] = 0
                stat["bytes"] = 0
                stat["last"] = now
            if nals:
                channel.broadcast(nals)

        return handle

    def _make_raw_handler(self, sink: Any, w: int, h: int) -> Any:
        """Pull handler that hands each raw frame to ``sink(rgba, cap, recv)``.

        ``rgba`` is a zero-copy ``(H, W, 4)`` view over the GStreamer buffer,
        valid only for the duration of the call — the sink must copy what it
        keeps. ``_RawBuffer`` (in-process) copies the full RGBA; the relay's
        shared-memory writer copies just RGB across the process boundary.
        """
        import numpy as np

        def handle(buf: Any, recv_perf: float) -> None:
            ok, mapinfo = buf.map(self._gst.MapFlags.READ)
            if not ok:
                return
            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                # nvvidconv may pad rows; trust the negotiated WxHx4 size.
                if arr.size < w * h * 4:
                    return
                rgba = arr[: w * h * 4].reshape(h, w, 4)
                sink(rgba, self._cap_perf_from_pts(buf.pts, recv_perf), recv_perf)
            finally:
                buf.unmap(mapinfo)

        return handle

    @staticmethod
    def _buffer_sink(raw: _RawBuffer) -> Any:
        """Default raw sink: copy the RGBA view into an in-process ``_RawBuffer``."""

        def sink(rgba: Any, cap_ts: float, recv_ts: float) -> None:
            raw.set(rgba.copy(), cap_ts, recv_ts)

        return sink

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
        self._pts_perf_offset_s = self._calibrate_pts_perf_offset()
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
        self._cancel_dataset_enable()
        self._dataset_enabled = False
        self._cancel_dataset_disable()
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
        raw_sink: Any = None,
        raw_socket_path: str | None = None,
        raw_dims: tuple[int, int] | None = None,
        dataset_fps: int | None = None,
    ) -> None:
        _GstPipelineBase.__init__(self)
        if resolution not in _RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported ZED X One resolution {resolution!r} "
                f"(expected one of {', '.join(_RESOLUTION_ENUM)})"
            )
        # A custom raw sink (the relay's shared-memory writer) or a shmsink socket
        # path both imply the raw branch and replace the in-process _RawBuffer: no
        # frame is stored locally — the consumer reads from the sink / shm instead.
        # ``raw_socket_path`` routes the raw branch through gst's native shmsink so
        # the relay does no Python per raw frame (the fix for the recording feed).
        want_raw = want_raw or raw_sink is not None or raw_socket_path is not None
        if not (want_encoded or want_raw):
            raise ValueError("ZedGstCamera needs at least one of encoded/raw")
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        self.stream_fps = HEADSET_STREAM_FPS
        self.dataset_fps = fps if dataset_fps is None else int(dataset_fps)
        if self.dataset_fps <= 0 or self.dataset_fps > self.fps:
            raise ValueError(
                f"dataset fps must be in [1, {self.fps}], got {self.dataset_fps}"
            )
        self.width, self.height = _RESOLUTION_DIMS[resolution]
        # The raw (dataset) branch can be downscaled on the VIC to cut the bytes
        # that cross to the control process; the encoded headset branch always
        # keeps the full capture resolution. ``raw_dims`` is the relay's target.
        self.raw_width, self.raw_height = raw_dims or (self.width, self.height)
        self._want_encoded = want_encoded
        self._want_raw = want_raw
        self._raw_sink_override = raw_sink
        self._raw_socket_path = raw_socket_path
        self._enc = _AUChannel(lambda: self.alive) if want_encoded else None
        self._raw = (
            _RawBuffer(self.raw_width, self.raw_height)
            if want_raw and raw_sink is None and raw_socket_path is None
            else None
        )
        self._alive_fn = lambda: self.alive

    def __repr__(self) -> str:
        return f"ZedGstCamera(serial={self.serial})"

    def _pipeline_str(self) -> str:
        bitrate = _bitrate_for(self.width, self.height, self.stream_fps)
        src = (
            f"zedxonesrc camera-sn={self.serial} "
            f"camera-resolution={_RESOLUTION_ENUM[self.resolution]} "
            f"camera-fps={self.fps} stream-type=1 do-timestamp=false "
            f"ctrl-auto-exposure-range-max={_MAX_AUTO_EXPOSURE_US} "
            "! video/x-raw(memory:NVMM),format=NV12"
        )
        rate = _stream_rate_limit(self.fps, self.stream_fps)
        enc = (
            f"{_QUEUE} ! {rate}"
            f"{_enc_branch(bitrate, self.stream_fps)} ! {_enc_appsink('enc')}"
        )
        # The dataset branch sits behind a `valve` so it can be gated shut at
        # runtime (see set_raw_enabled): its work is only needed while recording.
        # Defaults open so the SDK-less consumers (inference/run-policy) are
        # unchanged; `collect-data` explicitly closes it until an episode records.
        # nvvidconv (the VIC) resizes for free on the GPU, so the smaller dataset
        # dims downscale here without touching the CPU or the headset branch.
        # Recorder (shmsink) path: encode the dataset stream on the GPU and ship
        # H.264 AUs — no raw copy, no recorder re-encode (see
        # _dataset_enc_shmsink). In-process raw consumers (inference/run-policy,
        # or the pyshm RawFrameWriter fallback) still take RGBA off an appsink.
        if self._raw_socket_path:
            dataset_rate = _dataset_rate_limit(self.fps, self.dataset_fps)
            raw = (
                f"{_dataset_source_queue('dsenc')} ! {dataset_rate}"
                "valve name=rawvalve drop=false ! "
                + _dataset_enc_shmsink(
                    self._raw_socket_path,
                    self.raw_width,
                    self.raw_height,
                    self.dataset_fps,
                    "dsenc",
                )
            )
        else:
            raw = (
                f"{_QUEUE} ! valve name=rawvalve drop=false "
                f"! nvvidconv ! video/x-raw,format=RGBA,"
                f"width={self.raw_width},height={self.raw_height} ! {_raw_appsink('raw')}"
            )
        if self._want_encoded and self._want_raw:
            return f"{src} ! tee name=t  t. ! {enc}  t. ! {raw}"
        if self._want_encoded:
            return f"{src} ! {enc}"
        return f"{src} ! {raw}"

    def _raw_gates(self) -> tuple[tuple[str, str | None], ...]:
        if not self._want_raw:
            return ()
        return (("rawvalve", "dsenc" if self._raw_socket_path else None),)

    def begin_raw_disable(self) -> None:
        """Close this camera's all-intra dataset input."""
        self._begin_dataset_disable(self._raw_gates())

    def finish_raw_disable(self, deadline: float) -> None:
        """Complete a close previously armed by :meth:`begin_raw_disable`."""
        try:
            self._finish_dataset_disable(deadline)
        finally:
            self._set_dataset_queue_depths(self._raw_gates(), active=False)

    def begin_raw_enable(self, target_perf_s: float) -> None:
        """Arm this camera to open on a shared future exposure boundary."""
        self._begin_dataset_enable(self._raw_gates(), target_perf_s)

    def finish_raw_enable(self, deadline: float) -> None:
        """Wait for a common-boundary open armed by :meth:`begin_raw_enable`."""
        try:
            self._finish_dataset_enable(deadline)
        except Exception:
            # Local callers do not have the relay's outer multi-camera abort.
            # Preserve the same fail-closed queue/valve state either way.
            self._abort_dataset_enable(self._raw_gates())
            raise

    def abort_raw_enable(self) -> None:
        """Cancel an opening transaction and immediately close every branch."""
        self._abort_dataset_enable(self._raw_gates())

    def set_raw_enabled(self, enabled: bool) -> None:
        """Open or close the dataset branch at runtime (no pipeline reconfig).

        Toggles the dataset branch's ``valve``: when dropped, its GPU dataset
        encode (shmsink path) or VIC RGBA convert + appsink copy (in-process path)
        never runs, so the relay costs no more than the encode-only (``axol
        teleop``) path. ``collect-data`` opens this only while an episode is
        recording. The independent 30 fps headset branch keeps the same bitrate
        and spatial quality on both sides of this gate.
        """
        if not self._want_raw or self._pipeline is None:
            return
        if enabled:
            target = time.perf_counter() + _DATASET_ENABLE_LEAD_S
            self.begin_raw_enable(target)
            self.finish_raw_enable(target + _DATASET_GATE_TIMEOUT_S)
            return
        self.begin_raw_disable()
        self.finish_raw_disable(time.perf_counter() + _DATASET_GATE_TIMEOUT_S)

    def connect(self, warmup: bool = True) -> None:
        """Open the camera, start the pipeline, and block until it streams."""
        self._launch(self._pipeline_str())
        self._attach_dataset_queue_diagnostics(self._raw_gates())
        if self._enc is not None:
            self._start_pull(
                f"zedgst-{self.serial}-enc",
                "enc",
                self._make_au_handler(self._enc, f"sn{self.serial}"),
            )
        # On the shmsink path the frame copy happens in gst's C threads (no
        # Python pull loop here), so the relay's interpreter stays free for the
        # WebRTC send. Only the appsink path needs a Python pull thread.
        if self._want_raw and self._raw_socket_path is None:
            sink = self._raw_sink_override or self._buffer_sink(self._raw)
            self._start_pull(
                f"zedgst-{self.serial}-raw",
                "raw",
                self._make_raw_handler(sink, self.raw_width, self.raw_height),
            )
        channels = (self._enc,) if self._enc is not None else ()
        if not self._play_and_wait(channels):
            self.disconnect()
            raise RuntimeError(
                f"ZedGstCamera(serial={self.serial}) did not start streaming "
                f"within {_READY_TIMEOUT_S:.0f}s (camera absent or in use?)."
            )
        if self._want_raw and self._pts_perf_offset_s is None:
            self.disconnect()
            raise RuntimeError(
                "camera pipeline could not calibrate its sensor-PTS clock"
            )
        _logger.info(
            "ZedGstCamera connected (sn=%d %dx%d capture=%dfps stream=%dfps, "
            "encoded=%s raw=%s).",
            self.serial,
            self.width,
            self.height,
            self.fps,
            self.stream_fps,
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

    ``eyes`` selects which eye(s) are built: ``"both"`` (the head camera) crops
    and encodes the full pair, while ``"left"`` / ``"right"`` build a single eye
    only — the wrist policy, where a stereo camera streams/records just its left
    eye so it costs no more than a mono one. The unbuilt eye's view is ``None``.

    ``encoded_sbs`` packs both eyes into a **single side-by-side stream**
    instead of two per-eye ones: ``zedsrc`` already delivers the stereo pair as
    one double-width NVMM frame, so the SBS branch simply encodes it uncropped —
    one NVENC session here and, crucially, one decoder session on the headset
    (two 1920x1200 decoder sessions measurably drop the Quest's render rate).
    The packed stream is exposed as :attr:`sbs_view`; the raw (dataset) branch
    still crops per eye, so recordings are unchanged.
    """

    def __init__(
        self,
        serial: int,
        resolution: str = "HD1200",
        fps: int = 60,
        *,
        want_encoded: bool = True,
        want_raw: bool = False,
        left_raw_sink: Any = None,
        right_raw_sink: Any = None,
        left_raw_socket_path: str | None = None,
        right_raw_socket_path: str | None = None,
        raw_dims: tuple[int, int] | None = None,
        dataset_fps: int | None = None,
        eyes: str = "both",
        encoded_eyes: "list[str] | tuple[str, ...] | None" = None,
        raw_eyes: "list[str] | tuple[str, ...] | None" = None,
        encoded_sbs: bool = False,
    ) -> None:
        _GstPipelineBase.__init__(self)
        if resolution not in _STEREO_RESOLUTION_ENUM:
            raise ValueError(
                f"unsupported stereo ZED X resolution {resolution!r} "
                f"(expected one of {', '.join(_STEREO_RESOLUTION_ENUM)})"
            )
        # Which eye(s) to actually crop + encode/convert. A wrist stereo camera
        # streams/records only its left eye (``eyes="left"``), so the second
        # NVENC encode and VIC convert are never built — making a stereo wrist
        # cost exactly as much as a mono one. ``eyes="both"`` (the head camera)
        # builds the full per-eye pair.
        if eyes not in ("both", "left", "right"):
            raise ValueError(f"eyes must be 'both', 'left', or 'right'; got {eyes!r}.")
        # A per-eye raw sink (the relay's shared-memory writer) or a shmsink socket
        # path implies that eye's raw branch and replaces its in-process _RawBuffer.
        want_raw = (
            want_raw
            or left_raw_sink is not None
            or right_raw_sink is not None
            or left_raw_socket_path is not None
            or right_raw_socket_path is not None
        )
        # The encoded (headset stream) and raw (dataset recording) branches can
        # select different eyes — e.g. stream both eyes for depth while recording
        # only the left. ``encoded_eyes`` / ``raw_eyes`` override per branch; when
        # unset each falls back to ``eyes`` (the legacy coupled behaviour, gated by
        # the corresponding ``want_*``). Each eye is cropped once and tee'd into
        # whichever branch(es) want it, so a side present in neither is never built.
        base_sides = ("left", "right") if eyes == "both" else (eyes,)

        def _order(sides: "list[str] | tuple[str, ...]") -> tuple[str, ...]:
            for s in sides:
                if s not in ("left", "right"):
                    raise ValueError(f"eye must be 'left' or 'right'; got {s!r}.")
            return tuple(s for s in ("left", "right") if s in sides)

        # The SBS branch replaces the per-eye encodes (one packed stream instead
        # of two): with ``encoded_sbs`` no per-eye encoded side is built.
        self._encoded_sbs = encoded_sbs
        self._encoded_sides = (
            ()
            if encoded_sbs
            else _order(
                encoded_eyes
                if encoded_eyes is not None
                else (base_sides if want_encoded else ())
            )
        )
        self._raw_sides = _order(
            raw_eyes if raw_eyes is not None else (base_sides if want_raw else ())
        )
        want_encoded = bool(self._encoded_sides) or encoded_sbs
        want_raw = bool(self._raw_sides)
        # Build (crop) every eye either branch needs, ordered left-then-right.
        self._sides: tuple[str, ...] = _order(
            tuple(self._encoded_sides) + tuple(self._raw_sides)
        )
        self.eyes = eyes
        if not (want_encoded or want_raw):
            raise ValueError("ZedGstStereoCamera needs at least one of encoded/raw")
        self.serial = serial
        self.resolution = resolution
        self.fps = fps
        self.stream_fps = HEADSET_STREAM_FPS
        self.dataset_fps = fps if dataset_fps is None else int(dataset_fps)
        if self.dataset_fps <= 0 or self.dataset_fps > self.fps:
            raise ValueError(
                f"dataset fps must be in [1, {self.fps}], got {self.dataset_fps}"
            )
        self.width, self.height = _RESOLUTION_DIMS[resolution]
        # Per-eye downscale target for the raw (dataset) branch; encoded eyes keep
        # the full capture resolution. See ZedGstCamera for the rationale.
        self.raw_width, self.raw_height = raw_dims or (self.width, self.height)
        self._want_encoded = want_encoded
        self._want_raw = want_raw
        self._left_raw_sink = left_raw_sink
        self._right_raw_sink = right_raw_sink
        self._left_raw_socket_path = left_raw_socket_path
        self._right_raw_socket_path = right_raw_socket_path

        def eye(
            side: str, raw_sink: Any, socket_path: str | None
        ) -> tuple[_AUChannel | None, _RawBuffer | None, _GstEye]:
            enc = (
                _AUChannel(lambda: self.alive) if side in self._encoded_sides else None
            )
            raw = (
                _RawBuffer(self.raw_width, self.raw_height)
                if side in self._raw_sides and raw_sink is None and socket_path is None
                else None
            )
            view_fps = self.fps if side in self._raw_sides else self.stream_fps
            view = _GstEye(self, enc, raw, self.width, self.height, view_fps)
            return enc, raw, view

        # Only build the eye(s) named in ``self._sides``; the unbuilt eye's
        # view is ``None`` (callers wire up only the eyes they asked for).
        self._left_enc = self._left_raw = None
        self._right_enc = self._right_raw = None
        self.left_view: _GstEye | None = None
        self.right_view: _GstEye | None = None
        if "left" in self._sides:
            self._left_enc, self._left_raw, self.left_view = eye(
                "left", left_raw_sink, left_raw_socket_path
            )
        if "right" in self._sides:
            self._right_enc, self._right_raw, self.right_view = eye(
                "right", right_raw_sink, right_raw_socket_path
            )
        # Packed side-by-side stream: both eyes in one double-width frame/track.
        self._sbs_enc: _AUChannel | None = None
        self.sbs_view: _GstEye | None = None
        if encoded_sbs:
            self._sbs_enc = _AUChannel(lambda: self.alive)
            self.sbs_view = _GstEye(
                self,
                self._sbs_enc,
                None,
                self.width * 2,
                self.height,
                self.stream_fps,
            )

    def __repr__(self) -> str:
        return f"ZedGstStereoCamera(serial={self.serial})"

    def _eye_branch(self, side: str, sink_suffix: str) -> str:
        """One eye: crop its half on the VIC, then encode and/or raw appsink.

        Encode and raw are gated per eye (``self._encoded_sides`` /
        ``self._raw_sides``), so this eye may be encode-only (headset),
        raw-only (dataset), or both (tee'd) depending on which branch asked
        for it.
        """
        want_encoded = side in self._encoded_sides
        want_raw = side in self._raw_sides
        eye_w, eye_h = self.width, self.height
        left = 0 if side == "left" else eye_w
        right = eye_w if side == "left" else eye_w * 2
        bitrate = _bitrate_for(eye_w, eye_h, self.stream_fps)
        rate = _stream_rate_limit(self.fps, self.stream_fps)
        caps = (
            f"video/x-raw(memory:NVMM),format=NV12,width={eye_w},height={eye_h},"
            "pixel-aspect-ratio=1/1"
        )
        crop = (
            f"queue name=eye_{sink_suffix}_cropq leaky=downstream "
            "max-size-buffers=2 ! "
            f"nvvidconv left={left} right={right} top=0 bottom={eye_h} ! {caps}"
        )
        sock = (
            self._left_raw_socket_path
            if sink_suffix == "l"
            else self._right_raw_socket_path
        )
        # This eye's dataset branch: encode->shmsink (recorder) when it has a
        # socket, else RGBA appsink (in-process writer / inference). Both sit
        # behind a per-eye valve so set_raw_enabled can gate them while not
        # recording. See the mono _pipeline_str note.
        if sock:
            dataset_rate = _dataset_rate_limit(self.fps, self.dataset_fps)
            raw = (
                f"{_dataset_source_queue('dsenc_' + sink_suffix)} ! {dataset_rate}"
                f"valve name=rawvalve_{sink_suffix} drop=false ! "
                + _dataset_enc_shmsink(
                    sock,
                    self.raw_width,
                    self.raw_height,
                    self.dataset_fps,
                    "dsenc_" + sink_suffix,
                )
            )
        else:
            raw = (
                f"{_QUEUE} ! valve name=rawvalve_{sink_suffix} drop=false "
                f"! nvvidconv ! video/x-raw,format=RGBA,"
                f"width={self.raw_width},height={self.raw_height} ! "
                f"{_raw_appsink('raw_' + sink_suffix)}"
            )
        if want_encoded and want_raw:
            enc = (
                f"{_QUEUE} ! {rate}"
                f"{_enc_branch(bitrate, self.stream_fps, 'venc_' + sink_suffix)} ! "
                f"{_enc_appsink('enc_' + sink_suffix)}"
            )
            return f"{crop} ! tee name=t{sink_suffix}  t{sink_suffix}. ! {enc}  t{sink_suffix}. ! {raw}"
        if want_encoded:
            return (
                f"{crop} ! {rate}"
                f"{_enc_branch(bitrate, self.stream_fps, 'venc_' + sink_suffix)} ! "
                f"{_enc_appsink('enc_' + sink_suffix)}"
            )
        return f"{crop} ! {raw}"

    def _sbs_branch(self) -> str:
        """Encode the uncropped double-width stereo frame as one packed stream.

        ``zedsrc`` delivers the pair side-by-side in a single NVMM buffer, so no
        crop or compose is needed — the frame goes straight into NVENC. The
        bitrate is twice the per-eye default (same total bits as two per-eye
        streams, just in one track).
        """
        bitrate = 2 * _bitrate_for(self.width, self.height, self.stream_fps)
        rate = _stream_rate_limit(self.fps, self.stream_fps)
        return (
            f"{_QUEUE} ! {rate}"
            f"{_enc_branch(bitrate, self.stream_fps, 'venc_s')} ! "
            f"{_enc_appsink('enc_s')}"
        )

    def _raw_gates(self) -> tuple[tuple[str, str | None], ...]:
        gates: list[tuple[str, str | None]] = []
        for side in self._raw_sides:
            suffix = side[0]
            socket_path = (
                self._left_raw_socket_path
                if side == "left"
                else self._right_raw_socket_path
            )
            gates.append(
                (f"rawvalve_{suffix}", f"dsenc_{suffix}" if socket_path else None)
            )
        return tuple(gates)

    def begin_raw_disable(self) -> None:
        """Close every all-intra dataset-eye input."""
        self._begin_dataset_disable(self._raw_gates())

    def finish_raw_disable(self, deadline: float) -> None:
        """Complete a close previously armed by :meth:`begin_raw_disable`."""
        try:
            self._finish_dataset_disable(deadline)
        finally:
            self._set_dataset_queue_depths(self._raw_gates(), active=False)

    def begin_raw_enable(self, target_perf_s: float) -> None:
        """Arm both eyes to open on a shared future exposure boundary."""
        self._begin_dataset_enable(self._raw_gates(), target_perf_s)

    def finish_raw_enable(self, deadline: float) -> None:
        """Wait for a common-boundary open armed by :meth:`begin_raw_enable`."""
        try:
            self._finish_dataset_enable(deadline)
        except Exception:
            self._abort_dataset_enable(self._raw_gates())
            raise

    def abort_raw_enable(self) -> None:
        """Cancel an opening transaction and immediately close every eye."""
        self._abort_dataset_enable(self._raw_gates())

    def set_raw_enabled(self, enabled: bool) -> None:
        """Open or close both eyes' dataset branches at runtime.

        See :meth:`ZedGstCamera.set_raw_enabled`: gates the per-eye ``valve``s so
        each eye's dataset encode (or VIC convert + appsink copy) only runs while
        recording. The independent 30 fps headset branch is unchanged.
        """
        if not self._want_raw or self._pipeline is None:
            return
        if enabled:
            target = time.perf_counter() + _DATASET_ENABLE_LEAD_S
            self.begin_raw_enable(target)
            self.finish_raw_enable(target + _DATASET_GATE_TIMEOUT_S)
            return
        self.begin_raw_disable()
        self.finish_raw_disable(time.perf_counter() + _DATASET_GATE_TIMEOUT_S)

    def _pipeline_str(self) -> str:
        src = (
            f"zedsrc camera-sn={self.serial} "
            f"camera-resolution={_STEREO_RESOLUTION_ENUM[self.resolution]} "
            f"camera-fps={self.fps} stream-type=7 depth-mode=0 "
            "do-timestamp=false "
            "! video/x-raw(memory:NVMM),format=NV12 ! tee name=split"
        )
        parts = [f"split. ! {self._eye_branch(side, side[0])}" for side in self._sides]
        if self._encoded_sbs:
            parts.append(f"split. ! {self._sbs_branch()}")
        branches = "  ".join(parts)
        return f"{src}  {branches}"

    def connect(self, warmup: bool = True) -> None:
        self._launch(self._pipeline_str())
        self._attach_dataset_queue_diagnostics(self._raw_gates())
        eye_specs = [
            (
                "left",
                self._left_enc,
                self._left_raw,
                self._left_raw_sink,
                self._left_raw_socket_path,
                "l",
            ),
            (
                "right",
                self._right_enc,
                self._right_raw,
                self._right_raw_sink,
                self._right_raw_socket_path,
                "r",
            ),
        ]
        for side, enc, raw, raw_sink, sock, suffix in eye_specs:
            if side not in self._sides:
                continue
            if enc is not None:
                self._start_pull(
                    f"zedgst-{self.serial}-enc{suffix}",
                    f"enc_{suffix}",
                    self._make_au_handler(enc, f"sn{self.serial}-{suffix}"),
                )
            # shmsink writes the frame in C; only the appsink path needs a Python
            # pull thread (which would contend with the relay's send — the bug).
            if side in self._raw_sides and sock is None:
                sink = raw_sink or self._buffer_sink(raw)
                self._start_pull(
                    f"zedgst-{self.serial}-raw{suffix}",
                    f"raw_{suffix}",
                    self._make_raw_handler(sink, self.raw_width, self.raw_height),
                )
        if self._sbs_enc is not None:
            self._start_pull(
                f"zedgst-{self.serial}-encs",
                "enc_s",
                self._make_au_handler(self._sbs_enc, f"sn{self.serial}-sbs"),
            )
        channels = tuple(
            c for c in (self._left_enc, self._right_enc, self._sbs_enc) if c is not None
        )
        if not self._play_and_wait(channels):
            self.disconnect()
            raise RuntimeError(
                f"ZedGstStereoCamera(serial={self.serial}) did not start "
                f"streaming within {_READY_TIMEOUT_S:.0f}s."
            )
        if self._want_raw and self._pts_perf_offset_s is None:
            self.disconnect()
            raise RuntimeError(
                "camera pipeline could not calibrate its sensor-PTS clock"
            )
        _logger.info(
            "ZedGstStereoCamera connected (sn=%d %dx%d/eye capture=%dfps "
            "stream=%dfps).",
            self.serial,
            self.width,
            self.height,
            self.fps,
            self.stream_fps,
        )
