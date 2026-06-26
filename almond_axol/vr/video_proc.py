"""Out-of-process WebRTC video relay (GPU-resident gst grab + aiortc).

Sending video from inside the teleop process measurably starves the
control loops: pushing thousands of RTP packets per second plus encoding
is real CPU/GIL work, and it stretches the IK round-trip the moment a
headset connects. The same isolation pattern used for the IK solver
applies here — video runs in a dedicated subprocess and the control
process never touches it.

The subprocess owns the ZED cameras through :mod:`almond_axol.vr.gst_zed`:
the ``zedxonesrc`` / ``zedsrc`` GStreamer elements grab and NVENC-encode
entirely on the GPU, and Python only ever sees the encoded H.264 access
units, which aiortc forwards as pre-encoded packets (no Python encode step).
If the gst stack is unavailable the relay falls back to the ZED SDK grab +
in-Python NVENC path (``hw_video``). Either way aiortc owns the ICE / DTLS /
SRTP transport, which connects reliably on this multi-homed LAN where
gstreamer ``webrtcbin``'s libnice stalls. WebRTC media flows over the
subprocess's own UDP sockets, so the only traffic crossing the process
boundary is SDP signaling (a few messages per headset connection) over a
``multiprocessing`` pipe.

:class:`VideoRelayProcess` is the parent-side handle. It implements the
same async interface as ``WebRTCManager`` (``create_offer`` /
``set_answer`` / ``close`` / ``close_all`` / ``has_sources``), so
``VRServer.set_video_manager`` can use it as a drop-in.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.connection
import os
import threading

_logger = logging.getLogger(__name__)

# How long the relay subprocess may take to open every camera and report
# readiness (each camera takes a few seconds to start streaming).
_READY_TIMEOUT_S = 60.0

_REQUEST_TIMEOUT_S = 15.0


# ---------------------------------------------------------------------------
# Subprocess side
# ---------------------------------------------------------------------------


def _open_sdk_camera(name: str, spec: dict) -> object | None:
    """Open one ZED camera with the Python SDK, preferring 60 fps capture.

    Returns the connected ``ZedCamera`` / ``ZedStereoCamera`` (or ``None`` if
    the camera is absent). 60 fps halves frame staleness vs the GMSL 30 fps
    default; cameras that reject it fall back to their default rate.
    """
    from ..lerobot.camera.camera_zed import ZedCamera, ZedStereoCamera
    from ..lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS, ZedCameraConfig

    serial = spec["serial"]
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    dims = ZED_RESOLUTION_DIMS.get(resolution)
    width, height = dims if dims is not None else (None, None)
    cls = ZedStereoCamera if stereo else ZedCamera
    for fps in (spec.get("fps", 60), None):
        cam = cls(
            ZedCameraConfig(
                serial=serial,
                fps=fps,
                width=width,
                height=height,
                stereo=stereo,
            )
        )
        try:
            cam.connect(warmup=False)
            return cam
        except RuntimeError as exc:  # live-param mismatch (e.g. 60 fps) → retry
            _logger.info("video relay: %s rejected %s fps (%s)", name, fps, exc)
        except Exception as exc:  # noqa: BLE001 - camera absent → skip it
            _logger.warning("video relay: %s failed to open (%s)", name, exc)
            return None
    return None


def _raw_caps(width: int, height: int, fps: int) -> str:
    """gst caps for the raw RGBA frames the recorder's shmsrc must declare.

    Shared memory carries no caps, so the recorder's ``shmsrc`` needs these
    explicitly to interpret the bytes; they must match the relay's shmsink input
    (the ``nvvidconv`` RGBA output at the downscaled dataset dims).
    """
    return f"video/x-raw,format=RGBA,width={width},height={height},framerate={fps}/1"


def _gstshm_meta(
    socket_path: str, caps: str, width: int, height: int, fps: int, latency_s: float
) -> dict:
    return {
        "transport": "gstshm",
        "socket_path": socket_path,
        "caps": caps,
        "width": width,
        "height": height,
        "fps": fps,
        "latency_s": latency_s,
    }


def _pyshm_meta(shm_name: str, width: int, height: int, fps: int) -> dict:
    return {
        "transport": "pyshm",
        "shm_name": shm_name,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _eye_plan(name: str, spec: dict) -> list[tuple[str, str]]:
    """``[(eye_side, source_name)]`` for a stereo camera spec.

    ``spec["eyes"]`` lists the eyes to build (default both); ``spec["eye_suffix"]``
    (default True) decides naming. The head camera streams/records both eyes as
    ``{name}_left`` / ``{name}_right``; a wrist stereo camera builds only its left
    eye and exposes it under the plain ``{name}`` (``eye_suffix=False``), so it is
    indistinguishable from a mono wrist downstream — one encode, one source.
    """
    eyes = spec.get("eyes") or ["left", "right"]
    suffix = spec.get("eye_suffix", True)
    return [(side, f"{name}_{side}" if suffix else name) for side in eyes]


def _open_gst_camera_raw(
    name: str, spec: dict, cond: object, socket_dir: str | None
) -> tuple[object, dict[str, object], list, dict[str, dict]] | None:
    """Open one camera via the gst pipeline with both encoded + raw branches.

    Like :func:`_open_gst_camera`, but additionally exports each source's raw
    frames to the recorder process for the dataset. Two transports:

    * **gstshm** (``socket_dir`` set — gst's ``shm`` plugin is available): the raw
      branch ends in a native ``shmsink`` (pure C), so the relay does **zero**
      Python per raw frame and its interpreter stays free for the WebRTC send.
      The recorder reads via ``shmsrc`` (:class:`GstShmFrameReader`).
    * **pyshm** (fallback): a Python pull loop copies each frame into a
      :class:`RawFrameWriter` shared-memory block (the older path; runs the copy
      in the relay's interpreter).

    Returns ``(owned_camera, {track: source}, [writers], {source: meta})`` — where
    ``meta`` is the per-source dict from :func:`_gstshm_meta` / :func:`_pyshm_meta`
    — or ``None`` when the gst stack/camera is unavailable (the caller then falls
    back to the in-process camera pipeline).
    """
    from .gst_zed import (
        _RESOLUTION_DIMS,
        ZedGstCamera,
        ZedGstStereoCamera,
        zed_gst_available,
        zed_stereo_gst_available,
    )
    from .shm_frames import RawFrameWriter

    serial = int(spec["serial"])
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    if stereo and not zed_stereo_gst_available():
        return None
    if not stereo and not zed_gst_available():
        return None
    if resolution not in _RESOLUTION_DIMS:
        return None
    width, height = _RESOLUTION_DIMS[resolution]
    # The dataset (raw) frames are what cross to the recorder process and feed the
    # NVENC encoder; downscale them on the relay's VIC when the caller asks for a
    # smaller dataset resolution (clamped here, so it never upscales). The encoded
    # headset stream keeps the full capture resolution. The shm blocks/sockets,
    # raw_meta, and gst raw caps must all agree on these dims.
    raw_w, raw_h = width, height
    ds_name = spec.get("dataset_resolution")
    if ds_name in _RESOLUTION_DIMS:
        dw, dh = _RESOLUTION_DIMS[ds_name]
        if dw < width or dh < height:
            raw_w, raw_h = dw, dh
    raw_dims = (raw_w, raw_h)
    use_shm = socket_dir is not None

    for fps in (int(spec.get("fps", 60)), 30):
        writers: list = []
        try:
            if stereo and use_shm:
                plan = _eye_plan(name, spec)
                gst_eyes = "both" if len(plan) == 2 else plan[0][0]
                socks = {
                    side: os.path.join(socket_dir, f"{src}.sock") for side, src in plan
                }
                eye_kwargs: dict = {}
                if "left" in socks:
                    eye_kwargs["left_raw_socket_path"] = socks["left"]
                if "right" in socks:
                    eye_kwargs["right_raw_socket_path"] = socks["right"]
                cam: object = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    raw_dims=raw_dims,
                    eyes=gst_eyes,
                    **eye_kwargs,
                )
                cam.connect()
                caps = _raw_caps(raw_w, raw_h, fps)
                lat = cam.raw_latency_s
                sources = {}
                raw_meta = {}
                for side, src in plan:
                    sources[src] = cam.left_view if side == "left" else cam.right_view
                    raw_meta[src] = _gstshm_meta(
                        socks[side], caps, raw_w, raw_h, fps, lat
                    )
                return cam, sources, [], raw_meta
            if stereo:
                plan = _eye_plan(name, spec)
                gst_eyes = "both" if len(plan) == 2 else plan[0][0]
                eye_writers = {
                    side: RawFrameWriter.create(raw_w, raw_h, cond) for side, _ in plan
                }
                writers = list(eye_writers.values())
                eye_kwargs = {}
                if "left" in eye_writers:
                    eye_kwargs["left_raw_sink"] = eye_writers["left"].publish
                if "right" in eye_writers:
                    eye_kwargs["right_raw_sink"] = eye_writers["right"].publish
                cam = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    raw_dims=raw_dims,
                    eyes=gst_eyes,
                    **eye_kwargs,
                )
                cam.connect()
                sources = {}
                raw_meta = {}
                for side, src in plan:
                    sources[src] = cam.left_view if side == "left" else cam.right_view
                    raw_meta[src] = _pyshm_meta(
                        eye_writers[side].name, raw_w, raw_h, fps
                    )
                return cam, sources, writers, raw_meta
            if use_shm:
                sock = os.path.join(socket_dir, f"{name}.sock")
                cam = ZedGstCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    raw_socket_path=sock,
                    raw_dims=raw_dims,
                )
                cam.connect()
                caps = _raw_caps(raw_w, raw_h, fps)
                meta = {
                    name: _gstshm_meta(sock, caps, raw_w, raw_h, fps, cam.raw_latency_s)
                }
                return cam, {name: cam}, [], meta
            writer = RawFrameWriter.create(raw_w, raw_h, cond)
            writers = [writer]
            cam = ZedGstCamera(
                serial,
                resolution,
                fps,
                want_encoded=True,
                raw_sink=writer.publish,
                raw_dims=raw_dims,
            )
            cam.connect()
            return (
                cam,
                {name: cam},
                writers,
                {name: _pyshm_meta(writer.name, raw_w, raw_h, fps)},
            )
        except Exception as exc:  # noqa: BLE001 - try lower fps, then give up
            for w in writers:
                w.close()
            _logger.info("video relay: gst-raw %s @ %s fps failed (%s)", name, fps, exc)
    return None


def _open_gst_camera(name: str, spec: dict) -> tuple[object, dict[str, object]] | None:
    """Open one camera via the GPU-resident gst pipeline (encoded only).

    Returns ``(owned_camera, {track_name: source})`` where each source exposes
    ``subscribe()`` so the WebRTC manager forwards its pre-encoded H.264 AUs
    directly. The relay never needs raw frames, so the raw branch is omitted
    (lowest cost). Returns ``None`` when the gst stack or camera is unavailable
    so the caller can fall back to the SDK path.
    """
    from .gst_zed import (
        ZedGstCamera,
        ZedGstStereoCamera,
        zed_gst_available,
        zed_stereo_gst_available,
    )

    serial = int(spec["serial"])
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    if stereo and not zed_stereo_gst_available():
        return None
    if not stereo and not zed_gst_available():
        return None

    for fps in (int(spec.get("fps", 60)), 30):
        try:
            if stereo:
                plan = _eye_plan(name, spec)
                gst_eyes = "both" if len(plan) == 2 else plan[0][0]
                cam: object = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    want_raw=False,
                    eyes=gst_eyes,
                )
                cam.connect()
                sources = {
                    src: (cam.left_view if side == "left" else cam.right_view)
                    for side, src in plan
                }
                return cam, sources
            cam = ZedGstCamera(
                serial, resolution, fps, want_encoded=True, want_raw=False
            )
            cam.connect()
            return cam, {name: cam}
        except Exception as exc:  # noqa: BLE001 - try lower fps, then SDK
            _logger.info("video relay: gst %s @ %s fps failed (%s)", name, fps, exc)
    return None


def _relay_main(
    conn: multiprocessing.connection.Connection,
    cameras: dict[str, dict],
    log_level: int,
    want_raw: bool = False,
    raw_cond: object = None,
) -> None:
    """Relay subprocess entry point: open cameras, serve signaling requests.

    When ``want_raw`` is set (data collection), each camera is opened on the gst
    pipeline with a raw branch whose frames are published to shared memory for
    the control process; ``raw_cond`` is the shared
    :class:`multiprocessing.Condition` guarding those blocks. Cameras that can't
    provide raw via gst still stream to the headset (encoded only) but are
    omitted from ``raw_meta`` so the parent can fall back to the in-process
    camera path.
    """
    logging.basicConfig(level=log_level)

    # Disable the cyclic garbage collector in the relay. aiortc sends WebRTC media
    # on this process's asyncio loop, and a stop-the-world gen2 GC pause freezes
    # that loop for ~100ms — which is exactly what stalls the send during
    # recording (the raw branch's per-frame allocations push GC over its
    # threshold), making the headset feed laggy + grainy. The per-frame objects
    # (numpy frame views, encoded byte buffers) are all refcounted, so they free
    # promptly without the collector; only reference cycles would linger, which is
    # acceptable for a session-scoped subprocess.
    import gc

    gc.disable()

    # Pin the relay to its own cores (away from both the control loop and the
    # dataset recorder/encoders). Its WebRTC send is latency-sensitive; sharing
    # cores with the dataset throughput during recording starves the send and
    # makes the headset feed laggy + grainy. Isolated, it gets prompt CPU like in
    # teleop. Where affinity isn't available, fall back to a positive nice so it
    # at least doesn't preempt the control loop.
    from ..utils import affinity

    if not affinity.pin_relay():
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass

    from .video import WebRTCManager

    # Keep the camera objects alive for the relay's lifetime; ``sources`` maps
    # the per-track names the headset sees to a video source per camera/eye.
    # Prefer the GPU-resident gst pipeline; fall back to the SDK grab — a bare
    # ZedCamera/eye, which WebRTCManager adapts to a frame-driven NVENC source.
    owned: list[object] = []
    sources: dict[str, object] = {}
    writers: list[object] = []
    raw_meta: dict[str, dict] = {}
    # Prefer the gst-native shmsink transport for raw frames: it exports each
    # frame to the recorder in C, so the relay does zero Python per raw frame and
    # the WebRTC send keeps the GIL it needs (the recording-feed fix). Falls back
    # to the in-relay Python copy (RawFrameWriter) when gst's shm plugin is
    # absent. A per-relay-PID dir holds one socket per source; removed on exit.
    socket_dir: str | None = None
    if want_raw:
        from .gst_zed import _element_available

        if _element_available("shmsink") and _element_available("shmsrc"):
            import tempfile

            socket_dir = tempfile.mkdtemp(prefix="axol-raw-")
    for name, spec in cameras.items():
        if want_raw:
            raw = _open_gst_camera_raw(name, spec, raw_cond, socket_dir)
            if raw is not None:
                cam, gst_sources, cam_writers, cam_meta = raw
                owned.append(cam)
                sources.update(gst_sources)
                writers.extend(cam_writers)
                raw_meta.update(cam_meta)
                continue
            # No gst raw path for this camera; still stream it (encoded only).
        gst = _open_gst_camera(name, spec)
        if gst is not None:
            cam, gst_sources = gst
            owned.append(cam)
            sources.update(gst_sources)
            continue
        cam = _open_sdk_camera(name, spec)
        if cam is None:
            continue
        owned.append(cam)
        if spec.get("stereo"):
            # One grab, per-eye views. The head camera maps both eyes
            # (overhead_left / overhead_right); a wrist stereo camera maps only
            # its left eye under the plain name (see _eye_plan).
            for side, src in _eye_plan(name, spec):
                sources[src] = cam.left_view if side == "left" else cam.right_view
        else:
            sources[name] = cam

    manager = WebRTCManager(sources) if sources else None
    conn.send(("ready", sorted(sources), raw_meta))
    if manager is None:
        _logger.warning("video relay opened no cameras; nothing to stream")

    async def _loop_lag_monitor() -> None:
        """Log the relay event-loop's worst scheduling lag each second.

        This is the asyncio analog of the control loop's maxgap: it measures how
        late a fixed-interval wakeup actually fires on the relay's loop — the same
        loop aiortc sends WebRTC media on. A large lag during recording means the
        send is being starved *inside the relay process* (by the dataset raw
        branch), which core isolation can't fix; a small lag means the send is
        prompt and a degraded feed is downstream (network).
        """
        loop = asyncio.get_running_loop()
        period = 0.05
        worst = 0.0
        last = loop.time()
        while True:
            t0 = loop.time()
            await asyncio.sleep(period)
            worst = max(worst, loop.time() - t0 - period)
            now = loop.time()
            if now - last >= 1.0:
                _logger.info("relay event-loop maxlag=%.1fms", 1e3 * worst)
                worst = 0.0
                last = now

    async def serve() -> None:
        loop = asyncio.get_running_loop()
        # The WebRTC send-health logger (packets sent / lost / RTT) and event-loop
        # lag monitor were the recording-jitter instrumentation; only run them at
        # DEBUG so the default output stays quiet (they're dedicated diagnostic
        # tasks — no point spinning them when nothing logs).
        if _logger.isEnabledFor(logging.DEBUG):
            if manager is not None:
                loop.create_task(manager.log_stats_loop())
            loop.create_task(_loop_lag_monitor())
        while True:
            try:
                msg = await loop.run_in_executor(None, conn.recv)
            except (EOFError, OSError):  # parent exited — shut down
                break
            if msg is None:
                break
            kind = msg[0]
            try:
                if kind == "offer":
                    _, client_id = msg
                    if manager is None:
                        conn.send(("offer_err", client_id, "no cameras"))
                        continue
                    try:
                        sdp, tracks = await manager.create_offer(client_id)
                        conn.send(("offer_ok", client_id, sdp, tracks))
                    except Exception as exc:  # noqa: BLE001 - report upstream
                        _logger.error("video relay: offer failed: %s", exc)
                        conn.send(("offer_err", client_id, str(exc)))
                elif kind == "answer" and manager is not None:
                    _, client_id, sdp = msg
                    await manager.set_answer(client_id, sdp)
                elif kind == "close" and manager is not None:
                    _, client_id = msg
                    await manager.close(client_id)
                elif kind == "close_all" and manager is not None:
                    await manager.close_all()
                elif kind == "raw_enable":
                    _, enabled = msg
                    for cam in owned:
                        if hasattr(cam, "set_raw_enabled"):
                            cam.set_raw_enabled(enabled)
            except Exception as exc:  # noqa: BLE001 - keep serving
                _logger.error("video relay: error handling %s: %s", kind, exc)
        if manager is not None:
            await manager.close_all()

    # Dedicate one relay core to the WebRTC send: aiortc does the whole send on
    # this (the main) thread, and it's CPU-bound. Done here — after the cameras'
    # gst pull threads were created (so they keep the full relay group and land on
    # the other relay core) and right before the event loop runs on this thread.
    affinity.pin_relay_send_thread()

    try:
        asyncio.run(serve())
    finally:
        if manager is not None:
            manager.shutdown()
        for cam in owned:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        for writer in writers:
            try:
                writer.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        if socket_dir is not None:
            import shutil

            shutil.rmtree(socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class _RawCameraStub:
    """Dims-only stand-in for a relay raw source on the gst-shm transport.

    On the shmsink path the control process never reads raw frames — the recorder
    owns the ``shmsrc`` consumer — so the control process needs only the camera's
    width/height/fps to size the dataset observation features and to satisfy the
    robot's camera lifecycle (``connect``/``disconnect`` are no-ops on a proxy).
    Reads raise: nothing in the control process should pull frames from these.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, warmup: bool = True) -> None:
        pass

    def disconnect(self) -> None:
        pass

    close = disconnect

    def _no_read(self, *args: object, **kwargs: object):
        raise RuntimeError(
            "raw frames are read by the recorder subprocess on the gst-shm "
            "transport, not the control process."
        )

    read = read_at_or_after = read_latest = read_latest_with_ts = _no_read


class VideoRelayProcess:
    """Parent-side handle for the video relay subprocess.

    Implements the ``WebRTCManager`` interface so it can be handed to
    ``VRServer.set_video_manager``. Signaling requests are serialized over
    one pipe (they are rare — a handful per headset connection), each run
    in an executor so the caller's event loop never blocks.
    """

    def __init__(self, cameras: dict[str, dict], want_raw: bool = False) -> None:
        """Spawn the relay and block until its cameras are streaming.

        Args:
            cameras: Per-source spec: ``{name: {"serial": int,
                "resolution": str, "fps": int, "stereo": bool}}``.
            want_raw: Also publish each camera's raw RGB frames to shared memory
                for the control process (data collection). Successfully exported
                sources appear in :attr:`raw_cameras` as
                :class:`~almond_axol.vr.shm_frames.RawFrameReader` proxies.
        """
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        # One Condition guards every source's shared-memory metadata; it must be
        # created here and passed at spawn so parent (readers) and child
        # (writers) share the same underlying primitive.
        self._raw_cond = ctx.Condition() if want_raw else None
        self._proc = ctx.Process(
            target=_relay_main,
            args=(
                child_conn,
                cameras,
                logging.getLogger().level or logging.INFO,
                want_raw,
                self._raw_cond,
            ),
            daemon=True,
            name="video-relay",
        )
        self._proc.start()
        child_conn.close()
        self._lock = threading.Lock()

        self.sources: list[str] = []
        self.raw_cameras: dict[str, object] = {}
        # ``{source: meta}`` describing each raw source's transport (gstshm socket
        # + caps, or pyshm block name) and dims — exposed (with :attr:`raw_cond`)
        # so the recorder subprocess can attach its own consumer per source.
        self.raw_meta: dict[str, dict] = {}
        if self._conn.poll(_READY_TIMEOUT_S):
            msg = self._conn.recv()
            if isinstance(msg, tuple) and msg[0] == "ready":
                self.sources = list(msg[1])
                raw_meta = msg[2] if len(msg) > 2 else {}
                self.raw_meta = dict(raw_meta)
                self._attach_raw_readers(raw_meta)
        if not self.sources:
            _logger.warning("video relay started no camera streams")

    @property
    def raw_cond(self) -> object:
        """The shared ``multiprocessing.Condition`` guarding the raw shm blocks.

        Must be passed at spawn time to any other process that attaches a
        :class:`~almond_axol.vr.shm_frames.RawFrameReader` to these blocks.
        """
        return self._raw_cond

    def _attach_raw_readers(self, raw_meta: dict[str, dict]) -> None:
        """Expose each relay raw source to the control process.

        On the **gstshm** transport the control process never reads frames (the
        recorder owns the shmsrc consumer), so attach a dims-only
        :class:`_RawCameraStub`. On the **pyshm** fallback, attach a
        :class:`RawFrameReader` over the shared-memory block.
        """
        if not raw_meta:
            return
        from .shm_frames import RawFrameReader

        for source, meta in raw_meta.items():
            try:
                if meta["transport"] == "gstshm":
                    self.raw_cameras[source] = _RawCameraStub(
                        meta["width"], meta["height"], meta["fps"]
                    )
                elif self._raw_cond is not None:
                    self.raw_cameras[source] = RawFrameReader(
                        meta["shm_name"],
                        meta["width"],
                        meta["height"],
                        meta["fps"],
                        self._raw_cond,
                    )
            except Exception as exc:  # noqa: BLE001 - skip a source we can't map
                _logger.warning(
                    "video relay: could not attach raw frames for %s: %s",
                    source,
                    exc,
                )

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)

    def _request_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        with self._lock:
            self._conn.send(("offer", client_id))
            # Replies are strictly ordered on the pipe; the only inbound
            # messages are responses to "offer" requests.
            if not self._conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError("video relay did not answer the offer request")
            msg = self._conn.recv()
        if msg[0] == "offer_ok" and msg[1] == client_id:
            return msg[2], msg[3]
        raise RuntimeError(f"video relay offer failed: {msg}")

    def _send(self, msg: object) -> None:
        with self._lock:
            try:
                self._conn.send(msg)
            except (OSError, ValueError):
                pass  # relay already gone

    # -- WebRTCManager interface --------------------------------------------

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a peer connection in the relay; returns ``(sdp, tracks)``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_offer, client_id)

    async def set_answer(self, client_id: int, sdp: str) -> None:
        """Forward the headset's SDP answer to the relay."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("answer", client_id, sdp)
        )

    async def close(self, client_id: int) -> None:
        """Close the relay's peer connection for ``client_id``."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("close", client_id)
        )

    async def close_all(self) -> None:
        """Close every peer connection in the relay."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("close_all",)
        )

    def set_raw_enabled(self, enabled: bool) -> None:
        """Open/close the raw dataset branch in the relay (recording only).

        The raw RGBA branch (VIC convert + shared-memory copy for every camera)
        is the bulk of the relay's CPU. ``collect-data`` keeps it closed during
        the pre-record teleop phase — where nothing reads raw frames — so the
        control loop keeps the spare cores it needs, then opens it while an
        episode is actually recording. No-op if the relay has no raw sources.
        """
        if not self.raw_cameras:
            return
        self._send(("raw_enable", bool(enabled)))

    # -- Lifecycle ------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the relay subprocess (cameras and peer connections included)."""
        for reader in self.raw_cameras.values():
            try:
                reader.disconnect()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self.raw_cameras = {}
        try:
            with self._lock:
                self._conn.send(None)
        except (OSError, ValueError):
            pass
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
