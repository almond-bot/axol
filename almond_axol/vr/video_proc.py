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


def _open_gst_camera_raw(
    name: str, spec: dict, cond: object
) -> tuple[object, dict[str, object], list, dict[str, tuple]] | None:
    """Open one camera via the gst pipeline with both encoded + raw branches.

    Like :func:`_open_gst_camera`, but additionally allocates a shared-memory
    :class:`~almond_axol.vr.shm_frames.RawFrameWriter` per source and wires it
    as the camera's raw sink, so every captured frame is published to the
    control process for the dataset. Returns
    ``(owned_camera, {track: source}, [writers], {source: (shm_name, w, h, fps)})``
    or ``None`` when the gst stack/camera is unavailable (the caller then has no
    raw path for this camera and falls back to the in-process camera pipeline).
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
    # The dataset (raw) frames are what cross to the control process and feed the
    # NVENC encoder; downscale them on the relay's VIC when the caller asks for a
    # smaller dataset resolution (clamped here, so it never upscales). The encoded
    # headset stream keeps the full capture resolution. The shared-memory blocks,
    # raw_meta, and gst raw caps must all agree on these dims.
    raw_w, raw_h = width, height
    ds_name = spec.get("dataset_resolution")
    if ds_name in _RESOLUTION_DIMS:
        dw, dh = _RESOLUTION_DIMS[ds_name]
        if dw < width or dh < height:
            raw_w, raw_h = dw, dh
    raw_dims = (raw_w, raw_h)

    for fps in (int(spec.get("fps", 60)), 30):
        writers: list = []
        try:
            if stereo:
                left = RawFrameWriter.create(raw_w, raw_h, cond)
                right = RawFrameWriter.create(raw_w, raw_h, cond)
                writers = [left, right]
                cam: object = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    left_raw_sink=left.publish,
                    right_raw_sink=right.publish,
                    raw_dims=raw_dims,
                )
                cam.connect()
                sources = {
                    f"{name}_left": cam.left_view,
                    f"{name}_right": cam.right_view,
                }
                raw_meta = {
                    f"{name}_left": (left.name, raw_w, raw_h, fps),
                    f"{name}_right": (right.name, raw_w, raw_h, fps),
                }
                return cam, sources, writers, raw_meta
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
            return cam, {name: cam}, writers, {name: (writer.name, raw_w, raw_h, fps)}
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
                cam: object = ZedGstStereoCamera(
                    serial, resolution, fps, want_encoded=True, want_raw=False
                )
                cam.connect()
                return cam, {
                    f"{name}_left": cam.left_view,
                    f"{name}_right": cam.right_view,
                }
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
    raw_meta: dict[str, tuple] = {}
    for name, spec in cameras.items():
        if want_raw:
            raw = _open_gst_camera_raw(name, spec, raw_cond)
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
            # One camera, two eyes -> overhead_left / overhead_right.
            sources[f"{name}_left"] = cam.left_view
            sources[f"{name}_right"] = cam.right_view
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
        # Background WebRTC send-health logger (packets sent / lost / RTT) so we
        # can see whether a degraded headset feed is transport loss vs encoder.
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


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


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
        # ``{source: (shm_name, width, height, fps)}`` for the raw blocks the relay
        # created — exposed (with :attr:`raw_cond`) so a separate recorder
        # subprocess can attach its own readers to the same shared memory.
        self.raw_meta: dict[str, tuple] = {}
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

    def _attach_raw_readers(self, raw_meta: dict[str, tuple]) -> None:
        """Attach a RawFrameReader proxy to each shared-memory block the relay made."""
        if not raw_meta or self._raw_cond is None:
            return
        from .shm_frames import RawFrameReader

        for source, (shm_name, width, height, fps) in raw_meta.items():
            try:
                self.raw_cameras[source] = RawFrameReader(
                    shm_name, width, height, fps, self._raw_cond
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
