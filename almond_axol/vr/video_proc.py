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
) -> None:
    """Relay subprocess entry point: open cameras, serve signaling requests."""
    logging.basicConfig(level=log_level)

    from ..cli.teleop import _ZedFrameSource
    from .video import WebRTCManager

    # Keep the camera objects alive for the relay's lifetime; ``sources`` maps
    # the per-track names the headset sees to a frame source per camera/eye.
    # Prefer the GPU-resident gst pipeline; fall back to SDK grab + NVENC.
    owned: list[object] = []
    sources: dict[str, object] = {}
    for name, spec in cameras.items():
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
            sources[f"{name}_left"] = _ZedFrameSource(cam.left_view)
            sources[f"{name}_right"] = _ZedFrameSource(cam.right_view)
        else:
            sources[name] = _ZedFrameSource(cam)

    manager = WebRTCManager(sources) if sources else None
    conn.send(("ready", sorted(sources)))
    if manager is None:
        _logger.warning("video relay opened no cameras; nothing to stream")

    async def serve() -> None:
        loop = asyncio.get_running_loop()
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
            except Exception as exc:  # noqa: BLE001 - keep serving
                _logger.error("video relay: error handling %s: %s", kind, exc)
        if manager is not None:
            await manager.close_all()

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

    def __init__(self, cameras: dict[str, dict]) -> None:
        """Spawn the relay and block until its cameras are streaming.

        Args:
            cameras: Per-source spec: ``{name: {"serial": int,
                "resolution": str, "fps": int}}``.
        """
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_relay_main,
            args=(child_conn, cameras, logging.getLogger().level or logging.INFO),
            daemon=True,
            name="video-relay",
        )
        self._proc.start()
        child_conn.close()
        self._lock = threading.Lock()

        self.sources: list[str] = []
        if self._conn.poll(_READY_TIMEOUT_S):
            msg = self._conn.recv()
            if isinstance(msg, tuple) and msg[0] == "ready":
                self.sources = list(msg[1])
        if not self.sources:
            _logger.warning("video relay started no camera streams")

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

    # -- Lifecycle ------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the relay subprocess (cameras and peer connections included)."""
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
