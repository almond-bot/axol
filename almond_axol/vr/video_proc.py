"""Out-of-process WebRTC video relay (Jetson gst-native cameras).

Sending video from inside the teleop process measurably starves the
control loops: aiortc pushes thousands of RTP packets per second through
Python (packetize, SRTP, sendto), and that GIL traffic stretches the IK
round-trip from ~9 ms to ~23 ms the moment a headset connects. The same
isolation pattern used for the IK solver applies here — video runs in a
dedicated subprocess and the control process never touches it.

The subprocess owns the gst-native camera pipelines (``gst_zed``) and a
``WebRTCManager``; WebRTC media flows over its own UDP sockets, so the
only traffic crossing the process boundary is SDP signaling (a few
messages per headset connection) over a ``multiprocessing`` pipe.

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


def _relay_main(
    conn: multiprocessing.connection.Connection,
    cameras: dict[str, dict],
    log_level: int,
) -> None:
    """Relay subprocess entry point: open cameras, serve signaling requests."""
    logging.basicConfig(level=log_level)

    from .gst_zed import ZedXOneGstStream, ZedXStereoGstStream
    from .video import WebRTCManager

    # Keep the stream objects alive for the relay's lifetime; ``sources``
    # maps the per-track names the headset sees to the eye/camera channels.
    owned: list[ZedXOneGstStream | ZedXStereoGstStream] = []
    sources: dict[str, object] = {}
    for name, spec in cameras.items():
        resolution = spec.get("resolution", "HD1200")
        fps = spec.get("fps", 60)
        try:
            if spec.get("stereo"):
                # One camera, two encoded eyes -> overhead_left / overhead_right.
                stereo = ZedXStereoGstStream(spec["serial"], resolution, fps)
                if not stereo.wait_ready():
                    _logger.warning(
                        "video relay: %s stereo produced no frames; skipping", name
                    )
                    stereo.close()
                    continue
                owned.append(stereo)
                sources[f"{name}_left"] = stereo.left_view
                sources[f"{name}_right"] = stereo.right_view
            else:
                mono = ZedXOneGstStream(spec["serial"], resolution, fps)
                if not mono.wait_ready():
                    _logger.warning(
                        "video relay: %s produced no frames; skipping", name
                    )
                    mono.close()
                    continue
                owned.append(mono)
                sources[name] = mono
        except Exception as exc:  # noqa: BLE001 - bad spec / spawn failure
            _logger.warning("video relay: %s failed to start (%s)", name, exc)
            continue

    manager = WebRTCManager(sources)
    conn.send(("ready", sorted(sources)))

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
                    try:
                        sdp, tracks = await manager.create_offer(client_id)
                        conn.send(("offer_ok", client_id, sdp, tracks))
                    except Exception as exc:  # noqa: BLE001 - report upstream
                        _logger.error("video relay: offer failed: %s", exc)
                        conn.send(("offer_err", client_id, str(exc)))
                elif kind == "answer":
                    _, client_id, sdp = msg
                    await manager.set_answer(client_id, sdp)
                elif kind == "close":
                    _, client_id = msg
                    await manager.close(client_id)
                elif kind == "close_all":
                    await manager.close_all()
            except Exception as exc:  # noqa: BLE001 - keep serving
                _logger.error("video relay: error handling %s: %s", kind, exc)
        await manager.close_all()

    try:
        asyncio.run(serve())
    finally:
        for stream in owned:
            stream.close()


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
