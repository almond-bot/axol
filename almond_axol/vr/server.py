"""
VR WebSocket server for the Axol arm.

VRServer accepts secure WebSocket (WSS) connections from a VR headset and
surfaces the latest VRFrame to the caller. IK and motor control are handled
separately — this class is purely the network layer.

Communication is bidirectional:
  - headset → server: VRFrame JSON every XR frame
  - server → headset: arbitrary JSON (e.g. state feedback via broadcast_text)

Typical usage::

    async with VRServer() as vr:
        while True:
            frame = vr.get_frame()
            if frame is not None:
                print(frame.l_ee, frame.r_ee, frame.l_elbow, frame.r_elbow)
            await asyncio.sleep(0.01)

Or with an on_frame callback::

    def handle(frame: VRFrame) -> None:
        logging.getLogger(__name__).debug("frame: %s", frame)

    async with VRServer(on_frame=handle) as vr:
        await asyncio.sleep(float("inf"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..utils.certs import ACCEPT_PAGE_HTML, CERTFILE, KEYFILE, create_self_signed_cert
from ..utils.ports import open_listen_socket
from .config import VRServerConfig
from .control_channel import ControlChannelManager
from .ice import client_ice_servers
from .interp import PoseInterpolator
from .models import VRFrame

if TYPE_CHECKING:
    from .video import WebRTCManager

_logger = logging.getLogger(__name__)


class VRServer:
    """Secure WebSocket server that receives VRFrame data from a VR headset.

    Args:
        config:  Server configuration (port, TLS paths). Defaults to VRServerConfig().
    """

    def __init__(self, config: VRServerConfig = VRServerConfig()) -> None:
        """Configure the VR WebSocket server.

        The server is not started until :meth:`enable` (or ``async with``) is
        called.  A self-signed TLS certificate is auto-generated in
        ``~/.almond/vr/certs/`` on first use if no cert paths are provided.

        Args:
            config: Port, TLS certificate, and private-key paths.
        """
        self._port = config.port
        self._on_frame: Callable[[VRFrame], None] | None = None
        self._certfile = config.certfile or CERTFILE
        self._keyfile = config.keyfile or KEYFILE

        self._latest_frame: VRFrame | None = None
        # Adaptive playout buffer: reconstructs a smooth pose stream from
        # batched/jittered network arrivals. Consumers that want smoothing read
        # via get_render_frame(); get_frame() still returns the raw latest.
        self._interp = PoseInterpolator(
            enabled=config.interp_enabled,
            min_delay_s=config.interp_min_delay_s,
            max_delay_s=config.interp_max_delay_s,
        )
        self._client_count: int = 0
        self._active_clients: set[WebSocket] = set()
        self._server_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._listen_socket: socket.socket | None = None
        self._webrtc: WebRTCManager | Any | None = None
        # Dedicated pose data channel (low-latency control transport). Always
        # available — independent of whether any cameras are streaming.
        self._control = ControlChannelManager(self._ingest_pose_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> VRFrame | None:
        """Return the most recent frame received, or None if none yet."""
        return self._latest_frame

    def get_render_frame(self) -> VRFrame | None:
        """Return the smoothed playout frame for the current instant.

        Renders the pose from the adaptive interpolation buffer (motion is held
        slightly in the past and interpolated; control state is the latest
        received). Falls back to the raw latest frame when interpolation is
        disabled or there isn't enough history yet. The returned object is
        identity-stable so the IK loop can skip redundant solves while idle.
        """
        return self._interp.sample()

    def set_on_frame(self, callback: Callable[[VRFrame], None] | None) -> None:
        """Replace the on_frame callback. Safe to call after construction."""
        self._on_frame = callback

    def set_video_sources(self, sources: dict[str, Any] | None) -> None:
        """Register per-camera video sources to stream to the headset.

        Each value is a connected ``ZedCamera`` / stereo eye (registered
        directly — the relay adapts it to BGRA frames), a pre-encoded
        ``gst_zed`` camera, or any raw-frame source exposing ``width`` /
        ``height`` / ``fps`` + ``wait_next``; the manager picks the right
        WebRTC track per source (see :mod:`almond_axol.vr.video`). The headset
        negotiates a WebRTC connection over the existing ``/ws`` channel and
        receives one video track per source, encoded on the Jetson's hardware
        NVENC and shipped by aiortc.

        This is the in-process fallback; teleop normally runs the relay in a
        dedicated subprocess via :meth:`set_video_manager`. Pass ``None`` or an
        empty dict to disable video. Requires ``aiortc`` (a normal dependency);
        hardware NVENC additionally needs the system GStreamer stack from
        ``axol gst.install``. If aiortc is unavailable this logs a warning and
        leaves video disabled. Safe to call before or after :meth:`enable`.
        """
        if not sources:
            self._webrtc = None
            return
        try:
            from .video import WebRTCManager, webrtc_available

            if not webrtc_available():
                raise RuntimeError("aiortc unavailable")
            self._webrtc = WebRTCManager(sources)
        except Exception as exc:  # noqa: BLE001 - aiortc missing
            _logger.warning(
                "wrist video requested but the WebRTC stack (aiortc) is "
                "unavailable (%s); install the project dependencies (and "
                "`axol gst.install` for hardware NVENC). Continuing without "
                "wrist video.",
                exc,
            )
            self._webrtc = None
            return
        _logger.info("wrist video enabled for: %s", ", ".join(sources))

    def set_video_manager(self, manager: Any | None) -> None:
        """Register a pre-built WebRTC manager (e.g. an out-of-process relay).

        ``manager`` must implement the ``WebRTCManager`` signaling interface
        (``create_offer`` / ``set_answer`` / ``close`` / ``close_all``).
        Used by teleop to keep all video encoding and RTP traffic in a
        separate process (``almond_axol.vr.video_proc``) so it cannot
        contend with the control loops. Pass ``None`` to disable video.
        """
        self._webrtc = manager
        if manager is not None:
            _logger.info("wrist video enabled (external manager)")

    @property
    def connected(self) -> bool:
        """True if at least one VR client is currently connected."""
        return self._client_count > 0

    async def broadcast_text(self, text: str) -> None:
        """Send a text message to all currently connected VR clients."""
        for ws in list(self._active_clients):
            try:
                await ws.send_text(text)
            except Exception as exc:
                _logger.warning("Failed to send feedback to client: %s", exc)

    async def enable(self) -> None:
        """Start the WSS server in the background.

        The listening socket is bound *here* (reclaiming the port from a stale
        listener if needed) so a bind failure raises synchronously instead of
        being swallowed inside uvicorn's background task. uvicorn then adopts
        the already-bound socket via ``serve(sockets=...)``.
        """
        if self._server_task is not None:
            return

        if not os.path.isfile(self._certfile) or not os.path.isfile(self._keyfile):
            _logger.info("creating self-signed certificate")
            create_self_signed_cert(self._certfile, self._keyfile)

        sock = await asyncio.to_thread(open_listen_socket, "0.0.0.0", self._port)
        self._listen_socket = sock

        app = self._build_app()
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self._port,
            log_level="info",
            ssl_certfile=self._certfile,
            ssl_keyfile=self._keyfile,
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(
            self._uvicorn_server.serve(sockets=[sock])
        )
        _logger.info("listening on wss://0.0.0.0:%d/ws", self._port)

    async def disable(self) -> None:
        """Gracefully shut down the WSS server."""
        if self._webrtc is not None:
            await self._webrtc.close_all()
        await self._control.close_all()

        if self._uvicorn_server is not None:
            try:
                await self._uvicorn_server.shutdown()
            except Exception:
                pass
            self._uvicorn_server = None

        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            self._server_task = None

        # uvicorn closes the adopted socket on a clean shutdown, but close it
        # ourselves too so a cancelled/timed-out shutdown still frees the port
        # for the next ``enable()`` instead of leaking the bind.
        if self._listen_socket is not None:
            try:
                self._listen_socket.close()
            except OSError:
                pass
            self._listen_socket = None

        self._client_count = 0
        self._active_clients.clear()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> VRServer:
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_message(
        self, websocket: WebSocket, client_id: int, data: str
    ) -> None:
        """Dispatch one inbound text message.

        Signaling messages carry a ``type`` field; pose frames do not.
        """
        try:
            obj = json.loads(data)
        except Exception as exc:
            _logger.warning("invalid json: %s", exc)
            return

        if isinstance(obj, dict) and "type" in obj:
            await self._handle_signaling(websocket, client_id, obj)
            return

        self._ingest_frame_obj(obj)

    def _ingest_frame_obj(self, obj: Any) -> None:
        """Validate a decoded pose object and publish it to the consumer."""
        try:
            frame = VRFrame.model_validate(obj)
        except Exception as exc:
            _logger.warning("invalid frame: %s", exc)
            return
        self._latest_frame = frame
        self._interp.push(frame)
        if self._on_frame is not None:
            self._on_frame(frame)

    def _ingest_pose_text(self, data: str) -> None:
        """Ingest a pose frame from the control data channel (text message).

        Mirrors the WebSocket pose path; signaling never arrives here, so a
        message carrying a ``type`` field is ignored rather than validated.
        """
        try:
            obj = json.loads(data)
        except Exception as exc:
            _logger.warning("invalid json on pose channel: %s", exc)
            return
        if isinstance(obj, dict) and "type" in obj:
            return
        self._ingest_frame_obj(obj)

    async def _handle_signaling(
        self, websocket: WebSocket, client_id: int, obj: dict[str, Any]
    ) -> None:
        """Handle a WebRTC signaling message from the headset."""
        msg_type = obj.get("type")

        # Control data channel (pose transport): negotiated independently of the
        # cameras, so it's handled before the video-availability check below and
        # works even when no video sources are registered.
        if msg_type == "control-request":
            sdp = await self._control.create_offer(client_id)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "control-offer",
                        "sdp": sdp,
                        "iceServers": client_ice_servers(),
                    }
                )
            )
            return
        if msg_type == "control-answer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                await self._control.set_answer(client_id, sdp)
            return

        if self._webrtc is None:
            if msg_type == "webrtc-request":
                await websocket.send_text(json.dumps({"type": "webrtc-unavailable"}))
            return

        if msg_type == "webrtc-request":
            try:
                sdp, tracks = await self._webrtc.create_offer(client_id)
            except Exception as exc:
                _logger.error("failed to create webrtc offer: %s", exc)
                await websocket.send_text(json.dumps({"type": "webrtc-unavailable"}))
                return
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "webrtc-offer",
                        "sdp": sdp,
                        "tracks": tracks,
                        # Same TURN/STUN servers the aiortc peer used, so the
                        # browser gathers a matching relay candidate. Empty on
                        # a LAN (no env config) — harmless to the headset.
                        "iceServers": client_ice_servers(),
                    }
                )
            )
        elif msg_type == "webrtc-answer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                try:
                    await self._webrtc.set_answer(client_id, sdp)
                except Exception as exc:
                    _logger.error("failed to apply webrtc answer: %s", exc)
        else:
            _logger.debug("ignoring unknown signaling type: %s", msg_type)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        server = self

        @app.get("/__accept")
        async def _accept() -> HTMLResponse:
            """Self-closing page the web UI opens to approve the self-signed cert."""
            return HTMLResponse(ACCEPT_PAGE_HTML)

        @app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            await websocket.accept()
            _logger.info("client connected %s", websocket.client)
            server._client_count += 1
            server._active_clients.add(websocket)
            client_id = id(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    await server._handle_message(websocket, client_id, data)
            except WebSocketDisconnect:
                _logger.info("client disconnected %s", websocket.client)
            except Exception as exc:
                _logger.error("connection error: %s", exc)
                try:
                    await websocket.close()
                except Exception:
                    pass
            finally:
                server._active_clients.discard(websocket)
                server._client_count = max(0, server._client_count - 1)
                if server._webrtc is not None:
                    await server._webrtc.close(client_id)
                await server._control.close(client_id)
                # Last operator gone: drop buffered pose state so a fresh
                # session's capture timestamps aren't blended with this one's
                # stale frames (which would drive IK with incoherent poses
                # until the playout buffer drains).
                if server._client_count == 0:
                    server._latest_frame = None
                    server._interp.reset()

        return app
