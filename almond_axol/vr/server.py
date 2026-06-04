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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .certs import create_self_signed_cert
from .config import VRServerConfig
from .models import VRFrame

if TYPE_CHECKING:
    from .video import FrameSource, WebRTCManager

_logger = logging.getLogger(__name__)

_CERTS_DIR = os.path.join(os.path.expanduser("~"), ".almond", "vr", "certs")


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
        self._certfile = config.certfile or os.path.join(_CERTS_DIR, "cert.pem")
        self._keyfile = config.keyfile or os.path.join(_CERTS_DIR, "key.pem")

        self._latest_frame: VRFrame | None = None
        self._client_count: int = 0
        self._active_clients: set[WebSocket] = set()
        self._server_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._webrtc: WebRTCManager | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> VRFrame | None:
        """Return the most recent frame received, or None if none yet."""
        return self._latest_frame

    def set_on_frame(self, callback: Callable[[VRFrame], None] | None) -> None:
        """Replace the on_frame callback. Safe to call after construction."""
        self._on_frame = callback

    def set_video_sources(self, sources: dict[str, FrameSource] | None) -> None:
        """Register per-camera RGB frame sources to stream to the headset.

        Each source is a callable returning the latest RGB ``uint8`` numpy frame
        ``(H, W, 3)`` or ``None`` if no frame is available yet. The headset
        negotiates a WebRTC connection over the existing ``/ws`` channel and
        receives one video track per source.

        Pass ``None`` or an empty dict to disable video. Requires the ``video``
        extra (aiortc); if it is unavailable this logs a warning and leaves
        video disabled. Safe to call before or after :meth:`enable`.
        """
        if not sources:
            self._webrtc = None
            return
        try:
            from .video import WebRTCManager
        except ImportError as exc:
            _logger.warning(
                "wrist video requested but aiortc is unavailable (%s); install "
                "the 'video' extra. Continuing without wrist video.",
                exc,
            )
            self._webrtc = None
            return
        self._webrtc = WebRTCManager(sources)
        _logger.info("wrist video enabled for: %s", ", ".join(sources))

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
        """Start the WSS server in the background."""
        if self._server_task is not None:
            return

        if not os.path.isfile(self._certfile) or not os.path.isfile(self._keyfile):
            _logger.info("creating self-signed certificate")
            create_self_signed_cert(self._certfile, self._keyfile)

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
        self._server_task = asyncio.create_task(self._uvicorn_server.serve())
        _logger.info("listening on wss://0.0.0.0:%d/ws", self._port)

    async def disable(self) -> None:
        """Gracefully shut down the WSS server."""
        if self._webrtc is not None:
            await self._webrtc.close_all()

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

        try:
            frame = VRFrame.model_validate(obj)
            self._latest_frame = frame
            if self._on_frame is not None:
                self._on_frame(frame)
        except Exception as exc:
            _logger.warning("invalid frame: %s", exc)

    async def _handle_signaling(
        self, websocket: WebSocket, client_id: int, obj: dict[str, Any]
    ) -> None:
        """Handle a WebRTC signaling message from the headset."""
        msg_type = obj.get("type")

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
                json.dumps({"type": "webrtc-offer", "sdp": sdp, "tracks": tracks})
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

        return app
