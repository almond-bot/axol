"""
VR WebSocket server for the Axol arm.

VRServer accepts secure WebSocket (WSS) connections from a VR headset and
surfaces the latest VRFrame to the caller. IK and motor control are handled
separately — this class is purely the network layer.

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
import logging
import os
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .certs import create_self_signed_cert
from .models import VRFrame

_logger = logging.getLogger(__name__)

_CERTS_DIR = os.path.join(os.path.expanduser("~"), ".almond", "vr", "certs")


class VRServer:
    """Secure WebSocket server that receives VRFrame data from a VR headset.

    Args:
        port:      Port to listen on (default 8000).
        on_frame:  Optional callback invoked synchronously on each new frame.
                   Runs on the event-loop thread — keep it fast.
        certfile:  Path to TLS certificate PEM. Auto-generated if absent.
        keyfile:   Path to TLS private key PEM. Auto-generated if absent.
    """

    def __init__(
        self,
        port: int = 8000,
        on_frame: Callable[[VRFrame], None] | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
    ) -> None:
        self._port = port
        self._on_frame = on_frame
        self._certfile = certfile or os.path.join(_CERTS_DIR, "cert.pem")
        self._keyfile = keyfile or os.path.join(_CERTS_DIR, "key.pem")

        self._latest_frame: VRFrame | None = None
        self._client_count: int = 0
        self._server_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> VRFrame | None:
        """Return the most recent frame received, or None if none yet."""
        return self._latest_frame

    @property
    def connected(self) -> bool:
        """True if at least one VR client is currently connected."""
        return self._client_count > 0

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

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        server = self

        @app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            await websocket.accept()
            _logger.info("client connected %s", websocket.client)
            server._client_count += 1
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        frame = VRFrame.model_validate_json(data)
                        server._latest_frame = frame
                        if server._on_frame is not None:
                            server._on_frame(frame)
                    except Exception as exc:
                        _logger.warning("invalid frame: %s", exc)
            except WebSocketDisconnect:
                _logger.info("client disconnected %s", websocket.client)
            except Exception as exc:
                _logger.error("connection error: %s", exc)
                try:
                    await websocket.close()
                except Exception:
                    pass
            finally:
                server._client_count = max(0, server._client_count - 1)

        return app
