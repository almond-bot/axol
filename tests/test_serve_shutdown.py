"""Shutdown bounds for ``axol serve``: streaming WebSockets and the CAN wait.

A stop must not sit until systemd's kill timeout. The streaming endpoints have
to end on the client's close frame even with nothing queued, and the shutdown
hook has to give up on an unfinished CAN discovery instead of joining it
forever.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import Any
from unittest.mock import Mock, patch

import httpx

from almond_axol.cli.can import setup
from almond_axol.serve import app as app_module
from almond_axol.serve.manager import Session
from tests.test_serve_session_reservation import (
    _Manager,
    _Robot,
    _Runner,
    _hub_state,
    _test_app,
)

_TIMEOUT_S = 5.0
# The bound the shutdown hook is held to under test, small enough that the
# assertion fails on a wait that is not actually honouring it.
_TEST_DISCOVERY_BOUND_S = 0.1


async def _drive_websocket(app: Any, path: str) -> list[dict[str, Any]]:
    """Open ``path``, disconnect with nothing published, return the sent frames.

    Drives the ASGI protocol directly: a websocket client is the only way to
    observe that the handler task ends, and every timeout here is what would
    otherwise hang a stop.
    """
    incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    incoming.put_nowait({"type": "websocket.connect"})
    sent: list[dict[str, Any]] = []
    accepted = asyncio.Event()

    async def receive() -> dict[str, Any]:
        return await incoming.get()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] in ("websocket.accept", "websocket.close"):
            accepted.set()

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
        "subprotocols": [],
    }
    handler = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(accepted.wait(), _TIMEOUT_S)
    incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(handler, _TIMEOUT_S)
    return sent


class WebSocketDisconnectTest(unittest.IsolatedAsyncioTestCase):
    async def test_telemetry_stream_ends_on_client_disconnect(self) -> None:
        app = _test_app(_Manager(), _Runner())

        sent = await _drive_websocket(app, "/api/telemetry/ws")

        self.assertEqual(sent[0]["type"], "websocket.accept")
        self.assertIn("hello", sent[1]["text"])

    async def test_log_stream_ends_on_client_disconnect(self) -> None:
        session = Session("tracker.identify", {})
        session.status = "running"
        app = _test_app(_Manager([session]), _Runner())

        sent = await _drive_websocket(app, f"/api/sessions/{session.id}/logs")

        self.assertEqual(sent[0]["type"], "websocket.accept")
        self.assertIn("status", sent[1]["text"])


class CanDiscoveryShutdownBoundTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_gives_up_on_unfinished_can_discovery(self) -> None:
        attached = _hub_state(profiles={"axol"}, candidates=(("usb-2", "UNRESOLVED"),))
        entered = threading.Event()
        release = threading.Event()

        def never_finishes() -> setup.HeadlessHubSetupResult:
            entered.set()
            release.wait(_TIMEOUT_S)
            raise RuntimeError("discovery was released only by the test")

        robot = _Robot()
        robot.shutdown = Mock()  # type: ignore[method-assign]
        app = _test_app(_Manager(), _Runner(), robot)
        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=attached),
            patch.object(
                setup, "setup_detected_hubs", Mock(side_effect=never_finishes)
            ),
            patch.object(
                app_module,
                "_CAN_DISCOVERY_SHUTDOWN_TIMEOUT_SECONDS",
                _TEST_DISCOVERY_BOUND_S,
            ),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                discover = asyncio.create_task(client.post("/api/can/discover"))
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(entered.wait, _TIMEOUT_S), _TIMEOUT_S
                    )
                    started = time.monotonic()
                    await asyncio.wait_for(app.router.on_shutdown[-1](), _TIMEOUT_S)
                    elapsed = time.monotonic() - started
                finally:
                    release.set()
                    await asyncio.wait_for(discover, _TIMEOUT_S)

        self.assertLess(elapsed, _TEST_DISCOVERY_BOUND_S * 10)
        robot.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
