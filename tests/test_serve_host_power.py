"""Host shutdown/restart acknowledges the panel before powering off."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx

from almond_axol.serve import app as app_module
from tests.test_serve_session_reservation import _Manager, _Runner, _test_app


class _IdleRunner(_Runner):
    def is_running(self, **_kwargs: Any) -> bool:
        return self.running


async def _post(app: Any, path: str, events: list[str]) -> tuple[int, dict]:
    """POST through raw ASGI, logging when the response body is sent so a
    test can order it against the shutdown command."""
    status = 0
    body = b""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status, body
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body += message.get("body", b"")
            if not message.get("more_body", False):
                events.append("response")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    await app(scope, receive, send)
    return status, json.loads(body)


class HostPowerTest(unittest.IsolatedAsyncioTestCase):
    async def test_acknowledges_before_running_shutdown(self) -> None:
        app = _test_app(_Manager(), _IdleRunner())
        events: list[str] = []

        def run(cmd: list[str], **_kwargs: Any) -> SimpleNamespace:
            events.append(" ".join(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        for path, cmd in (
            ("/api/host/shutdown", "shutdown -h now"),
            ("/api/host/restart", "shutdown -r now"),
        ):
            events.clear()
            with (
                patch.object(app_module, "_HOST_POWER_DELAY_S", 0),
                patch.object(app_module.os, "geteuid", return_value=0),
                patch.object(app_module.subprocess, "run", side_effect=run),
            ):
                status, body = await _post(app, path, events)

            self.assertEqual(status, 200)
            self.assertEqual(body, {"ok": True})
            self.assertEqual(events, ["response", cmd])

    async def test_reservation_is_released_after_the_command(self) -> None:
        # A failed shutdown must not leave session launches wedged.
        app = _test_app(_Manager(), _IdleRunner())
        failed = SimpleNamespace(returncode=1, stdout="", stderr="nope")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            with (
                patch.object(app_module, "_HOST_POWER_DELAY_S", 0),
                patch.object(app_module.os, "geteuid", return_value=0),
                patch.object(app_module.subprocess, "run", return_value=failed) as run,
            ):
                first = await c.post("/api/host/shutdown")
                second = await c.post("/api/host/shutdown")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(run.call_count, 2)

    async def test_refuses_without_root_before_acknowledging(self) -> None:
        app = _test_app(_Manager(), _IdleRunner())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            with (
                patch.object(app_module.os, "geteuid", return_value=1000),
                patch.object(app_module, "prime_sudo", return_value=False),
                patch.object(app_module.subprocess, "run") as run,
            ):
                response = await c.post("/api/host/shutdown")

        self.assertEqual(response.status_code, 500)
        self.assertIn("root required", response.json()["error"])
        run.assert_not_called()

    async def test_refuses_while_busy(self) -> None:
        app = _test_app(_Manager(), _IdleRunner(running=True))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            with patch.object(app_module.subprocess, "run") as run:
                response = await c.post("/api/host/restart")

        self.assertEqual(response.status_code, 409)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
