from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from almond_axol.serve.app import create_app


async def _raw_asgi_get(app: Any, path: str) -> tuple[int, bytes]:
    """Issue a path without an HTTP client's dot-segment normalization."""
    messages: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.sleep(10)
        raise AssertionError("response unexpectedly requested another event")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"localhost:8001")],
            "client": ("test", 1),
            "server": ("localhost", 8001),
        },
        receive,
        send,
    )
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, body


class StaticSpaSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        parent = Path(self._temporary.name)
        self.static = parent / "static"
        self.static.mkdir()
        (self.static / "index.html").write_text("SPA INDEX", encoding="utf-8")
        assets = self.static / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("safe asset", encoding="utf-8")
        self.secret = parent / "secret.txt"
        self.secret.write_text("ROOT-READABLE SECRET", encoding="utf-8")
        self.app = create_app(self.static)

    def test_normal_asset_and_spa_fallback_keep_cache_contract(self) -> None:
        with TestClient(self.app) as client:
            asset = client.get("/assets/app.js")
            route = client.get("/control")

        self.assertEqual(asset.status_code, 200)
        self.assertEqual(asset.text, "safe asset")
        self.assertEqual(
            asset.headers["cache-control"],
            "public, max-age=31536000, immutable",
        )
        self.assertEqual(route.status_code, 200)
        self.assertEqual(route.text, "SPA INDEX")
        self.assertEqual(route.headers["cache-control"], "no-cache")

    def test_percent_encoded_parent_and_separator_cannot_escape_bundle(self) -> None:
        with TestClient(self.app) as client:
            for path in ("/%2e%2e/secret.txt", "/..%2fsecret.txt"):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 404)
                    self.assertNotIn("ROOT-READABLE SECRET", response.text)
                    self.assertNotIn("SPA INDEX", response.text)

    def test_raw_asgi_parent_path_does_not_receive_spa_fallback(self) -> None:
        status, body = asyncio.run(_raw_asgi_get(self.app, "/../secret.txt"))
        self.assertEqual(status, 404)
        self.assertNotIn(b"ROOT-READABLE SECRET", body)
        self.assertNotIn(b"SPA INDEX", body)

    def test_symlink_asset_cannot_escape_bundle(self) -> None:
        link = self.static / "outside.txt"
        try:
            link.symlink_to(self.secret)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        with TestClient(self.app) as client:
            response = client.get("/outside.txt")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("ROOT-READABLE SECRET", response.text)


if __name__ == "__main__":
    unittest.main()
