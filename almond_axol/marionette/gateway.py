"""The connection to a Marionette gateway, as a console.

The bridge says hello as a console that wants no camera frames, receives
every phone's config, pose and input, and sends two things back:

- ``status`` messages addressed to one phone (robot feedback for its
  screen; the gateway relays them to that phone and its desktop mirrors);
- ``camera_frame`` messages, JPEG data URLs for the streams the phone
  screen declares. Frames are coalesced per stream, so a slow gateway gets
  the newest frame instead of a backlog.

It reconnects with backoff forever; the session sees the link state through
:attr:`GatewayClient.connected` and treats a lost gateway as every phone
letting go.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

_logger = logging.getLogger(__name__)

DEFAULT_PORT = 8787
PROTOCOL_VERSION = 1
# Poses arrive at the ARKit frame rate; anything this large is not ours.
_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_ADDRESS = re.compile(
    r"^(?:(wss?|https?)://)?(\[[0-9a-f:.]+\]|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)"
    r"(?::(\d{1,5}))?(/\S*)?$",
    re.IGNORECASE,
)


def gateway_url(address: str) -> str:
    """The gateway WebSocket URL for an address as an operator types it.

    Accepts a host or IP, ``host:port``, or a ``ws://`` / ``wss://`` /
    ``http(s)://`` URL, like the Marionette app. The port defaults to 8787
    and the path to ``/ws``.
    """
    match = _ADDRESS.match(address.strip())
    if not match:
        raise ValueError(
            f"{address!r} is not a gateway address. Type an IP such as "
            "192.168.1.20, or host:port."
        )
    scheme, host, port, path = match.groups()
    port_number = int(port) if port else DEFAULT_PORT
    if not 1 <= port_number <= 65535:
        raise ValueError(f"{address!r} has an invalid port.")
    secure = (scheme or "ws").lower() in ("wss", "https")
    path = "/ws" if not path or path == "/" else path
    return f"{'wss' if secure else 'ws'}://{host.lower()}:{port_number}{path}"


class GatewayClient:
    """A reconnecting console connection to one gateway."""

    def __init__(
        self,
        address: str,
        on_message: Callable[[dict[str, Any]], None],
        on_lost: Callable[[], None],
        *,
        session_id: str = "axol",
    ) -> None:
        self.url = gateway_url(address)
        self._on_message = on_message
        self._on_lost = on_lost
        self._session_id = session_id
        self.connected = False
        self.error: str | None = None
        self._outbox: dict[str, str] = {}  # key -> latest message, coalesced
        self._wake = asyncio.Event()

    # -- sending (event-loop thread) -----------------------------------------

    def send_status(
        self,
        phone_id: str | None,
        control_id: str,
        values: dict[str, str | float | bool],
        message: str = "",
    ) -> None:
        """Queue a status for one phone (or every phone when None)."""
        payload: dict[str, Any] = {
            "type": "status",
            "version": PROTOCOL_VERSION,
            "sessionId": self._session_id,
            "controlId": control_id,
            "message": message,
            "values": values,
        }
        if phone_id is not None:
            payload["phoneId"] = phone_id
        # Feedback replaces the previous feedback for the same phone; a
        # notice is its own message and never coalesced away.
        key = f"status:{phone_id}:{control_id}"
        if control_id != "axol.feedback":
            key += f":{time.monotonic_ns()}"
        self._queue(key, json.dumps(payload))

    def send_frame(self, stream: str, jpeg: bytes, captured_ms: float) -> None:
        """Queue a camera frame; a newer frame of the stream replaces it."""
        payload = {
            "type": "camera_frame",
            "version": PROTOCOL_VERSION,
            "sessionId": self._session_id,
            "cameraId": stream,
            "dataUrl": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            "capturedAt": captured_ms,
        }
        self._queue(f"frame:{stream}", json.dumps(payload))

    def _queue(self, key: str, data: str) -> None:
        if not self.connected:
            return
        self._outbox[key] = data
        self._wake.set()

    # -- connection ----------------------------------------------------------

    async def run(self) -> None:
        """Connect, serve, and reconnect with backoff until cancelled."""
        from websockets.asyncio.client import connect

        delay = 0.5
        while True:
            try:
                async with connect(
                    self.url,
                    max_size=_MAX_MESSAGE_BYTES,
                    open_timeout=5,
                    ping_interval=5,
                    ping_timeout=5,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "version": PROTOCOL_VERSION,
                                "role": "console",
                                "cameras": False,
                            }
                        )
                    )
                    self.connected = True
                    self.error = None
                    delay = 0.5
                    _logger.info("marionette: connected to %s", self.url)
                    await self._serve(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported, then retried
                message = f"Can't reach the Marionette gateway at {self.url}"
                if self.error != message:
                    _logger.warning("marionette: %s (%s)", message, exc)
                self.error = message
            finally:
                if self.connected:
                    _logger.warning("marionette: lost the gateway")
                self.connected = False
                self._outbox.clear()
                self._on_lost()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def _serve(self, ws: Any) -> None:
        async def receive() -> None:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    try:
                        self._on_message(message)
                    except Exception:  # noqa: BLE001 - one bad message, not the link
                        _logger.warning("marionette: ignored a message", exc_info=True)

        async def send() -> None:
            while True:
                await self._wake.wait()
                self._wake.clear()
                while self._outbox:
                    # Status first: feedback is small and time-critical.
                    key = next(
                        (k for k in self._outbox if k.startswith("status:")),
                        next(iter(self._outbox)),
                    )
                    await ws.send(self._outbox.pop(key))

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
