"""VRServer connect-time announces and the client's ``get`` re-request.

The web client installs its message listeners a render after the socket
opens, so the announces the server pushes on accept can arrive before anyone
listens (and ``settings`` is only re-sent on change). ``{"type": "get"}``
asks for them again; these tests pin what it sends and that it goes only to
the requesting client.
"""

import asyncio
import json
import unittest

from almond_axol.vr.server import VRServer


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))


def _run(coro):
    return asyncio.run(coro)


class AnnounceTest(unittest.TestCase):
    def _server(self) -> VRServer:
        server = VRServer()
        server.set_mode("teleop")
        server.set_announce("settings", {"schema": [], "values": {"box_mode": True}})
        server.set_episode(3)
        return server

    def test_send_announces_order_and_content(self) -> None:
        server = self._server()
        ws = _FakeSocket()
        _run(server._send_announces(ws))
        self.assertEqual(
            ws.sent,
            [
                {"type": "mode", "value": "teleop"},
                {
                    "type": "settings",
                    "value": {"schema": [], "values": {"box_mode": True}},
                },
                {"type": "episode", "value": 3},
            ],
        )

    def test_get_resends_to_requesting_client_only(self) -> None:
        server = self._server()
        asker, other = _FakeSocket(), _FakeSocket()
        server._active_clients.update({asker, other})
        _run(server._handle_message(asker, id(asker), json.dumps({"type": "get"})))
        self.assertEqual(
            [m["type"] for m in asker.sent], ["mode", "settings", "episode"]
        )
        self.assertEqual(other.sent, [])

    def test_nothing_to_announce_sends_nothing(self) -> None:
        ws = _FakeSocket()
        _run(VRServer()._send_announces(ws))
        self.assertEqual(ws.sent, [])

    def test_send_failure_is_swallowed(self) -> None:
        class _Broken(_FakeSocket):
            async def send_text(self, data: str) -> None:
                raise RuntimeError("gone")

        server = self._server()
        _run(server._send_announces(_Broken()))  # must not raise


if __name__ == "__main__":
    unittest.main()
