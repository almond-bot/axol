"""VRServer connect-time announces and the client's session-config re-request.

The web client installs its message listeners a render after the socket
opens, so the announces the server pushes on accept can arrive before anyone
listens (and ``settings`` is only re-sent on change).
``{"type": "session-config-request"}`` asks for them again; these tests pin
that :meth:`VRServer.set_announce` entries ride along with the built-in
session config, in order, and that a replay goes only to the requesting
client.
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

    def test_announce_rides_with_session_config(self) -> None:
        server = self._server()
        ws = _FakeSocket()
        _run(server._send_session_config(ws))
        types = [m["type"] for m in ws.sent]
        # mode first, then the registered announces, then the built-ins.
        self.assertEqual(types[:2], ["mode", "settings"])
        self.assertIn("episode", types)
        self.assertLess(types.index("settings"), types.index("episode"))
        by_type = {m["type"]: m["value"] for m in ws.sent}
        self.assertEqual(by_type["mode"], "teleop")
        self.assertEqual(
            by_type["settings"], {"schema": [], "values": {"box_mode": True}}
        )
        self.assertEqual(by_type["episode"], 3)

    def test_set_announce_none_removes_entry(self) -> None:
        server = self._server()
        server.set_announce("settings", None)
        ws = _FakeSocket()
        _run(server._send_session_config(ws))
        self.assertNotIn("settings", [m["type"] for m in ws.sent])

    def test_request_replays_to_requesting_client_only(self) -> None:
        server = self._server()
        asker, other = _FakeSocket(), _FakeSocket()
        server._active_clients.update({asker, other})
        _run(
            server._handle_message(
                asker, id(asker), json.dumps({"type": "session-config-request"})
            )
        )
        types = [m["type"] for m in asker.sent]
        self.assertEqual(types[:2], ["mode", "settings"])
        self.assertIn("episode", types)
        self.assertEqual(other.sent, [])

    def test_send_failure_is_swallowed(self) -> None:
        class _Broken(_FakeSocket):
            async def send_text(self, data: str) -> None:
                raise RuntimeError("gone")

        server = self._server()
        _run(server._send_session_config(_Broken()))  # must not raise


if __name__ == "__main__":
    unittest.main()
