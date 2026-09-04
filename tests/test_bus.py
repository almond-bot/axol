"""CanBus lifecycle: state-specific errors for I/O on an unusable bus.

Exercises ``CanBus`` against an in-process fake ``axol-rt proxy`` (a Unix
socket server speaking the length-prefixed protocol) so no binary, CAN
interface, or subprocess is needed.
"""

from __future__ import annotations

import asyncio
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import can

from almond_axol.motor import bus as bus_module
from almond_axol.motor.bus import CanBus

# CanBus.start() imports rt.link lazily; import it up front so the heavy
# package import (which itself uses subprocess) happens before Popen is faked.
import almond_axol.rt.link  # noqa: E402,F401


def _frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


class FakeProxyServer:
    """Accepts one client, sends the ready marker, records what it receives."""

    def __init__(self, path: str, *, ready: bool = True) -> None:
        self.path = path
        self.ready = ready
        self.received: list[bytes] = []
        self.connected = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None
        self._server: asyncio.AbstractServer | None = None

    async def serve(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=self.path)

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writer = writer
        if self.ready:
            writer.write(_frame(b"R"))
            await writer.drain()
        self.connected.set()
        try:
            while True:
                (size,) = struct.unpack("<I", await reader.readexactly(4))
                payload = await reader.readexactly(size)
                self.received.append(payload)
                if payload == b"Q":
                    break
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        writer.close()

    async def drop_client(self) -> None:
        """Simulate the proxy dying under a live bus."""
        assert self._writer is not None
        self._writer.close()
        await self._writer.wait_closed()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


class FakeProc:
    """Stand-in for the ``axol-rt proxy`` Popen handle."""

    def __init__(self, server: FakeProxyServer) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self._server = server

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = 0


class CanBusStateTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.servers: list[FakeProxyServer] = []
        self.ready = True
        self._tmp = tempfile.TemporaryDirectory()

        def fake_popen(args: list[str], **_: object) -> FakeProc:
            socket_path = args[args.index("--socket") + 1]
            server = FakeProxyServer(socket_path, ready=self.ready)
            self.servers.append(server)
            asyncio.get_running_loop().create_task(server.serve())
            return FakeProc(server)

        self._patches = [
            patch.object(bus_module.subprocess, "Popen", fake_popen),
            patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt"),
        ]
        for p in self._patches:
            p.start()
        self.bus = CanBus("can_test")
        # Keep the socket path short and inside a scratch dir.
        self.bus._socket_path = os.path.join(self._tmp.name, "proxy.sock")

    async def asyncTearDown(self) -> None:
        await self.bus.close()
        for server in self.servers:
            await server.stop()
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    async def test_send_before_start_explains_ownership(self) -> None:
        self.assertFalse(self.bus.is_open)
        with self.assertRaises(can.CanOperationError) as ctx:
            await self.bus._send(0x7FF, b"\x01")
        message = str(ctx.exception)
        self.assertIn("has not been opened", message)
        self.assertIn("not a daemon", message)
        self.assertIn("Axol.connect()", message)
        self.assertIn("AxolArm.enable()", message)

    async def test_send_while_starting_says_still_starting(self) -> None:
        # A proxy that never sends the ready marker keeps start() waiting in
        # its handshake — exactly the window a racing enable() lands in.
        self.ready = False
        start_task = asyncio.create_task(self.bus.start())
        await asyncio.wait_for(self._wait_for_server_connected(), 2.0)

        self.assertFalse(self.bus.is_open)
        with self.assertRaises(can.CanOperationError) as ctx:
            await self.bus._send(0x7FF, b"\x01")
        self.assertIn("is still starting", str(ctx.exception))
        self.assertIn("await the in-flight", str(ctx.exception))

        start_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await start_task

    async def test_start_then_close_reports_closed(self) -> None:
        await self.bus.start()
        self.assertTrue(self.bus.is_open)
        await self.bus._send(0x123, b"\x01\x02")
        await asyncio.wait_for(self._wait_for_received(1), 2.0)
        self.assertEqual(self.servers[0].received[0][:1], b"S")

        await self.bus.close()
        self.assertFalse(self.bus.is_open)
        await asyncio.wait_for(self._wait_for_received(2), 2.0)
        self.assertEqual(self.servers[0].received[1], b"Q")
        with self.assertRaises(can.CanOperationError) as ctx:
            await self.bus._send(0x123, b"\x01")
        self.assertIn("was closed", str(ctx.exception))
        self.assertIn("Axol.connect()", str(ctx.exception))

    async def test_start_is_idempotent_while_open(self) -> None:
        await self.bus.start()
        await self.bus.start()
        self.assertEqual(len(self.servers), 1)
        self.assertTrue(self.bus.is_open)

    async def test_proxy_dying_under_live_bus_is_reported(self) -> None:
        await self.bus.start()
        with self.assertLogs(bus_module._logger, level="WARNING") as logs:
            await self.servers[0].drop_client()
            await asyncio.wait_for(self._wait_for_closed_reason(), 2.0)
        self.assertTrue(any("axol-rt proxy for can_test" in m for m in logs.output))

        self.assertFalse(self.bus.is_open)
        with self.assertRaises(can.CanOperationError) as ctx:
            await self.bus._send(0x123, b"\x01")
        message = str(ctx.exception)
        self.assertIn("axol-rt proxy for can_test", message)
        self.assertIn("to reopen the bus", message)

    async def test_missing_binary_is_an_initialization_error(self) -> None:
        with patch(
            "almond_axol.rt.link.find_binary",
            side_effect=FileNotFoundError("axol-rt binary not found"),
        ):
            with self.assertRaises(can.CanInitializationError) as ctx:
                await self.bus.start()
        self.assertIn("could not start axol-rt proxy", str(ctx.exception))
        self.assertIn("axol-rt binary not found", str(ctx.exception))
        self.assertEqual(self.servers, [])
        self.assertFalse(self.bus.is_open)

    async def test_popen_failure_is_an_initialization_error(self) -> None:
        def failing_popen(*_: object, **__: object) -> FakeProc:
            raise PermissionError("denied")

        with patch.object(bus_module.subprocess, "Popen", failing_popen):
            with self.assertRaises(can.CanInitializationError):
                await self.bus.start()
        self.assertFalse(self.bus.is_open)

    async def _wait_for_server_connected(self) -> None:
        while not self.servers or not self.servers[-1].connected.is_set():
            await asyncio.sleep(0.01)

    async def _wait_for_received(self, count: int) -> None:
        while len(self.servers[0].received) < count:
            await asyncio.sleep(0.01)

    async def _wait_for_closed_reason(self) -> None:
        while self.bus._closed_reason is None:
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
