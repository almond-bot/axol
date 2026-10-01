"""VR shutdown drains real TLS/WebSocket clients before its loop may close."""

from __future__ import annotations

import asyncio
import socket
import ssl
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from websockets.asyncio.client import connect

from almond_axol.utils.certs import create_self_signed_cert
from almond_axol.vr import server as server_module
from almond_axol.vr.config import VRServerConfig
from almond_axol.vr.server import VRServer


@pytest.fixture(scope="module")
def tls_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("vr-shutdown-certs")
    certificate = str(directory / "cert.pem")
    key = str(directory / "key.pem")
    create_self_signed_cert(certificate, key)
    return certificate, key


def _make_server(tls_paths):
    certificate, key = tls_paths
    return VRServer(VRServerConfig(port=0, certfile=certificate, keyfile=key))


def _client_tls():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


async def _wait_started(server):
    async with asyncio.timeout(3.0):
        while not server._uvicorn_server.started:
            if server._server_task.done():
                await server._server_task
            await asyncio.sleep(0.01)
    return server._listen_socket.getsockname()[1]


async def _assert_drained(server, serve_task, baseline_tasks, port):
    # Allow scheduled protocol callbacks to finish, without asyncio.run's
    # blanket final cancellation concealing a server-owned task leak.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert serve_task.done() and not serve_task.cancelled()
    assert serve_task.result() is None
    assert server._server_task is None
    assert server._uvicorn_server is None
    assert server._listen_socket is None
    assert server._tls_files is None
    assert not server._active_clients
    assert asyncio.all_tasks() - baseline_tasks == set()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))


def _unresponsive_tls_websocket(port):
    """Upgrade then stop reading, including TLS close-notify responses."""
    connection = _client_tls().wrap_socket(
        socket.create_connection(("127.0.0.1", port), timeout=3.0),
        server_hostname="localhost",
    )
    connection.sendall(
        b"GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
        b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
    )
    response = b""
    while b"\r\n\r\n" not in response:
        received = connection.recv(4096)
        if not received:
            connection.close()
            raise AssertionError("server closed before WebSocket upgrade")
        response += received
    assert response.startswith(b"HTTP/1.1 101 ")
    return connection


def _unfinished_tls_handshake(port):
    """Read the server hello, then leave the TLS negotiation incomplete."""
    incoming, outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = _client_tls().wrap_bio(incoming, outgoing, server_hostname="localhost")
    with suppress(ssl.SSLWantReadError):
        client.do_handshake()
    connection = socket.create_connection(("127.0.0.1", port), timeout=3.0)
    connection.sendall(outgoing.read())
    assert connection.recv(4096), "server must accept before shutdown starts"
    return connection


def test_enable_disable_reenable_without_clients(tls_paths):
    async def scenario():
        server = _make_server(tls_paths)
        baseline_tasks = asyncio.all_tasks()
        for _ in range(2):
            await server.enable()
            port = await _wait_started(server)
            serve_task = server._server_task
            await server.disable()
            await _assert_drained(server, serve_task, baseline_tasks, port)
        # Cleanup is idempotent after a completed shutdown.
        await server.disable()

    asyncio.run(scenario())


def test_disable_immediately_after_enable(tls_paths):
    async def scenario():
        server = _make_server(tls_paths)
        baseline_tasks = asyncio.all_tasks()
        await server.enable()
        serve_task = server._server_task
        port = server._listen_socket.getsockname()[1]
        await server.disable()
        await _assert_drained(server, serve_task, baseline_tasks, port)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "client_state", ["connected", "disconnected", "unresponsive", "tls-handshake"]
)
@pytest.mark.parametrize("server_abort_available", [True, False])
def test_disable_drains_tls_websockets(
    tls_paths, monkeypatch, client_state, server_abort_available
):
    monkeypatch.setattr(server_module, "_SERVER_SHUTDOWN_GRACE_S", 0.25)
    if not server_abort_available:
        # Exercise the Python 3.12 path even when CI uses Python 3.13+.
        monkeypatch.setattr(asyncio.Server, "abort_clients", None, raising=False)

    async def scenario():
        server = _make_server(tls_paths)
        baseline_tasks = asyncio.all_tasks()
        await server.enable()
        port = await _wait_started(server)
        serve_task = server._server_task
        client = None
        try:
            if client_state == "unresponsive":
                client = await asyncio.to_thread(_unresponsive_tls_websocket, port)
            elif client_state == "tls-handshake":
                client = await asyncio.to_thread(_unfinished_tls_handshake, port)
            else:
                client = await connect(f"wss://127.0.0.1:{port}/ws", ssl=_client_tls())
                if client_state == "disconnected":
                    await client.close()
            await asyncio.wait_for(server.disable(), timeout=3.0)
            if client_state == "connected":
                await client.wait_closed()
            await _assert_drained(server, serve_task, baseline_tasks, port)
        finally:
            if isinstance(client, socket.socket):
                client.close()
            elif client is not None:
                await client.close()
            if server._server_task is not None:
                await server.disable()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", [RuntimeError("serve failed"), TimeoutError("serve failed")]
)
def test_disable_propagates_server_failure_and_retains_ownership(failure):
    async def scenario():
        server = VRServer()

        async def fail():
            raise failure

        serve_task = asyncio.create_task(fail())
        owner = SimpleNamespace(should_exit=False)
        server._uvicorn_server = owner
        server._server_task = serve_task
        with pytest.raises(type(failure), match="serve failed"):
            await server.disable()
        assert owner.should_exit
        assert server._server_task is serve_task
        assert server._uvicorn_server is owner

    asyncio.run(scenario())


def test_cancelled_disable_keeps_server_task_available_for_retry():
    async def scenario():
        server = VRServer()
        released = asyncio.Event()
        serve_task = asyncio.create_task(released.wait())
        owner = SimpleNamespace(should_exit=False)
        server._uvicorn_server = owner
        server._server_task = serve_task
        cleanup = asyncio.create_task(server.disable())
        while not owner.should_exit:
            await asyncio.sleep(0)
        cleanup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cleanup
        assert not serve_task.done()
        assert server._server_task is serve_task
        assert server._uvicorn_server is owner
        released.set()
        await server.disable()
        assert server._server_task is None

    asyncio.run(scenario())


def test_failed_forced_shutdown_retains_task_ownership(monkeypatch):
    monkeypatch.setattr(server_module, "_SERVER_SHUTDOWN_GRACE_S", 0.01)
    monkeypatch.setattr(server_module, "_SERVER_SHUTDOWN_ABORT_S", 0.01)

    async def scenario():
        server = VRServer()
        transport = Mock()
        owner = SimpleNamespace(
            should_exit=False,
            server_state=SimpleNamespace(
                connections=[SimpleNamespace(transport=transport)]
            ),
        )
        serve_task = asyncio.create_task(asyncio.Event().wait())
        server._uvicorn_server = owner
        server._server_task = serve_task
        try:
            with pytest.raises(
                RuntimeError, match="VR server shutdown did not complete"
            ):
                await server.disable()
            transport.abort.assert_called_once_with()
            assert not serve_task.done()
            assert server._server_task is serve_task
            assert server._uvicorn_server is owner
        finally:
            serve_task.cancel()
            with suppress(asyncio.CancelledError):
                await serve_task

    asyncio.run(scenario())


def test_failed_shutdown_still_frees_the_vr_port(tls_paths, monkeypatch):
    """A stuck/failed disable keeps the task but must release the listener.

    Under ``axol serve`` the process outlives the operation: a listener kept
    open here would make every later VR operation fail to bind its port.
    """

    async def scenario():
        server = _make_server(tls_paths)
        await server.enable()
        port = await _wait_started(server)
        serve_task = server._server_task

        async def stuck(_server, _task):
            raise RuntimeError("VR server shutdown did not complete")

        monkeypatch.setattr(server, "_join_server_task", stuck)
        try:
            with pytest.raises(RuntimeError, match="did not complete"):
                await server.disable()
            assert server._server_task is serve_task  # ownership retained
            assert server._listen_socket is None
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", port))
                listener.listen()
        finally:
            with suppress(Exception):
                await asyncio.wait_for(serve_task, timeout=5.0)

    asyncio.run(scenario())


def test_wrapped_listener_is_reclaimable_by_the_next_bind(tls_paths):
    """The fd-owning wrapper, not the detached original, is registered."""
    from almond_axol.utils import ports

    async def scenario():
        server = _make_server(tls_paths)
        await server.enable()
        await _wait_started(server)
        try:
            assert ports._owned_listen_sockets[0] is server._listen_socket
        finally:
            await server.disable()

    asyncio.run(scenario())

    # A leaked (still listening) wrapper on a fixed port is closed and rebound.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    leaked = server_module._ClientTrackingSocket(
        ports.open_listen_socket("127.0.0.1", port)
    )
    ports.register_listen_socket(port, leaked)
    leaked.listen()
    rebound = ports.open_listen_socket("127.0.0.1", port)
    try:
        assert leaked.fileno() == -1
    finally:
        rebound.close()
