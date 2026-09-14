from __future__ import annotations

import asyncio
import errno
import signal
from types import SimpleNamespace

from almond_axol.utils import ports
from almond_axol.vr import control_channel


def test_listening_pids_and_port_reclaim(monkeypatch) -> None:
    monkeypatch.setattr(ports.shutil, "which", lambda name: "/usr/bin/ss")
    monkeypatch.setattr(ports.os, "getpid", lambda: 20)
    monkeypatch.setattr(
        ports.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='users:(("python",pid=10,fd=3)) users:(("self",pid=20,fd=4))'
        ),
    )
    assert ports.listening_pids(8000) == {10}
    monkeypatch.setattr(ports.shutil, "which", lambda name: None)
    assert ports.listening_pids(8000) == set()

    polls = iter([{10, 11}, {11}])
    monkeypatch.setattr(ports, "listening_pids", lambda port: next(polls))
    monkeypatch.setattr(ports.time, "sleep", lambda seconds: None)
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        ports, "_signal_pid", lambda pid, sig: signals.append((pid, sig))
    )
    ports.reclaim_port(8000)
    assert signals == [(10, signal.SIGTERM), (11, signal.SIGTERM), (11, signal.SIGKILL)]


def test_open_socket_retries_and_tracks_owner(monkeypatch) -> None:
    class FakeSocket:
        attempts = 0

        def __init__(self, *args) -> None:
            self.closed = False
            self.bound = None

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, address) -> None:
            FakeSocket.attempts += 1
            if FakeSocket.attempts == 1:
                raise OSError(errno.EADDRINUSE, "busy")
            self.bound = address

        def close(self) -> None:
            self.closed = True

    reclaimed: list[int] = []
    monkeypatch.setattr(ports.socket, "socket", FakeSocket)
    monkeypatch.setattr(ports.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(ports, "reclaim_port", reclaimed.append)
    ports._owned_listen_sockets.clear()
    sock = ports.open_listen_socket("127.0.0.1", 8002)
    assert sock.bound == ("127.0.0.1", 8002)
    assert ports._owned_listen_sockets[8002] is sock
    assert reclaimed == []


class _Channel:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, event):
        def decorate(callback):
            self.handlers[event] = callback
            return callback

        return decorate


class _Peer:
    def __init__(self, config) -> None:
        self.config = config
        self.channel = _Channel()
        self.handlers = {}
        self.connectionState = "new"
        self.localDescription = SimpleNamespace(sdp="offer-sdp")
        self.remote = None
        self.closed = False

    def createDataChannel(self, label, **kwargs):
        self.channel_args = (label, kwargs)
        return self.channel

    def on(self, event):
        def decorate(callback):
            self.handlers[event] = callback
            return callback

        return decorate

    async def createOffer(self):
        return SimpleNamespace(sdp="draft", type="offer")

    async def setLocalDescription(self, offer) -> None:
        self.offer = offer

    async def setRemoteDescription(self, answer) -> None:
        self.remote = answer

    async def close(self) -> None:
        self.closed = True


def test_control_channel_offer_messages_answer_and_close(monkeypatch) -> None:
    peers: list[_Peer] = []

    def make_peer(config):
        peer = _Peer(config)
        peers.append(peer)
        return peer

    monkeypatch.setattr(control_channel, "RTCPeerConnection", make_peer)
    monkeypatch.setattr(control_channel, "ice_servers", lambda: [])
    received: list[tuple[int, str]] = []
    manager = control_channel.ControlChannelManager(
        lambda client, msg: received.append((client, msg))
    )

    async def exercise() -> None:
        assert await manager.create_offer(7) == "offer-sdp"
        peer = peers[0]
        assert peer.channel_args == ("pose", {"ordered": False, "maxRetransmits": 0})
        peer.channel.handlers["message"]("pose-json")
        peer.channel.handlers["message"](b"ignored")
        assert received == [(7, "pose-json")]

        await manager.set_answer(7, "answer-sdp")
        assert peer.remote.sdp == "answer-sdp"
        await manager.set_answer(99, "ignored")

        peer.connectionState = "failed"
        await peer.handlers["connectionstatechange"]()
        assert peer.closed

        await manager.create_offer(8)
        await manager.create_offer(9)
        await manager.close_all()
        assert all(item.closed for item in peers)

    asyncio.run(exercise())
