"""Regression tests for VRServer's signaling and operator-lifecycle handling.

Each test drives a real ``VRServer`` (uvicorn + TLS on a loopback test port)
with WebSocket clients that mimic the headset app and the control panel.
They pin down the field failures fixed on this branch:

- a ``webrtc-request`` sent while the cameras are still starting is parked
  ("webrtc-pending") and answered with a pushed offer once video registers —
  not a terminal "unavailable" that hides the camera screens all session;
- a request retry landing while an offer is being built can't tear down the
  in-flight negotiation (double-offer race);
- the operator-gone notification (used to return the arms to rest on an app
  quit) fires despite stale once-streaming connections, exactly once for a
  headset's paired transports, and never for view-only clients;
- a relaunched headset whose frame seq counter restarted is not black-holed
  by the previous session's seq high-water mark while a view-only client
  (the control panel's camera mirror) stays connected.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import almond_axol.vr.server as server_mod
from almond_axol.vr.config import VRServerConfig
from almond_axol.vr.server import VRServer

from .conftest import collect_types, expect_type, make_frame, ws_connect


class FakeManager:
    """WebRTCManager stand-in whose offers can be held in flight."""

    def __init__(self, hold: bool = False) -> None:
        self.release = asyncio.Event()
        if not hold:
            self.release.set()
        self.create_calls = 0

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        self.create_calls += 1
        await self.release.wait()
        return "v=0 fake-sdp", {"0": "overhead"}

    async def set_answer(self, client_id: int, sdp: str) -> None:
        pass

    async def close(self, client_id: int) -> None:
        pass

    async def close_all(self) -> None:
        pass


async def test_video_pending_then_pushed_offer(test_port, ssl_ctx) -> None:
    """Camera bring-up: early requests wait, then get the offer pushed."""
    server = VRServer(VRServerConfig(port=test_port))
    server.set_video_expected(True)
    async with server:
        ws = await ws_connect(test_port, ssl_ctx)

        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-pending")

        # Retrying while still pending is answered pending again.
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-pending")

        # Video registers (from another thread, like the teleop CLI does):
        # the parked request gets its offer without asking again.
        manager = FakeManager()
        threading.Thread(target=server.set_video_manager, args=(manager,)).start()
        msg = await expect_type(ws, "webrtc-offer")
        assert msg["tracks"] == {"0": "overhead"}

        # Later requests are answered directly.
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-offer")
        assert manager.create_calls == 2
        await ws.close()


async def test_video_expected_but_none_resolves_unavailable(test_port, ssl_ctx) -> None:
    """Camera setup concluding with nothing releases parked clients."""
    server = VRServer(VRServerConfig(port=test_port))
    server.set_video_expected(True)
    async with server:
        ws = await ws_connect(test_port, ssl_ctx)
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-pending")

        threading.Thread(target=server.set_video_sources, args=(None,)).start()
        await expect_type(ws, "webrtc-unavailable")

        # Video is no longer expected: direct answer from here on.
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-unavailable")
        await ws.close()


async def test_unexpected_video_is_unavailable_immediately(test_port, ssl_ctx) -> None:
    """No cameras configured (sim): the reply must stay a prompt unavailable."""
    server = VRServer(VRServerConfig(port=test_port))
    async with server:
        ws = await ws_connect(test_port, ssl_ctx)
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-unavailable")
        await ws.close()


async def test_offer_race_single_offer(test_port, ssl_ctx) -> None:
    """A retry landing mid-build must not tear down the in-flight offer."""
    server = VRServer(VRServerConfig(port=test_port))
    server.set_video_expected(True)
    manager = FakeManager(hold=True)
    async with server:
        ws = await ws_connect(test_port, ssl_ctx)
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await expect_type(ws, "webrtc-pending")

        # Video registers; the pending flush blocks inside create_offer.
        threading.Thread(target=server.set_video_manager, args=(manager,)).start()
        for _ in range(100):
            if manager.create_calls == 1:
                break
            await asyncio.sleep(0.02)
        assert manager.create_calls == 1

        # The client's retry lands while the offer is in flight.
        await ws.send(json.dumps({"type": "webrtc-request"}))
        await asyncio.sleep(0.3)
        assert manager.create_calls == 1, "retry raced a second create_offer"

        manager.release.set()
        got = await collect_types(ws, "webrtc")
        assert got.count("webrtc-offer") == 1, got
        assert "webrtc-unavailable" not in got, got
        await ws.close()


async def test_operator_gone_recency(test_port, ssl_ctx, monkeypatch) -> None:
    """Quit detection despite stale tabs; single fire for paired transports."""
    monkeypatch.setattr(server_mod, "_OPERATOR_RECENCY_S", 1.0)
    gone: list[float] = []
    server = VRServer(VRServerConfig(port=test_port))
    server.set_on_operator_gone(lambda: gone.append(time.monotonic()))

    async with server:
        # Stale tab: streamed poses once, stays connected, goes idle.
        stale = await ws_connect(test_port, ssl_ctx)
        for i in range(5):
            await stale.send(make_frame(seq=i + 1))
        await asyncio.sleep(1.2)  # let its recency lapse

        # Operator quits: the stale tab must not mask it.
        operator = await ws_connect(test_port, ssl_ctx)
        for i in range(30):
            await operator.send(make_frame(seq=100 + i))
            await asyncio.sleep(0.01)
        await operator.close()
        await asyncio.sleep(0.5)
        assert len(gone) == 1, f"stale tab masked the quit: {gone}"

        # Paired transports (USB tunnel + network standby): one fire, on the
        # last close.
        a = await ws_connect(test_port, ssl_ctx)
        b = await ws_connect(test_port, ssl_ctx)
        for i in range(30):
            await a.send(make_frame(seq=200 + 2 * i))
            await b.send(make_frame(seq=201 + 2 * i))
            await asyncio.sleep(0.01)
        await a.close()
        await asyncio.sleep(0.5)
        assert len(gone) == 1, f"fired while a sibling was fresh: {gone}"
        await b.close()
        await asyncio.sleep(0.5)
        assert len(gone) == 2, f"second close did not fire: {gone}"

        # View-only client (never sent poses): its disconnect is a no-op.
        viewer = await ws_connect(test_port, ssl_ctx)
        await viewer.send(json.dumps({"type": "webrtc-request"}))
        await asyncio.sleep(0.2)
        await viewer.close()
        await asyncio.sleep(0.5)
        assert len(gone) == 2, f"viewer disconnect fired: {gone}"

        await stale.close()


async def test_relaunched_headset_not_seq_black_holed(test_port, ssl_ctx) -> None:
    """Pose state resets when the last pose sender leaves, not all clients."""
    server = VRServer(VRServerConfig(port=test_port))
    async with server:
        # View-only panel: connected the whole time, never sends poses.
        panel = await ws_connect(test_port, ssl_ctx)
        await panel.send(json.dumps({"type": "webrtc-request"}))

        # Session 1: headset streams seq 1..100.
        a = await ws_connect(test_port, ssl_ctx)
        for i in range(100):
            await a.send(make_frame(seq=i + 1))
        await asyncio.sleep(0.5)
        assert server.get_frame() is not None

        # App quit + relaunch: new connection, seq restarts at 1. With the
        # panel still connected, the old high-water mark must not drop the
        # new session's frames (no poses would mean no engage).
        await a.close()
        await asyncio.sleep(0.5)
        assert server.get_frame() is None, "pose state kept after operator quit"

        b = await ws_connect(test_port, ssl_ctx)
        for i in range(10):
            await b.send(make_frame(seq=i + 1))
        await asyncio.sleep(0.5)
        assert server.get_frame() is not None, "relaunched headset black-holed"

        await b.close()
        await panel.close()


async def test_pose_dedup_across_transports(test_port, ssl_ctx) -> None:
    """The seq dedup itself still works: duplicate copies are dropped."""
    server = VRServer(VRServerConfig(port=test_port))
    seen: list[int] = []
    server.set_on_frame(lambda f: seen.append(f.seq))
    async with server:
        usb = await ws_connect(test_port, ssl_ctx)
        net = await ws_connect(test_port, ssl_ctx)
        for i in range(20):
            await usb.send(make_frame(seq=i + 1))
            await net.send(make_frame(seq=i + 1))  # duplicate copy
        await asyncio.sleep(0.5)
        assert seen == list(range(1, 21)), seen
        await usb.close()
        await net.close()
