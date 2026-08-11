"""Shared helpers for the VR teleop regression tests.

These tests run against real ``VRServer`` instances (uvicorn + TLS on
loopback test ports) and the pure ``VRTeleopCore`` state machine. No robot,
cameras, or JAX are required, so the suite runs anywhere the base
dependencies are installed: ``uv run pytest``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import ssl
import time

import pytest
import websockets

# Ports are allocated sequentially per test so suites (and reruns against a
# lingering TIME_WAIT socket) never collide. VRServer reclaims its port from
# stale listeners, so strict uniqueness isn't required — just convenient.
_PORTS = itertools.count(8460)

POSE = {
    "position": {"x": 0.1, "y": 0.2, "z": 0.3},
    "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


def make_frame(
    seq: int | None = None,
    locks: bool = False,
    reset: bool = False,
    dz: float = 0.0,
) -> str:
    """A serialized VRFrame like the headset app sends every XR frame."""
    pose = {
        "position": {"x": 0.1, "y": 0.2, "z": 0.3 + dz},
        "quaternion": POSE["quaternion"],
    }
    payload = {
        "l_ee": pose,
        "r_ee": pose,
        "l_elbow": {"x": 0.1, "y": 0.3, "z": 0.2},
        "r_elbow": {"x": -0.1, "y": 0.3, "z": 0.2},
        "l_lock": locks,
        "r_lock": locks,
        "l_grip": 1.0,
        "r_grip": 1.0,
        "reset": reset,
        "t": time.monotonic() * 1000.0,
    }
    if seq is not None:
        payload["seq"] = seq
    return json.dumps(payload)


@pytest.fixture
def test_port() -> int:
    return next(_PORTS)


@pytest.fixture
def ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def ws_connect(port: int, ctx: ssl.SSLContext):
    """Connect to a VRServer's /ws, retrying while uvicorn starts up."""
    last: Exception | None = None
    for _ in range(50):
        try:
            return await websockets.connect(f"wss://localhost:{port}/ws", ssl=ctx)
        except OSError as exc:
            last = exc
            await asyncio.sleep(0.1)
    raise RuntimeError(f"server on port {port} never came up: {last}")


async def expect_type(ws, msg_type: str, timeout: float = 5.0) -> dict:
    """Receive until a message of ``msg_type`` arrives (skipping others)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"timed out waiting for {msg_type!r}"
        msg = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        if msg.get("type") == msg_type:
            return msg


async def collect_types(ws, prefix: str, quiet: float = 1.0) -> list[str]:
    """Message types starting with ``prefix`` until ``quiet`` s of silence."""
    got: list[str] = []
    while True:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), quiet))
        except (asyncio.TimeoutError, TimeoutError):
            return got
        if str(msg.get("type", "")).startswith(prefix):
            got.append(msg["type"])
