"""The control panel's live lidar stream (``/api/lidar/ws``)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from almond_axol.lidar.config import LidarConfig, LidarMount
from almond_axol.lidar.stream import LidarFrame, StreamStats
from almond_axol.serve import app as app_module
from almond_axol.serve import lidar_link
from almond_axol.serve.lidar_link import LidarHub, decode_frame, encode_frame


class _FakeStream:
    """Calls ``on_frame`` with a fixed frame ten times a second."""

    started = 0

    def __init__(self, config: LidarConfig, *, on_frame, **_: object) -> None:
        self.config = config
        self.on_frame = on_frame
        self.shared = False
        self.error = None
        self._stop = threading.Event()
        self._last = None

    def start(self) -> _FakeStream:
        type(self).started += 1

        def run() -> None:
            seq = 0
            while not self._stop.wait(0.05):
                xyz = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -0.5]], np.float32)
                self._last = time.monotonic()
                self.on_frame(
                    LidarFrame(
                        0.0,
                        0,
                        xyz,
                        np.array([10, 200], np.uint8),
                        np.zeros(2),
                        np.zeros(2),
                        0.1,
                        seq,
                    )
                )
                seq += 1

        threading.Thread(target=run, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def tap_readers(self) -> int:
        return 0

    def stats(self) -> StreamStats:
        age = None if self._last is None else time.monotonic() - self._last
        return StreamStats(2000, 40, 200, 10, 0, 0, age)


def _config() -> LidarConfig:
    return LidarConfig(
        lidar_ip="192.168.1.199",
        host_ip="192.168.1.5",
        mount=LidarMount(x=0.2, z=1.0, yaw=90.0),
    )


def test_encode_round_trip_thins_and_clips() -> None:
    xyz = np.random.default_rng(0).uniform(-5, 5, (5000, 3)).astype(np.float32)
    xyz[0] = (40.0, 0, 0)  # beyond int16 millimetres: dropped
    refl = np.arange(5000).astype(np.uint8)
    seq, out, out_refl = decode_frame(encode_frame(7, xyz, refl, 1000))
    assert seq == 7
    assert len(out) == 1000 == len(out_refl)
    assert np.abs(out).max() < 5.001
    np.testing.assert_allclose(out[0], xyz[1], atol=6e-4)
    with pytest.raises(ValueError):
        decode_frame(b"nope" + bytes(8))


def test_configured_needs_setup(tmp_path: Path) -> None:
    path = tmp_path / "lidar.json"
    assert not lidar_link.lidar_configured(path)
    LidarConfig().save(path)
    assert not lidar_link.lidar_configured(path)
    _config().save(path)
    assert lidar_link.lidar_configured(path)


def test_websocket_streams_base_frame_points(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeStream.started = 0
    monkeypatch.setattr(
        app_module,
        "LidarHub",
        lambda: LidarHub(
            load_config=_config, stream_factory=_FakeStream, linger=0.1, retry=0.1
        ),
    )
    app = app_module.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/api/lidar/ws?points=5000") as ws:
            first = ws.receive_json()
            assert first["type"] == "status"
            frame = None
            states = []
            for _ in range(40):
                msg = ws.receive()
                if msg.get("bytes"):
                    frame = msg["bytes"]
                    if states and states[-1] == "streaming":
                        break
                elif msg.get("text"):
                    states.append(json.loads(msg["text"])["state"])
            assert frame is not None
            _, xyz, refl = decode_frame(frame)
            # Mount: 0.2 m forward, 1 m up, turned 90° left.
            np.testing.assert_allclose(xyz[0], [0.2, 1.0, 1.0], atol=2e-3)
            np.testing.assert_allclose(xyz[1], [-1.8, 0.0, 0.5], atol=2e-3)
            assert list(refl) == [10, 200]
            assert "streaming" in states
    assert _FakeStream.started == 1


def test_hub_reports_setup_errors() -> None:
    import asyncio

    async def run() -> dict:
        hub = LidarHub(load_config=LidarConfig, linger=0.1, retry=10)
        sub = hub.subscribe()
        try:
            for _ in range(40):
                msg = await asyncio.wait_for(sub.queue.get(), 2)
                if isinstance(msg, dict) and msg["state"] == "error":
                    return msg
            raise AssertionError("no error status")
        finally:
            hub.unsubscribe(sub)
            hub.shutdown()

    msg = asyncio.run(run())
    assert "lidar.setup" in msg["error"]
