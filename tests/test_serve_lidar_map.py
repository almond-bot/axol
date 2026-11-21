"""The Sensors page's live map (``serve/lidar_map.py`` and ``/api/lidar/ws?map=1``)."""

from __future__ import annotations

import json
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from almond_axol.lidar.config import LidarConfig, LidarMount, pose_matrix
from almond_axol.lidar.stream import LidarFrame, StreamStats
from almond_axol.nav.geometry import Pose2D, planar_matrix, pose2d
from almond_axol.nav.mapping import MapBuilder
from almond_axol.nav.odometry import LidarOdometry
from almond_axol.nav.sim import SimLidar, office_route, office_world
from almond_axol.serve import app as app_module
from almond_axol.serve.lidar_link import LidarHub
from almond_axol.serve.lidar_map import LiveMap, decode_map_chunk, encode_map_chunk

MOUNT = pose_matrix(0.2, 0.0, 0.9, 0.0, 0.0, 0.0)
START = Pose2D(1.5, 3.5, 0.0)


def test_map_chunk_round_trip() -> None:
    pts = np.array([[1.234, -5.0, 0.5], [400.0, 0, 0], [-0.01, 2.0, 3.0]], np.float32)
    epoch, replace, out = decode_map_chunk(encode_map_chunk(7, pts, replace=True))
    assert (epoch, replace) == (7, True)
    # 400 m is out of range for int16 centimetres: dropped.
    np.testing.assert_allclose(out, [[1.23, -5.0, 0.5], [-0.01, 2.0, 3.0]], atol=6e-3)
    with pytest.raises(ValueError):
        decode_map_chunk(b"nope" + bytes(12))


def test_live_map_follows_a_drive_and_only_sends_new_voxels() -> None:
    lidar = SimLidar(office_world(), MOUNT, points_per_frame=4000, seed=2)
    live = LiveMap(MOUNT)
    poses = [Pose2D(1.5 + 0.05 * i, 3.5, 0.0) for i in range(50)]
    poses += [Pose2D(4.0, 3.5, i * math.pi / 40) for i in range(1, 11)]
    first = 0
    for k, p in enumerate(poses):
        live.update(lidar.scan(p, k * 0.1))
        if k == 0:
            first = len(live.take_new())
            assert first == live.size > 1000
    added = live.take_new()
    assert len(added) == live.size - first
    assert len(live.take_new()) == 0
    # Voxels are unique.
    assert len(np.unique(live.all_points(), axis=0)) == live.size

    est = pose2d(planar_matrix(START.x, START.y, START.yaw) @ live.pose)
    truth = poses[-1]
    assert math.hypot(est.x - truth.x, est.y - truth.y) < 0.08
    assert live.state == "mapping"
    assert live.distance == pytest.approx(2.5, abs=0.15)

    pose = live.pose_message(3)
    assert pose["epoch"] == 3 and len(pose["m"]) == 12
    trail = live.trail_message(3)
    assert len(trail["points"]) > 20
    assert live.summary()["points"] == live.size


def test_live_map_stops_growing_when_full() -> None:
    lidar = SimLidar(office_world(), MOUNT, points_per_frame=4000, seed=2)
    live = LiveMap(MOUNT, max_voxels=500)
    live.update(lidar.scan(START, 0.0))
    assert live.size == 500 and live.full
    assert live.summary()["full"]


class _FakeLocalizer:
    """Lost after the first fix, found again by the next search."""

    def __init__(self, nav_map, t_base_lidar) -> None:  # noqa: ANN001
        self.last = None
        self.state = "initializing"
        self.searches = 0

    def _fix(self, state: str, x: float):  # noqa: ANN202
        self.state = state
        self.last = SimpleNamespace(
            pose=planar_matrix(x, 0, 0), state=state, match_ratio=0.9
        )
        return self.last

    def initialize(self, frame, imu=None, guess=None):  # noqa: ANN001, ANN201
        self.searches += 1
        return self._fix("ok", 1.0)

    def update(self, frame, imu=None):  # noqa: ANN001, ANN201
        return self._fix("lost", self.last.pose[0, 3] + 0.05)


def test_saved_map_draws_the_map_and_searches_again_when_lost() -> None:
    cloud = np.array([[0, 0, 0], [0, 0, 0.01], [1, 2, 1]], np.float32)
    live = LiveMap(
        MOUNT,
        "office",
        load_map=lambda name: SimpleNamespace(name=name, cloud=cloud),
        localizer=_FakeLocalizer,
    )
    assert live.size == 2
    z = np.zeros(1, np.uint8)
    frame = LidarFrame(0.0, 0, np.zeros((1, 3), np.float32), z, z, z, 0.1, 0)
    live.update(frame)
    assert live.state == "ok"
    for _ in range(60):
        live.update(frame)
    assert live.state in ("ok", "lost")
    assert live._localizer.searches == 2
    # The saved map doesn't grow.
    assert live.size == 2


# -- the hub and the websocket --------------------------------------------


class _FakeStream:
    """Sends the simulated office's scans along a short drive."""

    def __init__(self, config: LidarConfig, *, on_frame, **_: object) -> None:
        self.on_frame = on_frame
        self.shared = False
        self.error = None
        self._stop = threading.Event()
        self._last = None

    def start(self) -> _FakeStream:
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=3000, seed=4)
        route = office_route(step=0.05)

        def run() -> None:
            k = 0
            while not self._stop.wait(0.03):
                frame = lidar.scan(route[min(k, len(route) - 1)], k * 0.1)
                self._last = time.monotonic()
                self.on_frame(frame)
                k += 1

        threading.Thread(target=run, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def tap_readers(self) -> int:
        return 0

    def imu_between(self, t0: float, t1: float) -> list:
        return []

    def stats(self) -> StreamStats:
        age = None if self._last is None else time.monotonic() - self._last
        return StreamStats(2000, 40, 200, 10, 0, 0, age)


def _config() -> LidarConfig:
    return LidarConfig(
        lidar_ip="192.168.1.199",
        host_ip="192.168.1.5",
        mount=LidarMount(x=0.2, z=0.9),
    )


def _next(ws, want):  # noqa: ANN001, ANN202
    """The next message matching ``want`` (bytes magic or JSON type)."""
    for _ in range(400):
        msg = ws.receive()
        data = msg.get("bytes")
        if data is not None and isinstance(want, bytes) and data[:4] == want:
            return data
        text = msg.get("text")
        if text is not None and isinstance(want, str):
            parsed = json.loads(text)
            if parsed.get("type") == want:
                return parsed
    raise AssertionError(f"no {want!r} message")


def test_websocket_streams_the_live_map(monkeypatch: pytest.MonkeyPatch) -> None:
    sources: list[str | None] = []

    def map_factory(mount: np.ndarray, source: str | None) -> LiveMap:
        sources.append(source)
        if source is not None:
            raise FileNotFoundError(source)
        return LiveMap(mount)

    monkeypatch.setattr(
        app_module,
        "LidarHub",
        lambda: LidarHub(
            load_config=_config,
            stream_factory=_FakeStream,
            linger=0.1,
            retry=0.1,
            map_factory=map_factory,
        ),
    )
    app = app_module.create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/api/lidar/ws?map=1") as ws:
            epoch, replace, _ = decode_map_chunk(_next(ws, b"LDM1"))
            assert replace and epoch == 1
            assert _next(ws, "trail")["epoch"] == 1
            # Later chunks only add voxels.
            for _ in range(20):
                e, replace, pts = decode_map_chunk(_next(ws, b"LDM1"))
                if len(pts):
                    break
            assert (e, replace) == (1, False) and len(pts)
            pose = _next(ws, "pose")
            assert pose["state"] == "mapping" and len(pose["m"]) == 12

            ws.send_text(json.dumps({"type": "reset-map"}))
            for _ in range(20):
                epoch, replace, _ = decode_map_chunk(_next(ws, b"LDM1"))
                if epoch == 2:
                    break
            assert (epoch, replace) == (2, True)

            ws.send_text(json.dumps({"type": "map-source", "source": "office"}))
            for _ in range(40):
                status = _next(ws, "status")
                if (status["map"] or {}).get("error"):
                    break
            assert status["map"]["source"] == "office"
            assert "no saved map named office" in status["map"]["error"]
            # Names that aren't map names are ignored.
            ws.send_text(json.dumps({"type": "map-source", "source": "../etc"}))
            ws.send_text("not json")
            assert _next(ws, "status")["map"]["source"] == "office"
        # A viewer without map=1 gets scans only.
        with client.websocket_connect("/api/lidar/ws") as ws:
            for _ in range(15):
                msg = ws.receive()
                assert not (msg.get("bytes") or b"").startswith(b"LDM1")
    assert sources[:3] == [None, None, "office"]


def test_maps_endpoint_lists_saved_maps(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # noqa: ANN001
    monkeypatch.setenv("ALMOND_HOME", str(tmp_path))
    builder = MapBuilder()
    lidar = SimLidar(office_world(), MOUNT, points_per_frame=2000, seed=1)
    odo = LidarOdometry(MOUNT)
    for k in range(3):
        r = odo.update(lidar.scan(START, k * 0.1))
        builder.add(r.pose, r.points)
    builder.build("lab").save()
    with TestClient(app_module.create_app()) as client:
        assert client.get("/api/lidar/maps").json() == {"maps": ["lab"]}


# -- the camera-only Sensors operation ---------------------------------------


def test_sensors_operation_streams_cameras_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from almond_axol.cli import jog, sensors
    from almond_axol.serve.commands import COMMANDS, is_robot_free

    cmd = COMMANDS["sensors"]
    assert not cmd.uses_can_bus and cmd.streams_video
    assert cmd.camera_mode == "teleop"
    assert not is_robot_free(cmd, {})  # no sim: it streams real cameras

    with pytest.raises(SystemExit, match="Settings > Cameras"):
        sensors._run(sensors.SensorsCmdConfig())

    seen = {}

    async def fake_run_with_cameras(cfg, session, stop_event, name):  # noqa: ANN001
        seen["name"] = name
        seen["cameras"] = cfg.cameras
        await session.run(stop_event)

    monkeypatch.setattr(jog, "_run_with_cameras", fake_run_with_cameras)
    stop = threading.Event()
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    started = time.monotonic()
    sensors._run(sensors.SensorsCmdConfig(cameras={"head": 123}), stop)
    assert time.monotonic() - started >= 0.25
    assert seen == {"name": "sensors", "cameras": {"head": 123}}
