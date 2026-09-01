from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from almond_axol.serve import app as app_module
from almond_axol.serve.commands import _format_value, _truthy, build_argv, command_specs
from almond_axol.serve.introspect import build_argparse_schema, build_schema
from almond_axol.serve.settings import SettingsStore


@dataclass
class NestedConfig:
    enabled: bool = False
    gains: list[float] = field(default_factory=lambda: [1.0, 2.0])


@dataclass
class ExampleConfig:
    """Example config.

    Attributes:
        count: Number of attempts.
    """

    count: int = 2
    nested: NestedConfig = field(default_factory=NestedConfig)


def test_draccus_schema_preserves_types_defaults_and_help() -> None:
    schema = build_schema(ExampleConfig)
    count = next(node for node in schema.nodes if node["key"] == "count")
    nested = next(node for node in schema.nodes if node["key"] == "nested")

    assert count["type"] == "number"
    assert count["default"] == 2
    assert count["help"] == "Number of attempts."
    assert [child["key"] for child in nested["children"]] == [
        "nested.enabled",
        "nested.gains",
    ]
    assert nested["children"][1]["type"] == "vector"


def test_argparse_schema_understands_required_flags_and_switches() -> None:
    def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
        parser = subparsers.add_parser("demo")
        parser.add_argument("name")
        parser.add_argument("--count", type=int, default=3)
        parser.add_argument("--verbose", action="store_true")

    schema = build_argparse_schema(add_parser)
    fields = {node["key"]: node for node in schema.nodes}

    assert fields["name"]["required"] is True
    assert fields["count"]["type"] == "number"
    assert fields["verbose"]["type"] == "boolean"


def test_settings_store_persists_merges_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    snapshot = store.update(
        values={"robot.has_gripper": False, "robot.left_channel": "can9"},
        cameras={"serials": {"overhead": "123"}},
    )
    assert snapshot["values"]["robot.has_gripper"] is False
    assert store.can_channels()[0] == "can9"
    assert store.has_gripper() is False
    assert SettingsStore(path).snapshot() == snapshot
    with pytest.raises(KeyError, match="unknown settings"):
        store.update(values={"does.not.exist": 1})


def test_command_catalog_and_argv_contracts() -> None:
    specs = command_specs()
    by_id = {spec["id"]: spec for spec in specs}
    assert {"teleop", "gravity-comp", "waypoints"} <= by_id.keys()
    assert by_id["teleop"]["simCapable"] is True
    assert by_id["teleop"]["isOperation"] is True

    argv = build_argv("teleop", {"sim": True, "cart_only": False})
    assert argv == ["--sim", "true", "--cart_only", "false"]
    assert _truthy("true") and not _truthy("off")
    assert _format_value([1, 2]) == "[1, 2]"
    assert _format_value(None) is None


def test_fastapi_read_only_routes_without_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        app_module, "SettingsStore", lambda: SettingsStore(tmp_path / "settings.json")
    )
    app = app_module.create_app(static_dir=tmp_path / "missing-dist")
    routes = {route.path for route in app.routes}
    assert {"/api/info", "/api/commands", "/api/settings", "/api/op/start"} <= routes

    with TestClient(app) as client:
        response = client.get("/__accept")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        command_response = client.get("/api/commands")
        assert command_response.status_code == 200
        assert any(item["id"] == "teleop" for item in command_response.json())


def test_fastapi_control_surface_and_error_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRobot:
        def __init__(self, left, right, **kwargs) -> None:
            self._channels = (left, right)
            self._state = "disconnected"

        def status(self):
            return {
                "state": self._state,
                "connected": self._state == "connected",
                "motors": [],
                "faults": [],
            }

        def channels(self):
            return self._channels

        def set_channels(self, left, right) -> None:
            self._channels = (left, right)

        def connect(self):
            self._state = "connected"
            return self.status()

        def disconnect(self):
            self._state = "disconnected"
            return self.status()

        def motor_details(self, arm, joint):
            if joint == "UNKNOWN":
                raise KeyError(joint)
            return {"arm": arm, "joint": joint, "status": "OK"}

        def motor_faults(self):
            return []

        def release(self) -> None:
            self._state = "busy"

        def reacquire(self) -> None:
            self._state = "connected"

        def shutdown(self) -> None:
            self._state = "disconnected"

    class FakeUpdater:
        version = "1.0.0"
        commit = "abc"
        release_install = False

        def __init__(self, is_idle) -> None:
            self.is_idle = is_idle
            self.provisioned = False

        def ensure_provisioned(self) -> None:
            self.provisioned = True

        async def status(self, *, force=False):
            return {
                "enabled": False,
                "version": self.version,
                "remoteVersion": None,
                "updateAvailable": False,
                "idle": self.is_idle(),
                "state": "idle",
                "phase": None,
                "error": None,
            }

        def start(self):
            return False, "not a release install"

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(
        app_module, "SettingsStore", lambda: SettingsStore(settings_path)
    )
    monkeypatch.setattr(app_module, "RobotLink", FakeRobot)
    monkeypatch.setattr(app_module, "SelfUpdater", FakeUpdater)
    monkeypatch.setattr(
        app_module, "_list_can_interfaces", lambda: [{"name": "can0", "up": True}]
    )
    monkeypatch.setattr(
        app_module,
        "_detect_cameras",
        lambda: {
            "devices": [{"serial": 7, "model": "ZED X", "kind": "stereo"}],
            "error": None,
        },
    )
    monkeypatch.setattr(
        app_module.adb,
        "status",
        lambda: SimpleNamespace(
            installed=True,
            serial="quest",
            state="device",
            reverse_active=True,
            ready=True,
        ),
    )
    monkeypatch.setattr(app_module.adb, "connect", app_module.adb.status)
    monkeypatch.setattr(
        app_module.adb, "set_proximity_disabled", lambda disabled: (True, None)
    )

    import almond_axol.recording.datasets as datasets_module
    import almond_axol.zed as zed_module
    import almond_axol.zed.snapshot as snapshot_module

    monkeypatch.setattr(datasets_module, "list_datasets", lambda base=None: [])
    monkeypatch.setattr(snapshot_module, "snapshot_jpeg", lambda serial: b"jpeg")
    monkeypatch.setattr(zed_module, "restart_zed_daemon", lambda: None)

    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>app</html>")
    (static / "assets" / "bundle.js").write_text("console.log('ok')")
    app = app_module.create_app(static_dir=static)

    with TestClient(app) as client:
        assert client.get("/api/info").json()["version"] == "1.0.0"
        assert client.get("/api/update/status?refresh=true").json()["idle"] is True
        assert client.post("/api/update/start").status_code == 409

        assert client.get("/api/robot/status").json()["state"] == "disconnected"
        assert (
            client.post(
                "/api/robot/connect",
                json={"channelsSet": True, "leftChannel": None, "rightChannel": None},
            ).status_code
            == 400
        )
        connected = client.post(
            "/api/robot/connect",
            json={"channelsSet": True, "leftChannel": "can0", "rightChannel": None},
        )
        assert connected.json()["state"] == "connected"
        assert client.post("/api/robot/disconnect").json()["state"] == "disconnected"
        assert (
            client.get("/api/can/interfaces").json()["interfaces"][0]["name"] == "can0"
        )
        assert client.get("/api/robot/motors/left/ELBOW").status_code == 200
        assert client.get("/api/robot/motors/left/UNKNOWN").status_code == 404

        assert client.get("/api/telemetry").json()["state"] == "disconnected"
        assert client.get("/api/telemetry/history?seconds=1&max_frames=5").json() == {
            "frames": []
        }
        with client.websocket_connect("/api/telemetry/ws") as websocket:
            assert websocket.receive_json()["type"] == "hello"

        assert (
            client.post("/api/diagnostics/run", json={"command": "missing"}).status_code
            == 400
        )
        assert client.get("/api/diagnostics/runs").json() == {"runs": []}
        assert client.delete("/api/diagnostics/runs").json() == {"removed": 0}
        assert client.get("/api/diagnostics/runs/missing").status_code == 404
        assert client.get("/api/cameras/detect").json()["devices"][0]["serial"] == 7
        preview = client.get("/api/cameras/preview/7")
        assert preview.content == b"jpeg"
        assert preview.headers["cache-control"] == "no-store"
        assert client.post("/api/cameras/restart-daemon").json()["ok"] is True

        assert client.get("/api/settings").status_code == 200
        assert (
            client.put(
                "/api/settings", json={"values": {"robot.has_gripper": False}}
            ).json()["values"]["robot.has_gripper"]
            is False
        )
        assert (
            client.put("/api/settings", json={"values": {"not.real": True}}).status_code
            == 400
        )
        assert client.get("/api/datasets").json() == {"datasets": []}
        assert client.get("/api/urdf/not-real.stl").status_code == 404

        assert client.get("/api/usb/status").json()["ready"] is True
        assert client.post("/api/usb/connect").json()["serial"] == "quest"
        assert client.post("/api/usb/proximity", json={"disabled": True}).json() == {
            "ok": True
        }
        assert client.get("/api/op/status").json()["running"] is False
        assert client.post("/api/op/start", json={"op": "missing"}).status_code == 400
        assert client.post("/api/op/stop").status_code == 404
        assert (
            client.post("/api/op/episode", json={"command": "save"}).status_code == 409
        )
        assert client.get("/api/sessions").json() == []
        assert client.post("/api/run", json={"command": "missing"}).status_code == 400
        assert client.post("/api/sessions/missing/stop").status_code == 404
        assert (
            client.post("/api/sessions/missing/input", json={"line": ""}).status_code
            == 404
        )
        assert client.get("/api/sessions/missing/log").status_code == 404
        with client.websocket_connect("/api/sessions/missing/logs") as websocket:
            assert websocket.receive_json()["type"] == "error"

        assert (
            client.get("/assets/bundle.js")
            .headers["cache-control"]
            .startswith("public")
        )
        assert client.get("/some/client/route").text == "<html>app</html>"
        assert client.get("/api/not-real").status_code == 404
