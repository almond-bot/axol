from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from almond_axol.serve import app as app_module
from almond_axol.serve.commands import _format_value, build_argv, command_specs
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
        values={"axol.has_gripper": False, "robot.left_channel": "can9"},
        cameras={"serials": {"overhead": "123"}},
    )
    assert snapshot["values"]["axol.has_gripper"] is False
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

    argv = build_argv("teleop", {"sim": True, "arms": False})
    assert argv == ["--sim", "true", "--arms", "false"]
    # Keys the schema does not know are dropped, never forwarded.
    assert build_argv("teleop", {"not_a_field": "x"}) == []
    assert _format_value([1, 2]) == "[1, 2]"
    assert _format_value(True) == "true"
    assert _format_value("  ") is None
    assert _format_value(None) is None
    with pytest.raises(KeyError):
        build_argv("missing", {})


def test_fastapi_read_only_routes_without_hardware(tmp_path: Path) -> None:
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html>app</html>")
    (static / "assets" / "bundle.js").write_text("console.log('ok')")
    app = app_module.create_app(static_dir=static)
    routes = {route.path for route in app.routes}
    assert {"/api/info", "/api/commands", "/api/settings", "/api/op/start"} <= routes

    with TestClient(app) as client:
        command_response = client.get("/api/commands")
        assert command_response.status_code == 200
        assert any(item["id"] == "teleop" for item in command_response.json())
        assert client.get("/api/settings").status_code == 200
        assert client.post("/api/op/start", json={"op": "missing"}).status_code == 400
        assert client.post("/api/run", json={"command": "missing"}).status_code == 400
        assert client.get("/api/sessions/missing/log").status_code == 404
        assert (
            client.get("/assets/bundle.js")
            .headers["cache-control"]
            .startswith("public")
        )
        assert client.get("/some/client/route").text == "<html>app</html>"
        assert client.get("/api/not-real").status_code == 404
