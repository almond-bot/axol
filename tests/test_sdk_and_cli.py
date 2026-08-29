from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from almond_axol.cli import main as cli_main
from almond_axol.robot.sim import Sim
from almond_axol.waypoints import JOINT_VECTOR_LEN, Waypoint, WaypointSet


def test_sim_sdk_tracks_both_arms_without_starting_viewer() -> None:
    async def exercise() -> None:
        sim = Sim(port=0)
        left = np.linspace(0.0, 0.7, JOINT_VECTOR_LEN, dtype=np.float32)
        right = -left
        await sim.motion_control(left=left, right=right)

        actual_left, actual_right = await sim.get_positions()
        np.testing.assert_array_equal(actual_left, left)
        np.testing.assert_array_equal(actual_right, right)
        np.testing.assert_array_equal(
            sim._build_q(), np.concatenate([left[:7], right[:7]])
        )

    asyncio.run(exercise())


def test_waypoints_round_trip_atomically(tmp_path: Path) -> None:
    waypoint = Waypoint(
        left=np.arange(JOINT_VECTOR_LEN, dtype=np.float32) / 10,
        right=np.arange(JOINT_VECTOR_LEN, dtype=np.float32) / -10,
    )
    waypoints = WaypointSet([waypoint])
    path = tmp_path / "nested" / "path.json"

    waypoints.save(path)
    loaded = WaypointSet.load(path)

    assert loaded[0].label == "waypoint 1"
    np.testing.assert_allclose(loaded[0].left, waypoint.left)
    assert not path.with_suffix(".json.tmp").exists()
    assert WaypointSet.load(tmp_path / "missing.json").waypoints == []


def test_waypoint_file_rejects_bad_shape_version_and_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="left waypoint"):
        Waypoint(np.zeros(2), np.zeros(JOINT_VECTOR_LEN))

    path = tmp_path / "path.json"
    path.write_text(json.dumps({"version": 999, "waypoints": []}))
    with pytest.raises(ValueError, match="version 999"):
        WaypointSet.load(path)
    path.write_text("not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        WaypointSet.load(path)


def test_cli_help_lists_hardware_and_sim_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["axol", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "teleop" in output
    assert "can.setup" in output
    assert "serve" in output
