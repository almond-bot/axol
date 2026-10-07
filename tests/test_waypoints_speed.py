"""Waypoint playback speed override (``speed_scale``, the panel's slider).

The scale must slow *every* move a session makes — straight-line legs in
both translation and rotation, the joint-space approach, and the return to
rest — and can never speed playback up past the configured speeds.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from almond_axol.cli import waypoints as wp_cli
from almond_axol.serve.commands import CommandDef, build_argv, command_specs
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.waypoints import Waypoint, WaypointSet

_SOLVER = SimpleNamespace(
    num_joints=14, left_indices=list(range(7)), right_indices=list(range(7, 14))
)


def _path(tmp_path: Path) -> Path:
    file = tmp_path / "path.json"
    pose = np.zeros(8, dtype=np.float32)
    WaypointSet([Waypoint(pose, pose), Waypoint(pose + 0.1, pose)]).save(file)
    return file


def _plan_kwargs(cfg: wp_cli.WaypointsCmdConfig) -> tuple[dict, dict]:
    """Plan a two-waypoint path; return the approach and leg planner kwargs."""
    calls: dict[str, dict] = {}

    def approach(*_args, **kwargs):
        calls["approach"] = kwargs
        return [np.zeros(14, dtype=np.float32)]

    def leg(*_args, **kwargs):
        calls["leg"] = kwargs
        return [np.zeros(14, dtype=np.float32)]

    with (
        patch.object(wp_cli, "_SolverHandle"),
        patch(
            "almond_axol.teleop.trajectory.plan_collision_aware_trajectory", approach
        ),
        patch("almond_axol.kinematics.path.plan_linear_segment", leg),
    ):
        session = wp_cli._Session(
            cfg, robot=None, control=None, stop_event=threading.Event()
        )  # type: ignore[arg-type]
        session._q_start = np.zeros(14, dtype=np.float32)
        session._plan(_SOLVER)
    return calls["approach"], calls["leg"]


def test_speed_scale_slows_every_planned_move(tmp_path: Path) -> None:
    file = _path(tmp_path)
    rest = VRTeleopConfig()
    full_approach, full_leg = _plan_kwargs(wp_cli.WaypointsCmdConfig(file=str(file)))
    slow_approach, slow_leg = _plan_kwargs(
        wp_cli.WaypointsCmdConfig(file=str(file), speed_scale=0.25)
    )

    assert full_leg["speed"] == pytest.approx(0.25)
    assert full_leg["ang_speed"] == pytest.approx(1.2)
    assert full_approach["speed"] == pytest.approx(rest.reset_speed)

    assert slow_leg["speed"] == pytest.approx(full_leg["speed"] * 0.25)
    assert slow_leg["ang_speed"] == pytest.approx(full_leg["ang_speed"] * 0.25)
    assert slow_approach["speed"] == pytest.approx(rest.reset_speed * 0.25)
    assert slow_approach["min_duration"] == pytest.approx(
        rest.reset_min_duration / 0.25
    )


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5, math.nan, math.inf])
def test_out_of_range_speed_scale_is_refused_before_anything_moves(
    tmp_path: Path, scale: float
) -> None:
    cfg = wp_cli.WaypointsCmdConfig(
        file=str(_path(tmp_path)), sim=True, speed_scale=scale
    )
    with (
        patch.object(wp_cli, "_session") as session,
        pytest.raises(ValueError, match="speed_scale"),
    ):
        wp_cli._run(cfg, threading.Event())
    session.assert_not_called()


def test_panel_offers_speed_scale_as_a_slider() -> None:
    spec = next(s for s in command_specs() if s["id"] == "waypoints")
    assert "speed_scale" in spec["perRunFields"]
    ui = spec["fieldUi"]["speed_scale"]
    assert ui["widget"] == "slider"
    assert 0.0 < ui["min"] < ui["max"] <= 1.0
    assert build_argv("waypoints", {"speed_scale": 0.4}) == ["--speed_scale", "0.4"]


def test_field_ui_must_name_a_per_run_field() -> None:
    with pytest.raises(ValueError, match="field_ui"):
        CommandDef(
            "x",
            "x",
            "X",
            "",
            "Operate",
            "draccus",
            lambda: None,
            per_run_fields=("a",),
            field_ui={"b": {"widget": "slider"}},
        )
