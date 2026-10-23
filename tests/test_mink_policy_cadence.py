"""Policy cadence reaches the real Mink speed gate before any device opens."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from almond_axol.lerobot.robot.config_axol import AxolRobotConfig
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.policy.mink_ik import MinkIKConfig


def config(tmp_path):
    return AxolRobotConfig(
        cameras={},
        calibration_dir=tmp_path,
        action_space="cartesian",
        cartesian_controller="mink",
    )


@pytest.mark.parametrize("rate", [15, 30, 60])
def test_rate_reaches_real_tracker_and_prepare_reuses_its_configuration(
    tmp_path, monkeypatch, rate
):
    connect = Mock(side_effect=AssertionError("hardware connect is forbidden"))
    monkeypatch.setattr(AxolRobot, "connect", connect)
    robot = AxolRobot(config(tmp_path), mink_solve_hz=rate)
    robot.prepare_cartesian_actions()
    solver = robot._ik
    assert solver.config.mink_solve_hz == rate
    assert solver.tracker._config.mink_solve_hz == rate
    robot.prepare_cartesian_actions()
    assert robot._ik is solver
    # Exercise the numerical solver rather than only inspecting an unused config.
    q = np.zeros(14, dtype=np.float32)
    left, right = solver.fk(q)
    result = solver.solve(q, left, right)
    assert result.shape == (14,)
    assert np.isfinite(result).all()
    connect.assert_not_called()


@pytest.mark.parametrize("override", [None, 60])
def test_custom_config_is_copied_and_only_explicit_rate_is_overridden(
    tmp_path, override
):
    custom = MinkIKConfig(
        mink_solve_hz=15,
        mink_posture_speed_gate=0.123,
        mink_posture_gate_tau=0.456,
        max_joint_delta=0.12,
    )
    options = {} if override is None else {"mink_solve_hz": override}
    robot = AxolRobot(config(tmp_path), ik_config=custom, **options)
    copied = robot._ik_config
    assert copied is not custom
    assert copied.mink_solve_hz == (15 if override is None else override)
    assert copied.mink_posture_speed_gate == custom.mink_posture_speed_gate
    assert copied.mink_posture_gate_tau == custom.mink_posture_gate_tau
    assert copied.max_joint_delta == custom.max_joint_delta
    assert custom.mink_solve_hz == 15
    custom.mink_solve_hz = 120
    assert copied.mink_solve_hz == (15 if override is None else override)


def test_omitting_rate_preserves_frozen_default(tmp_path):
    robot = AxolRobot(config(tmp_path))
    assert robot._ik_config == MinkIKConfig()
    assert robot._ik_config.mink_solve_hz == 30


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, -np.inf, True, "60"])
def test_invalid_rate_fails_before_camera_or_solver_creation(
    tmp_path, monkeypatch, rate
):
    build = Mock(side_effect=AssertionError("camera construction is forbidden"))
    monkeypatch.setattr(AxolRobot, "_build_cameras", build)
    with pytest.raises(ValueError, match="finite and positive"):
        AxolRobot(config(tmp_path), mink_solve_hz=rate)
    build.assert_not_called()


def test_invalid_explicit_config_rate_is_rejected_before_device_work(
    tmp_path, monkeypatch
):
    build = Mock(side_effect=AssertionError("camera construction is forbidden"))
    monkeypatch.setattr(AxolRobot, "_build_cameras", build)
    with pytest.raises(ValueError, match="finite and positive"):
        AxolRobot(config(tmp_path), ik_config=MinkIKConfig(mink_solve_hz=0))
    build.assert_not_called()


def test_mink_rate_is_not_silently_ignored_by_another_backend(tmp_path):
    other = config(tmp_path)
    other.cartesian_controller = "jax"
    with pytest.raises(ValueError, match="requires cartesian_controller"):
        AxolRobot(other, mink_solve_hz=60)
