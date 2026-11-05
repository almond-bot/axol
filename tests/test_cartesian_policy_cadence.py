"""Exercise real Cartesian shaping with a synthetic solver and no device."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from almond_axol.lerobot.robot import robot_axol as module
from almond_axol.lerobot.robot.config_axol import AxolRobotConfig


def robot(tmp_path, rate):
    value = module.AxolRobot(
        AxolRobotConfig(cameras={}, calibration_dir=tmp_path, action_space="cartesian"),
        cartesian_action_hz=rate,
    )
    value._axol = SimpleNamespace(
        left=SimpleNamespace(positions=np.zeros(8)),
        right=SimpleNamespace(positions=np.zeros(8)),
    )
    value._ensure_ik = lambda: SimpleNamespace(
        num_joints=14,
        left_indices=list(range(7)),
        right_indices=list(range(7, 14)),
        ik=lambda *args, **kwargs: np.full(14, 0.5),
    )
    action = dict.fromkeys((*module._LEFT_EE_KEYS, *module._RIGHT_EE_KEYS), 0.0)
    action.update({module._LEFT_GRIPPER_KEY: 0.2, module._RIGHT_GRIPPER_KEY: 0.8})
    return value, action


def run(value, action, rate):
    result = []
    for i in range(rate):
        with patch.object(module.time, "monotonic", return_value=1 + i / rate):
            left, right = value._cartesian_action_to_targets(action)
        assert left[-1] == pytest.approx(0.2) and right[-1] == pytest.approx(0.8)
        result.append(left[0])
    return np.asarray(result)


def test_low_rate_policy_matches_equal_elapsed_high_rate_motion(tmp_path):
    slow, action = robot(tmp_path, 12)
    fast, _ = robot(tmp_path, 60)
    a, b = run(slow, action, 12), run(fast, action, 60)
    np.testing.assert_allclose(a, b[4::5], atol=2e-5)
    default, _ = robot(tmp_path, None)
    c = run(default, action, 12)
    assert a[-1] > c[-1] + 0.05


def test_long_stall_reseeds_measured_state_without_spending_gap(tmp_path):
    value, action = robot(tmp_path, 12)
    run(value, action, 12)
    value._axol.left.positions[:7] = 0.1
    fresh, _ = robot(tmp_path, 12)
    fresh._axol.left.positions[:7] = 0.1
    with patch.object(module.time, "monotonic", return_value=100):
        a = value._cartesian_action_to_targets(action)
        b = fresh._cartesian_action_to_targets(action)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("rate", [True, 0, -1, 3, float("nan"), float("inf")])
def test_invalid_cadence_before_hardware(tmp_path, rate):
    with pytest.raises(ValueError, match="cartesian_action_hz"):
        robot(tmp_path, rate)


def test_repeated_same_time_call_cannot_advance_policy(tmp_path):
    value, action = robot(tmp_path, 12)
    with patch.object(module.time, "monotonic", return_value=1):
        first = value._cartesian_action_to_targets(action)
        second = value._cartesian_action_to_targets(action)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
