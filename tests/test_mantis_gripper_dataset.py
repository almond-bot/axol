"""The raw Mantis recording path preserves feedback while avoiding virtual FK."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from almond_axol.constants import Joint
from almond_axol.lerobot.robot.config_mantis import MantisRobotConfig
from almond_axol.lerobot.robot.robot_axol import (
    _LEFT_EE_KEYS,
    _LEFT_GRIPPER_KEY,
    _RIGHT_EE_KEYS,
    _RIGHT_GRIPPER_KEY,
)
from almond_axol.lerobot.robot.robot_mantis import MantisRobot
from almond_axol.mantis.relative import quat_xyzw_to_rotvec
from almond_axol.teleop.core import TCPPoseSnapshot


@pytest.mark.parametrize("torques", [False, True])
def test_gripper_only_path_matches_prior_full_rows_after_tracked_tcp_override(torques):
    robot = MantisRobot(
        MantisRobotConfig(observe_cartesian=True, observe_torques=torques)
    )
    left = np.linspace(-0.3, 0.8, 8, dtype=np.float32)
    right = np.linspace(0.5, -0.2, 8, dtype=np.float32)
    robot._axol = SimpleNamespace(
        left=SimpleNamespace(positions=left, torques=left * 0.1),
        right=SimpleNamespace(positions=right, torques=right * 0.2),
    )
    robot._fk = Mock()
    robot._fk.ee_poses.return_value = (np.arange(6), np.arange(6) + 10)
    action = {
        f"{side}_{joint.value}.pos": float(i) / 13
        for side in ("left", "right")
        for i, joint in enumerate(Joint)
    }
    original_state = robot.get_joint_observation()
    original_action = robot.action_to_dataset(action)
    assert robot._fk.ee_poses.call_count == 2
    robot._fk.ee_poses.reset_mock()
    robot._fk.ee_poses.side_effect = AssertionError("redundant virtual FK")
    raw_state = robot.get_gripper_observation()
    raw_action = robot.gripper_action_to_dataset(action)
    sample = TCPPoseSnapshot(
        pose_host_ts=123.0,
        left=(0.3, 0.4, 0.5, 0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)),
        right=(0.6, 0.7, 0.8, 0.0, 0.0, 1.0, 0.0),
    )
    for keys, pose in ((_LEFT_EE_KEYS, sample.left), (_RIGHT_EE_KEYS, sample.right)):
        tracked = dict(
            zip(keys, [*pose[:3], *quat_xyzw_to_rotvec(np.asarray(pose[3:]))])
        )
        for row in (original_state, original_action, raw_state, raw_action):
            row.update(tracked)
    assert raw_state == original_state
    assert raw_action == original_action
    assert raw_state[_LEFT_GRIPPER_KEY] == float(left[-1])
    assert raw_state[_RIGHT_GRIPPER_KEY] == float(right[-1])
    robot._fk.ee_poses.assert_not_called()


@pytest.mark.parametrize("torques", [False, True])
def test_capture_aligned_gripper_state_matches_upstream_without_fk(torques):
    from almond_axol.cli.collect_data import _row_observation

    robot = MantisRobot(
        MantisRobotConfig(observe_cartesian=True, observe_torques=torques)
    )
    captured_left = np.linspace(-0.3, 0.8, 8, dtype=np.float32)
    captured_right = np.linspace(0.5, -0.2, 8, dtype=np.float32)
    robot._axol = SimpleNamespace(
        left=SimpleNamespace(positions=np.zeros(8), torques=np.zeros(8)),
        right=SimpleNamespace(positions=np.zeros(8), torques=np.zeros(8)),
        state_nearest=Mock(
            return_value=(
                captured_left,
                captured_right,
                captured_left * 0.1,
                captured_right * 0.2,
                123.004,
            )
        ),
    )
    robot._fk = Mock()
    robot._fk.ee_poses.return_value = (np.arange(6), np.arange(6) + 10)
    expected = robot.joint_observation_at(123.0)
    assert expected is not None
    for key in (*_LEFT_EE_KEYS, *_RIGHT_EE_KEYS):
        expected.pop(key)
    robot._fk.ee_poses.reset_mock()
    robot._fk.ee_poses.side_effect = AssertionError("redundant virtual FK")
    live = robot.get_gripper_observation()
    actual = _row_observation(robot, live, 123.0)
    assert actual == expected
    assert actual[_LEFT_GRIPPER_KEY] != live[_LEFT_GRIPPER_KEY]
    robot._axol.state_nearest.assert_called_with(123.0, timeout=0.0)
    robot._fk.ee_poses.assert_not_called()

    # A telemetry sample outside the existing 10 ms bound must retain live
    # feedback rather than associating an unrelated measurement with the pose.
    assert _row_observation(robot, live, 122.0) is live
    robot._axol.state_nearest.return_value = None
    assert _row_observation(robot, live, 123.0) is live
