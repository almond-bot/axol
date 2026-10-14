"""run-policy's episode start: seeded execution filter and first-action check.

The scenario: the arms rest at REST_* and a policy whose inputs are out of
distribution returns a first action that puts the right elbow 0.6 rad away.
With an unseeded execution filter that target went out as one step, was
dropped by ``max_step_rad``, and every later target was dropped the same way
until one slipped under the limit as a single jump.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.cli.run_policy import (
    _build_axol_robot_client,
    _first_action_offset,
    _measured_joint_action,
)
from almond_axol.lerobot.inference_patch import import_robot_client_preserving_logging

JOINTS = (
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "gripper",
)
SCHEMA = tuple(f"{side}_{j}.pos" for side in ("left", "right") for j in JOINTS)
RIGHT_ELBOW = SCHEMA.index("right_elbow.pos")

# Measured rest pose (Joint order, per arm).
REST_LEFT = np.array([-0.22, 0.0, 0.0, 0.45, 0.0, 0.0, -0.23, 1.0])
REST_RIGHT = np.array([0.22, 0.0, 0.0, -0.45, 0.0, 0.0, 0.23, 1.0])
# A first action 0.6 rad off on the right elbow, with smaller offsets elsewhere
# and a gripper target the check must ignore.
FIRST_TARGET = np.concatenate([REST_LEFT, REST_RIGHT]).astype(np.float32)
FIRST_TARGET[[0, 3, 8, 14]] += np.array([-0.01, 0.03, -0.25, -0.1], dtype=np.float32)
FIRST_TARGET[RIGHT_ELBOW] = -1.05
FIRST_TARGET[SCHEMA.index("right_gripper.pos")] = 0.4


def _robot() -> SimpleNamespace:
    return SimpleNamespace(
        action_features=dict.fromkeys(SCHEMA, float),
        observation_features=dict.fromkeys(SCHEMA, float),
        config=SimpleNamespace(observe_cartesian=False),
        cartesian_actions=False,
        positions=(REST_LEFT.copy(), REST_RIGHT.copy()),
        connect=mock.Mock(),
        send_action=mock.Mock(side_effect=lambda action: action),
        torque_residuals=lambda: (np.zeros(7), np.zeros(7)),
    )


def _client(robot: SimpleNamespace, max_first_action_offset_rad: float):  # type: ignore[no-untyped-def]
    import_robot_client_preserving_logging()
    config = SimpleNamespace(
        fps=30,
        environment_dt=1 / 30,
        server_address="127.0.0.1:1",
        policy_type="act",
        pretrained_name_or_path="unused",
        actions_per_chunk=50,
        policy_device="cpu",
        client_device="cpu",
    )
    client = _build_axol_robot_client(
        config=config,
        robot=robot,
        publisher=None,
        max_first_action_offset_rad=max_first_action_offset_rad,
    )
    # Schema confirmation and the server reset are gRPC handshakes, covered
    # by test_action_schema; this exercises only the local dispatch path.
    client._action_schema_confirmed = True
    client._reset_server = mock.Mock()
    client.reset_episode_state()
    return client


class StartCheckTest(unittest.TestCase):
    def test_distant_first_action_is_refused_before_anything_moves(self) -> None:
        robot = _robot()
        client = _client(robot, 0.15)

        self.assertIsNone(client._shape_and_send(FIRST_TARGET))

        joint, offset = client.start_rejected
        self.assertEqual(joint, "right_elbow.pos")
        self.assertAlmostEqual(offset, 0.6, places=5)
        self.assertTrue(client.shutdown_event.is_set())
        # Later targets in the episode stay unsent, even ones under the limit.
        near = FIRST_TARGET.copy()
        near[RIGHT_ELBOW] = REST_RIGHT[3] - 0.05
        self.assertIsNone(client._shape_and_send(near))
        robot.send_action.assert_not_called()

    def test_reset_rearms_the_check_for_the_next_episode(self) -> None:
        robot = _robot()
        client = _client(robot, 0.15)
        client._shape_and_send(FIRST_TARGET)
        self.assertIsNotNone(client.start_rejected)

        client.reset_episode_state()
        healthy = np.concatenate([REST_LEFT, REST_RIGHT]).astype(np.float32)
        healthy[RIGHT_ELBOW] -= 0.05

        self.assertIsNone(client.start_rejected)
        self.assertIsNotNone(client._shape_and_send(healthy))
        robot.send_action.assert_called_once()

    def test_first_target_within_limit_ramps_from_the_measured_pose(self) -> None:
        robot = _robot()
        client = _client(robot, 0.15)
        target = np.concatenate([REST_LEFT, REST_RIGHT]).astype(np.float32)
        target[RIGHT_ELBOW] -= 0.12
        target[SCHEMA.index("right_gripper.pos")] = 0.2

        sent = client._shape_and_send(target)

        # Seeded at rest with zero velocity, the filter's first step is one
        # tick of acceleration, not the policy's whole 0.12 rad offset.
        step = abs(sent["right_elbow.pos"] - REST_RIGHT[3])
        self.assertGreater(step, 0.0)
        self.assertLess(step, 0.03)
        # Grippers bypass the filter.
        self.assertAlmostEqual(sent["right_gripper.pos"], 0.2, places=6)

    def test_disabled_check_still_ramps_a_distant_first_target(self) -> None:
        robot = _robot()
        client = _client(robot, 0.0)

        first = client._shape_and_send(FIRST_TARGET)

        self.assertIsNone(client.start_rejected)
        # The 0.6 rad target no longer goes out as a single step.
        self.assertLess(abs(first["right_elbow.pos"] - REST_RIGHT[3]), 0.03)
        for _ in range(90):
            last = client._shape_and_send(FIRST_TARGET)
        self.assertAlmostEqual(
            last["right_elbow.pos"], float(FIRST_TARGET[RIGHT_ELBOW]), places=4
        )

    def test_measured_pose_follows_the_gripperless_schema(self) -> None:
        names = tuple(n for n in SCHEMA if not n.endswith("gripper.pos"))
        measured = _measured_joint_action(names, REST_LEFT, REST_RIGHT)

        np.testing.assert_allclose(
            measured, np.concatenate([REST_LEFT[:7], REST_RIGHT[:7]]), rtol=0, atol=1e-6
        )
        joint, offset = _first_action_offset(
            measured + 0.01 * np.arange(14), measured, list(range(14)), names
        )
        self.assertEqual(joint, "right_wrist_3.pos")
        self.assertAlmostEqual(offset, 0.13, places=5)


if __name__ == "__main__":
    unittest.main()
