from __future__ import annotations

import unittest

import numpy as np

from almond_axol.cli.replay_dataset import _interpolate_action_values
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.mantis.smoothing import rotvec_to_quat_xyzw


_ACTION_NAMES = [
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "left_gripper.pos",
]


def _rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_matrix(rotvec_to_quat_xyzw(rotvec))


class ReplayInterpolationTests(unittest.TestCase):
    def test_cartesian_rotation_crossing_pi_follows_short_path(self) -> None:
        epsilon = 1e-3
        base = np.array([0, 0, 0, 0, 0, np.pi - epsilon, 0], dtype=np.float64)
        nxt = np.array([2, 4, 6, 0, 0, -np.pi + epsilon, 1], dtype=np.float64)

        midpoint = _interpolate_action_values(base, nxt, 0.5, _ACTION_NAMES)

        np.testing.assert_allclose(midpoint[:3], [1, 2, 3])
        self.assertAlmostEqual(float(midpoint[6]), 0.5)
        np.testing.assert_allclose(
            _rotation_matrix(midpoint[3:6]),
            _rotation_matrix(np.array([0.0, 0.0, np.pi])),
            atol=1e-6,
        )

    def test_cartesian_rotation_uses_ordinary_shortest_arc(self) -> None:
        base = np.zeros(7, dtype=np.float64)
        nxt = np.array([0, 0, 0, 0, np.pi / 2, 0, 0], dtype=np.float64)

        midpoint = _interpolate_action_values(base, nxt, 0.5, _ACTION_NAMES)

        np.testing.assert_allclose(
            _rotation_matrix(midpoint[3:6]),
            _rotation_matrix(np.array([0.0, np.pi / 4, 0.0])),
            atol=1e-6,
        )

    def test_joint_actions_keep_componentwise_interpolation(self) -> None:
        base = np.array([1.0, -2.0, 0.0])
        nxt = np.array([3.0, 2.0, 1.0])

        midpoint = _interpolate_action_values(
            base,
            nxt,
            0.25,
            ["left_shoulder_1.pos", "right_wrist_2.pos", "left_gripper.pos"],
        )

        np.testing.assert_allclose(midpoint, [1.5, -1.0, 0.25])


if __name__ == "__main__":
    unittest.main()
