from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np

from almond_axol.cli import replay_dataset
from almond_axol.cli.replay_dataset import _interpolate_action_values
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.mantis.smoothing import rotvec_to_quat_xyzw
from almond_axol.robot.base import (
    HardwareCleanupError,
    is_hardware_cleanup_uncertain,
)


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

    def test_session_error_is_preserved_and_marked_when_ik_cleanup_fails(
        self,
    ) -> None:
        primary = ValueError("playback failed")
        reset_failure = RuntimeError("IK child still alive")

        replay_dataset._finish_replay_cleanup(  # noqa: SLF001
            session_error=primary,
            playback_failure=None,
            disconnect_failure=None,
            reset_failure=reset_failure,
        )

        self.assertTrue(is_hardware_cleanup_uncertain(primary))
        self.assertTrue(any("IK reset worker" in note for note in primary.__notes__))

    def test_ik_cleanup_failure_is_hardware_cleanup_error_on_clean_exit(
        self,
    ) -> None:
        reset_failure = RuntimeError("IK child still alive")

        with self.assertRaisesRegex(
            HardwareCleanupError, "background ownership is uncertain"
        ) as raised:
            replay_dataset._finish_replay_cleanup(  # noqa: SLF001
                session_error=None,
                playback_failure=None,
                disconnect_failure=None,
                reset_failure=reset_failure,
            )

        self.assertIs(raised.exception.__cause__, reset_failure)

    def test_live_playback_future_blocks_robot_disconnect_until_exit_proof(
        self,
    ) -> None:
        playback_done = threading.Event()
        disconnect = mock.Mock()

        stopped, failure = replay_dataset._wait_for_replay_exit(  # noqa: SLF001
            playback_done,
            timeout=0.01,
        )

        self.assertFalse(stopped)
        self.assertIsInstance(failure, HardwareCleanupError)
        self.assertIsNone(
            replay_dataset._cleanup_replay_robot(  # noqa: SLF001
                playback_stopped=stopped,
                cleanup=disconnect,
            )
        )
        disconnect.assert_not_called()

        playback_done.set()
        stopped, failure = replay_dataset._wait_for_replay_exit(  # noqa: SLF001
            playback_done,
            timeout=0.01,
        )
        self.assertTrue(stopped)
        self.assertIsNone(failure)
        replay_dataset._cleanup_replay_robot(  # noqa: SLF001
            playback_stopped=stopped,
            cleanup=disconnect,
        )
        disconnect.assert_called_once_with()

    def test_live_playback_cleanup_error_is_propagated_as_hardware_uncertainty(
        self,
    ) -> None:
        playback_failure = HardwareCleanupError("playback still owns the robot")
        reset_failure = RuntimeError("IK reset child still alive")

        with self.assertRaises(HardwareCleanupError) as raised:
            replay_dataset._finish_replay_cleanup(  # noqa: SLF001
                session_error=None,
                playback_failure=playback_failure,
                disconnect_failure=None,
                reset_failure=reset_failure,
            )

        self.assertIs(raised.exception, playback_failure)
        self.assertTrue(
            any("IK reset worker" in note for note in playback_failure.__notes__)
        )

    def test_session_error_is_marked_when_playback_exit_is_unproved(self) -> None:
        primary = ValueError("replay command failed")
        playback_failure = HardwareCleanupError("playback still owns the robot")

        replay_dataset._finish_replay_cleanup(  # noqa: SLF001
            session_error=primary,
            playback_failure=playback_failure,
            disconnect_failure=None,
            reset_failure=None,
        )

        self.assertTrue(is_hardware_cleanup_uncertain(primary))
        self.assertTrue(any("replay playback" in note for note in primary.__notes__))


if __name__ == "__main__":
    unittest.main()
