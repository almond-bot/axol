"""``tune.motion`` refuses to replay from a pose the approach never reached."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.cli.tune.motion import _START_POSE_TOL, start_pose_stragglers

_LEFT = np.arange(0, 7)
_RIGHT = np.arange(7, 14)
_BOTH = [("left", _LEFT), ("right", _RIGHT)]


class StartPoseStragglersTest(unittest.TestCase):
    def test_all_joints_at_start_is_clean(self) -> None:
        q = np.linspace(-1.0, 1.0, 14)
        self.assertEqual(start_pose_stragglers(q, q.copy(), _BOTH), [])
        nudged = q + 0.5 * _START_POSE_TOL
        self.assertEqual(start_pose_stragglers(nudged, q, _BOTH), [])

    def test_names_the_joint_and_reports_the_error_in_degrees(self) -> None:
        q_start = np.zeros(14)
        q_now = q_start.copy()
        q_now[7 + 3] = math.radians(23.5)  # right.elbow left at rest, 23.5° short
        q_now[2] = -math.radians(4.0)  # left.shoulder_3 a little off the other way
        out = start_pose_stragglers(q_now, q_start, _BOTH)
        self.assertEqual([n for n, _ in out], ["left.shoulder_3", "right.elbow"])
        self.assertAlmostEqual(out[1][1], 23.5, places=6)
        self.assertAlmostEqual(out[0][1], -4.0, places=6)

    def test_an_arm_left_off_is_not_judged(self) -> None:
        # ``--arms right``: the left arm reads as rest while the motion's
        # first row is elsewhere — that is not a straggler.
        q_start = np.linspace(-1.0, 1.0, 14)
        q_now = q_start.copy()
        q_now[:7] = 0.0
        q_now[7 + 3] += math.radians(22.5)
        out = start_pose_stragglers(q_now, q_start, [("right", _RIGHT)])
        self.assertEqual(out, [("right.elbow", 22.5)])


if __name__ == "__main__":
    unittest.main()
