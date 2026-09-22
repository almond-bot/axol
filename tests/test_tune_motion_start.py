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


class RetimeMeasurementsTest(unittest.TestCase):
    """Cache reads are put back on the command clock before scoring."""

    def test_a_sawtooth_cache_age_is_removed(self) -> None:
        from almond_axol.cli.tune.motion import retime_measurements

        # A 240 Hz log of a joint moving at 1 rad/s whose cache was refreshed
        # at 400 Hz: each sample is 0..2.5 ms old, in a repeating pattern.
        t = np.arange(0, 1.0, 1 / 240)
        age = (np.arange(len(t)) % 5) * 0.0005
        offsets = np.zeros((len(t), 14))
        offsets[:, 7] = -age
        actual = np.full((len(t), 14), np.nan, dtype=np.float32)
        actual[:, 7] = (t - age).astype(np.float32)  # the position when sampled
        torque = np.full_like(actual, np.nan)
        fixed, _ = retime_measurements(t, offsets, actual, torque)
        # Raw error against the command clock is the sawtooth; re-timed it is ~0.
        raw = np.sqrt(np.mean((actual[:, 7] - t) ** 2))
        new = np.sqrt(np.mean((fixed[5:-5, 7] - t[5:-5]) ** 2))
        self.assertGreater(raw, 5e-4)
        self.assertLess(new, 2e-5)
        # An absent arm's NaN columns and unknown offsets pass through untouched.
        self.assertTrue(np.all(np.isnan(fixed[:, 0])))

    def test_unknown_offsets_leave_the_series_alone(self) -> None:
        from almond_axol.cli.tune.motion import retime_measurements

        t = np.arange(0, 0.1, 1 / 240)
        actual = np.random.default_rng(0).normal(size=(len(t), 14)).astype(np.float32)
        torque = actual.copy()
        a, q = retime_measurements(t, np.zeros((len(t), 14)), actual, torque)
        np.testing.assert_array_equal(a, actual)
        np.testing.assert_array_equal(q, torque)
