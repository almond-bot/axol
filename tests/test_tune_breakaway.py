"""The pure parts of ``axol tune.breakaway``: the escalation schedule, the
ramp shape, the trim search, the pose clamping, and the friction/bias split
the hardware loop feeds its measurements into."""

from __future__ import annotations

import math
import unittest

from almond_axol.cli.tune import breakaway as ba
from almond_axol.constants import Joint


class ScheduleTest(unittest.TestCase):
    def test_peaks_scale_with_fc_and_stop_at_the_cap(self) -> None:
        peaks = ba.peak_schedule(1.3, 3.0)
        self.assertEqual(peaks[0], 0.5 * 1.3)
        self.assertEqual(peaks[-1], 3.0 * 1.3)
        self.assertEqual(peaks, sorted(peaks))
        self.assertTrue(all(p <= 3.0 * 1.3 + 1e-9 for p in peaks))

    def test_low_friction_joints_get_the_floor(self) -> None:
        # A Damiao wrist with fc = 0.1 Nm would otherwise ramp to 0.05 Nm.
        peaks = ba.peak_schedule(0.1, 2.0)
        self.assertEqual(peaks[0], 0.5 * ba._FC_FLOOR_NM)
        self.assertEqual(peaks[-1], 2.0 * ba._FC_FLOOR_NM)

    def test_triangle_rises_to_one_and_returns_to_zero(self) -> None:
        self.assertEqual(ba.triangle(0.0, 4.0), 0.0)
        self.assertAlmostEqual(ba.triangle(1.0, 4.0), 0.5)
        self.assertAlmostEqual(ba.triangle(2.0, 4.0), 1.0)
        self.assertAlmostEqual(ba.triangle(3.0, 4.0), 0.5)
        self.assertEqual(ba.triangle(4.0, 4.0), 0.0)
        self.assertEqual(ba.triangle(5.0, 4.0), 0.0)


class SplitTest(unittest.TestCase):
    def test_symmetric_releases_are_pure_friction(self) -> None:
        f_static, bias = ba.split_breakaway([3.0, 3.1], [2.9, 3.0])
        self.assertAlmostEqual(f_static, 3.0)
        self.assertAlmostEqual(bias, -0.05)

    def test_a_feedforward_residual_shows_up_as_bias(self) -> None:
        # Hold torque over-supplies +0.4 Nm: the + direction releases 0.4
        # early, the - direction 0.4 late.
        f_static, bias = ba.split_breakaway([2.6], [3.4])
        self.assertAlmostEqual(f_static, 3.0)
        self.assertAlmostEqual(bias, 0.4)

    def test_predicted_stair_is_the_uncovered_torque_over_kp(self) -> None:
        # 3.0 Nm breakaway, 1.3 Nm fc, kp 250: (1.7 / 250) rad ≈ 0.39°.
        self.assertAlmostEqual(
            ba.predicted_stair_deg(3.0, 1.3, 250.0), 0.3896, places=3
        )
        self.assertEqual(ba.predicted_stair_deg(1.0, 1.3, 250.0), 0.0)
        self.assertTrue(math.isnan(ba.predicted_stair_deg(3.0, 1.3, 0.0)))


class TrimSearchTest(unittest.TestCase):
    def test_converges_on_a_standing_hold(self) -> None:
        search = ba.TrimSearch(step=0.2, max_trim=4.0)
        # The joint drifts + (torque too high) until the trim comes down.
        self.assertAlmostEqual(search.update(math.radians(0.5)), -0.2)
        self.assertAlmostEqual(search.update(math.radians(0.3)), -0.4)
        # Overshoot: drift flips sign, so the step halves.
        self.assertAlmostEqual(search.update(math.radians(-0.2)), -0.3)
        self.assertFalse(search.done)
        search.update(0.0)
        self.assertTrue(search.done)
        self.assertFalse(search.failed)

    def test_gives_up_when_the_trim_runs_away(self) -> None:
        search = ba.TrimSearch(step=1.0, max_trim=2.5)
        for _ in range(3):
            search.update(math.radians(1.0))
        self.assertTrue(search.failed)


class PosesTest(unittest.TestCase):
    def test_default_pose_is_rest(self) -> None:
        self.assertEqual(ba._probe_poses(Joint.SHOULDER_1, False, None), [0.0])

    def test_base_collision_joint_is_pushed_outboard(self) -> None:
        # Right shoulder_2's outboard side is +; the rest pose sits on the
        # boundary, so the probe moves 5° out where a release cannot cross it.
        (pose,) = ba._probe_poses(Joint.SHOULDER_2, False, [0.0])
        self.assertAlmostEqual(pose, ba._BOUNDARY_MARGIN)
        (pose_l,) = ba._probe_poses(Joint.SHOULDER_2, True, [0.0])
        self.assertAlmostEqual(pose_l, -ba._BOUNDARY_MARGIN)

    def test_poses_are_kept_inside_the_safe_range(self) -> None:
        poses = ba._probe_poses(Joint.SHOULDER_1, False, [-720.0, 720.0])
        lo, hi = ba.safe_limits(Joint.SHOULDER_1, False)
        self.assertGreater(poses[0], lo)
        self.assertLess(poses[1], hi)


if __name__ == "__main__":
    unittest.main()
