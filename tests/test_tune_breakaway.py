"""The breakaway probe's shape, detection and reporting math.

Hardware-free: the ramp shape, the release detector and the parameter
suggestions are pure functions, and they are what the measurement means.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.cli.tune.breakaway import (
    _ESCALATION,
    _FEEDBACK_LSB,
    _NO_FF,
    _triangle,
    detect_release,
)
from almond_axol.constants import ARM_JOINTS, Joint


class RampShapeTest(unittest.TestCase):
    def test_triangle_starts_and_ends_at_zero(self) -> None:
        # Coming back down is the safety property: past breakaway the torque
        # is already falling, so a release cannot run away.
        for n in (8, 9, 100, 961):
            tri = _triangle(n)
            self.assertEqual(len(tri), n)
            self.assertAlmostEqual(tri[0], 0.0)
            self.assertAlmostEqual(tri[-1], 0.0)
            self.assertAlmostEqual(tri.max(), 1.0)
            self.assertTrue(np.all(tri >= 0.0))
            peak = int(np.argmax(tri))
            self.assertTrue(np.all(np.diff(tri[: peak + 1]) >= -1e-12))
            self.assertTrue(np.all(np.diff(tri[peak:]) <= 1e-12))

    def test_escalation_is_ordered_and_starts_below_fc(self) -> None:
        # Starting under fc matters: a joint that releases near fc must never
        # see the larger peaks.
        self.assertEqual(list(_ESCALATION), sorted(_ESCALATION))
        self.assertLess(_ESCALATION[0], 1.0)

    def test_probe_feedforward_is_inert(self) -> None:
        # fc, k, fv, fo, j_eff, host_kd all zero: the joint feels exactly the
        # gravity + ramp the caller injects, with no friction model on top.
        self.assertEqual(_NO_FF[:6], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


class HomingTest(unittest.TestCase):
    """Every other joint is parked at rest, and the arm goes home afterwards."""

    def test_home_order_is_every_joint_distal_first(self) -> None:
        from almond_axol.cli.tune.breakaway import _HOME_ORDER

        self.assertEqual(set(_HOME_ORDER), set(ARM_JOINTS))
        self.assertEqual(len(_HOME_ORDER), len(set(_HOME_ORDER)))
        # Distal first: straightening the wrists and elbow before the
        # shoulders means each shoulder later swings a folded arm.
        self.assertEqual(_HOME_ORDER[0], Joint.WRIST_3)
        self.assertEqual(_HOME_ORDER[-1], Joint.SHOULDER_1)
        self.assertLess(
            _HOME_ORDER.index(Joint.ELBOW), _HOME_ORDER.index(Joint.SHOULDER_2)
        )

    def test_home_all_visits_every_joint_but_the_excluded_one(self) -> None:
        import asyncio
        from unittest.mock import patch

        from almond_axol.cli.tune import breakaway

        seen: list[Joint] = []

        async def fake_ramp(motors, targets):
            seen.extend(targets)

        motors = {j: object() for j in ARM_JOINTS}
        with patch.object(breakaway, "ramp_joints_to", fake_ramp):
            asyncio.run(breakaway._home_all(motors, exclude=Joint.SHOULDER_1))
        self.assertNotIn(Joint.SHOULDER_1, seen)
        self.assertEqual(set(seen), set(ARM_JOINTS) - {Joint.SHOULDER_1})
        self.assertEqual(seen, [j for j in breakaway._HOME_ORDER if j in seen])

    def test_home_all_skips_joints_absent_from_the_arm(self) -> None:
        import asyncio
        from unittest.mock import patch

        from almond_axol.cli.tune import breakaway

        seen: list[Joint] = []

        async def fake_ramp(motors, targets):
            seen.extend(targets)

        partial = {Joint.WRIST_2: object(), Joint.WRIST_3: object()}
        with patch.object(breakaway, "ramp_joints_to", fake_ramp):
            asyncio.run(breakaway._home_all(partial))
        self.assertEqual(set(seen), set(partial))


class ReleaseDetectionTest(unittest.TestCase):
    @staticmethod
    def _rows(positions):
        return [{"actual": p, "t": i / 240.0} for i, p in enumerate(positions)]

    def test_finds_the_torque_at_first_motion(self) -> None:
        n = 100
        ramp = 2.0 * _triangle(n)
        move = 3 * _FEEDBACK_LSB
        # Stuck until sample 40, then releases.
        pos = np.zeros(n)
        pos[40:] = 10 * _FEEDBACK_LSB
        got = detect_release(self._rows(pos), ramp, move)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, abs(ramp[40]), places=9)

    def test_no_motion_returns_none(self) -> None:
        n = 60
        ramp = 1.0 * _triangle(n)
        pos = np.zeros(n)
        self.assertIsNone(detect_release(self._rows(pos), ramp, 3 * _FEEDBACK_LSB))
        # Sub-threshold creep is not a release: one LSB of readout tick-over
        # is indistinguishable from a joint that has not moved.
        pos[30:] = 2 * _FEEDBACK_LSB
        self.assertIsNone(detect_release(self._rows(pos), ramp, 3 * _FEEDBACK_LSB))

    def test_dropped_frames_are_skipped_not_counted_as_motion(self) -> None:
        n = 80
        ramp = 2.0 * _triangle(n)
        pos = np.zeros(n)
        pos[10] = np.nan  # missed feedback frame
        pos[50:] = 10 * _FEEDBACK_LSB
        got = detect_release(self._rows(pos), ramp, 3 * _FEEDBACK_LSB)
        self.assertAlmostEqual(got, abs(ramp[50]), places=9)

    def test_too_few_good_samples_is_not_a_release(self) -> None:
        rows = [{"actual": np.nan, "t": 0.0}] * 20
        self.assertIsNone(detect_release(rows, _triangle(20), 1e-4))

    def test_release_on_the_falling_edge_is_still_reported(self) -> None:
        # A joint that only lets go after the peak still yields the torque it
        # let go at, not the peak.
        n = 100
        ramp = 2.0 * _triangle(n)
        pos = np.zeros(n)
        pos[70:] = 10 * _FEEDBACK_LSB
        got = detect_release(self._rows(pos), ramp, 3 * _FEEDBACK_LSB)
        self.assertAlmostEqual(got, abs(ramp[70]), places=9)
        self.assertLess(got, ramp.max())


class EntryPointTest(unittest.TestCase):
    """The arg-parse -> run() path, short of touching CAN.

    This is the seam that shipped broken: `run()` passed the `--log-level`
    *string* to `quiet_noisy_loggers`, which takes an int, and nothing
    exercised it because the unit tests all called the pure helpers.
    """

    def _args(self, argv):
        import argparse

        from almond_axol.cli.tune import breakaway

        parser = argparse.ArgumentParser()
        breakaway.add_parser(parser.add_subparsers(dest="cmd"))
        return parser.parse_args(["tune.breakaway", *argv])

    def test_run_reaches_the_async_body(self) -> None:
        from unittest.mock import patch

        from almond_axol.cli.tune import breakaway

        args = self._args(["--l", "--joint", "shoulder_1"])
        with patch.object(breakaway.asyncio, "run") as run_async:
            breakaway.run(args)
        self.assertEqual(run_async.call_count, 1)
        run_async.call_args.args[0].close()  # never awaited; don't warn

    def test_every_log_level_choice_is_accepted(self) -> None:
        from unittest.mock import patch

        from almond_axol.cli.tune import breakaway

        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            args = self._args(["--l", "--joint", "elbow", "--log-level", level])
            with patch.object(breakaway.asyncio, "run") as run_async:
                breakaway.run(args)
            run_async.call_args.args[0].close()

    def test_defaults_are_the_safe_ones(self) -> None:
        args = self._args(["--l", "--joint", "shoulder_1"])
        self.assertEqual(args.trials, 3)
        self.assertEqual(args.max_torque, 2.5)
        self.assertEqual(args.move_lsb, 3.0)
        self.assertIsNone(args.poses)
        self.assertGreater(args.kd, 0.0)  # the runaway brake is never off


class ReportTest(unittest.TestCase):
    """The report path itself — it runs on every measurement, and a crash
    there loses the run. `ndarray.ptp()` (removed in NumPy 2.0) shipped
    through here once already."""

    def test_renders_for_one_pose_two_poses_and_no_releases(self) -> None:
        from almond_axol.cli.tune.breakaway import _report

        two = [
            (math.radians(-20), -1.41, {"+": [1.30, 1.28], "-": [1.31, 1.29]}),
            (math.radians(20), -2.60, {"+": [1.42], "-": [1.40]}),
        ]
        for by_pose in (two, two[:1], [(0.0, -1.35, {"+": [], "-": []})]):
            _report(Joint.SHOULDER_1, 250.0, 0.909, by_pose)

    def test_splits_symmetric_breakaway_from_antisymmetric_bias(self) -> None:
        # The measurement that matters: probing both directions separates
        # static friction (symmetric) from an uncancelled standing torque
        # (antisymmetric). Averaging them together — as the first version
        # did — makes breakaway look direction-dependent when it is not.
        import io
        from contextlib import redirect_stdout

        from almond_axol.cli.tune.breakaway import _report

        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(
                Joint.SHOULDER_1,
                250.0,
                1.297,
                [
                    (
                        0.0,
                        -0.22,
                        {"+": [0.810], "-": [0.300], "trim_nm": [0.0]},
                    )
                ],
            )
        out = buf.getvalue()
        self.assertIn("0.555", out)  # (0.810 + 0.300) / 2 = breakaway
        # With no trim applied the residual is just the offset to the band
        # centre: (0.810 - 0.300) / 2.
        self.assertIn("0.255", out)

    def test_breakaway_is_independent_of_where_the_trim_stopped(self) -> None:
        """The invariant the whole loaded measurement rests on.

        The trim search stops at a band *edge*, not its centre. Writing the
        stiction band as [c-b, c+b] and the start as t, the two ramps cover
        (c+b)-t and t-(c-b): their mean is b for any t inside the band. So
        breakaway does not care where the search stopped — only the
        half-difference does, and that is reported as an offset.
        """
        rng = np.random.default_rng(0)
        for _ in range(50):
            c, b = rng.uniform(-1, 1), rng.uniform(0.2, 0.9)
            t = rng.uniform(c - b, c + b)
            plus, minus = (c + b) - t, t - (c - b)
            self.assertAlmostEqual((plus + minus) / 2, b, places=12)
            self.assertAlmostEqual((plus - minus) / 2, c - t, places=12)

    def test_residual_column_is_trim_plus_offset(self) -> None:
        import io
        from contextlib import redirect_stdout

        from almond_axol.cli.tune.breakaway import _report

        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(
                Joint.SHOULDER_1,
                250.0,
                1.297,
                [(0.26, 5.30, {"+": [0.311], "-": [1.010], "trim_nm": [-1.362]})],
            )
        out = buf.getvalue()
        # breakaway (0.311+1.010)/2, and residual -1.362 + (0.311-1.010)/2
        self.assertIn("0.660", out)
        self.assertIn("-1.712", out)

    def test_breakaway_below_fc_warns_against_more_compensation(self) -> None:
        # Static friction under the fitted fc means the feedforward is
        # already over-compensating; telling the operator to raise
        # stiction_gain there would push the wrong way.
        import io
        from contextlib import redirect_stdout

        from almond_axol.cli.tune.breakaway import _report

        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(
                Joint.SHOULDER_1,
                250.0,
                1.297,
                [(0.0, -0.22, {"+": [0.81], "-": [0.30], "drift_rad": [1e-4]})],
            )
        out = buf.getvalue()
        self.assertIn("BELOW the fitted fc", out)
        self.assertIn("do NOT raise stiction_gain", out)
        self.assertNotIn("stiction_err_deg", out)

    def test_breakaway_above_fc_suggests_stiction(self) -> None:
        import io
        from contextlib import redirect_stdout

        from almond_axol.cli.tune.breakaway import _report

        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(
                Joint.ELBOW,
                130.0,
                0.602,
                [(0.0, -0.1, {"+": [0.90], "-": [0.90], "drift_rad": [1e-4]})],
            )
        out = buf.getvalue()
        self.assertIn("stiction_gain", out)
        self.assertIn("stair height", out)

    def test_skipped_pose_renders_and_is_excluded_from_the_fit(self) -> None:
        import io
        from contextlib import redirect_stdout

        from almond_axol.cli.tune.breakaway import _report

        buf = io.StringIO()
        with redirect_stdout(buf):
            _report(
                Joint.SHOULDER_1,
                250.0,
                1.297,
                [
                    (0.0, -0.22, {"+": [0.81], "-": [0.30], "drift_rad": [1e-4]}),
                    (0.8, 14.9, {"+": [], "-": [], "drift_rad": [0.009]}),
                ],
            )
        out = buf.getvalue()
        self.assertIn("skipped", out)

    def test_one_sided_release_still_reports(self) -> None:
        # A joint that only breaks one way is a real outcome, not a crash.
        from almond_axol.cli.tune.breakaway import _report

        _report(Joint.ELBOW, 130.0, 0.602, [(0.0, -1.35, {"+": [0.71], "-": []})])


class SuggestionMathTest(unittest.TestCase):
    """The numbers the report turns a measurement into."""

    def test_stair_height_follows_the_breakaway_gap(self) -> None:
        fc, kp = 0.909, 250.0
        for ratio, expect_deg in ((1.0, 0.0), (1.5, math.degrees(0.5 * fc / kp))):
            stair = math.degrees(max(ratio * fc - fc, 0.0) / kp)
            self.assertAlmostEqual(stair, expect_deg, places=9)

    def test_load_slope_recovers_a_planted_gain(self) -> None:
        # friction_load_gain is the slope of breakaway against gravity load.
        fc, gain = 0.6, 0.08
        g = np.array([0.5, 2.0, 4.0])
        means = fc + gain * g
        self.assertAlmostEqual(float(np.polyfit(g, means, 1)[0]), gain, places=9)

    def test_feedback_lsb_matches_the_mit_frame(self) -> None:
        self.assertAlmostEqual(math.degrees(_FEEDBACK_LSB), 0.02197, places=5)


if __name__ == "__main__":
    unittest.main()
