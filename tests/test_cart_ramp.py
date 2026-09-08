"""Cart command ramp, wheel-scale mixing, and the radius calibration solver."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.robot.cart import (
    WHEEL_SIGNS,
    WHEELS,
    CartConfig,
    VectorRamp,
    mix,
    solve_wheel_scale,
)

DT = 0.02


def _drive(ramp: VectorRamp, cmd: list[float], target, cycles: int) -> list[float]:
    """Step the ramp, returning the per-cycle command magnitude trace."""
    trace = []
    for _ in range(cycles):
        ramp.step(cmd, target)
        trace.append(math.hypot(*cmd))
    return trace


class VectorRampTests(unittest.TestCase):
    def test_trapezoid_reaches_target_exactly_at_the_accel_rate(self) -> None:
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=0.0, dt=DT)
        cmd = [0.0, 0.0, 0.0]
        trace = _drive(ramp, cmd, (0.7, 0.0, 0.0), 200)
        steps = np.diff([0.0, *trace])
        self.assertLessEqual(steps.max(), 0.5 * DT + 1e-12)
        settled = next(i for i, v in enumerate(trace) if v == 0.7)
        self.assertAlmostEqual(settled * DT, 0.7 / 0.5, delta=DT)
        self.assertEqual(cmd, [0.7, 0.0, 0.0])

    def test_release_uses_the_faster_decel_and_lands_on_exact_zero(self) -> None:
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=0.0, dt=DT)
        cmd = [0.7, 0.0, 0.0]
        trace = _drive(ramp, cmd, (0.0, 0.0, 0.0), 100)
        steps = -np.diff([0.7, *trace])
        self.assertAlmostEqual(steps[0], 1.0 * DT)
        stopped = next(i for i, v in enumerate(trace) if v == 0.0)
        self.assertAlmostEqual(stopped * DT, 0.7 / 1.0, delta=DT)
        # Exactly zero, not merely small: the park state machine keys off it.
        self.assertEqual(cmd, [0.0, 0.0, 0.0])

    def test_speed_reduction_counts_as_decel(self) -> None:
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=0.0, dt=DT)
        cmd = [0.7, 0.0, 0.0]
        ramp.step(cmd, (0.3, 0.0, 0.0))
        self.assertAlmostEqual(cmd[0], 0.7 - 1.0 * DT)

    def test_reversal_decelerates_through_zero_then_accelerates(self) -> None:
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=0.0, dt=DT)
        cmd = [0.5, 0.0, 0.0]
        rates, values = [], []
        for _ in range(100):
            before = cmd[0]
            ramp.step(cmd, (-0.5, 0.0, 0.0))
            rates.append((cmd[0] - before) / DT)
            values.append(cmd[0])
        crossing = next(i for i, v in enumerate(values) if v < 0.0)
        # decel (1.0/s) all the way down to zero ...
        self.assertTrue(all(abs(r + 1.0) < 1e-9 for r in rates[: crossing - 1]))
        # ... then accel (0.5/s) out the other side
        self.assertTrue(
            all(abs(r + 0.5) < 1e-9 for r in rates[crossing + 1 : crossing + 10])
        )

    def test_direction_is_preserved_while_ramping(self) -> None:
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=2.0, dt=DT)
        cmd = [0.0, 0.0, 0.0]
        for _ in range(20):
            ramp.step(cmd, (0.6, 0.3, 0.0))
            if cmd[0] > 0.0:
                self.assertAlmostEqual(cmd[1] / cmd[0], 0.5)
        self.assertLess(math.hypot(cmd[0], cmd[1]), math.hypot(0.6, 0.3))

    def test_jerk_limit_gives_an_s_curve_that_still_settles_exactly(self) -> None:
        jerk = 2.0
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=jerk, dt=DT)
        cmd = [0.0, 0.0, 0.0]
        trace = _drive(ramp, cmd, (0.7, 0.0, 0.0), 200)
        rates = np.diff([0.0, *trace]) / DT
        accel = np.diff(rates) / DT
        # rate ramps up gradually (first step far below the accel limit) ...
        self.assertLess(rates[0], 0.1)
        self.assertLessEqual(rates.max(), 0.5 + 1e-9)
        # ... and never changes faster than the jerk limit (the final step
        # lands the sub-step remainder on the target, so exclude it).
        settled = next(i for i, v in enumerate(trace) if v == 0.7)
        self.assertLessEqual(np.abs(accel[: settled - 1]).max(), jerk + 1e-6)
        self.assertEqual(cmd, [0.7, 0.0, 0.0])
        # and the stop tail also lands on exact zero
        trace = _drive(ramp, cmd, (0.0, 0.0, 0.0), 200)
        self.assertIn(0.0, trace)
        self.assertEqual(cmd, [0.0, 0.0, 0.0])

    def test_release_mid_launch_swings_the_rate_smoothly_through_zero(self) -> None:
        jerk = 2.0
        ramp = VectorRamp(accel=0.5, decel=1.0, jerk=jerk, dt=DT)
        cmd = [0.0, 0.0, 0.0]
        trace = _drive(ramp, cmd, (0.7, 0.0, 0.0), 25)  # still accelerating
        self.assertGreater(ramp.vel[0], 0.4)
        trace += _drive(ramp, cmd, (0.0, 0.0, 0.0), 200)
        rates = np.diff([0.0, *trace]) / DT
        accel = np.diff(rates) / DT
        stopped = next(i for i, v in enumerate(trace) if v == 0.0)
        self.assertLessEqual(np.abs(accel[: stopped - 1]).max(), jerk + 1e-6)
        self.assertEqual(cmd, [0.0, 0.0, 0.0])

    def test_rejects_degenerate_rates(self) -> None:
        for kwargs in (
            dict(accel=0.0, decel=1.0, jerk=0.0),
            dict(accel=0.5, decel=0.0, jerk=0.0),
            dict(accel=0.5, decel=1.0, jerk=-1.0),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                VectorRamp(dt=DT, **kwargs)


class CartConfigTests(unittest.TestCase):
    def test_defaults_are_asymmetric_and_hold_is_on(self) -> None:
        cfg = CartConfig()
        self.assertGreater(cfg.decel, cfg.accel)
        self.assertGreater(cfg.jerk, 0.0)
        self.assertTrue(cfg.imu)
        self.assertEqual(cfg.wheel_scale, (1.0, 1.0, 1.0, 1.0))

    def test_rejects_ramps_that_would_never_stop(self) -> None:
        for kwargs in (
            dict(decel=0.0),
            dict(decel=-1.0),
            dict(accel=0.0),
            dict(jerk=-0.1),
            dict(decel=math.nan),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                CartConfig(**kwargs)

    def test_rejects_malformed_wheel_scale(self) -> None:
        for scale in ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 3.0), (1.0, 1.0, 1.0, 0.0)):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                CartConfig(wheel_scale=scale)
        cfg = CartConfig(wheel_scale=[1, 1.02, 0.98, 1])  # list from a CLI parse
        self.assertEqual(cfg.wheel_scale, (1.0, 1.02, 0.98, 1.0))


class MixTests(unittest.TestCase):
    def test_wheel_scale_multiplies_each_wheel_after_normalization(self) -> None:
        plain = mix(1.0, 0.0, 0.0, 20.0, 1.0)
        scaled = mix(1.0, 0.0, 0.0, 20.0, 1.0, (1.01, 0.99, 1.0, 1.0))
        self.assertEqual(scaled[0], plain[0] * 1.01)
        self.assertEqual(scaled[1], plain[1] * 0.99)
        self.assertEqual(scaled[2:], plain[2:])


class SolveWheelScaleTests(unittest.TestCase):
    """The solver is checked against the x-drive forward kinematics."""

    LEVER = (0.28 + 0.28) / math.sqrt(2)

    def _stroke(self, radii, motor_speeds, seconds):
        turns = tuple(s * seconds for s in motor_speeds)
        phi = np.array([WHEEL_SIGNS[w.motor_id] * t for w, t in zip(WHEELS, turns)])
        u = np.asarray(radii) * phi
        mx = np.array([w.mx for w in WHEELS])
        my = np.array([w.my for w in WHEELS])
        mw = np.array([w.mw for w in WHEELS])
        fwd = (mx * u).sum() / (2 * math.sqrt(2))
        left = (my * u).sum() / (2 * math.sqrt(2))
        heading = (mw * u).sum() / (4 * self.LEVER)
        return turns, fwd, left, heading

    def test_recovers_radius_spread_from_forward_and_left_strokes(self) -> None:
        radii = 0.05 * np.array([1.02, 0.97, 1.01, 1.0])
        strokes = [
            self._stroke(radii, mix(0.7, 0.0, 0.0, 20.0, 1.0), 3.0),
            self._stroke(radii, mix(0.0, 0.7, 0.0, 20.0, 1.0), 3.0),
        ]
        scales = solve_wheel_scale(strokes, self.LEVER)
        expected = radii.mean() / radii
        expected /= expected.mean()
        np.testing.assert_allclose(scales, expected, atol=1e-9)
        # applying the scales makes a forward stroke go exactly straight
        _, _, left, heading = self._stroke(
            radii, mix(0.7, 0.0, 0.0, 20.0, 1.0, scales), 3.0
        )
        self.assertAlmostEqual(left, 0.0, places=9)
        self.assertAlmostEqual(heading, 0.0, places=9)

    def test_identical_wheels_give_unit_scales(self) -> None:
        radii = [0.05] * 4
        strokes = [
            self._stroke(radii, mix(0.5, 0.0, 0.0, 20.0, 1.0), 2.0),
            self._stroke(radii, mix(0.0, -0.5, 0.0, 20.0, 1.0), 2.0),
        ]
        np.testing.assert_allclose(solve_wheel_scale(strokes, self.LEVER), 1.0)

    def test_underdetermined_strokes_are_rejected(self) -> None:
        radii = [0.05] * 4
        fwd = self._stroke(radii, mix(0.7, 0.0, 0.0, 20.0, 1.0), 3.0)
        with self.assertRaisesRegex(ValueError, "not determined"):
            solve_wheel_scale([fwd], self.LEVER)
        with self.assertRaisesRegex(ValueError, "not determined"):
            solve_wheel_scale(
                [fwd, self._stroke(radii, mix(0.4, 0.0, 0.0, 20.0, 1.0), 1.0)],
                self.LEVER,
            )
        with self.assertRaises(ValueError):
            solve_wheel_scale([fwd, fwd], 0.0)


if __name__ == "__main__":
    unittest.main()
