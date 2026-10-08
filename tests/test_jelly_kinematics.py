"""Jelly config validation, wheel-scale mixing, and the radius calibration solver.

The command ramp and the traction guard run in the Rust core; their unit
tests live in ``rust/axol-rt/src/ramp.rs``.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.robot.jelly import (
    WHEEL_SIGNS,
    WHEELS,
    JellyConfig,
    mix,
    solve_wheel_scale,
)


class JellyConfigTests(unittest.TestCase):
    def test_defaults_are_asymmetric_and_hold_is_on(self) -> None:
        cfg = JellyConfig()
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
                JellyConfig(**kwargs)

    def test_traction_is_on_by_default_and_validated(self) -> None:
        cfg = JellyConfig()
        self.assertTrue(cfg.traction)
        with self.assertRaises(ValueError):
            JellyConfig(traction_light=1.2)
        with self.assertRaises(ValueError):
            JellyConfig(traction_floor=0.0)
        JellyConfig(traction=False, traction_light=1.2)  # ignored when off

    def test_rejects_malformed_wheel_scale(self) -> None:
        for scale in ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 3.0), (1.0, 1.0, 1.0, 0.0)):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                JellyConfig(wheel_scale=scale)
        cfg = JellyConfig(wheel_scale=[1, 1.02, 0.98, 1])  # list from a CLI parse
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
