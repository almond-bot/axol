"""Jelly's thumbstick response curve.

Operators found the linear stick → speed map jumpy: the short throw of a
thumbstick reaches most of the base's speed with a small push, so they
overshoot. ``JellyConfig.stick_expo`` bends the map so the first part of the
travel is slow and fine while full deflection still gives full speed.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from almond_axol.robot.jelly import (
    Jelly,
    JellyConfig,
    deadzone,
    stick_command,
    stick_curve,
)


class StickCurveTest(unittest.TestCase):
    def test_full_deflection_is_still_full_speed(self) -> None:
        for expo in (0.0, 0.3, 0.6, 1.0):
            self.assertAlmostEqual(stick_curve(1.0, expo), 1.0)
            self.assertAlmostEqual(stick_curve(-1.0, expo), -1.0)
            self.assertEqual(stick_curve(0.0, expo), 0.0)

    def test_expo_slows_the_first_part_of_the_travel(self) -> None:
        self.assertAlmostEqual(stick_curve(0.5, 0.0), 0.5)  # linear
        self.assertAlmostEqual(stick_curve(0.5, 0.6), 0.275)
        self.assertAlmostEqual(stick_curve(0.25, 0.6), 0.109375)
        self.assertAlmostEqual(stick_curve(-0.5, 0.6), -0.275)

    def test_monotonic_and_clamped(self) -> None:
        for expo in (-1.0, 0.6, 2.0):
            xs = [i / 100.0 for i in range(101)]
            ys = [stick_curve(x, expo) for x in xs]
            self.assertTrue(all(b >= a for a, b in zip(ys, ys[1:])), expo)
        self.assertEqual(stick_curve(0.5, 2.0), stick_curve(0.5, 1.0))
        self.assertEqual(stick_curve(0.5, -1.0), stick_curve(0.5, 0.0))


class StickCommandTest(unittest.TestCase):
    def test_linear_matches_the_plain_deadzone(self) -> None:
        vx, vy, wz = stick_command(0.3, -0.7, 0.5, 0.15, 0.0)
        self.assertAlmostEqual(vx, -deadzone(-0.7, 0.15))
        self.assertAlmostEqual(vy, -deadzone(0.3, 0.15))
        self.assertAlmostEqual(wz, -deadzone(0.5, 0.15))

    def test_a_diagonal_keeps_its_heading(self) -> None:
        lin = stick_command(-0.5, -0.5, 0.0, 0.15, 0.0)
        cur = stick_command(-0.5, -0.5, 0.0, 0.15, 0.6)
        self.assertAlmostEqual(math.atan2(cur[1], cur[0]), math.atan2(lin[1], lin[0]))
        self.assertLess(math.hypot(cur[0], cur[1]), math.hypot(lin[0], lin[1]))

    def test_inside_the_deadzone_is_zero(self) -> None:
        self.assertEqual(stick_command(0.1, -0.1, 0.1, 0.15, 0.6), (0.0, 0.0, 0.0))


class VrFrameCurveTest(unittest.TestCase):
    def _target(self, expo: float, **sticks: float) -> tuple[float, float, float]:
        jelly = Jelly(JellyConfig(lift=False, channel=None, stick_expo=expo))
        frame = SimpleNamespace(
            reset=False,
            l_stick_x=0.0,
            l_stick_y=0.0,
            r_stick_x=0.0,
            l_stick_click=False,
            r_stick_click=False,
        )
        for key, value in sticks.items():
            setattr(frame, key, value)
        jelly.apply_vr_frame(frame)
        return jelly._target[:3]

    def test_the_headset_sticks_go_through_the_curve(self) -> None:
        self.assertEqual(JellyConfig().stick_expo, 0.6)
        # Half the travel past the deadzone, pushed forward.
        half = -(0.15 + 0.5 * 0.85)
        self.assertAlmostEqual(self._target(0.0, l_stick_y=half)[0], 0.5)
        self.assertAlmostEqual(self._target(0.6, l_stick_y=half)[0], 0.275)
        self.assertAlmostEqual(self._target(0.6, r_stick_x=-1.0)[2], 1.0)


if __name__ == "__main__":
    unittest.main()
