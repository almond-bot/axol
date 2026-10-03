"""Teleop's structural-mode command notch."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.filter import NotchFilter
from almond_axol.tuning.filtering import replay_filter_stack

FS = 240.0


def _gain(f: NotchFilter, hz: float, seconds: float = 12.0) -> tuple[float, float]:
    t = np.arange(int(seconds * FS)) / FS
    x = np.sin(2 * math.pi * hz * t)
    f.reset(np.zeros(1))
    y = np.array([f.update(np.array([v]))[0] for v in x])
    s = slice(int(6 * FS), None)
    g = np.std(y[s]) / np.std(x[s])
    lag = np.angle(np.sum(y[s] * np.exp(-2j * math.pi * hz * t[s]))) - np.angle(
        np.sum(x[s] * np.exp(-2j * math.pi * hz * t[s]))
    )
    return float(g), math.degrees(lag)


class NotchTest(unittest.TestCase):
    def test_kills_the_mode_and_passes_the_motion(self) -> None:
        f = NotchFilter([5.5], 2.0, FS)
        self.assertLess(_gain(f, 5.5)[0], 0.02)
        g, phase = _gain(f, 0.3)
        self.assertAlmostEqual(g, 1.0, delta=0.01)
        self.assertLess(abs(phase), 3.0)
        self.assertAlmostEqual(_gain(f, 30.0)[0], 1.0, delta=0.02)

    def test_cascade_and_seeded_hold_has_no_transient(self) -> None:
        f = NotchFilter([2.1, 5.5], [1.0, 2.0], FS)
        self.assertLess(_gain(f, 2.1)[0], 0.02)
        self.assertLess(_gain(f, 5.5)[0], 0.02)
        pose = np.array([0.3, -1.2, 0.0, 0.5, 0.1, 0.0, 0.2])
        f.reset(pose)
        for _ in range(50):
            np.testing.assert_allclose(f.update(pose), pose, atol=1e-6)

    def test_empty_is_pass_through_and_bad_specs_are_refused(self) -> None:
        f = NotchFilter([], 2.0, FS)
        x = np.array([0.1, 0.2])
        self.assertIs(f.update(x), x)
        self.assertFalse(f.active)
        with self.assertRaises(ValueError):
            NotchFilter([200.0], 2.0, FS)
        with self.assertRaises(ValueError):
            NotchFilter([5.0], 0.0, FS)

    def test_teleop_stack_default_is_unchanged_and_the_notch_applies(self) -> None:
        t = np.arange(0, 10, 1 / 90)
        x = np.stack(
            [
                0.3 * np.sin(2 * math.pi * 0.3 * t)
                + 0.01 * np.sin(2 * math.pi * 5.5 * t)
            ],
            1,
        )
        cfg = VRTeleopConfig()
        self.assertEqual(cfg.command_notch_hz, [])
        _, plain, _ = replay_filter_stack(t, x, config=cfg)
        notched_cfg = VRTeleopConfig()
        notched_cfg.command_notch_hz = [5.5]
        t_out, notched, _ = replay_filter_stack(t, x, config=notched_cfg)
        fs = 1 / np.median(np.diff(t_out))

        def band(y: np.ndarray) -> float:
            seg = y[int(3 * fs) :, 0]
            # Hann: the 0.3 Hz motion is 30x the 5.5 Hz content and would
            # leak into the band through a rectangular window.
            spec = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg))))
            f = np.fft.rfftfreq(len(y) - int(3 * fs), 1 / fs)
            return float(spec[(f > 5.0) & (f < 6.0)].max())

        self.assertLess(band(notched), 0.2 * band(plain))


if __name__ == "__main__":
    unittest.main()
