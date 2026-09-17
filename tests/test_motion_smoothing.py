"""The replay smoother must separate a recorded motion from tracker noise.

Measured on the ``slow_osc`` capture (28 s of teleop, right arm driven):
99.9 % of the commanded trajectory's energy sits below 0.59 Hz, while the
tracker contributes 3.36 mm rms of 1-3 Hz pose noise whose amplitude grows
with hand speed — 0.82 mm rms on the nearly-still left hand at 10 mm/s
against 3.36 mm on the right at 84 mm/s, and roughly isotropic, so it is not
a lag artifact. The arm tracked that noise faithfully (commanded elbow wobble
0.264° vs measured 0.269°) and it reached the operator as 1.86 mm rms of
vertical TCP bounce.

A recorded path carries no causality constraint, so it should be filtered far
harder than a live stream can be. The original cascaded one-pole rolled off at
only -40 dB/dec and defaulted to 6 Hz, which passed the entire band through.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import signal

from almond_axol.tuning import motion as mot


def _band(x: np.ndarray, rate: float, lo: float, hi: float) -> float:
    f, p = signal.welch(x - x.mean(axis=0), rate, nperseg=1024, axis=0)
    m = (f >= lo) & (f < hi)
    return float(p.sum(axis=1)[m].sum())


class ZeroPhaseSmoothingTest(unittest.TestCase):
    def test_no_phase_lag(self) -> None:
        rate = 120.0
        t = np.arange(0, 20, 1 / rate)
        x = np.sin(2 * np.pi * 0.2 * t)[:, None]
        y = mot._zero_phase_lowpass(x, rate, 0.8)
        # Cross-correlation peak at zero lag, not argmax of the signal: a
        # sine is flat at its crest, so argmax wanders by a sample or two on
        # numerical noise alone. A causal 0.8 Hz filter would shift a 0.2 Hz
        # sine by tens of milliseconds; that is the whole point of filtfilt.
        a, b = x[:, 0] - x[:, 0].mean(), y[:, 0] - y[:, 0].mean()
        lag = int(np.argmax(signal.correlate(b, a, "full"))) - (len(a) - 1)
        self.assertEqual(lag, 0)
        self.assertGreater(y[:, 0].std() / x[:, 0].std(), 0.99)

    def test_rejects_the_band_the_tracker_noise_lives_in(self) -> None:
        rate = 120.0
        t = np.arange(0, 30, 1 / rate)
        rng = np.random.default_rng(0)
        intent = np.sin(2 * np.pi * 0.2 * t)[:, None] * np.ones((1, 14))
        sos = signal.butter(2, [1.0, 3.0], "bp", fs=rate, output="sos")
        noise = signal.sosfiltfilt(
            sos, 0.05 * rng.standard_normal((len(t), 14)), axis=0
        )
        y = mot._zero_phase_lowpass(intent + noise, rate, 0.8)
        self.assertGreater(_band(y, rate, 0, 0.6) / _band(intent, rate, 0, 0.6), 0.98)
        self.assertLess(_band(y, rate, 1, 3) / _band(intent + noise, rate, 1, 3), 0.05)

    def test_edges_are_not_left_with_a_settling_transient(self) -> None:
        """A motion starts and ends at rest, and those waypoints are replayed
        into the arm. ``filtfilt``'s default padding is 3*(2*order+1) samples
        — 0.22 s at order 4 — against a ~1.25 s transient for a 0.8 Hz filter,
        which leaves a step exactly where a step is least welcome."""
        rate = 120.0
        t = np.arange(0, 20, 1 / rate)
        x = np.sin(2 * np.pi * 0.2 * t)[:, None]
        y = mot._zero_phase_lowpass(x, rate, 0.8)
        # In-band signal: the filter should be a near-identity everywhere,
        # including the first and last waypoint.
        self.assertLess(float(np.abs(y - x).max()), 0.01)
        self.assertLess(abs(float(y[0, 0] - x[0, 0])), 0.01)
        self.assertLess(abs(float(y[-1, 0] - x[-1, 0])), 0.01)

    def test_long_padding_never_exceeds_the_record(self) -> None:
        rate = 120.0
        t = np.arange(0, 1.5, 1 / rate)  # shorter than 3 cycles of 0.8 Hz
        x = np.sin(2 * np.pi * 0.2 * t)[:, None]
        y = mot._zero_phase_lowpass(x, rate, 0.8)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.isfinite(y).all())


class SpectralCutoffTest(unittest.TestCase):
    def test_tracks_the_motion_it_is_given(self) -> None:
        """One fixed cutoff cannot serve both a slow reach and a fast swing:
        it is either too high to clean the first or too low to pass the
        second, which is why the cutoff is read off the capture itself."""
        rate = 120.0
        t = np.arange(0, 30, 1 / rate)
        slow = np.sin(2 * np.pi * 0.2 * t)[:, None] * np.ones((1, 14))
        fast = np.sin(2 * np.pi * 3.0 * t)[:, None] * np.ones((1, 14))
        c_slow = mot._spectral_cutoff(slow, rate)
        c_fast = mot._spectral_cutoff(fast, rate)
        self.assertLess(c_slow, c_fast)
        self.assertEqual(c_slow, mot._SMOOTH_MIN_HZ)
        self.assertGreaterEqual(c_fast, 3.0)
        self.assertLessEqual(c_fast, mot._SMOOTH_MAX_HZ)

    def test_degenerate_captures_do_not_raise(self) -> None:
        x = np.zeros((5, 14))
        np.testing.assert_array_equal(mot._zero_phase_lowpass(x, 120.0, 0.8), x)
        self.assertEqual(
            mot._spectral_cutoff(np.zeros((4, 14)), 120.0), mot._SMOOTH_MAX_HZ
        )
        # Constant input has no spectrum to read a knee from.
        self.assertEqual(
            mot._spectral_cutoff(np.ones((2048, 14)), 120.0), mot._SMOOTH_MAX_HZ
        )


if __name__ == "__main__":
    unittest.main()
