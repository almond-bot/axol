"""The band an operator sees was defined as signal and never measured.

``metrics.BAND_LOW`` is 3 Hz and everything below it was reported as one
number called intentional motion. On the ``slow_osc`` capture the commanded
trajectory held 99.9 % of its energy below 0.59 Hz while the tracker added
~0.9 mm rms of 1-3 Hz pose noise, so the defect sat inside ``rms_low``
alongside 123 mm of real motion and was invisible. The pose filter passes
94 % of that band and 68 % of the 3-15 Hz band it was tuned for.
"""

from __future__ import annotations

import unittest

import numpy as np

from almond_axol.tuning.metrics import BAND_BOUNCE, BAND_HIGH, BAND_LOW, band_rms


class BounceBandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fs = 240.0
        self.t = np.arange(0, 40, 1 / self.fs)

    def test_bounce_band_sits_between_intent_and_the_felt_band(self) -> None:
        self.assertLess(BAND_BOUNCE, BAND_LOW)
        self.assertLess(BAND_LOW, BAND_HIGH)

    def test_isolates_a_2_hz_wobble_riding_a_large_slow_motion(self) -> None:
        """The case that hid: a 1 mm wobble on 120 mm of intentional travel."""
        big = 0.120 * np.sin(2 * np.pi * 0.12 * self.t)
        wobble = 0.001 * np.sqrt(2) * np.sin(2 * np.pi * 2.0 * self.t)
        b = band_rms(self.t, big + wobble)
        self.assertAlmostEqual(1e3 * b["rms_bounce"], 1.0, places=1)
        # ...and it is a subdivision of rms_low, which stays dominated by the
        # intentional motion — which is exactly why it hid there.
        self.assertGreater(b["rms_low"], 50 * b["rms_bounce"])

    def test_does_not_claim_intentional_motion_as_bounce(self) -> None:
        slow = 0.120 * np.sin(2 * np.pi * 0.12 * self.t)
        self.assertLess(1e3 * band_rms(self.t, slow)["rms_bounce"], 0.05)

    def test_does_not_leak_from_the_felt_band(self) -> None:
        felt = 0.001 * np.sin(2 * np.pi * 8.0 * self.t)
        b = band_rms(self.t, felt)
        self.assertLess(b["rms_bounce"], 0.05 * b["rms_mid"])

    def test_bands_still_sum_to_the_variance(self) -> None:
        """rms_bounce overlaps rms_low by construction; low/mid/high must
        still partition the signal or every stored run becomes unreadable."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(len(self.t))
        b = band_rms(self.t, x)
        total = b["rms_low"] ** 2 + b["rms_mid"] ** 2 + b["rms_high"] ** 2
        self.assertAlmostEqual(np.sqrt(total), x.std(), places=2)
        self.assertLessEqual(b["rms_bounce"], b["rms_low"] + 1e-12)

    def test_nan_shape_matches_the_success_shape(self) -> None:
        short = band_rms(np.arange(4) / self.fs, np.zeros(4))
        full = band_rms(self.t, np.sin(self.t))
        self.assertEqual(set(short), set(full))


if __name__ == "__main__":
    unittest.main()
