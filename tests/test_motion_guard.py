"""The unattended-run tracking guard: what trips it and what must not."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.tuning.motion_guard import MotionGuard

RATE = 240.0
T = np.arange(0.0, 8.0, 1.0 / RATE)
LAG = np.radians(0.4) * np.sin(2 * math.pi * 0.2 * T)  # slow tracking lag


def _first_trip(err: np.ndarray, **kw: float):
    g = MotionGuard(["right.elbow"], RATE, **kw)
    for k, e in enumerate(err):
        trip = g.update(np.array([e]))
        if trip is not None:
            return T[k], trip
    return None


class MotionGuardTest(unittest.TestCase):
    def test_slow_lag_and_feedback_gaps_never_trip(self) -> None:
        err = LAG.copy()
        err[::29] = np.nan  # missed feedback samples
        self.assertIsNone(_first_trip(err))

    def test_a_growing_ring_trips_as_an_oscillation_within_a_second(self) -> None:
        ring = np.radians(0.2) * np.exp(T / 1.5) * np.sin(2 * math.pi * 3.0 * T)
        hit = _first_trip(LAG + ring)
        assert hit is not None
        t, trip = hit
        self.assertEqual(trip.kind, "oscillation")
        self.assertEqual(trip.joint, "right.elbow")
        # Caught while the ring is still a degree or two, not after it ran away.
        self.assertLess(0.2 * math.exp(t / 1.5), 2.5)  # ring amplitude, degrees

    def test_a_buzz_trips_as_a_vibration(self) -> None:
        buzz = np.radians(0.3) * np.sin(2 * math.pi * 40.0 * T)
        hit = _first_trip(LAG + buzz)
        assert hit is not None
        self.assertEqual(hit[1].kind, "vibration")

    def test_a_deviation_must_persist_and_scale_relaxes_it(self) -> None:
        err = LAG.copy()
        err[500:503] = np.radians(15.0)  # three samples: a glitch, not a departure
        self.assertIsNone(_first_trip(err, osc_deg=100, vib_deg=100))
        err[1000:1100] = np.radians(15.0)
        hit = _first_trip(err, osc_deg=100, vib_deg=100)
        assert hit is not None
        self.assertEqual(hit[1].kind, "deviation")
        g = MotionGuard(["j"], RATE, osc_deg=100, vib_deg=100)
        self.assertIsNone(
            next(
                (t for e in err if (t := g.update(np.array([e]), scale=2.0))),
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
