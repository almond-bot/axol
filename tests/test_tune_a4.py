"""The pure parts of ``axol tune.a4``: the wave generator, the buzz guard, the
0xA4 frame, and the creep-smoothness scorecard."""

from __future__ import annotations

import math
import unittest

import numpy as np

from almond_axol.cli.tune import a4


class WaveformTest(unittest.TestCase):
    def test_sine_starts_at_centre_with_matching_velocity(self) -> None:
        w = a4.waveform("sine", 0.5, 0.1, 2.0, 100.0, freq=0.5)
        self.assertEqual(len(w), 200)
        t0, q0, v0 = w[0]
        self.assertEqual((t0, q0), (0.0, 0.5))
        self.assertAlmostEqual(v0, 0.1 * 2 * math.pi * 0.5)
        # Quarter period later the sine peaks and the velocity is zero.
        _, q, v = w[50]
        self.assertAlmostEqual(q, 0.6, places=6)
        self.assertAlmostEqual(v, 0.0, places=6)

    def test_triangle_runs_every_leg_at_the_set_speed(self) -> None:
        w = a4.waveform("triangle", 0.0, 0.2, 20.0, 100.0, speed=0.1)
        q = np.array([s[1] for s in w])
        v = np.array([s[2] for s in w])
        self.assertAlmostEqual(q.max(), 0.2, places=6)
        self.assertAlmostEqual(q.min(), -0.2, places=6)
        # Velocity is ±speed everywhere, and the position slope matches it on
        # every sample except the turnarounds.
        self.assertTrue(np.allclose(np.abs(v), 0.1))
        slope = np.abs(np.diff(q) * 100.0)
        self.assertGreater(np.mean(np.isclose(slope, 0.1, atol=1e-6)), 0.95)
        # First leg goes up from the centre at +speed.
        self.assertEqual(w[0][1], 0.0)
        self.assertGreater(w[1][1], 0.0)

    def test_zero_speed_or_frequency_holds(self) -> None:
        self.assertTrue(
            all(
                q == 1.0 and v == 0.0
                for _, q, v in a4.waveform("triangle", 1.0, 0.2, 1.0, 50.0, speed=0.0)
            )
        )
        self.assertTrue(
            all(
                q == 1.0
                for _, q, _v in a4.waveform("sine", 1.0, 0.2, 1.0, 50.0, freq=0.0)
            )
        )
        with self.assertRaises(ValueError):
            a4.waveform("square", 0.0, 0.1, 1.0, 50.0)


class BuzzGuardTest(unittest.TestCase):
    def test_smooth_creep_passes_and_vibration_trips(self) -> None:
        guard = a4.BuzzGuard(200.0, math.radians(0.3), 10.0)
        # A steady 3 deg/s creep never trips.
        for k in range(60):
            self.assertIsNone(guard.feed(math.radians(3.0) * k / 200.0, 1.0))
        # A 25 Hz, 0.6° oscillation does within one window.
        tripped = None
        for k in range(60, 120):
            tripped = guard.feed(
                math.radians(0.9)
                + math.radians(0.6) * math.sin(2 * math.pi * 25 * k / 200.0),
                1.0,
            )
            if tripped:
                break
        self.assertIsNotNone(tripped)
        self.assertIn("high-frequency", tripped)

    def test_current_limit_trips_immediately(self) -> None:
        guard = a4.BuzzGuard(200.0, math.radians(0.3), 5.0)
        self.assertIsNone(guard.feed(0.0, 4.0))
        self.assertIn("current", guard.feed(0.0, -6.0) or "")

    def test_limits_at_zero_are_off(self) -> None:
        guard = a4.BuzzGuard(200.0, 0.0, 0.0)
        for k in range(100):
            self.assertIsNone(guard.feed(math.sin(k), 50.0))


class FrameTest(unittest.TestCase):
    def test_a4_frame_matches_the_vendor_example(self) -> None:
        self.assertEqual(
            a4._a4_frame(2 * math.pi, 500.0),
            bytes([0xA4, 0x00, 0xF4, 0x01, 0xA0, 0x8C, 0x00, 0x00]),
        )
        iq, speed = a4._decode_a4_reply(
            bytes([0xA4, 0x32, 0x64, 0x00, 0xF4, 0x01, 0x2D, 0x00])
        )
        self.assertAlmostEqual(iq, 1.0)
        self.assertAlmostEqual(speed, math.radians(500.0))


class MetricsTest(unittest.TestCase):
    def _log(self, lag_s: float, ripple: float) -> list[dict]:
        rate = 200.0
        rng = np.random.default_rng(1)
        rows = []
        for k in range(int(10 * rate)):
            t = k / rate
            target = 0.05 * t
            actual = 0.05 * (t - lag_s) + ripple * math.sin(2 * math.pi * 2.0 * t)
            rows.append(
                {
                    "t": t,
                    "target": target,
                    "actual": actual + rng.normal(0, 1e-5),
                    "error": actual - target,
                    "torque": math.nan,
                    "speed": 0.05,
                    "iq": 1.5,
                    "v_cmd": 0.05,
                }
            )
        return rows

    def test_scores_lag_and_creep_smoothness(self) -> None:
        smooth = a4.a4_metrics(self._log(0.1, 0.0), 200.0)
        self.assertAlmostEqual(smooth["lag_ms"], 100.0, delta=6.0)
        self.assertLess(smooth["v_ripple"], 0.05)
        self.assertEqual(smooth["stuck_frac"], 0.0)
        self.assertLess(math.degrees(smooth["band_1_4"]), 0.01)
        self.assertAlmostEqual(smooth["iq_rms"], 1.5)
        # A constant current is a gravity hold, not vibration: no spread, no mode.
        self.assertAlmostEqual(smooth["iq_sd"], 0.0)
        self.assertAlmostEqual(smooth["iq_mode"], 0.0)
        # A 2 Hz ±0.3° wobble on the same creep shows up in the band and ripple.
        wobbly = a4.a4_metrics(self._log(0.1, math.radians(0.3)), 200.0)
        self.assertGreater(math.degrees(wobbly["band_1_4"]), 0.15)
        self.assertGreater(wobbly["v_ripple"], 0.5)


if __name__ == "__main__":
    unittest.main()


class SpeedCapTest(unittest.TestCase):
    def test_zero_track_is_the_fixed_cap(self) -> None:
        self.assertEqual(a4.speed_cap(math.radians(3.0), 60.0, 0.0, 1.0), 60.0)
        self.assertEqual(a4.speed_cap(0.0, 60.0, 0.0, 1.0), 60.0)

    def test_tracking_cap_follows_commanded_speed_with_floor_and_ceiling(self) -> None:
        # 3 deg/s commanded, 1.2× → 3.6 dps, sign-independent.
        self.assertAlmostEqual(a4.speed_cap(math.radians(3.0), 60.0, 1.2, 1.0), 3.6)
        self.assertAlmostEqual(a4.speed_cap(-math.radians(3.0), 60.0, 1.2, 1.0), 3.6)
        # A stationary target keeps the floor so it can still be corrected …
        self.assertEqual(a4.speed_cap(0.0, 60.0, 1.2, 1.0), 1.0)
        # … and the fixed cap remains the ceiling.
        self.assertEqual(a4.speed_cap(math.radians(100.0), 60.0, 1.2, 1.0), 60.0)

    def test_frame_carries_the_per_sample_cap(self) -> None:
        frame = a4._a4_frame(0.0, a4.speed_cap(math.radians(3.0), 60.0, 1.2, 1.0))
        self.assertEqual(int.from_bytes(frame[2:4], "little"), 4)  # 3.6 rounds to 4 dps


class CurrentModeMetricTest(unittest.TestCase):
    def test_mode_current_isolates_the_3_to_8_hz_shudder(self) -> None:
        rate = 200.0
        n = 2400
        t = np.arange(n) / rate
        # 5 Hz, 1 A amplitude on a -8 A gravity hold, plus a 60 Hz 0.3 A buzz.
        iq = (
            -8.0
            + 1.0 * np.sin(2 * math.pi * 5.0 * t)
            + 0.3 * np.sin(2 * math.pi * 60.0 * t)
        )
        log = [
            {
                "t": ti,
                "target": 0.0,
                "actual": 0.0,
                "error": 0.0,
                "v_cmd": 0.1,
                "iq": qi,
            }
            for ti, qi in zip(t, iq)
        ]
        m = a4.a4_metrics(log, rate)
        # sd of the two sines: sqrt(0.5 + 0.045) ≈ 0.738 A
        self.assertAlmostEqual(m["iq_sd"], math.sqrt(0.5 + 0.045), places=2)
        # The 3-8 Hz share is the 5 Hz tone's std alone: 1/sqrt(2) ≈ 0.707 A.
        self.assertAlmostEqual(m["iq_mode"], 1.0 / math.sqrt(2), places=1)
        self.assertLess(m["iq_mode"], m["iq_sd"])
