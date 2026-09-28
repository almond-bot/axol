"""Wrist-IMU damping: the estimator, the damper against a simulated hidden
structural mode (joint encoders cannot see it), and its self-protection."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.signal import butter, sosfilt

from almond_axol.tuning.imu_damping import (
    G,
    EncoderVelocity,
    TipDamper,
    VerticalVelocity,
)


def _structure(
    c: float, *, seconds: float = 30.0, delay_s: float = 0.012, seed: int = 0
):
    """Tool height on a 2 Hz, ζ 0.05 mode (2 kg), shaken by a 1-6 Hz force,
    pushed by one joint's torque through a 0.6 m lever. The IMU is tilted,
    noisy, delayed and at 200 Hz; control ticks at 250 Hz."""
    fs = 1000
    dt = 1 / fs
    m, w, zeta, lever = 2.0, 2 * math.pi * 2.0, 0.05, 0.6
    k, b = m * w * w, 2 * zeta * m * w
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    d = (
        sosfilt(
            butter(2, [1, 6], btype="band", fs=fs, output="sos"), rng.standard_normal(n)
        )
        * 3
    )
    damper = TipDamper(gain=c, columns=(0,), max_torque=2.0)
    damper.start(0.0)
    tilt = np.array([0.1, 0.05, 1.0]) / np.linalg.norm([0.1, 0.05, 1.0])
    z = v = tau = 0.0
    zs, accs = [], []
    for i in range(n):
        t = i * dt
        a = (-k * z - b * v + d[i] + tau / lever) / m
        v += a * dt
        z += v * dt
        zs.append(z)
        accs.append(a)
        if i % 5 == 0:
            j = max(0, i - int(delay_s * fs))
            meas = tilt * (G + accs[j]) + 0.02 * rng.standard_normal(3)
            damper.feed(np.array([[t - delay_s, *meas, 0, 0, 0]]))
        if i % 4 == 0:
            tau = damper.torque(t, np.array([lever, 0, 0, 0, 0, 0, 0]))[0]
    return float(np.std(zs[5 * fs :])), damper


class DamperTest(unittest.TestCase):
    def test_damps_a_mode_the_encoders_cannot_see(self) -> None:
        base, _ = _structure(0.0)
        damped, d = _structure(20.0)
        self.assertLess(damped, 0.6 * base)
        self.assertFalse(d.tripped)

    def test_too_much_gain_or_the_wrong_sign_trips_instead_of_running_away(
        self,
    ) -> None:
        base, _ = _structure(0.0)
        for c in (-10.0, 80.0):
            with self.subTest(c=c):
                rms, d = _structure(c)
                self.assertTrue(d.tripped)
                self.assertLess(rms, 6 * base)  # bounded by the trip (unbounded: ~40x)

    def test_clamp_ramp_and_stale_cutoff(self) -> None:
        d = TipDamper(
            gain=1e4, columns=(0, 3), max_torque=0.5, ramp_s=1.0, trip_speed=10.0
        )
        d.start(0.0)
        # At rest for 1.5 s, then pushed upward for 0.1 s: the band velocity
        # is positive (up).
        t = 0.0
        for i in range(320):
            t += 0.005
            push = 2.0 if i >= 300 else 0.0
            d.feed(np.array([[t, 0.0, 0.0, G + push, 0, 0, 0]]))
        jac = np.array([0.6, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.assertGreater(d.estimator.value, 0.0)
        early = d.torque(0.1, jac)
        late = d.torque(t, jac)
        self.assertLess(abs(early[0]), abs(late[0]) + 1e-12)  # ramping in
        self.assertLessEqual(np.abs(late).max(), 0.5 + 1e-12)
        self.assertLess(late[0], 0.0)  # opposes the upward velocity
        self.assertEqual(late[1], 0.0)  # only the chosen columns
        self.assertTrue(np.all(d.torque(t + 0.2, jac) == 0.0))  # IMU went quiet

    def test_deliberate_slow_motion_is_not_shake(self) -> None:
        # A 0.3 Hz, 50 mm vertical oscillation of the tool (deliberate motion):
        # the band velocity stays small next to its 94 mm/s peak.
        est = VerticalVelocity()
        fs = 200.0
        out = []
        for i in range(int(40 * fs)):
            t = i / fs
            acc_z = -0.05 * (2 * math.pi * 0.3) ** 2 * math.sin(2 * math.pi * 0.3 * t)
            out.append(est.update(t, np.array([0.0, 0.0, G + acc_z])))
        self.assertLess(np.abs(out[int(10 * fs) :]).max(), 0.008)
        # A 2 Hz, 1 mm shake passes at nearly full velocity (12.6 mm/s).
        est.reset()
        out = []
        for i in range(int(20 * fs)):
            t = i / fs
            acc_z = -0.001 * (2 * math.pi * 2) ** 2 * math.sin(2 * math.pi * 2 * t)
            out.append(est.update(t, np.array([0.0, 0.0, G + acc_z])))
        self.assertGreater(np.abs(out[int(5 * fs) :]).max(), 0.009)


class FlexTest(unittest.TestCase):
    def test_motion_the_encoders_see_is_not_flex(self) -> None:
        # The tool follows a 0.8 Hz, 20 mm deliberate move plus a 3 Hz, 1 mm
        # servo wobble — all visible to the encoders: the flex stays near 0
        # while the IMU velocity alone would not.
        # This IMU has no latency, so the encoder path is not delayed.
        d = TipDamper(
            gain=10.0,
            columns=(0,),
            trip_speed=10.0,
            encoder=EncoderVelocity(delay_s=0.0),
        )
        d.start(0.0)
        imu_v, flex = [], []
        for i in range(int(20 * 240)):
            t = i / 240.0
            z = 0.02 * math.sin(2 * math.pi * 0.8 * t) + 0.001 * math.sin(
                2 * math.pi * 3 * t
            )
            acc = -0.02 * (2 * math.pi * 0.8) ** 2 * math.sin(
                2 * math.pi * 0.8 * t
            ) - 0.001 * (2 * math.pi * 3) ** 2 * math.sin(2 * math.pi * 3 * t)
            if i % 6 < 5:  # ~200 Hz IMU
                d.feed(np.array([[t, 0.0, 0.0, G + acc, 0.0, 0.0, 0.0]]))
            d.feed_height(t, z)
            if i > 5 * 240:
                imu_v.append(d.estimator.value)
                flex.append(d.flex)
        self.assertGreater(np.std(imu_v), 0.01)
        self.assertLess(np.std(flex), 0.2 * np.std(imu_v))

    def test_gyro_keeps_a_rotating_camera_from_reading_as_shake(self) -> None:
        # A camera turning at 100°/s about a horizontal axis, not translating:
        # the accelerometer sees gravity swing through its axes.
        rate = math.radians(100.0)
        fs = 200.0
        results = {}
        for use_gyro in (False, True):
            est = VerticalVelocity()
            out = []
            for i in range(int(6 * fs)):
                t = i / fs
                th = 0.5 * math.sin(rate / 0.5 * t)  # swings ±29° at up to 100°/s
                w = rate * math.cos(rate / 0.5 * t)
                acc = G * np.array([math.sin(th), 0.0, math.cos(th)])
                gyro = np.array([0.0, -math.degrees(w), 0.0])
                out.append(est.update(t, acc, gyro if use_gyro else None))
            results[use_gyro] = float(np.std(out[int(2 * fs) :]))
        self.assertLess(results[True], 0.3 * results[False])


if __name__ == "__main__":
    unittest.main()
