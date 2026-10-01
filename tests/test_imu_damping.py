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
    c: float,
    *,
    seconds: float = 30.0,
    delay_s: float = 0.012,
    seed: int = 0,
    trip_speed: float = 1.0,
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
    damper = TipDamper(gain=c, columns=(0,), max_torque=2.0, trip_speed=trip_speed)
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
        # A low-frequency runaway (the wrong sign, or far too much gain on
        # this 2 Hz mode) stays under the default 0.3 m/s backstop once the
        # clamp bounds it; a tight flex-speed trip catches it.
        for c in (-10.0, 80.0):
            with self.subTest(c=c):
                rms, d = _structure(c, trip_speed=0.08)
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


class HighBandTripTest(unittest.TestCase):
    def _run(self, amp: float) -> TipDamper:
        d = TipDamper(gain=50.0, columns=(0,), ramp_s=0.5)
        d.start(0.0)
        for i in range(int(5 * 240)):
            t = i / 240.0
            if i % 6 < 5:  # ~200 Hz IMU
                a = amp * math.sin(2 * math.pi * 11.0 * t)
                d.feed(np.array([[t, 0.0, 0.0, G + a, 0.0, 0.0, 0.0]]))
            d.feed_height(t, 0.0)
            d.torque(t, np.array([-0.4, 0, 0, 0, 0, 0, 0]))
        return d

    def test_a_sustained_11hz_drive_trips_and_an_ordinary_one_does_not(self) -> None:
        # The runaway on jelly: ~1.8 m/s² of > 7 Hz acceleration over 2 s;
        # ordinary and fast passes at most ~1.55.
        hot = self._run(4.0)
        self.assertTrue(hot.tripped)
        self.assertIn("high-band acceleration", hot.trip_reason)
        self.assertFalse(self._run(1.5).tripped)


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


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _two_mass(c: float, seconds: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """A joint whose motor sits on its impedance spring (kp 450, kd 5) and
    carries a link through a compliant gearbox that rings at ~2 Hz. The
    encoder sees the motor, the wrist gyro the link. Returns the link's and
    the motor's angle after an initial link deflection, with a
    GyroFlexDamper of gain ``c`` on the motor torque."""
    from almond_axol.tuning.imu_damping import GyroFlexDamper

    fs, j1, j2 = 240.0, 0.05, 0.5
    kp, kd, k = 450.0, 5.0, 80.0
    d = GyroFlexDamper(gains={0: c}, mount=np.eye(3), max_torque=2.0, ramp_s=0.0)
    x1 = v1 = v2 = 0.0
    x2 = math.radians(0.3)
    sub = 10
    dt = 1.0 / (fs * sub)
    d.start(0.0)
    axes = np.zeros((7, 3))
    axes[0] = [0.0, 0.0, 1.0]
    link, motor, gyro = [], [], []
    for i in range(int(seconds * fs)):
        t = i / fs
        gyro.append((t, v2))
        # The IMU's sample is ~8 ms old when the loop reads it.
        arrived = [g for g in gyro if g[0] <= t - 0.008]
        if arrived:
            d.feed(
                np.array(
                    [[arrived[-1][0], 0, 0, G, 0, 0, math.degrees(arrived[-1][1])]]
                )
            )
        d.feed_pose(t, _rot_z(x1), axes)
        tau = d.torque(t)[0]
        for _ in range(sub):
            spring = k * (x2 - x1)
            a1 = (-kp * x1 - kd * v1 + spring + tau) / j1
            a2 = -spring / j2
            v1 += a1 * dt
            x1 += v1 * dt
            v2 += a2 * dt
            x2 += v2 * dt
        link.append(x2)
        motor.append(x1)
    return np.array(link), np.array(motor)


class GyroFlexTest(unittest.TestCase):
    def test_rigid_rotation_is_not_flex(self) -> None:
        from almond_axol.tuning.imu_damping import GyroFlexDamper

        mount = _rot_z(0.4) @ np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        d = GyroFlexDamper(gains={0: 10.0}, mount=mount, delay_s=0.0)
        d.start(0.0)
        axes = np.zeros((7, 3))
        axes[0] = [0.0, 0.0, 1.0]
        peak = 0.0
        for i in range(int(5 * 240)):
            t = i / 240.0
            a = 0.8 * math.sin(2 * math.pi * 0.3 * t)
            w = 0.8 * 2 * math.pi * 0.3 * math.cos(2 * math.pi * 0.3 * t)
            gyro_cam = mount @ np.array([0.0, 0.0, w])  # body rate = world rate about z
            d.feed(np.array([[t, 0, 0, G, *np.degrees(gyro_cam)]]))
            d.feed_pose(t, _rot_z(a), axes)
            if t > 2.0:
                peak = max(peak, abs(d.flex_axis[0]))
        # The encoder rate is a backward difference, half a tick behind the
        # gyro: that residue at 1.5 rad/s is ~0.02 rad/s, the rest is gone.
        self.assertLess(peak, 0.03)

    def test_damps_the_link_mode_the_encoder_cannot_see(self) -> None:
        free, _ = _two_mass(0.0)
        damped, _ = _two_mass(3.0)
        late = slice(240, None)
        self.assertLess(np.std(damped[late]), 0.5 * np.std(free[late]))

    def test_wrong_sign_trips_instead_of_running_away(self) -> None:
        link, _ = _two_mass(-60.0)
        self.assertTrue(np.all(np.isfinite(link)))
        self.assertLess(np.abs(link).max(), math.radians(5.0))

    def test_clamp_and_stale_cutoff(self) -> None:
        from almond_axol.tuning.imu_damping import GyroFlexDamper

        d = GyroFlexDamper(
            gains={3: 100.0}, mount=np.eye(3), max_torque=0.2, ramp_s=0.0, delay_s=0.0
        )
        d.start(0.0)
        axes = np.zeros((7, 3))
        axes[3] = [0.0, 0.0, 1.0]
        for i in range(60):
            t = i / 240.0
            d.feed(np.array([[t, 0, 0, G, 0, 0, 3.0 * math.sin(2 * math.pi * 2 * t)]]))
            d.feed_pose(t, np.eye(3), axes)
            tau = d.torque(t)
            self.assertLessEqual(abs(tau[3]), 0.2 + 1e-12)
            self.assertEqual(tau[0], 0.0)
        # No IMU sample for longer than stale_s: no torque.
        d.feed_pose(1.0, np.eye(3), axes)
        self.assertEqual(float(np.abs(d.torque(1.0)).max()), 0.0)

    def test_fit_mount_recovers_the_camera_rotation(self) -> None:
        from almond_axol.tuning.imu_damping import fit_mount

        true = _rot_z(0.7) @ np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        fs = 240.0
        t = np.arange(int(20 * fs)) / fs
        ax = np.stack(
            [
                np.sin(2 * math.pi * 0.2 * t),
                np.cos(2 * math.pi * 0.13 * t),
                0.5 + 0 * t,
            ],
            1,
        )
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        ang = 0.6 * np.sin(2 * math.pi * 0.25 * t)
        rots = []
        for a, u in zip(ang, ax):
            k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            rots.append(np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k)
        rots = np.array(rots)
        dr = np.einsum("nji,njk->nik", rots[:-1], rots[1:])
        w = (
            0.5
            * np.stack(
                [
                    dr[:, 2, 1] - dr[:, 1, 2],
                    dr[:, 0, 2] - dr[:, 2, 0],
                    dr[:, 1, 0] - dr[:, 0, 1],
                ],
                1,
            )
            * fs
        )
        w = np.vstack([w, w[-1:]])
        gyro = np.degrees(w @ true.T)
        mount, r2 = fit_mount(t, rots, t, gyro)
        self.assertGreater(r2, 0.99)
        self.assertLess(float(np.abs(mount - true).max()), 0.02)


class EncoderTipTest(unittest.TestCase):
    def test_tracking_the_command_is_not_flex_and_a_wobble_is(self) -> None:
        from almond_axol.tuning.imu_damping import EncoderTipDamper

        d = EncoderTipDamper(gain=100.0, columns=(0,), ramp_s=0.0)
        d.start(0.0)
        quiet, shaky = [], []
        for i in range(int(8 * 240)):
            t = i / 240.0
            z_cmd = 0.05 * math.sin(2 * math.pi * 0.2 * t)
            wobble = 0.0005 * math.sin(2 * math.pi * 2.0 * t) if t > 4 else 0.0
            d.feed_command(t, z_cmd)
            d.feed_height(t, z_cmd + wobble)
            tau = d.torque(t, np.array([-0.4, 0, 0, 0, 0, 0, 0]))
            (quiet if 2 < t < 4 else shaky if t > 5 else []).append(d.flex)
            self.assertLessEqual(abs(tau[0]), d.max_torque + 1e-12)
        self.assertLess(np.std(quiet), 1e-6)
        self.assertGreater(np.std(shaky), 0.003)


class ColumnLowPassTest(unittest.TestCase):
    def test_one_channel_is_filtered_and_the_other_is_not(self) -> None:
        from almond_axol.tuning.imu_damping import EncoderTipDamper

        d = EncoderTipDamper(
            gain=100.0, columns=(0, 3), ramp_s=0.0, max_torque=10.0, column_lp={3: 3.0}
        )
        d.start(0.0)
        jac = np.array([-0.4, 0, 0, -0.4, 0, 0, 0])
        taus = []
        for i in range(int(4 * 240)):
            t = i / 240.0
            d.feed_command(t, 0.0)
            d.feed_height(t, 0.0005 * math.sin(2 * math.pi * 11.0 * t))
            taus.append(d.torque(t, jac))
        taus = np.array(taus[240:])
        # 11 Hz through a 3 Hz pole: ~0.26 of the unfiltered channel.
        ratio = np.std(taus[:, 3]) / np.std(taus[:, 0])
        self.assertLess(ratio, 0.35)
        self.assertGreater(ratio, 0.15)


if __name__ == "__main__":
    unittest.main()
