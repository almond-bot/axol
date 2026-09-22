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


class PoseAndHeldJointsTest(unittest.TestCase):
    def test_pose_parses_validates_and_mirrors_shoulder_2_outboard(self) -> None:
        from almond_axol.constants import Joint

        pose = a4.parse_pose(["shoulder_1=-90", "elbow=-75"], Joint.SHOULDER_2, False)
        self.assertAlmostEqual(pose[Joint.SHOULDER_1], math.radians(-90))
        self.assertAlmostEqual(pose[Joint.ELBOW], math.radians(-75))
        self.assertEqual(a4.parse_pose(None, Joint.ELBOW, True), {})
        with self.assertRaisesRegex(SystemExit, "outside"):
            a4.parse_pose(["shoulder_1=120"], Joint.ELBOW, False)  # right limit is +90
        with self.assertRaisesRegex(SystemExit, "test joint"):
            a4.parse_pose(["elbow=10"], Joint.ELBOW, False)
        with self.assertRaisesRegex(SystemExit, "outboard"):
            a4.parse_pose(
                ["shoulder_2=-10"], Joint.SHOULDER_3, False
            )  # right: positive
        with self.assertRaisesRegex(SystemExit, "unknown joint"):
            a4.parse_pose(["hip=1"], Joint.ELBOW, True)

    def test_held_summary_scores_drift_and_oscillation(self) -> None:
        t = np.arange(0, 4.0, 1 / 80)
        # shoulder_2 oscillating at 2.5 Hz, ±0.5°, around a +10° hold.
        s2 = math.radians(10) + math.radians(0.5) * np.sin(2 * math.pi * 2.5 * t)
        # elbow let go: parked 6° below its 0° hold, no oscillation.
        el = np.full_like(t, math.radians(-6.0))
        out = a4.held_summary(
            {
                "shoulder_2": list(zip(t, s2)),
                "elbow": list(zip(t, el)),
                "wrist_1": [(0.0, 0.0)],
            },
            {"shoulder_2": math.radians(10), "elbow": 0.0},
        )
        self.assertNotIn("wrist_1", out)  # too few samples to score
        self.assertAlmostEqual(out["shoulder_2"]["hz"], 2.5, delta=0.3)
        self.assertAlmostEqual(out["shoulder_2"]["p2p"], 1.0, delta=0.05)
        self.assertAlmostEqual(out["shoulder_2"]["drift"], 0.0, delta=0.05)
        self.assertAlmostEqual(out["elbow"]["drift"], -6.0, places=6)
        self.assertEqual(out["elbow"]["std"], 0.0)


class RingPowerTest(unittest.TestCase):
    """Finding the joint that feeds a ring every held joint shares."""

    def _joint(
        self, phase: float, fs: float = 33.0, hz: float = 4.25, jitter: bool = True
    ) -> list[tuple[float, float, float]]:
        # Round-robin reads land unevenly; the scorer resamples.
        rng = np.random.default_rng(0)
        t = np.arange(0, 12.0, 1 / fs)
        if jitter:
            t = t + rng.uniform(0, 0.3 / fs, len(t))
        w = 0.1 * np.sin(2 * math.pi * hz * t)
        # Gravity's DC and the wave's slow content must not count.
        tau = 2.0 + 0.3 * np.sin(2 * math.pi * 0.07 * t)
        tau = tau + 0.5 * np.sin(2 * math.pi * hz * t + phase)
        return list(zip(t, w, tau))

    def test_the_driving_joint_is_the_only_positive_one(self) -> None:
        out = a4.ring_power(
            {
                "wrist_3": self._joint(0.0),  # torque in phase: drives
                "shoulder_2": self._joint(math.pi),  # opposes velocity: damps
                "elbow": self._joint(math.pi / 2),  # spring-like: no net power
            },
            4.25,
        )
        self.assertGreater(out["wrist_3"]["power_w"], 0.02)
        self.assertAlmostEqual(out["wrist_3"]["cos_phi"], 1.0, delta=0.05)
        self.assertLess(out["shoulder_2"]["power_w"], -0.02)
        self.assertAlmostEqual(out["shoulder_2"]["cos_phi"], -1.0, delta=0.05)
        self.assertAlmostEqual(out["elbow"]["cos_phi"], 0.0, delta=0.1)
        # 0.1 rad/s and 0.5 Nm amplitudes come back through the band-pass.
        self.assertAlmostEqual(out["wrist_3"]["vel_amp"], 0.1, delta=0.01)
        self.assertAlmostEqual(out["wrist_3"]["tau_amp"], 0.5, delta=0.05)

    def test_short_or_too_slow_joints_are_left_out(self) -> None:
        out = a4.ring_power(
            {
                "short": self._joint(0.0)[:10],
                "slow": self._joint(0.0, fs=8.0, jitter=False),  # Nyquist 4 Hz
            },
            4.25,
        )
        self.assertEqual(out, {})

    def test_ring_hz_is_the_most_moving_oscillating_joint(self) -> None:
        scores = {
            "shoulder_2": {"std": 0.075, "hz": 4.26},
            "wrist_3": {"std": 0.167, "hz": 4.25},
            "shoulder_1": {"std": 0.006, "hz": 6.9},  # quiet: not a ring
        }
        self.assertEqual(a4.ring_hz(scores), 4.25)
        self.assertIsNone(a4.ring_hz({"elbow": {"std": 0.01, "hz": 4.0}}))
        self.assertIsNone(a4.ring_hz({"elbow": {"std": 0.2, "hz": math.nan}}))

    def test_held_series_keys_each_joint_on_its_own_time_base(self) -> None:
        out = a4.held_series(
            {"elbow": [(0.0, 1.0), (0.1, 1.1)], "wrist_1": []},
            {"elbow": [(0.05, 0.2, 3.0)]},
        )
        self.assertEqual(
            sorted(out),
            [
                "held_elbow_dyn_t",
                "held_elbow_pos",
                "held_elbow_pos_t",
                "held_elbow_tau",
                "held_elbow_vel",
            ],
        )
        np.testing.assert_array_equal(out["held_elbow_pos"], [1.0, 1.1])
        np.testing.assert_array_equal(out["held_elbow_tau"], [3.0])


class DamiaoPathTest(unittest.TestCase):
    def test_dm_frame_is_two_little_endian_floats_in_rad_units(self) -> None:
        import struct

        frame = a4.dm_frame(1.25, 60.0)
        self.assertEqual(len(frame), 8)
        p, v = struct.unpack("<ff", frame)
        self.assertAlmostEqual(p, 1.25, places=6)
        self.assertAlmostEqual(v, math.radians(60.0), places=5)
        # A negative cap cannot be sent: the profiler's v_des is a magnitude.
        self.assertEqual(struct.unpack("<ff", a4.dm_frame(0.0, -5.0))[1], 0.0)

    def test_damiao_gain_registers_match_the_manual(self) -> None:
        # DM-J4310 register map: 0x19 KP_ASR, 0x1A KI_ASR, 0x1B KP_APR, 0x1C KI_APR.
        self.assertEqual(
            a4._DM_GAIN_REGS,
            {
                "speed_kp": 0x19,
                "speed_ki": 0x1A,
                "position_kp": 0x1B,
                "position_ki": 0x1C,
            },
        )
        self.assertEqual((a4._DM_REG_ACC, a4._DM_REG_DEC, a4._DM_REG_PM), (4, 5, 0x50))


class HeldSamplingTest(unittest.TestCase):
    """``_stream``'s round-robin reads of the held joints, on fake drivers."""

    def test_turns_alternate_position_and_dynamics_without_touching_the_wave(
        self,
    ) -> None:
        import asyncio
        import struct
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock

        from almond_axol.constants import Joint
        from almond_axol.motor.damiao import DamiaoMotor
        from almond_axol.motor.myactuator import MyActuatorMotor

        def myactuator() -> MagicMock:
            d = MagicMock(spec=MyActuatorMotor)
            d._kt = 2.0

            async def request(frame: bytes) -> bytes:
                if frame[0] == 0xA4:  # the wave: iq 1.5 A, 10 dps
                    return bytes([0xA4, 0]) + struct.pack("<hhh", 150, 10, 5)
                if frame[0] == 0x92:
                    return bytes([0x92, 0, 0, 0]) + struct.pack("<i", 1234)
                if frame[0] == 0x9C:  # held: iq -2.5 A, 7 dps
                    return bytes([0x9C, 30]) + struct.pack("<hhh", -250, 7, 0)
                raise AssertionError(hex(frame[0]))

            d._request = AsyncMock(side_effect=request)
            return d

        damiao = MagicMock(spec=DamiaoMotor)
        damiao._read_register = AsyncMock(return_value=0.5)
        damiao._request_feedback = AsyncMock(
            return_value=SimpleNamespace(velocity=0.3, torque=-0.4, position=0.5)
        )

        def held(driver: MagicMock) -> SimpleNamespace:
            return SimpleNamespace(
                motor=SimpleNamespace(_driver=driver), frame_offset=0.0
            )

        guard = MagicMock()
        guard.feed.return_value = None
        log, reason, pos, dyn = asyncio.run(
            a4._stream(
                SimpleNamespace(frame_offset=0.0),
                myactuator(),
                [(i / 400, 0.0, 0.0) for i in range(40)],
                60.0,
                400.0,
                guard,
                MagicMock(),
                held={Joint.ELBOW: held(myactuator()), Joint.WRIST_2: held(damiao)},
            )
        )
        self.assertIsNone(reason)
        # 40 ticks over 2 joints x 2 kinds: 10 of each, nothing double-counted.
        self.assertEqual(
            {k: len(v) for k, v in pos.items()}, {"elbow": 10, "wrist_2": 10}
        )
        self.assertEqual(
            {k: len(v) for k, v in dyn.items()}, {"elbow": 10, "wrist_2": 10}
        )
        _, vel, tau = dyn["elbow"][0]
        self.assertAlmostEqual(vel, math.radians(7))
        self.assertAlmostEqual(tau, -2.5 * 2.0)  # iq x kt
        self.assertEqual(dyn["wrist_2"][0][1:], (0.3, -0.4))
        # The held reads never leak into the wave's own iq/speed.
        for row in log:
            self.assertAlmostEqual(row["iq"], 1.5)
            self.assertAlmostEqual(row["speed"], math.radians(10))
            guard.feed.assert_any_call(row["actual"], 1.5)


class HeldGainTest(unittest.TestCase):
    """``--held-gain``: RAM gains for the joints held during another's wave."""

    def test_joint_or_side_qualified_specs_group_by_joint(self) -> None:
        from almond_axol.constants import Joint

        out = a4.parse_held_gains(
            [
                "shoulder_2.position_kp=0.5",
                "right.shoulder_2.speed_kp=0.06",
                "wrist_2.position_kp=200",
            ],
            Joint.SHOULDER_3,
            False,
        )
        self.assertEqual(
            out,
            {
                Joint.SHOULDER_2: {"position_kp": 0.5, "speed_kp": 0.06},
                Joint.WRIST_2: {"position_kp": 200.0},
            },
        )
        self.assertEqual(a4.parse_held_gains(None, Joint.ELBOW, True), {})

    def test_refuses_what_the_run_cannot_apply(self) -> None:
        from almond_axol.constants import Joint

        bad = {
            "shoulder_2.position_kp": r"not \[SIDE\.\]JOINT",
            "position_kp=0.5": r"not \[SIDE\.\]JOINT",
            "left.shoulder_2.position_kp=0.5": "this run is right",
            "hip.position_kp=1": "unknown joint",
            "gripper.position_kp=1": "not an arm joint",
            "shoulder_3.position_kp=1": "is the test joint",
            "elbow.bogus=1": "unknown gain",
            "elbow.position_kp=fast": "bad value",
            # The Damiao wrists' pv loop has no position D or current loop.
            "wrist_3.position_kd=0.1": "Damiao motor",
            "wrist_2.current_kp=1": "Damiao motor",
        }
        for spec, message in bad.items():
            with self.subTest(spec=spec), self.assertRaisesRegex(SystemExit, message):
                a4.parse_held_gains([spec], Joint.SHOULDER_3, False)

    def test_the_flag_is_repeatable_on_the_cli(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        a4.add_parser(parser.add_subparsers())
        ns = parser.parse_args(
            [
                "tune.a4",
                "--r",
                "--joint",
                "shoulder_3",
                "--held-gain",
                "shoulder_2.position_kp=0.5",
                "--held-gain",
                "wrist_2.position_kp=200",
            ]
        )
        self.assertEqual(
            ns.held_gain, ["shoulder_2.position_kp=0.5", "wrist_2.position_kp=200"]
        )
