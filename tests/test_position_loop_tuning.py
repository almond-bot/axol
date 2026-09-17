"""The position-loop tuner's safety properties and the gain-restore path.

Hardware-free. What matters here is not the search heuristic but the
invariants: RAM writes while searching, ROM only on --save, holders driven by
impedance rather than the servo under test, and a restore path that actually
writes gains back.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

from almond_axol.cli.tune import position_loop as pl
from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.motor.types import MotorGains


class PersistSemanticsTest(unittest.TestCase):
    """RAM vs ROM is the safety property this whole command rests on."""

    def test_ram_and_rom_use_different_commands(self) -> None:
        from almond_axol.motor.myactuator import (
            _MA_WRITE_GAINS_RAM,
            _MA_WRITE_GAINS_ROM,
        )

        self.assertEqual(_MA_WRITE_GAINS_RAM, 0x31)
        self.assertEqual(_MA_WRITE_GAINS_ROM, 0x32)
        self.assertNotEqual(_MA_WRITE_GAINS_RAM, _MA_WRITE_GAINS_ROM)

    def test_set_gains_defaults_to_persisting(self) -> None:
        # A caller that does not opt in must still get the old behaviour.
        import inspect

        from almond_axol.motor.motor import Motor
        from almond_axol.motor.myactuator import MyActuatorMotor

        for cls in (Motor, MyActuatorMotor):
            sig = inspect.signature(cls.set_gains)
            self.assertIs(sig.parameters["persist"].default, True, cls.__name__)
            self.assertEqual(
                sig.parameters["persist"].kind, inspect.Parameter.KEYWORD_ONLY
            )

    def test_damiao_accepts_persist_for_parity(self) -> None:
        import inspect

        from almond_axol.motor.damiao import DamiaoMotor

        self.assertIn("persist", inspect.signature(DamiaoMotor.set_gains).parameters)


class HoldersTest(unittest.TestCase):
    def test_holders_use_impedance_not_the_servo_under_test(self) -> None:
        # The premise of the command is that the position servo cannot hold a
        # loaded joint, so the holders must not be driven with it.
        import inspect

        src = inspect.getsource(pl._Holders)
        self.assertIn("set_impedance", src)
        self.assertNotIn("set_position_velocity", src)

    def test_holders_exclude_the_test_joint_and_track_wobble(self) -> None:
        from almond_axol.robot.config import AxolConfig

        motors = {}
        for j in ARM_JOINTS:
            m = MagicMock()
            m.get_position = AsyncMock(return_value=0.0)
            m.set_impedance = AsyncMock()
            m.position = 0.0
            motors[j] = m
        h = pl._Holders(motors, Joint.ELBOW, True, AxolConfig().resolved())
        asyncio.run(h.start())
        asyncio.run(h.stop())
        self.assertNotIn(Joint.ELBOW, h.peak_wobble)
        self.assertEqual(set(h.peak_wobble), set(ARM_JOINTS) - {Joint.ELBOW})


class SearchGuardsTest(unittest.TestCase):
    def test_escalation_starts_at_the_current_value(self) -> None:
        # x1 first: a joint already correct must not be stepped up at all.
        self.assertEqual(pl._KP_STEPS[0], 1.0)
        self.assertEqual(list(pl._KP_STEPS), sorted(pl._KP_STEPS))

    def test_ripple_limit_is_below_the_sag_tolerance(self) -> None:
        # Oscillation has to be caught before it can masquerade as a small
        # mean sag: a joint ringing +-0.3 deg averages to zero error.
        self.assertLess(pl._RIPPLE_LIMIT_DEG, pl._SAG_OK_DEG)

    def test_defaults_do_not_write_rom(self) -> None:
        parser = argparse.ArgumentParser()
        pl.add_parser(parser.add_subparsers())
        args = parser.parse_args(["tune.position-loop", "--r", "--joint", "elbow"])
        self.assertFalse(args.save)


class TeardownTest(unittest.TestCase):
    """The arm must be returned to rest before torque comes off.

    Disabling from wherever the probe finished drops the arm. This shipped
    broken once already in tune.breakaway; the same mistake reached here.
    """

    def test_homes_before_disabling(self) -> None:
        import inspect

        src = inspect.getsource(pl._run)
        block = src[src.index("finally:") :]
        self.assertIn("Returning to rest", block)
        self.assertLess(block.index("_HOME_ORDER"), block.index("m.disable()"))

    def test_home_order_is_distal_first_and_complete(self) -> None:
        self.assertEqual(set(pl._HOME_ORDER), set(ARM_JOINTS))
        self.assertEqual(pl._HOME_ORDER[0], Joint.WRIST_3)
        self.assertEqual(pl._HOME_ORDER[-1], Joint.SHOULDER_1)

    def test_holders_keep_streaming_while_their_target_moves(self) -> None:
        # A one-shot position command would leave the joint unsupported
        # mid-move; ramp_to only moves the target the hold loop is chasing.
        import inspect

        src = inspect.getsource(pl._Holders.ramp_to)
        self.assertIn("self._hold[joint]", src)
        self.assertNotIn("set_position_velocity", src)

    def test_ramp_to_walks_the_target_and_lands_on_it(self) -> None:
        from almond_axol.robot.config import AxolConfig

        motors = {}
        for j in ARM_JOINTS:
            m = MagicMock()
            m.get_position = AsyncMock(return_value=0.5)
            m.set_impedance = AsyncMock()
            m.position = 0.5
            motors[j] = m
        h = pl._Holders(motors, Joint.ELBOW, True, AxolConfig().resolved())

        async def go():
            await h.start()
            await h.stop()  # no streaming needed to exercise the target walk
            await h.ramp_to(Joint.WRIST_1, 0.0, 5.0)
            return h._hold[Joint.WRIST_1]

        self.assertAlmostEqual(asyncio.run(go()), 0.0)


class AccelerationTest(unittest.TestCase):
    """Zero is the value that selects direct tracking, and the clamp hid it."""

    def test_zero_is_clamped_unless_explicitly_allowed(self) -> None:
        import inspect

        from almond_axol.motor.myactuator import MyActuatorMotor

        sig = inspect.signature(MyActuatorMotor.set_acceleration)
        self.assertIs(sig.parameters["allow_zero"].default, False)
        src = inspect.getsource(MyActuatorMotor.set_acceleration)
        # The clamp still applies to every value except an explicit zero.
        self.assertIn("allow_zero and dps_s2 == 0", src)

    def test_track_mode_is_opt_in(self) -> None:
        parser = argparse.ArgumentParser()
        pl.add_parser(parser.add_subparsers())
        args = parser.parse_args(["tune.position-loop", "--r", "--joint", "elbow"])
        self.assertEqual(args.mode, "hold")
        self.assertIsNone(args.accel)

    def test_tracking_reads_position_explicitly_not_from_the_mit_cache(self) -> None:
        """The first hardware run returned all-NaN because of this.

        `motor.position` is fed by MIT impedance replies. 0xA4 answers on the
        0x240 control frame, so under a position-mode stream that cache is
        never populated and every sample raises.
        """
        import inspect

        src = inspect.getsource(pl._track)
        self.assertIn("await motor.get_position()", src)
        self.assertNotIn("motor.motor.position", src)

    def test_tracking_reaches_the_sine_start_before_timing(self) -> None:
        """Otherwise the first gain in a sweep measures the approach.

        Observed on hardware: kp=0.96 read 0.126 deg when it ran third and
        6.12 deg when it ran first, because the sine simply began commanding
        `center` from wherever the arm was parked and the 40 deg journey
        filled the window. The artefact attaches to whichever gain is
        measured first, which makes a sweep read backwards.
        """
        import inspect

        src = inspect.getsource(pl._track)
        approach = src[: src.index("t0 = time.monotonic()")]
        self.assertIn("set_position_velocity(start", approach)
        self.assertIn("_APPROACH_TOL_RAD", approach)
        # And it must not measure anyway: an unreached start yields no row.
        self.assertIn("no result for this pass", src)
        tail = src[src.index("_APPROACH_MAX_S:.0f}s") :]
        self.assertIn('return float("nan")', tail[: tail.index("dt = 1.0 / rate_hz")])

    def test_drive_starts_at_rest_so_the_approach_leaves_no_step(self) -> None:
        """The approach parks the joint at a standstill. A sine about the
        centre would then demand peak velocity at t=0, forcing a stiction
        breakaway inside the measured window; on the right elbow that cost
        23 % of the reported rms at position_kp 0.96 (0.126 -> 0.155 deg) and
        scaled inversely with gain, flattering high gains. A cosine from a
        turning point starts where the approach left off, at rest."""
        import inspect

        src = inspect.getsource(pl._track)
        self.assertIn("start = center - amp", src)
        self.assertIn("amp * math.cos(", src)

        # The demanded velocity at t=0 must be zero, not peak.
        center, amp, freq = -0.785, math.radians(10.0), 0.2

        def drive(t):
            return center - amp * math.cos(2.0 * math.pi * freq * t)

        dt = 1e-4
        self.assertAlmostEqual(drive(0.0), center - amp, places=9)
        self.assertLess(abs((drive(dt) - drive(0.0)) / dt), 1e-3)
        # ...and it still spans the full commanded amplitude.
        span = max(drive(t / 100) for t in range(0, 501)) - min(
            drive(t / 100) for t in range(0, 501)
        )
        self.assertAlmostEqual(span, 2 * amp, places=6)

    def test_tracking_sweep_tunes_the_integral_after_the_proportional(self) -> None:
        """kp alone cannot remove a velocity-following error.

        The measured law on the right elbow was rms = 0.0794/kp + 0.0404 deg
        and lag = 8.70/kp + 4.07 ms, so at kp=0.84 the P-loop lag is 10.3 ms
        against a 4.07 ms transport floor -- 70 % of the error is the term a
        type-1 loop cannot avoid. Raising kp to chase it runs into a cliff
        that moves with load (above 0.84 at -45 deg, below 0.72 at -90 deg),
        so the integral has to be swept too.
        """
        import inspect

        src = inspect.getsource(pl._run)
        # The track branch: from its banner to where the hold branch starts.
        track = src[src.index("streaming a") :]
        head = track[: track.index("{'sag':>9}")]
        self.assertIn("position_ki", head)
        # Ascending, and stopping on ripple: an integral winding up against
        # stiction limit-cycles rather than diverging.
        self.assertEqual(list(pl._KI_STEPS), sorted(pl._KI_STEPS))
        self.assertIn("winding up", head)
        # The ladder is relative to the winning kp, not absolute.
        self.assertTrue(all(0 < s <= 0.1 for s in pl._KI_STEPS))

    def test_approach_tolerance_is_tight_against_the_error_being_measured(self) -> None:
        # Tracking errors of interest are ~0.1 deg, so starting half a degree
        # off is already several times the signal.
        self.assertLessEqual(math.degrees(pl._APPROACH_TOL_RAD), 0.5)
        self.assertGreater(pl._APPROACH_MAX_S, 5.0)

    def test_tracking_ripple_separates_buzz_from_following_error(self) -> None:
        """A raised gain trades tracking error for vibration, and rms keeps
        falling right through it — an operator hears the buzz first. The
        ripple window must pass the drive sine and keep the buzz."""
        import numpy as np

        rate = 100.0
        k = max(3, int(0.15 * rate) | 1)

        def ripple(err):
            sm = np.convolve(err, np.ones(k) / k, mode="same")
            return float((err - sm)[k:-k].std())

        t = np.linspace(0, 15, 1500)
        smooth = 0.4 * np.sin(2 * np.pi * 0.2 * t)
        # The metric's job is separation, not clearing a threshold: the
        # threshold is uncalibrated and deliberately permissive.
        self.assertLess(ripple(smooth), 0.005)
        buzzy = smooth + 0.15 * np.sin(2 * np.pi * 18 * t)
        self.assertGreater(ripple(buzzy), 20 * ripple(smooth))
        self.assertGreater(ripple(buzzy), 0.05)

    def test_ripple_limit_is_permissive_and_overridable(self) -> None:
        """An uncalibrated threshold must not truncate the sweep.

        A tight default stopped the sweep at a gain the operator could not
        hear, which is exactly the run that would have calibrated it. The
        relative jump is the detector that needs no absolute level.
        """
        parser = argparse.ArgumentParser()
        pl.add_parser(parser.add_subparsers())
        args = parser.parse_args(["tune.position-loop", "--r", "--joint", "elbow"])
        self.assertEqual(args.ripple_limit, pl._TRACK_RIPPLE_LIMIT_DEG)
        # Permissive enough to clear the 0.088 deg that measured quiet.
        self.assertGreater(pl._TRACK_RIPPLE_LIMIT_DEG, 0.09)
        self.assertGreater(pl._TRACK_RIPPLE_JUMP, 1.0)

        over = parser.parse_args(
            ["tune.position-loop", "--r", "--joint", "elbow", "--ripple-limit", "0.04"]
        )
        self.assertEqual(over.ripple_limit, 0.04)

    def test_tracking_metric_reports_lag(self) -> None:
        # A held position cannot reveal profiled-motion mode; only a moving
        # target can, so the tracking path must report a following error.
        import inspect

        src = inspect.getsource(pl._track)
        self.assertIn("set_position_velocity", src)
        self.assertIn("lag", src)


class RestorePathTest(unittest.TestCase):
    """`motor.restore-config` recorded loop gains but never wrote them back."""

    def _motor(self, **over):
        g = MotorGains(
            speed_kp=1.0,
            speed_ki=2.0,
            position_kp=0.008,
            position_ki=0.0,
            position_kd=0.1,
            current_kp=3.0,
            current_ki=4.0,
        )
        m = MagicMock()
        m.get_gains = AsyncMock(return_value=replace(g, **over))
        m.set_gains = AsyncMock()
        return m

    def test_writes_saved_gains_back(self) -> None:
        from almond_axol.cli.motor import restore_config as rc

        m = self._motor()
        asyncio.run(rc._restore_loop_gains(m, {"loop_gains": {"position_kp": 12.5}}))
        self.assertEqual(m.set_gains.await_count, 1)
        self.assertEqual(m.set_gains.await_args.args[0].position_kp, 12.5)
        # Untouched fields keep their live values.
        self.assertEqual(m.set_gains.await_args.args[0].speed_kp, 1.0)

    def test_no_write_when_already_matching(self) -> None:
        from almond_axol.cli.motor import restore_config as rc

        m = self._motor()
        asyncio.run(rc._restore_loop_gains(m, {"loop_gains": {"position_kp": 0.008}}))
        self.assertEqual(m.set_gains.await_count, 0)

    def test_snapshot_without_gains_is_a_noop(self) -> None:
        from almond_axol.cli.motor import restore_config as rc

        m = self._motor()
        asyncio.run(rc._restore_loop_gains(m, {}))
        self.assertEqual(m.get_gains.await_count, 0)
        self.assertEqual(m.set_gains.await_count, 0)


if __name__ == "__main__":
    unittest.main()
