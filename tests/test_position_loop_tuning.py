"""The position-loop tuner's safety properties and the gain-restore path.

Hardware-free. What matters here is not the search heuristic but the
invariants: RAM writes while searching, ROM only on --save, holders driven by
impedance rather than the servo under test, and a restore path that actually
writes gains back.
"""

from __future__ import annotations

import argparse
import asyncio
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
