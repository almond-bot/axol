"""tune.position-loop must not leave the motor faulted or the arm raised.

Bench: every run ended with the elbow reporting MOTOR_STALL (bit 0x0002,
latched after STALL_TIME_LIMIT = 1.5 s of being driven without moving). The
teardown homed the test joint FIRST -- under whatever gains the sweep had
just left in RAM, often the ones it stopped at for buzzing, and in
direct-tracking mode -- and only then restored the stored gains and planner.
A loaded elbow asked to travel 90 deg on a buzzing or too-weak loop stalls,
the protection latches, the motor ignores 0xA4, and the next run starts on a
faulted motor. The driver's clear_errors() was a no-op.
"""

from __future__ import annotations

import asyncio
import inspect
import unittest

from almond_axol.cli.tune import position_loop as pl
from almond_axol.motor import myactuator as ma


class TeardownOrderTest(unittest.TestCase):
    def test_home_then_verify_then_clear_then_release(self) -> None:
        """End of a run: home under direct tracking (bench: after this tool's
        acceleration writes the profiled 0xA4 was ignored -- "has not moved
        for 3 s at -100.0°" -- while direct tracking at kp 0.24 brought the
        joint -100 -> -60 -> -21 -> rest), verify, clear the flag, and only
        then release. The planner and gains are restored by the helper once
        the joint is at rest, so _run must not restore them before homing."""
        src = inspect.getsource(pl._run)
        i_home = src.index('print("  Returning to rest ...")')
        i_verify = src.index("while not at_rest:")
        i_clear = src.rindex("await test.motor.clear_errors()")
        i_stop = src.index("await holders.stop()")
        i_disable = src.index("m.disable()")
        self.assertNotIn("RAM gains restored", src[:i_home])
        self.assertNotIn("acceleration restored", src[:i_home])
        self.assertIn("profiled_first=False", src[i_home:i_verify])
        self.assertLess(i_home, i_verify)
        self.assertLess(i_verify, i_clear)
        self.assertLess(i_clear, i_stop)
        self.assertLess(i_stop, i_disable)
        helper = inspect.getsource(pl._home_test_with_fallback)
        # direct tracking is attempted, and the stored planner + gains come
        # back after it, inside the helper
        self.assertLess(helper.index('"direct"'), helper.index("set_gains(original"))
        self.assertIn("set_acceleration(accel_before", helper)
        # the unverified 0x9B is only ever tried after a failed attempt
        self.assertLess(helper.index('"profiled"'), helper.index("clear_errors()"))
        # A raised arm is never released without the operator's word.
        self.assertIn("not releasing", src)
        self.assertIn("'drop'", src)

    def test_homing_is_verified_not_assumed(self) -> None:
        helper = inspect.getsource(pl._home_test_joint)
        self.assertIn("read_position(test)", helper)
        self.assertIn("abs(pos) < math.radians(2.0)", helper)
        # _run never homes the swept joint with a bare fire-and-forget command.
        src = inspect.getsource(pl._run)
        self.assertNotIn("await test.set_position_velocity(0.0", src)
        self.assertIn("_home_test_with_fallback(", src)

    def test_homing_is_observable_and_falls_back_to_direct_tracking(self) -> None:
        """Bench: after the planner and stock gains were restored, a 0xA4 to
        rest was acknowledged and ignored -- the elbow sat at its last target
        for 15 s at constant torque while the tool waited in silence. The
        target is re-sent, progress is logged, a joint that has not moved in
        3 s switches to the mode that homed it all day (direct tracking)."""
        helper = inspect.getsource(pl._home_test_joint) + inspect.getsource(
            pl._home_test_with_fallback
        )
        self.assertIn("has not moved for 3 s", helper)
        self.assertIn('_home_test_joint(test, joint, "profiled")', helper)
        self.assertIn('_home_test_joint(test, joint, "direct")', helper)
        self.assertIn("set_acceleration(0.0, allow_zero=True)", helper)

    def test_headless_launch_never_blocks_on_a_prompt(self) -> None:
        """From the dashboard there is no keyboard: input() would wait
        forever. Keep holding, retry, and let Stop end it."""
        src = inspect.getsource(pl._run)
        self.assertIn("sys.stdin.isatty()", src)
        self.assertIn("SIGTERM", src)
        self.assertIn("retrying homing in 5 s", src)
        # input() is only ever reached on a TTY
        self.assertLess(src.index("if interactive:"), src.index("input,"))


class FaultWatchTest(unittest.TestCase):
    def test_every_gain_point_checks_the_motor_protection(self) -> None:
        src = inspect.getsource(pl._run)
        point = src[src.index("async def point(") : src.index("def persist(")]
        self.assertIn("get_error_code()", point)
        self.assertIn("raise _MotorFaulted", point)
        # Both stages stop the sweep on it rather than measuring a dead motor.
        self.assertEqual(src.count("except _MotorFaulted:"), 2)


class _Capture:
    def __init__(self) -> None:
        self.frames: list[bytes] = []

    async def _request(self, data: bytes, timeout: float = 0.1) -> bytes:
        self.frames.append(data)
        return bytes(8)


class ClearErrorsTest(unittest.TestCase):
    def test_clear_errors_sends_0x9b(self) -> None:
        motor = object.__new__(ma.MyActuatorMotor)
        cap = _Capture()
        motor._request = cap._request  # type: ignore[method-assign]
        asyncio.run(motor.clear_errors())
        self.assertEqual(len(cap.frames), 1)
        self.assertEqual(cap.frames[0][0], 0x9B)

    def test_clear_errors_is_best_effort_on_old_firmware(self) -> None:
        motor = object.__new__(ma.MyActuatorMotor)

        async def dead(data: bytes, timeout: float = 0.1) -> bytes:
            raise ma.MotorError("no reply")

        motor._request = dead  # type: ignore[method-assign]
        asyncio.run(motor.clear_errors())  # must not raise


if __name__ == "__main__":
    unittest.main()


class HomeBeforeSweepTest(unittest.TestCase):
    def test_every_joint_is_homed_before_the_acceleration_is_written(self) -> None:
        """A sweep must start from a known pose. The holders otherwise hold
        whatever they were snapshotted at -- wherever the previous run stopped
        -- and the swept joint approached its sine from there."""
        src = inspect.getsource(pl._run)
        i_holders = src.index("await holders.start()")
        i_home = src.index(
            'print("  Homing all joints to rest (distal to proximal) ...")'
        )
        i_accel_write = src.index("-> writing {args.accel:.0f}")
        i_sweep = src.index("streaming a")
        self.assertLess(i_holders, i_home)
        self.assertLess(i_home, i_accel_write)
        self.assertLess(i_accel_write, i_sweep)
        # Same observable homing at start and end, and a start that fails
        # refuses to sweep rather than guessing the pose.
        self.assertGreaterEqual(src.count("_home_test_with_fallback("), 3)
        self.assertIn("would not come to rest before the sweep", src)
        # The planner is read (not written) before homing so a fallback can restore it.
        self.assertLess(src.index("get_acceleration()"), i_home)
