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
        self.assertLess(
            helper.index('"direct"'),
            helper.index('_write_gains_verified(test, original, "restore")'),
        )
        self.assertIn("set_acceleration(", helper[helper.index('"direct"') :])
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
        i = helper.index("set_acceleration(")
        self.assertIn("0.0, allow_zero=True, position_only=True", helper[i : i + 140])

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


class _AccelMotor:
    """Enough of MyActuatorMotor to exercise set_acceleration."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.slept: list[float] = []

    async def _request(self, data: bytes, timeout: float = 0.1) -> bytes:
        self.frames.append(data)
        return bytes(8)

    async def get_acceleration(self) -> float:
        return 0.0


class FlashWriteSettleTest(unittest.TestCase):
    """0x43 is a flash write and the motor drops commands sent during it.

    Bench: a gain restore sent straight after set_acceleration was acked and
    not applied -- the next run found the homing gain still in the motor,
    compounding 0.24 -> 0.96 -- and a 0xA4 sent straight after it was ignored
    for 15 s while the tool waited."""

    def test_settles_and_reads_back_after_the_last_write(self) -> None:
        m = _AccelMotor()

        async def fake_sleep(s: float) -> None:
            m.slept.append(s)

        orig_sleep = ma.asyncio.sleep
        ma.asyncio.sleep = fake_sleep  # type: ignore[assignment]
        try:
            asyncio.run(ma.MyActuatorMotor.set_acceleration(m, 0.0, allow_zero=True))
        finally:
            ma.asyncio.sleep = orig_sleep  # type: ignore[assignment]
        self.assertEqual(len(m.frames), 4)
        self.assertTrue(all(f[0] == 0x43 for f in m.frames))
        self.assertTrue(m.slept and m.slept[0] >= 0.5)

    def test_position_only_writes_two_frames(self) -> None:
        m = _AccelMotor()
        asyncio.run(
            ma.MyActuatorMotor.set_acceleration(
                m, 0.0, allow_zero=True, position_only=True, settle_s=0.0
            )
        )
        self.assertEqual(
            [f[1] for f in m.frames], [ma._MA_ACC_POS_PLAN, ma._MA_DEC_POS_PLAN]
        )


class VerifiedRestoreTest(unittest.TestCase):
    def test_restores_and_homing_gains_are_read_back_not_assumed(self) -> None:
        helper = inspect.getsource(pl._home_test_with_fallback)
        self.assertNotIn("set_gains(original, persist=False)", helper)
        self.assertIn('_write_gains_verified(test, original, "restore")', helper)
        self.assertIn('"homing gain"', helper)
        verify = inspect.getsource(pl._write_gains_verified)
        self.assertIn("get_gains()", verify)
        self.assertIn("retrying", verify)
        # the tuner never touches the speed planner
        src = inspect.getsource(pl._run) + helper
        for call in [s for s in src.split("set_acceleration(")[1:]]:
            self.assertIn("position_only=True", call[:120])


class HomingGainAndLatchedFaultTest(unittest.TestCase):
    def test_homing_gain_is_a_fixed_safe_band_not_scaled_from_the_motor(self) -> None:
        """Bench: 4x a leftover 0.96 produced a homing gain of 3.84."""
        helper = inspect.getsource(pl._home_test_with_fallback)
        self.assertNotIn("original.position_kp * 4.0", helper)
        self.assertIn("min(max(_HOME_KP_MIN, kp_hint), _HOME_KP_MAX)", helper)
        self.assertLessEqual(pl._HOME_KP_MAX, 0.5)
        self.assertGreaterEqual(pl._HOME_KP_MIN, 0.2)

    def test_a_fault_that_will_not_clear_is_reset_before_homing_again(self) -> None:
        """0x9B did not clear MOTOR_STALL on this firmware and a faulted motor
        ignores 0xA4; the 0x76 reset (mode switch) is what clears it."""
        src = inspect.getsource(pl._run)
        block = src[src.index("while not at_rest:") : src.index("motor status at rest")]
        self.assertIn("set_control_mode(ControlMode.POSITION_VELOCITY)", block)
        self.assertLess(block.index("clear_errors()"), block.index("set_control_mode("))

    def test_leftover_gains_are_called_out_at_the_start(self) -> None:
        src = inspect.getsource(pl._run)
        self.assertIn("is no stock value", src)
        self.assertLess(src.index("is no stock value"), src.index("Homing all joints"))
