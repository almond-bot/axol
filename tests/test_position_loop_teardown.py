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
    def test_restore_then_home_then_verify_then_clear_then_release(self) -> None:
        src = inspect.getsource(pl._run)
        i_restore = src.index("RAM gains restored")
        i_accel = src.index("acceleration restored")
        i_home = src.index('print("  Returning to rest ...")')
        i_verify = src.index("while not at_rest:")
        i_clear = src.rindex("await test.motor.clear_errors()")
        i_stop = src.index("await holders.stop()")
        i_disable = src.index("m.disable()")
        self.assertLess(i_accel, i_home)
        self.assertLess(i_restore, i_home)
        self.assertLess(i_home, i_verify)
        self.assertLess(i_verify, i_clear)
        self.assertLess(i_clear, i_stop)
        self.assertLess(i_stop, i_disable)
        # A raised arm is never released without the operator's word.
        self.assertIn("not releasing", src)
        self.assertIn("'drop'", src)

    def test_homing_is_verified_not_assumed(self) -> None:
        src = inspect.getsource(pl._run)
        self.assertIn("async def home_test_joint() -> bool:", src)
        self.assertIn("read_position(test)", src)


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
