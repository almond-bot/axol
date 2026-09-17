"""A tuner's teardown may never cut torque on a raised arm.

Bench, right shoulder_2 gravity sweep: one 0x92 read timed out (0.1 s) at the
top of the sweep; the abort's teardown tried to home, that read timed out too,
the failure was swallowed by ``except Exception: pass``, and the code went on
to reset (2 s of zero torque) and disable every motor with the arm raised. The
arm slammed down.
"""

from __future__ import annotations

import asyncio
import inspect
import unittest

from almond_axol.cli.tune import friction, gravity
from almond_axol.constants import Joint
from almond_axol.tuning import holders as H


class _Flaky:
    def __init__(self, fail: int, value: float = 0.3) -> None:
        self.fail, self.value, self.calls = fail, value, 0

    async def get_position(self) -> float:
        self.calls += 1
        if self.calls <= self.fail:
            raise TimeoutError("late reply")
        return self.value


class ReadPositionRetryTest(unittest.TestCase):
    def test_a_transient_timeout_is_retried_not_fatal(self) -> None:
        m = _Flaky(fail=2)
        self.assertEqual(asyncio.run(H.read_position(m, delay=0)), 0.3)
        self.assertEqual(m.calls, 3)

    def test_a_dead_motor_still_raises_after_the_retries(self) -> None:
        m = _Flaky(fail=99)
        with self.assertRaises(TimeoutError):
            asyncio.run(H.read_position(m, retries=4, delay=0))
        self.assertEqual(m.calls, 4)


class HoldersAdoptAndRestTest(unittest.TestCase):
    def _holders(self, positions: dict[Joint, float]) -> H.ImpedanceHolders:
        class _M:
            def __init__(self, p: float) -> None:
                self.p = p

            async def get_position(self) -> float:
                return self.p

        hs = H.ImpedanceHolders.__new__(H.ImpedanceHolders)
        hs._motors = {j: _M(p) for j, p in positions.items()}
        hs._hold, hs.peak_wobble, hs._drift_sum, hs._drift_n = {}, {}, {}, {}
        return hs

    def test_adopt_takes_hold_where_the_joint_is(self) -> None:
        hs = self._holders({Joint.ELBOW: 0.7})
        asyncio.run(hs.adopt(Joint.ELBOW))
        self.assertAlmostEqual(hs._hold[Joint.ELBOW], 0.7)
        asyncio.run(hs.adopt(Joint.ELBOW))  # idempotent
        self.assertEqual(len(hs._hold), 1)

    def test_at_rest_reads_the_motors_not_the_targets(self) -> None:
        hs = self._holders({Joint.ELBOW: 0.7, Joint.SHOULDER_1: 0.01})
        hs._hold = {Joint.ELBOW: 0.0, Joint.SHOULDER_1: 0.0}  # told to go home...
        off = asyncio.run(hs.at_rest())
        self.assertEqual([j for j, _ in off], [Joint.ELBOW])  # ...but did not.


class TeardownOrderTest(unittest.TestCase):
    def test_teardown_adopts_homes_verifies_then_releases_in_that_order(self) -> None:
        src = inspect.getsource(friction.safe_return_to_rest)
        i_adopt = src.index("holders.adopt(swept)")
        i_home = src.index("holders.ramp_to(j, 0.0, _RAMP_SPEED)")
        i_check = src.index("holders.at_rest()")
        i_stop = src.index("holders.stop()")
        i_reset = src.index("set_control_mode(ControlMode.IMPEDANCE)")
        i_disable = src.index("m.disable()")
        self.assertLess(i_adopt, i_home)
        self.assertLess(i_home, i_check)
        self.assertLess(i_check, i_stop)
        self.assertLess(i_stop, i_reset)
        self.assertLess(i_reset, i_disable)
        # It waits for the operator rather than releasing an off-rest arm,
        # and only an explicit word releases anyway.
        self.assertIn("NOT releasing", src)
        self.assertIn("'drop'", src)
        # No blanket swallow between the check and the release.
        tail = src[i_check:i_stop]
        self.assertNotIn("except Exception:\n                pass", tail)

    def test_both_sweep_tuners_use_it(self) -> None:
        for mod in (gravity, friction):
            with self.subTest(tuner=mod.__name__):
                src = inspect.getsource(mod._run)
                self.assertIn(
                    "await safe_return_to_rest(motors, holders, joint, kp, kd)", src
                )
                # ...and nothing in _run itself resets or disables any more.
                tail = src[src.index("Returning to rest") :]
                self.assertNotIn("set_control_mode", tail)
                self.assertNotIn(".disable()", tail)


if __name__ == "__main__":
    unittest.main()
