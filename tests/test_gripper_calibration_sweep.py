"""The enable-time gripper sweep stops on a hard stop, not a torque spike.

A jaw breaking free from where it rests, or passing a stiff patch of its
travel, pushes the torque past the stop threshold for a step or two while
the shaft keeps following the target. Only a shaft that stops following
with the torque held is a stop.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import AsyncMock, patch

from almond_axol.constants import Joint
from almond_axol.motor import MotorError
from almond_axol.robot import axol as axol_module
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.config import AxolConfig


class _Jaw:
    """A gripper motor under the calibration's impedance hold.

    The shaft follows the target between two hard stops; the reported
    torque is the spring's push into a stop, plus ``spike_nm`` along the
    motion while the shaft is inside ``stiff`` (a sticky patch it still
    moves through).
    """

    def __init__(
        self,
        lo: float,
        hi: float,
        start: float,
        stiff: tuple[float, float] | None = None,
        spike_nm: float = 0.0,
        stuck_nm: float = 0.0,
        spring: float = 0.0,
        inverted: bool = False,
    ) -> None:
        self.lo, self.hi = lo, hi
        self.spring = spring
        self.inverted = inverted
        self.sprung = 0
        self.position = start
        self.target = start
        self.stiff = stiff
        self.spike_nm = spike_nm
        self.stuck_nm = stuck_nm
        self.torque = 0.0

    async def set_impedance(self, p_des, v_des, kp, kd, t_ff) -> None:
        motion = p_des - self.target
        self.target = p_des
        if self.stuck_nm and abs(kp * (p_des - self.position)) < self.stuck_nm:
            # Stuck where it rests until the spring pulls hard enough.
            self.torque = kp * (p_des - self.position)
            return
        self.stuck_nm = 0.0
        on_stop = self.position in (self.lo, self.hi)
        self.position = min(max(p_des, self.lo), self.hi)
        if self.spring and on_stop and self.lo < self.position < self.hi:
            # Leaving a stop it was pressed on: the jaw springs off it,
            # ``spring`` rad ahead of the target, for a few steps.
            self.sprung = 3
        if self.sprung:
            self.sprung -= 1
            self.position = min(
                max(p_des + math.copysign(self.spring, motion), self.lo), self.hi
            )
        self.torque = kp * (p_des - self.position)
        if (
            self.stiff is not None
            and self.stiff[0] <= self.position <= self.stiff[1]
            and self.torque == 0.0
        ):
            self.torque = math.copysign(self.spike_nm, motion)

    async def get_torque(self) -> float:
        return -self.torque if self.inverted else self.torque

    async def get_position(self) -> float:
        return self.position


def _arm_with(jaw: _Jaw):
    arm = AxolHardware(AxolConfig()).left
    arm.motors[Joint.GRIPPER] = jaw
    return arm


class SweepTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module.asyncio, "sleep", AsyncMock()))
        self.enterContext(patch.object(axol_module, "_save_gripper_calibration"))

    async def test_breakaway_spike_off_the_open_stop_is_not_a_stop(self) -> None:
        # Resting on the open stop (-5.38), the first 2.4° toward closed
        # stick hard enough to read past the threshold — the failure seen
        # on the robot ("stops only 2.4° apart").
        jaw = _Jaw(lo=-5.38, hi=-2.20, start=-5.38, stiff=(-5.38, -5.338), spike_nm=0.8)
        arm = _arm_with(jaw)
        await arm._calibrate_gripper()
        self.assertAlmostEqual(arm._gripper_close, -2.20, delta=0.01)
        self.assertAlmostEqual(arm._gripper_open, -5.38, delta=0.01)
        self.assertAlmostEqual(arm.gripper_travel, 3.18, delta=0.02)

    async def test_stiff_patch_mid_travel_is_not_a_stop(self) -> None:
        jaw = _Jaw(lo=-5.38, hi=-2.20, start=-4.0, stiff=(-3.5, -3.3), spike_nm=1.5)
        arm = _arm_with(jaw)
        await arm._calibrate_gripper()
        self.assertAlmostEqual(arm._gripper_close, -2.20, delta=0.01)
        self.assertAlmostEqual(arm._gripper_open, -5.38, delta=0.01)

    async def test_jaw_stuck_on_the_open_stop_is_retried_harder(self) -> None:
        # Fully open and needing 1.4 Nm to break free: the first pair of
        # sweeps both stop where the jaw rests, a harder retry frees it.
        jaw = _Jaw(lo=0.53, hi=3.74, start=3.74, stuck_nm=1.4)
        arm = AxolHardware(AxolConfig()).right
        arm.motors[Joint.GRIPPER] = jaw
        with self.assertLogs(axol_module._logger, "WARNING") as logs:
            await arm._calibrate_gripper()
        self.assertIn("stuck on the stop", logs.output[0])
        self.assertAlmostEqual(arm._gripper_close, 0.53, delta=0.01)
        self.assertAlmostEqual(arm._gripper_open, 3.74, delta=0.01)

    async def test_jaw_resting_on_the_closed_stop_finds_it_at_once(self) -> None:
        jaw = _Jaw(lo=-5.38, hi=-2.20, start=-2.20)
        arm = _arm_with(jaw)
        await arm._calibrate_gripper()
        self.assertAlmostEqual(arm._gripper_close, -2.20, delta=0.001)
        self.assertAlmostEqual(arm._gripper_open, -5.38, delta=0.001)

    async def test_stop_is_held_under_the_abort_torque(self) -> None:
        jaw = _Jaw(lo=-5.38, hi=-2.20, start=-3.0)
        arm = _arm_with(jaw)
        await arm._seek_gripper_stop(1)
        self.assertLess(jaw.torque, axol_module._GRIPPER_CALIB_TORQUE_ABORT)

    async def test_jaw_springing_off_the_closed_stop_is_not_an_abort(self) -> None:
        # The right arm on the robot: leaving the closed stop it was pressed
        # on, the jaw sprang ~2.6° ahead of the target and the hold braked
        # it at -2.27 Nm, which aborted the opening sweep.
        jaw = _Jaw(lo=0.53, hi=3.74, start=2.0, spring=0.0454)
        arm = AxolHardware(AxolConfig()).right
        arm.motors[Joint.GRIPPER] = jaw
        await arm._calibrate_gripper()
        self.assertAlmostEqual(arm._gripper_close, 0.53, delta=0.06)
        self.assertAlmostEqual(arm._gripper_open, 3.74, delta=0.01)

    async def test_inverted_torque_sign_still_aborts(self) -> None:
        jaw = _Jaw(lo=-5.38, hi=-2.20, start=-3.0, inverted=True)
        arm = _arm_with(jaw)
        with self.assertRaisesRegex(MotorError, "against the sweep"):
            await arm._calibrate_gripper()
        # The failed sweep leaves the hold at the shaft, not wound up.
        self.assertEqual(jaw.target, jaw.position)

    async def test_jammed_jaw_still_fails(self) -> None:
        jaw = _Jaw(lo=-3.0, hi=-2.9, start=-2.95)
        arm = _arm_with(jaw)
        with self.assertRaisesRegex(MotorError, "apart"):
            await arm._calibrate_gripper()


if __name__ == "__main__":
    unittest.main()
