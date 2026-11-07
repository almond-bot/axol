"""The enable-time gripper sweep stops on a hard stop, not a torque spike.

A jaw breaking free from where it rests, or passing a stiff patch of its
travel, pushes the torque past the stop threshold for a step or two while
the shaft keeps following the target. Only a shaft that stops following
with the torque held is a stop.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from almond_axol.constants import Joint
from almond_axol.motor import MotorError
from almond_axol.robot import axol as axol_module
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.mantis import MantisGripperArm


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
    arm = AxolHardware(AxolConfig(gripper="parcel")).left
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
        arm = AxolHardware(AxolConfig(gripper="parcel")).right
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
        arm = AxolHardware(AxolConfig(gripper="parcel")).right
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


class GripperTypeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module.asyncio, "sleep", AsyncMock()))
        self.enterContext(patch.object(axol_module, "_save_gripper_calibration"))

    async def test_parallel_jaws_both_close_toward_positive(self) -> None:
        # The stock gripper's ~290° stroke, open at the negative stop on
        # both sides.
        hw = AxolHardware(AxolConfig(gripper="parallel"))
        for arm in (hw.left, hw.right):
            jaw = _Jaw(lo=-6.0, hi=-6.0 + math.radians(290), start=-4.0)
            arm.motors[Joint.GRIPPER] = jaw
            await arm._calibrate_gripper()
            self.assertAlmostEqual(arm._gripper_open, -6.0, delta=0.01)
            self.assertAlmostEqual(arm._gripper_close, jaw.hi, delta=0.01)
            self.assertEqual(jaw.position, jaw.lo)

    async def test_parcel_stroke_under_the_parallel_setting_fails(self) -> None:
        jaw = _Jaw(lo=0.53, hi=3.74, start=2.0)
        arm = AxolHardware(AxolConfig(gripper="parallel")).right
        arm.motors[Joint.GRIPPER] = jaw
        with self.assertRaisesRegex(
            MotorError,
            "matches the parcel gripper.*Empty the gripper.*--axol.gripper parcel",
        ):
            await arm._calibrate_gripper()
        self.assertEqual(jaw.target, jaw.position)
        axol_module._save_gripper_calibration.assert_not_called()

    async def test_parallel_stroke_under_the_parcel_setting_fails(self) -> None:
        jaw = _Jaw(lo=-6.0, hi=-6.0 + math.radians(290), start=-4.0)
        arm = AxolHardware(AxolConfig(gripper="parcel")).left
        arm.motors[Joint.GRIPPER] = jaw
        with self.assertRaisesRegex(MotorError, "matches the parallel gripper"):
            await arm._calibrate_gripper()

    async def test_short_stroke_matching_neither_type_is_accepted(self) -> None:
        # A parallel jaw stopped short by an object (240°, or 120°) matches
        # neither type: it calibrates, with a warning, rather than being
        # taken for the parcel gripper.
        for stroke_deg in (240, 120):
            with self.subTest(stroke_deg=stroke_deg):
                jaw = _Jaw(lo=-6.0, hi=-6.0 + math.radians(stroke_deg), start=-5.0)
                arm = AxolHardware(AxolConfig(gripper="parallel")).right
                arm.motors[Joint.GRIPPER] = jaw
                with self.assertLogs(axol_module._logger, "WARNING") as logs:
                    await arm._calibrate_gripper()
                self.assertIn("off the parallel gripper's", "\n".join(logs.output))
                self.assertAlmostEqual(
                    abs(arm._gripper_close - arm._gripper_open),
                    math.radians(stroke_deg),
                    delta=0.02,
                )

    async def test_restore_rejects_the_other_types_calibration(self) -> None:
        arm = AxolHardware(AxolConfig(gripper="parcel")).left
        arm.motors[Joint.GRIPPER] = _Jaw(lo=-6.0, hi=-1.0, start=-5.0)
        with patch.object(
            axol_module,
            "_load_gripper_calibration",
            return_value=(-6.0, -6.0 + math.radians(290)),
        ):
            with self.assertRaisesRegex(MotorError, "matches the parallel gripper"):
                await arm._restore_gripper_calibration()

    def test_unknown_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gripper must be one of"):
            AxolConfig(gripper="claw")


def _mantis_with(jaw: _Jaw) -> MantisGripperArm:
    bus = Mock()
    bus.channel = "can_mantis_l"
    with patch("almond_axol.robot.mantis.Motor", return_value=jaw):
        return MantisGripperArm(bus, SimpleNamespace(max_speed=10.0, torque_limit=0.5))


class MantisSweepTest(unittest.IsolatedAsyncioTestCase):
    """The Mantis gripper runs the robot gripper's two-stop sweep."""

    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module.asyncio, "sleep", AsyncMock()))

    async def test_stiff_patch_is_not_taken_for_the_open_stop(self) -> None:
        # The open-only sweep took a stiff patch on the way open for the
        # open stop, so the gripper never opened past it.
        lo, hi = -6.0, -6.0 + math.radians(290)
        jaw = _Jaw(lo=lo, hi=hi, start=-1.5, stiff=(-2.6, -2.4), spike_nm=1.5)
        arm = _mantis_with(jaw)
        await arm._calibrate()
        self.assertTrue(arm.is_calibrated)
        self.assertAlmostEqual(arm._open_pos, lo, delta=0.01)
        self.assertAlmostEqual(arm._closed_pos, hi, delta=0.01)
        # Closes first, then finishes open.
        self.assertEqual(jaw.position, lo)

    async def test_measured_stroke_replaces_the_nominal_one(self) -> None:
        jaw = _Jaw(lo=-6.0, hi=-6.0 + math.radians(240), start=-5.0)
        arm = _mantis_with(jaw)
        with self.assertLogs("almond_axol.robot.mantis", "WARNING") as logs:
            await arm._calibrate()
        self.assertIn("off the nominal", "\n".join(logs.output))
        self.assertAlmostEqual(arm.gripper_travel, math.radians(240), delta=0.02)
        self.assertAlmostEqual(arm._normalize(jaw.lo), 1.0, delta=0.01)
        self.assertAlmostEqual(arm._normalize(jaw.hi), 0.0, delta=0.01)

    async def test_jaw_stuck_on_the_open_stop_is_retried_harder(self) -> None:
        lo, hi = -6.0, -6.0 + math.radians(290)
        jaw = _Jaw(lo=lo, hi=hi, start=lo, stuck_nm=1.4)
        arm = _mantis_with(jaw)
        with self.assertLogs(axol_module._logger, "WARNING"):
            await arm._calibrate()
        self.assertAlmostEqual(arm._open_pos, lo, delta=0.01)
        self.assertAlmostEqual(arm._closed_pos, hi, delta=0.01)

    async def test_jammed_jaw_fails_uncalibrated(self) -> None:
        jaw = _Jaw(lo=-3.0, hi=-2.9, start=-2.95)
        arm = _mantis_with(jaw)
        with self.assertRaisesRegex(MotorError, "can_mantis_l gripper.*apart"):
            await arm._calibrate()
        self.assertFalse(arm.is_calibrated)


if __name__ == "__main__":
    unittest.main()
