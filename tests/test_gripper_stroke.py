"""The parcel gripper's working opening: 140° from closed, the stop in the angled grasp.

Closed is 0°. ``PositionForceConfig.open_limit_deg`` (140) is how far a
fully open command takes the blade in every mode but box mode's angled
(flush) grasp, which folds it to its open stop — wherever the
calibration found that; no stop angle is assumed anywhere. The teleop
core says when the stop is wanted (``gripper_full_stroke``), the
adapters hand that to the robot, and ``AxolArm`` maps the ``[0, 1]``
command over the right span.
"""

from __future__ import annotations

import logging
import math
import unittest

import numpy as np

from almond_axol.robot.axol import AxolArm, AxolHardware
from almond_axol.robot.config import AxolConfig, PositionForceConfig
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.teleop import _grip_seed


def _robot(close_direction: int = -1, limit: float = 140.0) -> AxolHardware:
    """An offline robot (buses and motors constructed, nothing opened)."""
    cfg = AxolConfig()
    for side in (cfg.left, cfg.right):
        side.gripper = PositionForceConfig(
            torque_limit=0.5,
            max_speed=10.0,
            close_direction=close_direction,
            open_limit_deg=limit,
        )
    return AxolHardware(cfg)


def _arm(stroke_deg: float, close_direction: int = -1, limit: float = 140.0) -> AxolArm:
    """The left arm with a calibrated gripper of the given stroke."""
    arm = _robot(close_direction, limit).left
    # As the calibration sweep would: the jaw closes in close_direction,
    # so the open stop is the other way, ``stroke`` away.
    arm._set_gripper_range(
        open_pos=-close_direction * math.radians(stroke_deg), close_pos=0.0
    )
    return arm


class SpanTest(unittest.TestCase):
    def test_full_open_command_stops_at_the_limit(self) -> None:
        arm = _arm(stroke_deg=180.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(140.0), places=9)
        self.assertAlmostEqual(arm._gripper_to_raw(0.5), math.radians(70.0), places=9)
        self.assertEqual(arm._gripper_to_raw(0.0), 0.0)

    def test_full_stroke_goes_to_the_stop_whatever_it_is(self) -> None:
        for stroke in (150.0, 180.0, 200.0):
            arm = _arm(stroke_deg=stroke)
            arm.set_gripper_full_stroke(True)
            self.assertTrue(arm.gripper_full_stroke)
            self.assertAlmostEqual(
                arm._gripper_to_raw(1.0), math.radians(stroke), places=9, msg=stroke
            )
            arm.set_gripper_full_stroke(False)
            self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(140.0))

    def test_a_short_stroke_is_not_limited(self) -> None:
        # The stop before the limit: the stop is the open position.
        arm = _arm(stroke_deg=120.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(120.0))

    def test_zero_limit_means_the_stop(self) -> None:
        arm = _arm(stroke_deg=180.0, limit=0.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(180.0))

    def test_mirrored_gripper_closes_the_other_way(self) -> None:
        arm = _arm(stroke_deg=180.0, close_direction=1)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), -math.radians(140.0), places=9)
        arm.set_gripper_full_stroke(True)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), -math.radians(180.0), places=9)

    def test_readings_are_over_the_whole_stroke(self) -> None:
        # A reading means the same thing in every mode: the blade at its
        # 140° limit on a 180° stroke reads 140/180 whether or not the full
        # stroke is on.
        arm = _arm(stroke_deg=180.0)
        raw = arm._gripper_to_raw(1.0)
        self.assertAlmostEqual(arm._gripper_from_raw(raw), 140.0 / 180.0, places=9)
        arm.set_gripper_full_stroke(True)
        self.assertAlmostEqual(arm._gripper_from_raw(raw), 140.0 / 180.0, places=9)
        self.assertAlmostEqual(arm._gripper_from_raw(arm._gripper_to_raw(1.0)), 1.0)

    def test_gripper_command_holds_a_reading(self) -> None:
        arm = _arm(stroke_deg=180.0)
        # At the stop after calibration: commanded 1.0, worked to 140°.
        self.assertEqual(arm.gripper_command(1.0), 1.0)
        # Half open (90°) reads 0.5 and takes 90/140 to hold.
        self.assertAlmostEqual(arm.gripper_command(0.5), 90.0 / 140.0, places=9)
        arm.set_gripper_full_stroke(True)
        self.assertAlmostEqual(arm.gripper_command(0.5), 0.5)

    def test_both_arms_together(self) -> None:
        robot = _robot()
        robot.set_gripper_full_stroke(True)
        self.assertTrue(robot.left.gripper_full_stroke)
        self.assertTrue(robot.right.gripper_full_stroke)
        robot.set_gripper_full_stroke(False)
        self.assertFalse(robot.left.gripper_full_stroke)

    def test_default_limit(self) -> None:
        self.assertEqual(AxolConfig().left.gripper.open_limit_deg, 140.0)
        self.assertEqual(AxolConfig().right.gripper.open_limit_deg, 140.0)
        self.assertNotIn("gripper_stroke_deg", VRTeleopConfig.__dataclass_fields__)


def _core(**cfg) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**cfg),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class CoreTest(unittest.TestCase):
    def test_only_the_angled_box_grasp_wants_the_stop(self) -> None:
        self.assertFalse(_core().gripper_full_stroke())
        self.assertFalse(
            _core(box_mode=True, box_grasp="straight").gripper_full_stroke()
        )
        self.assertTrue(_core(box_mode=True, box_grasp="flush").gripper_full_stroke())
        # Flush grasp configured but box mode off: plain teleop, 140°.
        self.assertFalse(_core(box_mode=False, box_grasp="flush").gripper_full_stroke())

    def test_stock_gripper_is_never_asked(self) -> None:
        core = _core(box_mode=True, box_grasp="flush", box_tool="urdf")
        self.assertFalse(core.gripper_full_stroke())

    def test_follows_the_stick_toggled_grasp(self) -> None:
        core = _core(box_mode=True, box_grasp="straight")
        self.assertFalse(core.gripper_full_stroke())
        core.config.box_grasp = "flush"  # what the worker's status mirrors back
        self.assertTrue(core.gripper_full_stroke())

    def test_triggers_pass_through_unscaled(self) -> None:
        core = _core()
        self.assertFalse(hasattr(core, "grip_command"))


class SeedTest(unittest.TestCase):
    def test_seed_converts_a_reading_through_the_arm(self) -> None:
        arm = _arm(stroke_deg=180.0)
        pos = np.zeros(8)
        pos[7] = 1.0  # resting at the stop after calibration
        self.assertEqual(_grip_seed(arm, pos), 1.0)
        pos[7] = 0.5
        self.assertAlmostEqual(_grip_seed(arm, pos), 90.0 / 140.0, places=9)

    def test_seed_without_an_arm_is_the_reading(self) -> None:
        pos = np.zeros(8)
        pos[7] = 0.3
        self.assertAlmostEqual(_grip_seed(None, pos), 0.3)
        self.assertIsNone(_grip_seed(None, None))
        self.assertIsNone(_grip_seed(None, np.zeros(7)))


class SyncTest(unittest.TestCase):
    """The native adapter tells the robot on change only, and lets go at the end."""

    def test_flag_reaches_the_robot_on_change(self) -> None:
        # Mirror of the adapter's _sync_squeeze: exercised through the core
        # and a recording robot rather than the whole control loop.
        core = _core(box_mode=True, box_grasp="straight")
        calls: list[bool] = []
        applied: bool | None = None

        def sync() -> None:
            nonlocal applied
            full = core.gripper_full_stroke()
            if full != applied:
                calls.append(full)
                applied = full

        sync()
        sync()
        core.config.box_grasp = "flush"
        sync()
        sync()
        core.config.box_grasp = "straight"
        sync()
        self.assertEqual(calls, [False, True, False])

    def test_gravity_hold_uses_the_reading_not_the_working_span(self) -> None:
        # Holding the gripper where it is must not map the reading through
        # the 140° span (it would pull a blade at the stop in to 109°).
        arm = _arm(stroke_deg=180.0)
        raw_at_stop = arm._gripper_open
        reading = arm._gripper_from_raw(raw_at_stop)
        held = arm._gripper_close + reading * (arm._gripper_open - arm._gripper_close)
        self.assertAlmostEqual(held, raw_at_stop, places=9)
        self.assertNotAlmostEqual(arm._gripper_to_raw(reading), raw_at_stop, places=3)


if __name__ == "__main__":
    unittest.main()
