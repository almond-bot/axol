"""The parcel gripper opens to its stop, except in box mode's angled grasp.

Closed is 0°. A fully open command takes the hinged blade to the
calibrated open stop — wherever the sweep found it; no stop angle is
assumed anywhere — in every mode but box mode's angled (flush) grasp,
which holds it at ``box_tool_open_deg`` (140°), the angle the grasp's
yaw lays the flat face along the box at. The teleop core says when the
limit is wanted (``gripper_open_limit``), the adapters hand it to the
robot, and ``AxolArm`` maps the ``[0, 1]`` command over the right span.
"""

from __future__ import annotations

import logging
import math
import unittest

import numpy as np

from almond_axol.motor import Joint
from almond_axol.robot.axol import AxolArm, AxolHardware
from almond_axol.robot.config import AxolConfig, PositionForceConfig
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.teleop import _grip_seed


def _robot(close_direction: int = -1) -> AxolHardware:
    """An offline robot (buses and motors constructed, nothing opened)."""
    cfg = AxolConfig()
    for side in (cfg.left, cfg.right):
        side.gripper = PositionForceConfig(
            torque_limit=0.5, max_speed=10.0, close_direction=close_direction
        )
    return AxolHardware(cfg)


def _arm(stroke_deg: float, close_direction: int = -1) -> AxolArm:
    """The left arm with a calibrated gripper of the given stroke."""
    arm = _robot(close_direction).left
    # As the calibration sweep would: the jaw closes in close_direction,
    # so the open stop is the other way, ``stroke`` away.
    arm._set_gripper_range(
        open_pos=-close_direction * math.radians(stroke_deg), close_pos=0.0
    )
    return arm


class SpanTest(unittest.TestCase):
    def test_full_open_goes_to_the_stop_whatever_it_is(self) -> None:
        for stroke in (150.0, 180.0, 200.0):
            arm = _arm(stroke_deg=stroke)
            self.assertIsNone(arm.gripper_open_limit)
            self.assertAlmostEqual(
                arm._gripper_to_raw(1.0), math.radians(stroke), places=9, msg=stroke
            )
            self.assertAlmostEqual(arm._gripper_to_raw(0.5), math.radians(stroke / 2))
            self.assertEqual(arm._gripper_to_raw(0.0), 0.0)

    def test_a_limit_stops_short_of_the_stop(self) -> None:
        arm = _arm(stroke_deg=180.0)
        arm.set_gripper_open_limit(140.0)
        self.assertAlmostEqual(arm.gripper_open_limit, 140.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(140.0), places=9)
        self.assertAlmostEqual(arm._gripper_to_raw(0.5), math.radians(70.0), places=9)
        arm.set_gripper_open_limit(None)
        self.assertIsNone(arm.gripper_open_limit)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(180.0), places=9)

    def test_a_limit_past_the_stop_is_the_stop(self) -> None:
        arm = _arm(stroke_deg=120.0)
        arm.set_gripper_open_limit(140.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), math.radians(120.0))

    def test_non_positive_limit_means_the_stop(self) -> None:
        arm = _arm(stroke_deg=180.0)
        arm.set_gripper_open_limit(0.0)
        self.assertIsNone(arm.gripper_open_limit)
        arm.set_gripper_open_limit(-5.0)
        self.assertIsNone(arm.gripper_open_limit)

    def test_mirrored_gripper_closes_the_other_way(self) -> None:
        arm = _arm(stroke_deg=180.0, close_direction=1)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), -math.radians(180.0), places=9)
        arm.set_gripper_open_limit(140.0)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), -math.radians(140.0), places=9)

    def test_readings_are_over_the_whole_stroke(self) -> None:
        # A reading means the same thing in every mode: a blade held at
        # 140° on a 180° stroke reads 140/180 with or without the limit.
        arm = _arm(stroke_deg=180.0)
        arm.set_gripper_open_limit(140.0)
        raw = arm._gripper_to_raw(1.0)
        self.assertAlmostEqual(arm._gripper_from_raw(raw), 140.0 / 180.0, places=9)
        arm.set_gripper_open_limit(None)
        self.assertAlmostEqual(arm._gripper_from_raw(raw), 140.0 / 180.0, places=9)
        self.assertAlmostEqual(arm._gripper_from_raw(arm._gripper_to_raw(1.0)), 1.0)

    def test_gripper_command_holds_a_reading(self) -> None:
        arm = _arm(stroke_deg=180.0)
        # No limit: readings and commands agree.
        self.assertEqual(arm.gripper_command(1.0), 1.0)
        self.assertAlmostEqual(arm.gripper_command(0.5), 0.5)
        # Under the limit a reading past it is capped; half open (90°)
        # takes 90/140 to hold.
        arm.set_gripper_open_limit(140.0)
        self.assertEqual(arm.gripper_command(1.0), 1.0)
        self.assertAlmostEqual(arm.gripper_command(0.5), 90.0 / 140.0, places=9)

    def test_both_arms_together(self) -> None:
        robot = _robot()
        robot.set_gripper_open_limit(140.0)
        self.assertAlmostEqual(robot.left.gripper_open_limit, 140.0)
        self.assertAlmostEqual(robot.right.gripper_open_limit, 140.0)
        robot.set_gripper_open_limit(None)
        self.assertIsNone(robot.left.gripper_open_limit)

    def test_no_stroke_angle_in_the_configs(self) -> None:
        self.assertNotIn("gripper_stroke_deg", VRTeleopConfig.__dataclass_fields__)
        self.assertNotIn("open_limit_deg", PositionForceConfig.__dataclass_fields__)
        self.assertEqual(VRTeleopConfig().box_tool_open_deg, 140.0)


class BladeHoldTest(unittest.TestCase):
    """The blade hold keeps a blade *at* its opening limit under a clamp."""

    def _arm(self, hold_deg: float = 10.0) -> AxolArm:
        arm = _arm(stroke_deg=180.0)
        arm._arm_config.gripper.hold_trim_deg = hold_deg
        arm.set_gripper_open_limit(140.0)
        return arm

    def _measure(self, arm: AxolArm, blade_deg: float) -> None:
        arm.motors[Joint.GRIPPER]._position = math.radians(blade_deg)

    def _run(self, arm: AxolArm, opening: float, seconds: float, t0: float) -> float:
        n = int(seconds * 120)
        hold = 0.0
        for i in range(n + 1):
            hold = arm._blade_hold(opening, now=t0 + i / 120.0)
        return hold

    def test_a_blade_falling_short_is_pushed_on(self) -> None:
        arm = self._arm()
        # Commanded to the 140° limit, the clamp folds the blade to 137°.
        self._measure(arm, 137.0)
        hold = self._run(arm, 1.0, 1.0, t0=10.0)
        # 3° short at 1°/s per degree: ~3° after a second, the right way.
        self.assertGreater(math.degrees(hold), 2.5)
        self.assertLess(math.degrees(hold), 3.5)
        self.assertAlmostEqual(arm.blade_hold_deg, math.degrees(hold))
        # The other way too (sign-agnostic).
        arm = self._arm()
        self._measure(arm, 143.0)
        hold = self._run(arm, 1.0, 1.0, t0=10.0)
        self.assertLess(math.degrees(hold), -2.5)

    def test_bounded_by_the_config(self) -> None:
        arm = self._arm(hold_deg=4.0)
        self._measure(arm, 134.0)
        hold = self._run(arm, 1.0, 10.0, t0=10.0)
        self.assertAlmostEqual(math.degrees(hold), 4.0, places=6)

    def test_a_travelling_blade_does_not_wind_up(self) -> None:
        # The limit just switched: the blade is still 30° away, moving.
        arm = self._arm()
        self._measure(arm, 110.0)
        hold = self._run(arm, 1.0, 2.0, t0=10.0)
        self.assertEqual(hold, 0.0)

    def test_off_the_limit_it_bleeds_away(self) -> None:
        arm = self._arm()
        self._measure(arm, 137.0)
        self._run(arm, 1.0, 2.0, t0=10.0)
        self.assertGreater(arm.blade_hold_deg, 1.5)
        # Trigger squeezed: the command is no longer the full open.
        hold = self._run(arm, 0.5, 5.0, t0=13.0)
        self.assertLess(abs(math.degrees(hold)), 0.2)
        # Or the limit lifted (the blade goes to the stop).
        arm = self._arm()
        self._measure(arm, 137.0)
        self._run(arm, 1.0, 2.0, t0=10.0)
        arm.set_gripper_open_limit(None)
        hold = self._run(arm, 1.0, 5.0, t0=13.0)
        self.assertLess(abs(math.degrees(hold)), 0.2)

    def test_off_at_zero_and_without_a_reading(self) -> None:
        arm = self._arm(hold_deg=0.0)
        self._measure(arm, 137.0)
        self.assertEqual(self._run(arm, 1.0, 1.0, t0=10.0), 0.0)
        arm = self._arm()
        arm.motors[Joint.GRIPPER]._position = None
        self.assertEqual(self._run(arm, 1.0, 1.0, t0=10.0), 0.0)

    def test_default_bound(self) -> None:
        self.assertEqual(
            PositionForceConfig(torque_limit=0.5, max_speed=10.0).hold_trim_deg, 10.0
        )


def _core(**cfg) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**cfg),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class CoreTest(unittest.TestCase):
    def test_only_the_angled_box_grasp_limits_the_opening(self) -> None:
        self.assertIsNone(_core().gripper_open_limit())
        self.assertIsNone(
            _core(box_mode=True, box_grasp="straight").gripper_open_limit()
        )
        self.assertEqual(
            _core(box_mode=True, box_grasp="flush").gripper_open_limit(), 140.0
        )
        # Flush grasp configured but box mode off: plain teleop, the stop.
        self.assertIsNone(_core(box_mode=False, box_grasp="flush").gripper_open_limit())

    def test_the_limit_is_the_blade_angle_setting(self) -> None:
        core = _core(box_mode=True, box_grasp="flush", box_tool_open_deg=146.0)
        self.assertEqual(core.gripper_open_limit(), 146.0)

    def test_stock_gripper_is_never_limited(self) -> None:
        core = _core(box_mode=True, box_grasp="flush", box_tool="urdf")
        self.assertIsNone(core.gripper_open_limit())

    def test_follows_the_stick_toggled_grasp(self) -> None:
        core = _core(box_mode=True, box_grasp="straight")
        self.assertIsNone(core.gripper_open_limit())
        core.config.box_grasp = "flush"  # what the worker's status mirrors back
        self.assertEqual(core.gripper_open_limit(), 140.0)

    def test_triggers_pass_through_unscaled(self) -> None:
        self.assertFalse(hasattr(_core(), "grip_command"))


class SeedTest(unittest.TestCase):
    def test_seed_converts_a_reading_through_the_arm(self) -> None:
        arm = _arm(stroke_deg=180.0)
        pos = np.zeros(8)
        pos[7] = 0.5
        self.assertAlmostEqual(_grip_seed(arm, pos), 0.5)  # no limit: as read
        arm.set_gripper_open_limit(140.0)
        self.assertAlmostEqual(_grip_seed(arm, pos), 90.0 / 140.0, places=9)

    def test_seed_without_an_arm_is_the_reading(self) -> None:
        pos = np.zeros(8)
        pos[7] = 0.3
        self.assertAlmostEqual(_grip_seed(None, pos), 0.3)
        self.assertIsNone(_grip_seed(None, None))
        self.assertIsNone(_grip_seed(None, np.zeros(7)))


class SyncTest(unittest.TestCase):
    """The adapters tell the robot on change only, and let go at the end."""

    def test_limit_reaches_the_robot_on_change(self) -> None:
        # Mirror of the adapter's _sync_squeeze: exercised through the core
        # and a recording setter rather than the whole control loop.
        core = _core(box_mode=True, box_grasp="straight")
        calls: list[float | None] = []
        applied: float | None = None

        def sync() -> None:
            nonlocal applied
            limit = core.gripper_open_limit()
            if limit != applied:
                calls.append(limit)
                applied = limit

        sync()
        sync()
        core.config.box_grasp = "flush"
        sync()
        sync()
        core.config.box_grasp = "straight"
        sync()
        self.assertEqual(calls, [140.0, None])

    def test_gravity_hold_uses_the_reading_not_the_working_span(self) -> None:
        # Holding the gripper where it is must not map the reading through
        # a 140° span (it would pull a blade at the stop in to 109°).
        arm = _arm(stroke_deg=180.0)
        arm.set_gripper_open_limit(140.0)
        raw_at_stop = arm._gripper_open
        reading = arm._gripper_from_raw(raw_at_stop)
        held = arm._gripper_close + reading * (arm._gripper_open - arm._gripper_close)
        self.assertAlmostEqual(held, raw_at_stop, places=9)
        self.assertNotAlmostEqual(arm._gripper_to_raw(reading), raw_at_stop, places=3)


if __name__ == "__main__":
    unittest.main()
