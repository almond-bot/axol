"""The parcel gripper opens to its calibrated stop in every mode.

Closed is 0°. A fully open command takes the hinged blade to the open stop
the enable-time sweep found — no stop angle is assumed anywhere — in plain
teleop and in both box-mode grasps. The angled grasp turns the grippers
inward (``box_flush_deg``) instead of holding the blade short of its stop.
"""

from __future__ import annotations

import logging
import math
import unittest

from almond_axol.robot.axol import AxolArm, AxolHardware
from almond_axol.robot.config import AxolConfig, PositionForceConfig
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore


def _arm(stroke_deg: float, close_direction: int = -1) -> AxolArm:
    """The left arm with a calibrated gripper of the given stroke (offline)."""
    cfg = AxolConfig()
    for side in (cfg.left, cfg.right):
        side.gripper = PositionForceConfig(
            torque_limit=0.5, max_speed=10.0, close_direction=close_direction
        )
    arm = AxolHardware(cfg).left
    # As the calibration sweep would: the jaw closes in close_direction,
    # so the open stop is the other way, ``stroke`` away.
    arm._set_gripper_range(
        open_pos=-close_direction * math.radians(stroke_deg), close_pos=0.0
    )
    return arm


class StrokeTest(unittest.TestCase):
    def test_full_open_goes_to_the_stop_whatever_it_is(self) -> None:
        for stroke in (150.0, 180.0, 200.0):
            arm = _arm(stroke_deg=stroke)
            self.assertAlmostEqual(
                arm._gripper_to_raw(1.0), math.radians(stroke), places=9, msg=stroke
            )
            self.assertAlmostEqual(arm._gripper_to_raw(0.5), math.radians(stroke / 2))
            self.assertEqual(arm._gripper_to_raw(0.0), 0.0)

    def test_mirrored_gripper_closes_the_other_way(self) -> None:
        arm = _arm(stroke_deg=180.0, close_direction=1)
        self.assertAlmostEqual(arm._gripper_to_raw(1.0), -math.radians(180.0), places=9)

    def test_readings_and_commands_share_the_stroke(self) -> None:
        arm = _arm(stroke_deg=180.0)
        for opening in (0.0, 0.3, 0.78, 1.0):
            raw = arm._gripper_to_raw(opening)
            self.assertAlmostEqual(arm._gripper_from_raw(raw), opening, places=9)

    def test_no_opening_limit_anywhere(self) -> None:
        arm = _arm(stroke_deg=180.0)
        for name in ("set_gripper_open_limit", "gripper_open_limit", "gripper_command"):
            self.assertFalse(hasattr(arm, name), name)
        self.assertFalse(hasattr(AxolHardware, "set_gripper_open_limit"))
        self.assertNotIn("hold_trim_deg", PositionForceConfig.__dataclass_fields__)
        core = VRTeleopCore(
            VRTeleopConfig(box_mode=True, box_grasp="flush"),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )
        self.assertFalse(hasattr(core, "gripper_open_limit"))

    def test_the_angled_grasp_is_a_yaw_setting(self) -> None:
        fields = VRTeleopConfig.__dataclass_fields__
        self.assertNotIn("box_tool_open_deg", fields)
        self.assertEqual(VRTeleopConfig().box_flush_deg, 39.0)


if __name__ == "__main__":
    unittest.main()
