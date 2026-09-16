"""The parcel gripper's 180° stroke: opened fully in box mode's parallel grasp only.

The hinged blade folds ``gripper_stroke_deg`` (180°) back at its open
stop, where the calibration puts grip 1.0. Plain teleop and the flush
grasp open it to ``box_tool_open_deg`` (140°) instead — a released
trigger commands that fold — so ``VRTeleopCore.grip_command`` scales the
trigger by 140/180 everywhere but the parallel grasp, and the flush yaw
is the 40° that fold leaves.
"""

from __future__ import annotations

import logging
import math
import unittest

from almond_axol.teleop.box import parcel_tool
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.dagger import DaggerTeleopCore
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

_OPEN = 140.0 / 180.0


def _frame(l_lock: bool, r_lock: bool, l_grip: float = 1.0, r_grip: float = 1.0):
    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=VRPose(position=zero, quaternion=identity),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=l_lock,
        r_lock=r_lock,
        l_grip=l_grip,
        r_grip=r_grip,
    )


def _core(**cfg) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**cfg),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class DefaultsTest(unittest.TestCase):
    def test_stroke_and_opening(self) -> None:
        cfg = VRTeleopConfig()
        self.assertEqual(cfg.gripper_stroke_deg, 180.0)
        self.assertEqual(cfg.box_tool_open_deg, 140.0)
        self.assertEqual(cfg.box_tool, "parcel")

    def test_flush_yaw_is_forty_degrees(self) -> None:
        tool = parcel_tool(VRTeleopConfig().box_tool_open_deg)
        self.assertAlmostEqual(math.degrees(tool.flush_tilt), 40.0, places=9)


class GripCommandTest(unittest.TestCase):
    def test_plain_teleop_opens_to_140_of_180(self) -> None:
        core = _core()
        self.assertAlmostEqual(core.grip_command(1.0), _OPEN)
        self.assertAlmostEqual(core.grip_command(0.5), 0.5 * _OPEN)
        self.assertEqual(core.grip_command(0.0), 0.0)

    def test_parallel_grasp_opens_the_full_stroke(self) -> None:
        core = _core(box_mode=True, box_grasp="straight")
        self.assertEqual(core.grip_command(1.0), 1.0)
        self.assertEqual(core.grip_command(0.25), 0.25)

    def test_flush_grasp_opens_to_140(self) -> None:
        core = _core(box_mode=True, box_grasp="flush")
        self.assertAlmostEqual(core.grip_command(1.0), _OPEN)

    def test_box_mode_off_is_plain_teleop_whatever_the_grasp(self) -> None:
        core = _core(box_mode=False, box_grasp="straight")
        self.assertAlmostEqual(core.grip_command(1.0), _OPEN)

    def test_stock_gripper_is_not_scaled(self) -> None:
        core = _core(box_tool="urdf")
        self.assertEqual(core.grip_command(1.0), 1.0)

    def test_opening_at_or_past_the_stroke_is_not_scaled(self) -> None:
        self.assertEqual(_core(box_tool_open_deg=180.0).grip_command(1.0), 1.0)
        self.assertEqual(_core(box_tool_open_deg=200.0).grip_command(1.0), 1.0)
        self.assertEqual(_core(gripper_stroke_deg=0.0).grip_command(1.0), 1.0)

    def test_reads_the_config_live(self) -> None:
        # Both angles are live settings (set_live lands them in the config
        # on the IK thread); the scale follows whatever is there now.
        core = _core()
        core.set_live("box_tool_open_deg", 90.0)  # accepted as a live key
        core.set_live("gripper_stroke_deg", 180.0)
        core.config.box_tool_open_deg = 90.0
        self.assertAlmostEqual(core.grip_command(1.0), 0.5)
        core.config.box_mode = True
        core.config.box_grasp = "straight"
        self.assertEqual(core.grip_command(1.0), 1.0)


class EngagePathTest(unittest.TestCase):
    """The scaled opening is what the engage paths store as the grip."""

    def test_per_arm_engage(self) -> None:
        core = _core()
        core.update_engage(_frame(l_lock=False, r_lock=False))
        core.update_engage(_frame(l_lock=True, r_lock=True))  # both engage
        core.update_engage(_frame(l_lock=True, r_lock=True, l_grip=1.0, r_grip=0.5))
        self.assertTrue(core.left_enabled and core.right_enabled)
        self.assertAlmostEqual(core.l_grip, _OPEN)
        self.assertAlmostEqual(core.r_grip, 0.5 * _OPEN)

    def test_box_pair_follows_the_leader_at_the_grasp_opening(self) -> None:
        core = _core(box_mode=True, box_grasp="flush")
        core.update_engage(_frame(l_lock=False, r_lock=False))
        core.update_engage(_frame(l_lock=False, r_lock=True))  # right leads
        self.assertTrue(core.teleop_enabled)
        core.update_engage(_frame(l_lock=False, r_lock=True, l_grip=0.0, r_grip=1.0))
        self.assertAlmostEqual(core.l_grip, _OPEN)
        self.assertAlmostEqual(core.r_grip, _OPEN)
        # The stick-click grasp toggle is mirrored back into the config;
        # the parallel grasp then opens the gripper the whole way.
        core.config.box_grasp = "straight"
        core.update_engage(_frame(l_lock=False, r_lock=True, r_grip=1.0))
        self.assertEqual(core.l_grip, 1.0)
        self.assertEqual(core.r_grip, 1.0)

    def test_dagger_takeover(self) -> None:
        core = DaggerTeleopCore(
            VRTeleopConfig(),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )
        core.intervention_allowed.set()
        core._sync_to_robot = lambda: None  # no worker in this test
        core.update_engage(_frame(l_lock=False, r_lock=False))
        core.update_engage(_frame(l_lock=True, r_lock=True))
        self.assertTrue(core.teleop_enabled)
        core.update_engage(_frame(l_lock=True, r_lock=True, l_grip=1.0, r_grip=1.0))
        self.assertAlmostEqual(core.l_grip, _OPEN)
        self.assertAlmostEqual(core.r_grip, _OPEN)


if __name__ == "__main__":
    unittest.main()
