"""Box mode exists only with the parcel gripper; the stock one has rigid mode.

``AxolConfig.gripper`` (adopted into ``VRTeleopConfig.gripper`` by the CLIs
and ``VRTeleop``) selects the fitted gripper. With the stock parallel gripper
box mode is absent — the core refuses it, the live settings drop every
box-only ``box_*`` key, and the control panel hides its box settings — and
rigid mode takes its place, sharing the lead hand and elbows-out knobs.
"""

from __future__ import annotations

import logging
import unittest

from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.robot.config import GRIPPER_TYPES
from almond_axol.serve import settings as serve_settings
from almond_axol.teleop.config import VRTeleopConfig, adopt_robot_gripper
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LIVE_SETTINGS, LiveSettings
from almond_axol.vr.config import VRServerConfig

_BOX_KEYS = {d.key for d in LIVE_SETTINGS if d.key.startswith("box_")}
# The box_* knobs rigid mode uses too.
_PAIR_KEYS = {"box_lead_hand", "box_elbow_out"}


def _core(**overrides) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**overrides),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class CoreGateTest(unittest.TestCase):
    def test_parallel_is_the_default_and_has_no_box_mode(self) -> None:
        # None follows the robot; with no robot to follow it is parallel.
        self.assertIsNone(VRTeleopConfig().gripper)
        self.assertFalse(_core().box_available)
        self.assertFalse(_core(gripper="parallel").box_available)
        self.assertTrue(_core(gripper="parcel").box_available)

    def test_box_mode_at_startup_is_forced_off(self) -> None:
        with self.assertLogs("test", "WARNING") as logs:
            core = _core(box_mode=True)
        self.assertFalse(core.box_mode)
        self.assertIn("parcel gripper", logs.output[0])
        self.assertTrue(_core(box_mode=True, gripper="parcel").box_mode)

    def test_switching_box_mode_on_is_refused(self) -> None:
        core = _core()
        with self.assertRaisesRegex(ValueError, "parcel gripper"):
            core.set_box_mode(True)
        core.set_box_mode(False)
        _core(gripper="parcel").set_box_mode(True)

    def test_rigid_mode_is_the_stock_grippers_pair_mode(self) -> None:
        core = _core(gripper="parallel")
        self.assertTrue(core.rigid_available)
        self.assertEqual(core.pair_name, "rigid")
        core.set_rigid_mode(True)  # queued, applied at the next frame
        self.assertTrue(_core().rigid_available)  # None = parallel
        parcel = _core(gripper="parcel")
        self.assertFalse(parcel.rigid_available)
        self.assertEqual(parcel.pair_name, "box")
        with self.assertRaisesRegex(ValueError, "box mode"):
            parcel.set_rigid_mode(True)


class AdoptRobotGripperTest(unittest.TestCase):
    def test_follows_the_robot(self) -> None:
        config = VRTeleopConfig()
        adopt_robot_gripper(config, "parcel")
        self.assertEqual(config.gripper, "parcel")

    def test_a_disagreeing_teleop_value_is_overridden_with_a_warning(self) -> None:
        config = VRTeleopConfig(gripper="parcel")
        with self.assertLogs("almond_axol.teleop.config", "WARNING") as logs:
            adopt_robot_gripper(config, "parallel")
        self.assertEqual(config.gripper, "parallel")
        self.assertIn("ignored", logs.output[0])

    def test_vrteleop_adopts_an_axol_robots_gripper(self) -> None:
        from almond_axol.teleop.teleop import VRTeleop

        class _Arm:
            gripper_type = "parcel"

        class _Robot:
            left = _Arm()
            right = None

        config = VRTeleopConfig()
        teleop = VRTeleop(
            _Robot(),
            config=config,
            kinematics_config=KinematicsConfig(robot_model="mobile"),
            vr_server_config=VRServerConfig(),
        )
        self.assertTrue(teleop._core.box_available)
        # ...and the IK solver and headset overlay use the parcel gripper's
        # model of Axol on Jelly.
        self.assertEqual(teleop._kinematics_config.gripper, "parcel")
        self.assertEqual(teleop._core._urdf_file, "axol_jelly_parcel.urdf")


class LiveGateTest(unittest.TestCase):
    def _live(self, gripper: str) -> LiveSettings:
        return LiveSettings(_core(gripper=gripper), object(), lambda _s: None)

    def test_parallel_publishes_rigid_mode_and_no_box_only_settings(self) -> None:
        live = self._live("parallel")
        keys = {d["key"] for d in live.schema()}
        self.assertEqual(keys & _BOX_KEYS, _PAIR_KEYS)
        self.assertEqual(set(live.values()) & _BOX_KEYS, _PAIR_KEYS)
        self.assertIn("rigid_mode", keys)
        self.assertIs(live.values()["rigid_mode"], False)
        self.assertIn("reengage", keys)
        with self.assertRaises(ValueError):
            live.apply("box_mode", "toggle")
        live.apply("rigid_mode", "toggle")  # queued on the core

    def test_parcel_publishes_box_mode(self) -> None:
        live = self._live("parcel")
        keys = {d["key"] for d in live.schema()}
        self.assertTrue({"box_mode", "box_grasp", "box_flush_deg"} <= keys)
        self.assertNotIn("rigid_mode", keys)
        with self.assertRaises(ValueError):
            live.apply("rigid_mode", "toggle")

    def test_box_tool_is_gone(self) -> None:
        self.assertNotIn("box_tool", {d.key for d in LIVE_SETTINGS})
        self.assertNotIn("box_tool", VRTeleopConfig.__dataclass_fields__)


class PanelGateTest(unittest.TestCase):
    def test_gripper_select_on_the_robot_tab(self) -> None:
        robot = next(c for c in serve_settings.SETTINGS if c.key == "robot")
        gripper = next(s for s in robot.settings if s.key == "axol.gripper")
        self.assertEqual(gripper.options, GRIPPER_TYPES)

    def test_every_box_setting_needs_the_parcel_gripper(self) -> None:
        box = [
            s
            for c in serve_settings.SETTINGS
            for s in c.settings
            if s.key.startswith("teleop.box_")
        ]
        self.assertTrue(box)
        for s in box:
            if s.key in serve_settings._PAIR_SETTINGS:
                # Shared with rigid mode: shown whichever gripper is fitted.
                self.assertNotIn("showWhen", s.ui, s.key)
                continue
            self.assertEqual(
                s.ui.get("showWhen"),
                {"key": "axol.gripper", "equals": "parcel"},
                s.key,
            )

    def test_the_old_box_tool_setting_is_dropped(self) -> None:
        self.assertEqual(serve_settings.canonical_keys("teleop.box_tool"), ())
        keys = {s.key for c in serve_settings.SETTINGS for s in c.settings}
        self.assertNotIn("teleop.box_tool", keys)


if __name__ == "__main__":
    unittest.main()
