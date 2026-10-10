"""Box mode exists only with the parcel gripper.

``AxolConfig.gripper`` (adopted into ``VRTeleopConfig.gripper`` by the CLIs
and ``VRTeleop``) selects the fitted gripper. With the stock parallel gripper
the whole of box mode is absent: the core refuses it, the live settings drop every ``box_*``
key, and the control panel hides its box settings.
"""

from __future__ import annotations

import logging
import unittest

from almond_axol.serve import settings as serve_settings
from almond_axol.robot.config import GRIPPER_TYPES
from almond_axol.teleop.config import VRTeleopConfig, adopt_robot_gripper
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LIVE_SETTINGS, LiveSettings
from almond_axol.vr.config import VRServerConfig

_BOX_KEYS = {d.key for d in LIVE_SETTINGS if d.key.startswith("box_")}


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
            kinematics_config=object(),
            vr_server_config=VRServerConfig(),
        )
        self.assertTrue(teleop._core.box_available)


class LiveGateTest(unittest.TestCase):
    def _live(self, gripper: str) -> LiveSettings:
        return LiveSettings(_core(gripper=gripper), object(), lambda _s: None)

    def test_parallel_publishes_no_box_settings(self) -> None:
        live = self._live("parallel")
        keys = {d["key"] for d in live.schema()}
        self.assertFalse(keys & _BOX_KEYS)
        self.assertFalse(set(live.values()) & _BOX_KEYS)
        self.assertIn("reengage", keys)
        with self.assertRaises(ValueError):
            live.apply("box_mode", "toggle")

    def test_parcel_publishes_box_mode(self) -> None:
        keys = {d["key"] for d in self._live("parcel").schema()}
        self.assertTrue({"box_mode", "box_grasp", "box_flush_deg"} <= keys)

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
