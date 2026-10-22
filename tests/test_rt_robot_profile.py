"""Tracking profile selection is explicit and does not change default wire config."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.config import AxolConfig
from almond_axol.rt.robot import Axol


def _hardware():
    config = AxolConfig()
    return SimpleNamespace(
        left=SimpleNamespace(
            _config=config,
            _arm_config=config.left,
            motors=dict.fromkeys(ARM_JOINTS),
            _has_gripper=True,
        ),
        right=None,
        _left_bus=SimpleNamespace(_channel="can_fake"),
    )


class TrackingProfileTest(TestCase):
    def setUp(self):
        patcher = patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_explicit_profile_is_the_only_wire_change(self):
        default = Axol._wrap(_hardware())._config_text()
        mink = Axol._wrap(_hardware(), tracking_profile="mink")._config_text()
        self.assertNotIn("tracking_profile", default)
        self.assertEqual(mink.count("tracking_profile mink\n"), 1)
        self.assertEqual(mink.replace("tracking_profile mink\n", ""), default)

    def test_public_constructor_forwards_profile_without_starting_hardware(self):
        with patch("almond_axol.rt.robot.AxolHardware", return_value=_hardware()):
            robot = Axol(tracking_profile="mink")
        self.assertIn("tracking_profile mink\n", robot._config_text())
        self.assertFalse(robot._armed)
        self.assertFalse(robot._core_started)

    def test_unknown_profile_is_rejected_before_runtime_start(self):
        with self.assertRaisesRegex(ValueError, "Unknown realtime tracking profile"):
            Axol._wrap(_hardware(), tracking_profile="invalid_profile")
