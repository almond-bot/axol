"""``axol teleop`` drives the hardware it finds, gated by the Robot tab switches.

Nothing is enabled by hand: the arms are used when their CAN interfaces
exist, Jelly's wheels when ``can_alm_axol_b`` does, its lift when its bus
does. The ``arms`` / ``jelly.wheels`` / ``jelly.lift`` switches are the
operator's opt-out for attached hardware.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.cli.config import TeleopCmdConfig
from almond_axol.cli.teleop import select_hardware
from almond_axol.constants import CAN_BASE, CAN_CHEST, CAN_LEFT, CAN_RIGHT
from almond_axol.robot import jelly as jelly_module
from almond_axol.robot import lift as lift_module
from almond_axol.robot.jelly import JellyConfig


class SelectHardwareTest(unittest.TestCase):
    def _select(self, present: tuple[str, ...], cfg: TeleopCmdConfig | None = None):
        with tempfile.TemporaryDirectory() as directory:
            for name in present:
                (Path(directory) / name).mkdir()
            with (
                patch.object(lift_module, "_SYS_NET", Path(directory)),
                patch.object(jelly_module, "_SYS_NET", Path(directory)),
            ):
                return select_hardware(cfg or TeleopCmdConfig())

    def test_defaults_drive_the_arms(self) -> None:
        cfg = TeleopCmdConfig()
        self.assertTrue(cfg.arms)
        self.assertFalse(hasattr(cfg, "jelly_only"))

    def test_arms_only_robot(self) -> None:
        arms, jelly = self._select((CAN_LEFT, CAN_RIGHT))
        self.assertTrue(arms)
        self.assertIsNone(jelly)

    def test_full_robot_drives_arms_wheels_and_lift(self) -> None:
        arms, jelly = self._select((CAN_LEFT, CAN_RIGHT, CAN_BASE, CAN_CHEST))
        self.assertTrue(arms)
        assert jelly is not None
        self.assertEqual(jelly.channel, CAN_BASE)
        self.assertTrue(jelly.lift)

    def test_jelly_only_robot_skips_the_arms_automatically(self) -> None:
        # No hub attached but Jelly is: drive Jelly, arms untouched (the old
        # --jelly_only, now inferred).
        with self.assertLogs("almond_axol.cli.teleop", level="WARNING") as logs:
            arms, jelly = self._select((CAN_BASE,))
        self.assertFalse(arms)
        assert jelly is not None
        self.assertEqual(jelly.channel, CAN_BASE)
        self.assertIn("driving Jelly only", "\n".join(logs.output))

    def test_no_interfaces_keeps_the_arms_so_the_axol_error_names_them(self) -> None:
        # Nothing attached at all: no silent Jelly-only fallback — the arms
        # stay requested and the Axol connection reports the missing hub.
        arms, jelly = self._select(())
        self.assertTrue(arms)
        self.assertIsNone(jelly)

    def test_arms_switch_off_drives_just_jelly(self) -> None:
        arms, jelly = self._select(
            (CAN_LEFT, CAN_RIGHT, CAN_BASE), TeleopCmdConfig(arms=False)
        )
        self.assertFalse(arms)
        assert jelly is not None
        self.assertEqual(jelly.channel, CAN_BASE)

    def test_arms_off_without_jelly_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "nothing to drive"):
            self._select((CAN_LEFT, CAN_RIGHT), TeleopCmdConfig(arms=False))

    def test_wheels_and_lift_switches_narrow_jelly(self) -> None:
        cfg = TeleopCmdConfig(jelly=JellyConfig(wheels=False))
        arms, jelly = self._select((CAN_LEFT, CAN_RIGHT, CAN_BASE, CAN_CHEST), cfg)
        self.assertTrue(arms)
        assert jelly is not None
        self.assertIsNone(jelly.channel)
        self.assertTrue(jelly.lift)

        cfg = TeleopCmdConfig(jelly=JellyConfig(wheels=False, lift=False))
        arms, jelly = self._select((CAN_LEFT, CAN_RIGHT, CAN_BASE, CAN_CHEST), cfg)
        self.assertTrue(arms)
        self.assertIsNone(jelly)

    def test_a_single_configured_arm_counts_as_the_arms(self) -> None:
        cfg = TeleopCmdConfig(right_channel=None)
        arms, jelly = self._select((CAN_LEFT, CAN_BASE), cfg)
        self.assertTrue(arms)
        assert jelly is not None
        # A disabled ("null") arm never counts as present.
        cfg = TeleopCmdConfig(left_channel=None)
        arms, _ = self._select((CAN_LEFT, CAN_BASE), cfg)
        self.assertFalse(arms)

    def test_sim_models_the_arms_and_never_jelly(self) -> None:
        arms, jelly = self._select((CAN_BASE, CAN_CHEST), TeleopCmdConfig(sim=True))
        self.assertTrue(arms)
        self.assertIsNone(jelly)
        with self.assertRaisesRegex(ValueError, "sim models the arms"):
            self._select((CAN_BASE,), TeleopCmdConfig(sim=True, arms=False))


if __name__ == "__main__":
    unittest.main()
