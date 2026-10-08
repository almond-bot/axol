"""``tune.motion`` starts from what teleop runs: the panel settings (a
gripperless robot, a custom end-effector's mass, per-joint gains), unless
``--defaults``."""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from dataclasses import replace
from unittest import mock

from almond_axol.cli.tune import motion as cli
from almond_axol.robot.config import AxolConfig


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli.add_parser(parser.add_subparsers())
    return parser.parse_args(["tune.motion", "--motion", "slow_osc", *argv])


def _panel() -> AxolConfig:
    cfg = AxolConfig(has_gripper=False, left_stiffness=0.8, right_stiffness=0.8)
    cfg.right.wrist_3 = replace(cfg.right.wrist_3, mass=1.3)
    return cfg


class RunConfigTest(unittest.TestCase):
    def _config(self, *argv: str) -> tuple[AxolConfig, str]:
        out = io.StringIO()
        with (
            mock.patch.object(cli, "shared_axol_config", _panel),
            contextlib.redirect_stdout(out),
        ):
            return cli._run_config(_args(*argv)), out.getvalue()

    def test_panel_settings_by_default(self) -> None:
        cfg, text = self._config()
        self.assertFalse(cfg.has_gripper)
        self.assertEqual(cfg.right.wrist_3.mass, 1.3)
        self.assertEqual(cfg.left_stiffness, 0.8)
        self.assertIn("panel settings, gripper no", text)

    def test_defaults_ignore_the_panel(self) -> None:
        cfg, text = self._config("--defaults")
        self.assertTrue(cfg.has_gripper)
        self.assertEqual(cfg.right.wrist_3.mass, AxolConfig().right.wrist_3.mass)
        self.assertIn("calibrated defaults", text)

    def test_flags_override_the_panel(self) -> None:
        cfg, _ = self._config("--stiffness", "1.0")
        self.assertEqual((cfg.left_stiffness, cfg.right_stiffness), (1.0, 1.0))
        self.assertEqual(cfg.right.wrist_3.mass, 1.3)
        cfg, _ = self._config("--defaults", "--no-gripper")
        self.assertFalse(cfg.has_gripper)
        with self.assertRaises(SystemExit):
            self._config("--stiffness", "1.5")


if __name__ == "__main__":
    unittest.main()
