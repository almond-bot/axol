"""Teleop's in-core tip damping: the per-arm config and IMU forwarding set up
at startup, and the damper following each arm's engage state."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.cli import teleop as teleop_cli
from almond_axol.cli.config import TeleopCmdConfig
from almond_axol.teleop.teleop import VRTeleop

_CHAIN = {"w": np.tile([0.0, -1.0, 0.0], (7, 1)), "r": np.zeros((7, 3)), "m": np.eye(4)}


class ConfigureTest(unittest.TestCase):
    def _configure(self, cfg: TeleopCmdConfig) -> dict[str, int]:
        with (
            mock.patch("almond_axol.kinematics.solver.KinematicsSolver", lambda: None),
            mock.patch("almond_axol.rt.tipdamp.poe_chain", lambda *a, **k: _CHAIN),
            mock.patch("almond_axol.rt.tipdamp.check_chain", lambda *a, **k: 0.0),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("AXOL_IMU_UDP", None)
            out = teleop_cli._configure_tip_damping(cfg)
            self.env = os.environ.get("AXOL_IMU_UDP")
        return out

    def test_off_by_default(self) -> None:
        cfg = TeleopCmdConfig(cameras={"right_arm": 308521090})
        self.assertEqual(self._configure(cfg), {})
        self.assertIsNone(self.env)
        self.assertIsNone(cfg.axol.right.tip_damp)

    def test_each_arm_with_a_wrist_camera_gets_one(self) -> None:
        cfg = TeleopCmdConfig(cameras={"overhead": 1, "right_arm": 308521090})
        cfg.tip_damp.enable = True
        with self.assertLogs("almond_axol.cli.teleop", level="WARNING") as logs:
            out = self._configure(cfg)
        self.assertEqual(out, {"right": 308521090})
        self.assertIn("left arm needs its wrist camera", "\n".join(logs.output))
        tip = cfg.axol.right.tip_damp
        self.assertEqual(tip.gain, 120.0)
        self.assertEqual(tip.joints, {"shoulder_1": 1.0, "elbow": 0.4})
        self.assertEqual(tip.imu_port, 47811)
        self.assertIs(tip.chain, _CHAIN)
        self.assertIsNone(cfg.axol.left.tip_damp)
        self.assertEqual(self.env, "308521090:47811")

    def test_a_bad_joint_is_refused(self) -> None:
        cfg = TeleopCmdConfig(cameras={"right_arm": 1})
        cfg.tip_damp.enable = True
        cfg.tip_damp.joints = "shoulder_1:1,knee:1"
        with self.assertRaises(SystemExit):
            self._configure(cfg)


class SyncTest(unittest.TestCase):
    def test_the_damper_follows_each_arms_engage_state(self) -> None:
        calls: list[tuple[str, bool]] = []
        fake = SimpleNamespace(
            _robot=SimpleNamespace(set_tip_damping=lambda s, on: calls.append((s, on))),
            _core=SimpleNamespace(left_enabled=False, right_enabled=False),
            _tip_on={"left": False, "right": False},
        )
        sync = VRTeleop._sync_tip_damping
        sync(fake)
        self.assertEqual(calls, [])  # nothing changed
        fake._core.right_enabled = True
        sync(fake)
        sync(fake)
        self.assertEqual(calls, [("right", True)])
        fake._core.left_enabled = True
        fake._core.right_enabled = False
        sync(fake)
        self.assertEqual(calls[1:], [("left", True), ("right", False)])
        sync(fake, force_off=True)
        self.assertEqual(calls[3:], [("left", False)])

    def test_a_robot_without_it_is_left_alone(self) -> None:
        fake = SimpleNamespace(
            _robot=SimpleNamespace(),
            _core=SimpleNamespace(left_enabled=True, right_enabled=True),
            _tip_on={"left": False, "right": False},
        )
        VRTeleop._sync_tip_damping(fake)
        self.assertEqual(fake._tip_on, {"left": False, "right": False})


if __name__ == "__main__":
    unittest.main()
