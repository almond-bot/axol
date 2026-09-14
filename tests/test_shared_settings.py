"""The shared settings file is the default for the CLI and the SDK.

``~/.almond/settings.json`` used to be read only by ``axol serve``; a direct
``axol teleop`` or an SDK ``Axol()`` silently ran the built-in defaults, so
the three surfaces could disagree about gains and channels. These tests pin
that all three now read the same file the same way, and that every layer
that should override it does.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from almond_axol.cli.config import GravityCompCmdConfig, TeleopCmdConfig, parse
from almond_axol.constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.robot import Axol, AxolConfig, Jelly, JellyConfig, Mantis
from almond_axol.serve.settings import SettingsStore
from almond_axol.settings import (
    SHARED,
    load_store,
    shared_axol_config,
    shared_can_channels,
    shared_config,
    shared_overlay,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.teleop import VRTeleop
from almond_axol.vr.config import VRServerConfig

_REST_LEFT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def _real_tmp() -> str:
    """A temp root with no symlinked components (secure state I/O refuses them)."""
    return os.path.realpath(tempfile.gettempdir())


class _StoreCase(unittest.TestCase):
    """A populated settings file in a private directory for every test."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory(dir=_real_tmp())
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "settings.json"
        self.store = SettingsStore(self.path)
        self.store.update(
            values={
                "axol.left_stiffness": 0.7,
                "axol.left.gripper.torque_limit": 0.42,
                "axol.right.gripper.torque_limit": 0.42,
                "axol.left.elbow.mass": 9.9,
                "robot.right_channel": "null",
                "gravity.kd": 0.9,
                "jelly.enabled": True,
                "jelly.max_speed": 0.25,
                "teleop.rest_pose_left": _REST_LEFT,
                "teleop.position_multiplier": 1.5,
                "kinematics.pos_weight": 123.0,
                "vr_server.port": 8123,
            },
        )
        # Make the SDK's default store *this* file rather than the host's.
        patcher = patch("almond_axol.settings.load_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)


class SharedOverlayTest(_StoreCase):
    def test_overlay_is_the_panels_fold_nested_and_typed(self) -> None:
        overlay = shared_overlay("teleop", store=self.store)
        self.assertEqual(overlay["axol"]["left_stiffness"], 0.7)
        self.assertEqual(overlay["axol"]["left"]["elbow"]["mass"], 9.9)
        self.assertEqual(overlay["axol"]["left"]["gripper"]["torque_limit"], 0.42)
        # ``"null"`` is the stored spelling of a disabled arm; the overlay
        # carries a real None, exactly as the panel's argv would decode.
        self.assertIn("right_channel", overlay)
        self.assertIsNone(overlay["right_channel"])
        self.assertEqual(overlay["teleop"]["rest_pose_left"], _REST_LEFT)
        self.assertEqual(overlay["vr_server"]["port"], 8123)
        self.assertEqual(overlay["jelly"]["enabled"], True)

        gravity = shared_overlay("gravity-comp", store=self.store)
        self.assertEqual(gravity["kd"], 0.9)
        self.assertEqual(gravity["axol"]["left"]["elbow"]["mass"], 9.9)
        # Teleop-only settings never leak into another op's overlay.
        self.assertNotIn("teleop", gravity)

    def test_unknown_operation_has_no_settings(self) -> None:
        self.assertEqual(shared_overlay("not-a-command", store=self.store), {})

    def test_shared_config_decodes_a_subtree_over_the_defaults(self) -> None:
        axol = shared_axol_config(self.store)
        self.assertIsInstance(axol, AxolConfig)
        self.assertEqual(axol.left_stiffness, 0.7)
        self.assertEqual(axol.left.elbow.mass, 9.9)
        # Untouched fields keep the (calibrated) defaults.
        self.assertEqual(axol.right.elbow.mass, AxolConfig().right.elbow.mass)
        self.assertEqual(axol.left.gripper.torque_limit, 0.42)

        teleop = shared_config(VRTeleopConfig, "teleop", "teleop", store=self.store)
        np.testing.assert_allclose(teleop.rest_pose_left, _REST_LEFT, rtol=1e-6)
        self.assertEqual(teleop.position_multiplier, 1.5)
        self.assertEqual(shared_can_channels(self.store), (CAN_LEFT, None))


class CliSharedSettingsTest(_StoreCase):
    def test_direct_cli_reads_the_settings_file_by_default(self) -> None:
        cfg = parse(
            TeleopCmdConfig, ["--settings_path", str(self.path)], settings_op="teleop"
        )
        self.assertEqual(cfg.axol.left_stiffness, 0.7)
        self.assertEqual(cfg.axol.left.elbow.mass, 9.9)
        self.assertIsNone(cfg.right_channel)
        self.assertEqual(cfg.vr_server.port, 8123)
        self.assertTrue(cfg.jelly.enabled)
        np.testing.assert_allclose(cfg.teleop.rest_pose_left, _REST_LEFT, rtol=1e-6)

        gravity = parse(
            GravityCompCmdConfig,
            ["--settings_path", str(self.path)],
            settings_op="gravity-comp",
        )
        self.assertEqual(gravity.kd, 0.9)
        self.assertEqual(gravity.axol.left.elbow.mass, 9.9)
        self.assertIsNone(gravity.right_channel)

    def test_default_path_is_the_almond_home_settings_file(self) -> None:
        # Without --settings_path the CLI opens the store the panel writes.
        with patch("almond_axol.settings.load_store") as load_store:
            load_store.return_value = self.store
            cfg = parse(TeleopCmdConfig, [], settings_op="teleop")
        load_store.assert_called_once_with(None)
        self.assertEqual(cfg.axol.left_stiffness, 0.7)

    def test_settings_sit_below_config_file_and_flags(self) -> None:
        config_path = Path(self._dir.name) / "teleop.json"
        config_path.write_text(
            json.dumps({"axol": {"left_stiffness": 0.5, "right_stiffness": 0.6}})
        )
        cfg = parse(
            TeleopCmdConfig,
            [
                "--settings_path",
                str(self.path),
                "--config_path",
                str(config_path),
                "--axol.right_stiffness",
                "0.3",
            ],
            settings_op="teleop",
        )
        self.assertEqual(cfg.axol.left_stiffness, 0.5)  # file beats settings
        self.assertEqual(cfg.axol.right_stiffness, 0.3)  # flag beats file
        self.assertEqual(cfg.axol.left.elbow.mass, 9.9)  # settings still apply

    def test_no_settings_and_missing_settings_op_use_bare_defaults(self) -> None:
        for argv, settings_op in (
            (["--settings_path", str(self.path), "--no_settings"], "teleop"),
            ([], None),
        ):
            with self.subTest(argv=argv, settings_op=settings_op):
                cfg = parse(TeleopCmdConfig, argv, settings_op=settings_op)
                self.assertEqual(cfg.axol.left_stiffness, 1.0)
                self.assertEqual(cfg.axol.left.elbow.mass, AxolConfig().left.elbow.mass)

    def test_settings_flags_are_not_config_fields(self) -> None:
        cfg = parse(
            TeleopCmdConfig,
            ["--settings_path", str(self.path), "--no_settings"],
            settings_op="teleop",
        )
        self.assertFalse(hasattr(cfg, "settings_path"))
        self.assertFalse(hasattr(cfg, "no_settings"))
        # Without a settings op the flags don't exist at all.
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse(TeleopCmdConfig, ["--no_settings"])

    def test_explicit_missing_settings_file_is_a_usage_error(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse(
                TeleopCmdConfig,
                ["--settings_path", str(self.path.with_name("nope.json"))],
                settings_op="teleop",
            )

    def test_unreadable_settings_file_fails_closed(self) -> None:
        # A file that exists but cannot be read must not silently become the
        # calibrated defaults: that is a different robot. Both the explicit
        # path and the default one refuse, and the message names the exit.
        corrupt = self.path.with_name("corrupt.json")
        corrupt.write_text("{not json")
        # _StoreCase stubs load_store with this test's store; use the real one.
        real_loader = patch("almond_axol.settings.load_store", load_store)
        for argv in (["--settings_path", str(corrupt)], []):
            with self.subTest(argv=argv):
                stderr = StringIO()
                with (
                    real_loader,
                    patch("almond_axol.serve.settings.SETTINGS_PATH", corrupt),
                    redirect_stderr(stderr),
                    self.assertRaises(SystemExit),
                ):
                    parse(TeleopCmdConfig, argv, settings_op="teleop")
                self.assertIn("could not read the settings file", stderr.getvalue())
                self.assertIn(str(corrupt), stderr.getvalue())
                self.assertIn("--no_settings", stderr.getvalue())
        # The escape hatch works, and a *missing* default file is still fine.
        with real_loader, patch("almond_axol.serve.settings.SETTINGS_PATH", corrupt):
            cfg = parse(TeleopCmdConfig, ["--no_settings"], settings_op="teleop")
        self.assertEqual(cfg.axol.left_stiffness, 1.0)
        with (
            real_loader,
            patch(
                "almond_axol.serve.settings.SETTINGS_PATH",
                self.path.with_name("none.json"),
            ),
        ):
            cfg = parse(TeleopCmdConfig, [], settings_op="teleop")
        self.assertEqual(cfg.axol.left_stiffness, 1.0)

    def test_sdk_store_is_strict_but_tolerates_a_missing_file(self) -> None:
        corrupt = self.path.with_name("corrupt.json")
        corrupt.write_text("{not json")
        with self.assertRaises(ValueError):
            load_store(corrupt)
        # Serve-style tolerance is opt-in and loud.
        with self.assertLogs("almond_axol.serve.settings", "ERROR"):
            tolerant = load_store(corrupt, strict=False)
        self.assertEqual(tolerant.snapshot()["values"], {})
        self.assertEqual(
            load_store(self.path.with_name("none.json")).snapshot()["values"], {}
        )

    def test_mantis_args_select_the_rig_channel_map(self) -> None:
        cfg = parse(
            TeleopCmdConfig,
            ["--settings_path", str(self.path), "--mantis", "true"],
            settings_op="teleop",
            settings_args={"mantis": True},
        )
        self.assertEqual(
            (cfg.left_channel, cfg.right_channel), (CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT)
        )


class SdkSharedSettingsTest(_StoreCase):
    def setUp(self) -> None:
        super().setUp()
        # ``Axol()`` is the realtime-core robot; constructing it only needs
        # the core binary to *resolve*, which these tests never launch.
        patcher = patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_axol_defaults_come_from_the_shared_settings(self) -> None:
        with patch("almond_axol.robot.axol.CanBus") as can_bus:
            axol = Axol()
        # ``robot.right_channel`` is saved as null: only the left bus opens.
        can_bus.assert_called_once_with(CAN_LEFT)
        self.assertIsNone(axol.right)
        assert axol.left is not None
        # Stiffness is baked into the gains at construction; the saved link
        # mass reaches the gravity model the arm actually runs.
        self.assertEqual(axol.left._config.left.elbow.mass, 9.9)
        self.assertEqual(axol.left._config.left.gripper.torque_limit, 0.42)
        self.assertNotEqual(
            axol.left._config.left.elbow.kp, AxolConfig().resolved().left.elbow.kp
        )

    def test_axol_explicit_arguments_override_the_settings(self) -> None:
        with patch("almond_axol.robot.axol.CanBus") as can_bus:
            axol = Axol(config=AxolConfig(), right_channel="can_bench")
        self.assertEqual(
            [call.args for call in can_bus.call_args_list],
            [(CAN_LEFT,), ("can_bench",)],
        )
        assert axol.left is not None
        self.assertEqual(
            axol.left._config.left.elbow.mass, AxolConfig().left.elbow.mass
        )

        with patch("almond_axol.robot.axol.CanBus") as can_bus:
            Axol(left_channel=None, right_channel="can_only")
        can_bus.assert_called_once_with("can_only")

    def test_shared_sentinel_is_distinct_from_disabled(self) -> None:
        self.assertIsNot(SHARED, None)
        self.assertEqual(repr(SHARED), "SHARED")

    def test_mantis_defaults_come_from_the_shared_settings(self) -> None:
        self.store.update(values={"mantis.left_channel": "can_rig_a"})
        with patch("almond_axol.robot.mantis.CanBus") as can_bus:
            mantis = Mantis()
        self.assertEqual(
            [call.args for call in can_bus.call_args_list],
            [("can_rig_a",), (CAN_MANTIS_RIGHT,)],
        )
        assert mantis.left is not None
        self.assertEqual(mantis.left._gripper_config.torque_limit, 0.42)

    def test_jelly_and_vr_teleop_defaults_come_from_the_shared_settings(self) -> None:
        jelly = Jelly()
        self.assertTrue(jelly._config.enabled)
        self.assertEqual(jelly._config.max_speed, 0.25)
        self.assertFalse(Jelly(JellyConfig())._config.enabled)

        with patch("almond_axol.teleop.teleop.VRServer") as server:
            teleop = VRTeleop(MagicMock())
        np.testing.assert_allclose(teleop._config.rest_pose_left, _REST_LEFT, rtol=1e-6)
        self.assertEqual(teleop._config.position_multiplier, 1.5)
        self.assertEqual(teleop._kinematics_config.pos_weight, 123.0)
        self.assertEqual(server.call_args.args[0].port, 8123)

        with patch("almond_axol.teleop.teleop.VRServer") as server:
            explicit = VRTeleop(
                MagicMock(),
                config=VRTeleopConfig(),
                kinematics_config=KinematicsConfig(),
                vr_server_config=VRServerConfig(),
            )
        self.assertEqual(explicit._config.position_multiplier, 1.0)
        self.assertEqual(
            explicit._kinematics_config.pos_weight, KinematicsConfig().pos_weight
        )
        self.assertEqual(server.call_args.args[0].port, VRServerConfig().port)


if __name__ == "__main__":
    unittest.main()
