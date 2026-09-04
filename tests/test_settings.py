from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.constants import CAN_LEFT
from almond_axol.serve.commands import COMMANDS, build_argv, normalize_boolean_args
from almond_axol.serve.settings import SettingsStore
from almond_axol.utils import certs, state_files


class DiagnosticSettingsTest(unittest.TestCase):
    def test_web_catalog_excludes_local_firmware_paths(self) -> None:
        self.assertNotIn("motor.flash", COMMANDS)

    def test_boolean_args_are_canonical_before_argv_emission(self) -> None:
        args = normalize_boolean_args(
            "teleop",
            {
                "mantis": "yes",
                "sim": "OFF",
                "cart_only": "no",
                "axol.has_gripper": "on",
            },
        )

        self.assertEqual(
            args,
            {
                "mantis": True,
                "sim": False,
                "cart_only": False,
                "axol.has_gripper": True,
            },
        )
        self.assertEqual(
            build_argv("teleop", args),
            [
                "--mantis",
                "true",
                "--sim",
                "false",
                "--cart_only",
                "false",
                "--axol.has_gripper",
                "true",
            ],
        )

    def test_boolean_args_reject_ambiguous_values(self) -> None:
        for value in (1, 0, "1", "0", "maybe", [], {}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "mantis must be a boolean"),
            ):
                normalize_boolean_args("teleop", {"mantis": value})

    def test_argparse_side_flags_use_the_same_boolean_parser(self) -> None:
        args = normalize_boolean_args(
            "diag.lift-cycle", {"no_left": "on", "no_right": "off"}
        )

        self.assertEqual(args, {"no_left": True, "no_right": False})
        self.assertEqual(build_argv("diag.lift-cycle", args), ["--no-left"])

    def test_boolean_settings_are_canonical_and_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.has_gripper": "yes"})
            self.assertIs(store.snapshot()["values"]["robot.has_gripper"], True)
            self.assertTrue(store.has_gripper())

            store.update(values={"robot.has_gripper": "off"})
            self.assertIs(store.snapshot()["values"]["robot.has_gripper"], False)
            self.assertFalse(store.has_gripper())

            with self.assertRaisesRegex(
                ValueError, "robot.has_gripper must be a boolean"
            ):
                store.update(values={"robot.has_gripper": 1})

    def test_lift_cycle_does_not_inherit_gripper_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.has_gripper": False})

            args = store.merged_args("diag.lift-cycle", {"cycles": 3})

        self.assertEqual(args, {"cycles": 3})
        self.assertEqual(build_argv("diag.lift-cycle", args), ["--cycles", "3"])

    def test_lift_cycle_inherits_axol_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "robot.left_channel": "can-custom-left",
                    "robot.right_channel": "can-custom-right",
                }
            )

            args = store.merged_args("diag.lift-cycle", {"cycles": 1})

        self.assertEqual(args["left_channel"], "can-custom-left")
        self.assertEqual(args["right_channel"], "can-custom-right")

    def test_lift_cycle_translates_disabled_channel_to_skip_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.left_channel": "null"})

            args = store.merged_args("diag.lift-cycle", {"cycles": 1})

        self.assertNotIn("left_channel", args)
        self.assertTrue(args["no_left"])
        self.assertIn("--no-left", build_argv("diag.lift-cycle", args))

    def test_axol_channels_must_be_distinct_when_both_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            with self.assertRaisesRegex(ValueError, "distinct interfaces"):
                store.update(
                    values={
                        "robot.left_channel": "can-shared",
                        "robot.right_channel": "can-shared",
                    }
                )
            self.assertEqual(store.snapshot()["values"], {})

            store.update(values={"robot.left_channel": "null"})
            self.assertIsNone(store.can_channels()[0])

    def test_effective_axol_channels_match_direct_and_nested_operation_args(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "robot.left_channel": "can-saved-left",
                    "robot.right_channel": "can-saved-right",
                }
            )

            self.assertEqual(
                store.effective_axol_can_channels(
                    "teleop", {"left_channel": "can-run-left"}
                ),
                ("can-run-left", "can-saved-right"),
            )
            self.assertEqual(
                store.effective_axol_can_channels(
                    "collect-dagger",
                    {"robot_config.right_channel": "can-run-right"},
                ),
                ("can-saved-left", "can-run-right"),
            )

    def test_effective_axol_channels_follow_build_argv_null_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.left_channel": "can-saved-left"})

            self.assertEqual(
                store.effective_axol_can_channels("teleop", {"left_channel": None})[0],
                CAN_LEFT,
            )
            self.assertIsNone(
                store.effective_axol_can_channels("teleop", {"left_channel": "null"})[0]
            )

    def test_hosted_request_cannot_override_tls_or_calibration_paths(self) -> None:
        cases = {
            "teleop": {
                "vr_server.certfile": "/etc/host-cert",
                "vr_server.keyfile": "/etc/host-key",
            },
            "collect-data": {
                "teleop_config.vr_server_config.certfile": "/etc/data-cert",
                "teleop_config.vr_server_config.keyfile": "/etc/data-key",
                "robot_config.calibration_dir": "/etc/data-robot",
                "robot_config.id": "../../data-robot",
                "teleop_config.calibration_dir": "/etc/data-teleop",
                "teleop_config.id": "../../data-teleop",
            },
            "collect-dagger": {
                "teleop_config.vr_server_config.certfile": "/etc/dagger-cert",
                "teleop_config.vr_server_config.keyfile": "/etc/dagger-key",
                "robot_config.calibration_dir": "/etc/dagger-robot",
                "robot_config.id": "../../dagger-robot",
                "teleop_config.calibration_dir": "/etc/dagger-teleop",
                "teleop_config.id": "../../dagger-teleop",
                "policy_path": "operator/policy",
                "rerun_ip": "192.0.2.10",
            },
            "replay-dataset": {
                "robot_config.calibration_dir": "/etc/replay-robot",
                "robot_config.id": "../../replay-robot",
            },
            "run-policy": {
                "robot_config.calibration_dir": "/etc/policy-robot",
                "robot_config.id": "../../policy-robot",
                "policy_path": "operator/policy",
                "server_host": "192.0.2.20",
                "rerun_ip": "192.0.2.21",
            },
        }
        calibration_fields = {
            "robot_config.calibration_dir",
            "robot_config.id",
            "teleop_config.calibration_dir",
            "teleop_config.id",
        }

        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            with patch.object(
                state_files, "privileged_service_active", return_value=True
            ):
                merged = {
                    operation: store.merged_args(operation, request)
                    for operation, request in cases.items()
                }

        self.assertEqual(merged["teleop"]["vr_server.certfile"], certs.CERTFILE)
        self.assertEqual(merged["teleop"]["vr_server.keyfile"], certs.KEYFILE)
        for operation in ("collect-data", "collect-dagger"):
            self.assertEqual(
                merged[operation]["teleop_config.vr_server_config.certfile"],
                certs.CERTFILE,
            )
            self.assertEqual(
                merged[operation]["teleop_config.vr_server_config.keyfile"],
                certs.KEYFILE,
            )
        for operation in (
            "collect-data",
            "collect-dagger",
            "replay-dataset",
            "run-policy",
        ):
            self.assertTrue(calibration_fields.isdisjoint(merged[operation]))

        # Hosted path confinement must not remove intentional model/network
        # selections that are independent of local filesystem ownership.
        self.assertEqual(merged["collect-dagger"]["policy_path"], "operator/policy")
        self.assertEqual(merged["collect-dagger"]["rerun_ip"], "192.0.2.10")
        self.assertEqual(merged["run-policy"]["policy_path"], "operator/policy")
        self.assertEqual(merged["run-policy"]["server_host"], "192.0.2.20")
        self.assertEqual(merged["run-policy"]["rerun_ip"], "192.0.2.21")

    def test_hosted_saved_advanced_paths_are_confined_for_every_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "teleop.id": "../../saved-teleop",
                    "teleop.calibration_dir": "/etc/saved-teleop",
                },
                advanced={
                    "vr_server.certfile": "/etc/saved-cert",
                    "vr_server.keyfile": "/etc/saved-key",
                    "lerobot.id": "../../saved-robot",
                    "lerobot.calibration_dir": "/etc/saved-robot",
                },
            )
            with patch.object(
                state_files, "privileged_service_active", return_value=True
            ):
                merged = {
                    operation: store.merged_args(operation, {})
                    for operation in (
                        "teleop",
                        "collect-data",
                        "collect-dagger",
                        "replay-dataset",
                        "run-policy",
                    )
                }

        self.assertEqual(merged["teleop"]["vr_server.certfile"], certs.CERTFILE)
        self.assertEqual(merged["teleop"]["vr_server.keyfile"], certs.KEYFILE)
        for operation in ("collect-data", "collect-dagger"):
            self.assertEqual(
                merged[operation]["teleop_config.vr_server_config.certfile"],
                certs.CERTFILE,
            )
            self.assertEqual(
                merged[operation]["teleop_config.vr_server_config.keyfile"],
                certs.KEYFILE,
            )
            self.assertNotIn("teleop_config.id", merged[operation])
            self.assertNotIn("teleop_config.calibration_dir", merged[operation])
        for operation in (
            "collect-data",
            "collect-dagger",
            "replay-dataset",
            "run-policy",
        ):
            self.assertNotIn("robot_config.id", merged[operation])
            self.assertNotIn("robot_config.calibration_dir", merged[operation])

    def test_non_root_serve_preserves_custom_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            with patch.object(
                state_files, "privileged_service_active", return_value=False
            ):
                teleop = store.merged_args(
                    "teleop",
                    {
                        "vr_server.certfile": "/srv/custom-cert",
                        "vr_server.keyfile": "/srv/custom-key",
                    },
                )
                collection = store.merged_args(
                    "collect-dagger",
                    {
                        "teleop_config.vr_server_config.certfile": "/srv/data-cert",
                        "teleop_config.vr_server_config.keyfile": "/srv/data-key",
                        "robot_config.calibration_dir": "/srv/robot-calibration",
                        "robot_config.id": "robot-one",
                        "teleop_config.calibration_dir": "/srv/teleop-calibration",
                        "teleop_config.id": "teleop-one",
                    },
                )
                policy = store.merged_args(
                    "run-policy",
                    {
                        "robot_config.calibration_dir": "/srv/policy-calibration",
                        "robot_config.id": "policy-robot",
                    },
                )

        self.assertEqual(teleop["vr_server.certfile"], "/srv/custom-cert")
        self.assertEqual(teleop["vr_server.keyfile"], "/srv/custom-key")
        self.assertEqual(
            collection["teleop_config.vr_server_config.certfile"], "/srv/data-cert"
        )
        self.assertEqual(
            collection["teleop_config.vr_server_config.keyfile"], "/srv/data-key"
        )
        self.assertEqual(
            collection["robot_config.calibration_dir"], "/srv/robot-calibration"
        )
        self.assertEqual(collection["robot_config.id"], "robot-one")
        self.assertEqual(
            collection["teleop_config.calibration_dir"], "/srv/teleop-calibration"
        )
        self.assertEqual(collection["teleop_config.id"], "teleop-one")
        self.assertEqual(
            policy["robot_config.calibration_dir"], "/srv/policy-calibration"
        )
        self.assertEqual(policy["robot_config.id"], "policy-robot")


if __name__ == "__main__":
    unittest.main()
