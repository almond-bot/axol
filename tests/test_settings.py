from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.constants import CAN_LEFT
from almond_axol.serve.commands import COMMANDS, build_argv, normalize_boolean_args
from almond_axol.serve.settings import SettingsStore
from almond_axol.utils import certs, state_files


class DiagnosticSettingsTest(unittest.TestCase):
    def test_web_catalog_includes_motor_flash(self) -> None:
        self.assertIn("motor.flash", COMMANDS)
        self.assertEqual(COMMANDS["motor.flash"].hardware_profiles, ("axol",))

    def test_boolean_args_are_canonical_before_argv_emission(self) -> None:
        args = normalize_boolean_args(
            "teleop",
            {
                "mantis": "yes",
                "sim": "OFF",
                "jelly_only": "no",
                "axol.has_gripper": "on",
            },
        )

        self.assertEqual(
            args,
            {
                "mantis": True,
                "sim": False,
                "jelly_only": False,
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
                "--jelly_only",
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
            store.update(values={"axol.has_gripper": "yes"})
            self.assertIs(store.snapshot()["values"]["axol.has_gripper"], True)
            self.assertTrue(store.has_gripper())

            store.update(values={"axol.has_gripper": "off"})
            self.assertIs(store.snapshot()["values"]["axol.has_gripper"], False)
            self.assertFalse(store.has_gripper())

            with self.assertRaisesRegex(
                ValueError, "axol.has_gripper must be a boolean"
            ):
                store.update(values={"axol.has_gripper": 1})

    def test_file_is_a_nested_tree_of_canonical_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(path)
            store.update(
                values={
                    "axol.left_stiffness": 0.7,
                    "axol.left.elbow.kp": 60,
                    "teleop.rest_pose_left": [0, 0.1, 0, 0, 0, 0, 0],
                    "robot.right_channel": "null",
                    "recording.root": "/data",
                },
                cameras={"serials": {"overhead": "1"}},
            )

            document = json.loads(path.read_text())
            self.assertEqual(
                document,
                {
                    "version": 2,
                    "axol": {
                        "left": {"elbow": {"kp": 60}},
                        "left_stiffness": 0.7,
                    },
                    "teleop": {"rest_pose_left": [0, 0.1, 0, 0, 0, 0, 0]},
                    "robot": {"right_channel": "null"},
                    "recording": {"root": "/data"},
                    "cameras": {"serials": {"overhead": "1"}},
                },
            )
            # The file leads with the version and follows the section order
            # the registry declares, so it reads top-down like a config.
            self.assertEqual(
                list(document),
                ["version", "axol", "teleop", "robot", "recording", "cameras"],
            )
            self.assertEqual(store.document(), document)

            reloaded = SettingsStore(path)
            self.assertEqual(reloaded.snapshot(), store.snapshot())
            self.assertNotIn("advanced", reloaded.snapshot())
            # The teleop subtree is the same shape as TeleopCmdConfig, so the
            # merged args are just the flattened file.
            merged = reloaded.merged_args("teleop", {})
            self.assertEqual(merged["axol.left.elbow.kp"], 60)
            self.assertEqual(merged["axol.left_stiffness"], 0.7)
            self.assertEqual(merged["right_channel"], "null")
            self.assertNotIn("root", merged)

    def test_version_one_file_migrates_to_canonical_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "values": {
                            "robot.left_stiffness": 0.7,
                            "robot.gripper_torque_limit": 0.4,
                            "robot.has_gripper": False,
                            "robot.gravity_kd": 0.9,
                            "teleop.mantis_source": "quest",
                            "teleop.frequency": 240,
                            "recording.observe_torques": True,
                            "teleop.id": "op",
                        },
                        "advanced": {
                            "axol.left.elbow.kp": 60,
                            "vr_teleop.frequency": 120,
                            "vr_teleop.hold_to_engage": True,
                            "lerobot.id": "robot-one",
                        },
                        "cameras": None,
                    }
                )
            )
            store = SettingsStore(path)

            self.assertEqual(
                store.snapshot()["values"],
                {
                    "axol.left_stiffness": 0.7,
                    "axol.left.gripper.torque_limit": 0.4,
                    "axol.right.gripper.torque_limit": 0.4,
                    "axol.has_gripper": False,
                    "gravity.kd": 0.9,
                    "mantis.source": "quest",
                    # The visible curated value wins over the hidden one.
                    "teleop.frequency": 240,
                    "lerobot.observe_torques": True,
                    "lerobot_teleop.id": "op",
                    "axol.left.elbow.kp": 60,
                    "teleop.hold_to_engage": True,
                    "lerobot.id": "robot-one",
                },
            )
            self.assertFalse(store.has_gripper())
            collect = store.merged_args("collect-data", {"mantis": True})
            self.assertEqual(collect["mantis_source"], "quest")
            self.assertEqual(collect["teleop_hz"], 240)
            self.assertEqual(collect["teleop_config.vr_teleop_config.frequency"], 240)
            self.assertEqual(collect["robot_config.observe_torques"], True)
            self.assertEqual(collect["teleop_config.id"], "op")
            self.assertEqual(
                collect["robot_config.axol_config.right.gripper.torque_limit"], 0.4
            )
            self.assertEqual(store.merged_args("gravity-comp", {})["kd"], 0.9)

            # The next save rewrites the file in the nested layout.
            store.update(values={"kinematics.pos_weight": 99})
            document = json.loads(path.read_text())
            self.assertEqual(document["version"], 2)
            self.assertNotIn("values", document)
            self.assertNotIn("advanced", document)
            self.assertEqual(
                document["teleop"], {"frequency": 240, "hold_to_engage": True}
            )
            self.assertEqual(document["kinematics"], {"pos_weight": 99})

    def test_update_accepts_pre_v2_names_from_cached_panels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "robot.gripper_max_speed": 3,
                    "teleop.mantis_source": "ultimate",
                },
                advanced={"vr_teleop.position_multiplier": 1.5},
            )
            values = store.snapshot()["values"]
            self.assertEqual(values["axol.left.gripper.max_speed"], 3)
            self.assertEqual(values["axol.right.gripper.max_speed"], 3)
            self.assertEqual(values["mantis.source"], "ultimate")
            self.assertEqual(values["teleop.position_multiplier"], 1.5)

            store.update(values={"robot.gripper_max_speed": None})
            values = store.snapshot()["values"]
            self.assertNotIn("axol.left.gripper.max_speed", values)
            self.assertNotIn("axol.right.gripper.max_speed", values)

    def test_update_rejects_unknown_sections_and_curated_only_leaves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            with self.assertRaisesRegex(KeyError, "nope.x"):
                store.update(values={"nope.x": 1})
            # Sections whose leaves carry explicit per-op targets accept only
            # their curated keys …
            with self.assertRaisesRegex(KeyError, "recording.bogus"):
                store.update(values={"recording.bogus": 1})
            with self.assertRaisesRegex(KeyError, "axol"):
                store.update(values={"axol": 1})
            # … while any leaf of a grafted subsystem is fine: the op schemas
            # decide what it means and build_argv drops what they don't know.
            store.update(values={"axol.left.wrist.kp": 5, "axol.not_a_field": 1})
            merged = store.merged_args("run-policy", {})
            self.assertEqual(merged["robot_config.axol_config.left.wrist.kp"], 5)
            self.assertNotIn(
                "--robot_config.axol_config.not_a_field",
                build_argv("run-policy", merged),
            )

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
                    "lerobot_teleop.id": "../../saved-teleop",
                    "lerobot_teleop.calibration_dir": "/etc/saved-teleop",
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
