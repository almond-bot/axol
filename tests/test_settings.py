from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from almond_axol.constants import CAN_LEFT
from almond_axol.serve.commands import build_argv, normalize_boolean_args
from almond_axol.serve.settings import SettingsStore


class DiagnosticSettingsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
