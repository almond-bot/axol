from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from almond_axol.serve.commands import build_argv
from almond_axol.serve.settings import SettingsStore


class DiagnosticSettingsTest(unittest.TestCase):
    def test_lift_cycle_inherits_gripperless_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.has_gripper": False})

            args = store.merged_args("diag.lift-cycle", {"cycles": 3})

        self.assertEqual(args, {"has_gripper": False, "cycles": 3})
        self.assertEqual(
            build_argv("diag.lift-cycle", args),
            ["--no-gripper", "--cycles", "3"],
        )

    def test_lift_cycle_run_value_overrides_shared_gripper_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"robot.has_gripper": False})

            args = store.merged_args(
                "diag.lift-cycle", {"cycles": 1, "has_gripper": True}
            )

        self.assertTrue(args["has_gripper"])
        self.assertNotIn("--no-gripper", build_argv("diag.lift-cycle", args))


if __name__ == "__main__":
    unittest.main()
