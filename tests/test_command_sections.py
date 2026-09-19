"""The Diagnostics dashboard renders each command in exactly one section."""

from __future__ import annotations

import unittest

from almond_axol.serve.commands import COMMANDS

# The dashboard's homes for a Diagnostics command: the Tests and Helpers card
# grids, or a tab of the tuning workbench.
_SECTIONS = {"test", "helper", "tuning"}

# Tuning-section commands the workbench has no tab for stay CLI-only; the
# dashboard renders no other grid for them. Keep this list honest.
_CLI_ONLY_TUNING = {"tune.filter"}
_WORKBENCH_TABS = {
    "tune.a4",
    "tune.pid",
    "tune.friction",
    "tune.gravity",
    "tune.factory",
    "tune.motion",
    "motion.build",
    "diag.offline",
}


class DiagnosticsSectionsTest(unittest.TestCase):
    def test_every_diagnostics_command_has_one_valid_section(self) -> None:
        for cmd_id, cmd in COMMANDS.items():
            if cmd.category != "Diagnostics":
                continue
            with self.subTest(cmd_id):
                self.assertIn(cmd.section, _SECTIONS)

    def test_section_is_only_used_inside_diagnostics(self) -> None:
        for cmd_id, cmd in COMMANDS.items():
            if cmd.category == "Diagnostics":
                continue
            with self.subTest(cmd_id):
                self.assertIsNone(cmd.section)

    def test_tuning_commands_match_the_workbench_tabs(self) -> None:
        tuning = {
            cmd_id
            for cmd_id, cmd in COMMANDS.items()
            if cmd.category == "Diagnostics" and cmd.section == "tuning"
        }
        self.assertEqual(tuning, _WORKBENCH_TABS | _CLI_ONLY_TUNING)

    def test_expected_test_and_helper_cards(self) -> None:
        by_section: dict[str, set[str]] = {"test": set(), "helper": set()}
        for cmd_id, cmd in COMMANDS.items():
            if cmd.category == "Diagnostics" and cmd.section in by_section:
                by_section[cmd.section].add(cmd_id)
        self.assertEqual(
            by_section["test"],
            {"diag.rom-enable", "diag.lift-cycle", "diag.zed-cable"},
        )
        self.assertEqual(
            by_section["helper"], {"diag.rom-disable", "lift.home", "lift.goto"}
        )


if __name__ == "__main__":
    unittest.main()
