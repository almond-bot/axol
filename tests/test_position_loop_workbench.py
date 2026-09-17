"""tune.position-loop is driven from the dashboard's tuning workbench.

The workbench posts form values keyed by argparse dest; the server derives
the schema from the parser, so what the UI can send is exactly what the
command accepts. Each gain point becomes one persisted run (kind
``position_loop``) so the workbench can chart target vs actual and score it.
"""

from __future__ import annotations

import inspect
import unittest

from almond_axol.cli.tune import position_loop as pl
from almond_axol.serve.commands import COMMANDS, build_argv, get_schema


class RegistryTest(unittest.TestCase):
    def test_registered_as_a_motor_driving_axol_tuning_tool(self) -> None:
        cmd = COMMANDS["tune.position-loop"]
        self.assertTrue(cmd.drives_motors)
        self.assertTrue(cmd.requires_hardware)
        self.assertEqual(cmd.hardware_profiles, ("axol",))
        self.assertEqual(cmd.section, "tuning")

    def test_schema_exposes_what_the_workbench_sends(self) -> None:
        emit = get_schema("tune.position-loop").emit
        self.assertEqual(emit["arm"]["t"], "choice")
        self.assertEqual(emit["arm"]["map"], {"left": "--l", "right": "--r"})
        self.assertEqual(emit["kp"]["t"], "optlist")
        self.assertEqual(emit["ki"]["t"], "optlist")
        self.assertEqual(emit["save_run"]["t"], "flag")
        for key in (
            "joint",
            "mode",
            "accel",
            "center",
            "amp",
            "freq",
            "duration",
            "rate",
            "repeat",
            "ripple_limit",
            "label",
        ):
            self.assertIn(key, emit)
        # The inner speed loop is sweepable; the current loop is not exposed.
        self.assertEqual(emit["speed_kp"]["t"], "optlist")
        self.assertEqual(emit["speed_ki"]["t"], "optlist")
        self.assertNotIn("current_kp", emit)
        self.assertNotIn("current_ki", emit)

    def test_argv_round_trip(self) -> None:
        argv = build_argv(
            "tune.position-loop",
            {
                "arm": "right",
                "joint": "elbow",
                "mode": "track",
                "accel": "0",
                "center": "-90",
                "kp": "0.24 0.36",
                "ki": "0",
                "save_run": True,
                "repeat": "1",
                "label": "wb",
            },
        )
        self.assertEqual(
            argv,
            [
                "--r",
                "--joint",
                "elbow",
                "--mode",
                "track",
                "--accel",
                "0",
                "--center",
                "-90",
                "--kp",
                "0.24",
                "0.36",
                "--ki",
                "0",
                "--save-run",
                "--repeat",
                "1",
                "--label",
                "wb",
            ],
        )


class PersistedRunTest(unittest.TestCase):
    def test_every_gain_point_is_saved_with_a_chartable_series(self) -> None:
        src = inspect.getsource(pl._run)
        self.assertIn('save_run(\n                        "position_loop",', src)
        # Persisted after each printed row, in both stages, only when asked.
        self.assertEqual(src.count("persist(kp, 0.0, m, noisy)"), 1)
        self.assertEqual(src.count("persist(best[0], ki, m, noisy)"), 1)
        self.assertIn("if not args.save_run:", src)
        self.assertIn('"position_kp": kp,', src)
        self.assertIn('"speed_kp": inner["speed_kp"],', src)
        self.assertIn("group=sweep_group", src)

    def test_track_returns_the_series_the_chart_draws(self) -> None:
        src = inspect.getsource(pl._track)
        for key in ('"t"', '"target"', '"actual"', '"current"'):
            self.assertIn(key, src)


if __name__ == "__main__":
    unittest.main()


class DirectTrackingPresetTest(unittest.TestCase):
    def test_a_numeric_zero_accel_preset_reaches_the_command_line(self) -> None:
        """The tab presets accel=0 (direct tracking). Zero is the one value an
        argv builder is most likely to drop as falsy -- and dropping it would
        leave the motor in profiled-motion mode, where the sine test tracks
        nothing at any gain. The browser sends the preset as a number."""
        argv = build_argv(
            "tune.position-loop",
            {
                "arm": "right",
                "joint": "elbow",
                "mode": "track",
                "accel": 0,
                "kp": "0.24",
            },
        )
        i = argv.index("--accel")
        self.assertEqual(argv[i + 1], "0")

    def test_cli_treats_zero_as_a_value_not_as_unset(self) -> None:
        src = inspect.getsource(pl._run)
        self.assertIn("if args.accel is not None:", src)
        self.assertIn("if args.accel == 0.0:", src)
        self.assertIn("direct tracking mode", src)
