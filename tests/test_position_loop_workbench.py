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
        self.assertIn('gains={"position_kp": kp, "position_ki": ki}', src)
        self.assertIn("group=sweep_group", src)

    def test_track_returns_the_series_the_chart_draws(self) -> None:
        src = inspect.getsource(pl._track)
        for key in ('"t"', '"target"', '"actual"', '"current"'):
            self.assertIn(key, src)


if __name__ == "__main__":
    unittest.main()
