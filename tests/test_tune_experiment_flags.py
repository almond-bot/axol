"""The replay tuning commands actually carry the control experiments.

``tune.motion`` / ``tune.repeatability`` drive the robot through production
``motion_control``, so the realtime core applies whatever ``experiments`` the
config holds. Both used to build a bare ``AxolConfig()``, which silently
pinned every replay to the shipped control law — the one thing the
deterministic A/B harness must not do. These lock that down.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.cli.tune._experiments import (
    announce,
    base_config,
    describe,
    parse_experiment_overrides,
)
from almond_axol.robot.config import AxolConfig, ControlExperiments


class ExperimentOverrideParsingTest(unittest.TestCase):
    def test_parses_each_field_type(self) -> None:
        got = parse_experiment_overrides(
            [
                "friction_k_max=400",
                "friction_slew=30",
                "dither_square=true",
                "wire_mode=a9",
            ]
        )
        self.assertEqual(got["friction_k_max"], 400.0)
        self.assertEqual(got["friction_slew"], 30.0)
        self.assertIs(got["dither_square"], True)
        self.assertEqual(got["wire_mode"], "a9")
        self.assertEqual(parse_experiment_overrides(None), {})

    def test_rejects_bad_specs_before_the_arm_moves(self) -> None:
        for spec in (
            "friction_k_max",  # no value
            "bogus=1",  # unknown field
            "friction_k_max=fast",  # number wanted
            "dither_square=maybe",  # boolean wanted
        ):
            with self.assertRaises(SystemExit, msg=spec):
                parse_experiment_overrides([spec])

    def test_invalid_combinations_fail_at_config_time(self) -> None:
        # The core would refuse this; failing here keeps it off the robot.
        with self.assertRaises(SystemExit):
            base_config(
                stiffness=1.0,
                has_gripper=True,
                experiment_overrides={"wire_mode": "A9"},
                settings=False,
            )


class BaseConfigTest(unittest.TestCase):
    def test_overrides_reach_the_config_and_others_keep_defaults(self) -> None:
        config = base_config(
            stiffness=1.0,
            has_gripper=True,
            experiment_overrides={"friction_k_max": 400.0, "friction_slew": 30.0},
            settings=False,
        )
        self.assertEqual(config.experiments.friction_k_max, 400.0)
        self.assertEqual(config.experiments.friction_slew, 30.0)
        self.assertEqual(config.experiments.stiction_gain, 0.0)
        self.assertFalse(config.experiments.is_default())

    def test_shared_settings_experiments_are_honoured(self) -> None:
        # The panel writes an `experiments` block; a replay must run it,
        # which is the whole bug this guards.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {"axol": {"experiments": {"integrator_hz": 0.3, "dither_nm": 0.2}}}
                )
            )
            from almond_axol import settings as settings_mod

            store = settings_mod.load_store(path)
            with patch.object(settings_mod, "load_store", return_value=store):
                config = base_config(stiffness=1.0, has_gripper=True)
        self.assertEqual(config.experiments.integrator_hz, 0.3)
        self.assertEqual(config.experiments.dither_nm, 0.2)

    def test_no_settings_ignores_the_file(self) -> None:
        config = base_config(stiffness=0.8, has_gripper=False, settings=False)
        self.assertTrue(config.experiments.is_default())
        self.assertEqual(config.left_stiffness, 0.8)
        self.assertFalse(config.has_gripper)

    def test_stiffness_and_gripper_survive_the_settings_base(self) -> None:
        config = base_config(stiffness=0.5, has_gripper=False, settings=False)
        self.assertEqual((config.left_stiffness, config.right_stiffness), (0.5, 0.5))


class ProvenanceTest(unittest.TestCase):
    def test_describe_reports_only_what_differs(self) -> None:
        self.assertEqual(describe(ControlExperiments()), {})
        self.assertEqual(
            describe(ControlExperiments(friction_k_max=400.0)),
            {"friction_k_max": 400.0},
        )

    def test_announce_returns_the_set_it_prints(self) -> None:
        exp = ControlExperiments(dither_nm=0.2, wire_mode="a9")
        active = announce(exp)
        self.assertEqual(active, {"dither_nm": 0.2, "wire_mode": "a9"})
        self.assertEqual(announce(AxolConfig().experiments), {})


class ParserWiringTest(unittest.TestCase):
    """Both replay commands expose the flags and neither builds a bare config."""

    def test_both_commands_register_the_flags(self) -> None:
        import argparse

        from almond_axol.cli.tune import motion, repeatability

        for module in (motion, repeatability):
            subparsers = argparse.ArgumentParser().add_subparsers()
            module.add_parser(subparsers)
            (created,) = subparsers.choices.values()
            flags = {
                opt for action in created._actions for opt in action.option_strings
            }
            self.assertIn("--experiment", flags, module.__name__)
            self.assertIn("--no-settings", flags, module.__name__)

    def test_neither_constructs_a_bare_axolconfig(self) -> None:
        # The regression that made every replay run the shipped law: a
        # hand-built AxolConfig() ignores both the settings file and the
        # --experiment flag, so the harness scored the shipped controller
        # whatever the operator selected.
        from almond_axol.cli.tune import motion, repeatability

        for module in (motion, repeatability):
            source = Path(module.__file__).read_text()
            self.assertNotIn("AxolConfig(\n", source, module.__name__)


if __name__ == "__main__":
    unittest.main()
