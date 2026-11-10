"""Per-motor tools can target a CAN interface outside the arms (a test stand)."""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr

from almond_axol.cli.motor import (
    dump_config,
    flash,
    resolve_channel,
    set_can_id,
    set_config,
    set_zero_pos,
)
from almond_axol.cli.tune import pid
from almond_axol.serve.app import _off_arm_motor_launch
from almond_axol.serve.introspect import build_argparse_schema

_TOOLS = (set_can_id, dump_config, set_config, flash, set_zero_pos)


def _parser(module) -> argparse.ArgumentParser:  # noqa: ANN001
    parser = argparse.ArgumentParser()
    module.add_parser(parser.add_subparsers())
    return parser


class ChannelReplacesSideTest(unittest.TestCase):
    def test_per_motor_tools_make_the_arm_optional(self) -> None:
        for module in _TOOLS:
            with self.subTest(module.__name__):
                arm = next(
                    n
                    for n in build_argparse_schema(module.add_parser).nodes
                    if n["key"] == "arm"
                )
                self.assertFalse(arm["required"])

    def test_arm_tools_still_require_an_arm(self) -> None:
        arm = next(
            n for n in build_argparse_schema(pid.add_parser).nodes if n["key"] == "arm"
        )
        self.assertTrue(arm["required"])

    def test_channel_alone_names_the_bus(self) -> None:
        base = ["motor.set-can-id", "--current-id", "1", "--new-id", "2"]
        base += ["--type", "damiao"]
        parser = _parser(set_can_id)
        self.assertEqual(
            resolve_channel(parser.parse_args([*base, "--channel", "can0"])), "can0"
        )
        self.assertEqual(
            resolve_channel(parser.parse_args([*base, "--l"])), "can_alm_axol_l"
        )
        with self.assertRaisesRegex(SystemExit, "--l or --r"):
            resolve_channel(parser.parse_args(base))

    def test_left_and_right_stay_exclusive(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            _parser(set_can_id).parse_args(
                ["motor.set-can-id", "--l", "--r", "--current-id", "1"]
            )


class OffArmLaunchTest(unittest.TestCase):
    _STATUS = {"channels": {"left": "can_alm_axol_l", "right": "can9"}}

    def _off_arm(self, args: dict, command: str = "motor.set-zero-pos") -> bool:
        return _off_arm_motor_launch(command, args, self._STATUS)

    def test_single_motor_on_a_stand_bus_skips_the_arm_survey(self) -> None:
        self.assertTrue(self._off_arm({"channel": "can0", "id": 1}))

    def test_arm_buses_keep_the_survey(self) -> None:
        for args in (
            {"channel": "can_alm_axol_l", "id": 1},
            {"channel": "can_alm_axol_r", "id": 1},
            {"channel": "can9", "id": 1},  # mapped to the right arm
            {"channel": "can0", "arm": "left", "id": 1},
            {"channel": "can0", "guided": True},
            {"id": 1},
        ):
            with self.subTest(args):
                self.assertFalse(self._off_arm(args))

    def test_only_listed_commands_qualify(self) -> None:
        self.assertFalse(self._off_arm({"channel": "can0"}, "diag.rom-enable"))


if __name__ == "__main__":
    unittest.main()
