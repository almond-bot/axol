"""ROM soak payload: the gripped item's mass reaches the gravity model.

A full-robot ``diag.rom-enable`` run clamps an item in each gripper; the
``--payload-kg`` mass (default 2 kg) is added to the gravity model as each
gripper closes. The panel renders the option as a number field from the
argparse schema, so the default and the emitted flag are part of the contract.
"""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock, patch

from almond_axol.constants import GRIPPER_TIP_OFFSET
from almond_axol.diagnostics.rom import enable as rom
from almond_axol.robot.gravity import MAX_PAYLOAD_KG
from almond_axol.serve.commands import build_argv
from almond_axol.serve.introspect import build_argparse_schema


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    rom._add_arguments(parser)
    return parser.parse_args(argv)


def _fields(nodes: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for node in nodes:
        if node.get("kind") == "field":
            out[node["key"]] = node
        out.update(_fields(node.get("children", [])))
    return out


class PayloadArgumentTest(unittest.TestCase):
    def test_default_is_two_kg(self) -> None:
        self.assertEqual(_parse([]).payload_kg, 2.0)
        self.assertEqual(rom.DEFAULT_PAYLOAD_KG, 2.0)

    def test_value_is_parsed_as_kg(self) -> None:
        self.assertEqual(_parse(["--payload-kg", "3.5"]).payload_kg, 3.5)

    def test_check_accepts_zero_through_the_model_limit(self) -> None:
        self.assertEqual(rom.check_payload_kg(0.0), 0.0)
        self.assertEqual(rom.check_payload_kg(MAX_PAYLOAD_KG), MAX_PAYLOAD_KG)

    def test_check_refuses_out_of_range_before_bring_up(self) -> None:
        for bad in (-0.5, MAX_PAYLOAD_KG + 0.1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                rom.check_payload_kg(bad)

    def test_run_cli_passes_the_payload_to_the_run(self) -> None:
        captured: dict = {}

        async def fake_run_axol(**kwargs: object) -> None:
            captured.update(kwargs)

        with patch.object(rom, "run_axol", fake_run_axol):
            rom.run_cli(_parse(["--payload-kg", "1.5"]))
        self.assertEqual(captured["payload_kg"], 1.5)

    def test_run_cli_refuses_a_bad_payload_without_starting(self) -> None:
        run = MagicMock()
        with patch.object(rom, "run_axol", run), self.assertRaises(SystemExit):
            rom.run_cli(_parse(["--payload-kg", "9"]))
        run.assert_not_called()


class PanelFieldTest(unittest.TestCase):
    def test_panel_renders_a_number_field_defaulting_to_two_kg(self) -> None:
        schema = build_argparse_schema(rom.add_parser)
        field = _fields(schema.nodes)["payload_kg"]
        self.assertEqual(field["type"], "number")
        self.assertEqual(field["default"], 2.0)
        self.assertEqual(field["label"], "Payload kg per gripper")

    def test_panel_value_becomes_the_flag(self) -> None:
        argv = build_argv("diag.rom-enable", {"payload_kg": 2.5})
        self.assertEqual(argv[argv.index("--payload-kg") + 1], "2.5")


class ApplyGraspPayloadTest(unittest.TestCase):
    def test_item_is_a_point_mass_at_the_fingertips(self) -> None:
        robot = MagicMock()
        rom.apply_grasp_payload(robot, "right", 2.0)
        robot.set_payload.assert_called_once_with("right", 2.0, GRIPPER_TIP_OFFSET)

    def test_zero_leaves_the_model_alone(self) -> None:
        robot = MagicMock()
        rom.apply_grasp_payload(robot, "left", 0.0)
        robot.set_payload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
