from __future__ import annotations

import unittest
from pathlib import Path

import pytest

from almond_axol.tracker.survive_config import native_seed, parse_survive_config


# config_save() writes the global members without outer braces, then appends
# lighthouse objects without commas between those top-level groups. Values
# written by json_write_float/json_write_uint32 are strings, not JSON numbers.
NATIVE_CONFIG = (
    Path(__file__).parent / "fixtures" / "libsurvive-config.native"
).read_text()


class ParseSurviveConfigTest(unittest.TestCase):
    def test_reads_native_writer_output_without_requiring_json_wrapper(self) -> None:
        config = parse_survive_config(NATIVE_CONFIG)
        self.assertEqual(config["v"], "0")
        self.assertEqual(config["poser"], "MPFIT")
        self.assertEqual(config["floor-offset"], "-0.481664836407")
        self.assertEqual(config["lighthouse0"]["id"], "1234567890")
        self.assertEqual(config["lighthouse1"]["id"], "987654321")
        self.assertEqual(len(config["lighthouse1"]["pose"]), 7)

    def test_accepts_standard_json_object(self) -> None:
        self.assertEqual(
            parse_survive_config(
                ' {"v": "0", "lighthouse0": {"index": "0", "pose": [1, 2]}}\n'
            ),
            {"v": "0", "lighthouse0": {"index": "0", "pose": [1, 2]}},
        )
        self.assertEqual(parse_survive_config("{}\n"), {})

    def test_accepts_native_globals_without_lighthouses(self) -> None:
        self.assertEqual(
            parse_survive_config('"v":"0",\n"poser":"MPFIT"\n'),
            {"v": "0", "poser": "MPFIT"},
        )

    def test_accepts_native_lighthouse_groups_without_globals(self) -> None:
        self.assertEqual(
            parse_survive_config(
                '"lighthouse0":{"id":"1"}\n"lighthouse2":{"id":"3"}\n'
            ),
            {"lighthouse0": {"id": "1"}, "lighthouse2": {"id": "3"}},
        )

    def test_preserves_escaped_strings(self) -> None:
        self.assertEqual(
            parse_survive_config(r'"label":"a \"quoted\" value"'),
            {"label": 'a "quoted" value'},
        )

    def test_rejects_empty_truncated_or_non_object_inputs(self) -> None:
        for text in (
            "",
            " \n\t",
            "[]",
            '"scalar"',
            "42",
            "null",
            '{"v":',
            '"v":',
            '"v":"unfinished',
            NATIVE_CONFIG[:-3],
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_survive_config(text)

    def test_rejects_malformed_nested_json_instead_of_repairing_it(self) -> None:
        for text in (
            '"lighthouse0":{"index":"0" "id":"1"}',
            '"lighthouse0":{"pose":["0" "1"]}',
            '"lighthouse0":{"pose":["0",]}',
            '"lighthouse0":{"index":"0",}',
            '"v":"0"\n"poser":"MPFIT"',
            '"v":"0"\n"lighthouse0":"not an object"',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_survive_config(text)

    def test_rejects_trailing_garbage_and_extra_documents(self) -> None:
        for text in (
            NATIVE_CONFIG + "garbage",
            NATIVE_CONFIG + "{}",
            '{"v":"0"} {"v":"1"}',
            '"v":"0",',
            '"v":"0",,"poser":"MPFIT"',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_survive_config(text)

    def test_rejects_duplicate_keys(self) -> None:
        for text in (
            '"v":"0",\n"v":"1"',
            '"lighthouse0":{"id":"1"}\n"lighthouse0":{"id":"2"}',
            '{"v":"0", "v":"1"}',
            '"lighthouse0":{"id":"1", "id":"2"}',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_survive_config(text)


@pytest.mark.parametrize("saved", [None, b"", b"  \n", b"{}\n", b"{ }"])
def test_native_seed_never_hands_survive_cli_an_empty_object(saved) -> None:
    # survive-cli 1.0 (f1e6edd) segfaults on a configuration holding only "{}";
    # an empty file calibrates from scratch like a missing one (seen on an
    # AGX Orin with four base stations).
    assert native_seed(saved) == b""


def test_native_seed_keeps_a_saved_calibration_verbatim() -> None:
    saved = b'"v":"0",\n"lighthouse0":{"index":"0"}\n'
    assert native_seed(saved) is saved
