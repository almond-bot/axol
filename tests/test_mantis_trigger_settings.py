"""Mantis trigger endpoints stay per-side across settings and bridge config."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from almond_axol.serve.settings import SettingsStore, settings_schema
from almond_axol.tracker.config import (
    TrackerConfig,
    load_tracker_config,
    save_tracker_config,
)


class MantisTriggerSettingsTest(unittest.TestCase):
    def test_panel_fields_reach_teleop_and_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(path)
            store.update(values={"mantis.left_trigger_full_scale": 0.90})

            teleop = store.merged_args("teleop", {"mantis": True})
            collect = store.merged_args("collect-data", {"mantis": True})
            self.assertEqual(teleop["mantis_left_trigger_full_scale"], 0.90)
            self.assertEqual(collect["mantis_left_trigger_full_scale"], 0.90)
            self.assertNotIn("mantis_right_trigger_full_scale", teleop)
            self.assertNotIn("mantis_right_trigger_full_scale", collect)

            # A per-run override remains possible, and the saved value survives
            # reopening the store for a subsequent operation.
            override = store.merged_args(
                "teleop", {"mantis": True, "mantis_left_trigger_full_scale": 0.93}
            )
            self.assertEqual(override["mantis_left_trigger_full_scale"], 0.93)
            self.assertEqual(
                SettingsStore(path).snapshot()["values"][
                    "mantis.left_trigger_full_scale"
                ],
                0.90,
            )

        fields = {
            setting["key"]: setting
            for category in settings_schema()
            for setting in category["settings"]
        }
        self.assertEqual(fields["mantis.left_trigger_full_scale"]["default"], 1.0)
        self.assertEqual(fields["mantis.right_trigger_full_scale"]["default"], 1.0)

    def test_invalid_endpoint_is_rejected_before_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = SettingsStore(path)
            for value in (0, -0.1, 0.49, 1.01, math.nan, math.inf, True, "0.90"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "left_trigger_full_scale"):
                        store.update(values={"mantis.left_trigger_full_scale": value})
                    self.assertNotIn(
                        "mantis.left_trigger_full_scale", store.snapshot()["values"]
                    )

    def test_standalone_bridge_config_persists_both_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.json"
            config = TrackerConfig(
                trigger_full_scale_left=0.90,
                trigger_full_scale_right=1.0,
            )
            save_tracker_config(config, path)
            restored = load_tracker_config(path)
            self.assertEqual(restored.trigger_full_scale_left, 0.90)
            self.assertEqual(restored.trigger_full_scale_right, 1.0)


if __name__ == "__main__":
    unittest.main()
