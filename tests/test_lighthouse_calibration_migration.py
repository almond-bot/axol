"""Discovery aliases must never hide a custom mount when binding hardware serials."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.collect_data import _validate_mantis_calibration
from almond_axol.mantis.calibration import INVALID_TRANSFORM_ENTRY
from almond_axol.serve import tracker_setup
from almond_axol.teleop.config import VRTeleopConfig, apply_mantis_teleop_profile
from almond_axol.tracker.config import TrackerConfig, save_tracker_config

_CUSTOM = [0.12, 0.02, 0.03, 0.0, 0.0, 0.0, 1.0]
_ENTRY = {"pos": _CUSTOM[:3], "quat": _CUSTOM[3:]}
_KEYS = {"left": "survive:LHR-LEFT", "right": "survive:LHR-RIGHT"}


class LighthouseCalibrationMigrationTest(unittest.TestCase):
    def test_alias_override_blocks_factory_until_exact_serial_is_measured(self) -> None:
        for alias in ("survive:WM0", "survive:T20"):
            for exact in (False, True):
                with self.subTest(alias=alias, exact=exact):
                    saved = {"left": {alias: list(_CUSTOM)}}
                    if exact:
                        saved["left"][_KEYS["left"]] = list(_CUSTOM)
                    vr = VRTeleopConfig()
                    cfg = SimpleNamespace(
                        mantis_allow_uncalibrated=False,
                        mantis_source="lighthouse",
                        teleop_config=SimpleNamespace(vr_teleop_config=vr),
                    )
                    with (
                        patch(
                            "almond_axol.mantis.calibration.load_tcp_transforms",
                            return_value=saved,
                        ),
                        patch(
                            "almond_axol.mantis.calibration.tracker_key_for_side",
                            side_effect=lambda side, **_: (
                                _KEYS[side],
                                "serial binding",
                            ),
                        ),
                    ):
                        apply_mantis_teleop_profile(vr, tracker_source="lighthouse")
                        if exact:
                            self.assertEqual(vr.tcp_transform_left, _CUSTOM)
                            _validate_mantis_calibration(cfg)
                        else:
                            self.assertIsNone(vr.tcp_transform_left)
                            with self.assertRaisesRegex(
                                ValueError, "no verified.*LHR-LEFT"
                            ):
                                _validate_mantis_calibration(cfg)
                    self.assertIn(alias, saved["left"])

    def test_invalid_alias_also_blocks_factory_fallback(self) -> None:
        def saved(**kwargs):
            statuses = kwargs.get("entry_statuses")
            if statuses is not None:
                statuses[("left", "survive:WM0")] = INVALID_TRANSFORM_ENTRY
            return {}

        vr = VRTeleopConfig()
        cfg = SimpleNamespace(
            mantis_allow_uncalibrated=False,
            mantis_source="lighthouse",
            teleop_config=SimpleNamespace(vr_teleop_config=vr),
        )
        with (
            patch(
                "almond_axol.mantis.calibration.load_tcp_transforms", side_effect=saved
            ),
            patch(
                "almond_axol.mantis.calibration.tracker_key_for_side",
                side_effect=lambda side, **_: (_KEYS[side], "serial binding"),
            ),
        ):
            apply_mantis_teleop_profile(vr, tracker_source="lighthouse")
            self.assertIsNone(vr.tcp_transform_left)
            with self.assertRaisesRegex(ValueError, "no verified"):
                _validate_mantis_calibration(cfg)

    def test_api_exposes_old_alias_for_review_and_accepts_explicit_serial_mapping(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "tracker.json"
            transform_path = root / "tcp.json"
            save_tracker_config(
                TrackerConfig(left="LHR-LEFT", right="LHR-RIGHT"), config_path
            )
            original = {"left": {"survive:WM0": dict(_ENTRY)}}
            transform_path.write_text(json.dumps(original))
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                snapshot = tracker_setup.calibration_snapshot(
                    "lighthouse", path=transform_path
                )
                self.assertEqual(snapshot["left"]["status"], "missing")
                self.assertIn("survive:WM0", snapshot["left"]["overrideKeys"])
                self.assertEqual(snapshot["right"]["status"], "factory")
                self.assertEqual(json.loads(transform_path.read_text()), original)
                original["left"][_KEYS["left"]] = dict(_ENTRY)
                transform_path.write_text(json.dumps(original))
                snapshot = tracker_setup.calibration_snapshot(
                    "lighthouse", path=transform_path
                )
                self.assertEqual(snapshot["left"]["status"], "measured")
                self.assertEqual(snapshot["left"]["pos"], _CUSTOM[:3])
                self.assertEqual(json.loads(transform_path.read_text()), original)


if __name__ == "__main__":
    unittest.main()
