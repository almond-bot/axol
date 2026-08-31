from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli.collect_data import CollectDataConfig
from almond_axol.cli.config import TeleopCmdConfig, parse
from almond_axol.cli.mantis_bridge import (
    add_quest_key_to_direct_fallback,
    load_direct_mantis_fallback,
    managed_mantis_bridge,
)
from almond_axol.constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT
from almond_axol.serve.app import _mantis_channel_mismatch_message
from almond_axol.serve.runner import (
    _bind_managed_mantis_trigger_channels,
    _managed_mantis_run_channels,
)
from almond_axol.serve.settings import SettingsStore
from almond_axol.utils.can_channels import require_mantis_channels


class MantisChannelFlowTest(unittest.TestCase):
    def test_direct_fallback_matches_saved_rig_source_channels_and_quest_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "teleop.mantis_source": "quest",
                    "mantis.left_channel": "can_mantis_r",
                    "mantis.right_channel": "can_mantis_l",
                    "mantis.quest_tracker_key": "quest:oculus-touch-v3:grip",
                }
            )
            with patch("almond_axol.serve.settings.SettingsStore", return_value=store):
                teleop_fallback, quest_key = load_direct_mantis_fallback(
                    collection=False
                )
                collect_fallback, collect_key = load_direct_mantis_fallback(
                    collection=True
                )

        self.assertEqual(teleop_fallback["mantis_source"], "quest")
        self.assertEqual(teleop_fallback["left_channel"], "can_mantis_r")
        self.assertEqual(teleop_fallback["right_channel"], "can_mantis_l")
        self.assertEqual(
            collect_fallback["robot_config"],
            {"left_channel": "can_mantis_r", "right_channel": "can_mantis_l"},
        )
        self.assertEqual(quest_key, "quest:oculus-touch-v3:grip")
        self.assertEqual(collect_key, quest_key)

    def test_direct_fallback_is_below_config_file_and_cli(self) -> None:
        fallback: dict[str, object] = {
            "mantis_source": "quest",
            "left_channel": "saved-left",
            "right_channel": "saved-right",
        }
        add_quest_key_to_direct_fallback(
            fallback, "quest:saved-profile:grip", collection=False
        )
        saved = parse(
            TeleopCmdConfig,
            ["--mantis", "true"],
            fallback_overlay=fallback,
        )
        self.assertEqual(saved.mantis_source, "quest")
        self.assertEqual(
            (saved.left_channel, saved.right_channel), ("saved-left", "saved-right")
        )
        self.assertEqual(saved.teleop.tracker_key, "quest:saved-profile:grip")

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "teleop.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mantis": True,
                        "mantis_source": "ultimate",
                        "left_channel": "file-left",
                        "right_channel": "file-right",
                        "teleop": {"tracker_key": "file-key"},
                    }
                )
            )
            overridden = parse(
                TeleopCmdConfig,
                [
                    "--config_path",
                    str(config_path),
                    "--left_channel",
                    "cli-left",
                ],
                fallback_overlay=fallback,
            )
        self.assertEqual(overridden.mantis_source, "ultimate")
        self.assertEqual(
            (overridden.left_channel, overridden.right_channel),
            ("cli-left", "file-right"),
        )
        self.assertEqual(overridden.teleop.tracker_key, "file-key")

    def test_nested_collection_fallback_has_the_same_precedence(self) -> None:
        fallback: dict[str, object] = {
            "mantis_source": "quest",
            "robot_config": {
                "left_channel": "saved-left",
                "right_channel": "saved-right",
            },
        }
        add_quest_key_to_direct_fallback(
            fallback, "quest:saved-profile:grip", collection=True
        )
        cfg = parse(
            CollectDataConfig,
            [
                "--repo_id",
                "test/direct-fallback",
                "--task",
                "test",
                "--mantis",
                "true",
                "--robot_config.left_channel",
                "cli-left",
                "--teleop_config.vr_teleop_config.tracker_key",
                "quest:cli-profile:grip",
            ],
            fallback_overlay=fallback,
        )
        self.assertEqual(cfg.mantis_source, "quest")
        self.assertEqual(
            (cfg.robot_config.left_channel, cfg.robot_config.right_channel),
            ("cli-left", "saved-right"),
        )
        self.assertEqual(
            cfg.teleop_config.vr_teleop_config.tracker_key,
            "quest:cli-profile:grip",
        )

    def test_quest_key_is_applied_only_to_quest_mantis_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            key = "quest:meta-quest-touch-plus:grip"
            store.update(values={"mantis.quest_tracker_key": key})

            quest = store.merged_args(
                "teleop", {"mantis": True, "mantis_source": "quest"}
            )
            self.assertEqual(quest["teleop.tracker_key"], key)
            lighthouse = store.merged_args(
                "teleop", {"mantis": True, "mantis_source": "lighthouse"}
            )
            self.assertNotIn("teleop.tracker_key", lighthouse)
            axol = store.merged_args("teleop", {"mantis": False})
            self.assertNotIn("teleop.tracker_key", axol)

    def test_legacy_advanced_quest_key_migrates_to_visible_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            key = "quest:meta-quest-touch-plus:grip"
            path.write_text(
                json.dumps(
                    {
                        "values": {},
                        "advanced": {"vr_teleop.tracker_key": key},
                    }
                )
            )
            snapshot = SettingsStore(path).snapshot()
            self.assertEqual(snapshot["values"]["mantis.quest_tracker_key"], key)
            self.assertNotIn("vr_teleop.tracker_key", snapshot["advanced"])

    def test_mantis_channels_must_be_two_nonempty_distinct_names(self) -> None:
        self.assertEqual(
            require_mantis_channels((" can_left ", "can_right")),
            ("can_left", "can_right"),
        )
        with self.assertRaisesRegex(ValueError, "left CAN channel is empty"):
            require_mantis_channels((None, "can_right"))
        with self.assertRaisesRegex(ValueError, "distinct interfaces"):
            require_mantis_channels(("can_same", "can_same"))

    def test_settings_reject_invalid_mantis_mapping_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            with self.assertRaisesRegex(ValueError, "left CAN channel is empty"):
                store.update(values={"mantis.left_channel": " "})
            with self.assertRaisesRegex(ValueError, "distinct interfaces"):
                store.update(
                    values={
                        "mantis.left_channel": "can_same",
                        "mantis.right_channel": "can_same",
                    }
                )
            self.assertEqual(store.snapshot()["values"], {})

    def test_effective_channels_include_stored_mapping_and_run_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(
                values={
                    "mantis.left_channel": "can_hub_b",
                    "mantis.right_channel": "can_hub_a",
                }
            )

            self.assertEqual(
                store.effective_mantis_can_channels("teleop", {"mantis": True}),
                ("can_hub_b", "can_hub_a"),
            )
            self.assertEqual(
                store.effective_mantis_can_channels(
                    "teleop",
                    {
                        "mantis": True,
                        "left_channel": "can_override",
                        "right_channel": "null",
                    },
                ),
                ("can_override", None),
            )
            self.assertEqual(
                store.effective_mantis_can_channels(
                    "collect-data",
                    {
                        "mantis": True,
                        "robot_config.left_channel": CAN_LEFT,
                        "robot_config.right_channel": CAN_RIGHT,
                    },
                ),
                (CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT),
            )

    def test_direct_managed_bridge_ignores_standalone_trigger_overrides(self) -> None:
        config = SimpleNamespace(
            backend="survive",
            left="tracker-left",
            right="tracker-right",
            allow_single_side=False,
            trigger_can_left="stale-trigger-left",
            trigger_can_right="stale-trigger-right",
        )
        captured: list[SimpleNamespace] = []

        def run_bridge(selected, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(selected)
            kwargs["on_ready"]()
            kwargs["controls"].quit.wait(1.0)

        with (
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch("almond_axol.tracker.config.select_tracker_backend"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.cli.tracker_bridge.run_configured_bridge",
                side_effect=run_bridge,
            ),
        ):
            with managed_mantis_bridge(
                "lighthouse",
                left_channel="can_run_left",
                right_channel="can_run_right",
                port=8000,
                pose_source_id="managed-test",
            ):
                pass

        self.assertEqual(len(captured), 1)
        self.assertEqual(config.trigger_can_left, "can_run_left")
        self.assertEqual(config.trigger_can_right, "can_run_right")

        with self.assertRaisesRegex(ValueError, "distinct interfaces"):
            with managed_mantis_bridge(
                "quest",
                left_channel="can_same",
                right_channel="can_same",
                port=8000,
            ):
                pass

    def test_serve_managed_bridge_uses_effective_run_channels(self) -> None:
        tracker = SimpleNamespace(
            trigger_can_left="stale-trigger-left",
            trigger_can_right="stale-trigger-right",
        )
        collect_config = SimpleNamespace(
            robot_config=SimpleNamespace(
                left_channel="can_run_right", right_channel="can_run_left"
            )
        )
        _bind_managed_mantis_trigger_channels(tracker, collect_config)
        self.assertEqual(tracker.trigger_can_left, "can_run_right")
        self.assertEqual(tracker.trigger_can_right, "can_run_left")

        teleop_config = SimpleNamespace(
            left_channel=CAN_LEFT,
            right_channel=CAN_RIGHT,
        )
        _bind_managed_mantis_trigger_channels(tracker, teleop_config)
        self.assertEqual(tracker.trigger_can_left, CAN_MANTIS_LEFT)
        self.assertEqual(tracker.trigger_can_right, CAN_MANTIS_RIGHT)

        with self.assertRaisesRegex(ValueError, "distinct interfaces"):
            _managed_mantis_run_channels(
                SimpleNamespace(left_channel="can_same", right_channel="can_same")
            )

    def test_preflight_rejects_connected_link_on_old_channel_map(self) -> None:
        self.assertIsNone(
            _mantis_channel_mismatch_message(
                ("can_new_left", "can_new_right"),
                ("can_new_left", "can_new_right"),
            )
        )
        message = _mantis_channel_mismatch_message(
            ("can_old_left", "can_old_right"),
            ("can_new_left", "can_new_right"),
        )
        self.assertIsNotNone(message)
        self.assertIn("changed after this link connected", message or "")
        self.assertIn("Disconnect and reconnect Mantis", message or "")


if __name__ == "__main__":
    unittest.main()
