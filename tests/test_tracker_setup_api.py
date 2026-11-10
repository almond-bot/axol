from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from almond_axol.serve import app as app_module
from almond_axol.serve import tracker_setup
from almond_axol.serve.commands import COMMANDS, get_schema
from almond_axol.tracker.config import (
    TrackerConfig,
    load_tracker_config,
    save_tracker_config,
)
from tests.safety_isolation import setUpModule, tearDownModule  # noqa: F401

_IDENTITY = {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}


def _keyed(key: str, entry: dict[str, list[float]] = _IDENTITY) -> dict[str, Any]:
    return {"key": key, **entry}


class TrackerSetupPersistenceTest(unittest.TestCase):
    def test_atomic_writer_preserves_existing_operator_owner_when_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            path.write_text("{}")
            with patch.object(
                tracker_setup, "secure_atomic_write_json"
            ) as secure_write:
                tracker_setup._atomic_write_json(path, {"ok": True})  # noqa: SLF001
            secure_write.assert_called_once_with(path, {"ok": True})

    def _tracker_config(self, path: Path) -> None:
        save_tracker_config(
            TrackerConfig(
                backend="survive",
                left="LHR-LEFT",
                right="LHR-RIGHT",
                bindings={
                    "survive": {"left": "LHR-LEFT", "right": "LHR-RIGHT"},
                },
            ),
            path,
        )

    def test_calibration_partial_update_preserves_unrelated_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            original = {
                "metadata": {"fixture": "bench-a"},
                "left": {"survive:OTHER": dict(_IDENTITY)},
                "right": {"quest:meta-quest-touch-plus:grip": dict(_IDENTITY)},
            }
            calibration_path.write_text(json.dumps(original))
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                result = tracker_setup.save_calibration(
                    "lighthouse",
                    {
                        "left": {
                            "key": "survive:LHR-LEFT",
                            "pos": [0.1, 0.2, 0.3],
                            "quat": [0.0, 0.0, 0.0, 2.0],
                        }
                    },
                    path=calibration_path,
                )

            self.assertEqual(result["keys"]["left"], "survive:LHR-LEFT")
            self.assertEqual(result["left"]["status"], "measured")
            self.assertEqual(result["left"]["quat"], [0.0, 0.0, 0.0, 1.0])
            # The factory value applies without appearing as measured editor
            # data for the untouched side.
            self.assertEqual(result["right"]["status"], "factory")
            self.assertIsNone(result["right"]["pos"])
            self.assertIsNone(result["right"]["quat"])
            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["metadata"], original["metadata"])
            self.assertEqual(
                saved["left"]["survive:OTHER"],
                original["left"]["survive:OTHER"],
            )
            self.assertEqual(saved["right"], original["right"])
            self.assertEqual(calibration_path.stat().st_mode & 0o777, 0o600)

    def test_calibration_remove_preserves_unrelated_entries_and_restores_factory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            active_key = "survive:LHR-LEFT"
            quest_key = "quest:meta-quest-touch-plus:grip"
            original = {
                "metadata": {"fixture": "bench-a", "revision": 4},
                "left": {
                    active_key: dict(_IDENTITY),
                    quest_key: {
                        "pos": [0.1, 0.2, 0.3],
                        "quat": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                "right": {"quest:meta-quest-touch-plus:grip": dict(_IDENTITY)},
            }
            calibration_path.write_text(json.dumps(original))

            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                before = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
                result = tracker_setup.remove_calibration(
                    "lighthouse",
                    "left",
                    active_key,
                    active_key,
                    before["left"]["overrideRevisions"][active_key],
                    path=calibration_path,
                )

            self.assertEqual(result["left"]["status"], "factory")
            self.assertIsNone(result["left"]["pos"])
            self.assertIsNone(result["left"]["quat"])
            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["metadata"], original["metadata"])
            self.assertNotIn(active_key, saved["left"])
            self.assertEqual(saved["left"][quest_key], original["left"][quest_key])
            self.assertEqual(saved["right"], original["right"])
            self.assertEqual(calibration_path.stat().st_mode & 0o777, 0o600)

    def test_calibration_remove_rejects_stale_identity_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            original = {
                "left": {"survive:LHR-LEFT": dict(_IDENTITY)},
                "right": {"survive:LHR-RIGHT": dict(_IDENTITY)},
            }
            calibration_path.write_text(json.dumps(original))

            with (
                patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path),
                self.assertRaisesRegex(
                    tracker_setup.TrackerSetupError,
                    "identity changed.*refresh",
                ),
            ):
                tracker_setup.remove_calibration(
                    "lighthouse",
                    "left",
                    "survive:LHR-LEFT",
                    "survive:OLD-TRACKER",
                    "identity-check-runs-first",
                    path=calibration_path,
                )

            self.assertEqual(json.loads(calibration_path.read_text()), original)

    def test_calibration_remove_exposes_malformed_active_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            active_key = "survive:LHR-LEFT"
            quest_key = "quest:meta-quest-touch-plus:grip"
            calibration_path.write_text(
                json.dumps(
                    {
                        "left": {
                            active_key: ["malformed"],
                            quest_key: dict(_IDENTITY),
                        }
                    }
                )
            )

            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                before = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
                self.assertEqual(before["left"]["status"], "missing")
                self.assertEqual(before["left"]["overrideKeys"], [active_key])
                after = tracker_setup.remove_calibration(
                    "lighthouse",
                    "left",
                    active_key,
                    active_key,
                    before["left"]["overrideRevisions"][active_key],
                    path=calibration_path,
                )

            self.assertEqual(after["left"]["status"], "factory")
            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["left"], {quest_key: dict(_IDENTITY)})

    def test_calibration_remove_cleans_only_selected_legacy_and_family_blockers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            active_key = "survive:LHR-LEFT"
            old_key = "survive:OLD-TRACKER"
            quest_key = "quest:meta-quest-touch-plus:grip"
            original = {
                "metadata": {"fixture": "bench-b"},
                "left": {
                    **_IDENTITY,
                    old_key: dict(_IDENTITY),
                    quest_key: dict(_IDENTITY),
                },
                "right": {"quest:meta-quest-touch-plus:grip": dict(_IDENTITY)},
            }
            calibration_path.write_text(json.dumps(original))

            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                before = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
                self.assertEqual(
                    before["left"]["overrideKeys"],
                    ["legacy", old_key],
                )
                with self.assertRaisesRegex(
                    tracker_setup.TrackerSetupError,
                    "selected override no longer applies",
                ):
                    tracker_setup.remove_calibration(
                        "lighthouse",
                        "left",
                        quest_key,
                        active_key,
                        "not-a-removable-entry",
                        path=calibration_path,
                    )
                after_old = tracker_setup.remove_calibration(
                    "lighthouse",
                    "left",
                    old_key,
                    active_key,
                    before["left"]["overrideRevisions"][old_key],
                    path=calibration_path,
                )
                self.assertEqual(after_old["left"]["overrideKeys"], ["legacy"])
                after_legacy = tracker_setup.remove_calibration(
                    "lighthouse",
                    "left",
                    "legacy",
                    active_key,
                    after_old["left"]["overrideRevisions"]["legacy"],
                    path=calibration_path,
                )

            self.assertEqual(after_legacy["left"]["status"], "factory")
            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["metadata"], original["metadata"])
            self.assertEqual(saved["left"], {quest_key: dict(_IDENTITY)})
            self.assertEqual(saved["right"], original["right"])

    def test_calibration_remove_rejects_same_key_concurrent_resave(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            active_key = "survive:LHR-LEFT"
            calibration_path.write_text(
                json.dumps({"left": {active_key: dict(_IDENTITY)}})
            )

            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                before = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
                old_revision = before["left"]["overrideRevisions"][active_key]
                tracker_setup.save_calibration(
                    "lighthouse",
                    {
                        "left": {
                            "key": active_key,
                            "pos": [0.1, 0.2, 0.3],
                            "quat": [0.0, 0.0, 0.0, 1.0],
                        }
                    },
                    path=calibration_path,
                )
                with self.assertRaisesRegex(
                    tracker_setup.TrackerSetupError,
                    "selected override changed.*refresh",
                ):
                    tracker_setup.remove_calibration(
                        "lighthouse",
                        "left",
                        active_key,
                        active_key,
                        old_revision,
                        path=calibration_path,
                    )

            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["left"][active_key]["pos"], [0.1, 0.2, 0.3])

    def test_calibration_resolves_lighthouse_binding_and_quest_configured_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                lighthouse = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
                self.assertEqual(
                    lighthouse["keys"],
                    {"left": "survive:LHR-LEFT", "right": "survive:LHR-RIGHT"},
                )
                for side in ("left", "right"):
                    self.assertEqual(lighthouse[side]["status"], "factory")
                    self.assertIsNone(lighthouse[side]["pos"])
                    self.assertIsNone(lighthouse[side]["quat"])

                with self.assertRaisesRegex(
                    tracker_setup.TrackerSetupError, "quest_tracker_key"
                ):
                    tracker_setup.save_calibration(
                        "quest",
                        {"left": _keyed("quest:test:grip")},
                        path=calibration_path,
                    )
                quest_key = "quest:meta-quest-touch-plus:grip"
                quest = tracker_setup.save_calibration(
                    "quest",
                    {"right": _keyed(quest_key)},
                    quest_tracker_key=quest_key,
                    path=calibration_path,
                )
                self.assertEqual(quest["right"]["key"], quest_key)
                self.assertEqual(quest["right"]["status"], "measured")

    def test_malformed_exact_override_suppresses_factory_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            calibration_path.write_text(
                json.dumps(
                    {
                        "left": {"survive:LHR-LEFT": []},
                        "right": {"survive:LHR-RIGHT": "invalid"},
                    }
                )
            )
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                snapshot = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
            self.assertEqual(snapshot["left"]["status"], "missing")
            self.assertEqual(snapshot["right"]["status"], "missing")

            mixed = dict(_IDENTITY)
            mixed["survive:LHR-LEFT"] = dict(_IDENTITY)
            calibration_path.write_text(
                json.dumps(
                    {
                        "left": mixed,
                        "right": dict(_IDENTITY),
                    }
                )
            )
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                snapshot = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
            self.assertEqual(snapshot["left"]["status"], "missing")
            self.assertEqual(snapshot["right"]["status"], "missing")

            for override_key in ("survive", "survive:previous-device"):
                with self.subTest(override_key=override_key):
                    calibration_path.write_text(
                        json.dumps(
                            {
                                "left": {override_key: dict(_IDENTITY)},
                                "right": {override_key: dict(_IDENTITY)},
                            }
                        )
                    )
                    with patch.object(
                        tracker_setup, "TRACKER_CONFIG_FILE", config_path
                    ):
                        snapshot = tracker_setup.calibration_snapshot(
                            "lighthouse", path=calibration_path
                        )
                    self.assertEqual(snapshot["left"]["status"], "missing")
                    self.assertEqual(snapshot["right"]["status"], "missing")

            calibration_path.write_text(
                json.dumps(
                    {
                        "left": dict(_IDENTITY),
                        "right": dict(_IDENTITY),
                    }
                )
            )
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                snapshot = tracker_setup.calibration_snapshot(
                    "lighthouse", path=calibration_path
                )
            self.assertEqual(snapshot["left"]["status"], "missing")
            self.assertEqual(snapshot["right"]["status"], "missing")

    def test_calibration_rejects_nonfinite_and_zero_quaternion(self) -> None:
        for entry in (
            {"pos": [math.inf, 0, 0], "quat": [0, 0, 0, 1]},
            {"pos": [0, 0, 0], "quat": [0, 0, 0, 0]},
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(tracker_setup.TrackerSetupError):
                    tracker_setup.save_calibration(
                        "quest",
                        {"left": _keyed("quest:test:grip", entry)},
                        quest_tracker_key="quest:test:grip",
                        path=Path("unused"),
                    )

    def test_calibration_rejects_stale_identity_from_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            with self.assertRaisesRegex(
                tracker_setup.TrackerSetupError,
                "identity changed.*refresh",
            ):
                tracker_setup.save_calibration(
                    "quest",
                    {"left": _keyed("quest:old-controller:grip")},
                    quest_tracker_key="quest:new-controller:grip",
                    path=path,
                )
            self.assertFalse(path.exists())

    def test_calibration_refuses_to_erase_malformed_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            malformed = "{ definitely not json"
            path.write_text(malformed)
            with self.assertRaisesRegex(
                tracker_setup.TrackerSetupError, "fix it before saving"
            ):
                tracker_setup.save_calibration(
                    "quest",
                    {"left": _keyed("quest:test:grip")},
                    quest_tracker_key="quest:test:grip",
                    path=path,
                )
            self.assertEqual(path.read_text(), malformed)


class _Settings:
    def __init__(self, quest_key: str | None = None) -> None:
        self.quest_key = quest_key

    def can_channels(self) -> tuple[str, str]:
        return "can-left", "can-right"

    def mantis_can_channels(self) -> tuple[str, str]:
        return "can-mantis-left", "can-mantis-right"

    def has_gripper(self) -> bool:
        return True

    def snapshot(self) -> dict[str, Any]:
        values = (
            {"mantis.quest_tracker_key": self.quest_key}
            if self.quest_key is not None
            else {}
        )
        return {"values": values, "cameras": None, "advanced": {}}


def _test_app(settings: _Settings) -> Any:
    manager = MagicMock()
    manager.list.return_value = []
    runner = MagicMock()
    runner.is_running.return_value = False
    with (
        patch.object(app_module, "SessionManager", return_value=manager),
        patch.object(app_module, "OperationRunner", return_value=runner),
        patch.object(app_module, "SettingsStore", return_value=settings),
        patch.object(app_module, "RobotLink", return_value=MagicMock()),
    ):
        return app_module.create_app()


class TrackerSetupApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_lighthouse_quality_api_saves_valid_modes_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            save_tracker_config(TrackerConfig(left="LHR-L", right="LHR-R"), path)

            def load(*_args: Any) -> TrackerConfig:
                return load_tracker_config(path)

            def save(config: TrackerConfig, *_args: Any) -> None:
                save_tracker_config(config, path)

            app = _test_app(_Settings())
            transport = httpx.ASGITransport(app=app)
            with (
                patch("almond_axol.tracker.config.load_tracker_config", load),
                patch("almond_axol.tracker.config.save_tracker_config", save),
            ):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    rejected = await client.put(
                        "/api/tracker/lighthouse/quality", json={"mode": "lax"}
                    )
                    saved = await client.put(
                        "/api/tracker/lighthouse/quality", json={"mode": "strict"}
                    )
            self.assertEqual(rejected.status_code, 400)
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json(), {"mode": "strict"})
            data = json.loads(path.read_text())
            self.assertEqual(data["lighthouse_quality"], "strict")
            # The rest of the tracker setup is untouched.
            self.assertEqual((data["left"], data["right"]), ("LHR-L", "LHR-R"))

    async def test_calibration_api_supports_partial_quest_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tcp.json"
            quest_key = "quest:meta-quest-touch-plus:grip"
            app = _test_app(_Settings(quest_key))
            transport = httpx.ASGITransport(app=app)
            with patch.object(tracker_setup, "MANTIS_TCP_TRANSFORM_FILE", path):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.put(
                        "/api/tracker/calibration/quest",
                        json={"left": _keyed(quest_key)},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    value = response.json()
                    self.assertEqual(value["left"]["key"], quest_key)
                    self.assertEqual(value["left"]["status"], "measured")
                    # Quest 3 Touch Plus ships a factory transform, so the side
                    # without an override falls back to it (factory values
                    # are not echoed as editable pos/quat).
                    self.assertEqual(value["right"]["status"], "factory")
                    self.assertIsNone(value["right"]["pos"])
                    response = await client.get("/api/tracker/calibration/quest")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["left"]["pos"], [0.0, 0.0, 0.0])

                    response = await client.delete(
                        "/api/tracker/calibration/quest/left",
                        params={
                            "key": quest_key,
                            "active_key": quest_key,
                            "revision": value["left"]["overrideRevisions"][quest_key],
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["left"]["status"], "factory")
                    self.assertIsNone(response.json()["left"]["pos"])
                    self.assertNotIn("left", json.loads(path.read_text()))

    async def test_calibration_api_rejects_unknown_source(self) -> None:
        app = _test_app(_Settings())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/tracker/calibration/unknown")
        self.assertEqual(response.status_code, 400)

    def test_lighthouse_check_is_a_runnable_serve_command(self) -> None:
        self.assertIn("tracker.lighthouse.check", COMMANDS)
        schema = get_schema("tracker.lighthouse.check")
        self.assertEqual(len(schema.nodes), 1)
        self.assertEqual(schema.nodes[0]["key"], "expected_stations")
        self.assertEqual(schema.nodes[0]["options"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
