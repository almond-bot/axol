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
from almond_axol.mantis.calibration import ULTIMATE_POSE_CONVENTION_FIELD
from almond_axol.tracker.config import TrackerConfig, save_tracker_config
from almond_axol.tracker.ultimate import ultimate_wifi_config_error

_IDENTITY = {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}


def _keyed(key: str, entry: dict[str, list[float]] = _IDENTITY) -> dict[str, Any]:
    return {"key": key, **entry}


class TrackerSetupPersistenceTest(unittest.TestCase):
    def test_wifi_first_save_is_restrictive_and_never_returns_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tracker" / "ultimate_wifi.json"
            with self.assertRaisesRegex(tracker_setup.TrackerSetupError, "first save"):
                tracker_setup.save_ultimate_wifi(
                    {"ssid": "AXOL", "country": "US", "freq": 5240}, path
                )

            secret = "not-for-an-api-response"
            result = tracker_setup.save_ultimate_wifi(
                {
                    "ssid": "AXOL",
                    "pass": secret,
                    "country": "us",
                    "freq": 5240,
                },
                path,
            )
            self.assertEqual(result["status"], "valid")
            self.assertTrue(result["configured"])
            self.assertTrue(result["passwordSet"])
            self.assertNotIn("pass", result)
            self.assertNotIn(secret, json.dumps(result))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["pass"], secret)
            self.assertEqual(json.loads(path.read_text())["country"], "US")

    def test_wifi_omitted_password_preserves_existing_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            tracker_setup.save_ultimate_wifi(
                {
                    "ssid": "old",
                    "pass": "keep-me!",
                    "country": "US",
                    "freq": 5180,
                },
                path,
            )
            result = tracker_setup.save_ultimate_wifi(
                {"ssid": "new", "country": "CA", "freq": 5200}, path
            )
            saved = json.loads(path.read_text())
            self.assertEqual(saved["pass"], "keep-me!")
            self.assertEqual(saved["ssid"], "new")
            self.assertNotIn("keep-me!", json.dumps(result))

    def test_wifi_rejects_unknown_fields_without_echoing_secret(self) -> None:
        secret = "do-not-echo-this"
        with self.assertRaises(tracker_setup.TrackerSetupError) as caught:
            tracker_setup.save_ultimate_wifi(
                {
                    "ssid": "AXOL",
                    "pass": secret,
                    "country": "US",
                    "freq": 5240,
                    "unexpected": "value",
                },
                Path("unused"),
            )
        self.assertNotIn(secret, str(caught.exception))

    def test_wifi_snapshot_reports_insecure_mode_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            path.write_text(
                json.dumps(
                    {
                        "ssid": "AXOL",
                        "pass": "hidden!!",
                        "country": "US",
                        "freq": 5240,
                    }
                )
            )
            path.chmod(0o644)
            result = tracker_setup.ultimate_wifi_snapshot(path)
            self.assertEqual(result["status"], "permissions-warning")
            self.assertFalse(result["configured"])
            self.assertNotIn("hidden!!", json.dumps(result))

    def test_wifi_rejects_values_that_cannot_be_valid_credentials(self) -> None:
        baseline: dict[str, object] = {
            "ssid": "AXOL",
            "pass": "valid-password",
            "country": "US",
            "freq": 5240,
        }
        invalid = {
            "oversized SSID": {"ssid": "x" * 33},
            "control character": {"ssid": "AXOL\n"},
            "short passphrase": {"pass": "short"},
            "unknown country": {"country": "ZZ"},
            "non-Wi-Fi frequency": {"freq": 1},
        }
        for label, replacement in invalid.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "wifi.json"
                payload = {**baseline, **replacement}
                with self.assertRaises(tracker_setup.TrackerSetupError) as caught:
                    tracker_setup.save_ultimate_wifi(payload, path)
                self.assertFalse(path.exists())
                self.assertNotIn(str(payload["pass"]), str(caught.exception))

    def test_runtime_wifi_validation_uses_the_same_rules_as_the_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            path.write_text(
                json.dumps(
                    {
                        "ssid": "AXOL",
                        "pass": "valid-password",
                        "country": "ZZ",
                        "freq": 1,
                    }
                )
            )
            self.assertIn("country", ultimate_wifi_config_error(path) or "")

    def test_atomic_writer_preserves_existing_operator_owner_when_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
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
                    "ultimate": {
                        "left": "a:b:c:d:e:f",
                        "right": "1:2:3:4:5:6",
                    },
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
                "right": {"ultimate:1:2:3:4:5:6": dict(_IDENTITY)},
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
            # The CAD candidate is status-only; it must not appear as current
            # measured editor values for the untouched side.
            self.assertEqual(result["right"]["status"], "candidate")
            self.assertIsNone(result["right"]["pos"])
            self.assertIsNone(result["right"]["quat"])
            saved = json.loads(calibration_path.read_text())
            self.assertEqual(saved["metadata"], original["metadata"])
            self.assertEqual(
                saved["left"]["survive:OTHER"],
                original["left"]["survive:OTHER"],
            )
            self.assertNotIn(
                ULTIMATE_POSE_CONVENTION_FIELD,
                saved["left"]["survive:LHR-LEFT"],
            )
            self.assertEqual(saved["right"], original["right"])
            self.assertEqual(calibration_path.stat().st_mode & 0o777, 0o600)

    def test_calibration_resolves_ultimate_binding_and_quest_configured_key(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                ultimate = tracker_setup.calibration_snapshot(
                    "ultimate", path=calibration_path
                )
                self.assertEqual(
                    ultimate["keys"],
                    {
                        "left": "ultimate:a:b:c:d:e:f",
                        "right": "ultimate:1:2:3:4:5:6",
                    },
                )
                for side in ("left", "right"):
                    self.assertEqual(ultimate[side]["status"], "candidate")
                    self.assertIsNone(ultimate[side]["pos"])
                    self.assertIsNone(ultimate[side]["quat"])

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
                saved = json.loads(calibration_path.read_text())
                self.assertNotIn(
                    ULTIMATE_POSE_CONVENTION_FIELD,
                    saved["right"][quest_key],
                )

    def test_ultimate_calibration_requires_explicit_convention_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "tracker.json"
            calibration_path = root / "tcp.json"
            self._tracker_config(config_path)
            keys = {
                "left": "ultimate:a:b:c:d:e:f",
                "right": "ultimate:1:2:3:4:5:6",
            }
            calibration_path.write_text(
                json.dumps({side: {key: dict(_IDENTITY)} for side, key in keys.items()})
            )

            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                stale = tracker_setup.calibration_snapshot(
                    "ultimate", path=calibration_path
                )
                self.assertEqual(
                    stale["activePoseConvention"],
                    {"quatOrder": "wxyz", "upAxis": "z"},
                )
                for side in ("left", "right"):
                    self.assertEqual(stale[side]["status"], "stale")
                    self.assertEqual(stale[side]["pos"], _IDENTITY["pos"])
                    self.assertIsNone(stale[side]["poseConvention"])

                adopted = tracker_setup.save_calibration(
                    "ultimate",
                    {"left": _keyed(keys["left"])},
                    path=calibration_path,
                )
                self.assertEqual(adopted["left"]["status"], "measured")
                self.assertEqual(adopted["right"]["status"], "stale")

            saved = json.loads(calibration_path.read_text())
            self.assertEqual(
                saved["left"][keys["left"]][ULTIMATE_POSE_CONVENTION_FIELD],
                {"quat_order": "wxyz", "up_axis": "z"},
            )

            config = TrackerConfig(
                backend="ultimate",
                left="a:b:c:d:e:f",
                right="1:2:3:4:5:6",
                bindings={"ultimate": {"left": "a:b:c:d:e:f", "right": "1:2:3:4:5:6"}},
                ultimate_quat_order="wxyz",
                ultimate_up_axis="y",
            )
            save_tracker_config(config, config_path)
            with patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path):
                mismatched = tracker_setup.calibration_snapshot(
                    "ultimate", path=calibration_path
                )
                self.assertEqual(mismatched["left"]["status"], "stale")
                self.assertEqual(
                    mismatched["left"]["poseConvention"],
                    {"quatOrder": "wxyz", "upAxis": "z"},
                )
                self.assertEqual(
                    mismatched["activePoseConvention"],
                    {"quatOrder": "wxyz", "upAxis": "y"},
                )

                resaved = tracker_setup.save_calibration(
                    "ultimate",
                    {"left": _keyed(keys["left"])},
                    path=calibration_path,
                )
                self.assertEqual(resaved["left"]["status"], "measured")
                self.assertEqual(
                    resaved["left"]["poseConvention"],
                    {"quatOrder": "wxyz", "upAxis": "y"},
                )

            config.ultimate_up_axis = "invalid"
            save_tracker_config(config, config_path)
            with (
                patch.object(tracker_setup, "TRACKER_CONFIG_FILE", config_path),
                self.assertRaisesRegex(
                    tracker_setup.TrackerSetupError,
                    "pose convention is invalid",
                ),
            ):
                tracker_setup.save_calibration(
                    "ultimate",
                    {"left": _keyed(keys["left"])},
                    path=calibration_path,
                )

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
    async def test_bindings_readiness_surfaces_stale_ultimate_convention(
        self,
    ) -> None:
        app = _test_app(_Settings())
        transport = httpx.ASGITransport(app=app)
        config = TrackerConfig(
            backend="ultimate",
            left="a:b:c:d:e:f",
            right="1:2:3:4:5:6",
            bindings={
                "ultimate": {
                    "left": "a:b:c:d:e:f",
                    "right": "1:2:3:4:5:6",
                }
            },
        )
        calibration = {
            "left": {"status": "stale"},
            "right": {"status": "measured"},
        }
        with (
            patch(
                "almond_axol.cli.tracker_install.lighthouse_readiness",
                return_value={},
            ),
            patch(
                "almond_axol.cli.tracker_ultimate.ultimate_runtime_readiness",
                return_value={},
            ),
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch(
                "almond_axol.mantis.calibration.load_tcp_transforms",
                return_value={},
            ),
            patch(
                "almond_axol.vr.server.get_last_quest_pose_datum",
                return_value=None,
            ),
            patch.object(
                tracker_setup,
                "calibration_snapshot",
                return_value=calibration,
            ),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/api/tracker/bindings")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()["sources"]["ultimate"]["transforms"],
            {"left": "stale", "right": "measured"},
        )

    async def test_untrusted_browser_cannot_read_or_probe_api(self) -> None:
        app = _test_app(_Settings())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://attacker.example:8001"
        ) as client:
            explicit_origin = await client.get(
                "/api/tracker/ultimate/wifi",
                headers={"Origin": "http://attacker.example:8001"},
            )
            originless_navigation = await client.get(
                "/api/tracker/ultimate/wifi",
                headers={"Sec-Fetch-Site": "cross-site"},
            )
            native_client = await client.get("/api/tracker/ultimate/wifi")

        self.assertEqual(explicit_origin.status_code, 403)
        self.assertEqual(originless_navigation.status_code, 403)
        self.assertEqual(native_client.status_code, 200)

    async def test_untrusted_browser_origin_cannot_write_tracker_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            app = _test_app(_Settings())
            transport = httpx.ASGITransport(app=app)
            with patch.object(tracker_setup, "ULTIMATE_WIFI_CONFIG_FILE", path):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://attacker.example:8001"
                ) as client:
                    response = await client.put(
                        "/api/tracker/ultimate/wifi",
                        # Matching Origin and Host is still untrusted: both are
                        # attacker-controlled in a plain-HTTP DNS rebind.
                        headers={"Origin": "http://attacker.example:8001"},
                        json={
                            "ssid": "AXOL",
                            "pass": "must-not-write",
                            "country": "US",
                            "freq": 5240,
                        },
                    )
            self.assertEqual(response.status_code, 403)
            self.assertFalse(path.exists())

    async def test_hosted_browser_origin_can_write_tracker_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            app = _test_app(_Settings())
            transport = httpx.ASGITransport(app=app)
            with patch.object(tracker_setup, "ULTIMATE_WIFI_CONFIG_FILE", path):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.put(
                        "/api/tracker/ultimate/wifi",
                        headers={"Origin": "https://axol.almond.bot"},
                        json={
                            "ssid": "AXOL",
                            "pass": "allowed-write",
                            "country": "US",
                            "freq": 5240,
                        },
                    )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(path.exists())

    async def test_wifi_api_never_echoes_password_and_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wifi.json"
            app = _test_app(_Settings())
            transport = httpx.ASGITransport(app=app)
            with patch.object(tracker_setup, "ULTIMATE_WIFI_CONFIG_FILE", path):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    secret = "api-secret-value"
                    response = await client.put(
                        "/api/tracker/ultimate/wifi",
                        json={
                            "ssid": "AXOL",
                            "pass": secret,
                            "country": "US",
                            "freq": 5240,
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn(secret, response.text)
                    response = await client.put(
                        "/api/tracker/ultimate/wifi",
                        json={"ssid": "NEW", "country": "US", "freq": 5200},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn(secret, response.text)
                    response = await client.get("/api/tracker/ultimate/wifi")
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn(secret, response.text)
            self.assertEqual(json.loads(path.read_text())["pass"], secret)

    async def test_wifi_api_validation_error_does_not_echo_body(self) -> None:
        app = _test_app(_Settings())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            secret = "bad-request-secret"
            response = await client.put(
                "/api/tracker/ultimate/wifi",
                json={
                    "ssid": "AXOL",
                    "pass": secret,
                    "country": "US",
                    "freq": 5240,
                    "extra": True,
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(secret, response.text)

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
                    self.assertEqual(value["right"]["status"], "missing")
                    self.assertIsNone(value["right"]["pos"])
                    response = await client.get("/api/tracker/calibration/quest")
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["left"]["pos"], [0.0, 0.0, 0.0])

    async def test_calibration_api_rejects_unknown_source(self) -> None:
        app = _test_app(_Settings())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/tracker/calibration/unknown")
        self.assertEqual(response.status_code, 400)

    def test_ultimate_check_is_a_runnable_serve_command(self) -> None:
        self.assertIn("tracker.ultimate.check", COMMANDS)
        schema = get_schema("tracker.ultimate.check")
        self.assertEqual(schema.nodes, [])


if __name__ == "__main__":
    unittest.main()
