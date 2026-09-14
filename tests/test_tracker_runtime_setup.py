from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.cli import (
    mantis_bridge,
    tracker_identify,
    tracker_install,
    tracker_pair,
    tracker_ultimate,
)


def _ultimate_probe(*, log_suppression: bool = True) -> dict[str, object]:
    return {
        "hid_ok": True,
        "hid_device_api": True,
        "hid_enumerate_api": True,
        "hid_error": "",
        "interfaces": [
            {"interface": 0, "path": "/dev/hidraw-test", "accessible": True}
        ],
        "pyvut_ok": True,
        "pyvut_error": "",
        "api_compatible": True,
        "log_suppression_api": log_suppression,
        "packaged_wifi": "customized",
    }


def _write_lighthouse_manifest(
    manifest: Path,
    *,
    rule: Path,
    cli: Path,
    artifacts: tuple[Path, ...] = (),
) -> None:
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    records = tracker_install._runtime_artifact_records(  # noqa: SLF001
        (cli, *artifacts), require_root_control=False
    )
    assert records is not None
    manifest.write_text(
        json.dumps(
            {
                "schema": tracker_install._MANIFEST_SCHEMA,  # noqa: SLF001
                "pinnedRef": tracker_install._PINNED_REF,  # noqa: SLF001
                "buildRevision": tracker_install._BUILD_REVISION,  # noqa: SLF001
                "udevRuleSha256": tracker_install._file_digest(rule),  # noqa: SLF001
                "surviveCliPath": str(cli),
                "runtimeArtifacts": records,
            }
        )
    )


class LighthouseRuntimeReadinessTest(unittest.TestCase):
    def test_requires_exact_manifest_runtime_and_matching_udev_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed_rule = root / "etc" / "81-vive.rules"
            manifest = root / "libsurvive-manifest.json"
            cli = root / "usr" / "local" / "bin" / "survive-cli"
            cli.parent.mkdir(parents=True)
            installed_rule.parent.mkdir(parents=True)
            installed_rule.write_text("vive permission rule\n")
            _write_lighthouse_manifest(manifest, rule=installed_rule, cli=cli)

            ready = tracker_install.lighthouse_readiness(
                installed_udev_rule=installed_rule, manifest_path=manifest
            )
            installed_rule.write_text("different rule\n")
            stale_rule = tracker_install.lighthouse_readiness(
                installed_udev_rule=installed_rule, manifest_path=manifest
            )
            payload = json.loads(manifest.read_text())
            payload["buildRevision"] = "older-build"
            manifest.write_text(json.dumps(payload))
            stale_stamp = tracker_install.lighthouse_readiness(
                installed_udev_rule=installed_rule, manifest_path=manifest
            )

        self.assertTrue(ready["installed"])
        self.assertTrue(ready["pinnedBuild"])
        self.assertTrue(ready["udevReady"])
        self.assertFalse(stale_rule["installed"])
        self.assertFalse(stale_rule["udevReady"])
        self.assertFalse(stale_stamp["installed"])
        self.assertFalse(stale_stamp["pinnedBuild"])

    def test_executable_alone_is_not_a_supported_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = tracker_install.lighthouse_readiness(
                installed_udev_rule=Path(directory) / "81-vive.rules",
                manifest_path=Path(directory) / "missing-manifest",
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["installed"])

    def test_importable_python_backend_cannot_replace_attested_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed_rule = root / "etc" / "81-vive.rules"
            installed_rule.parent.mkdir(parents=True)
            installed_rule.write_text("vive permission rule\n")
            result = tracker_install.lighthouse_readiness(
                installed_udev_rule=installed_rule,
                manifest_path=root / "missing-manifest",
            )

        self.assertFalse(result["available"])
        self.assertFalse(result["pairingCli"])
        self.assertFalse(result["installed"])
        self.assertIn("attested", " ".join(result["issues"]))

    def test_machine_manifest_is_independent_of_callers_build_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule = root / "etc" / "81-vive.rules"
            manifest = root / "var" / "libsurvive-build-stamp"
            rule.parent.mkdir(parents=True)
            manifest.parent.mkdir(parents=True)
            rule.write_text("installed pinned rule\n")
            cli = root / "usr" / "local" / "bin" / "survive-cli"
            cli.parent.mkdir(parents=True)
            _write_lighthouse_manifest(manifest, rule=rule, cli=cli)
            result = tracker_install.lighthouse_readiness(
                src=root / "unrelated-operator-cache",
                installed_udev_rule=rule,
                manifest_path=manifest,
            )

        self.assertTrue(result["installed"])
        self.assertEqual(result["stampPath"], str(manifest))

    def test_runtime_byte_drift_invalidates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rule = root / "81-vive.rules"
            rule.write_text("installed pinned rule\n")
            manifest = root / "manifest.json"
            cli = root / "survive-cli"
            _write_lighthouse_manifest(manifest, rule=rule, cli=cli)
            cli.write_text("changed after installation\n")
            cli.chmod(0o755)

            result = tracker_install.lighthouse_readiness(
                installed_udev_rule=rule, manifest_path=manifest
            )

        self.assertFalse(result["installed"])
        self.assertFalse(result["runtimeArtifacts"])


class LighthouseReattestationTest(unittest.TestCase):
    def _cache(self, root: Path, revision: str) -> Path:
        src = root / "libsurvive"
        (src / "build").mkdir(parents=True)
        (src / tracker_install._STAMP).write_text(  # noqa: SLF001
            f"{tracker_install._PINNED_REF}\n{revision}\nold-digest\n"  # noqa: SLF001
        )
        (src / tracker_install._INSTALL_MANIFEST).write_text(  # noqa: SLF001
            "/usr/local/bin/survive-cli\n"
        )
        return src

    def test_stale_manifest_format_reattests_without_rebuilding(self) -> None:
        readiness = [
            {"installed": False, "available": False, "pinnedBuild": False},
            {"installed": True, "available": True, "pinnedBuild": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            src = self._cache(Path(directory), "libusb-v1")
            with (
                patch.object(tracker_install, "_src_dir", return_value=src),
                patch.object(
                    tracker_install, "lighthouse_readiness", side_effect=readiness
                ),
                patch.object(tracker_install, "prime_sudo", return_value=True),
                patch.object(
                    tracker_install, "_install_udev_rule", return_value=True
                ) as udev,
                patch.object(
                    tracker_install, "_install_machine_stamp", return_value=True
                ) as stamp,
                patch.object(tracker_install, "_install_build_deps") as deps,
                patch.object(tracker_install, "_build_and_install") as build,
            ):
                self.assertTrue(tracker_install.ensure_installed())

        udev.assert_called_once_with(src)
        stamp.assert_called_once_with(src)
        deps.assert_not_called()
        build.assert_not_called()

    def test_root_run_via_sudo_considers_the_operators_cache(self) -> None:
        operator = SimpleNamespace(pw_dir="/home/operator")
        with (
            patch.object(tracker_install.os, "geteuid", return_value=0),
            patch.dict(tracker_install.os.environ, {"SUDO_USER": "operator"}),
            patch.object(tracker_install.pwd, "getpwnam", return_value=operator),
        ):
            caches = tracker_install._build_caches()  # noqa: SLF001
        with (
            patch.object(tracker_install.os, "geteuid", return_value=0),
            patch.dict(tracker_install.os.environ, {}, clear=True),
        ):
            service_caches = tracker_install._build_caches()  # noqa: SLF001

        self.assertEqual(
            caches,
            (
                Path("/opt/almond/libsurvive"),
                Path("/home/operator/.almond/libsurvive"),
            ),
        )
        self.assertEqual(service_caches, (Path("/opt/almond/libsurvive"),))

    def test_different_options_or_unfinished_build_are_not_reattestable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            other_options = self._cache(Path(directory) / "a", "hidapi-v0")
            unfinished = self._cache(Path(directory) / "b", "libusb-v1")
            (unfinished / tracker_install._INSTALL_MANIFEST).unlink()  # noqa: SLF001

            self.assertFalse(
                tracker_install._reattestable_build(other_options)  # noqa: SLF001
            )
            self.assertFalse(
                tracker_install._reattestable_build(unfinished)  # noqa: SLF001
            )


class UltimateRuntimeReadinessTest(unittest.TestCase):
    def _inspect(
        self,
        *,
        probe: dict[str, object] | None = None,
        commit: str | None = tracker_ultimate._PYVUT_REF,  # noqa: SLF001
        rules: list[Path] | None = None,
    ) -> dict[str, object]:
        with (
            patch.object(
                tracker_ultimate,
                "_python_probe",
                return_value=probe or _ultimate_probe(),
            ),
            patch.object(tracker_ultimate, "_missing_system_packages", return_value=[]),
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", commit),
            ),
            patch.object(
                tracker_ultimate, "ultimate_dongle_present", return_value=True
            ),
            patch.object(
                tracker_ultimate,
                "_matching_udev_rules",
                return_value=rules if rules is not None else [Path("/etc/test.rules")],
            ),
            patch.object(tracker_ultimate, "_operator_has_dialout", return_value=False),
            patch.object(
                tracker_ultimate,
                "_wifi_config_status",
                return_value=("OK", "redacted valid config", "valid"),
            ),
        ):
            return tracker_ultimate.ultimate_runtime_readiness(cached=False)

    def test_pinned_noninvasive_runtime_and_endpoint_are_ready(self) -> None:
        result = self._inspect()

        self.assertTrue(result["installed"])
        self.assertTrue(result["pinnedPyvut"])
        self.assertTrue(result["logSuppression"])
        self.assertTrue(result["udevReady"])
        self.assertEqual(result["endpointStatus"], "accessible")
        # uaccess on the live endpoint is sufficient even without dialout.
        self.assertTrue(result["operatorAccess"])

    def test_unpinned_or_unsafe_runtime_fails_closed(self) -> None:
        probe = _ultimate_probe(log_suppression=False)
        result = self._inspect(probe=probe, commit="unexpected", rules=[])

        self.assertFalse(result["installed"])
        self.assertFalse(result["pinnedPyvut"])
        self.assertFalse(result["logSuppression"])
        self.assertFalse(result["udevReady"])
        self.assertIn("not pinned", " ".join(result["issues"]))

    def test_udev_match_requires_a_rule_that_actually_grants_access(self) -> None:
        prefix = (
            'SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0bb4", ATTRS{idProduct}=="0350", '
        )
        self.assertFalse(
            tracker_ultimate._udev_line_grants_access(  # noqa: SLF001
                (prefix + 'MODE="0600"').lower().replace(" ", "")
            )
        )
        self.assertFalse(
            tracker_ultimate._udev_line_grants_access(  # noqa: SLF001
                (prefix + '# TAG+="uaccess"').lower().replace(" ", "")
            )
        )
        self.assertTrue(
            tracker_ultimate._udev_line_grants_access(  # noqa: SLF001
                (prefix + 'MODE="0660", GROUP="dialout", TAG+="uaccess"')
                .lower()
                .replace(" ", "")
            )
        )

    def test_ui_probe_cache_avoids_relaunching_child_interpreter(self) -> None:
        tracker_ultimate._clear_runtime_probe_cache()  # noqa: SLF001
        with patch.object(
            tracker_ultimate, "_python_probe", return_value=_ultimate_probe()
        ) as probe:
            first = tracker_ultimate._cached_python_probe(max_age_s=60.0)  # noqa: SLF001
            second = tracker_ultimate._cached_python_probe(max_age_s=60.0)  # noqa: SLF001
        tracker_ultimate._clear_runtime_probe_cache()  # noqa: SLF001

        self.assertEqual(first, second)
        self.assertEqual(probe.call_count, 1)

    def test_root_owned_shared_environment_uses_narrow_sudo_install(self) -> None:
        installed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch.object(
                tracker_ultimate.shutil,
                "which",
                side_effect=lambda name: f"/usr/bin/{name}",
            ),
            patch.object(
                tracker_ultimate.sysconfig, "get_path", return_value="/opt/axol/site"
            ),
            patch.object(tracker_ultimate.os, "access", return_value=False),
            patch.object(tracker_ultimate, "prime_sudo", return_value=True),
            patch.object(
                tracker_ultimate, "run_root", return_value=installed
            ) as run_root,
            patch.object(tracker_ultimate, "_run") as unprivileged,
        ):
            self.assertTrue(tracker_ultimate._pip_install_pyvut())  # noqa: SLF001

        unprivileged.assert_not_called()
        command = run_root.call_args.args[0]
        self.assertEqual(command[:3], ["/usr/bin/uv", "pip", "install"])
        self.assertIn(tracker_ultimate.sys.executable, command)

    def test_installer_repairs_native_deps_even_when_pyvut_is_pinned(self) -> None:
        probe = _ultimate_probe()
        readiness = {"installed": False, "issues": ["missing HID libraries"]}
        with (
            patch.object(tracker_ultimate, "_python_probe", return_value=probe),
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", tracker_ultimate._PYVUT_REF),  # noqa: SLF001
            ),
            patch.object(
                tracker_ultimate, "_install_system_packages", return_value=False
            ) as native,
            patch.object(tracker_ultimate, "_install_udev_rule", return_value=True),
            patch.object(tracker_ultimate, "_print_wifi_action"),
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_readiness",
                return_value=readiness,
            ),
            self.assertRaisesRegex(SystemExit, "missing HID libraries"),
        ):
            tracker_ultimate.run_install()

        native.assert_called_once_with()


class ManagedRuntimeGateTest(unittest.TestCase):
    def test_quest_needs_no_local_tracker_runtime(self) -> None:
        mantis_bridge.require_mantis_tracker_readiness("quest")

    def test_lighthouse_requires_the_supported_install(self) -> None:
        with (
            patch.object(
                tracker_install,
                "lighthouse_readiness",
                return_value={
                    "installed": False,
                    "issues": ["the pinned build stamp is stale"],
                },
            ),
            self.assertRaisesRegex(RuntimeError, "axol tracker.install"),
        ):
            mantis_bridge.require_mantis_tracker_readiness("lighthouse")

    def test_ultimate_requires_live_access_and_protected_wifi(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_readiness",
                return_value={
                    "installed": True,
                    "issues": [],
                    "dongleConnected": True,
                    "endpointStatus": "accessible",
                    "operatorAccess": True,
                    "wifiConfig": "permissions-warning",
                },
            ),
            self.assertRaisesRegex(RuntimeError, "mode-0600"),
        ):
            mantis_bridge.require_mantis_tracker_readiness("ultimate")


class PersistentTrackerSetupGateTest(unittest.TestCase):
    def test_physical_identify_requires_runtime_before_opening_or_saving(self) -> None:
        for backend, source_name in (
            ("survive", "lighthouse"),
            ("ultimate", "ultimate"),
        ):
            with self.subTest(backend=backend):
                config = SimpleNamespace(backend=backend)
                with (
                    patch(
                        "almond_axol.tracker.load_tracker_config",
                        return_value=config,
                    ),
                    patch("almond_axol.tracker.create_source") as create_source,
                    patch(
                        "almond_axol.tracker.config.save_tracker_config"
                    ) as save_config,
                    patch.object(
                        mantis_bridge,
                        "require_mantis_tracker_readiness",
                        side_effect=RuntimeError("runtime unsupported"),
                    ) as readiness,
                    self.assertRaisesRegex(SystemExit, "runtime unsupported"),
                ):
                    tracker_identify.run(
                        SimpleNamespace(backend=None, web_prompts=False)
                    )

                readiness.assert_called_once_with(source_name)
                create_source.assert_not_called()
                save_config.assert_not_called()

    def test_synthetic_identify_is_exempt_from_runtime_gate(self) -> None:
        config = SimpleNamespace(backend="synthetic")
        source = SimpleNamespace(
            start=Mock(side_effect=RuntimeError("synthetic started")),
            stop=Mock(),
        )
        with (
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch("almond_axol.tracker.create_source", return_value=source),
            patch.object(
                mantis_bridge, "require_mantis_tracker_readiness"
            ) as readiness,
            self.assertRaisesRegex(RuntimeError, "synthetic started"),
        ):
            tracker_identify.run(SimpleNamespace(backend=None, web_prompts=False))

        readiness.assert_not_called()
        source.stop.assert_called_once_with()

    def test_lighthouse_pair_requires_pinned_runtime_before_spawning(self) -> None:
        with (
            patch.object(
                tracker_install,
                "lighthouse_readiness",
                return_value={
                    "installed": False,
                    "issues": ["the pinned build stamp is stale"],
                },
            ) as readiness,
            patch.object(
                tracker_install, "verified_survive_cli"
            ) as verified_survive_cli,
            patch.object(tracker_pair.subprocess, "Popen") as popen,
            self.assertRaisesRegex(SystemExit, "axol tracker.install"),
        ):
            tracker_pair.run(SimpleNamespace(timeout=90.0))

        readiness.assert_called_once_with()
        verified_survive_cli.assert_not_called()
        popen.assert_not_called()


class UltimateCheckExitStatusTest(unittest.TestCase):
    def test_missing_or_insecure_wifi_fails_the_check(self) -> None:
        readiness = {
            "dongleStatus": "connected",
            "missingNativeDependencies": [],
            "pythonHid": True,
            "hidError": "",
            "pyvutVersion": "0.0.test",
            "installedRef": tracker_ultimate._PYVUT_REF,  # noqa: SLF001
            "apiCompatible": True,
            "pinnedPyvut": True,
            "pinnedRef": tracker_ultimate._PYVUT_REF,  # noqa: SLF001
            "pyvutError": "",
            "logSuppression": True,
            "wifiConfig": "permissions-warning",
            "wifiDetail": "valid values but mode 0644",
            "interfaces": [
                {"interface": 0, "path": "/dev/hidraw-test", "accessible": True}
            ],
            "endpointStatus": "accessible",
            "udevRules": ["/etc/udev/rules.d/test.rules"],
            "operatorAccess": True,
            "durableOperatorAccess": True,
        }
        with (
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_readiness",
                return_value=readiness,
            ),
            patch.object(
                tracker_ultimate,
                "_read_bindings",
                return_value=("AA:BB", "CC:DD", None),
            ),
            patch.object(
                tracker_ultimate,
                "is_ultimate_tracker_key",
                return_value=True,
            ),
            patch.object(
                tracker_ultimate,
                "_read_pose_conventions",
                return_value=("wxyz", "z"),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            tracker_ultimate.run_check()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
