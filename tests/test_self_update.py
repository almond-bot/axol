from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call, patch

from almond_axol.cli import tracker_ultimate, update_preflight
from almond_axol.serve import update


class _Process:
    def __init__(self, returncode: int = 0, output: bytes = b"") -> None:
        self.returncode = returncode
        self._output = output

    async def communicate(self) -> tuple[bytes, None]:
        return self._output, None


class _BlockingProcess:
    def __init__(self, *, terminate_exits: bool = True) -> None:
        self.returncode: int | None = None
        self.started = asyncio.Event()
        self.exited = asyncio.Event()
        self.terminate_exits = terminate_exits
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self) -> tuple[bytes, None]:
        self.started.set()
        await self.exited.wait()
        return b"", None

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.terminate_exits:
            self.returncode = -15
            self.exited.set()

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.exited.set()


class _HTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return json.dumps(self._payload).encode()


class _ByteHTTPResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.status = status
        self._stream = io.BytesIO(payload)

    def __enter__(self) -> _ByteHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _release_wheel(payload: bytes = b"wheel") -> update._ReleaseWheel:  # noqa: SLF001
    return update._ReleaseWheel(  # noqa: SLF001
        "almond_axol-0.1.37-py3-none-any.whl",
        "https://files.pythonhosted.org/packages/aa/almond_axol-0.1.37-py3-none-any.whl",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )


def _wheel_payload(
    *,
    name: str = "almond-axol",
    version: str = "0.1.37",
    installer: bytes = b"#!/usr/bin/env bash\necho verified\n",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            f"almond_axol-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n",
        )
        info = zipfile.ZipInfo("almond_axol/_installer.sh")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, installer)
    return output.getvalue()


def _updater() -> update.SelfUpdater:
    with (
        patch.object(update, "installed_origin", return_value=("https://repo", "abc")),
        patch.object(update, "installed_version", return_value="0.1.36"),
        patch.object(update, "installed_commit", return_value="abc"),
    ):
        result = update.SelfUpdater(lambda: True)
    result._remote_tag = "release-v0.1.37"  # noqa: SLF001
    result._remote_version = "0.1.37"  # noqa: SLF001
    result._state = "updating"  # noqa: SLF001
    result._launches_blocked = True  # noqa: SLF001
    result._release_wheel = Mock(return_value=_release_wheel())  # type: ignore[method-assign]  # noqa: SLF001
    stage = Path("/var/lib/almond-axol/update-workers/release-0.1.37-test")
    result._stage_release = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=update._StagedRelease(  # noqa: SLF001
            stage,
            stage / "almond_axol-0.1.37-py3-none-any.whl",
            stage / "install",
            "a" * 64,
        )
    )
    result._arm_durable_update_guard = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=None
    )
    result._disarm_durable_update_guard = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=None
    )
    result._operator_user_for_update = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=("robot", None)
    )
    return result


class UltimateUpdateRequirementTests(unittest.TestCase):
    def test_runtime_source_metadata_must_be_exact_and_unambiguous(self) -> None:
        exact_metadata = {
            "url": tracker_ultimate._PYVUT_REPO,  # noqa: SLF001
            "vcs_info": {
                "vcs": "git",
                "requested_revision": tracker_ultimate._PYVUT_REF,  # noqa: SLF001
                "commit_id": tracker_ultimate._PYVUT_REF,  # noqa: SLF001
            },
        }
        dist = Mock(version="0.0.test")
        dist.read_text.return_value = json.dumps(exact_metadata)
        with patch.object(tracker_ultimate, "distributions", return_value=[dist]):
            self.assertEqual(
                tracker_ultimate._installed_pyvut(),  # noqa: SLF001
                ("0.0.test", tracker_ultimate._PYVUT_REF),  # noqa: SLF001
            )

        dist.read_text.return_value = json.dumps(
            {**exact_metadata, "subdirectory": "operator-controlled"}
        )
        with patch.object(tracker_ultimate, "distributions", return_value=[dist]):
            self.assertEqual(
                tracker_ultimate._installed_pyvut(),  # noqa: SLF001
                ("0.0.test", None),
            )

        with patch.object(
            tracker_ultimate,
            "distributions",
            return_value=[dist, Mock(version="duplicate")],
        ):
            self.assertEqual(
                tracker_ultimate._installed_pyvut(),  # noqa: SLF001
                ("present-but-ambiguous", None),
            )

    def test_present_non_exact_runtime_blocks_force_replacement(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", "different-pin"),
            ),
            patch.object(tracker_ultimate, "_packaged_wifi_status") as wifi_status,
        ):
            requirement, error = tracker_ultimate.ultimate_runtime_update_requirement()
            self.assertIsNone(requirement)
            self.assertIn("VIVE Ultimate update blocked", error or "")
        wifi_status.assert_not_called()

    def test_absent_runtime_needs_no_preservation_requirement(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=(None, None),
            ),
            patch.object(tracker_ultimate, "_packaged_wifi_status") as wifi_status,
        ):
            self.assertEqual(
                tracker_ultimate.ultimate_runtime_update_requirement(),
                (None, None),
            )
        wifi_status.assert_not_called()

    def test_exact_runtime_with_uninspectable_wifi_blocks(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", tracker_ultimate._PYVUT_REF),  # noqa: SLF001
            ),
            patch.object(
                tracker_ultimate,
                "_packaged_wifi_status",
                return_value="missing",
            ),
        ):
            requirement, error = tracker_ultimate.ultimate_runtime_update_requirement()
        self.assertIsNone(requirement)
        self.assertIn("cannot be inspected safely", error or "")

    def test_current_pin_returns_the_exact_vcs_requirement(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", tracker_ultimate._PYVUT_REF),  # noqa: SLF001
            ),
            patch.object(
                tracker_ultimate,
                "_packaged_wifi_status",
                return_value="placeholder",
            ),
        ):
            self.assertEqual(
                tracker_ultimate.ultimate_runtime_update_requirement(),
                (tracker_ultimate._PYVUT_SPEC, None),  # noqa: SLF001
            )

    def test_custom_package_wifi_requires_a_valid_durable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "ultimate_wifi.json")
            with (
                patch.object(
                    tracker_ultimate,
                    "_installed_pyvut",
                    return_value=("0.0.test", tracker_ultimate._PYVUT_REF),  # noqa: SLF001
                ),
                patch.object(
                    tracker_ultimate,
                    "_packaged_wifi_status",
                    return_value="customized",
                ),
                patch.object(tracker_ultimate, "ULTIMATE_WIFI_CONFIG_FILE", config),
            ):
                config.write_text(
                    json.dumps(
                        {
                            "ssid": "private-map",
                            "pass": "not-in-the-error",
                            "country": "US",
                            "freq": 123,
                        }
                    )
                )
                requirement, error = (
                    tracker_ultimate.ultimate_runtime_update_requirement()
                )
                self.assertIsNone(requirement)
                self.assertIn("package-local Wi-Fi", error or "")
                self.assertNotIn("not-in-the-error", error or "")

                config.write_text(
                    json.dumps(
                        {
                            "ssid": "private-map",
                            "pass": "not-in-the-error",
                            "country": "US",
                            "freq": 5180,
                        }
                    )
                )
                self.assertEqual(
                    tracker_ultimate.ultimate_runtime_update_requirement(),
                    (tracker_ultimate._PYVUT_SPEC, None),  # noqa: SLF001
                )

    def test_cli_preflight_has_machine_readable_absent_and_preserve_states(
        self,
    ) -> None:
        stdout = io.StringIO()
        with (
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_update_requirement",
                return_value=(None, None),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            tracker_ultimate.run_update_preflight()
        self.assertEqual(stdout.getvalue(), "")

        stdout = io.StringIO()
        requirement = "git+https://github.com/nijkah/pyvut.git@" + "a" * 40
        with (
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_update_requirement",
                return_value=(requirement, None),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            tracker_ultimate.run_update_preflight()
        self.assertEqual(stdout.getvalue(), requirement + "\n")

    def test_cli_preflight_block_is_secret_safe_and_distinct(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                tracker_ultimate,
                "ultimate_runtime_update_requirement",
                return_value=(None, "move the redacted Wi-Fi config"),
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            tracker_ultimate.run_update_preflight()
        self.assertEqual(raised.exception.code, 20)
        self.assertEqual(stderr.getvalue(), "move the redacted Wi-Fi config\n")


class TorchUpdatePreflightTests(unittest.TestCase):
    def test_non_aarch64_does_not_import_or_inspect_torch(self) -> None:
        with (
            patch.object(update_preflight, "_is_aarch64", return_value=False),
            patch.object(update_preflight, "distribution") as get_distribution,
            patch.object(update_preflight.subprocess, "run") as run,
        ):
            self.assertIsNone(update_preflight._torch_replacement_error())  # noqa: SLF001
        get_distribution.assert_not_called()
        run.assert_not_called()

    def test_absent_aarch64_torch_is_safe_for_a_fresh_install(self) -> None:
        with (
            patch.object(update_preflight, "_is_aarch64", return_value=True),
            patch.object(
                update_preflight,
                "distribution",
                side_effect=update_preflight.PackageNotFoundError,
            ),
            patch.object(update_preflight.subprocess, "run") as run,
        ):
            self.assertIsNone(update_preflight._torch_replacement_error())  # noqa: SLF001
        run.assert_not_called()

    def test_default_pypi_aarch64_cpu_torch_is_safe(self) -> None:
        torch_dist = Mock(version="2.10.0")
        torch_dist.read_text.return_value = None
        probe = Mock(
            returncode=0,
            stdout=json.dumps({"version": "2.10.0+cpu", "cuda_build": False}),
        )
        with (
            patch.object(update_preflight, "_is_aarch64", return_value=True),
            patch.object(update_preflight, "distribution", return_value=torch_dist),
            patch.object(update_preflight.subprocess, "run", return_value=probe),
        ):
            self.assertIsNone(update_preflight._torch_replacement_error())  # noqa: SLF001

    def test_cuda_torch_blocks_with_fixed_safe_text(self) -> None:
        torch_dist = Mock(version="2.10.0")
        torch_dist.read_text.return_value = None
        probe = Mock(
            returncode=0,
            stdout=json.dumps({"version": "2.10.0+private-index", "cuda_build": True}),
        )
        with (
            patch.object(update_preflight, "_is_aarch64", return_value=True),
            patch.object(update_preflight, "distribution", return_value=torch_dist),
            patch.object(update_preflight.subprocess, "run", return_value=probe),
        ):
            error = update_preflight._torch_replacement_error()  # noqa: SLF001
        self.assertIn("CPU-only torch==2.10.0", error or "")
        self.assertNotIn("private-index", error or "")

    def test_custom_direct_or_versioned_torch_blocks_without_import(self) -> None:
        for version, direct_url in (
            ("2.10.0+nv-custom", None),
            ("2.10.0", '{"url": "https://private.invalid/wheel"}'),
        ):
            with self.subTest(version=version, direct_url=direct_url):
                torch_dist = Mock(version=version)
                torch_dist.read_text.return_value = direct_url
                with (
                    patch.object(update_preflight, "_is_aarch64", return_value=True),
                    patch.object(
                        update_preflight, "distribution", return_value=torch_dist
                    ),
                    patch.object(update_preflight.subprocess, "run") as run,
                ):
                    error = update_preflight._torch_replacement_error()  # noqa: SLF001
                self.assertIn("custom PyTorch", error or "")
                self.assertNotIn("private.invalid", error or "")
                run.assert_not_called()

    def test_shared_contract_blocks_before_ultimate_inspection(self) -> None:
        with (
            patch.object(
                update_preflight,
                "_torch_replacement_error",
                return_value="fixed torch block",
            ),
            patch.object(
                update_preflight, "ultimate_runtime_update_requirement"
            ) as ultimate,
        ):
            self.assertEqual(
                update_preflight.release_update_requirements(),
                ([], "fixed torch block"),
            )
        ultimate.assert_not_called()

    def test_shared_cli_block_has_distinct_status_and_safe_output(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(
                update_preflight,
                "release_update_requirements",
                return_value=([], "fixed safe error"),
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            update_preflight.run()
        self.assertEqual(raised.exception.code, 20)
        self.assertEqual(stderr.getvalue(), "fixed safe error\n")

    def test_unexpected_inspection_error_is_sanitized(self) -> None:
        with patch.object(
            update_preflight,
            "_torch_replacement_error",
            side_effect=RuntimeError("https://user:secret@private.invalid"),
        ):
            requirements, error = update_preflight.release_update_requirements()
        self.assertEqual(requirements, [])
        self.assertIn("could not be inspected safely", error or "")
        self.assertNotIn("secret", error or "")
        self.assertNotIn("private.invalid", error or "")


class PluginUpdatePreflightTests(unittest.TestCase):
    def test_absent_plugin_needs_no_preservation(self) -> None:
        with patch.object(
            update_preflight,
            "distribution",
            side_effect=update_preflight.PackageNotFoundError,
        ):
            self.assertEqual(
                update_preflight._lerobot_plugin_update_requirement(),  # noqa: SLF001
                (None, None),
            )

    def test_published_plugin_preserves_its_exact_version(self) -> None:
        plugin_dist = Mock(version="0.1.1")
        plugin_dist.read_text.return_value = None
        with patch.object(update_preflight, "distribution", return_value=plugin_dist):
            self.assertEqual(
                update_preflight._lerobot_plugin_update_requirement(),  # noqa: SLF001
                ("lerobot_robot_axol==0.1.1", None),
            )

    def test_direct_plugin_source_blocks_without_exposing_its_url(self) -> None:
        plugin_dist = Mock(version="0.1.1")
        plugin_dist.read_text.return_value = (
            '{"url":"https://user:secret@private.invalid/plugin.whl"}'
        )
        with patch.object(update_preflight, "distribution", return_value=plugin_dist):
            requirement, error = update_preflight._lerobot_plugin_update_requirement()  # noqa: SLF001
        self.assertIsNone(requirement)
        self.assertIn("direct or customized", error or "")
        self.assertNotIn("secret", error or "")
        self.assertNotIn("private.invalid", error or "")

    def test_unexpected_plugin_inspection_error_is_sanitized(self) -> None:
        with (
            patch.object(
                update_preflight,
                "_torch_replacement_error",
                return_value=None,
            ),
            patch.object(
                update_preflight,
                "ultimate_runtime_update_requirement",
                return_value=(None, None),
            ),
            patch.object(
                update_preflight,
                "_lerobot_plugin_update_requirement",
                side_effect=RuntimeError("https://user:secret@private.invalid"),
            ),
        ):
            requirements, error = update_preflight.release_update_requirements()

        self.assertEqual(requirements, [])
        self.assertIn("could not be inspected safely", error or "")
        self.assertNotIn("secret", error or "")
        self.assertNotIn("private.invalid", error or "")

    def test_shared_cli_emits_each_allowlisted_requirement_on_its_own_line(
        self,
    ) -> None:
        requirements = [
            "git+https://github.com/nijkah/pyvut.git@" + "a" * 40,
            "lerobot_robot_axol==0.1.1",
        ]
        stdout = io.StringIO()
        with (
            patch.object(
                update_preflight,
                "release_update_requirements",
                return_value=(requirements, None),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            update_preflight.run()
        self.assertEqual(stdout.getvalue(), "\n".join(requirements) + "\n")


class InstallerUltimatePreservationTests(unittest.TestCase):
    _PYVUT_REF = "fcfcd33f4c1f16b0d84f5f741dc1319abdc7942a"
    _PYVUT_REPO_URL = "https://github.com/nijkah/pyvut.git"
    _PYVUT_SPEC = f"git+{_PYVUT_REPO_URL}@{_PYVUT_REF}"
    _PYVUT_DEFAULT_WIFI = b"""{
    "ssid": "test_5G",
    "pass": "testtest",
    "country": "US",
    "freq": 5240
}"""

    def _run_legacy_ultimate_probe(
        self,
        *,
        direct_url: object | None = None,
        write_direct_url: bool = True,
        wifi: bytes | None = _PYVUT_DEFAULT_WIFI,
        wifi_record_count: int = 1,
        install_distribution: bool = True,
        duplicate_distribution: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        installer_path = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        )
        installer = installer_path.read_text()
        function_start = installer.index("legacy_ultimate_update_requirement() {")
        function_end = installer.index("\n}\n", function_start) + len("\n}\n")
        function = installer[function_start:function_end]

        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            if install_distribution:
                package = site / "pyvut"
                dist_info = site / "pyvut-0.0.test.dist-info"
                package.mkdir()
                dist_info.mkdir()
                # A successful probe proves it inspected metadata without
                # importing pyvut (which may require unavailable HID libraries).
                (package / "__init__.py").write_text(
                    'raise RuntimeError("legacy probe imported pyvut")\n'
                )
                if wifi is not None:
                    (package / "wifi_info.json").write_bytes(wifi)
                (dist_info / "METADATA").write_text(
                    "Metadata-Version: 2.1\nName: pyvut\nVersion: 0.0.test\n"
                )
                if write_direct_url:
                    value = direct_url
                    if value is None:
                        value = {
                            "url": self._PYVUT_REPO_URL,
                            "vcs_info": {
                                "vcs": "git",
                                "requested_revision": self._PYVUT_REF,
                                "commit_id": self._PYVUT_REF,
                            },
                        }
                    (dist_info / "direct_url.json").write_text(json.dumps(value))
                records = ["pyvut/__init__.py,,"]
                records.extend(
                    "pyvut/wifi_info.json,," for _ in range(wifi_record_count)
                )
                records.extend(
                    [
                        "pyvut-0.0.test.dist-info/METADATA,,",
                        "pyvut-0.0.test.dist-info/direct_url.json,,",
                        "pyvut-0.0.test.dist-info/RECORD,,",
                    ]
                )
                (dist_info / "RECORD").write_text("\n".join(records) + "\n")
                if duplicate_distribution:
                    duplicate = site / "pyvut-0.0.duplicate.dist-info"
                    duplicate.mkdir()
                    for filename in ("METADATA", "direct_url.json", "RECORD"):
                        source = dist_info / filename
                        if source.exists():
                            (duplicate / filename).write_bytes(source.read_bytes())

            script = "\n".join(
                [
                    "set -euo pipefail",
                    f'PYVUT_REPO_URL="{self._PYVUT_REPO_URL}"',
                    f'PYVUT_REF="{self._PYVUT_REF}"',
                    'PYVUT_SPEC="git+${PYVUT_REPO_URL}@${PYVUT_REF}"',
                    "PYVUT_DEFAULT_WIFI_SHA256="
                    '"fd64dd89b6dd61d06e91b1a5c913aa7fcae5ac2654903eb3f7e6dac8aeee2b67"',
                    "UPDATE_WITH_ARGS=()",
                    "ULTIMATE_PRESERVED=0",
                    "say() { :; }",
                    "die() { printf '%s\\n' \"$*\" >&2; exit 1; }",
                    function,
                    'legacy_ultimate_update_requirement "$1"',
                    "printf 'preserved=%s\\n' \"${ULTIMATE_PRESERVED}\"",
                    'for arg in "${UPDATE_WITH_ARGS[@]}"; do '
                    "printf 'arg=%s\\n' \"${arg}\"; done",
                ]
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(site)
            environment["PYTHONNOUSERSITE"] = "1"
            return subprocess.run(
                ["bash", "-c", script, "bash", sys.executable],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

    def test_legacy_ultimate_probe_preserves_only_the_exact_safe_runtime(
        self,
    ) -> None:
        for url in (self._PYVUT_REPO_URL, self._PYVUT_REPO_URL.removesuffix(".git")):
            with self.subTest(url=url):
                result = self._run_legacy_ultimate_probe(
                    direct_url={
                        "url": url,
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": self._PYVUT_REF,
                        },
                    }
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    result.stdout,
                    f"preserved=1\narg=--with\narg={self._PYVUT_SPEC}\n",
                )

    def test_legacy_ultimate_probe_is_absent_safe_and_fail_closed(self) -> None:
        absent = self._run_legacy_ultimate_probe(install_distribution=False)
        self.assertEqual(absent.returncode, 0, absent.stderr)
        self.assertEqual(absent.stdout, "preserved=0\n")

        cases = {
            "credentialed-url": {
                "direct_url": {
                    "url": "https://operator:private@github.com/nijkah/pyvut.git",
                    "vcs_info": {"vcs": "git", "commit_id": self._PYVUT_REF},
                }
            },
            "wrong-commit": {
                "direct_url": {
                    "url": self._PYVUT_REPO_URL,
                    "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                }
            },
            "custom-wifi": {"wifi": b'{"pass":"private-value"}\n'},
            "missing-wifi": {"wifi": None},
            "missing-record": {"wifi_record_count": 0},
            "ambiguous-record": {"wifi_record_count": 2},
            "ambiguous-distribution": {"duplicate_distribution": True},
            "missing-direct-url": {"write_direct_url": False},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                result = self._run_legacy_ultimate_probe(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("VIVE Ultimate update blocked", result.stderr)
                self.assertNotIn("private-value", result.stderr)
                self.assertNotIn("operator:private", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_installer_preserves_and_verifies_before_service_start(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        dataset_gate = installer.index("Existing datasets were found under")
        preflight = installer.index('"${AXOL}" update-preflight')
        pypi_check = installer.index(
            '"https://pypi.org/pypi/almond-axol/${VERSION}/json"'
        )
        uv_install = installer.index('UV_ARCHIVE_URL="https://github.com/astral-sh/uv')
        uv_verify = installer.index("sha256sum --check --status", uv_install)
        guard_write = installer.index(
            'durable_replace "${GUARD_TMP}" "${UPDATE_GUARD_FILE}"'
        )
        mantis_guard_write = installer.index(
            'durable_replace "${MANTIS_GUARD_TMP}" "${MANTIS_UPDATE_GUARD_FILE}"'
        )
        guard_reload = installer.index("systemctl daemon-reload", mantis_guard_write)
        marker_write = installer.index(
            "Could not durably arm the update restart guard", guard_reload
        )
        stop_mantis = installer.index(
            'systemctl stop "${MANTIS_SERVICE_NAME}.service"', marker_write
        )
        stop = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"', marker_write
        )
        inactive_check = installer.index(
            'systemctl is-active --quiet "${SERVICE_NAME}.service"', stop
        )
        force_install = installer.index('"${UV}" tool install')
        verify = installer.index('"${AXOL}" tracker.ultimate.install', force_install)
        enable = installer.index('systemctl enable "${SERVICE_NAME}"', verify)
        token = installer.index("write_update_start_token", enable)
        start = installer.index('systemctl start "${SERVICE_NAME}"', token)
        restore_mantis = installer.index(
            'restore_mantis_for_candidate "${VERSION}"', start
        )
        promote = installer.index("    promote_pending_rollback", restore_mantis)
        self.assertLess(dataset_gate, pypi_check)
        self.assertLess(pypi_check, uv_install)
        self.assertLess(uv_install, uv_verify)
        self.assertLess(uv_verify, preflight)
        self.assertLess(uv_install, preflight)
        self.assertLess(preflight, guard_write)
        self.assertLess(guard_write, mantis_guard_write)
        self.assertLess(mantis_guard_write, guard_reload)
        self.assertLess(guard_reload, marker_write)
        self.assertLess(marker_write, stop_mantis)
        self.assertLess(stop_mantis, stop)
        self.assertLess(marker_write, stop)
        self.assertLess(stop, inactive_check)
        self.assertLess(inactive_check, force_install)
        self.assertLess(force_install, verify)
        self.assertLess(verify, enable)
        self.assertLess(enable, token)
        self.assertLess(token, start)
        self.assertLess(start, restore_mantis)
        self.assertLess(restore_mantis, promote)
        self.assertIn("MANTIS_SERVICE_SHOULD_RUN=1", installer)
        self.assertIn(
            'MANTIS_UPDATE_GUARD_DIR="/etc/systemd/system/${MANTIS_SERVICE_NAME}.service.d"',
            installer,
        )
        self.assertIn(
            '[ ! -L "${MANTIS_UPDATE_GUARD_DIR}" ]',
            installer,
        )
        self.assertIn(
            "ConditionPathExists=|!%s\\nConditionPathExists=|%s",
            installer,
        )
        self.assertIn(
            "ExecStartPre=/usr/bin/rm -f -- %s",
            installer,
        )
        self.assertIn("ExecStartPost=${BIN_DIR}/axol update-healthcheck", installer)
        self.assertIn("/usr/bin/mv -fT -- ${UPDATE_START_TOKEN}", installer)
        self.assertIn(
            "current Axol service and runtime were not changed",
            installer,
        )
        self.assertIn("one verifiable canonical wheel", installer)
        self.assertIn('"${RELEASE_WHEEL}[${EXTRAS}]"', installer)
        self.assertNotIn('"almond-axol[${EXTRAS}]==${VERSION}"', installer)
        self.assertIn('UV="${BIN_DIR}/uv"', installer)
        system_path = installer.index('PATH="/usr/sbin:/usr/bin:/sbin:/bin"')
        root_check = installer.index('if [ "$(id -u)" -ne 0 ]')
        local_entry_check = installer.index('UNSAFE_LOCAL_ENTRY="$(find -P')
        trusted_local_path = installer.index(
            'PATH="/usr/sbin:/usr/bin:/sbin:/bin:${BIN_DIR}"'
        )
        self.assertLess(system_path, root_check)
        self.assertLess(root_check, local_entry_check)
        self.assertLess(local_entry_check, trusted_local_path)
        self.assertNotIn(
            'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"',
            installer,
        )
        self.assertIn('"refs/tags/release-v*"', installer)
        self.assertNotIn('"refs/tags/v*"', installer)
        self.assertIn("AXOL_RELEASE_TAG", installer)
        self.assertIn('MIN_SAFE_RELEASE_VERSION="0.1.37"', installer)
        self.assertLess(installer.index("OLDEST_VERSION="), pypi_check)
        self.assertIn("hardened migration release is not published yet", installer)
        self.assertIn('"uv ${UV_VERSION}"|"uv ${UV_VERSION} "*', installer)
        self.assertNotIn('== "uv ${UV_VERSION}"*', installer)
        self.assertNotIn('!= "uv ${UV_VERSION}"*', installer)
        self.assertNotIn('UV="$(command -v uv', installer)
        self.assertIn('trusted_root_executable "${UV}"', installer)
        self.assertNotIn("astral.sh/uv/${UV_VERSION}/install.sh", installer)
        self.assertIn('find -P "${executable}" -maxdepth 0 -perm /022', installer)
        self.assertIn('"${UPDATE_WITH_ARGS[@]}"', installer)
        self.assertIn('20)\n            die "${UPDATE_PREFLIGHT_OUTPUT}"', installer)
        self.assertLess(installer.index("LEGACY_TORCH_STATUS"), force_install)
        self.assertIn("PyPI torch 2.10 is CPU-only on aarch64", installer)

    def test_installer_persists_migration_ack_identity_and_safe_lock(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()

        symlink_validation = installer.index("validate_local_symlink_target() {")
        local_path_enabled = installer.index(
            'PATH="/usr/sbin:/usr/bin:/sbin:/bin:${BIN_DIR}"'
        )
        self.assertLess(symlink_validation, local_path_enabled)
        self.assertIn('chain="$(namei -l -- "${link}")"', installer)
        self.assertIn("has a non-root-controlled symlink component", installer)
        self.assertIn("has a writable symlink-chain component", installer)
        self.assertIn('target="$(realpath -e -- "${link}")"', installer)
        self.assertIn("has writable target ancestry at ${current}", installer)
        self.assertIn(
            "Environment=PATH=/usr/sbin:/usr/bin:/sbin:/bin:${BIN_DIR}",
            installer,
        )

        dataset_gate = installer.index("Existing datasets were found under")
        lock_open = installer.index('exec {UPDATE_LOCK_FD}<>"${UPDATE_LOCK}"')
        acknowledgement = installer.index(
            'durable_replace "${LEGACY_ACK_TMP}" "${LEGACY_DATASET_ACK}"'
        )
        self.assertLess(dataset_gate, lock_open)
        self.assertLess(lock_open, acknowledgement)
        self.assertIn("LEGACY_DATASET_ACKED=1", installer[:dataset_gate])
        self.assertIn("legacy-datasets-cli-only=1", installer)

        lock_section = installer[
            installer.index("# Serialize curl-based installs") : installer.index(
                "# Keep the exact previously running uv environment"
            )
        ]
        self.assertIn('[ -f "/proc/$$/fd/${UPDATE_LOCK_FD}" ]', lock_section)
        self.assertIn("stat -Lc '%d:%i'", lock_section)
        self.assertIn("stat -c '%d:%i'", lock_section)
        self.assertNotIn(
            'durable_replace "${UPDATE_LOCK_TMP}" "${UPDATE_LOCK}"', lock_section
        )

        self.assertIn(
            'OPERATOR_USER_ENV_LINE="$(systemd_environment_line AXOL_OPERATOR_USER',
            installer,
        )
        self.assertIn("${OPERATOR_USER_ENV_LINE}", installer)
        self.assertIn('PERSISTED_OPERATOR_USER="${AXOL_OPERATOR_USER:-}"', installer)
        self.assertNotIn(
            'install -d -o root -g "${DATASET_GROUP}" -m 0750 /var/lib/almond-axol',
            installer,
        )
        self.assertIn(
            "install -d -o root -g root -m 0751 /var/lib/almond-axol",
            installer,
        )
        self.assertIn(
            'install -d -o root -g "${DATASET_GROUP}" -m 2750 '
            '"${SERVICE_DATASET_ROOT}"',
            installer,
        )

    def test_installer_fsyncs_guard_payloads_and_directory_entries(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        helper = installer[
            installer.index("durable_replace() {") : installer.index(
                "durable_remove() {"
            )
        ]
        payload_sync = helper.index('sync -- "${temporary}"')
        atomic_replace = helper.index('mv -fT -- "${temporary}" "${destination}"')
        directory_sync = helper.index('sync -- "${parent}"')
        self.assertLess(payload_sync, atomic_replace)
        self.assertLess(atomic_replace, directory_sync)

    def test_installer_accepts_only_hardened_release_tags(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        start = installer.index("is_hardened_release_tag() {")
        end = installer.index("\n}\n", start) + len("\n}\n")
        helper = installer[start:end]

        for tag, accepted in (
            ("release-v0.1.37", True),
            ("release-v1.2.3", True),
            ("v0.1.36", False),
            ("v0.1.37", False),
            ("release-v0.1.37rc1", False),
        ):
            with self.subTest(tag=tag):
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'{helper}\nis_hardened_release_tag "$1"',
                        "bash",
                        tag,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, accepted)

        guard_reload = installer.index(
            "systemctl daemon-reload",
            installer.index(
                'durable_replace "${MANTIS_GUARD_TMP}" "${MANTIS_UPDATE_GUARD_FILE}"'
            ),
        )
        marker_commit = installer.index(
            "Could not durably arm the update restart guard", guard_reload
        )
        stop_service = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"', marker_commit
        )
        mutate_runtime = installer.index('"${UV}" tool install')
        self.assertLess(marker_commit, stop_service)
        self.assertLess(marker_commit, mutate_runtime)
        self.assertIn("command -v sync >/dev/null 2>&1", installer)

        directory_helper = installer[
            installer.index("durable_install_directory() {") : installer.index(
                "durable_replace() {"
            )
        ]
        create_directory = directory_helper.index(
            'install -d -o "${owner}" -g "${group}" -m "${mode}" "${path}"'
        )
        persist_parent = directory_helper.index('sync -- "${parent}"')
        self.assertLess(create_directory, persist_parent)

        state_directory = installer.index(
            'durable_install_directory "/var/lib/almond-axol" root root 0751'
        )
        axol_dropin_directory = installer.index(
            'durable_install_directory "${UPDATE_GUARD_DIR}" root root 0755'
        )
        mantis_dropin_directory = installer.index(
            'durable_install_directory "${MANTIS_UPDATE_GUARD_DIR}" root root 0755'
        )
        first_guard_payload = installer.index('GUARD_TMP="$(mktemp', state_directory)
        cleanup_trap = installer.index(
            "trap cleanup_update_guard_temporaries EXIT", state_directory
        )
        cleanup_disarm = installer.index("trap - EXIT", first_guard_payload)
        self.assertLess(state_directory, axol_dropin_directory)
        self.assertLess(axol_dropin_directory, mantis_dropin_directory)
        self.assertLess(mantis_dropin_directory, cleanup_trap)
        self.assertLess(cleanup_trap, first_guard_payload)
        self.assertLess(first_guard_payload, cleanup_disarm)
        self.assertIn('rm -f -- "${GUARD_TMP}" || true', installer)
        self.assertIn('rm -f -- "${MANTIS_GUARD_TMP}" || true', installer)

        marker_helper = installer[
            installer.index("write_update_marker() {") : installer.index(
                '[ "$(uname -s)" = "Linux"'
            )
        ]
        self.assertIn('rm -f -- "${marker_tmp}" || true', marker_helper)

        remove_helper = installer[
            installer.index("durable_remove() {") : installer.index(
                '[ "$(uname -s)" = "Linux"'
            )
        ]
        self.assertLess(
            remove_helper.index('rm -f -- "${path}"'),
            remove_helper.index('sync -- "${parent}"'),
        )

    def test_installer_keeps_guard_until_one_shot_candidate_is_healthy(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        guard_reload = installer.index(
            "systemctl daemon-reload",
            installer.index(
                'durable_replace "${MANTIS_GUARD_TMP}" "${MANTIS_UPDATE_GUARD_FILE}"'
            ),
        )
        marker = installer.index(
            "Could not durably arm the update restart guard", guard_reload
        )
        enable = installer.index('systemctl enable "${SERVICE_NAME}"', marker)
        token = installer.index("write_update_start_token", enable)
        start = installer.index('systemctl start "${SERVICE_NAME}"', token)
        verify_marker = installer.index('[ ! -f "${UPDATE_MARKER}" ]', start)
        reject_failed_candidate = installer.index(
            'reject_candidate_after_health_failure "${VERSION}"', start
        )
        reject_start = installer.index("reject_candidate_after_failure() {")
        reject_end = installer.index("\n}\n", reject_start)
        rejection = installer[reject_start:reject_end]
        restore_marker = rejection.index("write_update_marker || marker_armed=0")
        clear_token = rejection.index(
            'durable_remove "${UPDATE_START_TOKEN}" || attempt_state_cleared=0'
        )
        disable_failed_service = rejection.index(
            'systemctl disable --now "${SERVICE_NAME}.service"'
        )
        verify_disabled = rejection.index('main_enable_state="$(systemctl is-enabled')
        self.assertLess(marker, token)
        self.assertLess(token, start)
        self.assertLess(start, verify_marker)
        self.assertLess(verify_marker, reject_failed_candidate)
        self.assertLess(restore_marker, clear_token)
        self.assertLess(clear_token, disable_failed_service)
        self.assertLess(disable_failed_service, verify_disabled)
        self.assertIn('[ "${marker_armed}" -ne 1 ]', rejection)
        self.assertIn('[ "${attempt_state_cleared}" -ne 1 ]', rejection)
        self.assertIn('[ "${main_active}" -ne 0 ]', rejection)
        self.assertIn('[ "${main_enabled}" -ne 0 ]', rejection)
        self.assertIn('[ "${mantis_active}" -ne 0 ]', rejection)
        self.assertIn("local main_active=1", rejection)
        self.assertIn("--property=ActiveState", rejection)
        self.assertIn('[ "${main_enable_state}" = "disabled" ]', rejection)
        # The installer never opens a pre-health gap by clearing the guard.
        transaction = installer[token:reject_failed_candidate]
        self.assertNotIn('durable_remove "${UPDATE_MARKER}"', transaction)

    def test_installer_retains_and_can_restore_exact_previous_runtime(self) -> None:
        installer_path = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        )
        installer = installer_path.read_text()

        syntax = subprocess.run(
            ["bash", "-n", str(installer_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        dataset_gate = installer.index("Existing datasets were found under")
        update_lock = installer.index('flock -n "${UPDATE_LOCK_FD}"')
        restore_dispatch = installer.index('if [ "${AXOL_RESTORE_PREVIOUS:-}" = "1" ]')
        release_resolution = installer.index('say "Resolving the latest release..."')
        self.assertLess(dataset_gate, update_lock)
        self.assertLess(update_lock, restore_dispatch)
        self.assertLess(restore_dispatch, release_resolution)

        prepare_start = installer.index("prepare_pending_rollback() {")
        prepare_end = installer.index("\n}\n", prepare_start)
        prepare = installer[prepare_start:prepare_end]
        publish_metadata = prepare.index('mv -T -- "${stage}" "${PENDING_ROLLBACK}"')
        self.assertGreater(publish_metadata, 0)
        self.assertNotIn(
            'mv -T -- "${canonical_tool}" "${PENDING_ROLLBACK}/tool"',
            prepare,
        )
        self.assertNotIn('durable_remove "${AXOL}"', prepare)
        self.assertIn('"version=${old_version}"', prepare)
        self.assertIn('"wrapper-kind=${wrapper_kind}"', prepare)

        retain_start = installer.index("retain_pending_rollback_tool() {")
        retain_end = installer.index("\n}\n", retain_start)
        retain = installer[retain_start:retain_end]
        retain_tool = retain.index(
            'mv -T -- "${canonical_tool}" "${PENDING_ROLLBACK}/tool"'
        )
        remove_wrapper = retain.index('durable_remove "${AXOL}"')
        self.assertLess(retain_tool, remove_wrapper)

        guard_reload = installer.index(
            "systemctl daemon-reload",
            installer.index(
                'durable_replace "${MANTIS_GUARD_TMP}" "${MANTIS_UPDATE_GUARD_FILE}"'
            ),
        )
        marker = installer.index(
            "Could not durably arm the update restart guard", guard_reload
        )
        prepare_call = installer.index("    prepare_pending_rollback", guard_reload)
        stop_mantis = installer.index(
            'systemctl stop "${MANTIS_SERVICE_NAME}.service"', marker
        )
        mantis_inactive = installer.index(
            'systemctl is-active --quiet "${MANTIS_SERVICE_NAME}.service"',
            stop_mantis,
        )
        stop = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"', marker
        )
        inactive = installer.index(
            'systemctl is-active --quiet "${SERVICE_NAME}.service"', stop
        )
        retain_call = installer.index("    retain_pending_rollback_tool", inactive)
        force_install = installer.index('"${UV}" tool install', retain_call)
        self.assertLess(guard_reload, prepare_call)
        self.assertLess(prepare_call, marker)
        self.assertLess(marker, stop)
        self.assertLess(stop_mantis, mantis_inactive)
        self.assertLess(mantis_inactive, retain_call)
        self.assertLess(stop, inactive)
        self.assertLess(inactive, retain_call)
        self.assertLess(retain_call, force_install)

        start = installer.index('systemctl start "${SERVICE_NAME}"', force_install)
        health_commit_check = installer.index('[ ! -f "${UPDATE_MARKER}" ]', start)
        restore_mantis = installer.index(
            'restore_mantis_for_candidate "${VERSION}"', health_commit_check
        )
        promote = installer.index("    promote_pending_rollback", restore_mantis)
        self.assertLess(health_commit_check, restore_mantis)
        self.assertLess(restore_mantis, promote)
        self.assertIn("AXOL_RESTORE_PREVIOUS=1", installer)
        self.assertIn(
            'mv -T -- "${slot}/tool" "${canonical_tool}"',
            installer,
        )

    def test_mantis_restart_is_part_of_candidate_commit_and_recovery(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()

        restore_start = installer.index("restore_mantis_for_candidate() {")
        restore_end = installer.index("\n}\n", restore_start)
        restore = installer[restore_start:restore_end]
        start = restore.index('systemctl start "${MANTIS_SERVICE_NAME}.service"')
        recovery_stable = restore.index("mantis_service_is_stable")
        stable = restore.index("mantis_service_is_stable", start)
        reject = restore.index(
            'reject_candidate_after_mantis_failure "${candidate_version}"',
            stable,
        )
        self.assertLess(recovery_stable, start)
        self.assertLess(start, stable)
        self.assertLess(stable, reject)

        stable_start = installer.index("mantis_service_is_stable() {")
        stable_end = installer.index("\n}\n", stable_start)
        stability_check = installer[stable_start:stable_end]
        self.assertIn("--property=MainPID", stability_check)
        self.assertIn(
            'systemctl is-active --quiet "${MANTIS_SERVICE_NAME}.service"',
            stability_check,
        )
        self.assertIn('[ "${stable_samples}" -ge 5 ]', stability_check)

        reject_start = installer.index("reject_candidate_after_failure() {")
        reject_end = installer.index("\n}\n", reject_start)
        rejection = installer[reject_start:reject_end]
        rearm = rejection.index("write_update_marker || marker_armed=0")
        stop_mantis = rejection.index('systemctl stop "${MANTIS_SERVICE_NAME}.service"')
        disable_axol = rejection.index(
            'systemctl disable --now "${SERVICE_NAME}.service"'
        )
        restore_command = rejection.index("AXOL_RESTORE_PREVIOUS=1")
        self.assertLess(rearm, stop_mantis)
        self.assertLess(rearm, disable_axol)
        self.assertLess(disable_axol, restore_command)
        self.assertNotIn("promote_pending_rollback", rejection)
        self.assertNotIn("safe_remove_rollback_path", rejection)

        recovered_restore = installer.index(
            'restore_mantis_for_candidate "${RECOVERED_CANDIDATE_VERSION}"'
        )
        recovered_promote = installer.index(
            "            promote_pending_rollback", recovered_restore
        )
        recovered_commit = installer.index(
            'durable_remove "${UPDATE_MARKER}"', recovered_restore
        )
        self.assertLess(recovered_restore, recovered_commit)
        self.assertLess(recovered_commit, recovered_promote)
        self.assertLess(recovered_restore, recovered_promote)

        final_health = installer.index('[ ! -f "${UPDATE_MARKER}" ]', recovered_promote)
        final_restore = installer.index(
            'restore_mantis_for_candidate "${VERSION}"', final_health
        )
        final_commit = installer.index(
            'durable_remove "${UPDATE_MARKER}"', final_restore
        )
        final_promote = installer.index("    promote_pending_rollback", final_restore)
        self.assertLess(final_health, final_restore)
        self.assertLess(final_restore, final_commit)
        self.assertLess(final_commit, final_promote)
        self.assertEqual(installer.count("promote_pending_rollback\n"), 2)

        recovered_exit = installer.index("            exit 0", recovered_promote)
        self.assertLess(recovered_promote, recovered_exit)
        self.assertLess(recovered_exit, final_health)

    def test_recovery_exits_same_release_but_continues_explicit_newer_release(
        self,
    ) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        recovery_start = installer.index(
            '            if [ -z "${AXOL_RELEASE_TAG:-}" ]'
        )
        recovery_end = installer.index("\n            fi", recovery_start) + len(
            "\n            fi"
        )
        decision = installer[recovery_start:recovery_end]
        harness = f"""
set -u
say() {{ :; }}
RECOVERED_CANDIDATE_VERSION=0.1.37
{decision}
printf continued
"""

        def run_decision(tag: str | None) -> subprocess.CompletedProcess[str]:
            environment = dict(os.environ)
            if tag is None:
                environment.pop("AXOL_RELEASE_TAG", None)
            else:
                environment["AXOL_RELEASE_TAG"] = tag
            return subprocess.run(
                ["bash", "-c", harness],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

        no_explicit_release = run_decision(None)
        same_release = run_decision("release-v0.1.37")
        newer_release = run_decision("release-v0.1.38")

        self.assertEqual(no_explicit_release.returncode, 0)
        self.assertEqual(no_explicit_release.stdout, "")
        self.assertEqual(same_release.returncode, 0)
        self.assertEqual(same_release.stdout, "")
        self.assertEqual(newer_release.returncode, 0)
        self.assertEqual(newer_release.stdout, "continued")

    def test_installer_allowlists_and_legacy_preserves_the_published_plugin(
        self,
    ) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        force_install = installer.index('"${UV}" tool install')

        self.assertIn('[[ "${update_requirement}" =~ ^lerobot_robot_axol==', installer)
        self.assertIn('UPDATE_WITH_ARGS+=(--with "${update_requirement}")', installer)
        self.assertLess(
            installer.index("legacy_plugin_update_requirement()"), force_install
        )
        self.assertLess(
            installer.index("legacy_ultimate_update_requirement()"), force_install
        )
        self.assertLess(
            installer.index(
                'legacy_plugin_update_requirement "${EXISTING_AXOL_PYTHON}"'
            ),
            force_install,
        )
        self.assertEqual(
            installer.count('legacy_ultimate_update_requirement "'),
            2,
        )
        self.assertIn(
            'legacy_ultimate_update_requirement "${LEGACY_PREFLIGHT_PYTHON}"',
            installer,
        )
        self.assertIn(
            'legacy_ultimate_update_requirement "${EXISTING_AXOL_PYTHON}"',
            installer,
        )
        self.assertIn(f'PYVUT_REF="{self._PYVUT_REF}"', installer)
        self.assertIn(
            'PYVUT_DEFAULT_WIFI_SHA256="fd64dd89b6dd61d06e91b1a5c913aa7fcae5ac2654903eb3f7e6dac8aeee2b67"',
            installer,
        )
        self.assertIn("direct or customized lerobot_robot_axol", installer)

    def test_installer_stops_before_mutation_and_stays_stopped_on_failure(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        stop = installer.index('systemctl disable --now "${SERVICE_NAME}.service"')
        force_install = installer.index('"${UV}" tool install')
        provision = installer.index('"${AXOL}" provision 2>&1')
        fatal = installer.index('if [ "${PROVISION_STATUS}" -ne 0 ]')
        start = installer.index('systemctl start "${SERVICE_NAME}"')
        self.assertLess(stop, force_install)
        self.assertLess(provision, fatal)
        self.assertLess(fatal, start)
        self.assertIn('PROVISION_STATUS="${PIPESTATUS[0]}"', installer)
        self.assertIn("service remains blocked and disabled", installer)

    def test_recovered_candidate_cannot_fall_back_to_legacy_info_health(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        function_start = installer.index("service_payload_has_version() {")
        health_start = installer.index("rollback_service_is_healthy() {")
        function_end = installer.index("\n}\n", health_start) + len("\n}\n")
        functions = installer[function_start:function_end]
        harness = f"""
set -u
SERVICE_NAME=axol
seq() {{ printf '%s\\n' 1 2 3 4 5; }}
sleep() {{ :; }}
systemctl() {{ printf '%s\\n' 4242; }}
curl() {{
    case "$*" in
        */api/health*) printf '%s' "${{HEALTH_PAYLOAD}}" ;;
        */api/info*) printf '%s' "${{INFO_PAYLOAD}}" ;;
        *) return 1 ;;
    esac
}}
{functions}
rollback_service_is_healthy 0.1.37 "${{ALLOW_LEGACY_INFO}}"
"""

        def health_result(
            health: dict[str, object], *, allow_legacy_info: bool
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["bash", "-c", harness],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "HEALTH_PAYLOAD": json.dumps(health),
                    "INFO_PAYLOAD": json.dumps({"version": "0.1.37"}),
                    "ALLOW_LEGACY_INFO": "1" if allow_legacy_info else "0",
                },
            )

        valid = health_result(
            {"ready": True, "version": "0.1.37", "pid": 4242},
            allow_legacy_info=False,
        )
        not_ready = health_result(
            {"ready": False, "version": "0.1.37", "pid": 4242},
            allow_legacy_info=False,
        )
        wrong_pid = health_result(
            {"ready": True, "version": "0.1.37", "pid": 9999},
            allow_legacy_info=False,
        )
        legacy = health_result(
            {"ready": False, "version": "0.1.37", "pid": 4242},
            allow_legacy_info=True,
        )

        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertNotEqual(not_ready.returncode, 0)
        self.assertNotEqual(wrong_pid.returncode, 0)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertIn('rollback_service_is_healthy "${old_version}" 1', installer)
        self.assertIn(
            'rollback_service_is_healthy "${RECOVERED_CANDIDATE_VERSION}" 0',
            installer,
        )

    def test_operator_state_writes_are_privilege_contained(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        state = installer[installer.index("# -- Shared operator state") :]
        self.assertIn('"${RUNUSER}" -u "${OPERATOR_USER}" -- "$@"', installer)
        self.assertIn(
            "operator_run cp -a --no-clobber --no-target-directory",
            state,
        )
        self.assertIn('operator_state_copy "${legacy_item}" "${operator_item}"', state)
        self.assertNotIn('cp -a -- "${legacy_item}" "${operator_item}"', state)
        self.assertIn('operator_run chmod 2775 -- "${ALMOND_STATE_HOME}"', state)
        self.assertNotIn(
            'chown -hR "${OPERATOR_USER}:${OPERATOR_GROUP}" "${ALMOND_STATE_HOME}"',
            state,
        )
        self.assertIn(
            'install -d -o root -g "${DATASET_GROUP}" -m 2750 '
            '"${SERVICE_DATASET_ROOT}"',
            installer,
        )


class SelfUpdateSafetyBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def test_existing_incomplete_marker_blocks_launches_on_process_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory, "update-incomplete")
            marker.write_text("incomplete\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(
                    update,
                    "installed_origin",
                    return_value=("https://repo", "abc"),
                ),
                patch.object(update, "installed_version", return_value="0.1.34"),
                patch.object(update, "installed_commit", return_value="abc"),
            ):
                updater = update.SelfUpdater(lambda: True)

        self.assertTrue(updater.launches_blocked)
        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertIn("previous update", updater._error or "")  # noqa: SLF001

    def test_verified_candidate_commit_clears_only_its_in_memory_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            verifying.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.37"),
                patch.object(update, "installed_commit", return_value="abc"),
            ):
                updater = update.SelfUpdater(lambda: True)
                self.assertTrue(updater.launches_blocked)
                verifying.unlink()
                marker.unlink()
                self.assertFalse(updater.launches_blocked)
                self.assertEqual(updater._state, "idle")  # noqa: SLF001

    def test_deleting_generic_failed_marker_does_not_revive_old_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.36"),
                patch.object(update, "installed_commit", return_value="abc"),
            ):
                updater = update.SelfUpdater(lambda: True)
                marker.unlink()
                self.assertTrue(updater.launches_blocked)

    async def test_start_raises_launch_barrier_before_background_task_runs(
        self,
    ) -> None:
        updater = _updater()
        updater._state = "idle"  # noqa: SLF001
        updater._launches_blocked = False  # noqa: SLF001
        gate = asyncio.Event()

        async def run_update() -> None:
            await gate.wait()

        updater._run_update = run_update  # type: ignore[method-assign]  # noqa: SLF001
        with patch.object(update.shutil, "which", return_value="/usr/bin/uv"):
            started, error = updater.start()

        self.assertTrue(started)
        self.assertIsNone(error)
        self.assertTrue(updater.launches_blocked)
        self.assertTrue(updater.maintenance_active)
        gate.set()
        assert updater._update_task is not None  # noqa: SLF001
        await updater._update_task  # noqa: SLF001

    async def test_retry_preflight_failure_preserves_inherited_durable_barrier(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.36"),
                patch.object(update, "installed_commit", return_value="abc"),
                patch.object(update.shutil, "which", return_value="/usr/bin/uv"),
                patch.object(
                    update,
                    "release_update_requirements",
                    return_value=([], "update preflight failed safely"),
                ),
            ):
                updater = update.SelfUpdater(lambda: True)
                updater._remote_tag = "release-v0.1.37"  # noqa: SLF001
                updater._remote_version = "0.1.37"  # noqa: SLF001

                started, error = updater.start()
                self.assertTrue(started)
                self.assertIsNone(error)
                assert updater._update_task is not None  # noqa: SLF001
                await updater._update_task  # noqa: SLF001

                self.assertTrue(marker.exists())
                self.assertTrue(updater.launches_blocked)
                self.assertEqual(updater._state, "error")  # noqa: SLF001
                self.assertEqual(  # noqa: SLF001
                    updater._error, "update preflight failed safely"
                )

    async def test_healthy_interrupted_candidate_can_retry_same_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            marker.chmod(0o600)
            verifying.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.37"),
                patch.object(update, "installed_commit", return_value="abc"),
                patch.object(update.shutil, "which", return_value="/usr/bin/uv"),
            ):
                updater = update.SelfUpdater(lambda: True)
                updater._remote_tag = "release-v0.1.37"  # noqa: SLF001
                updater._remote_version = "0.1.37"  # noqa: SLF001
                updater._schedule_remote_refresh = Mock()  # type: ignore[method-assign]  # noqa: SLF001
                updater._run_update = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

                # ExecStartPost proved Axol healthy; the installer crashed before
                # completing the shared Mantis/rollback transaction.
                verifying.unlink()

                status = await updater.status()
                self.assertTrue(status["updateAvailable"])
                self.assertTrue(status["idle"])

                started, error = updater.start()
                self.assertTrue(started)
                self.assertIsNone(error)
                assert updater._update_task is not None  # noqa: SLF001
                await updater._update_task  # noqa: SLF001

    def test_generic_failed_marker_cannot_authorize_same_release_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            marker.chmod(0o600)
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.37"),
                patch.object(update, "installed_commit", return_value="abc"),
                patch.object(update.shutil, "which", return_value="/usr/bin/uv"),
            ):
                updater = update.SelfUpdater(lambda: True)
                updater._remote_tag = "release-v0.1.37"  # noqa: SLF001
                updater._remote_version = "0.1.37"  # noqa: SLF001

                started, error = updater.start()

        self.assertFalse(started)
        self.assertEqual(error, "no update available")

    async def test_new_release_namespace_is_resolved_with_legacy_history(self) -> None:
        output = b"\n".join(
            (
                b"a refs/tags/v0.1.35",
                b"b refs/tags/release-v0.1.36",
                b"c refs/tags/release-v0.1.37^{}",
                b"d refs/tags/release-v0.1.37",
                b"e refs/tags/release-v0.1.37rc1",
                b"f refs/tags/v0.1.99",
            )
        )
        create = AsyncMock(return_value=_Process(output=output))
        updater = _updater()

        with patch.object(asyncio, "create_subprocess_exec", create):
            latest = await updater._resolve_latest_release("https://repo")  # noqa: SLF001

        self.assertEqual(latest, ("release-v0.1.37", "0.1.37"))
        self.assertEqual(
            create.await_args.args[:6],
            (
                "git",
                "ls-remote",
                "--tags",
                "https://repo",
                "refs/tags/v*",
                "refs/tags/release-v*",
            ),
        )
        self.assertEqual(update.parse_version("v0.1.35"), (0, 1, 35))
        self.assertEqual(update.parse_version("release-v0.1.36"), (0, 1, 36))

    def test_exact_pypi_release_must_have_one_verifiable_pure_wheel(self) -> None:
        filename = "almond_axol-0.1.35-py3-none-any.whl"
        available = {
            "info": {"name": "almond-axol", "version": "0.1.35"},
            "urls": [
                {
                    "packagetype": "bdist_wheel",
                    "yanked": False,
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/packages/aa/{filename}",
                    "digests": {"sha256": "a" * 64},
                    "size": 1234,
                }
            ],
        }
        with patch.object(
            update.urllib.request,
            "urlopen",
            return_value=_HTTPResponse(available),
        ) as open_url:
            self.assertTrue(update._release_available_on_pypi("0.1.35"))  # noqa: SLF001
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://pypi.org/pypi/almond-axol/0.1.35/json",
        )

        invalid_artifacts = (
            [{**available["urls"][0], "yanked": True}],
            [
                {
                    **available["urls"][0],
                    "url": f"https://example.invalid/{filename}",
                }
            ],
            [{**available["urls"][0], "digests": {"sha256": "bad"}}],
            [available["urls"][0], dict(available["urls"][0])],
        )
        for artifacts in invalid_artifacts:
            with (
                self.subTest(artifacts=artifacts),
                patch.object(
                    update.urllib.request,
                    "urlopen",
                    return_value=_HTTPResponse({**available, "urls": artifacts}),
                ),
            ):
                self.assertFalse(  # noqa: SLF001
                    update._release_available_on_pypi("0.1.35")
                )

    def test_target_wheel_installer_is_hash_verified_and_root_staged(self) -> None:
        payload = _wheel_payload()
        artifact = _release_wheel(payload)
        with tempfile.TemporaryDirectory() as directory:
            worker_root = Path(directory) / "update-workers"
            worker_root.mkdir(mode=0o700)
            with (
                patch.object(update, "_UPDATE_WORKER_ROOT", worker_root),
                patch.object(update, "_validate_update_worker_root"),
                patch.object(update, "_validate_release_stage"),
                patch.object(update.os, "chown"),
                patch.object(update.os, "fchown"),
                patch.object(
                    update.urllib.request,
                    "urlopen",
                    return_value=_ByteHTTPResponse(payload),
                ),
            ):
                staged = update._stage_release(artifact, "0.1.37")  # noqa: SLF001

            self.assertEqual(
                staged.installer.read_bytes(),
                b"#!/usr/bin/env bash\necho verified\n",
            )
            self.assertEqual(stat.S_IMODE(staged.installer.stat().st_mode), 0o700)
            self.assertEqual(staged.wheel.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(staged.wheel.stat().st_mode), 0o600)
            self.assertEqual(staged.sha256, hashlib.sha256(payload).hexdigest())

    def test_target_wheel_digest_or_identity_mismatch_is_never_staged(self) -> None:
        cases = (
            (_wheel_payload(), "0" * 64),
            (
                _wheel_payload(name="operator-controlled"),
                hashlib.sha256(_wheel_payload(name="operator-controlled")).hexdigest(),
            ),
        )
        for payload, digest in cases:
            with self.subTest(digest=digest):
                artifact = update._ReleaseWheel(  # noqa: SLF001
                    "almond_axol-0.1.37-py3-none-any.whl",
                    "https://files.pythonhosted.org/packages/aa/almond_axol-0.1.37-py3-none-any.whl",
                    digest,
                    len(payload),
                )
                with tempfile.TemporaryDirectory() as directory:
                    worker_root = Path(directory) / "update-workers"
                    worker_root.mkdir(mode=0o700)
                    with (
                        patch.object(update, "_UPDATE_WORKER_ROOT", worker_root),
                        patch.object(update, "_validate_update_worker_root"),
                        patch.object(update, "_validate_release_stage"),
                        patch.object(update.os, "chown"),
                        patch.object(update.os, "fchown"),
                        patch.object(
                            update.urllib.request,
                            "urlopen",
                            return_value=_ByteHTTPResponse(payload),
                        ),
                        self.assertRaises(ValueError),
                    ):
                        update._stage_release(artifact, "0.1.37")  # noqa: SLF001
                    self.assertEqual(list(worker_root.iterdir()), [])

    async def test_missing_pypi_release_does_not_arm_or_mutate(self) -> None:
        updater = _updater()
        updater._release_wheel = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._provision = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock()

        with (
            patch.object(
                update, "release_update_requirements", return_value=([], None)
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        updater._arm_durable_update_guard.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        create.assert_not_awaited()
        self.assertFalse(updater.launches_blocked)
        self.assertIn("verifiable pure-Python wheel", updater._error or "")  # noqa: SLF001
        self.assertIn("release publishing", updater._error or "")  # noqa: SLF001

    async def test_unmanaged_service_never_launches_update_worker(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value="not the managed service"
        )
        create = AsyncMock()

        with (
            patch.object(
                update, "release_update_requirements", return_value=([], None)
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        create.assert_not_awaited()
        self.assertFalse(updater.launches_blocked)
        self.assertEqual(updater._error, "not the managed service")  # noqa: SLF001

    def test_durable_guard_is_live_before_marker_and_disable(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        events: list[tuple[str, object]] = []
        mantis_stopped = False

        def systemctl(*args: str) -> Mock:
            nonlocal mantis_stopped
            events.append(("systemctl", args))
            if args == ("stop", update._MANTIS_SERVICE_NAME):  # noqa: SLF001
                mantis_stopped = True
            if (
                args == ("is-active", "--quiet", update._MANTIS_SERVICE_NAME)  # noqa: SLF001
                and mantis_stopped
            ):
                return Mock(returncode=3, stdout="inactive\n")
            if args[:2] == ("show", "--property=MainPID"):
                return Mock(returncode=0, stdout=str(os.getpid()))
            if args[:2] == ("show", "--property=LoadState"):
                return Mock(returncode=0, stdout="loaded\n")
            if args[:2] == ("show", "--property=DropInPaths"):
                dropin = (
                    update._MANTIS_UPDATE_GUARD_DROPIN  # noqa: SLF001
                    if args[-1] == update._MANTIS_SERVICE_NAME  # noqa: SLF001
                    else update._UPDATE_GUARD_DROPIN  # noqa: SLF001
                )
                return Mock(
                    returncode=0,
                    stdout=str(dropin),
                )
            return Mock(returncode=0, stdout="")

        def write(path: Path, _content: str, *, mode: int) -> None:
            events.append(("write", (path, mode)))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.os, "geteuid", return_value=0),
            patch.object(update.Path, "is_file", return_value=True),
            patch.dict(os.environ, update._MANAGED_UPDATE_ENV, clear=False),  # noqa: SLF001
            patch.object(update, "_write_durable_root_file", side_effect=write),
        ):
            error = update.SelfUpdater._arm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIsNone(error)
        reload_index = events.index(("systemctl", ("daemon-reload",)))
        mantis_dropin_index = events.index(
            (
                "write",
                (update._MANTIS_UPDATE_GUARD_DROPIN, 0o644),  # noqa: SLF001
            )
        )
        marker_index = events.index(
            ("write", (update._UPDATE_GUARD_MARKER, 0o600))  # noqa: SLF001
        )
        stop_mantis_index = events.index(
            ("systemctl", ("stop", update._MANTIS_SERVICE_NAME))  # noqa: SLF001
        )
        disable_index = events.index(
            ("systemctl", ("disable", update._SERVICE_NAME))  # noqa: SLF001
        )
        self.assertLess(mantis_dropin_index, reload_index)
        self.assertLess(reload_index, marker_index)
        self.assertLess(marker_index, stop_mantis_index)
        self.assertLess(stop_mantis_index, disable_index)
        self.assertLess(marker_index, disable_index)
        self.assertTrue(updater._mantis_restore_requested)  # noqa: SLF001
        self.assertTrue(updater._mantis_enable_requested)  # noqa: SLF001

    def test_durable_guard_rejects_an_unmanaged_uv_layout_before_mutation(self) -> None:
        updater = _updater()
        updater._systemctl = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        environment = dict(update._MANAGED_UPDATE_ENV)  # noqa: SLF001
        environment["UV_TOOL_DIR"] = "/root/.local/share/uv/tools"
        write = Mock()

        with (
            patch.object(update.os, "geteuid", return_value=0),
            patch.object(update.Path, "is_file", return_value=True),
            patch.dict(os.environ, environment, clear=True),
            patch.object(update, "_write_durable_root_file", write),
        ):
            error = update.SelfUpdater._arm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIn("hosted installer layout", error or "")
        self.assertIn("UV_TOOL_DIR", error or "")
        updater._systemctl.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        write.assert_not_called()

    def test_durable_guard_allows_an_absent_optional_mantis_service(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        writes: list[Path] = []

        def systemctl(*args: str) -> Mock:
            if args[:2] == ("show", "--property=MainPID"):
                return Mock(returncode=0, stdout=str(os.getpid()))
            if args[:2] == ("show", "--property=LoadState"):
                return Mock(returncode=0, stdout="not-found\n")
            if args[:2] == ("show", "--property=DropInPaths"):
                return Mock(
                    returncode=0,
                    stdout=str(update._UPDATE_GUARD_DROPIN),  # noqa: SLF001
                )
            return Mock(returncode=0, stdout="")

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.os, "geteuid", return_value=0),
            patch.object(
                update.shutil,
                "which",
                side_effect=lambda name: (
                    "/usr/bin/systemctl"
                    if name == "systemctl"
                    else update._MANAGED_UV_EXECUTABLE  # noqa: SLF001
                ),
            ),
            patch.dict(os.environ, update._MANAGED_UPDATE_ENV, clear=False),  # noqa: SLF001
            patch.object(
                update,
                "_write_durable_root_file",
                side_effect=lambda path, *_args, **_kwargs: writes.append(path),
            ),
        ):
            error = update.SelfUpdater._arm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIsNone(error)
        self.assertIn(update._MANTIS_UPDATE_GUARD_DROPIN, writes)  # noqa: SLF001
        self.assertFalse(updater._mantis_restore_requested)  # noqa: SLF001
        self.assertFalse(updater._mantis_enable_requested)  # noqa: SLF001
        self.assertNotIn(
            call("stop", update._MANTIS_SERVICE_NAME),  # noqa: SLF001
            updater._systemctl.call_args_list,  # type: ignore[attr-defined]
        )

    def test_verified_update_enables_before_removing_marker(self) -> None:
        updater = _updater()
        events: list[tuple[str, object]] = []

        def systemctl(*args: str) -> Mock:
            events.append(("systemctl", args))
            return Mock(returncode=0, stdout="enabled")

        def remove(path: Path) -> None:
            events.append(("remove", path))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_remove_durable_file", side_effect=remove),
        ):
            error = update.SelfUpdater._disarm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIsNone(error)
        enable_index = events.index(
            ("systemctl", ("enable", update._SERVICE_NAME))  # noqa: SLF001
        )
        remove_index = events.index(
            ("remove", update._UPDATE_GUARD_MARKER)  # noqa: SLF001
        )
        self.assertLess(enable_index, remove_index)

    def test_verified_update_restores_mantis_before_axol_process_exits(self) -> None:
        updater = _updater()
        updater._mantis_restore_requested = True  # noqa: SLF001
        updater._mantis_enable_requested = True  # noqa: SLF001
        events: list[tuple[str, object]] = []

        def systemctl(*args: str) -> Mock:
            events.append(("systemctl", args))
            if args[:2] == ("show", "--property=MainPID"):
                return Mock(returncode=0, stdout="4242\n")
            return Mock(returncode=0, stdout="enabled")

        def remove(path: Path) -> None:
            events.append(("remove", path))

        def write_mantis_token(version: str) -> None:
            events.append(("write-mantis-token", version))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_remove_durable_file", side_effect=remove),
            patch.object(
                update, "_write_mantis_start_token", side_effect=write_mantis_token
            ),
            patch.object(update.Path, "exists", return_value=False),
            patch.object(update.Path, "is_symlink", return_value=False),
            patch.object(update.time, "sleep"),
        ):
            error = update.SelfUpdater._disarm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIsNone(error)
        remove_index = events.index(
            ("remove", update._UPDATE_GUARD_MARKER)  # noqa: SLF001
        )
        enable_index = events.index(
            ("systemctl", ("enable", update._MANTIS_SERVICE_NAME))  # noqa: SLF001
        )
        start_index = events.index(
            ("systemctl", ("start", update._MANTIS_SERVICE_NAME))  # noqa: SLF001
        )
        active_index = events.index(
            (
                "systemctl",
                ("is-active", "--quiet", update._MANTIS_SERVICE_NAME),  # noqa: SLF001
            )
        )
        self.assertLess(enable_index, remove_index)
        self.assertLess(start_index, remove_index)
        self.assertLess(start_index, active_index)
        self.assertIsNone(updater._mantis_restore_requested)  # noqa: SLF001
        self.assertIsNone(updater._mantis_enable_requested)  # noqa: SLF001

    def test_mantis_stability_rejects_main_pid_churn(self) -> None:
        updater = _updater()
        pids = iter(("101\n", "101\n", "202\n", "202\n"))

        def systemctl(*args: str) -> Mock:
            if args[:2] == ("show", "--property=MainPID"):
                return Mock(returncode=0, stdout=next(pids))
            return Mock(returncode=0, stdout="active\n")

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update, "_MANTIS_STABILITY_ATTEMPTS", 4),
            patch.object(update, "_MANTIS_STABLE_SAMPLES", 3),
            patch.object(update.time, "sleep") as sleep,
        ):
            stable = update.SelfUpdater._mantis_service_is_stable(updater)  # noqa: SLF001

        self.assertFalse(stable)
        self.assertEqual(sleep.call_count, 4)

    def test_mantis_restore_failure_rearms_guard_and_stops_helper(self) -> None:
        updater = _updater()
        updater._mantis_restore_requested = True  # noqa: SLF001
        updater._mantis_enable_requested = True  # noqa: SLF001
        events: list[tuple[str, object]] = []

        def systemctl(*args: str) -> Mock:
            events.append(("systemctl", args))
            if args == ("start", update._MANTIS_SERVICE_NAME):  # noqa: SLF001
                return Mock(returncode=1, stdout="")
            return Mock(returncode=0, stdout="enabled")

        def write(path: Path, content: str, *, mode: int) -> None:
            events.append(("write", (path, content, mode)))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_remove_durable_file"),
            patch.object(update, "_write_durable_root_file", side_effect=write),
            patch.object(update, "_write_mantis_start_token"),
            patch.object(update.Path, "exists", return_value=False),
            patch.object(update.Path, "is_symlink", return_value=False),
        ):
            error = update.SelfUpdater._disarm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIn("axol-mantis.service could not be started", error or "")
        marker_index = events.index(
            (
                "write",
                (update._UPDATE_GUARD_MARKER, "target-version=0.1.36\n", 0o600),  # noqa: SLF001
            )
        )
        stop_index = events.index(
            ("systemctl", ("stop", update._MANTIS_SERVICE_NAME))  # noqa: SLF001
        )
        disable_index = events.index(
            ("systemctl", ("disable", update._SERVICE_NAME))  # noqa: SLF001
        )
        self.assertLess(marker_index, stop_index)
        self.assertLess(stop_index, disable_index)
        self.assertNotIn(
            ("systemctl", ("disable", update._MANTIS_SERVICE_NAME)),  # noqa: SLF001
            events,
        )

    def test_failed_marker_commit_rearms_and_contains_both_services(self) -> None:
        updater = _updater()
        updater._mantis_restore_requested = True  # noqa: SLF001
        updater._mantis_enable_requested = True  # noqa: SLF001
        events: list[tuple[str, object]] = []

        def systemctl(*args: str) -> Mock:
            events.append(("systemctl", args))
            if args[:2] == ("show", "--property=MainPID"):
                return Mock(returncode=0, stdout="4242\n")
            if args[:2] == ("show", "--property=LoadState"):
                return Mock(
                    returncode=0,
                    stdout="LoadState=loaded\nActiveState=inactive\n",
                )
            if args[:2] == ("show", "--property=UnitFileState"):
                return Mock(returncode=0, stdout="disabled\n")
            return Mock(returncode=0, stdout="enabled\n")

        def remove(path: Path) -> None:
            events.append(("remove", path))
            if path == update._UPDATE_GUARD_MARKER:  # noqa: SLF001
                raise OSError("directory fsync failed")

        def write(path: Path, content: str, *, mode: int) -> None:
            events.append(("write", (path, content, mode)))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update.Path, "exists", return_value=False),
            patch.object(update.Path, "is_symlink", return_value=False),
            patch.object(update, "_write_mantis_start_token"),
            patch.object(update, "_remove_durable_file", side_effect=remove),
            patch.object(update, "_write_durable_root_file", side_effect=write),
            patch.object(update.time, "sleep"),
        ):
            error = update.SelfUpdater._disarm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIn("could not durably commit", error or "")
        self.assertIn("services remain update-guarded", error or "")
        failed_commit = events.index(
            ("remove", update._UPDATE_GUARD_MARKER)  # noqa: SLF001
        )
        rearm = events.index(
            (
                "write",
                (update._UPDATE_GUARD_MARKER, "target-version=0.1.36\n", 0o600),  # noqa: SLF001
            )
        )
        stop = events.index(
            ("systemctl", ("stop", update._MANTIS_SERVICE_NAME))  # noqa: SLF001
        )
        disable = events.index(
            ("systemctl", ("disable", update._SERVICE_NAME))  # noqa: SLF001
        )
        self.assertLess(failed_commit, rearm)
        self.assertLess(rearm, stop)
        self.assertLess(stop, disable)

    async def test_startup_provision_holds_launch_barrier_until_complete(self) -> None:
        updater = _updater()
        updater._state = "idle"  # noqa: SLF001
        updater._launches_blocked = False  # noqa: SLF001
        gate = asyncio.Event()

        async def provision() -> None:
            await gate.wait()
            return None

        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._provision = AsyncMock(side_effect=provision)  # type: ignore[method-assign]  # noqa: SLF001
        await updater.ensure_provisioned()
        await asyncio.sleep(0)
        self.assertTrue(updater.maintenance_active)
        self.assertTrue(updater.launches_blocked)

        gate.set()
        assert updater._provision_task is not None  # noqa: SLF001
        await updater._provision_task  # noqa: SLF001
        self.assertFalse(updater.maintenance_active)
        self.assertFalse(updater.launches_blocked)

    async def test_startup_provision_skips_manual_root_release_serve(self) -> None:
        updater = _updater()
        updater._state = "idle"  # noqa: SLF001
        updater._launches_blocked = False  # noqa: SLF001
        updater._managed_service_error = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value="self-update must run from the managed axol.service process"
        )
        updater._provision = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

        await updater.ensure_provisioned()

        self.assertTrue(updater._provision_started)  # noqa: SLF001
        self.assertIsNone(updater._provision_task)  # noqa: SLF001
        self.assertFalse(updater.maintenance_active)
        self.assertFalse(updater.launches_blocked)
        self.assertEqual(updater._state, "idle")  # noqa: SLF001
        updater._arm_durable_update_guard.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001


class SelfUpdaterShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovered_candidate_worker_exit_commits_in_place(self) -> None:
        status = Mock(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=inactive\nSubState=dead\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "update-incomplete"
            verifying = root / "update-verifying"
            token = root / "update-start-once"
            marker.write_text("target-version=0.1.37\n")
            verifying.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_VERIFYING_MARKER", verifying),
                patch.object(update, "_UPDATE_START_TOKEN", token),
                patch.object(update, "_UPDATE_MONITOR_INTERVAL_S", 0.0),
                patch.object(update, "installed_origin", return_value=("repo", "abc")),
                patch.object(update, "installed_version", return_value="0.1.37"),
                patch.object(update, "installed_commit", return_value="abc"),
            ):
                updater = update.SelfUpdater(lambda: True)
                updater._state = "updating"  # noqa: SLF001
                updater._phase = "restarting"  # noqa: SLF001
                updater._systemctl = Mock(return_value=status)  # type: ignore[method-assign]  # noqa: SLF001

                # Axol health and the resumed installer's shared commit both
                # completed without replacing this already-running candidate.
                verifying.unlink()
                marker.unlink()
                await updater._monitor_update_worker(  # noqa: SLF001
                    "axol-update-test.service"
                )

        self.assertEqual(updater._state, "idle")  # noqa: SLF001
        self.assertIsNone(updater._error)  # noqa: SLF001
        self.assertIsNone(updater._phase)  # noqa: SLF001
        self.assertFalse(updater.launches_blocked)

    async def test_confirmed_handoff_terminal_exit_stays_fail_closed(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._systemctl = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=Mock(
                returncode=0,
                stdout=("LoadState=loaded\nActiveState=inactive\nSubState=dead\n"),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "update-incomplete"
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_MONITOR_INTERVAL_S", 0.0),
                patch.object(
                    update,
                    "release_update_requirements",
                    return_value=([], None),
                ),
                patch.object(update.Path, "is_file", return_value=True),
                patch.object(
                    asyncio,
                    "create_subprocess_exec",
                    AsyncMock(return_value=_Process()),
                ),
            ):
                await updater._run_update()  # noqa: SLF001
                assert updater._update_monitor_task is not None  # noqa: SLF001
                await updater._update_monitor_task  # noqa: SLF001

        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertTrue(updater.launches_blocked)
        updater._systemctl.assert_called_once()  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_failed_handoff_without_marker_still_stays_fail_closed(self) -> None:
        updater = _updater()
        status = Mock(
            returncode=0,
            stdout="LoadState=loaded\nActiveState=failed\nSubState=failed\n",
        )
        updater._systemctl = Mock(return_value=status)  # type: ignore[method-assign]  # noqa: SLF001

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "update-incomplete"
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_MONITOR_INTERVAL_S", 0.0),
            ):
                await updater._monitor_update_worker("axol-update-test.service")  # noqa: SLF001

        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertTrue(updater.launches_blocked)
        self.assertIn("worker stopped before replacing", updater._error or "")  # noqa: SLF001

    async def test_failed_handoff_with_durable_marker_stays_fail_closed(self) -> None:
        updater = _updater()
        status = Mock(
            returncode=0,
            stdout="LoadState=not-found\nActiveState=inactive\nSubState=dead\n",
        )
        updater._systemctl = Mock(return_value=status)  # type: ignore[method-assign]  # noqa: SLF001

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "update-incomplete"
            marker.write_text("target-version=0.1.37\n")
            with (
                patch.object(update, "_UPDATE_GUARD_MARKER", marker),
                patch.object(update, "_UPDATE_MONITOR_INTERVAL_S", 0.0),
            ):
                await updater._monitor_update_worker("axol-update-test.service")  # noqa: SLF001

        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertTrue(updater.launches_blocked)
        self.assertIn("worker stopped before replacing", updater._error or "")  # noqa: SLF001

    async def test_shutdown_reaps_blocking_provision_and_retains_guard(self) -> None:
        updater = _updater()
        process = _BlockingProcess()
        updater._provision_task = asyncio.create_task(  # noqa: SLF001
            updater._run_startup_provision()  # noqa: SLF001
        )

        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            await process.started.wait()
            await updater.shutdown()
            await updater.shutdown()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertTrue(updater._provision_task.done())  # noqa: SLF001
        updater._arm_durable_update_guard.assert_called_once_with()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._disarm_durable_update_guard.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertTrue(updater.launches_blocked)

    async def test_shutdown_escalates_blocking_child_and_reaps_it(self) -> None:
        updater = _updater()
        process = _BlockingProcess(terminate_exits=False)
        updater._provision_task = asyncio.create_task(  # noqa: SLF001
            updater._run_startup_provision()  # noqa: SLF001
        )

        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_PROCESS_TERMINATE_TIMEOUT_S", 0.0),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            await process.started.wait()
            await updater.shutdown()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertTrue(updater._provision_task.done())  # noqa: SLF001
        updater._disarm_durable_update_guard.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_cancelled_shutdown_still_finishes_shared_drain(self) -> None:
        updater = _updater()
        process = _BlockingProcess(terminate_exits=False)
        updater._provision_task = asyncio.create_task(  # noqa: SLF001
            updater._run_startup_provision()  # noqa: SLF001
        )

        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_PROCESS_TERMINATE_TIMEOUT_S", 0.01),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            await process.started.wait()
            shutdown = asyncio.create_task(updater.shutdown())
            await asyncio.sleep(0)
            shutdown.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await shutdown

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertTrue(updater._shutdown_task.done())  # noqa: SLF001
        self.assertTrue(updater._provision_task.done())  # noqa: SLF001

    async def test_shutdown_stops_unconfirmed_update_unit(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._systemctl = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=Mock(
                returncode=0,
                stdout="LoadState=not-found\nActiveState=inactive\n",
            )
        )
        process = _BlockingProcess()

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            updater._update_task = asyncio.create_task(updater._run_update())  # noqa: SLF001
            await process.started.wait()
            await updater.shutdown()

        unit_arg = next(
            arg
            for arg in updater._systemctl.call_args.args  # type: ignore[attr-defined]  # noqa: SLF001
            if arg.startswith("axol-update-")
        )
        self.assertTrue(unit_arg.endswith(".service"))
        self.assertEqual(
            updater._systemctl.call_args_list,  # type: ignore[attr-defined]  # noqa: SLF001
            [
                call("stop", unit_arg),
                call(
                    "show",
                    "--property=LoadState",
                    "--property=ActiveState",
                    unit_arg,
                ),
            ],
        )
        self.assertEqual(process.terminate_calls, 1)
        self.assertTrue(updater._update_task.done())  # noqa: SLF001

    async def test_shutdown_preserves_confirmed_systemd_handoff(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._systemctl = Mock(return_value=Mock(returncode=0))  # type: ignore[method-assign]  # noqa: SLF001

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(
                asyncio,
                "create_subprocess_exec",
                AsyncMock(return_value=_Process()),
            ),
        ):
            updater._update_task = asyncio.create_task(updater._run_update())  # noqa: SLF001
            await updater._update_task  # noqa: SLF001
            await updater.shutdown()

        self.assertTrue(updater._update_worker_handed_off)  # noqa: SLF001
        self.assertTrue(updater._update_monitor_task.done())  # noqa: SLF001
        updater._systemctl.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001


class SelfUpdateWorkerTests(unittest.IsolatedAsyncioTestCase):
    def test_operator_identity_is_validated_from_the_service_environment(self) -> None:
        account = Mock(pw_uid=1000, pw_name="robot")
        with (
            patch.dict(os.environ, {"AXOL_OPERATOR_USER": "robot"}, clear=False),
            patch.object(update.pwd, "getpwnam", return_value=account),
        ):
            self.assertEqual(
                update.SelfUpdater._operator_user_for_update(),  # noqa: SLF001
                ("robot", None),
            )

        with patch.dict(os.environ, {}, clear=True):
            operator, error = update.SelfUpdater._operator_user_for_update()  # noqa: SLF001
        self.assertIsNone(operator)
        self.assertIn("persisted", error or "")

        with patch.dict(os.environ, {"AXOL_OPERATOR_USER": "root"}, clear=True):
            self.assertEqual(
                update.SelfUpdater._operator_user_for_update(),  # noqa: SLF001
                ("root", None),
            )

        with (
            patch.dict(
                os.environ,
                {"AXOL_OPERATOR_USER": "bad user"},
                clear=True,
            ),
            patch.object(update.pwd, "getpwnam") as lookup,
        ):
            operator, error = update.SelfUpdater._operator_user_for_update()  # noqa: SLF001
        self.assertIsNone(operator)
        self.assertIn("persisted", error or "")
        lookup.assert_not_called()

        with (
            patch.dict(
                os.environ,
                {"AXOL_OPERATOR_USER": "deleted-user"},
                clear=True,
            ),
            patch.object(update.pwd, "getpwnam", side_effect=KeyError),
        ):
            operator, error = update.SelfUpdater._operator_user_for_update()  # noqa: SLF001
        self.assertIsNone(operator)
        self.assertIn("no longer exists", error or "")

    async def test_update_uses_verified_installer_from_exact_target_wheel(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(return_value=_Process())

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001
            await updater.shutdown()

        command = create.await_args_list[0].args
        self.assertEqual(command[0], update._SYSTEMD_RUN_EXECUTABLE)  # noqa: SLF001
        self.assertIn("--no-block", command)
        self.assertIn("--property=RuntimeMaxSec=45min", command)
        self.assertIn("--setenv=AXOL_RELEASE_TAG=release-v0.1.37", command)
        self.assertIn("--setenv=AXOL_OPERATOR_USER=robot", command)
        stage = "/var/lib/almond-axol/update-workers/release-0.1.37-test"
        installer = f"{stage}/install"
        wheel = f"{stage}/almond_axol-0.1.37-py3-none-any.whl"
        self.assertIn(
            f"--property=ExecStopPost={update._RM_EXECUTABLE} -rf -- {stage}",  # noqa: SLF001
            command,
        )
        self.assertIn(f"--setenv=AXOL_RELEASE_WHEEL={wheel}", command)
        self.assertIn(f"--setenv=AXOL_RELEASE_WHEEL_SHA256={'a' * 64}", command)
        self.assertEqual(command[-2:], (update._BASH_EXECUTABLE, installer))  # noqa: SLF001
        self.assertNotIn("curl", " ".join(command))
        self.assertNotIn("-c", command)
        self.assertNotIn("uv tool install", " ".join(command))
        updater._stage_release.assert_called_once()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertEqual(create.await_count, 1)
        self.assertEqual(updater._phase, "restarting")  # noqa: SLF001

    async def test_pinned_ultimate_preflight_is_repeated_by_installer_worker(
        self,
    ) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        vcs = "git+https://github.com/nijkah/pyvut.git@" + "a" * 40
        create = AsyncMock(return_value=_Process())

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([vcs], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001
            await updater.shutdown()

        self.assertEqual(create.await_count, 1)
        command = create.await_args.args
        self.assertIn("AXOL_RELEASE_TAG=release-v0.1.37", " ".join(command))
        self.assertNotIn(vcs, " ".join(command))

    async def test_published_plugin_preflight_allows_hosted_worker(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(return_value=_Process())
        plugin = "lerobot_robot_axol==0.1.1"

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([plugin], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001
            await updater.shutdown()

        self.assertEqual(create.await_count, 1)
        self.assertNotIn(plugin, " ".join(create.await_args.args))

    async def test_credential_preflight_stops_before_uv(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        create = AsyncMock()

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], "move package-local Wi-Fi config first"),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        create.assert_not_awaited()
        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertEqual(updater._error, "move package-local Wi-Fi config first")  # noqa: SLF001

    async def test_worker_launch_failure_is_sanitized(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._systemctl = Mock(return_value=Mock(returncode=0))  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(
            return_value=_Process(
                1, b"https://user:secret@private.invalid/worker failed\n"
            )
        )

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertIn("worker could not be started", updater._error or "")  # noqa: SLF001
        self.assertNotIn("secret", updater._error or "")  # noqa: SLF001
        self.assertNotIn("private.invalid", updater._error or "")  # noqa: SLF001

    async def test_missing_worker_executable_does_not_launch_any_process(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock()

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(update.Path, "is_file", return_value=False),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        create.assert_not_awaited()
        self.assertIn("worker is unavailable", updater._error or "")  # noqa: SLF001

    async def test_update_worker_never_mutates_live_process_environment(self) -> None:
        updater = _updater()
        updater._managed_service_error = Mock(return_value=None)  # type: ignore[method-assign]  # noqa: SLF001
        updater._provision = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        updater._verify_ultimate_runtime = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=(
                    ["git+https://github.com/nijkah/pyvut.git@" + "a" * 40],
                    None,
                ),
            ),
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=_Process())
            ),
        ):
            await updater._run_update()  # noqa: SLF001
            await updater.shutdown()

        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._verify_ultimate_runtime.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertEqual(updater._phase, "restarting")  # noqa: SLF001

    async def test_provision_subprocess_error_is_sanitized(self) -> None:
        updater = _updater()
        create = AsyncMock(
            return_value=_Process(7, b"host detail with private-value\n")
        )
        with (
            patch.object(asyncio, "create_subprocess_exec", create),
            patch.object(update.Path, "is_file", return_value=True),
            self.assertLogs(update._logger, level="WARNING") as logs,  # noqa: SLF001
        ):
            error = await updater._provision()  # noqa: SLF001

        self.assertIn("provisioning failed (7)", error or "")
        self.assertNotIn("private-value", error or "")
        self.assertNotIn("private-value", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
