from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
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


def _updater() -> update.SelfUpdater:
    with (
        patch.object(update, "installed_origin", return_value=("https://repo", "abc")),
        patch.object(update, "installed_version", return_value="0.1.34"),
        patch.object(update, "installed_commit", return_value="abc"),
    ):
        result = update.SelfUpdater(lambda: True)
    result._remote_tag = "v0.1.35"  # noqa: SLF001
    result._remote_version = "0.1.35"  # noqa: SLF001
    result._state = "updating"  # noqa: SLF001
    return result


class UltimateUpdateRequirementTests(unittest.TestCase):
    def test_only_the_current_expected_pin_is_preserved(self) -> None:
        with (
            patch.object(
                tracker_ultimate,
                "_installed_pyvut",
                return_value=("0.0.test", "different-pin"),
            ),
            patch.object(tracker_ultimate, "_packaged_wifi_status") as wifi_status,
        ):
            self.assertEqual(
                tracker_ultimate.ultimate_runtime_update_requirement(),
                (None, None),
            )
        wifi_status.assert_not_called()

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
    def test_installer_preserves_and_verifies_before_service_restart(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        preflight = installer.index('"${BIN_DIR}/axol" update-preflight')
        force_install = installer.index('"${UV}" tool install')
        verify = installer.index("axol tracker.ultimate.install", force_install)
        restart = installer.index('systemctl restart "${SERVICE_NAME}"')
        self.assertLess(preflight, force_install)
        self.assertLess(force_install, verify)
        self.assertLess(verify, restart)
        self.assertIn('"${UPDATE_WITH_ARGS[@]}"', installer)
        self.assertIn('20)\n            die "${UPDATE_PREFLIGHT_OUTPUT}"', installer)
        self.assertLess(installer.index("LEGACY_TORCH_STATUS"), force_install)
        self.assertIn("PyPI torch 2.10 is CPU-only on aarch64", installer)

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
            installer.index(
                'legacy_plugin_update_requirement "${EXISTING_AXOL_PYTHON}"'
            ),
            force_install,
        )
        self.assertIn("direct or customized lerobot_robot_axol", installer)

    def test_installer_stops_before_restart_when_provisioning_fails(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        provision = installer.index("axol provision 2>&1")
        fatal = installer.index('if [ "${PROVISION_STATUS}" -ne 0 ]')
        restart = installer.index('systemctl restart "${SERVICE_NAME}"')
        self.assertLess(provision, fatal)
        self.assertLess(fatal, restart)
        self.assertIn('PROVISION_STATUS="${PIPESTATUS[0]}"', installer)

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


class SelfUpdateUltimateTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_ultimate_runtime_remains_an_explicit_opt_in(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(return_value=_Process())

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        command = create.await_args_list[0].args
        self.assertNotIn("--with", command)
        self.assertEqual(command[-1], "almond-axol[lerobot,sim,tracker]==0.1.35")
        self.assertEqual(create.await_count, 1)
        updater._maybe_restart.assert_called_once_with()  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_pinned_ultimate_is_preserved_and_verified(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        vcs = "git+https://github.com/nijkah/pyvut.git@" + "a" * 40
        create = AsyncMock(side_effect=[_Process(), _Process()])

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([vcs], None),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
            patch.object(update.shutil, "which", return_value="/usr/local/bin/axol"),
        ):
            await updater._run_update()  # noqa: SLF001

        self.assertEqual(
            create.await_args_list[0].args,
            (
                "uv",
                "tool",
                "install",
                "--python",
                "3.13",
                "--force",
                "--with",
                vcs,
                "almond-axol[lerobot,sim,tracker]==0.1.35",
            ),
        )
        self.assertEqual(
            create.await_args_list[1],
            call(
                "/usr/local/bin/axol",
                "tracker.ultimate.install",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            ),
        )
        updater._maybe_restart.assert_called_once_with()  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_published_plugin_is_preserved_without_ultimate_repair(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        updater._verify_ultimate_runtime = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(return_value=_Process())
        plugin = "lerobot_robot_axol==0.1.1"

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([plugin], None),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        self.assertEqual(
            create.await_args_list[0].args,
            (
                "uv",
                "tool",
                "install",
                "--python",
                "3.13",
                "--force",
                "--with",
                plugin,
                "almond-axol[lerobot,sim,tracker]==0.1.35",
            ),
        )
        updater._verify_ultimate_runtime.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._maybe_restart.assert_called_once_with()  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_credential_preflight_stops_before_uv(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
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
        updater._maybe_restart.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertEqual(updater._error, "move package-local Wi-Fi config first")  # noqa: SLF001

    async def test_restore_failure_does_not_restart(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value=None
        )
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(
            side_effect=[
                _Process(),
                _Process(1, b"https://user:secret@private.invalid/simple\n"),
            ]
        )

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=(
                    ["git+https://github.com/nijkah/pyvut.git@" + "a" * 40],
                    None,
                ),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
            patch.object(update.shutil, "which", return_value="/usr/local/bin/axol"),
        ):
            await updater._run_update()  # noqa: SLF001

        updater._maybe_restart.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertFalse(updater._restart_pending)  # noqa: SLF001
        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertIn("VIVE Ultimate runtime restore failed", updater._error or "")  # noqa: SLF001
        self.assertNotIn("secret", updater._error or "")  # noqa: SLF001
        self.assertNotIn("private.invalid", updater._error or "")  # noqa: SLF001

    async def test_uv_install_failure_is_sanitized_and_does_not_provision(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock(
            return_value=_Process(
                9, b"https://user:secret@private.invalid/simple failed\n"
            )
        )

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=([], None),
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._maybe_restart.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertIn("uv tool install failed (9)", updater._error or "")  # noqa: SLF001
        self.assertNotIn("secret", updater._error or "")  # noqa: SLF001
        self.assertNotIn("private.invalid", updater._error or "")  # noqa: SLF001

    async def test_provision_failure_does_not_restore_or_restart(self) -> None:
        updater = _updater()
        updater._provision = AsyncMock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value="updated Axol, but provisioning failed (1)"
        )
        updater._verify_ultimate_runtime = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        updater._maybe_restart = Mock()  # type: ignore[method-assign]  # noqa: SLF001

        with (
            patch.object(
                update,
                "release_update_requirements",
                return_value=(
                    ["git+https://github.com/nijkah/pyvut.git@" + "a" * 40],
                    None,
                ),
            ),
            patch.object(
                asyncio, "create_subprocess_exec", AsyncMock(return_value=_Process())
            ),
        ):
            await updater._run_update()  # noqa: SLF001

        updater._verify_ultimate_runtime.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        updater._maybe_restart.assert_not_called()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertFalse(updater._restart_pending)  # noqa: SLF001
        self.assertEqual(updater._state, "error")  # noqa: SLF001
        self.assertEqual(  # noqa: SLF001
            updater._error, "updated Axol, but provisioning failed (1)"
        )

    async def test_provision_subprocess_error_is_sanitized(self) -> None:
        updater = _updater()
        create = AsyncMock(
            return_value=_Process(7, b"host detail with private-value\n")
        )
        with (
            patch.object(asyncio, "create_subprocess_exec", create),
            patch.object(update.shutil, "which", return_value="/usr/local/bin/axol"),
            self.assertLogs(update._logger, level="WARNING") as logs,  # noqa: SLF001
        ):
            error = await updater._provision()  # noqa: SLF001

        self.assertIn("provisioning failed (7)", error or "")
        self.assertNotIn("private-value", error or "")
        self.assertNotIn("private-value", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
