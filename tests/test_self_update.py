from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
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


class _HTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def _updater() -> update.SelfUpdater:
    with (
        patch.object(update, "installed_origin", return_value=("https://repo", "abc")),
        patch.object(update, "installed_version", return_value="0.1.34"),
        patch.object(update, "installed_commit", return_value="abc"),
    ):
        result = update.SelfUpdater(lambda: True)
    result._remote_tag = "release-v0.1.35"  # noqa: SLF001
    result._remote_version = "0.1.35"  # noqa: SLF001
    result._state = "updating"  # noqa: SLF001
    result._launches_blocked = True  # noqa: SLF001
    result._release_available = Mock(return_value=True)  # type: ignore[method-assign]  # noqa: SLF001
    result._arm_durable_update_guard = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=None
    )
    result._disarm_durable_update_guard = Mock(  # type: ignore[method-assign]  # noqa: SLF001
        return_value=None
    )
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
    def test_installer_preserves_and_verifies_before_service_start(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        preflight = installer.index('"${BIN_DIR}/axol" update-preflight')
        pypi_check = installer.index(
            '"https://pypi.org/pypi/almond-axol/${VERSION}/json"'
        )
        uv_install = installer.index(
            'curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh"'
        )
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
        stop_mantis = installer.index('systemctl stop "${MANTIS_SERVICE_NAME}.service"')
        stop = installer.index('systemctl disable --now "${SERVICE_NAME}.service"')
        inactive_check = installer.index(
            'systemctl is-active --quiet "${SERVICE_NAME}.service"'
        )
        force_install = installer.index('"${UV}" tool install')
        verify = installer.index("axol tracker.ultimate.install", force_install)
        enable = installer.index('systemctl enable "${SERVICE_NAME}"')
        clear_marker = installer.index('durable_remove "${UPDATE_MARKER}"')
        start = installer.index('systemctl start "${SERVICE_NAME}"')
        restore_mantis = installer.index(
            'systemctl start "${MANTIS_SERVICE_NAME}.service"'
        )
        self.assertLess(pypi_check, uv_install)
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
        self.assertLess(enable, clear_marker)
        self.assertLess(clear_marker, start)
        self.assertLess(start, restore_mantis)
        self.assertIn("MANTIS_SERVICE_SHOULD_RUN=1", installer)
        self.assertIn(
            'MANTIS_UPDATE_GUARD_DIR="/etc/systemd/system/${MANTIS_SERVICE_NAME}.service.d"',
            installer,
        )
        self.assertIn(
            '[ ! -L "${MANTIS_UPDATE_GUARD_DIR}" ]',
            installer,
        )
        self.assertEqual(
            installer.count(
                "printf '[Unit]\\nConditionPathExists=!%s\\n' \"${UPDATE_MARKER}\""
            ),
            2,
        )
        self.assertIn(
            'ConditionPathExists=!%s\\n\' "${UPDATE_MARKER}"',
            installer,
        )
        self.assertIn(
            "current Axol service and runtime were not changed",
            installer,
        )
        self.assertIn("has no non-yanked PyPI artifacts", installer)
        self.assertIn("wrong almond-axol version", installer)
        self.assertIn('UV="${BIN_DIR}/uv"', installer)
        self.assertIn('"refs/tags/v*" "refs/tags/release-v*"', installer)
        self.assertIn("^(release-)?v", installer)
        self.assertIn('MIN_SAFE_RELEASE_VERSION="0.1.36"', installer)
        self.assertLess(installer.index("OLDEST_VERSION="), pypi_check)
        self.assertIn("hardened migration release is not published yet", installer)
        self.assertIn('"uv ${UV_VERSION}"|"uv ${UV_VERSION} "*', installer)
        self.assertNotIn('== "uv ${UV_VERSION}"*', installer)
        self.assertNotIn('!= "uv ${UV_VERSION}"*', installer)
        self.assertNotIn('UV="$(command -v uv', installer)
        self.assertIn('"${UPDATE_WITH_ARGS[@]}"', installer)
        self.assertIn('20)\n            die "${UPDATE_PREFLIGHT_OUTPUT}"', installer)
        self.assertLess(installer.index("LEGACY_TORCH_STATUS"), force_install)
        self.assertIn("PyPI torch 2.10 is CPU-only on aarch64", installer)

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

        marker_commit = installer.index(
            "Could not durably arm the update restart guard"
        )
        stop_service = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"'
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
            'durable_install_directory "/var/lib/almond-axol" root root 0750'
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

    def test_installer_restores_guard_when_verified_service_fails_to_start(
        self,
    ) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        clear_marker = installer.index('durable_remove "${UPDATE_MARKER}"')
        remove_guard = installer.index('if ! durable_remove "${UPDATE_MARKER}"; then')
        remove_failure_rearm = installer.index(
            "write_update_marker || MARKER_RESTORED=0", remove_guard
        )
        remove_failure_disable = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"', remove_guard
        )
        start_guard = installer.index(
            'if ! systemctl start "${SERVICE_NAME}"; then', clear_marker
        )
        restore_marker = installer.index(
            "write_update_marker || MARKER_RESTORED=0", start_guard
        )
        disable_failed_service = installer.index(
            'systemctl disable --now "${SERVICE_NAME}.service"', start_guard
        )
        failed_start_exit = installer.index(
            "Axol was verified but failed to start; the service is blocked and disabled",
            start_guard,
        )
        self.assertLess(clear_marker, start_guard)
        self.assertLess(remove_guard, remove_failure_rearm)
        self.assertLess(remove_failure_rearm, remove_failure_disable)
        self.assertLess(remove_failure_disable, start_guard)
        self.assertLess(start_guard, restore_marker)
        self.assertLess(start_guard, disable_failed_service)
        self.assertLess(restore_marker, failed_start_exit)
        self.assertLess(disable_failed_service, failed_start_exit)

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

    def test_installer_stops_before_mutation_and_stays_stopped_on_failure(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        stop = installer.index('systemctl disable --now "${SERVICE_NAME}.service"')
        force_install = installer.index('"${UV}" tool install')
        provision = installer.index("axol provision 2>&1")
        fatal = installer.index('if [ "${PROVISION_STATUS}" -ne 0 ]')
        start = installer.index('systemctl start "${SERVICE_NAME}"')
        self.assertLess(stop, force_install)
        self.assertLess(provision, fatal)
        self.assertLess(fatal, start)
        self.assertIn('PROVISION_STATUS="${PIPESTATUS[0]}"', installer)
        self.assertIn("service remains blocked and disabled", installer)

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

    async def test_new_release_namespace_is_resolved_with_legacy_history(self) -> None:
        output = b"\n".join(
            (
                b"a refs/tags/v0.1.35",
                b"b refs/tags/release-v0.1.36",
                b"c refs/tags/release-v0.1.37^{}",
                b"d refs/tags/release-v0.1.37",
                b"e refs/tags/release-v0.1.37rc1",
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

    def test_exact_pypi_release_must_have_a_non_yanked_artifact(self) -> None:
        available = {
            "info": {"version": "0.1.35"},
            "urls": [{"packagetype": "bdist_wheel", "yanked": False}],
        }
        unavailable = {
            "info": {"version": "0.1.35"},
            "urls": [{"packagetype": "bdist_wheel", "yanked": True}],
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

        with patch.object(
            update.urllib.request,
            "urlopen",
            return_value=_HTTPResponse(unavailable),
        ):
            self.assertFalse(update._release_available_on_pypi("0.1.35"))  # noqa: SLF001

    async def test_missing_pypi_release_does_not_arm_or_mutate(self) -> None:
        updater = _updater()
        updater._release_available = Mock(return_value=False)  # type: ignore[method-assign]  # noqa: SLF001
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
        self.assertIn("not yet available from PyPI", updater._error or "")  # noqa: SLF001
        self.assertIn("retry later", updater._error or "")  # noqa: SLF001

    async def test_guard_failure_happens_before_environment_mutation(self) -> None:
        updater = _updater()
        updater._arm_durable_update_guard = Mock(  # type: ignore[method-assign]  # noqa: SLF001
            return_value="could not establish durable guard"
        )
        updater._provision = AsyncMock()  # type: ignore[method-assign]  # noqa: SLF001
        create = AsyncMock()

        with (
            patch.object(
                update, "release_update_requirements", return_value=([], None)
            ),
            patch.object(asyncio, "create_subprocess_exec", create),
        ):
            await updater._run_update()  # noqa: SLF001

        create.assert_not_awaited()
        updater._provision.assert_not_awaited()  # type: ignore[attr-defined]  # noqa: SLF001
        self.assertTrue(updater.launches_blocked)

    def test_durable_guard_is_live_before_marker_and_disable(self) -> None:
        updater = _updater()
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
            patch.object(
                update.shutil,
                "which",
                side_effect=lambda name: (
                    "/usr/bin/systemctl"
                    if name == "systemctl"
                    else update._MANAGED_UV_EXECUTABLE  # noqa: SLF001
                ),
            ),
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
        self.assertLess(remove_index, start_index)
        self.assertLess(start_index, active_index)
        self.assertIsNone(updater._mantis_restore_requested)  # noqa: SLF001
        self.assertIsNone(updater._mantis_enable_requested)  # noqa: SLF001

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

        def write(path: Path, _content: str, *, mode: int) -> None:
            events.append(("write", (path, mode)))

        updater._systemctl = Mock(side_effect=systemctl)  # type: ignore[method-assign]  # noqa: SLF001
        with (
            patch.object(update.Path, "is_file", return_value=True),
            patch.object(update, "_remove_durable_file"),
            patch.object(update, "_write_durable_root_file", side_effect=write),
        ):
            error = update.SelfUpdater._disarm_durable_update_guard(updater)  # noqa: SLF001

        self.assertIn("axol-mantis.service could not be started", error or "")
        marker_index = events.index(
            ("write", (update._UPDATE_GUARD_MARKER, 0o600))  # noqa: SLF001
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

    async def test_startup_provision_holds_launch_barrier_until_complete(self) -> None:
        updater = _updater()
        updater._state = "idle"  # noqa: SLF001
        updater._launches_blocked = False  # noqa: SLF001
        gate = asyncio.Event()

        async def provision() -> None:
            await gate.wait()
            return None

        updater._provision = AsyncMock(side_effect=provision)  # type: ignore[method-assign]  # noqa: SLF001
        updater.ensure_provisioned()
        await asyncio.sleep(0)
        self.assertTrue(updater.maintenance_active)
        self.assertTrue(updater.launches_blocked)

        gate.set()
        assert updater._provision_task is not None  # noqa: SLF001
        await updater._provision_task  # noqa: SLF001
        self.assertFalse(updater.maintenance_active)
        self.assertFalse(updater.launches_blocked)


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
