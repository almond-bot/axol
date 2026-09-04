from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.cli import mantis_session


class MantisSessionSafetyTest(unittest.TestCase):
    def test_retired_service_is_stopped_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_service = root / "old.service"
            new_service = root / "new.service"
            old_service.write_text("legacy")
            with (
                patch.object(mantis_session, "_PRE_MANTIS_SERVICE_PATH", old_service),
                patch.object(mantis_session, "_SERVICE_PATH", new_service),
                patch.object(mantis_session.shutil, "which", return_value="/opt/axol"),
                patch.object(mantis_session, "_operator_user", return_value="operator"),
                patch.object(mantis_session, "run_root") as run_root,
            ):
                mantis_session._install()

        calls = run_root.call_args_list
        self.assertEqual(
            calls[0].args[0],
            ["systemctl", "stop", mantis_session._PRE_MANTIS_SERVICE_NAME],
        )
        self.assertEqual(calls[0].kwargs, {"check": True})
        self.assertEqual(
            calls[1].args[0],
            ["systemctl", "disable", mantis_session._PRE_MANTIS_SERVICE_NAME],
        )
        self.assertEqual(calls[1].kwargs, {"check": True})
        self.assertEqual(calls[2].args[0], ["rm", "-f", str(old_service)])
        unit = calls[3].kwargs["input_text"]
        self.assertIn("User=operator\n", unit)
        self.assertIn("Wants=axol.service\n", unit)
        self.assertIn("After=network-online.target axol.service\n", unit)
        # Triggering (``|``) so the updater's one-shot token drop-in can admit a
        # start; a non-triggering condition is ANDed with that OR group.
        self.assertIn(
            "ConditionPathExists=|!/var/lib/almond-axol/update-incomplete\n", unit
        )
        self.assertNotIn("ConditionPathExists=!", unit)

    def test_retired_service_shutdown_failure_keeps_definition(self) -> None:
        for failed_action in ("stop", "disable"):
            with (
                self.subTest(failed_action=failed_action),
                tempfile.TemporaryDirectory() as directory,
            ):
                old_service = Path(directory) / "old.service"
                old_service.write_text("legacy")

                def run_root(command: list[str], **_kwargs: object) -> object:
                    if command == [
                        "systemctl",
                        failed_action,
                        mantis_session._PRE_MANTIS_SERVICE_NAME,
                    ]:
                        raise RuntimeError(f"{failed_action} denied")
                    return object()

                with (
                    patch.object(
                        mantis_session, "_PRE_MANTIS_SERVICE_PATH", old_service
                    ),
                    patch.object(
                        mantis_session.shutil, "which", return_value="/opt/axol"
                    ),
                    patch.object(
                        mantis_session, "_operator_user", return_value="operator"
                    ),
                    patch.object(mantis_session, "run_root", side_effect=run_root),
                    self.assertRaisesRegex(RuntimeError, f"{failed_action} denied"),
                ):
                    mantis_session._install()

                self.assertTrue(old_service.exists())

    def test_sudo_install_uses_validated_invoking_operator(self) -> None:
        account = SimpleNamespace(pw_name="robot")
        with (
            patch.object(mantis_session.os, "geteuid", return_value=0),
            patch.dict(
                mantis_session.os.environ,
                {"SUDO_UID": "1234", "SUDO_USER": "robot"},
            ),
            patch.object(
                mantis_session.pwd, "getpwuid", return_value=account
            ) as lookup,
        ):
            self.assertEqual(mantis_session._operator_user(), "robot")
        lookup.assert_called_once_with(1234)

    def test_root_install_without_operator_is_rejected(self) -> None:
        with (
            patch.object(mantis_session.os, "geteuid", return_value=0),
            patch.dict(
                mantis_session.os.environ,
                {"SUDO_UID": "", "SUDO_USER": ""},
            ),
            self.assertRaisesRegex(SystemExit, "non-root operator"),
        ):
            mantis_session._operator_user()

    def test_fallback_is_blocked_by_managed_service(self) -> None:
        with (
            patch.object(mantis_session, "_update_is_incomplete", return_value=False),
            patch.object(
                mantis_session, "_managed_serve_is_installed", return_value=True
            ),
        ):
            self.assertIn("axol.service", mantis_session._fallback_serve_block_reason())

    def test_managed_service_inspection_failure_is_closed_on_systemd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            failed = SimpleNamespace(returncode=1, stdout="")
            with (
                patch.object(mantis_session, "_SYSTEMD_RUNTIME", runtime),
                patch.object(
                    mantis_session.shutil, "which", return_value="/bin/systemctl"
                ),
                patch.object(mantis_session.subprocess, "run", return_value=failed),
            ):
                self.assertTrue(mantis_session._managed_serve_is_installed())
            with (
                patch.object(mantis_session, "_SYSTEMD_RUNTIME", runtime),
                patch.object(mantis_session.shutil, "which", return_value=None),
            ):
                self.assertTrue(mantis_session._managed_serve_is_installed())

    def test_clean_systemd_not_found_allows_standalone_fallback(self) -> None:
        not_found = SimpleNamespace(returncode=0, stdout="not-found\n")
        with (
            patch.object(mantis_session.shutil, "which", return_value="/bin/systemctl"),
            patch.object(mantis_session.subprocess, "run", return_value=not_found),
        ):
            self.assertFalse(mantis_session._managed_serve_is_installed())

    def test_fallback_is_blocked_by_update_marker(self) -> None:
        with (
            patch.object(mantis_session, "_update_is_incomplete", return_value=True),
            patch.object(mantis_session, "_managed_serve_is_installed") as managed,
        ):
            self.assertIn("update", mantis_session._fallback_serve_block_reason())
        managed.assert_not_called()

    def test_update_marker_check_fails_closed_on_inspection_error(self) -> None:
        with patch.object(
            mantis_session.os, "lstat", side_effect=PermissionError("denied")
        ):
            self.assertTrue(mantis_session._update_is_incomplete())
        with patch.object(mantis_session.os, "lstat", side_effect=FileNotFoundError):
            self.assertFalse(mantis_session._update_is_incomplete())

    def test_session_never_spawns_fallback_while_blocked(self) -> None:
        with (
            patch.object(mantis_session, "_adb", return_value=None),
            patch.object(
                mantis_session,
                "_fallback_serve_block_reason",
                return_value="axol.service owns the server",
            ),
            patch.object(mantis_session, "_serve_is_up", return_value=False),
            patch.object(mantis_session, "_spawn") as spawn,
            patch.object(mantis_session.time, "sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            mantis_session._session()
        spawn.assert_not_called()

    def test_running_fallback_is_stopped_when_managed_service_appears(self) -> None:
        process = Mock()
        with (
            patch.object(mantis_session, "_adb", return_value=None),
            patch.object(
                mantis_session,
                "_fallback_serve_block_reason",
                side_effect=[None, "axol.service owns the server"],
            ),
            patch.object(mantis_session, "_serve_is_up", return_value=False),
            patch.object(mantis_session, "_spawn", return_value=process),
            patch.object(mantis_session, "_stop_process") as stop,
            patch.object(mantis_session.time, "sleep", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            mantis_session._session()
        stop.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
