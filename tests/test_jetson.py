from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from almond_axol.utils import jetson


class JetsonPowerModeTest(unittest.TestCase):
    def _preferred_mode(self, config: str) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "nvpmodel.conf"
            config_path.write_text(config)
            with patch.object(jetson, "_NVPMODEL_CONFIG", config_path):
                return jetson._preferred_max_power_mode()  # noqa: SLF001

    def test_prefers_maxn_super_using_its_configured_id(self) -> None:
        self.assertEqual(
            self._preferred_mode(
                """
                < POWER_MODEL ID=0 NAME=15W >
                < POWER_MODEL ID=1 NAME=25W >
                < POWER_MODEL ID=2 NAME=MAXN_SUPER >
                """
            ),
            ("2", "MAXN SUPER"),
        )

    def test_uses_configured_maxn_when_super_is_unavailable(self) -> None:
        self.assertEqual(
            self._preferred_mode(
                """
                < POWER_MODEL ID=0 NAME=MODE_15W >
                < POWER_MODEL NAME=MAXN ID=3 >
                """
            ),
            ("3", "MAXN"),
        )

    def test_falls_back_to_mode_zero_when_names_are_unavailable(self) -> None:
        self.assertEqual(
            self._preferred_mode("< POWER_MODEL ID=1 NAME=MODE_15W >"),
            ("0", "MAXN"),
        )

    def test_sets_maxn_super_instead_of_mode_zero(self) -> None:
        escalator = Mock()
        escalator.run.return_value = (True, "")
        with (
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson.shutil, "which", return_value="/usr/sbin/nvpmodel"),
            patch.object(
                jetson,
                "_preferred_max_power_mode",
                return_value=("2", "MAXN SUPER"),
            ),
            patch.object(jetson, "_query_power_mode", side_effect=["1", "2"]),
        ):
            jetson._set_max_power_mode(escalator)  # noqa: SLF001

        escalator.run.assert_called_once_with(
            ["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"
        )
        escalator.write.assert_not_called()

    def test_persists_selected_super_mode_when_switch_needs_reboot(self) -> None:
        escalator = Mock()
        escalator.run.return_value = (
            False,
            "NVPM WARN: Reboot required for changing to this power mode: 2",
        )
        escalator.write.return_value = (True, "")
        with (
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson.shutil, "which", return_value="/usr/sbin/nvpmodel"),
            patch.object(
                jetson,
                "_preferred_max_power_mode",
                return_value=("2", "MAXN SUPER"),
            ),
            patch.object(jetson, "_query_power_mode", side_effect=["1", "1"]),
        ):
            jetson._set_max_power_mode(escalator)  # noqa: SLF001

        self.assertEqual(
            escalator.method_calls,
            [
                call.run(["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"),
                call.write(jetson._NVPMODEL_STATUS, "pmode:0002"),  # noqa: SLF001
            ],
        )

    def test_failed_deferred_status_write_is_reported(self) -> None:
        escalator = Mock()
        escalator.run.return_value = (False, "reboot required")
        escalator.write.return_value = (False, "read-only filesystem")
        with (
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson.shutil, "which", return_value="/usr/sbin/nvpmodel"),
            patch.object(
                jetson,
                "_preferred_max_power_mode",
                return_value=("2", "MAXN SUPER"),
            ),
            patch.object(jetson, "_query_power_mode", side_effect=["1", "1"]),
            self.assertLogs("almond_axol.utils.jetson", level="WARNING") as logs,
        ):
            jetson._set_max_power_mode(escalator)  # noqa: SLF001

        self.assertIn("recording it for the next boot failed", "\n".join(logs.output))
        self.assertIn("read-only filesystem", "\n".join(logs.output))

    def test_root_command_failure_is_not_retried_through_sudo(self) -> None:
        failure = subprocess.CompletedProcess(
            ["/usr/sbin/nvpmodel", "-m", "2"],
            234,
            stdout="",
            stderr="NVPM WARN: Reboot required",
        )
        escalator = jetson._RootEscalator(interactive=False)  # noqa: SLF001
        with (
            patch.object(jetson.os, "geteuid", return_value=0),
            patch.object(jetson.subprocess, "run", return_value=failure) as run,
        ):
            ok, detail = escalator.run(
                ["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"
            )

        self.assertFalse(ok)
        self.assertIn("Reboot required", detail)
        run.assert_called_once_with(
            ["/usr/sbin/nvpmodel", "-m", "2"],
            input="n\n",
            capture_output=True,
            text=True,
        )

    def test_missing_sudo_is_a_reported_non_root_failure(self) -> None:
        failure = subprocess.CompletedProcess(
            ["/usr/sbin/nvpmodel", "-m", "2"],
            1,
            stdout="",
            stderr="permission denied",
        )
        escalator = jetson._RootEscalator(interactive=True)  # noqa: SLF001
        with (
            patch.object(jetson.os, "geteuid", return_value=1000),
            patch.object(jetson.subprocess, "run", return_value=failure) as run,
            patch.object(
                jetson,
                "prime_sudo",
                side_effect=FileNotFoundError("sudo is unavailable"),
            ),
        ):
            ok, detail = escalator.run(["/usr/sbin/nvpmodel", "-m", "2"])

        self.assertFalse(ok)
        self.assertIn("sudo is unavailable", detail)
        self.assertEqual(run.call_count, 1)

    def test_non_root_command_failure_retries_once_through_sudo(self) -> None:
        direct_failure = subprocess.CompletedProcess(
            ["/usr/sbin/nvpmodel", "-m", "2"],
            1,
            stdout="",
            stderr="permission denied",
        )
        sudo_success = subprocess.CompletedProcess(
            ["sudo", "-n", "/usr/sbin/nvpmodel", "-m", "2"],
            0,
            stdout="",
            stderr="",
        )
        escalator = jetson._RootEscalator(interactive=True)  # noqa: SLF001
        with (
            patch.object(jetson.os, "geteuid", return_value=1000),
            patch.object(
                jetson.subprocess,
                "run",
                side_effect=[direct_failure, sudo_success],
            ) as run,
            patch.object(jetson, "prime_sudo", return_value=True) as prime,
        ):
            ok, detail = escalator.run(
                ["/usr/sbin/nvpmodel", "-m", "2"], input_text="n\n"
            )

        self.assertTrue(ok)
        self.assertEqual(detail, "")
        prime.assert_called_once_with()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[1],
            call(
                ["sudo", "-n", "/usr/sbin/nvpmodel", "-m", "2"],
                input="n\n",
                capture_output=True,
                text=True,
            ),
        )

    def test_root_write_failure_is_not_retried_through_sudo(self) -> None:
        escalator = jetson._RootEscalator(interactive=False)  # noqa: SLF001
        with (
            patch.object(jetson.os, "geteuid", return_value=0),
            patch.object(Path, "write_text", side_effect=OSError("read-only")),
            patch.object(jetson.subprocess, "run") as run,
        ):
            ok, detail = escalator.write(Path("/var/lib/nvpmodel/status"), "mode")

        self.assertFalse(ok)
        self.assertIn("read-only", detail)
        run.assert_not_called()

    def test_non_root_write_failure_retries_once_through_sudo(self) -> None:
        success = subprocess.CompletedProcess(
            ["sudo", "-n", "tee", "/var/lib/nvpmodel/status"],
            0,
            stdout="pmode:0002",
            stderr="",
        )
        escalator = jetson._RootEscalator(interactive=True)  # noqa: SLF001
        status = Path("/var/lib/nvpmodel/status")
        with (
            patch.object(jetson.os, "geteuid", return_value=1000),
            patch.object(Path, "write_text", side_effect=PermissionError("denied")),
            patch.object(jetson.subprocess, "run", return_value=success) as run,
            patch.object(jetson, "prime_sudo", return_value=True) as prime,
        ):
            ok, detail = escalator.write(status, "pmode:0002")

        self.assertTrue(ok)
        self.assertEqual(detail, "")
        prime.assert_called_once_with()
        run.assert_called_once_with(
            ["sudo", "-n", "tee", str(status)],
            input="pmode:0002",
            capture_output=True,
            text=True,
        )

    def test_declined_sudo_prompt_is_attempted_only_once(self) -> None:
        escalator = jetson._RootEscalator(interactive=True)  # noqa: SLF001
        with (
            patch.object(jetson.os, "geteuid", return_value=1000),
            patch.object(Path, "write_text", side_effect=PermissionError("denied")),
            patch.object(
                jetson.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["sudo", "-n", "tee"],
                    1,
                    stdout="",
                    stderr="sudo unavailable",
                ),
            ),
            patch.object(jetson, "prime_sudo", return_value=False) as prime,
        ):
            self.assertFalse(escalator.write(Path("/sys/mock/one"), "1")[0])
            self.assertFalse(escalator.write(Path("/sys/mock/two"), "2")[0])

        prime.assert_called_once_with()

    def test_installed_service_waits_for_nvpmodel(self) -> None:
        installer = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()

        self.assertIn(
            "After=nvpmodel.service network-online.target",
            installer,
        )


if __name__ == "__main__":
    unittest.main()
