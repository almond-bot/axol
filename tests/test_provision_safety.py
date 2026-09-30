from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.cli import provision


class ProvisionSafetyTest(unittest.TestCase):
    def test_upgrade_scrubs_only_exact_legacy_can_root_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe_unit = root / "axol-can-up.service"
            safe_unit = root / "unrelated.service"
            unsafe_unit.write_text(
                "[Service]\nExecStart=/bin/bash /home/operator/.almond/can/startup.sh\n"
            )
            safe_content = (
                "[Service]\nExecStart=/bin/bash /etc/almond-axol/can/startup.sh\n"
            )
            safe_unit.write_text(safe_content)
            legacy_cron = (
                "@reboot /root/.almond/can/startup_mantis.sh\n"
                "@reboot /home/operator/bin/backup\n"
            )

            calls: list[tuple[list[str], dict[str, object]]] = []

            def run_root(
                command: list[str], **kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                calls.append((command, kwargs))
                if command == ["env", "LC_ALL=C", "crontab", "-l"]:
                    return subprocess.CompletedProcess(command, 0, legacy_cron, "")
                return subprocess.CompletedProcess(command, 0, "", "")

            output = io.StringIO()
            with (
                patch.object(
                    provision.shutil, "which", return_value="/usr/bin/crontab"
                ),
                patch.object(
                    provision,
                    "_LEGACY_CAN_UNIT_FILES",
                    (unsafe_unit, safe_unit),
                ),
                patch.object(provision, "run_root", side_effect=run_root),
                contextlib.redirect_stdout(output),
            ):
                self.assertTrue(provision._neutralize_legacy_can_root_execution())

            crontab_write = next(
                kwargs["input_text"]
                for command, kwargs in calls
                if command == ["crontab", "-"]
            )
            self.assertEqual(crontab_write, "@reboot /home/operator/bin/backup\n")
            self.assertIn(
                (["systemctl", "stop", unsafe_unit.name], {"check": True}),
                calls,
            )
            self.assertIn(
                (["systemctl", "disable", unsafe_unit.name], {"check": True}),
                calls,
            )
            self.assertIn((["rm", "-f", str(unsafe_unit)], {"check": True}), calls)
            self.assertNotIn((["rm", "-f", str(safe_unit)], {"check": True}), calls)
            self.assertEqual(safe_unit.read_text(), safe_content)
            self.assertIn("Run `axol can.setup`", output.getvalue())
            self.assertNotIn("sudo axol", output.getvalue())

    def test_safe_cron_and_unit_are_left_untouched_without_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            safe_unit = Path(directory) / "axol-can-up.service"
            safe_unit.write_text(
                "[Service]\nExecStart=/bin/bash /etc/almond-axol/can/startup.sh\n"
            )
            current = subprocess.CompletedProcess(
                ["env", "LC_ALL=C", "crontab", "-l"],
                0,
                "@reboot /home/operator/bin/backup\n",
                "",
            )
            output = io.StringIO()
            with (
                patch.object(
                    provision.shutil, "which", return_value="/usr/bin/crontab"
                ),
                patch.object(provision, "_LEGACY_CAN_UNIT_FILES", (safe_unit,)),
                patch.object(provision, "run_root", return_value=current) as run_root,
                contextlib.redirect_stdout(output),
            ):
                self.assertFalse(provision._neutralize_legacy_can_root_execution())

            run_root.assert_called_once_with(["env", "LC_ALL=C", "crontab", "-l"])
            self.assertEqual(output.getvalue(), "")

    def test_unit_shutdown_failure_aborts_before_any_reference_is_removed(
        self,
    ) -> None:
        for failed_action in ("stop", "disable"):
            with (
                self.subTest(failed_action=failed_action),
                tempfile.TemporaryDirectory() as directory,
            ):
                unsafe_unit = Path(directory) / "axol-can-up.service"
                unsafe_unit.write_text(
                    "[Service]\n"
                    "ExecStart=/bin/bash /home/operator/.almond/can/startup.sh\n"
                )
                legacy_cron = "@reboot /root/.almond/can/startup_mantis.sh\n"
                calls: list[tuple[list[str], dict[str, object]]] = []

                def run_root(
                    command: list[str], **kwargs: object
                ) -> subprocess.CompletedProcess[str]:
                    calls.append((command, kwargs))
                    if command == ["env", "LC_ALL=C", "crontab", "-l"]:
                        return subprocess.CompletedProcess(command, 0, legacy_cron, "")
                    if command == ["systemctl", failed_action, unsafe_unit.name]:
                        raise RuntimeError(
                            f"`systemctl` failed: {failed_action} denied"
                        )
                    return subprocess.CompletedProcess(command, 0, "", "")

                output = io.StringIO()
                with (
                    patch.object(
                        provision.shutil, "which", return_value="/usr/bin/crontab"
                    ),
                    patch.object(provision, "_LEGACY_CAN_UNIT_FILES", (unsafe_unit,)),
                    patch.object(provision, "run_root", side_effect=run_root),
                    contextlib.redirect_stdout(output),
                    self.assertRaisesRegex(RuntimeError, f"{failed_action} denied"),
                ):
                    provision._neutralize_legacy_can_root_execution()

                mutations = (
                    ["crontab", "-"],
                    ["rm", "-f", str(unsafe_unit)],
                )
                self.assertFalse(any(command in mutations for command, _ in calls))
                self.assertTrue(unsafe_unit.exists())
                self.assertEqual(output.getvalue(), "")

    def test_localized_host_uses_c_locale_for_no_crontab_diagnostic(self) -> None:
        no_crontab = subprocess.CompletedProcess(
            ["env", "LC_ALL=C", "crontab", "-l"],
            1,
            "",
            "no crontab for root\n",
        )
        output = io.StringIO()
        with (
            patch.object(provision.shutil, "which", return_value="/usr/bin/crontab"),
            patch.object(provision, "_LEGACY_CAN_UNIT_FILES", ()),
            patch.object(provision, "run_root", return_value=no_crontab) as run_root,
            contextlib.redirect_stdout(output),
        ):
            self.assertFalse(provision._neutralize_legacy_can_root_execution())

        run_root.assert_called_once_with(["env", "LC_ALL=C", "crontab", "-l"])
        self.assertEqual(output.getvalue(), "")

    def test_migration_failure_aborts_before_best_effort_steps(self) -> None:
        with (
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(
                provision,
                "_neutralize_legacy_can_root_execution",
                side_effect=RuntimeError("crontab unreadable"),
            ),
            patch.object(provision, "_step") as step,
            self.assertRaisesRegex(RuntimeError, "crontab unreadable"),
        ):
            provision.run()
        step.assert_not_called()

    def test_step_failures_are_aggregated_after_all_repairs_are_attempted(self) -> None:
        attempted: list[str] = []

        def fail(label: str) -> None:
            attempted.append(label)
            raise RuntimeError(f"{label} failed")

        def succeed(label: str) -> None:
            attempted.append(label)

        with (
            tempfile.TemporaryDirectory() as zed_sdk,
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(provision, "_neutralize_legacy_can_root_execution"),
            patch.object(provision, "_ZED_SDK", Path(zed_sdk)),
            patch.object(provision.adb, "install", side_effect=lambda: succeed("adb")),
            patch.object(
                provision.tracker_install,
                "run",
                side_effect=lambda: fail("tracker"),
            ),
            patch.object(
                provision.zed_driver,
                "ensure_driver",
                side_effect=lambda: succeed("driver"),
            ),
            patch.object(
                provision.gyro, "install", side_effect=lambda: succeed("gyro")
            ),
            patch.object(
                provision.zed_install,
                "run",
                side_effect=lambda: fail("pyzed"),
            ),
            patch.object(
                provision.zed_calibration,
                "share_calibration_files",
                side_effect=lambda: succeed("calibration"),
            ),
            patch.object(
                provision.gst_install,
                "run",
                side_effect=lambda: succeed("gst"),
            ),
            patch.object(
                provision.gst_build_zed,
                "run",
                side_effect=lambda: succeed("gst-build"),
            ),
            patch.object(
                provision,
                "tune_host",
                side_effect=lambda **_: succeed("tuning"),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            provision.run()

        self.assertEqual(
            attempted,
            [
                "adb",
                "tracker",
                "driver",
                "gyro",
                "pyzed",
                "calibration",
                "gst",
                "gst-build",
                "tuning",
            ],
        )
        message = str(raised.exception)
        self.assertIn("Lighthouse tracking (tracker.install)", message)
        self.assertIn("pyzed (zed.install)", message)
        self.assertNotIn("GStreamer + PyGObject", message)


class ProvisionEscalationTest(unittest.TestCase):
    def _holder(self, ready: str) -> Mock:
        holder = Mock()
        holder.stdout.readline.return_value = ready
        return holder

    def test_operator_run_keeps_steps_unprivileged_behind_a_sudo_lock(self) -> None:
        events: list[str] = []
        holder = self._holder("locked\n")
        holder.stdin.close.side_effect = lambda: events.append("released")
        with (
            patch.object(provision.os, "geteuid", return_value=1000),
            patch.object(provision, "prime_sudo", return_value=True),
            patch.object(provision.subprocess, "Popen", return_value=holder) as popen,
            patch.object(provision, "host_update_lock") as direct_lock,
            patch.object(
                provision, "_run_locked", side_effect=lambda: events.append("steps")
            ),
        ):
            provision.run()

        self.assertEqual(events, ["steps", "released"])
        self.assertEqual(
            popen.call_args.args[0],
            [
                "sudo",
                "-n",
                provision.sys.executable,
                "-m",
                "almond_axol.utils.host_update_lock",
            ],
        )
        holder.wait.assert_called_once_with()
        direct_lock.assert_not_called()

    def test_holder_failure_stops_before_any_step(self) -> None:
        holder = self._holder("")
        with (
            patch.object(provision.os, "geteuid", return_value=1000),
            patch.object(provision, "prime_sudo", return_value=True),
            patch.object(provision.subprocess, "Popen", return_value=holder),
            patch.object(provision, "_run_locked") as locked,
            self.assertRaisesRegex(SystemExit, "lock holder did not start"),
        ):
            provision.run()
        locked.assert_not_called()
        holder.stdin.close.assert_called_once_with()

    def test_operator_without_sudo_gets_a_message_not_a_traceback(self) -> None:
        with (
            patch.object(provision.os, "geteuid", return_value=1000),
            patch.object(provision, "prime_sudo", return_value=False),
            patch.object(provision.subprocess, "Popen") as popen,
            patch.object(provision, "_run_locked") as locked,
            self.assertRaisesRegex(SystemExit, "needs sudo"),
        ):
            provision.run()
        popen.assert_not_called()
        locked.assert_not_called()

    def test_lock_contention_is_reported_without_a_traceback(self) -> None:
        with (
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision,
                "host_update_lock",
                side_effect=provision.HostUpdateLockError("another transaction"),
            ),
            patch.object(provision, "_run_locked") as locked,
            self.assertRaisesRegex(SystemExit, "another transaction"),
        ):
            provision.run()
        locked.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class HostTuningTest(unittest.TestCase):
    """`axol provision` is the only command a host needs, Jetson or not."""

    def test_boot_runs_only_the_tuning_without_the_update_lock(self) -> None:
        with (
            patch.object(provision, "tune_host") as tune,
            patch.object(provision, "host_update_lock") as lock,
            patch.object(provision, "_sudo_held_update_lock") as sudo_lock,
            patch.object(provision, "_run_locked") as locked,
        ):
            provision.run(SimpleNamespace(boot=True))
        tune.assert_called_once()
        lock.assert_not_called()
        sudo_lock.assert_not_called()
        locked.assert_not_called()

    def test_inside_serve_the_tuning_is_left_to_the_unit(self) -> None:
        # The self-updater's provision can overlap a robot session; the
        # restart it ends with runs the unit's ExecStartPre tuning anyway.
        with (
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(provision, "_neutralize_legacy_can_root_execution"),
            patch.object(provision, "_step", return_value=True) as step,
            patch.object(provision, "privileged_service_active", return_value=True),
            patch.object(provision, "tune_host") as tune,
        ):
            provision.run()
        labels = [c.args[0] for c in step.call_args_list]
        self.assertFalse(any(label.startswith("host tuning") for label in labels))
        tune.assert_not_called()

    def test_jetson_hosts_get_the_realtime_tuning(self) -> None:
        with (
            patch.object(provision.jetson, "_is_jetson", return_value=True),
            patch.object(
                provision.jetson,
                "host_model",
                return_value="NVIDIA Jetson AGX Thor Developer Kit",
            ),
            patch.object(provision.jetson, "pin_realtime_clocks") as pin,
            self.assertLogs(provision._logger, "INFO") as logs,
        ):
            provision.tune_host(interactive=False)
        pin.assert_called_once_with(interactive=False)
        self.assertIn("AGX Thor", logs.output[0])
        self.assertIn("core layout:", logs.output[-1])

    def test_other_hosts_are_left_alone(self) -> None:
        with (
            patch.object(provision.jetson, "_is_jetson", return_value=False),
            patch.object(
                provision.jetson,
                "host_model",
                return_value="Raspberry Pi 5 Model B Rev 1.0",
            ),
            patch.object(provision.jetson, "pin_realtime_clocks") as pin,
            self.assertLogs(provision._logger, "INFO") as logs,
        ):
            provision.tune_host()
        pin.assert_not_called()
        self.assertIn("no Jetson tuning", logs.output[0])

    def test_jetson_setup_is_an_alias_of_the_boot_tuning(self) -> None:
        from almond_axol.cli.jetson import setup as jetson_setup

        with patch.object(provision, "tune_host") as tune:
            jetson_setup.run()
        tune.assert_called_once_with(interactive=True)

    def test_installer_leaves_tuning_to_provision_and_the_unit(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        self.assertIn("ExecStartPre=-${BIN_DIR}/axol provision --boot", script)
        self.assertIn("After=network-online.target nvpmodel.service", script)
        # No separate operator/installer step any more.
        self.assertNotIn("axol jetson.setup", script)
