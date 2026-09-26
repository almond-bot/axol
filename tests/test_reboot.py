"""Reboot-required bookkeeping and ``axol provision``'s reboot decision."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from almond_axol.cli import provision
from almond_axol.utils import reboot


class _FakeRoot:
    """Stand-in for ``run_root`` that applies file commands under a temp dir."""

    def __init__(self) -> None:
        self.rebooted = 0

    def __call__(self, cmd, *, input_text=None, check=False):
        match cmd:
            case ["mkdir", "-p", path]:
                Path(path).mkdir(parents=True, exist_ok=True)
            case ["tee", "-a", path]:
                with open(path, "a") as f:
                    f.write(input_text or "")
            case ["tee", path]:
                Path(path).write_text(input_text or "")
            case ["rm", "-f", path]:
                Path(path).unlink(missing_ok=True)
            case ["systemctl", "reboot"]:
                self.rebooted += 1
            case _:
                raise AssertionError(f"unexpected root command {cmd}")
        return subprocess.CompletedProcess(cmd, 0, "", "")


class RebootStateTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.root = _FakeRoot()
        for target, value in (
            ("MARKER", root / "run" / "reboot-required"),
            ("_LAST_ATTEMPT", root / "lib" / "last-auto-reboot"),
            ("run_root", self.root),
        ):
            p = patch.object(reboot, target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_requests_accumulate_without_duplicates(self) -> None:
        self.assertEqual(reboot.pending(), [])
        reboot.request("driver")
        reboot.request("power mode")
        reboot.request("driver")
        self.assertEqual(reboot.pending(), ["driver", "power mode"])

    def test_same_reasons_never_reboot_twice(self) -> None:
        # A step that still asks after the reboot it caused would otherwise
        # loop the host through the service's startup provision.
        self.assertTrue(reboot.reboot_host(["driver"]))
        with self.assertLogs(reboot._logger, "WARNING"):
            self.assertFalse(reboot.reboot_host(["driver"]))
        self.assertEqual(self.root.rebooted, 1)
        # A different reason is a new reboot.
        self.assertTrue(reboot.reboot_host(["power mode"]))
        # Once nothing is pending, the guard resets.
        reboot.clear_attempt()
        self.assertTrue(reboot.reboot_host(["driver"]))
        self.assertEqual(self.root.rebooted, 3)


class ProvisionRebootTest(unittest.TestCase):
    def _run(self, args=None, *, pending=(), reboots=True, env=None):
        out = io.StringIO()
        with (
            patch.dict(os.environ, env or {}, clear=False),
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(provision, "_run_locked") as locked,
            patch.object(provision.reboot, "pending", return_value=list(pending)),
            patch.object(provision.reboot, "clear_attempt") as clear,
            patch.object(
                provision.reboot, "reboot_host", return_value=reboots
            ) as reboot_host,
            contextlib.redirect_stdout(out),
        ):
            provision.run(args)
        return locked, reboot_host, clear, out.getvalue()

    def setUp(self) -> None:
        p = patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("AXOL_PRIVILEGED_SERVICE", None)

    def test_clean_run_with_a_pending_reason_reboots(self) -> None:
        locked, reboot_host, _, out = self._run(pending=["driver"])
        locked.assert_called_once()
        reboot_host.assert_called_once_with(["driver"])
        self.assertIn("REBOOT REQUIRED: driver. Rebooting now.", out)

    def test_nothing_pending_resets_the_loop_guard(self) -> None:
        _, reboot_host, clear, _ = self._run()
        reboot_host.assert_not_called()
        clear.assert_called_once()

    def test_no_reboot_leaves_it_pending(self) -> None:
        _, reboot_host, _, out = self._run(
            SimpleNamespace(no_reboot=True), pending=["driver"]
        )
        reboot_host.assert_not_called()
        self.assertIn("Left pending for the caller", out)

    def test_never_reboots_under_axol_serve_even_from_an_old_updater(self) -> None:
        # Pre-`--no-reboot` serve builds still mark their children.
        _, reboot_host, _, _ = self._run(
            pending=["driver"], env={"AXOL_PRIVILEGED_SERVICE": "1"}
        )
        reboot_host.assert_not_called()

    def test_apply_reboot_provisions_nothing(self) -> None:
        locked, reboot_host, _, _ = self._run(
            SimpleNamespace(apply_reboot=True), pending=["driver"]
        )
        locked.assert_not_called()
        reboot_host.assert_called_once_with(["driver"])

    def test_loop_guard_is_an_error(self) -> None:
        with self.assertRaisesRegex(SystemExit, "not rebooting again"):
            self._run(pending=["driver"], reboots=False)

    def test_failed_run_never_reboots_and_says_one_is_pending(self) -> None:
        with (
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(provision, "_neutralize_legacy_can_root_execution"),
            patch.object(provision, "_step", return_value=False),
            patch.object(provision.reboot, "pending", return_value=["driver"]),
            patch.object(provision.reboot, "reboot_host") as reboot_host,
            self.assertRaisesRegex(SystemExit, "A reboot is also pending"),
        ):
            provision.run()
        reboot_host.assert_not_called()

    def test_power_mode_pending_reboot_is_recorded_by_the_tuning_step(self) -> None:
        steps: dict[str, object] = {}

        def record(label, fn):
            steps[label] = fn
            return True

        with (
            patch.object(provision.os, "geteuid", return_value=0),
            patch.object(
                provision, "host_update_lock", return_value=contextlib.nullcontext()
            ),
            patch.object(provision, "_neutralize_legacy_can_root_execution"),
            patch.object(provision, "_step", side_effect=record),
            patch.object(provision, "privileged_service_active", return_value=False),
            patch.object(provision, "_reboot_if_pending"),
        ):
            provision.run()
        tuning = next(fn for label, fn in steps.items() if label.startswith("host"))
        with (
            patch.object(provision, "tune_host", return_value=True),
            patch.object(provision.reboot, "request") as request,
        ):
            tuning()
        request.assert_called_once_with("Jetson maximum power mode")

    def test_boot_tuning_never_records_or_reboots(self) -> None:
        with (
            patch.object(provision, "tune_host", return_value=True),
            patch.object(provision.reboot, "request") as request,
            patch.object(provision, "_reboot_if_pending") as decide,
        ):
            provision.run(SimpleNamespace(boot=True))
        request.assert_not_called()
        decide.assert_not_called()

    def test_installer_defers_the_reboot_to_its_last_step(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "web" / "app" / "public" / "install"
        ).read_text()
        self.assertIn("provision --require-rt --no-reboot", script)
        self.assertTrue(
            script.rstrip().endswith('"${BIN_DIR}/axol" provision --apply-reboot')
        )
        self.assertIn(str(reboot.MARKER), script)


if __name__ == "__main__":
    unittest.main()
