"""ZED calibration cache sharing between the root service and operator logins."""

from __future__ import annotations

import grp
import logging
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from almond_axol.zed import calibration


def _own_group() -> str:
    return grp.getgrgid(os.getgid()).gr_name


def _run_directly(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    """Stand-in for ``run_root`` that runs the command as the test user."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"`{cmd[0]}` failed: {proc.stderr.strip()}")
    return proc


class SharePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = Path(self._tmp.name) / "settings"
        self.settings.mkdir(mode=0o700)
        self.gid = os.getgid()

    def _conf(self, serial: int, mode: int) -> Path:
        path = self.settings / f"SN{serial}.conf"
        path.write_text("[LEFT_CAM_FHD1200]\nfx=1\n")
        path.chmod(mode)
        return path

    def test_root_umask_layout_needs_every_repair(self) -> None:
        # What a fresh hosted install leaves behind: directory without setgid
        # or group rwx, files 0640 (SDK default through the service's 027
        # umask). Group ids are our own here, so only mode repairs remain.
        stereo = self._conf(51617969, 0o640)
        mono = self._conf(302271563, 0o640)
        plan = calibration._share_plan(self.settings, self.gid)
        self.assertEqual(
            plan,
            [
                ["chmod", "g+rwx,g+s", str(self.settings)],
                # Files are listed in sorted (name) order.
                ["chmod", "g+rw", str(mono), str(stereo)],
            ],
        )

    def test_foreign_group_is_regrouped(self) -> None:
        conf = self._conf(51617969, 0o660)
        self.settings.chmod(0o2770)
        foreign_gid = self.gid + 1
        plan = calibration._share_plan(self.settings, foreign_gid)
        self.assertEqual(
            plan,
            [
                ["chgrp", str(foreign_gid), str(self.settings)],
                ["chgrp", str(foreign_gid), str(conf)],
            ],
        )

    def test_shared_layout_is_a_no_op(self) -> None:
        self._conf(51617969, 0o660)
        self.settings.chmod(0o2770)
        self.assertEqual(calibration._share_plan(self.settings, self.gid), [])

    def test_only_calibration_files_are_touched(self) -> None:
        self.settings.chmod(0o2770)
        other = self.settings / "notes.txt"
        other.write_text("x")
        other.chmod(0o600)
        self.assertEqual(calibration._share_plan(self.settings, self.gid), [])


class ShareCalibrationFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = Path(self._tmp.name) / "settings"
        self.settings.mkdir(mode=0o700)
        self.conf = self.settings / "SN51617969.conf"
        self.conf.write_text("[LEFT_CAM_FHD1200]\n")
        self.conf.chmod(0o640)

    def test_applies_group_bits_and_setgid(self) -> None:
        with (
            patch.object(calibration, "prime_sudo", return_value=True),
            patch.object(calibration, "run_root", side_effect=_run_directly),
        ):
            self.assertTrue(
                calibration.share_calibration_files(
                    directory=self.settings, group=_own_group()
                )
            )
        dir_mode = self.settings.stat().st_mode
        self.assertTrue(dir_mode & stat.S_ISGID)
        self.assertEqual(dir_mode & 0o070, 0o070)
        self.assertEqual(self.conf.stat().st_mode & 0o060, 0o060)
        # Second run finds nothing to do and never escalates.
        with (
            patch.object(calibration, "prime_sudo") as prime,
            patch.object(calibration, "run_root") as run_root,
        ):
            self.assertTrue(
                calibration.share_calibration_files(
                    directory=self.settings, group=_own_group()
                )
            )
        prime.assert_not_called()
        run_root.assert_not_called()

    def test_missing_cache_is_a_quiet_success(self) -> None:
        with patch.object(calibration, "run_root") as run_root:
            self.assertTrue(
                calibration.share_calibration_files(
                    directory=self.settings / "absent", group=_own_group()
                )
            )
        run_root.assert_not_called()

    def test_unknown_group_is_reported_not_raised(self) -> None:
        with (
            patch.object(calibration, "run_root") as run_root,
            self.assertLogs(calibration._logger, level=logging.WARNING) as logs,
        ):
            self.assertFalse(
                calibration.share_calibration_files(
                    directory=self.settings, group="no-such-group-axol"
                )
            )
        run_root.assert_not_called()
        self.assertIn("no `no-such-group-axol` group", logs.output[0])

    def test_no_escalation_path_logs_the_manual_fix(self) -> None:
        with (
            patch.object(calibration, "prime_sudo", return_value=False),
            patch.object(calibration, "run_root") as run_root,
            self.assertLogs(calibration._logger, level=logging.WARNING) as logs,
        ):
            self.assertFalse(
                calibration.share_calibration_files(
                    directory=self.settings, group=_own_group()
                )
            )
        run_root.assert_not_called()
        self.assertIn("sudo axol provision", logs.output[0])
        self.assertIn(str(self.settings), logs.output[0])

    def test_command_failure_is_reported_not_raised(self) -> None:
        with (
            patch.object(calibration, "prime_sudo", return_value=True),
            patch.object(
                calibration, "run_root", side_effect=RuntimeError("`chmod` failed")
            ),
            self.assertLogs(calibration._logger, level=logging.WARNING),
        ):
            self.assertFalse(
                calibration.share_calibration_files(
                    directory=self.settings, group=_own_group()
                )
            )


class CalibrationHintTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.settings = Path(self._tmp.name)

    def test_readable_or_missing_file_gives_no_hint(self) -> None:
        conf = self.settings / "SN51617969.conf"
        conf.write_text("x")
        self.assertEqual(calibration.calibration_hint(51617969, self.settings), "")
        self.assertEqual(calibration.calibration_hint(1, self.settings), "")
        self.assertEqual(calibration.calibration_hint(None, self.settings), "")

    @unittest.skipIf(os.geteuid() == 0, "root can read anything")
    def test_unreadable_file_names_the_file_and_the_fix(self) -> None:
        conf = self.settings / "SN51617969.conf"
        conf.write_text("x")
        conf.chmod(0o000)
        self.addCleanup(conf.chmod, 0o600)
        hint = calibration.calibration_hint(51617969, self.settings)
        self.assertIn(str(conf), hint)
        self.assertIn("not readable by this user", hint)
        self.assertIn("sudo axol provision", hint)
        # No serial: any unreadable cached file counts.
        self.assertIn(str(conf), calibration.calibration_hint(None, self.settings))
        self.assertEqual(
            calibration.unreadable_calibration_files(self.settings), [conf]
        )


class EnsureCalibrationReadableTest(unittest.TestCase):
    def test_root_always_reconciles(self) -> None:
        with (
            patch.object(calibration.os, "geteuid", return_value=0),
            patch.object(calibration, "unreadable_calibration_files") as unreadable,
            patch.object(calibration, "share_calibration_files") as share,
        ):
            calibration.ensure_calibration_readable()
        share.assert_called_once_with()
        unreadable.assert_not_called()

    def test_operator_only_acts_on_unreadable_files(self) -> None:
        with (
            patch.object(calibration.os, "geteuid", return_value=1000),
            patch.object(calibration, "unreadable_calibration_files", return_value=[]),
            patch.object(calibration, "share_calibration_files") as share,
        ):
            calibration.ensure_calibration_readable()
        share.assert_not_called()
        with (
            patch.object(calibration.os, "geteuid", return_value=1000),
            patch.object(
                calibration,
                "unreadable_calibration_files",
                return_value=[Path("/usr/local/zed/settings/SN1.conf")],
            ),
            patch.object(calibration, "share_calibration_files") as share,
        ):
            calibration.ensure_calibration_readable()
        share.assert_called_once_with()

    def test_never_raises(self) -> None:
        with (
            patch.object(calibration.os, "geteuid", return_value=0),
            patch.object(
                calibration,
                "share_calibration_files",
                side_effect=RuntimeError("boom"),
            ),
            self.assertLogs(calibration._logger, level=logging.WARNING),
        ):
            calibration.ensure_calibration_readable()


if __name__ == "__main__":
    unittest.main()
