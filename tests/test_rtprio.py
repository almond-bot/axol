import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from almond_axol.utils import rtprio
from almond_axol.utils.affinity import MAX_FIFO_PRIORITY


class OperatorUserTest(TestCase):
    def test_prefers_sudo_user(self) -> None:
        with patch.dict(rtprio.os.environ, {"SUDO_USER": "shawn"}):
            self.assertEqual(rtprio.operator_user(), "shawn")

    def test_root_sudo_user_falls_through_to_home(self) -> None:
        home = Path(tempfile.mkdtemp())
        (home / "alice").mkdir()
        with (
            patch.dict(rtprio.os.environ, {"SUDO_USER": "root"}),
            patch.object(Path, "iterdir", lambda self: iter([home / "alice"])),
            patch.object(Path, "owner", lambda self: "alice"),
        ):
            self.assertEqual(rtprio.operator_user(), "alice")

    def test_none_without_homes(self) -> None:
        with (
            patch.dict(rtprio.os.environ, {}, clear=True),
            patch.object(Path, "iterdir", side_effect=OSError),
        ):
            self.assertIsNone(rtprio.operator_user())


class InstallTest(TestCase):
    def setUp(self) -> None:
        self.limits = Path(tempfile.mkdtemp()) / "limits.d" / "50-axol-rtprio.conf"
        patches = [
            patch.object(rtprio, "LIMITS_PATH", self.limits),
            patch.object(rtprio, "operator_user", return_value="shawn"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run_root(self, argv, *, input_text=None, check=False):
        self.runs.append(argv)
        if argv[0] == "mkdir":
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
        elif argv[0] == "tee":
            Path(argv[-1]).write_text(input_text)
        return True

    def test_writes_the_grant_at_the_stacks_fifo_ceiling(self) -> None:
        self.runs: list[list[str]] = []
        with (
            patch.object(rtprio, "prime_sudo", return_value=True),
            patch.object(rtprio, "run_root", self._run_root),
        ):
            rtprio.install()
        text = self.limits.read_text()
        self.assertIn(f"shawn\t-\trtprio\t{MAX_FIFO_PRIORITY}\n", text)
        # Matches what the relay's capture chain and the CAN loops request.
        self.assertGreaterEqual(MAX_FIFO_PRIORITY, 20)

    def test_rerun_is_a_no_op(self) -> None:
        self.limits.parent.mkdir(parents=True)
        self.limits.write_text(rtprio.limits_text("shawn"))
        with (
            patch.object(rtprio, "prime_sudo", side_effect=AssertionError),
            patch.object(rtprio, "run_root", side_effect=AssertionError),
        ):
            rtprio.install()

    def test_drifted_file_is_rewritten(self) -> None:
        self.runs = []
        self.limits.parent.mkdir(parents=True)
        self.limits.write_text("shawn - rtprio 5\n")
        with (
            patch.object(rtprio, "prime_sudo", return_value=True),
            patch.object(rtprio, "run_root", self._run_root),
        ):
            rtprio.install()
        self.assertEqual(self.limits.read_text(), rtprio.limits_text("shawn"))

    def test_no_root_only_warns_with_the_manual_command(self) -> None:
        with (
            patch.object(rtprio, "prime_sudo", return_value=False),
            patch.object(rtprio, "run_root", side_effect=AssertionError),
            self.assertLogs(rtprio._logger, level="WARNING") as logs,
        ):
            rtprio.install()
        self.assertIn("sudo tee", "\n".join(logs.output))
        self.assertFalse(self.limits.exists())

    def test_no_operator_is_skipped(self) -> None:
        with (
            patch.object(rtprio, "operator_user", return_value=None),
            patch.object(rtprio, "prime_sudo", side_effect=AssertionError),
        ):
            rtprio.install()
        self.assertFalse(self.limits.exists())
