import tempfile
from pathlib import Path
from types import SimpleNamespace
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


class VerifySessionTest(TestCase):
    """Provisioning must not report a grant the current session cannot use."""

    def test_sufficient_limit_is_reported_as_available(self) -> None:
        with (
            patch.object(rtprio.os, "geteuid", return_value=1000),
            patch.object(rtprio, "_current_user", return_value="shawn"),
            patch.object(rtprio, "current_limit", return_value=MAX_FIFO_PRIORITY),
            self.assertLogs(rtprio._logger, level="INFO") as logs,
        ):
            self.assertTrue(rtprio.verify_session("shawn"))
        self.assertIn("real-time scheduling available", "\n".join(logs.output))

    def test_zero_limit_warns_with_the_cause_and_both_remedies(self) -> None:
        # The 2026-09-17 case: the file is on disk, the session cannot see it.
        with (
            patch.object(rtprio.os, "geteuid", return_value=1000),
            patch.object(rtprio, "_current_user", return_value="shawn"),
            patch.object(rtprio, "current_limit", return_value=0),
            patch.object(rtprio, "_limit_inherited_from", return_value="tailscaled"),
            patch.object(rtprio, "_raise_session_limits", return_value=[]),
            self.assertLogs(rtprio._logger, level="WARNING") as logs,
        ):
            self.assertFalse(rtprio.verify_session("shawn"))
        message = "\n".join(logs.output)
        self.assertIn("pam_limits", message)
        self.assertIn("tailscaled", message)
        self.assertIn("prlimit", message)

    def test_root_is_silent_because_cap_sys_nice_bypasses_the_limit(self) -> None:
        # Root's own rtprio default is 0 on a stock Ubuntu, so warning here
        # would fire on every curl-installer run and mean nothing.
        with (
            patch.object(rtprio.os, "geteuid", return_value=0),
            patch.object(rtprio, "current_limit", side_effect=AssertionError),
            self.assertNoLogs(rtprio._logger, level="WARNING"),
        ):
            self.assertTrue(rtprio.verify_session("shawn"))

    def test_provisioning_for_another_user_is_silent(self) -> None:
        # Under sudo the grant is for the operator, not for whoever runs this,
        # so this process's limit says nothing about their future logins.
        with (
            patch.object(rtprio.os, "geteuid", return_value=1000),
            patch.object(rtprio, "_current_user", return_value="builder"),
            patch.object(rtprio, "current_limit", side_effect=AssertionError),
            self.assertNoLogs(rtprio._logger, level="WARNING"),
        ):
            self.assertTrue(rtprio.verify_session("shawn"))

    def test_zero_limit_is_fixed_in_place_instead_of_asked_for(self) -> None:
        # The terminal-host case: a shell under node (e.g. an IDE terminal)
        # never runs pam_limits. Provision raises it itself.
        with (
            patch.object(rtprio.os, "geteuid", return_value=1000),
            patch.object(rtprio, "_current_user", return_value="shawn"),
            patch.object(rtprio, "current_limit", return_value=0),
            patch.object(
                rtprio,
                "_raise_session_limits",
                return_value=[(10, "bash"), (9, "node")],
            ),
            self.assertNoLogs(rtprio._logger, level="WARNING"),
            self.assertLogs(rtprio._logger, level="INFO") as logs,
        ):
            self.assertTrue(rtprio.verify_session("shawn"))
        message = "\n".join(logs.output)
        self.assertIn("bash (10), node (9)", message)
        self.assertIn("new terminals from node", message)


def _proc_tree(tree: dict[int, tuple[str, int, int]]):
    """A ``Path.read_text`` stand-in serving ``/proc/<pid>/{comm,status}``."""

    def read_text(self: Path) -> str:
        comm, uid, ppid = tree[int(self.parts[2])]
        if self.parts[3] == "comm":
            return comm + "\n"
        return f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nPPid:\t{ppid}\n"

    return read_text


class RaiseSessionLimitsTest(TestCase):
    def test_raises_owned_ancestors_and_stops_at_other_accounts(self) -> None:
        # bash -> node -> sshd (root): the operator's own processes only.
        tree = {10: ("bash", 1000, 11), 11: ("node", 1000, 12), 12: ("sshd", 0, 1)}
        runs: list[list[str]] = []

        def run_root(argv, **_):
            runs.append(argv)
            return SimpleNamespace(returncode=0)

        with (
            patch.object(rtprio.os, "getuid", return_value=1000),
            patch.object(rtprio.os, "getppid", return_value=10),
            patch.object(Path, "read_text", _proc_tree(tree)),
            patch.object(Path, "exists", return_value=True),
            patch.object(rtprio, "prime_sudo", return_value=True),
            patch.object(rtprio, "run_root", run_root),
        ):
            raised = rtprio._raise_session_limits()
        self.assertEqual(raised, [(10, "bash"), (11, "node")])
        ceiling = f"--rtprio={MAX_FIFO_PRIORITY}:{MAX_FIFO_PRIORITY}"
        self.assertEqual(
            [argv[-3:] for argv in runs],
            [
                ["--pid", "10", ceiling],
                ["--pid", "11", ceiling],
            ],
        )

    def test_stops_at_the_user_systemd_manager(self) -> None:
        tree = {10: ("bash", 1000, 11), 11: ("systemd", 1000, 1)}
        with (
            patch.object(rtprio.os, "getuid", return_value=1000),
            patch.object(rtprio.os, "getppid", return_value=10),
            patch.object(Path, "read_text", _proc_tree(tree)),
        ):
            self.assertEqual(rtprio._owned_ancestors(), [(10, "bash")])

    def test_without_sudo_nothing_is_raised(self) -> None:
        with (
            patch.object(rtprio, "_owned_ancestors", return_value=[(10, "bash")]),
            patch.object(Path, "exists", return_value=True),
            patch.object(rtprio, "prime_sudo", return_value=False),
            patch.object(rtprio, "run_root", side_effect=AssertionError),
        ):
            self.assertEqual(rtprio._raise_session_limits(), [])


class LimitInheritedFromTest(TestCase):
    def test_skips_shells_and_names_the_real_ancestor(self) -> None:
        # bash -> tailscaled: the shell only passes the limit down, so the
        # ancestor worth naming is the one that had it.
        tree = {10: ("bash", 11), 11: ("tailscaled", 1)}

        def read_text(self: Path) -> str:
            parts = self.parts
            pid = int(parts[2])
            if parts[3] == "comm":
                return tree[pid][0] + "\n"
            return f"Name:\t{tree[pid][0]}\nPPid:\t{tree[pid][1]}\n"

        with (
            patch.object(rtprio.os, "getppid", return_value=10),
            patch.object(Path, "read_text", read_text),
        ):
            self.assertEqual(rtprio._limit_inherited_from(), "tailscaled")

    def test_unreadable_proc_is_not_fatal(self) -> None:
        with (
            patch.object(rtprio.os, "getppid", return_value=10),
            patch.object(Path, "read_text", side_effect=OSError),
        ):
            self.assertIsNone(rtprio._limit_inherited_from())


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

    def test_an_already_present_grant_still_verifies_the_session(self) -> None:
        # The exact 2026-09-17 shape: the drop-in was written days earlier, so
        # `axol provision` took this path and reported success while the
        # session it ran in stayed at zero.
        self.limits.parent.mkdir(parents=True)
        self.limits.write_text(rtprio.limits_text("shawn"))
        with (
            patch.object(rtprio, "prime_sudo", side_effect=AssertionError),
            patch.object(rtprio, "run_root", side_effect=AssertionError),
            patch.object(rtprio, "verify_session") as verify,
        ):
            rtprio.install()
        verify.assert_called_once_with("shawn")

    def test_a_freshly_written_grant_verifies_the_session(self) -> None:
        self.runs = []
        with (
            patch.object(rtprio, "prime_sudo", return_value=True),
            patch.object(rtprio, "run_root", self._run_root),
            patch.object(rtprio, "verify_session") as verify,
        ):
            rtprio.install()
        verify.assert_called_once_with("shawn")

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
