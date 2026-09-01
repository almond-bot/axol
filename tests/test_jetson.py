import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from almond_axol.utils import jetson


def _stat_line(tid: int, comm: str, rt_priority: int, policy: int) -> str:
    """A proc(5) ``stat`` line with the given rt_priority (40) / policy (41)."""
    fields = ["S", "1"] + ["0"] * 35  # state .. field 39
    fields += [str(rt_priority), str(policy), "0", "0"]
    return f"{tid} ({comm}) " + " ".join(fields) + "\n"


class ThreadsAtFifoTest(TestCase):
    def _proc(self, tasks: dict[int, tuple[int, int]]) -> Path:
        root = Path(tempfile.mkdtemp())
        for tid, (rt_priority, policy) in tasks.items():
            task = root / "4242" / "task" / str(tid)
            task.mkdir(parents=True)
            # A comm with a space and a ")" is legal and must not shift fields.
            (task / "stat").write_text(
                _stat_line(tid, "nvargus (x) d", rt_priority, policy)
            )
        return root

    def test_all_threads_fifo_at_priority(self) -> None:
        root = self._proc({4242: (6, 1), 4300: (6, 1)})
        self.assertTrue(jetson._threads_at_fifo(4242, 6, proc_root=root))

    def test_any_cfs_thread_is_false(self) -> None:
        root = self._proc({4242: (6, 1), 4300: (0, 0)})
        self.assertFalse(jetson._threads_at_fifo(4242, 6, proc_root=root))

    def test_wrong_priority_is_false(self) -> None:
        root = self._proc({4242: (5, 1)})
        self.assertFalse(jetson._threads_at_fifo(4242, 6, proc_root=root))

    def test_missing_process_is_none(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.assertIsNone(jetson._threads_at_fifo(4242, 6, proc_root=root))


class _Escalator:
    """Records root operations; ``write`` really writes so re-runs can compare."""

    def __init__(self) -> None:
        self.runs: list[list[str]] = []
        self.writes: list[Path] = []

    def run(self, argv, *, input_text=None):
        self.runs.append(argv)
        if argv[0] == "mkdir":
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
        return True, ""

    def write(self, path, value):
        self.writes.append(path)
        path.write_text(value)
        return True, ""


class PrioritizeCaptureDaemonsTest(TestCase):
    def setUp(self) -> None:
        self.unit_dir = Path(tempfile.mkdtemp())
        patches = [
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson, "_SYSTEMD_UNIT_DIR", self.unit_dir),
            patch.object(jetson, "_CAPTURE_DAEMON_UNITS", ("nvargus-daemon.service",)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.dropin = (
            self.unit_dir / "nvargus-daemon.service.d" / jetson._CAPTURE_DAEMON_DROPIN
        )

    def test_installs_dropin_and_reschedules_live_daemon(self) -> None:
        esc = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=False),
        ):
            jetson._prioritize_capture_daemons(esc)

        text = self.dropin.read_text()
        self.assertIn("CPUSchedulingPolicy=fifo", text)
        self.assertIn(
            f"CPUSchedulingPriority={jetson._CAPTURE_DAEMON_FIFO_PRIORITY}", text
        )
        self.assertIn(["systemctl", "daemon-reload"], esc.runs)
        self.assertIn(
            [
                "chrt",
                "-f",
                "-a",
                "-p",
                str(jetson._CAPTURE_DAEMON_FIFO_PRIORITY),
                "777",
            ],
            esc.runs,
        )

    def test_rerun_is_a_no_op_once_applied(self) -> None:
        esc = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=False),
        ):
            jetson._prioritize_capture_daemons(esc)
        again = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=True),
        ):
            jetson._prioritize_capture_daemons(again)
        self.assertEqual(again.runs, [])
        self.assertEqual(again.writes, [])

    def test_absent_daemon_is_skipped_entirely(self) -> None:
        esc = _Escalator()
        with patch.object(jetson, "_service_main_pid", return_value=0):
            jetson._prioritize_capture_daemons(esc)
        self.assertFalse(self.dropin.exists())
        self.assertEqual(esc.runs, [])

    def test_stopped_daemon_with_dropin_only_refreshes_dropin(self) -> None:
        self.dropin.parent.mkdir(parents=True)
        self.dropin.write_text("stale\n")
        esc = _Escalator()
        with patch.object(jetson, "_service_main_pid", return_value=0):
            jetson._prioritize_capture_daemons(esc)
        self.assertIn("CPUSchedulingPolicy=fifo", self.dropin.read_text())
        self.assertNotIn("chrt", [argv[0] for argv in esc.runs])

    def test_gpu_is_pinned_but_not_a_jetson_marker(self) -> None:
        self.assertIn("*.gpu", jetson._ENGINE_CLOCK_GLOBS)
        self.assertNotIn("*.gpu", jetson._JETSON_ENGINE_GLOBS)
