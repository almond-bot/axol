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


class ThreadsOnCpusTest(TestCase):
    def _proc(self, tasks: dict[int, str]) -> Path:
        root = Path(tempfile.mkdtemp())
        for tid, allowed in tasks.items():
            task = root / "4242" / "task" / str(tid)
            task.mkdir(parents=True)
            (task / "status").write_text(
                f"Name:\tnvargus-daemon\nCpus_allowed_list:\t{allowed}\nMems_allowed_list:\t0\n"
            )
        return root

    def test_parse_cpu_list_handles_ranges(self) -> None:
        self.assertEqual(jetson._parse_cpu_list("0-2,5"), {0, 1, 2, 5})
        self.assertEqual(jetson._parse_cpu_list(" 7 "), {7})
        self.assertEqual(jetson._cpu_list({5, 1}), "1,5")

    def test_all_threads_confined(self) -> None:
        root = self._proc({4242: "1,5", 4300: "1,5"})
        self.assertTrue(jetson._threads_on_cpus(4242, {1, 5}, proc_root=root))

    def test_roaming_thread_is_false(self) -> None:
        root = self._proc({4242: "1,5", 4300: "0-7"})
        self.assertFalse(jetson._threads_on_cpus(4242, {1, 5}, proc_root=root))

    def test_missing_process_is_none(self) -> None:
        root = Path(tempfile.mkdtemp())
        self.assertIsNone(jetson._threads_on_cpus(4242, {1, 5}, proc_root=root))


_INTERRUPTS = """\
           CPU0       CPU1       CPU2       CPU3
 11:    5000000    4999000    5001000    5000500     GICv3  27 Level     arch_timer
123:    7900000          0          0          0     GICv3 251 Level     xhci-hcd:usb1
124:          0          0          0          0     GICv3 252 Level     xhci-hcd:usb2
200:          0        120          0          0     GICv3 300 Level     tegra-can-ish
ERR:          0
"""


def _fake_proc(
    interrupts: str | None,
    affinity: dict[int, str] = {},
    effective: dict[int, str] = {},
) -> Path:
    """A ``/proc`` stand-in: an interrupts table plus per-irq affinity files.

    ``effective`` adds ``effective_affinity_list`` nodes (the CPU the GIC
    actually picked out of the nominal mask) for the given irqs.
    """
    root = Path(tempfile.mkdtemp())
    if interrupts is not None:
        (root / "interrupts").write_text(interrupts)
    for irq, cpus in affinity.items():
        (root / "irq" / str(irq)).mkdir(parents=True, exist_ok=True)
        (root / "irq" / str(irq) / "smp_affinity_list").write_text(cpus + "\n")
    for irq, cpus in effective.items():
        (root / "irq" / str(irq)).mkdir(parents=True, exist_ok=True)
        (root / "irq" / str(irq) / "effective_affinity_list").write_text(cpus + "\n")
    return root


def _fake_sys(devices: dict[str, str]) -> Path:
    """A ``/sys`` stand-in where each CAN interface's ``device`` link resolves
    to a USB function directory named like the kernel does (``1-2.2:1.0``)."""
    root = Path(tempfile.mkdtemp())
    for iface, function in devices.items():
        target = root / "bus/usb/devices" / function
        target.mkdir(parents=True)
        net = root / "class/net" / iface
        net.mkdir(parents=True)
        (net / "device").symlink_to(target)
    return root


class CanUsbIrqsTest(TestCase):
    def test_resolves_the_bus_from_the_interfaces_usb_function(self) -> None:
        sys_root = _fake_sys(
            {jetson.CAN_LEFT: "1-2.2:1.0", jetson.CAN_RIGHT: "1-2.2:1.1"}
        )
        self.assertEqual(jetson._can_usb_buses(sys_root=sys_root), {"1"})

    def test_only_the_controller_the_adapters_hang_off(self) -> None:
        # usb2 is a different controller: it must not be steered.
        sys_root = _fake_sys({jetson.CAN_LEFT: "1-2.2:1.0"})
        root = _fake_proc(_INTERRUPTS)
        self.assertEqual(
            jetson._can_usb_irqs(proc_root=root, sys_root=sys_root),
            {123: "xhci-hcd:usb1"},
        )

    def test_every_xhci_row_when_no_interface_resolves(self) -> None:
        root = _fake_proc(_INTERRUPTS)
        self.assertEqual(
            jetson._can_usb_irqs(proc_root=root, sys_root=Path(tempfile.mkdtemp())),
            {123: "xhci-hcd:usb1", 124: "xhci-hcd:usb2"},
        )

    def test_unreadable_table_is_empty(self) -> None:
        root = _fake_proc(None)
        self.assertEqual(
            jetson._can_usb_irqs(proc_root=root, sys_root=Path(tempfile.mkdtemp())),
            {},
        )

    def test_irq_affinity_parses_and_tolerates_absence(self) -> None:
        root = _fake_proc(None, {123: "0-7"})
        self.assertEqual(jetson._irq_affinity(123, proc_root=root), set(range(8)))
        self.assertIsNone(jetson._irq_affinity(9, proc_root=root))

    def test_effective_affinity_prefers_the_gic_choice(self) -> None:
        # Nominal mask says "anywhere"; the GIC actually delivers to CPU0.
        root = _fake_proc(None, {123: "0-7"}, {123: "0"})
        self.assertEqual(jetson._irq_effective_affinity(123, proc_root=root), {0})
        # No effective node (or an empty one): fall back to the nominal mask.
        root = _fake_proc(None, {124: "0-7"}, {124: ""})
        self.assertEqual(
            jetson._irq_effective_affinity(124, proc_root=root), set(range(8))
        )
        self.assertIsNone(jetson._irq_effective_affinity(9, proc_root=root))


class CanIrqCpusTest(TestCase):
    """Live placement of the CAN adapters' interrupt, as the camera pool sees it."""

    def setUp(self) -> None:
        self.sys_root = _fake_sys({jetson.CAN_LEFT: "1-2.2:1.0"})

    def test_unsteered_interrupt_reports_cpu0(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "0-7", 124: "0-7"}, {123: "0"})
        self.assertEqual(
            jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root), {0}
        )

    def test_nominal_mask_when_the_kernel_tracks_no_effective_one(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "0-7", 124: "0-7"})
        self.assertEqual(
            jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root),
            set(range(8)),
        )

    def test_steered_interrupt_reports_the_can_core_only(self) -> None:
        # The other controller (usb2) stays on CPU0 but is not the CAN one.
        root = _fake_proc(_INTERRUPTS, {123: "7", 124: "0-7"}, {123: "7", 124: "0"})
        self.assertEqual(
            jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root), {7}
        )

    def test_every_xhci_row_counts_when_no_interface_resolves(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "7", 124: "0-7"}, {123: "7", 124: "0"})
        self.assertEqual(
            jetson.can_irq_cpus(proc_root=root, sys_root=Path(tempfile.mkdtemp())),
            {0, 7},
        )

    def test_unknown_when_no_row_or_no_affinity(self) -> None:
        root = _fake_proc("           CPU0\n 11:   1   GICv3  arch_timer\n")
        self.assertIsNone(jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root))
        root = _fake_proc(_INTERRUPTS)  # row present, affinity files missing
        self.assertIsNone(jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root))
        root = _fake_proc(None)
        self.assertIsNone(jetson.can_irq_cpus(proc_root=root, sys_root=self.sys_root))


class SteerCanIrqTest(TestCase):
    def setUp(self) -> None:
        self.sys_root = _fake_sys({jetson.CAN_LEFT: "1-2.2:1.0"})
        patches = [
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson, "can_irq_cpu", return_value=7),
            patch.object(jetson, "_irqbalance_active", return_value=False),
            patch.object(jetson, "_SYS_ROOT", self.sys_root),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_steers_the_can_controller_irq_onto_the_can_core(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "0-7", 124: "0-7"})
        esc = _Escalator()
        with patch.object(jetson, "_PROC_ROOT", root):
            jetson._steer_can_irq(esc)
        self.assertEqual(esc.writes, [root / "irq/123/smp_affinity_list"])
        self.assertEqual(jetson._irq_affinity(123, proc_root=root), {7})
        # The other controller was left alone.
        self.assertEqual(jetson._irq_affinity(124, proc_root=root), set(range(8)))

    def test_rerun_is_a_no_op_once_applied(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "7"})
        esc = _Escalator()
        with patch.object(jetson, "_PROC_ROOT", root):
            jetson._steer_can_irq(esc)
        self.assertEqual(esc.writes, [])

    def test_missing_controller_only_warns(self) -> None:
        root = _fake_proc("           CPU0\n 11:   1   GICv3  arch_timer\n")
        esc = _Escalator()
        with (
            patch.object(jetson, "_PROC_ROOT", root),
            self.assertLogs(jetson._logger, level="WARNING"),
        ):
            jetson._steer_can_irq(esc)
        self.assertEqual(esc.writes, [])

    def test_failed_write_warns_with_the_manual_command(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "0-7"})

        class _Denied(_Escalator):
            def write(self, path, value):
                return False, "sudo unavailable"

        with (
            patch.object(jetson, "_PROC_ROOT", root),
            self.assertLogs(jetson._logger, level="WARNING") as logs,
        ):
            jetson._steer_can_irq(_Denied())
        self.assertIn("echo 7 | sudo tee", "\n".join(logs.output))

    def test_irqbalance_is_called_out(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "7"})
        with (
            patch.object(jetson, "_PROC_ROOT", root),
            patch.object(jetson, "_irqbalance_active", return_value=True),
            self.assertLogs(jetson._logger, level="WARNING") as logs,
        ):
            jetson._steer_can_irq(_Escalator())
        self.assertIn("irqbalance", "\n".join(logs.output))

    def test_no_can_partition_is_a_no_op(self) -> None:
        root = _fake_proc(_INTERRUPTS, {123: "0-7"})
        esc = _Escalator()
        with (
            patch.object(jetson, "_PROC_ROOT", root),
            patch.object(jetson, "can_irq_cpu", return_value=None),
        ):
            jetson._steer_can_irq(esc)
        self.assertEqual(esc.writes, [])


class PrioritizeCaptureDaemonsTest(TestCase):
    def setUp(self) -> None:
        self.unit_dir = Path(tempfile.mkdtemp())
        patches = [
            patch.object(jetson, "_is_jetson", return_value=True),
            patch.object(jetson, "_SYSTEMD_UNIT_DIR", self.unit_dir),
            patch.object(jetson, "_CAPTURE_DAEMON_UNITS", ("nvargus-daemon.service",)),
            patch.object(jetson, "realtime_camera_cores", return_value={1, 5}),
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
            patch.object(jetson, "_threads_on_cpus", return_value=False),
        ):
            jetson._prioritize_capture_daemons(esc)

        text = self.dropin.read_text()
        self.assertIn("CPUSchedulingPolicy=fifo", text)
        self.assertIn(
            f"CPUSchedulingPriority={jetson._CAPTURE_DAEMON_FIFO_PRIORITY}", text
        )
        self.assertIn("CPUAffinity=1 5", text)
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
        self.assertIn(["taskset", "-a", "-c", "-p", "1,5", "777"], esc.runs)

    def test_rerun_is_a_no_op_once_applied(self) -> None:
        esc = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=False),
            patch.object(jetson, "_threads_on_cpus", return_value=False),
        ):
            jetson._prioritize_capture_daemons(esc)
        again = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=True),
            patch.object(jetson, "_threads_on_cpus", return_value=True),
        ):
            jetson._prioritize_capture_daemons(again)
        self.assertEqual(again.runs, [])
        self.assertEqual(again.writes, [])

    def test_affinity_alone_is_reapplied_when_daemon_roams(self) -> None:
        esc = _Escalator()
        with (
            patch.object(jetson, "_service_main_pid", return_value=777),
            patch.object(jetson, "_threads_at_fifo", return_value=True),
            patch.object(jetson, "_threads_on_cpus", return_value=False),
        ):
            jetson._prioritize_capture_daemons(esc)
        self.assertNotIn("chrt", [argv[0] for argv in esc.runs])
        self.assertIn(["taskset", "-a", "-c", "-p", "1,5", "777"], esc.runs)

    def test_dropin_omits_affinity_without_a_partition(self) -> None:
        esc = _Escalator()
        with (
            patch.object(jetson, "realtime_camera_cores", return_value=None),
            patch.object(jetson, "_service_main_pid", return_value=0),
        ):
            self.dropin.parent.mkdir(parents=True)
            self.dropin.write_text("stale\n")
            jetson._prioritize_capture_daemons(esc)
        self.assertNotIn("CPUAffinity", self.dropin.read_text())

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
