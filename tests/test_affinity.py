import io
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import call, patch

from almond_axol.utils import affinity


def _proc_files(comms: dict[str, str], process_comm: str = "python"):
    """``open`` stand-in serving ``/proc/self/comm`` and per-task ``comm`` files."""

    def _open(path, *args, **kwargs):
        if path == "/proc/self/comm":
            return io.StringIO(process_comm + "\n")
        prefix = "/proc/self/task/"
        if path.startswith(prefix) and path.endswith("/comm"):
            tid = path[len(prefix) : -len("/comm")]
            if tid in comms:
                return io.StringIO(comms[tid] + "\n")
            raise FileNotFoundError(path)
        raise AssertionError(f"unexpected open({path!r})")

    return _open


class PrioritizeCaptureThreadsTest(TestCase):
    comms = {
        "101": "python",  # Python main thread (excluded via threading)
        "201": "camsrc:src",  # zedsrc streaming thread
        "202": "camsrc:src",  # a second camera's
        "203": "python",  # ZED SDK worker: never renamed its comm
        "204": "eye_l_cropq:src",  # crop VIC dispatch: holds a camera surface
        "205": "V4L2_EncThread",  # NVENC: stays CFS
        "206": "cuda-EvtHandlr",
        "207": "dsenc_l_srcq:sr",  # 15-char comm truncation of dsenc_l_srcq:src
        "208": "dsenc_l_outq:sr",  # post-encode: stays CFS
    }
    python_threads = [SimpleNamespace(native_id=101)]
    wanted = ("camsrc:src", "eye_l_cropq:src", "dsenc_l_srcq:sr")

    def test_elevates_capture_chain_and_sdk_threads_only(self) -> None:
        # "999" is listed but exited before its comm is read; "x" isn't a tid.
        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(
                affinity.os, "listdir", return_value=[*self.comms, "999", "x"]
            ),
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
            patch.object(
                affinity.os,
                "sched_param",
                create=True,
                side_effect=lambda p: ("param", p),
            ),
            patch.object(affinity.os, "SCHED_FIFO", 1, create=True),
            patch("builtins.open", _proc_files(self.comms)),
            patch("threading.enumerate", return_value=self.python_threads),
        ):
            moved = affinity.prioritize_capture_threads(self.wanted)

        self.assertEqual(moved, 5)
        elevated = (201, 202, 203, 204, 207)
        param = ("param", affinity.CAPTURE_FIFO_PRIORITY)
        self.assertCountEqual(
            setsched.call_args_list,
            [call(tid, 1, param) for tid in elevated],
        )
        # Every FIFO thread is kept off CPU0, where the CAN adapters' softirq
        # bottom half runs as CFS work, and off the relay's Python core.
        self.assertCountEqual(
            set_affinity.call_args_list,
            [call(tid, {1, 5}) for tid in elevated],
        )

    def test_permission_denied_leaves_threads_cfs(self) -> None:
        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(affinity.os, "listdir", return_value=list(self.comms)),
            patch.object(
                affinity.os,
                "sched_setscheduler",
                create=True,
                side_effect=PermissionError("EPERM"),
            ) as setsched,
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
            patch.object(
                affinity.os, "sched_param", create=True, side_effect=lambda p: p
            ),
            patch.object(affinity.os, "SCHED_FIFO", 1, create=True),
            patch("builtins.open", _proc_files(self.comms)),
            patch("threading.enumerate", return_value=self.python_threads),
            self.assertLogs(affinity._logger, level="INFO") as logs,
        ):
            moved = affinity.prioritize_capture_threads(self.wanted)

        self.assertEqual(moved, 0)
        self.assertEqual(setsched.call_count, 1)  # stops at the first EPERM
        self.assertEqual(set_affinity.call_count, 0)  # CFS threads keep the pool
        self.assertTrue(any("CAP_SYS_NICE" in line for line in logs.output))

    def test_noop_without_scheduler_api(self) -> None:
        with patch.object(affinity, "os", SimpleNamespace(listdir=lambda _p: [])):
            self.assertEqual(affinity.prioritize_capture_threads(self.wanted), 0)


class RealtimeCameraCoresTest(TestCase):
    """The FIFO camera pool follows the *live* placement of the CAN interrupt."""

    def _cores(self, n: int, irq_cpus: set[int] | None) -> set[int] | None:
        with (
            patch.object(affinity.os, "cpu_count", return_value=n),
            patch("almond_axol.utils.jetson.can_irq_cpus", return_value=irq_cpus),
        ):
            return affinity.realtime_camera_cores()

    def test_eight_cores_excludes_irq_cpu_while_the_interrupt_can_land_there(
        self,
    ) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=8):
            groups = affinity.core_groups()
        assert groups is not None
        self.assertEqual(groups["irq"], {0})
        # Unsteered: the GIC delivers to CPU0 (effective) or anywhere (nominal).
        for irq_cpus in ({0}, set(range(8))):
            cores = self._cores(8, irq_cpus)
            assert cores is not None
            self.assertEqual(cores, {1, 5}, irq_cpus)
            # Disjoint from everything a FIFO camera thread must never preempt.
            for group in ("can", "realtime", "ik", "irq"):
                self.assertTrue(cores.isdisjoint(groups[group]), group)
            self.assertNotIn(min(groups["relay"]), cores)

    def test_unknown_interrupt_placement_keeps_cpu0_excluded(self) -> None:
        # No /proc row, unreadable affinity, not a Jetson: the conservative
        # (pre-existing) layout.
        self.assertEqual(self._cores(8, None), {1, 5})

    def test_steered_interrupt_returns_cpu0_to_the_camera_pool(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=8):
            groups = affinity.core_groups()
        assert groups is not None
        cores = self._cores(8, {affinity.can_irq_cpu()})
        assert cores is not None
        self.assertEqual(cores, {0, 1, 5})
        # Still never a control, IK, CAN, or relay-Python core.
        for group in ("can", "realtime", "ik"):
            self.assertTrue(cores.isdisjoint(groups[group]), group)
        self.assertNotIn(min(groups["relay"]), cores)

    def test_smaller_layouts_still_avoid_cpu0(self) -> None:
        # Below 8 cores CPU0 is a CAN core, so steering never changes the pool.
        for n, expected in ((6, {4, 5}), (5, {3, 4}), (4, {3})):
            for irq_cpus in (None, {0}, {n - 1}):
                self.assertEqual(self._cores(n, irq_cpus), expected, (n, irq_cpus))

    def test_none_when_partitioning_is_not_applicable(self) -> None:
        self.assertIsNone(self._cores(2, None))


class CanIrqCpuTest(TestCase):
    def test_is_the_highest_can_core_and_never_a_camera_core(self) -> None:
        for n in (8, 6, 5, 4):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
                target = affinity.can_irq_cpu()
                assert target is not None
                # Whether the interrupt is still on CPU0 or already steered
                # onto the CAN core, that core is never a camera core.
                for irq_cpus in (None, {0}, {target}):
                    with patch(
                        "almond_axol.utils.jetson.can_irq_cpus",
                        return_value=irq_cpus,
                    ):
                        cameras = affinity.realtime_camera_cores()
                    assert cameras is not None
                    self.assertNotIn(target, cameras, (n, irq_cpus))
            assert groups is not None
            self.assertEqual(target, max(groups["can"]), n)
            self.assertNotIn(target, groups["irq"], n)

    def test_none_without_a_can_partition(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=2):
            self.assertIsNone(affinity.can_irq_cpu())


class IsolateRelayCpuTest(TestCase):
    def test_gstreamer_threads_share_relay_and_background_cores(self) -> None:
        python_threads = [
            SimpleNamespace(native_id=101),
            SimpleNamespace(native_id=102),
        ]

        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(
                affinity.os,
                "listdir",
                return_value=["101", "102", "201", "202", "not-a-tid"],
            ),
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
            patch("threading.enumerate", return_value=python_threads),
        ):
            self.assertTrue(affinity.isolate_relay_cpu())

        self.assertCountEqual(
            set_affinity.call_args_list[:2],
            [call(101, {4}), call(102, {4})],
        )
        self.assertEqual(
            set_affinity.call_args_list[2:],
            [call(201, {0, 1, 5}), call(202, {0, 1, 5})],
        )

    def test_gstreamer_set_excludes_python_core_when_groups_overlap(self) -> None:
        groups = {
            "relay": {2, 3},
            "background": {2, 3},
        }
        python_threads = [SimpleNamespace(native_id=101)]

        with (
            patch.object(affinity, "core_groups", return_value=groups),
            patch.object(affinity.os, "listdir", return_value=["101", "201"]),
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
            patch("threading.enumerate", return_value=python_threads),
        ):
            self.assertTrue(affinity.isolate_relay_cpu())

        self.assertEqual(
            set_affinity.call_args_list,
            [call(101, {2}), call(201, {3})],
        )
