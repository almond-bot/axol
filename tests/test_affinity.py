import io
from contextlib import ExitStack
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

    def test_larger_jetsons_add_their_extra_cores_to_the_pool(self) -> None:
        # AGX Orin 64GB (12) and Thor T5000 (14): the middle cores join the
        # 8-core pool except the recorder's pair; CPU0 follows the interrupt
        # exactly as on the Orin NX.
        for n in (12, 14):
            extra = set(range(6, n - 4))
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
                steered = {affinity.can_irq_cpu()}
            assert groups is not None
            self.assertEqual(self._cores(n, {0}), {1, 5} | extra, n)
            self.assertEqual(self._cores(n, None), {1, 5} | extra, n)
            cores = self._cores(n, steered)
            self.assertEqual(cores, {0, 1, 5} | extra, n)
            assert cores is not None
            for group in ("can", "realtime", "ik"):
                self.assertTrue(cores.isdisjoint(groups[group]), (n, group))
            self.assertNotIn(min(groups["relay"]), cores)

    def test_smaller_layouts_still_avoid_cpu0(self) -> None:
        # 6-7 cores: CPU0 is a CAN core, so steering never changes the pool.
        # 5 cores: CPU0 is the relay's Python core and the interrupt CPU, so
        # the one throughput core left is the whole pool. 4 cores (Pi 5): no
        # CPU is free of control, CAN, relay-Python or the interrupt — no
        # FIFO camera pool at all (the Pi runs no ZED cameras).
        for n, expected in ((6, {4, 5}), (5, {2}), (4, None)):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                steered = affinity.can_irq_cpu()
            assert steered is not None
            for irq_cpus in (None, {0}, {steered}):
                self.assertEqual(self._cores(n, irq_cpus), expected, (n, irq_cpus))

    def test_an_interrupt_moved_onto_a_camera_core_leaves_the_pool(self) -> None:
        # irqbalance (or an operator) can move the CAN interrupt after
        # jetson.setup steered it. A FIFO camera thread on that CPU stalls
        # both arms' feedback — the 2026-09-02 fault — so the pool drops it.
        self.assertEqual(self._cores(8, {5}), {0, 1})
        self.assertEqual(self._cores(12, {6}), {0, 1, 5, 7})
        self.assertEqual(self._cores(6, {5}), {4})
        # Two controllers on two CPUs: both are left out.
        self.assertEqual(self._cores(14, {5, 13}), {0, 1, 6, 7, 8, 9})
        # A wide nominal mask is the unsteered default: the GIC picks CPU0.
        self.assertEqual(self._cores(12, set(range(12))), {1, 5, 6, 7})

    def test_none_when_partitioning_is_not_applicable(self) -> None:
        self.assertIsNone(self._cores(2, None))


class CoreGroupsTest(TestCase):
    """Every partitioned host gives the Rust core two CAN CPUs of its own."""

    def test_can_cores_are_a_disjoint_pair_on_every_layout(self) -> None:
        # rt.link only exports AXOL_RT_CPU_LEFT/RIGHT + the SCHED_FIFO request
        # when there are two CAN cores disjoint from Python control. The old
        # 4-5 core layout shared them (can == realtime), which left a Pi 5's
        # bus loops as unpinned CFS threads: a 43.8 ms overrun sent the core
        # limp mid-ROM-sweep, and every >0.5 ms late tick had already gated
        # the shoulder host damping off.
        for n in (4, 5, 6, 8, 12, 14):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
            assert groups is not None
            self.assertEqual(len(groups["can"]), 2, n)
            self.assertTrue(groups["can"].isdisjoint(groups["realtime"]), n)
            self.assertTrue(groups["can"].isdisjoint(groups["ik"]), n)
            self.assertTrue(groups["can"].isdisjoint(groups["relay"]), n)
            self.assertTrue(groups["can"].isdisjoint(groups["background"]), n)
            # Control never shares with throughput work either.
            self.assertTrue(groups["realtime"].isdisjoint(groups["relay"]), n)
            self.assertTrue(groups["realtime"].isdisjoint(groups["background"]), n)
            # Every CPU is assigned, none is invented. A 12-core AGX Orin
            # used to leave 6-9 unassigned and a 14-core Thor 6-11.
            used = set().union(
                *(groups[g] for g in ("can", "realtime", "ik", "relay", "background"))
            )
            self.assertEqual(used, set(range(n)), n)

    def test_pi5_layout(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=4):
            groups = affinity.core_groups()
        self.assertEqual(
            groups,
            {
                "can": {2, 3},
                "realtime": {1},
                "ik": {1},
                "relay": {0},
                "background": {0},
                "recorder": {0},
                "camera": set(),
                "irq": {0},
            },
        )

    def _jetson_layout(
        self, n: int, background: set[int], recorder: set[int], camera: set[int]
    ) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=n):
            groups = affinity.core_groups()
        self.assertEqual(
            groups,
            {
                "can": {n - 2, n - 1},
                "realtime": {2},
                "ik": {3},
                "relay": {4, 5},
                "background": background,
                "recorder": recorder,
                "camera": camera,
                "irq": {0},
            },
            n,
        )

    def test_orin_nx_16gb_layout(self) -> None:
        # Also the 8-core AGX Orin 32GB: no spare pair, so the recorder
        # shares the background cores with the FIFO camera set.
        self._jetson_layout(8, {0, 1}, recorder={0, 1}, camera={0, 1, 5})

    def test_agx_orin_64gb_layout(self) -> None:
        # The extra cores go to throughput work; control, IK and relay stay
        # on the same CPUs as on the Orin NX, CAN stays on the last pair.
        # The two below CAN are the recorder's, out of the FIFO camera pool.
        self._jetson_layout(
            12, {0, 1, 6, 7, 8, 9}, recorder={8, 9}, camera={0, 1, 5, 6, 7}
        )

    def test_thor_t5000_layout(self) -> None:
        self._jetson_layout(
            14,
            {0, 1, 6, 7, 8, 9, 10, 11},
            recorder={10, 11},
            camera={0, 1, 5, 6, 7, 8, 9},
        )

    def test_dedicated_recorder_cores_never_carry_fifo_camera_work(self) -> None:
        # On 8 cores FIFO camera threads preempt the recorder on every core
        # it has (~5 % of each on the 2026-09-14 policy ops). Where a spare
        # pair exists it is the recorder's alone among throughput work that
        # may preempt it; CFS GStreamer workers may still share it.
        for n in (12, 14):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
            assert groups is not None
            self.assertEqual(len(groups["recorder"]), 2, n)
            self.assertTrue(groups["recorder"] <= groups["background"], n)
            self.assertTrue(groups["recorder"].isdisjoint(groups["camera"]), n)
            for group in ("can", "realtime", "ik", "relay", "irq"):
                self.assertTrue(
                    groups["recorder"].isdisjoint(groups[group]), (n, group)
                )

    def test_five_core_layout_keeps_a_throughput_core(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=5):
            groups = affinity.core_groups()
        assert groups is not None
        self.assertEqual(groups["can"], {3, 4})
        self.assertEqual(groups["realtime"], {1})
        self.assertEqual(groups["relay"], {0, 2})
        self.assertEqual(groups["background"], {0, 2})

    def test_none_below_four_cores(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=3):
            self.assertIsNone(affinity.core_groups())


class CanIrqCpuTest(TestCase):
    def test_is_the_highest_can_core_and_never_a_camera_core(self) -> None:
        for n in (14, 12, 8, 6, 5, 4):
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
                    if cameras is None:
                        # 4 cores: no FIFO camera pool exists at all.
                        self.assertEqual(n, 4)
                        continue
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


class BackgroundAndIkTest(TestCase):
    """The recorder borrows a dedicated IK core (imports; policy-op steady state), never control's."""

    def _startup_cores(self, n: int) -> set[int]:
        applied: list[set[int]] = []
        with (
            patch.object(affinity.os, "cpu_count", return_value=n),
            patch.object(
                affinity.os, "sched_setaffinity", lambda pid, c: applied.append(set(c))
            ),
        ):
            self.assertTrue(affinity.pin_background_and_ik())
        return applied[-1]

    def test_eight_cores_widen_onto_the_idle_ik_core(self) -> None:
        # 2026-09-14: the torch/lerobot import confined to the two background
        # cores competed with the relay's SCHED_FIFO camera threads there,
        # took 56 s, and the recorder's ready handshake timed out before the
        # operator could start an episode.
        with patch.object(affinity.os, "cpu_count", return_value=8):
            groups = affinity.core_groups()
        assert groups is not None
        self.assertEqual(self._startup_cores(8), groups["background"] | groups["ik"])
        self.assertTrue(self._startup_cores(8).isdisjoint(groups["realtime"]))

    def test_dedicated_recorder_cores_replace_the_ik_loan(self) -> None:
        # 12+ cores: the recorder's own pair has no FIFO camera work, so
        # nothing starves the import or the policy-op steady state and the
        # IK core stays the IK worker's.
        for n in (12, 14):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
            assert groups is not None
            self.assertEqual(self._startup_cores(n), {n - 4, n - 3}, n)
            self.assertEqual(self._startup_cores(n), groups["recorder"], n)

    def test_pin_recorder_is_background_without_a_spare_pair(self) -> None:
        for n, expected in ((4, {0}), (6, {5}), (8, {0, 1}), (12, {8, 9})):
            applied: list[set[int]] = []
            with (
                patch.object(affinity.os, "cpu_count", return_value=n),
                patch.object(
                    affinity.os,
                    "sched_setaffinity",
                    lambda pid, c: applied.append(set(c)),
                ),
            ):
                self.assertTrue(affinity.pin_recorder())
            self.assertEqual(applied, [expected], n)

    def test_smaller_layouts_never_touch_the_control_core(self) -> None:
        # Below 8 cores ``ik`` collapses onto the control core; the import
        # must not follow it there.
        for n in (4, 5, 6):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                groups = affinity.core_groups()
            assert groups is not None
            cores = self._startup_cores(n)
            self.assertEqual(cores, groups["background"], n)
            self.assertTrue(cores.isdisjoint(groups["realtime"]), n)

    def test_noop_without_partitioning(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=2):
            self.assertFalse(affinity.pin_background_and_ik())


class _NoSchedOs:
    """``os`` as it looks on a platform with no real-time scheduling at all."""

    environ: dict[str, str] = {}


class ControlThreadFifoTest(TestCase):
    """The control thread goes SCHED_FIFO, thread-scoped, children reset to CFS."""

    def test_sets_fifo_with_reset_on_fork_on_the_calling_thread(self) -> None:
        with (
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
            patch.object(
                affinity.os, "sched_param", create=True, side_effect=lambda p: p
            ),
            patch.object(affinity.os, "SCHED_FIFO", 1, create=True),
            patch.object(affinity.os, "SCHED_RESET_ON_FORK", 0x40000000, create=True),
        ):
            self.assertTrue(affinity.prioritize_control_thread())
        setsched.assert_called_once_with(
            0, 1 | 0x40000000, affinity.CONTROL_FIFO_PRIORITY
        )

    def test_without_the_reset_flag_it_declines(self) -> None:
        # Otherwise every thread spawned from the control thread (the 1 kHz
        # IK-dispatch poll included) would inherit FIFO.
        with (
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
            patch.object(affinity.os, "SCHED_FIFO", 1, create=True),
        ):
            if hasattr(affinity.os, "SCHED_RESET_ON_FORK"):
                with patch.object(affinity.os, "SCHED_RESET_ON_FORK", None):
                    self.assertFalse(affinity.prioritize_control_thread())
            else:
                self.assertFalse(affinity.prioritize_control_thread())
        setsched.assert_not_called()

    def _deny_fifo(self, stack: ExitStack) -> None:
        """Enter patches for a host that offers SCHED_FIFO and refuses it."""
        stack.enter_context(
            patch.object(
                affinity.os,
                "sched_setscheduler",
                create=True,
                side_effect=PermissionError(1, "Operation not permitted"),
            )
        )
        stack.enter_context(
            patch.object(
                affinity.os, "sched_param", create=True, side_effect=lambda p: p
            )
        )
        stack.enter_context(patch.object(affinity.os, "SCHED_FIFO", 1, create=True))
        stack.enter_context(
            patch.object(affinity.os, "SCHED_RESET_ON_FORK", 0x40000000, create=True)
        )

    def test_permission_denied_refuses_to_run(self) -> None:
        # The whole point: a CFS control thread is a silent ~1-in-50 late tick,
        # indistinguishable from a code regression, so it stops instead.
        with ExitStack() as stack:
            self._deny_fifo(stack)
            env = dict(affinity.os.environ)
            env.pop(affinity.ALLOW_CFS_CONTROL_ENV, None)
            stack.enter_context(patch.dict(affinity.os.environ, env, clear=True))
            with self.assertRaises(affinity.ControlSchedulingError) as caught:
                affinity.prioritize_control_thread()
        message = str(caught.exception)
        self.assertIn("refusing to run without real-time scheduling", message)
        # It has to name the cause an operator cannot see from the code.
        self.assertIn("pam_limits", message)
        self.assertIn(affinity.ALLOW_CFS_CONTROL_ENV, message)

    def test_permission_denied_is_tolerated_when_not_required(self) -> None:
        with ExitStack() as stack:
            self._deny_fifo(stack)
            logs = stack.enter_context(
                self.assertLogs(affinity._logger, level="WARNING")
            )
            self.assertFalse(affinity.prioritize_control_thread(required=False))
        self.assertIn("SCHED_OTHER", logs.output[0])

    def test_env_escape_hatch_downgrades_the_refusal_to_a_warning(self) -> None:
        with ExitStack() as stack:
            self._deny_fifo(stack)
            stack.enter_context(
                patch.dict(affinity.os.environ, {affinity.ALLOW_CFS_CONTROL_ENV: "1"})
            )
            logs = stack.enter_context(
                self.assertLogs(affinity._logger, level="WARNING")
            )
            self.assertFalse(affinity.prioritize_control_thread())
        self.assertIn("SCHED_OTHER", logs.output[0])

    def test_a_platform_without_the_syscall_is_a_silent_no_op(self) -> None:
        # Nothing was on offer, so there is nothing to refuse — this mirrors
        # axol-rt, which only insists when the launcher asked for a priority.
        # Without this, every non-Linux dev box would fail to start.
        with patch.object(affinity, "os", _NoSchedOs()):
            self.assertFalse(affinity.prioritize_control_thread())

    def test_release_puts_the_thread_back_on_cfs(self) -> None:
        with (
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
            patch.object(
                affinity.os, "sched_param", create=True, side_effect=lambda p: p
            ),
            patch.object(affinity.os, "SCHED_OTHER", 0, create=True),
        ):
            affinity.release_control_thread()
        setsched.assert_called_once_with(0, 0, 0)

    def test_control_sits_between_capture_and_can_on_the_ladder(self) -> None:
        self.assertGreater(
            affinity.CONTROL_FIFO_PRIORITY, affinity.CAPTURE_FIFO_PRIORITY
        )
        self.assertLess(affinity.CONTROL_FIFO_PRIORITY, affinity.MAX_FIFO_PRIORITY)

    def test_enter_control_thread_pins_then_goes_fifo(self) -> None:
        with (
            patch.object(affinity, "pin_realtime", return_value=False) as pin,
            patch.object(
                affinity, "prioritize_control_thread", return_value=True
            ) as fifo,
        ):
            self.assertTrue(affinity.enter_control_thread())
        pin.assert_called_once_with()
        fifo.assert_called_once_with(required=True)

    def test_enter_control_thread_forwards_a_tolerated_denial(self) -> None:
        with (
            patch.object(affinity, "pin_realtime", return_value=True),
            patch.object(
                affinity, "prioritize_control_thread", return_value=False
            ) as fifo,
        ):
            self.assertTrue(affinity.enter_control_thread(required=False))
        fifo.assert_called_once_with(required=False)


class DescribeLayoutTest(TestCase):
    def test_names_every_role_per_host(self) -> None:
        for n, expected in (
            (
                14,
                "14 cores online: CAN 12-13, control 2, IK 3, relay 4-5, "
                "recorder 10-11, camera 0-1,5-9",
            ),
            (
                4,
                "4 cores online: CAN 2-3, control 1, IK 1, relay 0, "
                "recorder 0, camera -",
            ),
            (2, "2 cores online: too few to partition, nothing is pinned"),
        ):
            with patch.object(affinity.os, "cpu_count", return_value=n):
                self.assertEqual(affinity.describe_layout(), expected, n)


class OnlineCpusTest(TestCase):
    """The layout lands on CPUs that exist even if a mode offlines a middle one."""

    def _groups(self, n: int, online: str) -> dict[str, set[int]] | None:
        with (
            patch.object(affinity.os, "cpu_count", return_value=n),
            patch.object(affinity.Path, "read_text", return_value=online + "\n"),
        ):
            return affinity.core_groups()

    def test_contiguous_online_list_is_the_plain_numbering(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=12):
            plain = affinity.core_groups()
        self.assertEqual(self._groups(12, "0-11"), plain)

    def test_a_gap_maps_every_role_onto_online_cpus(self) -> None:
        # 12 online out of 14, with 6-7 offline: CAN still gets the top two
        # online CPUs and nothing names an offline one.
        groups = self._groups(12, "0-5,8-13")
        assert groups is not None
        self.assertEqual(groups["can"], {12, 13})
        self.assertEqual(groups["realtime"], {2})
        self.assertEqual(groups["recorder"], {10, 11})
        used = set().union(*groups.values())
        self.assertTrue(used <= set(range(6)) | set(range(8, 14)))
        self.assertEqual(len(used), 12)

    def test_a_disagreeing_or_unreadable_list_falls_back(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=8):
            plain = affinity.core_groups()
        self.assertEqual(self._groups(8, "0-11"), plain)
        self.assertEqual(self._groups(8, "garbage"), plain)
