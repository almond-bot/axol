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

    def test_encode_chain_gets_the_lower_fifo_tier(self) -> None:
        # The NVENC feed/dequeue/drain threads go SCHED_FIFO one notch under
        # the capture chain (2026-09-15: left CFS they starved behind the
        # recorder's spill during every DAgger intervention, the encoder's
        # input pool ran dry and dsenc_inq overran). A name in both lists is
        # capture-tier. Other CFS threads (cuda) still stay put.
        comms = {**self.comms, "209": "dsenc_l_inq:src", "210": "dsenc_l:src"}
        encode = ("V4L2_EncThread", "dsenc_l_inq:src", "dsenc_l:src", "dsenc_l_outq:sr")
        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(affinity.os, "listdir", return_value=list(comms)),
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
            patch.object(
                affinity.os,
                "sched_param",
                create=True,
                side_effect=lambda p: ("param", p),
            ),
            patch.object(affinity.os, "SCHED_FIFO", 1, create=True),
            patch("builtins.open", _proc_files(comms)),
            patch("threading.enumerate", return_value=self.python_threads),
            patch("almond_axol.utils.jetson.can_irq_cpus", return_value={7}),
        ):
            moved = affinity.prioritize_capture_threads(
                (*self.wanted, "dsenc_l:src"), encode
            )

        self.assertEqual(moved, 9)
        capture = ("param", affinity.CAPTURE_FIFO_PRIORITY)
        encode_param = ("param", affinity.ENCODE_FIFO_PRIORITY)
        self.assertLess(affinity.ENCODE_FIFO_PRIORITY, affinity.CAPTURE_FIFO_PRIORITY)
        self.assertCountEqual(
            setsched.call_args_list,
            [call(tid, 1, capture) for tid in (201, 202, 203, 204, 207, 210)]
            + [call(tid, 1, encode_param) for tid in (205, 208, 209)],
        )
        # Every FIFO thread lands on the relay's share of the camera pool:
        # the interrupt is steered, so the pool is {0, 1, 5}, and core 5 is
        # the Argus daemon's (capture_daemon_cores).
        self.assertCountEqual(
            set_affinity.call_args_list,
            [
                call(tid, {0, 1})
                for tid in (201, 202, 203, 204, 205, 207, 208, 209, 210)
            ],
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
        # 6-7 cores: CPU0 is a CAN core, so steering never changes the pool.
        # 5 cores: CPU0 is the relay's Python core and the interrupt CPU, so
        # the one throughput core left is the whole pool. 4 cores (Pi 5): no
        # CPU is free of control, CAN, relay-Python or the interrupt — no
        # FIFO camera pool at all (the Pi runs no ZED cameras).
        for n, expected in ((6, {4, 5}), (5, {2}), (4, None)):
            for irq_cpus in (None, {0}, {n - 1}):
                self.assertEqual(self._cores(n, irq_cpus), expected, (n, irq_cpus))

    def test_none_when_partitioning_is_not_applicable(self) -> None:
        self.assertIsNone(self._cores(2, None))


class CameraPoolSplitTest(TestCase):
    """The Argus daemon and the relay's FIFO chain share no CPU once the pool has three."""

    def _split(
        self, n: int, irq_cpus: set[int] | None
    ) -> tuple[set[int] | None, set[int] | None]:
        with (
            patch.object(affinity.os, "cpu_count", return_value=n),
            patch("almond_axol.utils.jetson.can_irq_cpus", return_value=irq_cpus),
        ):
            return affinity.capture_daemon_cores(), affinity.relay_capture_cores()

    def test_steered_eight_core_pool_gives_the_daemon_its_own_core(self) -> None:
        # 2026-09-15: ~160 % of a core of FIFO work (daemon ~72 % + relay
        # chain) shared three CPUs; per-CPU FIFO load peaked at 96-101 %, the
        # RT throttle fired, and the throttled daemon lost its capture
        # scheduler — every camera on the box went with it.
        daemon, relay = self._split(8, {affinity.can_irq_cpu()})
        self.assertEqual(daemon, {5})
        self.assertEqual(relay, {0, 1})
        assert daemon is not None and relay is not None
        self.assertTrue(daemon.isdisjoint(relay))
        self.assertEqual(daemon | relay, {0, 1, 5})

    def test_two_core_pool_stays_shared(self) -> None:
        # Unsteered: the pool is {1, 5}; splitting it would serialize one
        # side onto a single CPU, so both keep the pair (pre-existing layout).
        for irq_cpus in (None, {0}):
            self.assertEqual(self._split(8, irq_cpus), ({1, 5}, {1, 5}), irq_cpus)
        self.assertEqual(self._split(6, {5}), ({4, 5}, {4, 5}))
        self.assertEqual(self._split(5, None), ({2}, {2}))

    def test_none_without_a_pool(self) -> None:
        self.assertEqual(self._split(4, None), (None, None))
        self.assertEqual(self._split(2, None), (None, None))


class ApplyPinsEveryThreadTest(TestCase):
    """The dedicated-process pins move the process's existing threads too."""

    def _run(self, pin, tasks=("101", "202", "303", "gone", "x")):
        calls: list[tuple[int, set[int]]] = []

        def _set(tid, cores):
            if tid == 303:
                raise OSError("exited")
            calls.append((tid, set(cores)))

        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(affinity.os, "listdir", return_value=list(tasks)),
            patch.object(affinity.os, "sched_setaffinity", _set),
            patch.object(affinity.threading, "get_native_id", return_value=101),
        ):
            self.assertTrue(pin())
        return calls

    def test_pin_ik_leaves_the_xla_pool_on_the_startup_mask(self) -> None:
        # pin_ik_startup widened the worker to {2, 3} and JAX spawned its
        # tf_XLAEigen pool there. Narrowing the pool onto the IK core with the
        # solve loop (axol #301's first cut) serialized the solve onto one
        # CPU: engaged IK fell from ~100-115 Hz to 62-78 Hz and teleop
        # juddered (station, 2026-09-15 15:45). Only the loop thread narrows.
        calls = self._run(affinity.pin_ik)
        self.assertEqual(calls, [(0, {3})])

    def test_pin_ik_startup_widens_every_thread(self) -> None:
        calls = self._run(affinity.pin_ik_startup)
        self.assertEqual(calls, [(0, {2, 3}), (202, {2, 3})])

    def test_recorder_narrowing_moves_its_import_threads(self) -> None:
        with patch.object(affinity.os, "cpu_count", return_value=8):
            groups = affinity.core_groups()
        assert groups is not None
        for pin, cores in (
            (affinity.pin_background, groups["background"]),
            (affinity.pin_background_and_ik, groups["background"] | groups["ik"]),
            (affinity.pin_relay, groups["relay"]),
        ):
            calls = self._run(pin)
            self.assertEqual(calls, [(0, cores), (202, cores)], pin.__name__)

    def test_pin_realtime_is_thread_scoped(self) -> None:
        # Under `axol serve` the control loop is one thread of the shared
        # server process; its web/VR threads must not follow it onto core 2.
        calls = self._run(affinity.pin_realtime)
        self.assertEqual(calls, [(0, {2})])

    def test_without_proc_the_caller_is_still_pinned(self) -> None:
        with (
            patch.object(affinity.os, "cpu_count", return_value=8),
            patch.object(affinity.os, "listdir", side_effect=OSError("no /proc")),
            patch.object(affinity.os, "sched_setaffinity") as set_affinity,
        ):
            self.assertTrue(affinity.pin_background())
        self.assertEqual(set_affinity.call_args_list, [call(0, {0, 1})])


class CoreGroupsTest(TestCase):
    """Every partitioned host gives the Rust core two CAN CPUs of its own."""

    def test_can_cores_are_a_disjoint_pair_on_every_layout(self) -> None:
        # rt.link only exports AXOL_RT_CPU_LEFT/RIGHT + the SCHED_FIFO request
        # when there are two CAN cores disjoint from Python control. The old
        # 4-5 core layout shared them (can == realtime), which left a Pi 5's
        # bus loops as unpinned CFS threads: a 43.8 ms overrun sent the core
        # limp mid-ROM-sweep, and every >0.5 ms late tick had already gated
        # the shoulder host damping off.
        for n in (4, 5, 6, 8, 12):
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
            # No CPU is invented, and the small layouts use every one.
            used = set().union(
                *(groups[g] for g in ("can", "realtime", "ik", "relay", "background"))
            )
            self.assertTrue(used <= set(range(n)), n)
            if n <= 8:
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
                "irq": {0},
            },
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
