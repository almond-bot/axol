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
            patch.object(
                affinity.os, "listdir", return_value=[*self.comms, "999", "x"]
            ),
            patch.object(affinity.os, "sched_setscheduler", create=True) as setsched,
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
        param = ("param", affinity.CAPTURE_FIFO_PRIORITY)
        self.assertCountEqual(
            setsched.call_args_list,
            [call(tid, 1, param) for tid in (201, 202, 203, 204, 207)],
        )

    def test_permission_denied_leaves_threads_cfs(self) -> None:
        with (
            patch.object(affinity.os, "listdir", return_value=list(self.comms)),
            patch.object(
                affinity.os,
                "sched_setscheduler",
                create=True,
                side_effect=PermissionError("EPERM"),
            ) as setsched,
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
        self.assertTrue(any("CAP_SYS_NICE" in line for line in logs.output))

    def test_noop_without_scheduler_api(self) -> None:
        with patch.object(affinity, "os", SimpleNamespace(listdir=lambda _p: [])):
            self.assertEqual(affinity.prioritize_capture_threads(self.wanted), 0)


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
