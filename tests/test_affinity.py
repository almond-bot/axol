from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import call, patch

from almond_axol.utils import affinity


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
