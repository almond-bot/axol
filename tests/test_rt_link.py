import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from almond_axol.rt import link
from almond_axol.utils import affinity


class _FakeProc:
    returncode = None

    def poll(self):
        return None


async def _start_and_capture_env(cpu_count: int) -> dict[str, str]:
    """Run ``RtLink.start`` against a fake core and return the env it spawned with."""
    reader = MagicMock()
    # The reader task exits cleanly on a closed stream.
    reader.readexactly = AsyncMock(side_effect=asyncio.IncompleteReadError(b"", 4))
    writer = MagicMock()
    popen = MagicMock(return_value=_FakeProc())
    with (
        patch.object(affinity.os, "cpu_count", return_value=cpu_count),
        patch.object(link, "find_binary", return_value="/fake/axol-rt"),
        patch.object(link.subprocess, "Popen", popen),
        patch.object(
            link.asyncio,
            "open_unix_connection",
            AsyncMock(return_value=(reader, writer)),
        ),
    ):
        rt = link.RtLink()
        await rt.start()
        assert rt._reader_task is not None
        await rt._reader_task
    return popen.call_args.kwargs["env"]


class RtLinkSchedulingEnvTest(unittest.TestCase):
    """The core is pinned and given SCHED_FIFO on every partitioned host."""

    def _env(self, cpu_count: int) -> dict[str, str]:
        return asyncio.run(_start_and_capture_env(cpu_count))

    def test_pi5_gets_pinned_can_cores_and_fifo(self) -> None:
        # Before the Pi layout, a 4-core host shared CAN with Python control
        # and none of these were set: the bus loops ran as CFS threads and
        # the ROM soak went limp on a 43.8 ms overrun.
        env = self._env(4)
        self.assertEqual(env["AXOL_RT_CPU_LEFT"], "2")
        self.assertEqual(env["AXOL_RT_CPU_RIGHT"], "3")
        self.assertEqual(env["AXOL_RT_FIFO_PRIORITY"], "20")
        self.assertEqual(env["AXOL_RT_BACKGROUND_CPUS"], "0")

    def test_jetson_layout_unchanged(self) -> None:
        env = self._env(8)
        self.assertEqual(env["AXOL_RT_CPU_LEFT"], "6")
        self.assertEqual(env["AXOL_RT_CPU_RIGHT"], "7")
        self.assertEqual(env["AXOL_RT_FIFO_PRIORITY"], "20")
        self.assertEqual(env["AXOL_RT_BACKGROUND_CPUS"], "0,1")

    def test_unpartitioned_host_requests_nothing(self) -> None:
        env = self._env(2)
        for key in (
            "AXOL_RT_CPU_LEFT",
            "AXOL_RT_CPU_RIGHT",
            "AXOL_RT_FIFO_PRIORITY",
            "AXOL_RT_BACKGROUND_CPUS",
        ):
            self.assertNotIn(key, env)


if __name__ == "__main__":
    unittest.main()
