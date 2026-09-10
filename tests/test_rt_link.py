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


class _ExitedProc:
    """A core that has already exited (the proto-refusal path)."""

    returncode = 1

    def poll(self):
        return 1


class RtLinkConfigureTest(unittest.IsolatedAsyncioTestCase):
    """Package/binary protocol skew fails at configure time, and says so.

    Before the ``proto`` line existed, a package that slotted joints by motor
    id armed against a core that slotted them by list order, and the core
    then rejected every target on its max-step gate: the arms enabled and
    held, and nothing moved. Now the core refuses the config and exits; the
    link has to turn that exit into a message naming the binary and the fix.
    """

    def _link(self, proc) -> link.RtLink:
        rt = link.RtLink(binary="/opt/axol-rt")
        rt._proc = proc
        rt._writer = MagicMock()
        rt._writer.is_closing.return_value = False
        return rt

    def test_config_header_declares_the_protocol(self) -> None:
        self.assertEqual(link.config_header(), [f"proto {link.CONFIG_PROTO}"])
        self.assertEqual(link.CONFIG_PROTO, 2)

    async def test_configure_names_a_stale_binary_when_the_core_exits(self) -> None:
        rt = self._link(_ExitedProc())
        started = asyncio.get_running_loop().time()
        with self.assertRaises(link.RtLinkError) as ctx:
            await rt.configure("proto 2\nloop_hz 240\n")
        message = str(ctx.exception)
        self.assertIn("/opt/axol-rt", message)
        self.assertIn("proto 2", message)
        self.assertIn("axol rt.install", message)
        # The exit is noticed in well under the 5 s ack timeout.
        self.assertLess(asyncio.get_running_loop().time() - started, 2.0)
        sent = rt._writer.write.call_args.args[0]
        self.assertTrue(sent.endswith(b"Cproto 2\nloop_hz 240\n"))

    async def test_configure_keeps_a_generic_error_while_the_core_runs(self) -> None:
        # A core that is alive but silent is a different failure (not skew):
        # the plain timeout message stands. Short timeout via a patched wait.
        rt = self._link(_FakeProc())
        with patch.object(
            rt, "_await_state", side_effect=link.RtLinkError("timed out")
        ):
            with self.assertRaises(link.RtLinkError) as ctx:
                await rt.configure("proto 2\n")
        self.assertEqual(str(ctx.exception), "timed out")

    async def test_await_state_still_takes_an_ack_sent_just_before_exit(self) -> None:
        # A disarm ack can land after the core has already exited; the exit
        # check must not pre-empt a state that is (or is about to be) queued.
        rt = self._link(_ExitedProc())

        async def late_ack() -> None:
            await asyncio.sleep(0.05)
            rt._states.put_nowait("disarmed")

        asyncio.ensure_future(late_ack())
        await rt._await_state("disarmed", 5.0, tolerate_fault=True)


if __name__ == "__main__":
    unittest.main()
