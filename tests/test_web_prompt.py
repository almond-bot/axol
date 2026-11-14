"""The panel's Stop (one SIGINT) ends a run waiting on a Continue prompt."""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from almond_axol.cli import web_prompt

# A command under asyncio.run, blocked on a web prompt inside a coroutine:
# the shape of guided zeroing and motor.burn-in.
_SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    from almond_axol.cli import web_prompt

    async def main():
        try:
            answer = web_prompt.ask("[prompt] hold the arm")
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            raise
        print(f"answered {answer!r}", flush=True)
        await asyncio.sleep(0)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
    """
)


@unittest.skipIf(sys.platform == "win32", "POSIX signals")
class StopDuringPromptTest(unittest.TestCase):
    def _start(self) -> subprocess.Popen[str]:
        proc = subprocess.Popen(
            [sys.executable, "-c", _SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        self.assertEqual(proc.stdout.readline(), "[prompt] hold the arm\n")
        return proc

    def test_first_sigint_interrupts_the_prompt(self) -> None:
        proc = self._start()
        try:
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertEqual(proc.returncode, 130, err)
        self.assertIn("interrupted", out)

    def test_continue_answers_the_prompt(self) -> None:
        proc = self._start()
        try:
            out, err = proc.communicate("\n", timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertEqual(proc.returncode, 0, err)
        self.assertIn("answered '\\n'", out)


class HandlerRestoredTest(unittest.TestCase):
    def test_previous_sigint_handler_is_restored(self) -> None:
        def marker(signum, frame):  # pragma: no cover - never delivered
            pass

        previous = signal.signal(signal.SIGINT, marker)
        try:
            with (
                mock.patch("sys.stdin.readline", return_value="\n"),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(web_prompt.ask("[prompt] x"), "\n")
            self.assertIs(signal.getsignal(signal.SIGINT), marker)
        finally:
            signal.signal(signal.SIGINT, previous)


class GuidedZeroStopTest(unittest.IsolatedAsyncioTestCase):
    """A Stop at a joint's prompt ends guided zeroing instead of skipping on."""

    async def test_stop_at_a_joint_prompt_ends_the_run(self) -> None:
        from contextlib import asynccontextmanager

        from almond_axol.cli.motor import set_zero_pos

        class FakeMotor:
            _driver = object()

            async def set_direction_flag(self) -> bool:
                return False

            async def get_position(self) -> float:
                return 0.0

        @asynccontextmanager
        async def fake_bus(_channel):
            yield object()

        prompts: list[str] = []

        def prompt(instruction: str, web_prompts: bool, button=None) -> bool:
            prompts.append(instruction)
            if len(prompts) == 1:
                return True  # hold every joint in range
            raise KeyboardInterrupt  # Stop at the first joint's end stop

        args = mock.Mock(
            l=True, r=False, channel="can-test", joints=None, web_prompts=True
        )
        with (
            mock.patch.object(set_zero_pos, "CanBus", fake_bus),
            mock.patch.object(set_zero_pos, "Motor", lambda bus, joint: FakeMotor()),
            mock.patch.object(set_zero_pos, "resolve_channel", return_value="can-test"),
            mock.patch.object(set_zero_pos, "_prompt", side_effect=prompt),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                await set_zero_pos._run_guided(args)
        self.assertEqual(len(prompts), 2)


if __name__ == "__main__":
    unittest.main()
