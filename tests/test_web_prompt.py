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

    async def test_stop_after_a_wrist_was_zeroed_asks_for_a_power_cycle(self) -> None:
        """The zero already saved stays, and a Damiao wrist needs a power cycle."""
        from contextlib import asynccontextmanager

        from almond_axol.cli.motor import set_zero_pos
        from almond_axol.constants import Joint
        from almond_axol.motor.damiao import DamiaoMotor

        zeroed: list[Joint] = []

        class FakeWrist:
            def __init__(self, joint: Joint) -> None:
                self.joint = joint
                self._driver = object.__new__(DamiaoMotor)
                self._reads = 0

            async def set_direction_flag(self) -> bool:
                return False

            async def get_position(self) -> float:
                # In range for the start reference, then at an end stop.
                self._reads += 1
                return 0.0 if self._reads == 1 else 1.5

            async def set_zero_position(self) -> None:
                zeroed.append(self.joint)

        @asynccontextmanager
        async def fake_bus(_channel):
            yield object()

        prompts: list[str] = []

        def prompt(instruction: str, web_prompts: bool, button=None) -> bool:
            prompts.append(instruction)
            if len(prompts) <= 2:
                return True  # the start reference, then wrist_2 at its stop
            raise KeyboardInterrupt  # Stop at wrist_3's prompt

        args = mock.Mock(l=True, r=False, channel="can-test", joints="wrist_2,wrist_3")
        args.web_prompts = True
        power_cycle = mock.AsyncMock()
        with (
            mock.patch.object(set_zero_pos, "CanBus", fake_bus),
            mock.patch.object(
                set_zero_pos, "Motor", lambda bus, joint: FakeWrist(joint)
            ),
            mock.patch.object(set_zero_pos, "resolve_channel", return_value="can-test"),
            mock.patch.object(set_zero_pos, "_prompt", side_effect=prompt),
            mock.patch.object(set_zero_pos, "_apply_damiao_zeros", power_cycle),
            mock.patch("builtins.print") as printed,
        ):
            with self.assertRaises(KeyboardInterrupt):
                await set_zero_pos._run_guided(args)
        self.assertEqual(zeroed, [Joint.WRIST_2])
        output = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("WRIST_2      zeroed", output)
        self.assertIn("Damiao motors zeroed (WRIST_2)", output)
        self.assertIn("power-cycle required", output)
        # The run ended: it did not go on to the power-cycle prompt.
        power_cycle.assert_not_called()


class StopIsReportedTest(unittest.TestCase):
    """A Stop at a web prompt ends the command with exit status 130."""

    def test_guided_zeroing_reports_the_stop(self) -> None:
        from almond_axol.cli.motor import set_zero_pos

        async def stopped_at_a_prompt(_args) -> None:
            raise KeyboardInterrupt

        with (
            mock.patch.object(set_zero_pos, "_run", stopped_at_a_prompt),
            mock.patch("builtins.print") as printed,
        ):
            with self.assertRaises(SystemExit) as exit_:
                set_zero_pos.run(mock.Mock())
        self.assertEqual(exit_.exception.code, 130)
        printed.assert_called_with("\nstopped.")

    def test_burn_in_reports_the_stop(self) -> None:
        from almond_axol.cli.motor import burn_in

        async def stopped_at_a_prompt(_args, _buses) -> bool:
            # The panel's Stop lands while a step waits on Continue.
            return burn_in._web_prompt("Connect motor 1.")

        with (
            mock.patch.object(burn_in, "_validate"),
            mock.patch.object(burn_in, "_run", stopped_at_a_prompt),
            mock.patch.object(web_prompt, "ask", side_effect=KeyboardInterrupt),
            mock.patch("builtins.print") as printed,
        ):
            with self.assertRaises(SystemExit) as exit_:
                burn_in.run(mock.Mock())
        self.assertEqual(exit_.exception.code, 130)
        printed.assert_called_with("\n  Stopped. Every motor was switched off.")


if __name__ == "__main__":
    unittest.main()
