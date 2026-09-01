from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from almond_axol.utils.control_loop import (
    run_blocking_with_control_ticks,
    run_blocking_with_sync_control_ticks,
)


class RunBlockingWithControlTicksTest(unittest.IsolatedAsyncioTestCase):
    async def test_ticks_continue_until_blocking_operation_returns(self) -> None:
        release = threading.Event()
        ticks = 0

        def operation() -> int:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")
            return 42

        async def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 3:
                release.set()

        result = await run_blocking_with_control_ticks(operation, tick, 0.001)

        self.assertEqual(result, 42)
        self.assertGreaterEqual(ticks, 3)

    async def test_operation_exception_propagates_after_ticks(self) -> None:
        release = threading.Event()
        ticks = 0

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")
            raise RuntimeError("recorder failed")

        async def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 2:
                release.set()

        with self.assertRaisesRegex(RuntimeError, "recorder failed"):
            await run_blocking_with_control_ticks(operation, tick, 0.001)

        self.assertGreaterEqual(ticks, 2)

    async def test_cancellation_drains_worker_while_ticks_continue(self) -> None:
        release = threading.Event()
        operation_done = threading.Event()
        first_tick = asyncio.Event()
        finish_first_tick = asyncio.Event()
        ticks = 0

        def operation() -> None:
            try:
                if not release.wait(1.0):
                    raise TimeoutError("test did not release operation")
            finally:
                operation_done.set()

        async def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 1:
                first_tick.set()
                await finish_first_tick.wait()
            elif ticks == 3:
                release.set()

        task = asyncio.create_task(
            run_blocking_with_control_ticks(operation, tick, 0.001)
        )
        await asyncio.wait_for(first_tick.wait(), 1.0)
        task.cancel()
        finish_first_tick.set()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1.0)

        self.assertTrue(operation_done.is_set())
        self.assertGreaterEqual(ticks, 3)

    async def test_slow_tick_does_not_create_catch_up_burst(self) -> None:
        release = threading.Event()
        starts: list[float] = []
        loop = asyncio.get_running_loop()
        period_s = 0.02

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")

        async def tick() -> None:
            starts.append(loop.time())
            if len(starts) == 1:
                await asyncio.sleep(3 * period_s)
            elif len(starts) == 4:
                release.set()

        await run_blocking_with_control_ticks(operation, tick, period_s)

        self.assertGreaterEqual(len(starts), 4)
        self.assertGreaterEqual(starts[1] - starts[0], 3 * period_s * 0.8)
        # An absolute-deadline loop would run ticks 2-4 back-to-back to repay
        # the first tick's delay. Relative pacing leaves each later gap intact.
        self.assertGreaterEqual(starts[2] - starts[1], period_s * 0.8)
        self.assertGreaterEqual(starts[3] - starts[2], period_s * 0.8)

    async def test_driver_cancelled_error_does_not_spin_forever(self) -> None:
        ticks = 0

        def operation() -> None:
            raise asyncio.CancelledError

        async def tick() -> None:
            nonlocal ticks
            ticks += 1

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(
                run_blocking_with_control_ticks(operation, tick, 0.001),
                timeout=1.0,
            )

        self.assertGreaterEqual(ticks, 1)

    async def test_tick_cancelled_error_does_not_spin_forever(self) -> None:
        release = threading.Event()

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")

        async def tick() -> None:
            release.set()
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(
                run_blocking_with_control_ticks(operation, tick, 0.001),
                timeout=1.0,
            )

    async def test_tick_error_uses_drain_tick_until_worker_finishes(self) -> None:
        release = threading.Event()
        drain_ticks = 0

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")

        async def tick() -> None:
            raise RuntimeError("tracking contact")

        async def drain_tick() -> None:
            nonlocal drain_ticks
            drain_ticks += 1
            if drain_ticks == 3:
                release.set()

        with self.assertRaisesRegex(RuntimeError, "tracking contact"):
            await run_blocking_with_control_ticks(
                operation,
                tick,
                0.001,
                drain_tick=drain_tick,
            )

        self.assertGreaterEqual(drain_ticks, 3)

    async def test_drain_tick_error_still_drains_worker(self) -> None:
        release = threading.Event()
        operation_done = threading.Event()

        def operation() -> None:
            try:
                if not release.wait(1.0):
                    raise TimeoutError("test did not release operation")
            finally:
                operation_done.set()

        async def tick() -> None:
            raise RuntimeError("tracking contact")

        async def drain_tick() -> None:
            release.set()
            raise OSError("hold failed")

        with self.assertRaisesRegex(RuntimeError, "tracking contact") as raised:
            await run_blocking_with_control_ticks(
                operation,
                tick,
                0.001,
                drain_tick=drain_tick,
            )

        self.assertTrue(operation_done.is_set())
        self.assertTrue(
            any("hold failed" in note for note in raised.exception.__notes__)
        )


class RunBlockingWithSyncControlTicksTest(unittest.TestCase):
    def test_interrupt_from_loop_condition_is_drained(self) -> None:
        release = threading.Event()
        operation_done = threading.Event()
        ticks = 0
        real_event = threading.Event

        class InterruptOnceEvent:
            def __init__(self) -> None:
                self._event = real_event()
                self._interrupted = False

            def is_set(self) -> bool:
                if not self._interrupted:
                    self._interrupted = True
                    raise KeyboardInterrupt
                return self._event.is_set()

            def set(self) -> None:
                self._event.set()

            def wait(self, timeout: float | None = None) -> bool:
                return self._event.wait(timeout)

        events = 0

        def make_event():
            nonlocal events
            events += 1
            return InterruptOnceEvent() if events == 1 else real_event()

        def operation() -> None:
            try:
                if not release.wait(1.0):
                    raise TimeoutError("test did not release operation")
            finally:
                operation_done.set()

        def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 3:
                release.set()

        with (
            patch("almond_axol.utils.control_loop.threading.Event", make_event),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_blocking_with_sync_control_ticks(operation, tick, 0.001)

        self.assertTrue(operation_done.is_set())
        self.assertGreaterEqual(ticks, 3)

    def test_interrupt_delivered_after_worker_start_is_drained(self) -> None:
        release = threading.Event()
        operation_done = threading.Event()
        ticks = 0

        def operation() -> None:
            try:
                if not release.wait(1.0):
                    raise TimeoutError("test did not release operation")
            finally:
                operation_done.set()

        def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 3:
                release.set()

        # Model SIGINT becoming pending while Thread.start() is protected and
        # being delivered as soon as the main thread restores its old mask.
        with (
            patch(
                "almond_axol.utils.control_loop.signal.pthread_sigmask",
                side_effect=(set(), KeyboardInterrupt()),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_blocking_with_sync_control_ticks(operation, tick, 0.001)

        self.assertTrue(operation_done.is_set())
        self.assertGreaterEqual(ticks, 3)

    def test_ticks_continue_until_blocking_operation_returns(self) -> None:
        release = threading.Event()
        ticks = 0

        def operation() -> int:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")
            return 42

        def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 3:
                release.set()

        result = run_blocking_with_sync_control_ticks(operation, tick, 0.001)

        self.assertEqual(result, 42)
        self.assertGreaterEqual(ticks, 3)

    def test_operation_exception_propagates_after_ticks(self) -> None:
        release = threading.Event()
        ticks = 0

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")
            raise RuntimeError("recorder failed")

        def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 2:
                release.set()

        with self.assertRaisesRegex(RuntimeError, "recorder failed"):
            run_blocking_with_sync_control_ticks(operation, tick, 0.001)

        self.assertGreaterEqual(ticks, 2)

    def test_tick_error_uses_drain_tick_until_worker_finishes(self) -> None:
        release = threading.Event()
        drain_ticks = 0

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")

        def tick() -> None:
            raise RuntimeError("command failed")

        def drain_tick() -> None:
            nonlocal drain_ticks
            drain_ticks += 1
            if drain_ticks == 3:
                release.set()

        with self.assertRaisesRegex(RuntimeError, "command failed"):
            run_blocking_with_sync_control_ticks(
                operation,
                tick,
                0.001,
                drain_tick=drain_tick,
            )

        self.assertGreaterEqual(drain_ticks, 3)

    def test_keyboard_interrupt_is_delayed_while_ticks_continue(self) -> None:
        release = threading.Event()
        operation_done = threading.Event()
        ticks = 0

        def operation() -> None:
            try:
                if not release.wait(1.0):
                    raise TimeoutError("test did not release operation")
            finally:
                operation_done.set()

        def tick() -> None:
            nonlocal ticks
            ticks += 1
            if ticks == 1:
                raise KeyboardInterrupt
            if ticks == 3:
                release.set()

        with self.assertRaises(KeyboardInterrupt):
            run_blocking_with_sync_control_ticks(operation, tick, 0.001)

        self.assertTrue(operation_done.is_set())
        self.assertGreaterEqual(ticks, 3)

    def test_slow_tick_does_not_create_catch_up_burst(self) -> None:
        release = threading.Event()
        starts: list[float] = []
        period_s = 0.02

        def operation() -> None:
            if not release.wait(1.0):
                raise TimeoutError("test did not release operation")

        def tick() -> None:
            starts.append(time.perf_counter())
            if len(starts) == 1:
                time.sleep(3 * period_s)
            elif len(starts) == 4:
                release.set()

        run_blocking_with_sync_control_ticks(operation, tick, period_s)

        self.assertGreaterEqual(len(starts), 4)
        self.assertGreaterEqual(starts[1] - starts[0], 3 * period_s * 0.8)
        self.assertGreaterEqual(starts[2] - starts[1], period_s * 0.8)
        self.assertGreaterEqual(starts[3] - starts[2], period_s * 0.8)


if __name__ == "__main__":
    unittest.main()
