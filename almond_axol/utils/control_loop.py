"""Helpers for keeping robot command loops alive across blocking work."""

from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_blocking_with_control_ticks(
    operation: Callable[[], T],
    tick: Callable[[], Awaitable[None]],
    period_s: float,
    *,
    drain_tick: Callable[[], Awaitable[None]] | None = None,
) -> T:
    """Run blocking ``operation`` off-loop while awaiting ``tick`` regularly.

    This is for bounded lifecycle calls which cannot safely interrupt a robot's
    command stream (for example, recorder IPC at an episode boundary). Merely
    awaiting :func:`asyncio.to_thread` is not enough when the awaiting coroutine
    is itself the only command producer, so this helper continues calling
    ``tick`` on the caller's event loop until the worker finishes. If ``tick``
    fails and ``drain_tick`` is provided, the first error is retained while the
    safer fallback tick keeps the command stream alive until the worker drains.

    Tick pacing is relative to the start of each tick. A slow tick therefore
    delays the next one instead of accruing an absolute-deadline debt and
    emitting a burst of catch-up commands. Caller cancellation is delayed until
    the blocking worker has completed: Python cannot cancel work already running
    in a thread, and abandoning such a worker could let it keep using an IPC pipe
    while teardown or a later command races it. The operation must consequently
    be bounded internally (for example, its IPC call must have a finite timeout);
    the helper deliberately does not pretend it can kill a stuck Python thread.

    Args:
        operation: Bounded synchronous callable to execute in a worker thread.
        tick: Async control/hold step, run serially on the caller's event loop.
        period_s: Minimum target period between the starts of normal-duration
            ticks. Must be positive.
        drain_tick: Optional safe hold/fallback step used after ``tick`` raises.
            If it also raises, the worker is still drained but no further ticks
            are attempted; the original tick error remains primary.

    Returns:
        The value returned by ``operation``.

    Raises:
        ValueError: If ``period_s`` is not positive.
        BaseException: Any exception from ``operation`` or ``tick``. A tick
            error takes precedence over a concurrent operation error. If the
            caller cancels this coroutine, cancellation wins after the worker
            has been drained.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive")

    loop = asyncio.get_running_loop()

    async def drive() -> T:
        worker = asyncio.create_task(asyncio.to_thread(operation))
        active_tick = tick
        tick_error: BaseException | None = None
        drain_error: BaseException | None = None
        try:
            while not worker.done():
                tick_started = loop.time()
                try:
                    await active_tick()
                except BaseException as exc:
                    if tick_error is None:
                        tick_error = exc
                        if drain_tick is None:
                            break
                        active_tick = drain_tick
                    else:
                        drain_error = exc
                        break
                remaining = period_s - (loop.time() - tick_started)
                if remaining > 0 and not worker.done():
                    # Unlike wait_for(), asyncio.wait() does not cancel the
                    # non-cancellable thread-backed task when the period ends.
                    await asyncio.wait((worker,), timeout=remaining)
            if not worker.done():
                # No fallback exists, or the fallback itself failed. Preserve
                # IPC sequencing even though there is no longer a useful tick.
                with contextlib.suppress(BaseException):
                    await asyncio.shield(worker)

            operation_error: BaseException | None = None
            result: T | None = None
            try:
                result = worker.result()
            except BaseException as exc:
                operation_error = exc

            if tick_error is not None:
                if drain_error is not None:
                    tick_error.add_note(f"drain tick also failed: {drain_error!r}")
                if operation_error is not None:
                    tick_error.add_note(
                        f"blocking operation also failed: {operation_error!r}"
                    )
                raise tick_error
            if operation_error is not None:
                raise operation_error
            return result  # type: ignore[return-value]
        except BaseException:
            # A failed tick must not orphan a thread which may still hold an IPC
            # lock. There is no useful heartbeat after the heartbeat itself has
            # failed, but draining the bounded operation preserves sequencing.
            if not worker.done():
                with contextlib.suppress(BaseException):
                    await asyncio.shield(worker)
            else:
                # Retrieve a worker exception so asyncio does not report an
                # unobserved task while the tick exception remains primary.
                with contextlib.suppress(BaseException):
                    worker.result()
            raise

    # Shield a dedicated driver so cancellation of this public coroutine does
    # not stop its heartbeat while the underlying thread continues to run.
    driver = asyncio.create_task(drive())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(driver)
        except asyncio.CancelledError as exc:
            # ``shield`` also raises CancelledError when the *driver* finished
            # cancelled (for example, operation() itself raised it). That is
            # an outcome, not a new caller cancellation: retrying await on the
            # already-cancelled task would spin forever.
            if driver.done() and driver.cancelled():
                if cancellation is not None:
                    raise cancellation
                raise
            if cancellation is None:
                cancellation = exc
            # Keep waiting through repeated cancellation requests. The shielded
            # driver continues ticking until its bounded worker has completed.
            continue
        except BaseException:
            if cancellation is not None:
                # The await above retrieved the driver's exception; preserve
                # the caller's earlier cancellation as the primary outcome.
                raise cancellation
            raise
        if cancellation is not None:
            raise cancellation
        return result


def run_blocking_with_sync_control_ticks(
    operation: Callable[[], T],
    tick: Callable[[], None],
    period_s: float,
    *,
    drain_tick: Callable[[], None] | None = None,
) -> T:
    """Synchronous counterpart to :func:`run_blocking_with_control_ticks`.

    ``operation`` runs on one non-daemon worker thread while the calling thread
    invokes ``tick`` at a relative cadence. This is intended for synchronous
    robot supervisors whose command API itself hops to the robot event loop.

    A ``KeyboardInterrupt`` is remembered but delayed until the operation has
    completed; ticks continue in the meantime so Ctrl+C cannot turn a bounded
    recorder transition into a lost target stream. A regular tick failure uses
    ``drain_tick`` when supplied. As with the async form, ``operation`` must
    enforce its own finite timeout because a running Python thread cannot be
    killed safely.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive")

    worker_done = threading.Event()
    result: T | None = None
    operation_error: BaseException | None = None

    def run_operation() -> None:
        nonlocal result, operation_error
        try:
            result = operation()
        except BaseException as exc:
            operation_error = exc
        finally:
            worker_done.set()

    worker = threading.Thread(
        target=run_operation,
        name="axol-blocking-control-work",
        daemon=False,
    )
    active_tick = tick
    tick_error: BaseException | None = None
    drain_error: BaseException | None = None
    cancellation: KeyboardInterrupt | None = None

    # On Linux, defer SIGINT across the complete worker lifetime. CPython can
    # deliver it between any two bytecodes (including a finally clause's loop
    # condition), and no arrangement of Python-level try blocks can make every
    # one of those gaps atomic. Restoring the mask only after join delivers a
    # pending Ctrl+C at the first point where abandoning this function is safe.
    previous_mask: set[signal.Signals] | None = None
    if threading.current_thread() is threading.main_thread() and hasattr(
        signal, "pthread_sigmask"
    ):
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    worker_started = False
    try:
        worker.start()
        worker_started = True

        while True:
            try:
                # Keep the condition inside the handler: CPython can deliver
                # SIGINT between any two bytecodes, including loop bookkeeping.
                if worker_done.is_set():
                    break
                tick_started = time.perf_counter()
                active_tick()
                remaining = period_s - (time.perf_counter() - tick_started)
                if remaining > 0 and not worker_done.is_set():
                    worker_done.wait(remaining)
            except KeyboardInterrupt as exc:
                if cancellation is None:
                    cancellation = exc
                # Ctrl+C is external to the tick itself. Continue the same safe
                # target stream while the bounded worker drains.
            except BaseException as exc:
                if tick_error is None:
                    tick_error = exc
                    if drain_tick is None:
                        break
                    active_tick = drain_tick
                else:
                    drain_error = exc
                    break
    finally:
        # A failed tick or a signal anywhere after Thread.start() cannot justify
        # abandoning a worker which may still own an IPC transaction. Keep every
        # loop condition inside its KeyboardInterrupt handler and drain the
        # operation's internally bounded timeout before returning to teardown.
        try:
            if worker_started:
                while True:
                    try:
                        if worker_done.is_set():
                            break
                        worker_done.wait(period_s)
                    except KeyboardInterrupt as exc:
                        if cancellation is None:
                            cancellation = exc
                while True:
                    try:
                        worker.join(period_s)
                        if not worker.is_alive():
                            break
                    except KeyboardInterrupt as exc:
                        if cancellation is None:
                            cancellation = exc
        finally:
            if previous_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    if cancellation is not None:
        raise cancellation
    if tick_error is not None:
        if drain_error is not None:
            tick_error.add_note(f"drain tick also failed: {drain_error!r}")
        if operation_error is not None:
            tick_error.add_note(f"blocking operation also failed: {operation_error!r}")
        raise tick_error
    if operation_error is not None:
        raise operation_error
    return result  # type: ignore[return-value]
