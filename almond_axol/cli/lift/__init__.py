"""Shared plumbing for the ``axol lift.*`` commands.

Both commands (``lift.home``, ``lift.goto``) talk to the jelly_legs lift
controller on the chest CAN bus through the :class:`~almond_axol.robot.lift.
Lift` driver, watch its status to completion, and stop the motion on
Ctrl-C / a control-panel Stop (the serve session manager sends SIGINT
first, so ``KeyboardInterrupt`` covers both).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import time

from ...constants import CAN_CHEST
from ...robot.lift import Lift, LiftStatus

# How long to wait for the board's first status reply before giving up.
FIRST_STATUS_TIMEOUT_S = 3.0
# Status print cadence while watching a motion.
WATCH_INTERVAL_S = 0.5
# A command may never be judged from a cached pre-command frame, and a lost
# controller must abort rather than leave HOME/SET_POS running unattended.
STATUS_STALE_S = 1.0


def add_channel_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--channel",
        default=CAN_CHEST,
        help="SocketCAN interface of the chest bus carrying the jelly_legs "
        f"lift controller (default: {CAN_CHEST})",
    )


def fmt_status(st: LiftStatus) -> str:
    """One human-readable status line (mirrors the firmware's flag set)."""
    pct = st.height_percent
    pos = f"{pct:5.1f}%" if pct is not None else "  --- "
    flags = [
        name
        for name, on in (
            ("homed", st.homed),
            ("moving", st.moving),
            ("pos_move", st.pos_move),
            ("STALL", st.stall_fault),
            ("at_lower", st.at_lower),
            ("at_upper", st.at_upper),
            ("homing", st.homing),
            ("jog", st.jog),
        )
        if on
    ]
    return f"pos={pos} vel={st.velocity:+5d} [{' '.join(flags) or 'idle'}]"


async def open_lift(channel: str) -> Lift:
    """Open the chest bus and wait for the board's first status reply.

    Raises ``SystemExit`` with an actionable message when the interface is
    missing or the board never answers.
    """
    lift = Lift(channel)
    try:
        await lift.start()
    except BaseException as exc:
        cleanup_error: BaseException | None = None
        try:
            await lift.close()
        except BaseException as close_exc:
            cleanup_error = close_exc
            exc.add_note(
                "lift startup cleanup also failed: "
                f"{type(close_exc).__name__}: {close_exc}"
            )
        if isinstance(cleanup_error, (KeyboardInterrupt, asyncio.CancelledError)):
            cleanup_error.add_note(
                "lift startup also failed before cleanup was interrupted: "
                f"{type(exc).__name__}: {exc}"
            )
            raise cleanup_error
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            raise
        cleanup_detail = (
            " The partially opened lift bus also could not be closed; restart "
            "the process after restoring CAN."
            if cleanup_error is not None
            else ""
        )
        raise SystemExit(
            f"ERROR: {exc}\nRun `axol can.setup` once to name and bring up "
            f"the chest bus, or pass --channel.{cleanup_detail}"
        ) from exc
    deadline = time.monotonic() + FIRST_STATUS_TIMEOUT_S
    while lift.status is None and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    if lift.status is None:
        await lift.close()
        raise SystemExit(
            f"ERROR: no status from the jelly_legs board on {channel} — "
            f"is the chest powered and wired to this adapter?"
        )
    return lift


def require_motion_preflight(
    lift: Lift,
    *,
    operation: str,
    require_homed: bool,
) -> LiftStatus:
    """Fail closed on stale, faulted, or already-moving lift state."""
    st = lift.status
    if st is None or not lift.status_is_fresh(STATUS_STALE_S):
        raise SystemExit(
            f"ERROR: cannot start {operation}: lift status is stale; check the "
            "chest CAN connection and controller power."
        )
    health = (
        st.driver_fault_mask,
        st.drivers_enabled,
        st.vm_present,
        st.flash_interlock,
        st.save_pending,
    )
    if any(value is None for value in health):
        raise SystemExit(
            f"ERROR: cannot start {operation}: the controller does not report "
            "current driver/interlock health; flash the current Jelly Legs "
            "firmware first."
        )
    if st.driver_fault_mask:
        raise SystemExit(
            f"ERROR: cannot start {operation}: DRV8245 fault mask "
            f"0x{st.driver_fault_mask:02x} is active."
        )
    if st.stall_fault:
        raise SystemExit(
            f"ERROR: cannot start {operation}: a leg stall fault is latched; "
            "inspect and clear the obstruction first."
        )
    if st.flash_interlock:
        raise SystemExit(
            f"ERROR: cannot start {operation}: the saved-position flash "
            "interlock is active; reboot the lift controller first."
        )
    if not st.vm_present:
        raise SystemExit(
            f"ERROR: cannot start {operation}: the 24 V motor supply is absent."
        )
    if not st.drivers_enabled:
        raise SystemExit(
            f"ERROR: cannot start {operation}: the lift motor drivers are disabled."
        )
    if st.save_pending:
        raise SystemExit(
            f"ERROR: cannot start {operation}: a position save is still pending."
        )
    if st.moving or st.pos_move or st.homing or st.jog:
        raise SystemExit(
            f"ERROR: cannot start {operation}: the lift is already moving "
            f"({fmt_status(st)})."
        )
    if require_homed and not st.homed:
        raise SystemExit(
            "ERROR: the lift is not homed, so it has no height scale — run "
            "`axol lift.home` once first."
        )
    return st


class MotionNeverStarted(RuntimeError):
    """The firmware never reported the commanded motion as running.

    Either it silently refused the command, or the command was already
    satisfied (e.g. a position move to the current height) — the caller
    decides which by looking at the final state.
    """


class StopNotVerified(RuntimeError):
    """STOP was sent but no fresh idle controller state confirmed it."""


async def _stop_motion_verified(lift: Lift) -> LiftStatus:
    """Send canonical STOP and require a post-command idle status frame."""
    await lift.stop_motion()
    sent_at = time.monotonic()
    deadline = sent_at + STATUS_STALE_S
    last_stamp = sent_at
    while True:
        stamp = lift.last_status_monotonic
        if stamp is not None and stamp > last_stamp:
            last_stamp = stamp
            status = lift.status
            if status is not None and lift.status_is_fresh(STATUS_STALE_S):
                if not (
                    status.moving or status.pos_move or status.homing or status.jog
                ):
                    return status
        if time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.02)
    raise StopNotVerified(
        "lift STOP was not confirmed by a fresh idle status; keep clear, "
        "restore CAN/controller power, and stop or power off the lift"
    )


class Interrupted(Exception):
    """The operator asked to stop (Ctrl-C, or the control panel's Stop)."""


@contextlib.contextmanager
def interrupt_event():
    """An event set on SIGINT/SIGTERM, handled inside the running loop.

    ``asyncio.run`` offers no reliable way for a coroutine to intercept
    Ctrl-C (the task is simply cancelled), but a lift motion must be
    *stopped on the wire* before the process exits — the firmware runs
    HOME / SET_POS to completion on its own. Loop signal handlers keep the
    loop alive so the stop frame can still be sent. The serve session
    manager's Stop sends SIGINT first (mimicking Ctrl-C), so the control
    panel gets the same clean abort.
    """
    loop = asyncio.get_running_loop()
    event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, event.set)
    try:
        yield event
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


async def watch_motion(
    lift: Lift,
    *,
    started,
    finished,
    start_timeout_s: float,
    timeout_s: float,
    interrupted: asyncio.Event,
    commanded_at: float,
) -> LiftStatus:
    """Print status until a motion runs to completion; return the final status.

    ``started``/``finished`` are predicates on :class:`LiftStatus`. The
    motion must satisfy ``started`` within ``start_timeout_s`` (else
    :class:`MotionNeverStarted`), and the watch ends when ``finished`` holds
    after that. Raises ``TimeoutError`` when ``timeout_s`` passes without
    completion and :class:`Interrupted` when ``interrupted`` is set. The first
    state considered must have arrived strictly after ``commanded_at`` (a
    monotonic timestamp captured after the command send completed). Canonical
    STOP is sent on every exit, including normal completion, timeout, signal,
    stale status, and task cancellation.
    """

    async def _watch_loop() -> LiftStatus:
        t0 = time.monotonic()
        seen_start = False
        next_print = 0.0
        last_stamp = commanded_at
        while True:
            status_deadline = time.monotonic() + STATUS_STALE_S
            while True:
                if interrupted.is_set():
                    raise Interrupted
                stamp = lift.last_status_monotonic
                if stamp is not None and stamp > last_stamp:
                    break
                if time.monotonic() >= status_deadline:
                    raise TimeoutError("lift status stopped updating")
                await asyncio.sleep(0.02)

            if not lift.status_is_fresh(STATUS_STALE_S):
                raise TimeoutError("lift status is stale")
            st = lift.status
            assert st is not None
            last_stamp = stamp
            now = time.monotonic()
            if not seen_start:
                if started(st):
                    seen_start = True
                elif now - t0 > start_timeout_s:
                    raise MotionNeverStarted
            if seen_start and finished(st):
                print(f"  {fmt_status(st)}")
                return st
            if now >= next_print:
                next_print = now + WATCH_INTERVAL_S
                print(f"  {fmt_status(st)}")
            if now - t0 > timeout_s:
                raise TimeoutError

    primary_error: BaseException | None = None
    primary_traceback = None
    result: LiftStatus | None = None
    try:
        result = await _watch_loop()
    except BaseException as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__

    try:
        await _stop_motion_verified(lift)
    except BaseException as cleanup_error:
        if primary_error is not None:
            cleanup_error.add_note(
                "motion watcher also exited with "
                f"{type(primary_error).__name__}: {primary_error}"
            )
            raise cleanup_error from primary_error
        raise

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    assert result is not None
    return result
