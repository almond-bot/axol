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
    except RuntimeError as exc:
        await lift.close()
        raise SystemExit(
            f"ERROR: {exc}\nRun `axol can.setup` once to name and bring up "
            f"the chest bus, or pass --channel."
        ) from None
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


class MotionNeverStarted(RuntimeError):
    """The firmware never reported the commanded motion as running.

    Either it silently refused the command, or the command was already
    satisfied (e.g. a position move to the current height) — the caller
    decides which by looking at the final state.
    """


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
) -> LiftStatus:
    """Print status until a motion runs to completion; return the final status.

    ``started``/``finished`` are predicates on :class:`LiftStatus`. The
    motion must satisfy ``started`` within ``start_timeout_s`` (else
    :class:`MotionNeverStarted`), and the watch ends when ``finished`` holds
    after that. Raises ``TimeoutError`` when ``timeout_s`` passes without
    completion and :class:`Interrupted` when ``interrupted`` is set — the
    motion is left running in both cases; callers stop it.
    """
    t0 = time.monotonic()
    seen_start = False
    next_print = 0.0
    while True:
        now = time.monotonic()
        if interrupted.is_set():
            raise Interrupted
        st = lift.status
        assert st is not None  # open_lift guarantees a first status
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
        await asyncio.sleep(0.1)
