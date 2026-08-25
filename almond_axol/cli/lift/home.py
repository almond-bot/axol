"""
axol lift.home

One-time calibration of the telescoping lift: runs the jelly_legs firmware's
two-ended homing sequence. Both legs drive down until they stall at the
bottom stop, then up to the top stop; on success the firmware rebases its
counters (bottom = 0), sets soft limits a margin inside the hard stops, and
saves everything to flash. Position and limits persist across power cycles
(the legs are self-locking), so homing normally happens **once ever** —
re-run it only if the columns were turned by hand or the firmware died
mid-move.

The sequence takes ~1-2 minutes and intentionally touches both end stops.
Ctrl-C (or the control panel's Stop) aborts it; an aborted homing rolls back
to the previous calibration, so nothing is ever half-homed.

Usage:
    axol lift.home
    axol lift.home --channel can0
"""

from __future__ import annotations

import argparse
import asyncio

from . import (
    Interrupted,
    MotionNeverStarted,
    add_channel_argument,
    fmt_status,
    interrupt_event,
    open_lift,
    watch_motion,
)

# Homing must report itself as running within this window, or we conclude
# the firmware refused/dropped the command.
_START_TIMEOUT_S = 3.0
# Hard cap well past the ~1-2 min a healthy sequence takes.
_HOMING_TIMEOUT_S = 300.0


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``lift.home`` subcommand."""
    p = subparsers.add_parser(
        "lift.home",
        help="Calibrate (home) the telescoping lift against its end stops.",
    )
    add_channel_argument(p)
    p.set_defaults(func=run)


async def _run(args: argparse.Namespace) -> None:
    lift = await open_lift(args.channel)
    try:
        with interrupt_event() as interrupted:
            st = lift.status
            assert st is not None
            if st.homed:
                print(
                    "Lift is already homed (calibration persists in flash) — "
                    "re-homing anyway."
                )
            print("Starting the homing sequence (~1-2 min; Ctrl-C aborts safely)...")
            await lift.home()
            try:
                st = await watch_motion(
                    lift,
                    started=lambda s: s.homing,
                    finished=lambda s: not s.homing,
                    start_timeout_s=_START_TIMEOUT_S,
                    timeout_s=_HOMING_TIMEOUT_S,
                    interrupted=interrupted,
                )
            except MotionNeverStarted:
                raise SystemExit(
                    "ERROR: the board never started homing — check the legs "
                    "are connected and 24 V is on, then re-run."
                ) from None
            except Interrupted:
                await lift.stop_motion()
                raise SystemExit(
                    "\nInterrupted — homing aborted (rolled back to the "
                    "previous calibration)."
                ) from None
            except TimeoutError:
                await lift.stop_motion()
                raise SystemExit(
                    f"ERROR: homing did not finish within "
                    f"{_HOMING_TIMEOUT_S:.0f}s — stopped it. Last status: "
                    f"{fmt_status(lift.status) if lift.status else 'none'}"
                ) from None
        if st.homed and not st.stall_fault:
            print("Homing complete — calibration saved to the board's flash.")
        else:
            raise SystemExit(
                "ERROR: homing did not complete cleanly (rolled back) — check "
                "that both legs are plugged in and powered, then re-run."
            )
    finally:
        await lift.close()


def run(args: argparse.Namespace) -> None:
    """Home the lift and wait for completion."""
    asyncio.run(_run(args))
