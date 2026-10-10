"""
axol lift.home

One-time calibration of the telescoping lift: runs the jelly_legs firmware's
two-ended homing sequence. The legs drive up until they stall at the top
stop, then down to the bottom stop; on success the firmware rebases its
counters (bottom = 0), sets soft limits a margin inside the hard stops, and
saves everything to flash. Position and limits persist across power cycles
(the legs are self-locking), so homing normally happens **once ever** —
re-run it only if the columns were turned by hand or the firmware died
mid-move.

By default the legs home **together** — the mode for legs bolted to the
robot: they move as one and the first leg to reach a stop stops both, so the
frame is never racked (a robot on top is fine). ``--independent`` homes each
leg against its own stops instead, which levels loose legs that start at
different heights — use it only with the legs off the robot, where one leg
pushing on alone cannot rack anything. Lift firmware before v0.9 ignores the
choice and always homes the legs independently.

The sequence takes ~1-2 minutes and intentionally touches both end stops.
Ctrl-C (or the control panel's Stop) aborts it; an aborted homing rolls back
to the previous calibration, so nothing is ever half-homed.

Usage:
    axol lift.home
    axol lift.home --independent      # loose legs, off the robot
    axol lift.home --channel can0
"""

from __future__ import annotations

import argparse
import asyncio
import time

from . import (
    Interrupted,
    MotionNeverStarted,
    StopNotVerified,
    add_channel_argument,
    fmt_status,
    interrupt_event,
    open_lift,
    require_motion_preflight,
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
    p.add_argument(
        "--independent",
        action="store_true",
        help="Home each leg against its own end stops, levelling legs that "
        "start at different heights. Only with the legs OFF the robot - "
        "mounted, one leg would push on alone and rack the frame. Default: "
        "the legs home together, as one (safe on the robot).",
    )
    add_channel_argument(p)
    p.set_defaults(func=run)


async def _run(args: argparse.Namespace) -> None:
    lift = await open_lift(args.channel)
    try:
        with interrupt_event() as interrupted:
            st = require_motion_preflight(
                lift,
                operation="homing",
                require_homed=False,
            )
            if st.homed:
                print(
                    "Lift is already homed (calibration persists in flash) — "
                    "re-homing anyway."
                )
            if args.independent:
                print(
                    "Independent homing: each leg runs to its own stops. Only "
                    "with the legs OFF the robot."
                )
            mode = "independent" if args.independent else "legs-together"
            print(
                f"Starting the {mode} homing sequence "
                "(~1-2 min; Ctrl-C aborts safely)..."
            )
            saw_independent = False

            def started(s) -> bool:  # noqa: ANN001
                nonlocal saw_independent
                saw_independent |= bool(s.independent_homing)
                return s.homing

            def finished(s) -> bool:  # noqa: ANN001
                nonlocal saw_independent
                saw_independent |= bool(s.independent_homing)
                return not s.homing

            async def verify_before_send() -> None:
                if interrupted.is_set():
                    raise Interrupted
                require_motion_preflight(
                    lift,
                    operation="homing",
                    require_homed=False,
                )
                if interrupted.is_set():
                    raise Interrupted

            try:
                await lift.home(
                    independent=args.independent, before_send=verify_before_send
                )
                commanded_at = time.monotonic()
                st = await watch_motion(
                    lift,
                    started=started,
                    finished=finished,
                    start_timeout_s=_START_TIMEOUT_S,
                    timeout_s=_HOMING_TIMEOUT_S,
                    interrupted=interrupted,
                    commanded_at=commanded_at,
                )
            except MotionNeverStarted:
                raise SystemExit(
                    "ERROR: the board never started homing — check the legs "
                    "are connected and 24 V is on, then re-run."
                ) from None
            except Interrupted:
                raise SystemExit(
                    "\nInterrupted — homing aborted (rolled back to the "
                    "previous calibration)."
                ) from None
            except StopNotVerified as exc:
                raise SystemExit(f"ERROR: {exc}.") from exc
            except TimeoutError:
                raise SystemExit(
                    f"ERROR: homing did not finish within "
                    f"{_HOMING_TIMEOUT_S:.0f}s — stopped it. Last status: "
                    f"{fmt_status(lift.status) if lift.status else 'none'}"
                ) from None
        if st.homed and not st.stall_fault:
            print("Homing complete — calibration saved to the board's flash.")
            if args.independent and not saw_independent:
                print(
                    "Note: the lift firmware never reported independent mode — "
                    "it predates v0.9, which always homes the legs "
                    "independently anyway."
                )
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
