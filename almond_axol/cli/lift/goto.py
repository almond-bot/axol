"""
axol lift.goto

Move the telescoping lift to a target height, as percent of its homed
travel. The default target (75%) is the **robot install height** — high
enough to slide the robot onto its cart mount comfortably.

Requires a homed lift (``axol lift.home``, a one-time calibration persisted
in the board's flash). The firmware decelerates into the target and keeps
both legs level; Ctrl-C (or the control panel's Stop) halts the move
immediately, otherwise it runs to completion on its own.

Usage:
    axol lift.goto                 # 75% — the install height
    axol lift.goto --percent 30
    axol lift.goto --percent 100 --speed 300
"""

from __future__ import annotations

import argparse
import asyncio
import math
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

# The move must report itself as running within this window; one that never
# does is a retarget to (near) the current height, which is already done.
_START_TIMEOUT_S = 2.0
# Full travel is ~18 s at full speed; the operator can cap the speed down to
# the firmware's 40 counts/s floor, which stretches it toward ~5 min.
_MOVE_TIMEOUT_S = 420.0
# How far off the target the final height may land before we call it a
# failure. The firmware stops within ±3 counts (~0.03%); 1% of headroom also
# absorbs the permille rounding of the status readback.
_TOLERANCE_PCT = 1.0


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``lift.goto`` subcommand."""
    p = subparsers.add_parser(
        "lift.goto",
        help="Move the lift to a target height (default 75%% — the robot "
        "install height).",
    )
    p.add_argument(
        "--percent",
        type=float,
        default=75.0,
        help="Target height as percent of homed travel, 0 (fully lowered) "
        "to 100 (fully raised). Default: 75 — the robot install height.",
    )
    p.add_argument(
        "--speed",
        type=int,
        default=0,
        help="Speed cap in encoder counts/s (~650 = full speed, floor 40); "
        "0 = full speed. Default: 0.",
    )
    add_channel_argument(p)
    p.set_defaults(func=run)


async def _run(args: argparse.Namespace) -> None:
    if not math.isfinite(args.percent) or not 0.0 <= args.percent <= 100.0:
        raise SystemExit("ERROR: --percent must be between 0 and 100.")
    if not 0 <= args.speed <= 0xFFFF:
        raise SystemExit("ERROR: --speed must be between 0 and 65535.")
    lift = await open_lift(args.channel)
    try:
        with interrupt_event() as interrupted:
            require_motion_preflight(
                lift,
                operation="position move",
                require_homed=True,
            )
            print(
                f"Moving the lift to {args.percent:.1f}% of travel (Ctrl-C stops it)..."
            )
            moving = lambda s: s.moving or s.pos_move  # noqa: E731

            async def verify_before_send() -> None:
                if interrupted.is_set():
                    raise Interrupted
                require_motion_preflight(
                    lift,
                    operation="position move",
                    require_homed=True,
                )
                if interrupted.is_set():
                    raise Interrupted

            try:
                await lift.set_position(
                    round(args.percent * 10),
                    args.speed,
                    before_send=verify_before_send,
                )
                commanded_at = time.monotonic()
                st = await watch_motion(
                    lift,
                    started=moving,
                    finished=lambda s: not moving(s),
                    start_timeout_s=_START_TIMEOUT_S,
                    timeout_s=_MOVE_TIMEOUT_S,
                    interrupted=interrupted,
                    commanded_at=commanded_at,
                )
            except MotionNeverStarted:
                # A retarget to (near) the current height stops immediately
                # without ever flagging a move — verified against the target
                # below like any other completion.
                st = lift.status
                assert st is not None
            except Interrupted:
                raise SystemExit("\nInterrupted — lift stopped.") from None
            except StopNotVerified as exc:
                raise SystemExit(f"ERROR: {exc}.") from exc
            except TimeoutError:
                raise SystemExit(
                    f"ERROR: the move did not finish within "
                    f"{_MOVE_TIMEOUT_S:.0f}s — stopped it. Last status: "
                    f"{fmt_status(lift.status) if lift.status else 'none'}"
                ) from None
        if st.stall_fault:
            raise SystemExit(
                "ERROR: a leg stalled mid-move and the move was aborted — "
                "clear the obstruction and re-run."
            )
        height = st.height_percent
        if height is None or abs(height - args.percent) > _TOLERANCE_PCT:
            raise SystemExit(
                f"ERROR: move ended at "
                f"{f'{height:.1f}%' if height is not None else 'unknown'} "
                f"instead of {args.percent:.1f}% — re-run, and re-home "
                f"(`axol lift.home`) if it persists."
            )
        print(f"Done — lift at {height:.1f}% of travel.")
    finally:
        await lift.close()


def run(args: argparse.Namespace) -> None:
    """Move the lift to the target height and wait for completion."""
    asyncio.run(_run(args))
