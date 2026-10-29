"""
axol safety.status / safety.stop / safety.start / safety.power-cycle

Jelly's safety board (on Jelly builds that carry it) from the command line,
through the :class:`~almond_axol.robot.safety.SafetyBoard` driver:

- ``safety.status`` prints the E-stop chain once (``--watch`` keeps printing).
- ``safety.stop`` is the software E-stop: it cuts power to Axol, like the
  E-Stop, and waits for the board to confirm.
- ``safety.start`` switches power back on, like the green on-switch. The
  board refuses while the E-Stop is pressed.
- ``safety.power-cycle`` switches power off for a few seconds and back on,
  which resets every motor on Axol's supply: it clears motor faults and
  applies new Damiao zeros. The arms drop while power is off.

Usage:
    axol safety.status
    axol safety.status --watch
    axol safety.stop
    axol safety.start
    axol safety.power-cycle
"""

from __future__ import annotations

import argparse
import asyncio

from ..constants import CAN_BASE, CAN_CHEST
from ..robot.safety import SafetyBoard, SafetyError, SafetyStatus

_WATCH_INTERVAL_S = 0.5


def fmt_status(st: SafetyStatus) -> str:
    """One human-readable status line."""
    if st.output_on:
        state = "POWER ON"
    elif st.update_locked:
        state = "FIRMWARE UPDATE (power locked off until it finishes)"
    elif st.estop_pressed:
        state = "E-STOP PRESSED"
    elif st.software_stopped:
        state = f"SOFTWARE STOP ({st.stop_cause})"
    else:
        state = "power off (press the green on-switch or run safety.start)"
    faults = [
        name
        for name, on in (
            ("k2_stuck", st.k2_stuck),
            ("k2_no_close", st.k2_no_close),
            ("k1_pole_b", st.k1_pole_b),
            ("start_failed", st.remote_start_failed),
            ("heartbeat_lost", st.heartbeat_lost),
            ("supply_dip", st.supply_dip),
        )
        if on
    ]
    return (
        f"{state} | battery {st.supply_volts:.1f} V, output {st.output_volts:.1f} V"
        f" | faults: {' '.join(faults) or 'none'} | firmware {st.firmware_version}"
    )


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``safety.*`` subcommands."""
    for name, help_text, func in (
        ("safety.status", "Show the state of Jelly's E-stop chain.", _status),
        ("safety.stop", "Software E-stop: cut power to Axol.", _stop),
        ("safety.start", "Switch power back on, like the green on-switch.", _start),
        (
            "safety.power-cycle",
            "Switch power off and back on to reset Axol's motors.",
            _power_cycle,
        ),
    ):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument(
            "--channel",
            default=None,
            help="CAN interface of the safety board (default: the lift's bus, "
            f"{CAN_CHEST} when it exists, otherwise {CAN_BASE})",
        )
        if name == "safety.status":
            p.add_argument(
                "--watch", action="store_true", help="Keep printing until Ctrl-C."
            )
        p.set_defaults(func=lambda args, func=func: _main(func, args))


def _main(func, args: argparse.Namespace) -> None:  # noqa: ANN001
    try:
        asyncio.run(_with_board(func, args))
    except SafetyError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    except KeyboardInterrupt:
        pass


async def _with_board(func, args: argparse.Namespace) -> None:  # noqa: ANN001
    board = SafetyBoard(args.channel)
    await board.start()
    try:
        try:
            await board.request_status()
        except SafetyError:
            raise SafetyError(
                f"No safety board answered on {board.channel}. Check that "
                "Jelly is on and that this Jelly has a safety board."
            ) from None
        await func(board, args)
    finally:
        await board.close()


async def _status(board: SafetyBoard, args: argparse.Namespace) -> None:
    while True:
        print(fmt_status(await board.request_status()), flush=True)
        if not args.watch:
            return
        await asyncio.sleep(_WATCH_INTERVAL_S)


async def _stop(board: SafetyBoard, args: argparse.Namespace) -> None:
    status = await board.stop()
    print("Power is off.")
    print(fmt_status(status))
    print("Press the green on-switch or run `axol safety.start` to switch it back on.")


async def _start(board: SafetyBoard, args: argparse.Namespace) -> None:
    status = await board.remote_start()
    print("Power is on.")
    print(fmt_status(status))


async def _power_cycle(board: SafetyBoard, args: argparse.Namespace) -> None:
    print("Switching power off for a few seconds. The arms drop.", flush=True)
    status = await board.power_cycle()
    print("Power is back on. The motors have restarted.")
    print(fmt_status(status))
