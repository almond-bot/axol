"""
axol safety.update

Update the firmware of Jelly's safety board over CAN.

The board only takes an update with power off, so this command cuts power to
Axol first (a software stop). The image streams into the board's spare flash
partition; the board only boots it once the whole image has arrived, its CRC
matches, and it is a safety board build. The new firmware then boots *on
trial* with power locked off, and this command confirms it once it answers
with the expected build. An image that never confirms (crashes, hangs, power
loss, or this command dying) reverts to the previous firmware within 15 s.
Power stays off afterwards until the green on-switch or ``axol
safety.start``. An update takes a few seconds; Ctrl-C (or the control
panel's Stop) aborts it before anything is installed.

The safety board shares the lift's bus with the jelly_legs lift board. It
answers on its own IDs and rejects the lift's firmware, so updating one
board never touches the other.

Without ``--firmware`` the command only reports the running firmware.

Needs safety board firmware 0.3 or newer, installed once over USB together
with its A/B partition table (see the almond_safety firmware README). Older
boards do not answer and must be updated over USB.

Usage:
    axol safety.update                               # show the running firmware
    axol safety.update --firmware build/firmware.bin
    axol safety.update --firmware firmware.bin --force   # reflash the same build
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ..constants import CAN_BASE, CAN_CHEST
from ..motor import CanBus
from ..robot.safety import resolve_safety_channel
from ..robot.safety_firmware import (
    FirmwareUpdateError,
    SafetyFirmwareUpdater,
    parse_image,
)
from .lift import interrupt_event


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``safety.update`` subcommand."""
    p = subparsers.add_parser(
        "safety.update",
        help="Update the safety board's firmware over CAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--firmware",
        type=Path,
        default=None,
        metavar="FIRMWARE_BIN",
        help="The safety board firmware file (firmware.bin). Omit it to only "
        "report the installed firmware version.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reflash even when the board already runs this build.",
    )
    p.add_argument(
        "--channel",
        default=None,
        help="CAN interface of the safety board (default: the lift's bus, "
        f"{CAN_CHEST} when it exists, otherwise {CAN_BASE})",
    )
    p.set_defaults(func=run)


async def _run(args: argparse.Namespace) -> None:
    image = None
    if args.firmware is not None:
        try:
            data = args.firmware.read_bytes()
        except OSError as exc:
            raise SystemExit(f"ERROR: cannot read {args.firmware}: {exc}") from exc
        try:
            image = parse_image(data)
        except FirmwareUpdateError as exc:
            raise SystemExit(f"ERROR: {args.firmware}: {exc}") from None
        print(f"image: {image.describe()}")

    from .can.setup import bring_up_interfaces, iface_up

    channel = resolve_safety_channel(args.channel)
    if not iface_up(channel):
        bring_up_interfaces([channel])
    # The safety bus is shared and never stalls a motor; keep it out of the
    # stall record like the server's own safety link.
    bus = CanBus(channel, publishes_stall=False)
    updater = SafetyFirmwareUpdater(bus)
    try:
        await bus.start()
    except Exception as exc:  # noqa: BLE001 - reported with a fix
        raise SystemExit(
            f"ERROR: {exc}\nRun `axol can.setup` once to name and bring up the "
            "lift's bus, or pass --channel."
        ) from exc
    try:
        if image is None:
            await updater.quiet_status()
            info = await updater.info()
            if info is None:
                raise SystemExit(
                    f"ERROR: no firmware-info reply on {channel}: the safety "
                    "board is offline, or runs firmware older than 0.3 "
                    "(update it over USB once)."
                )
            print(info.describe())
            return

        print(f"Updating the safety board firmware on {channel} (Ctrl-C aborts)...")
        next_pct = 0

        def progress(done: int, total: int) -> None:
            nonlocal next_pct
            pct = 100 * done // total
            if pct >= next_pct or done == total:
                print(f"  sent {pct:3d}%", flush=True)
                next_pct = pct + 10

        with interrupt_event() as interrupted:
            try:
                info = await updater.flash(
                    image,
                    force=args.force,
                    progress=progress,
                    cancelled=interrupted.is_set,
                )
            except asyncio.CancelledError:
                if not interrupted.is_set():
                    raise
                raise SystemExit(
                    "\nInterrupted. Update aborted; the board keeps its "
                    "current firmware."
                ) from None
            except FirmwareUpdateError as exc:
                raise SystemExit(f"ERROR: {exc}.") from None
        if info is not None:
            print(f"Safety board firmware updated to {info.version}.")
            print(
                "Power is off. Press the green on-switch or run "
                "`axol safety.start` to switch it back on."
            )
    finally:
        await bus.close()


def run(args: argparse.Namespace) -> None:
    """Report or update the safety board's firmware."""
    asyncio.run(_run(args))
