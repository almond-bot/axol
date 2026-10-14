"""
axol lift.update

Update the jelly_legs lift controller's firmware over CAN.

The image streams into the board's spare flash partition while the running
firmware keeps working; the board only boots it once the whole image has
arrived, its CRC matches, and it is a Jelly Legs build. The new firmware then
boots *on trial* with motion locked, and this command confirms it once it
answers with the expected build. An image that never confirms (crashes, hangs,
power loss, or this command dying) reverts to the previous firmware within
15 s, and the lift's saved position and homing survive the update. An update
takes about a minute; Ctrl-C (or the control panel's Stop) aborts it before
anything is installed.

Without ``--firmware`` the command only reports the running firmware.

Needs lift firmware 0.9 or newer, installed once over USB together with its
A/B partition table (see the jelly_legs firmware README) — older boards do not
answer and must be updated over USB.

Usage:
    axol lift.update                               # show the running firmware
    axol lift.update --firmware build/firmware.bin
    axol lift.update --firmware firmware.bin --force   # reflash the same build
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ...motor import CanBus
from ...robot.lift import resolve_lift_channel
from ...robot.lift_firmware import (
    FirmwareImage,
    FirmwareUpdateError,
    LiftFirmwareUpdater,
)
from . import add_channel_argument, interrupt_event


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``lift.update`` subcommand."""
    p = subparsers.add_parser(
        "lift.update",
        help="Update the lift controller's firmware over CAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--firmware",
        type=Path,
        default=None,
        metavar="FIRMWARE_BIN",
        help="The jelly_legs build's firmware.bin. Omit to only report the "
        "running firmware.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reflash even when the board already runs this build.",
    )
    add_channel_argument(p)
    p.set_defaults(func=run)


async def _run(args: argparse.Namespace) -> None:
    image = None
    if args.firmware is not None:
        try:
            data = args.firmware.read_bytes()
        except OSError as exc:
            raise SystemExit(f"ERROR: cannot read {args.firmware}: {exc}") from exc
        try:
            image = FirmwareImage.parse(data)
        except FirmwareUpdateError as exc:
            raise SystemExit(f"ERROR: {args.firmware}: {exc}.") from None
        print(f"image: {image.describe()}")

    from ..can.setup import bring_up_interfaces, iface_up

    channel = resolve_lift_channel(args.channel)
    if not iface_up(channel):
        bring_up_interfaces([channel])
    bus = CanBus(channel)
    updater = LiftFirmwareUpdater(bus)
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
                    f"ERROR: no firmware-info reply on {channel}: the lift "
                    "board is offline, or runs firmware older than 0.9 "
                    "(update it over USB once)."
                )
            print(info.describe())
            return

        print(f"Updating the lift firmware on {channel} (~1 min; Ctrl-C aborts)...")
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
                    "\nInterrupted — update aborted; the board keeps its "
                    "current firmware."
                ) from None
            except FirmwareUpdateError as exc:
                raise SystemExit(f"ERROR: {exc}.") from None
        if info is not None:
            print(f"Lift firmware updated to {info.version}.")
    finally:
        await bus.close()


def run(args: argparse.Namespace) -> None:
    """Report or update the lift controller's firmware."""
    asyncio.run(_run(args))
