"""
axol mantis.update

Update the firmware of both Mantis rigs over CAN, one rig after the other.

The image streams into the rig's spare flash partition while the trigger
keeps working; the rig only boots it once the whole image has arrived, its
CRC matches, and it is a Mantis build. The new firmware then boots *on trial*,
and this command confirms it once it answers with the expected build. An image
that never confirms (crashes, hangs, power loss, or this command dying)
reverts to the previous firmware within 15 s. Each rig takes a few seconds;
Ctrl-C (or the control panel's Stop) aborts it before anything is installed.

Leave the triggers released: each rig measures its trigger's rest position
when it restarts into the new firmware. Don't run this during teleop or data
collection.

The buses come from Settings -> Mantis (``can_mantis_l`` / ``can_mantis_r``
by default). Without ``--firmware`` the command only reports the running
firmware.

Needs Mantis firmware 0.2 or newer, installed once over USB together with its
A/B partition table (see the mantis firmware README). Older rigs do not
answer and must be updated over USB.

Usage:
    axol mantis.update                                # show the running firmware
    axol mantis.update --firmware build/firmware.bin  # both rigs
    axol mantis.update --firmware firmware.bin --side left
    axol mantis.update --firmware firmware.bin --force   # reflash the same build
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ..motor import CanBus
from ..robot.mantis_firmware import (
    FirmwareImage,
    FirmwareUpdateError,
    MantisFirmwareUpdater,
    parse_image,
)
from .lift import interrupt_event

_SIDES = ("left", "right")


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``mantis.update`` subcommand."""
    p = subparsers.add_parser(
        "mantis.update",
        help="Update the Mantis rigs' firmware over CAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--firmware",
        type=Path,
        default=None,
        metavar="FIRMWARE_BIN",
        help="The Mantis firmware file (firmware.bin). Omit it to only report "
        "the installed firmware version.",
    )
    p.add_argument(
        "--side",
        choices=("both", *_SIDES),
        default="both",
        help="Which Mantis rig to update (default: both, left first).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reflash even when the rig already runs this build.",
    )
    p.add_argument(
        "--channel",
        default=None,
        help="CAN interface of one Mantis rig, instead of the buses from "
        "Settings -> Mantis. Updates only that rig.",
    )
    p.set_defaults(func=run)


def resolve_targets(side: str, channel: str | None) -> list[tuple[str, str]]:
    """The ``(name, interface)`` pairs to update, in order ("left rig")."""
    if channel is not None:
        return [(f"{side} rig" if side in _SIDES else "rig", channel)]
    from ..settings import shared_mantis_can_channels

    try:
        channels = dict(zip(_SIDES, shared_mantis_can_channels(), strict=True))
    except Exception as exc:  # noqa: BLE001 - reported with a fix
        raise SystemExit(
            f"ERROR: cannot read the Mantis CAN channels from the settings: {exc}"
        ) from exc
    sides = _SIDES if side == "both" else (side,)
    targets = []
    for name in sides:
        iface = channels[name]
        if not iface:
            if side != "both":
                raise SystemExit(
                    f"ERROR: the {name} rig has no CAN channel. Set it in "
                    "Settings -> Mantis, or pass --channel."
                )
            print(f"Skipping the {name} rig: it has no CAN channel.")
            continue
        targets.append((f"{name} rig", iface))
    return targets


async def _update_one(
    who: str,
    channel: str,
    image: FirmwareImage | None,
    force: bool,
    interrupted: asyncio.Event,
) -> bool:
    """Report or update one board; False when it failed (already printed)."""
    from .can.setup import bring_up_interfaces, iface_up

    bus = CanBus(channel)
    updater = MantisFirmwareUpdater(bus)
    try:
        # A missing interface fails this rig only; the other still updates.
        if not iface_up(channel):
            bring_up_interfaces([channel])
        await bus.start()
    except Exception as exc:  # noqa: BLE001 - reported with a fix
        print(
            f"ERROR: {who}: {exc}\nRun `axol can.setup` once to name "
            "and bring up the Mantis buses, or pass --channel."
        )
        return False
    try:
        if image is None:
            info = await updater.info()
            if info is None:
                print(
                    f"ERROR: {who}: no firmware-info reply on {channel}. "
                    "The rig is unplugged or the hub has no power, or the rig "
                    "runs firmware older than 0.2 (update it over USB once)."
                )
                return False
            print(f"{who} ({channel}): {info.describe()}")
            return True

        print(f"Updating the {who} on {channel} (Ctrl-C aborts)...")
        next_pct = 0

        def progress(done: int, total: int) -> None:
            nonlocal next_pct
            pct = 100 * done // total
            if pct >= next_pct or done == total:
                print(f"  sent {pct:3d}%", flush=True)
                next_pct = pct + 10

        try:
            info = await updater.flash(
                image,
                force=force,
                log=lambda line: print(f"  {line}"),
                progress=progress,
                cancelled=interrupted.is_set,
            )
        except FirmwareUpdateError as exc:
            print(f"ERROR: {who}: {exc}.")
            return False
        if info is not None:
            print(
                f"{who[0].upper()}{who[1:]}: Mantis firmware updated to {info.version}."
            )
        return True
    finally:
        await bus.close()


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

    targets = resolve_targets(args.side, args.channel)
    if not targets:
        raise SystemExit("ERROR: no Mantis CAN channel is set in Settings -> Mantis.")
    failed: list[str] = []
    with interrupt_event() as interrupted:
        for who, channel in targets:
            if interrupted.is_set():
                raise SystemExit(f"\nInterrupted before the {who} update.")
            try:
                ok = await _update_one(who, channel, image, args.force, interrupted)
            except asyncio.CancelledError:
                if not interrupted.is_set():
                    raise
                raise SystemExit(
                    f"\nInterrupted. The {who} update was aborted, and "
                    "it keeps its current firmware."
                ) from None
            if not ok:
                failed.append(who)
    if failed:
        raise SystemExit(f"ERROR: failed on the {' and the '.join(failed)}.")


def run(args: argparse.Namespace) -> None:
    """Report or update the Mantis rigs' firmware."""
    asyncio.run(_run(args))
