"""
axol motor.flash

Flash a firmware image to a single MyActuator motor over CAN.

The motor reboots into its bootloader and the image is streamed to it as a
YMODEM-1K transfer (see ``almond_axol.motor.firmware``). Nothing else may talk
to the motor while this runs, so power the arm but leave it idle.

Interrupting a flash leaves the motor sitting in its bootloader with a partially
written image: it will not answer normal commands, but re-running this command
recovers it, so do not power-cycle and assume the motor is lost.

Examples:
    axol motor.flash --l --id 0x01 ./RMD-X6-V4.4.bin
    axol motor.flash --r --id 0x03 ./firmware.bin --yes
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from ...motor.bus import CanBus
from ...motor.firmware import FirmwareUpdater
from . import add_side_and_channel_arguments, resolve_channel

# Guards against handing the tool a .zip/.hex or the wrong file entirely; the
# vendor images are tens to a few hundred KiB.
_MAX_REASONABLE_BYTES = 4 * 1024 * 1024


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.flash`` subcommand."""
    p = subparsers.add_parser(
        "motor.flash",
        help="Flash a firmware .bin to a MyActuator motor over CAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # Mantis grippers are Damiao motors; this updater speaks only the
    # MyActuator bootloader protocol.
    add_side_and_channel_arguments(p, supports_mantis=False)
    p.add_argument(
        "firmware",
        type=Path,
        help="Path to the firmware image (.bin)",
    )
    p.add_argument(
        "--id",
        required=True,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="CAN ID of the motor to flash (hex or decimal, e.g. 0x01 or 1)",
    )
    p.add_argument(
        "--frame-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Pause between CAN frames; raise if the adapter's TX queue overruns "
        "(default: 0, matching the vendor tool)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Flash a firmware image to one motor."""
    asyncio.run(_run(args))


def _load_image(path: Path) -> bytes:
    if not path.is_file():
        raise SystemExit(f"error: {path} is not a file")
    image = path.read_bytes()
    if not image:
        raise SystemExit(f"error: {path} is empty")
    if len(image) > _MAX_REASONABLE_BYTES:
        raise SystemExit(
            f"error: {path} is {len(image) / 1024:.0f} KiB, which does not look "
            f"like an RMD firmware image — refusing to flash it"
        )
    if path.suffix.lower() != ".bin":
        print(f"  warning: {path.name} is not a .bin — flashing it anyway")
    return image


def _confirm(channel: str, motor_id: int, path: Path, size: int) -> None:
    print(
        "\n  This overwrites the motor's firmware. A failed or interrupted flash\n"
        "  leaves it in the bootloader until you re-run this command.\n"
    )
    answer = input(
        f"  Flash {path.name} ({size} B) to {motor_id:#04x} on {channel}? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        raise SystemExit("aborted")


async def _run(args: argparse.Namespace) -> None:
    channel = resolve_channel(args)
    path: Path = args.firmware
    image = _load_image(path)

    print(f"\nflash — {channel}  motor {args.id:#04x}")
    print(f"  image:  {path}  ({len(image)} B)")

    if not args.yes:
        _confirm(channel, args.id, path, len(image))

    started = time.monotonic()
    last_pct = -1

    def on_progress(sent: int, total: int) -> None:
        nonlocal last_pct
        pct = sent * 100 // total
        if pct != last_pct:
            last_pct = pct
            print(f"\r  writing: {pct:3d}%  ({sent}/{total} B)", end="", flush=True)

    async with CanBus(channel) as bus:
        updater = FirmwareUpdater(bus, args.id)
        print("  entering bootloader ...")
        await updater.flash(
            image,
            name=path.name,
            frame_delay=args.frame_delay,
            on_progress=on_progress,
        )

    print(f"\n  done in {time.monotonic() - started:.1f}s — power-cycle the motor")
    sys.stdout.flush()
