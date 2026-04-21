"""
axol motor.set-can-id

Change the CAN ID of a single motor and persist it to flash.

The motor must be the only device on the bus, or you must know its current CAN ID.

Examples:
    axol motor.set-can-id --l --current-id 0x01 --new-id 0x03 --type myactuator
    axol motor.set-can-id --r --current-id 0x06 --new-id 0x07 --type damiao
"""

import argparse
import asyncio

from ...motor.bus import CanBus
from ...motor.motor import make_driver


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "motor.set-can-id",
        help="Change the CAN ID of a motor and persist it to flash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help="Left arm (can_alm_axol_l)")
    side.add_argument("--r", action="store_true", help="Right arm (can_alm_axol_r)")
    p.add_argument(
        "--current-id",
        required=True,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="Current CAN ID of the motor (hex or decimal, e.g. 0x01 or 1)",
    )
    p.add_argument(
        "--new-id",
        required=True,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="New CAN ID to assign (hex or decimal, e.g. 0x03 or 3)",
    )
    p.add_argument(
        "--type",
        required=True,
        choices=["myactuator", "damiao"],
        help="Motor driver type",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    channel = "can_alm_axol_l" if args.l else "can_alm_axol_r"
    print(f"\nset-can-id — {channel}  {args.current_id:#04x} → {args.new_id:#04x}")

    async with CanBus(channel) as bus:
        motor = make_driver(bus, args.current_id, args.type)

        print("  sending set-can-id command ...")
        await motor.set_can_id(args.new_id)
        print(f"  done — new CAN ID is {args.new_id:#04x}")

        print("  verifying ...")
        voltage = await motor.get_voltage()
        temperature = await motor.get_temperature()
        position = await motor.get_position()
        print(f"  voltage:     {voltage:.1f} V")
        print(f"  temperature: {temperature:.0f} °C")
        print(f"  position:    {position:.4f} rad")
