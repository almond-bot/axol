"""
almond-axol set-can-id

Change the CAN ID of a single motor and persist it to flash.

The motor must be the only device on the bus, or you must know its current CAN ID.
After running this command, power-cycle the motor to confirm the new ID is active.

Examples:
    almond-axol set-can-id --type myactuator --current-id 0x01 --new-id 0x03
    almond-axol set-can-id --type damiao     --current-id 0x06 --new-id 0x07
    almond-axol set-can-id --channel can_alm_axol_l --type myactuator --current-id 0x01 --new-id 0x03
"""

import argparse
import asyncio

from ..motor.bus import CanBus
from ..motor.damiao import DamiaoMotor
from ..motor.myactuator import MyActuatorMotor


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "set-can-id",
        help="Change the CAN ID of a motor and persist it to flash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--channel",
        default="can0",
        metavar="CHANNEL",
        help="CAN interface name (default: can0)",
    )
    p.add_argument(
        "--type",
        required=True,
        choices=["myactuator", "damiao"],
        help="Motor driver type",
    )
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
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    print(
        f"\nset-can-id — {args.channel}  type={args.type}"
        f"  {args.current_id:#04x} → {args.new_id:#04x}"
    )

    async with CanBus(args.channel) as bus:
        if args.type == "myactuator":
            motor = MyActuatorMotor(bus, args.current_id)
        else:
            motor = DamiaoMotor(
                bus, args.current_id, feedback_id=0x10 + args.current_id
            )

        print("  sending set-can-id command ...")
        await motor.set_can_id(args.new_id)
        print(f"  done — new CAN ID is {args.new_id:#04x}")
        print("  power-cycle the motor to confirm the change.")
