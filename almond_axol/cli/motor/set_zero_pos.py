"""
almond-axol motor.set-zero-pos

Set the zero position of a single motor to its current position.
The motor type is inferred automatically from the CAN ID.

The motor must be powered and on the CAN bus. The command sets the current
mechanical position as the new zero reference (persisted to flash).

Examples:
    almond-axol motor.set-zero-pos --l --id 0x01
    almond-axol motor.set-zero-pos --r --id 0x06
"""

import argparse
import asyncio

from ...motor.bus import CanBus
from ...motor.damiao import DamiaoMotor
from ...motor.motor import make_driver


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "motor.set-zero-pos",
        help="Set the zero position of a motor to its current position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help="Left arm (can_alm_axol_l)")
    side.add_argument("--r", action="store_true", help="Right arm (can_alm_axol_r)")
    p.add_argument(
        "--id",
        required=True,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="CAN ID of the motor (hex or decimal, e.g. 0x01 or 1)",
    )
    p.add_argument(
        "--type",
        choices=["myactuator", "damiao"],
        default=None,
        help="Motor driver type (inferred from ID if omitted)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    channel = "can_alm_axol_l" if args.l else "can_alm_axol_r"
    print(f"\nset-zero-pos — {channel}  id={args.id:#04x}")

    async with CanBus(channel) as bus:
        motor = make_driver(bus, args.id, args.type)

        position_before = await motor.get_position()
        print(f"  position before: {position_before:.4f} rad")

        print("  setting zero position ...")
        await motor.set_zero_position()

        position_after = await motor.get_position()
        print(f"  position after:  {position_after:.4f} rad")
        print("  done")

        if isinstance(motor, DamiaoMotor):
            print(
                "\n  ⚠  WARNING: Damiao motors require a power cycle to apply the new"
                " zero position. Please restart the motor now."
            )
