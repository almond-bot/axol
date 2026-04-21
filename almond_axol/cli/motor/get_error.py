"""
axol motor.get-error

Read and print the error / status code from a single motor.
The motor type is inferred automatically from the CAN ID.

Examples:
    axol motor.get-error --l --id 0x01
    axol motor.get-error --r --id 0x06
"""

import argparse
import asyncio

from ...motor.bus import CanBus
from ...motor.motor import make_driver
from ...motor.types import MotorStatus


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "motor.get-error",
        help="Read the error / status code from a motor.",
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
        help="Motor CAN ID (hex or decimal, e.g. 0x01 or 1)",
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
    print(f"\nget-error — {channel}  id={args.id:#04x}\n")

    async with CanBus(channel) as bus:
        motor = make_driver(bus, args.id, args.type)

        try:
            status = await motor.get_error_code()
        except Exception as e:
            print(f"  ERROR: could not read motor — {e}")
            print("  Check that the motor is powered and the CAN ID is correct.")
            return

        ok = status == MotorStatus.OK
        label = "OK" if ok else "FAULT"
        print(f"  status  {label}  ({status.value})")
