"""
almond-axol motor-info

Read and print status information from a single motor.
Useful for verifying that a motor is reachable at a given CAN ID.

Examples:
    almond-axol motor-info --type myactuator --id 0x01
    almond-axol motor-info --type damiao     --id 0x06
    almond-axol motor-info --type myactuator --id 0x01 --channel can_alm_axol_l
"""

import argparse
import asyncio
import math

from ..motor.bus import CanBus
from ..motor.damiao import DamiaoMotor
from ..motor.myactuator import MyActuatorMotor


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "motor-info",
        help="Read status from a motor to verify it is reachable.",
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
        "--id",
        required=True,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="Motor CAN ID (hex or decimal, e.g. 0x01 or 1)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    print(f"\nmotor-info — {args.channel}  type={args.type}  id={args.id:#04x}\n")

    async with CanBus(args.channel) as bus:
        if args.type == "myactuator":
            motor = MyActuatorMotor(bus, args.id)
        else:
            motor = DamiaoMotor(bus, args.id, feedback_id=0x10 + args.id)

        try:
            position = await motor.get_position()
            velocity = await motor.get_velocity()
            torque = await motor.get_torque()
            temperature = await motor.get_temperature()
            voltage = await motor.get_voltage()
            status = await motor.get_error_code()
        except Exception as e:
            print(f"  ERROR: could not read motor — {e}")
            print("  Check that the motor is powered and the CAN ID is correct.")
            return

        torque_label = "current" if args.type == "myactuator" else "torque"
        torque_unit = "A" if args.type == "myactuator" else "Nm"

        print(f"  status      {status.value}")
        print(f"  position    {position:.4f} rad  ({math.degrees(position):.2f}°)")
        print(f"  velocity    {velocity:.4f} rad/s  ({math.degrees(velocity):.2f}°/s)")
        print(f"  {torque_label:<10}  {torque:.4f} {torque_unit}")
        print(f"  temperature {temperature:.1f} °C")
        print(f"  voltage     {voltage:.1f} V")
