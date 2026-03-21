"""
axol motor.info

Read and print status information from a single motor.
Useful for verifying that a motor is reachable at a given CAN ID.
The motor type is inferred automatically from the CAN ID.

Examples:
    axol motor.info --l --id 0x01
    axol motor.info --r --id 0x06
"""

import argparse
import asyncio
import math

from ...motor.bus import CanBus
from ...motor.motor import make_driver


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "motor.info",
        help="Read status from a motor to verify it is reachable.",
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
    async with CanBus(channel) as bus:
        motor = make_driver(bus, args.id, kt=1.0, motor_type=args.type)
        motor_type = type(motor).__name__.removesuffix("Motor").lower()
        print(f"\nmotor-info — {channel}  type={motor_type}  id={args.id:#04x}\n")

        try:
            position = await motor.get_position()
            velocity = await motor.get_velocity()
            torque = await motor.get_torque()
            temperature = await motor.get_temperature()
            voltage = await motor.get_voltage()
            status = await motor.get_error_code()
            control_mode = await motor.get_control_mode()
        except Exception as e:
            print(f"  ERROR: could not read motor — {e}")
            print("  Check that the motor is powered and the CAN ID is correct.")
            return

        print(f"  status      {status.value}")
        if control_mode is not None:
            print(f"  mode        {control_mode.name}")
        else:
            print("  mode        N/A (determined per-command)")
        print(
            f"  position    {position:.4f} rad  ({math.degrees(position):.2f}°)  ({position / (2 * math.pi):.4f} rev)"
        )
        print(f"  velocity    {velocity:.4f} rad/s  ({math.degrees(velocity):.2f}°/s)")
        print(f"  torque      {torque:.4f} Nm")
        print(f"  temperature {temperature:.1f} °C")
        print(f"  voltage     {voltage:.1f} V")
