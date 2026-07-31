"""
axol motor.set-config

Read or write a single configuration parameter on either motor family —
MyActuator's 0xC0 parameters or Damiao's 0x7FF registers. The type is inferred
from the CAN ID, and parameter names are unique across families.

With ``--value`` the parameter is written and persisted; without it the current
value is just read back. Use ``axol motor.dump-config`` to see everything at
once.

Read-only parameters (firmware versions, measured winding constants) are always
refused. Protected ones — factory calibration, CAN IDs, baud rate — need
``--force-protected``, since getting one wrong can leave a motor unable to
commutate or unreachable on the bus.

Damiao's CAN timeout is handled in milliseconds here; the underlying register
counts 50 us ticks, and the conversion is applied for you.

Examples:
    axol motor.set-config --l --id 0x01 --param LOW_VOLTAGE
    axol motor.set-config --l --id 0x01 --param LOW_VOLTAGE --value -2.0
    axol motor.set-config --r --id 0x07 --param TIMEOUT --value 1000
"""

import argparse
import asyncio

from ...motor.bus import CanBus
from ...motor.config import ALL_PARAM_NAMES, Access
from ...motor.errors import MotorError
from ...motor.motor import make_driver


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.set-config`` subcommand."""
    p = subparsers.add_parser(
        "motor.set-config",
        help="Read or write one motor configuration parameter.",
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
        "--param",
        required=True,
        choices=ALL_PARAM_NAMES,
        help="Parameter to read or write",
    )
    p.add_argument(
        "--value",
        type=float,
        default=None,
        help="New value; omit to read the current value without writing",
    )
    p.add_argument(
        "--type",
        choices=["myactuator", "damiao"],
        default=None,
        help="Motor driver type (inferred from ID if omitted)",
    )
    p.add_argument(
        "--force-protected",
        action="store_true",
        help="Allow writing a protected parameter (calibration, CAN ID, baud)",
    )
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Read or write one configuration parameter."""
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    channel = "can_alm_axol_l" if args.l else "can_alm_axol_r"

    async with CanBus(channel) as bus:
        try:
            motor = make_driver(bus, args.id, kt=1.0, motor_type=args.type)
        except ValueError as e:
            print(f"error: {e}")
            raise SystemExit(1) from e

        kind = type(motor).MOTOR_TYPE
        try:
            param = type(motor).resolve_param(args.param)
        except MotorError as e:
            print(f"error: {e}")
            print(f"  ({args.param} belongs to the other motor family)")
            raise SystemExit(1) from e

        spec = motor.PARAMS[param]
        unit = f" {spec.unit}" if spec.unit else ""
        fmt = "{:.0f}" if spec.integer else "{:.4f}"
        print(
            f"\nmotor-config — {channel}  type={kind}  id={args.id:#04x}  "
            f"{param.name} ({int(param):#04x})"
        )

        if args.value is not None:
            if spec.access is Access.READ_ONLY:
                print(f"\n  {param.name} is read-only and cannot be written.")
                raise SystemExit("aborted")
            if spec.access is Access.PROTECTED and not args.force_protected:
                print(
                    f"\n  {param.name} is a protected parameter (factory calibration,\n"
                    "  CAN identity, or baud rate). Changing it can leave the motor\n"
                    "  unable to commutate or unreachable on the bus. Pass\n"
                    "  --force-protected if you really mean to change it."
                )
                raise SystemExit("aborted")

        current = await motor.read_config(param)
        print(f"  current: {fmt.format(current)}{unit}")

        if args.value is None:
            return
        if current == args.value:
            print("  already at the requested value — nothing written")
            return

        if not args.yes:
            answer = input(
                f"  Set {param.name} to {fmt.format(args.value)}{unit}? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                raise SystemExit("aborted")

        await motor.write_config(param, args.value)
        readback = await motor.read_config(param)
        print(f"  wrote:   {fmt.format(readback)}{unit}  (persisted)")
        if readback != args.value:
            print(
                f"  warning: motor stored {fmt.format(readback)}{unit}, not "
                f"{fmt.format(args.value)}{unit} — it may clamp this parameter"
            )
