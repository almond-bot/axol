"""
axol motor.dump-config

Read every configuration parameter from one or all motors on a bus and print
them, optionally saving a JSON snapshot. The snapshot is what
``axol motor.restore-config`` consumes, so this is the first thing to ask a
remote customer for when their motors are misbehaving.

Works for both motor families — MyActuator's 0xC0 parameters and Damiao's
0x7FF registers — and the type is inferred from the CAN ID.

If ``--id`` is omitted, every motor on the bus (IDs 1-8) is dumped.

Examples:
    axol motor.dump-config --l --id 0x01
    axol motor.dump-config --l --out arm-left.json
    axol motor.dump-config --l --id 0x01 --raw
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

from ...motor.bus import CanBus
from ...motor.config import Access
from ...motor.motor import make_driver
from ...utils.paths import almond_home
from ...utils.state_files import (
    privileged_service_active,
    require_path_beneath,
    secure_atomic_write_json,
)
from . import add_side_and_channel_arguments, resolve_channel


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.dump-config`` subcommand."""
    p = subparsers.add_parser(
        "motor.dump-config",
        help="Read all configuration parameters from a motor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--id",
        default=None,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="Motor CAN ID (hex or decimal, e.g. 0x01 or 1). "
        "If omitted, all motors on the bus (IDs 1-8) are dumped.",
    )
    p.add_argument(
        "--type",
        choices=["myactuator", "damiao"],
        default=None,
        help="Motor driver type (inferred from ID if omitted)",
    )
    p.add_argument(
        "--out",
        default=None,
        type=Path,
        metavar="FILE",
        help="Write a JSON snapshot for motor.restore-config",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Sweep raw parameter indices instead of the known table, to "
        "identify indices that are not yet named (MyActuator only — Damiao's "
        "register table is complete)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Dump configuration parameters from one or all motors."""
    asyncio.run(_run(args))


_ACCESS_TAG = {
    Access.READ_WRITE: "",
    Access.PROTECTED: "  (protected)",
    Access.READ_ONLY: "  (read-only)",
}


async def _dump_motor(
    bus: CanBus, motor_id: int, channel: str, motor_type: str | None, raw: bool
) -> dict[str, Any] | None:
    """Dump one motor, printing a table and returning its JSON record."""
    try:
        motor = make_driver(bus, motor_id, kt=1.0, motor_type=motor_type)
    except ValueError as e:
        print(f"\nmotor-config — {channel}  id={motor_id:#04x}\n")
        print(f"  ERROR: {e}")
        print("  Pass --type myactuator|damiao to query an ID outside the known range.")
        return None

    kind = type(motor).MOTOR_TYPE
    print(f"\nmotor-config — {channel}  type={kind}  id={motor_id:#04x}\n")

    sweep = motor.PARAM_SWEEP_RANGE if raw else None
    if raw and sweep is None:
        print(f"  --raw is not supported for {kind}: its parameter table is complete")
        return None

    try:
        values = await motor.dump_config(sweep)
    except Exception as e:
        print(f"  ERROR: could not read configuration — {e}")
        return None

    if not values:
        print("  no parameters returned")
        return None

    params: dict[str, float] = {}
    for index, value in values.items():
        spec = motor.PARAMS.get(index) if not raw else None
        if spec is None:
            print(f"  {int(index):#04x}{'':<26} {value:>14.4f}")
            params[f"{int(index):#04x}"] = value
            continue
        shown = f"{value:>14.0f}" if spec.integer else f"{value:>14.4f}"
        print(
            f"  {index.name:<30} {shown} {spec.unit:<7}{_ACCESS_TAG[spec.access]}".rstrip()
        )
        params[index.name] = value

    return {"motor_id": motor_id, "type": kind, "raw": raw, "params": params}


async def _run(args: argparse.Namespace) -> None:
    channel = resolve_channel(args)
    motor_ids = (
        [args.id]
        if args.id is not None
        else ([8] if args.target == "mantis" else list(range(1, 9)))
    )

    motors: list[dict[str, Any]] = []
    async with CanBus(channel) as bus:
        for motor_id in motor_ids:
            record = await _dump_motor(bus, motor_id, channel, args.type, args.raw)
            if record is not None:
                motors.append(record)

    if args.out is None:
        return
    if not motors:
        print("\nnothing to save — no motor answered")
        return

    output = args.out
    if privileged_service_active():
        output = require_path_beneath(
            output,
            almond_home(),
            label="motor configuration output",
        )
    secure_atomic_write_json(
        output,
        {"channel": channel, "motors": motors},
        sort_keys=False,
    )
    print(f"\nsaved {len(motors)} motor(s) to {output}")
