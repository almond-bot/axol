"""
axol motor.restore-config

Write a JSON snapshot produced by ``axol motor.dump-config`` back to the motors
it came from, returning them to a known-good configuration. Works for both
motor families; each snapshot record carries the motor type it was taken from.

Parameters already holding the saved value are skipped, so restoring onto a
matching motor writes nothing. Read-only parameters are always skipped, and
protected ones — factory calibration, CAN IDs, baud rate — unless
``--include-protected`` is passed, since rewriting those can leave a motor
unable to commutate or unreachable on the bus.

Examples:
    axol motor.restore-config --l arm-left.json
    axol motor.restore-config --l --id 0x01 arm-left.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ...motor.bus import CanBus
from ...motor.config import MotorParam
from ...motor.driver import MotorDriver
from ...motor.errors import MotorError
from ...motor.motor import make_driver
from . import add_side_and_channel_arguments, resolve_channel


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.restore-config`` subcommand."""
    p = subparsers.add_parser(
        "motor.restore-config",
        help="Write a saved configuration snapshot back to motors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument("snapshot", type=Path, help="JSON file from motor.dump-config")
    p.add_argument(
        "--id",
        default=None,
        type=lambda x: int(x, 0),
        metavar="ID",
        help="Restore only this motor CAN ID (hex or decimal). "
        "If omitted, every motor in the snapshot is restored.",
    )
    p.add_argument(
        "--include-protected",
        action="store_true",
        help="Also write protected parameters — factory calibration, CAN IDs, "
        "baud rate (dangerous)",
    )
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Restore configuration parameters to one or all motors in a snapshot."""
    asyncio.run(_run(args))


def _parse_params(
    record: dict[str, Any], motor: MotorDriver
) -> dict[MotorParam, float]:
    """Resolve a snapshot record's parameter names against this motor's table.

    Raw sweeps are rejected rather than partially applied: their keys are bare
    indices with no known meaning, so writing them back could hit anything.
    """
    if record.get("raw"):
        raise ValueError(
            "snapshot was taken with --raw; raw index dumps cannot be restored"
        )
    params: dict[MotorParam, float] = {}
    for name, value in record.get("params", {}).items():
        try:
            params[type(motor).resolve_param(name)] = float(value)
        except MotorError:
            print(f"    skipping unknown parameter {name!r}")
    return params


def _confirm(motor_ids: list[int], channel: str, include_protected: bool) -> bool:
    ids = ", ".join(f"{i:#04x}" for i in motor_ids)
    extra = " including PROTECTED parameters" if include_protected else ""
    reply = input(f"  Restore config to {ids} on {channel}{extra}? [y/N] ")
    return reply.strip().lower() in ("y", "yes")


async def _restore_motor(
    bus: CanBus, record: dict[str, Any], include_protected: bool
) -> None:
    """Restore one motor from its snapshot record."""
    motor_id = int(record["motor_id"])
    print(f"\n  motor {motor_id:#04x}")
    try:
        motor = make_driver(bus, motor_id, kt=1.0, motor_type=record.get("type"))
    except ValueError as e:
        print(f"    ERROR: {e}")
        return

    try:
        params = _parse_params(record, motor)
    except ValueError as e:
        print(f"    ERROR: {e}")
        return
    if not params:
        print("    nothing to restore")
        return

    try:
        written = await motor.restore_config(
            params, include_protected=include_protected
        )
    except Exception as e:
        print(f"    ERROR: could not restore — {e}")
        return

    if not written:
        print("    already matches the snapshot")
        return
    for param in written:
        spec = motor.PARAMS[param]
        shown = f"{params[param]:.0f}" if spec.integer else f"{params[param]:.4f}"
        print(f"    wrote {param.name:<30} {shown:>14} {spec.unit}".rstrip())
    print(f"    persisted {len(written)} parameter(s)")


async def _run(args: argparse.Namespace) -> None:
    channel = resolve_channel(args)

    try:
        snapshot = json.loads(args.snapshot.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {args.snapshot}: {e}")
        return

    records = [
        r
        for r in snapshot.get("motors", [])
        if args.id is None or int(r["motor_id"]) == args.id
    ]
    if not records:
        print(f"no matching motors in {args.snapshot}")
        return

    saved_channel = snapshot.get("channel")
    if saved_channel and saved_channel != channel:
        print(f"note: snapshot was taken on {saved_channel}, restoring to {channel}")

    if not args.yes and not _confirm(
        [int(r["motor_id"]) for r in records], channel, args.include_protected
    ):
        print("aborted")
        return

    async with CanBus(channel) as bus:
        for record in records:
            await _restore_motor(bus, record, args.include_protected)
