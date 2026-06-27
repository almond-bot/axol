"""
axol motor.health

Probe every motor on both arms and report which responded.
Runs the same status reads as motor.info on all 16 motors.

Examples:
    axol motor.health
"""

import argparse
import asyncio

from ...constants import CAN_LEFT, CAN_RIGHT, Joint
from ...motor.bus import CanBus
from ...motor.motor import Motor

# CAN IDs 0x01–0x08 in Joint control order.
_MOTOR_IDS: dict[Joint, int] = {
    joint: motor_id for joint, motor_id in zip(Joint, range(0x01, 0x09), strict=True)
}

_OK = "\033[32mOK\033[0m"


async def _probe_motor(motor: Motor) -> str | None:
    """Run the same reads as motor.info; return an error string on failure."""
    try:
        await motor.get_position()
        await motor.get_velocity()
        await motor.get_torque()
        await motor.get_temperature()
        await motor.get_voltage()
        await motor.get_error_code()
        await motor.get_control_mode()
        await motor.get_firmware_version()
        await motor.get_model()
    except Exception as e:
        return str(e)
    return None


async def _check_arm(channel: str) -> list[tuple[Joint, str | None]]:
    results: list[tuple[Joint, str | None]] = []
    async with CanBus(channel) as bus:
        for joint in Joint:
            result = await _probe_motor(Motor(bus, joint))
            results.append((joint, result))
    return results


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.health`` subcommand."""
    p = subparsers.add_parser(
        "motor.health",
        help="Probe all motors and report which responded.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Probe every motor on both arms."""
    failed = asyncio.run(_run())
    if failed:
        raise SystemExit(1)


async def _run() -> list[tuple[str, Joint]]:
    """Return the list of motors that failed to respond."""
    failed: list[tuple[str, Joint]] = []
    arms = [("left", CAN_LEFT), ("right", CAN_RIGHT)]

    for side, channel in arms:
        print(f"{side.upper()} ({channel})")
        for joint, error in await _check_arm(channel):
            motor_id = _MOTOR_IDS[joint]
            label = f"  {joint.name:<11} id={motor_id:#04x}"
            if error is not None:
                print(f"{label}  {error}")
                failed.append((side, joint))
            else:
                print(f"{label}  {_OK}")
        print()

    return failed
