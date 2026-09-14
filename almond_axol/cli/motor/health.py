"""
axol motor.health

Probe every motor on both arms and report which responded.
Runs the same status reads as motor.info on all 16 motors.

Examples:
    axol motor.health
    axol motor.health --no-right --left-channel can0   # one arm, non-hub adapter
"""

import argparse
import asyncio

from ...constants import (
    CAN_LEFT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    CAN_RIGHT,
    Joint,
)
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


async def _check_arm(
    channel: str, joints: list[Joint] | None = None
) -> list[tuple[Joint, str | None]]:
    results: list[tuple[Joint, str | None]] = []
    async with CanBus(channel) as bus:
        for joint in joints or Joint:
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
    p.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    p.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    p.add_argument(
        "--target",
        choices=["axol", "mantis"],
        default="axol",
        help="Probe the Axol arms or Mantis grippers (default: %(default)s).",
    )
    p.add_argument(
        "--left-channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface for the left arm, for setups without the "
        "selected hardware's default bus.",
    )
    p.add_argument(
        "--right-channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface for the right arm, for setups without the "
        "selected hardware's default bus.",
    )
    p.add_argument(
        "--joints",
        default=None,
        help="Comma-separated joints to probe. Defaults to all Axol joints or "
        "the gripper only for Mantis.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Probe every motor on the selected arms."""
    if args.no_left and args.no_right:
        raise SystemExit("Cannot skip both arms.")
    defaults = (
        (CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT)
        if args.target == "mantis"
        else (CAN_LEFT, CAN_RIGHT)
    )
    joints = _parse_joints(args.joints, target=args.target)
    arms = []
    if not args.no_left:
        arms.append(("left", args.left_channel or defaults[0]))
    if not args.no_right:
        arms.append(("right", args.right_channel or defaults[1]))
    failed = asyncio.run(_run(arms, joints))
    if failed:
        raise SystemExit(1)


def _parse_joints(spec: str | None, *, target: str) -> list[Joint]:
    if not spec:
        return [Joint.GRIPPER] if target == "mantis" else list(Joint)
    by_name = {joint.value: joint for joint in Joint}
    names = [name.strip().lower() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"Unknown joint(s): {', '.join(unknown)}")
    return [joint for joint in Joint if joint.value in names]


async def _run(
    arms: list[tuple[str, str]], joints: list[Joint] | None = None
) -> list[tuple[str, Joint]]:
    """Return the list of motors that failed to respond."""
    failed: list[tuple[str, Joint]] = []

    for side, channel in arms:
        print(f"{side.upper()} ({channel})")
        for joint, error in await _check_arm(channel, joints):
            motor_id = _MOTOR_IDS[joint]
            label = f"  {joint.name:<11} id={motor_id:#04x}"
            if error is not None:
                print(f"{label}  {error}")
                failed.append((side, joint))
            else:
                print(f"{label}  {_OK}")
        print()

    return failed
