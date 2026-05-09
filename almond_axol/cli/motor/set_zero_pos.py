"""
axol motor.set-zero-pos

Set the zero position of a single motor, or walk every arm joint with
``--guided`` and zero each one against its closer end stop. The current
mechanical position becomes the new zero reference (persisted to flash).

Examples:
    axol motor.set-zero-pos --l --id 0x01
    axol motor.set-zero-pos --r --id 0x06
    axol motor.set-zero-pos --l --guided
"""

import argparse
import asyncio
import math

from ...motor.bus import CanBus
from ...motor.damiao import DamiaoMotor
from ...motor.motor import Motor, make_driver
from ...robot.axol import closer_end_stop
from ...shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, Joint


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
        type=lambda x: int(x, 0),
        default=None,
        metavar="ID",
        help="CAN ID (hex or decimal).  Required unless --guided.",
    )
    p.add_argument(
        "--type",
        choices=["myactuator", "damiao"],
        default=None,
        help="Motor driver type (inferred from ID if omitted)",
    )
    p.add_argument(
        "--guided",
        action="store_true",
        help="Walk every arm joint, zeroing each at its closer end stop.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    if args.guided:
        if args.id is not None:
            print("note: --id is ignored in --guided mode.")
        await _run_guided(args)
        return

    if args.id is None:
        raise SystemExit("error: --id is required (or use --guided).")
    await _run_single(args)


async def _run_single(args: argparse.Namespace) -> None:
    channel = CAN_LEFT if args.l else CAN_RIGHT
    print(f"\nset-zero-pos — {channel}  id={args.id:#04x}")

    async with CanBus(channel) as bus:
        motor = make_driver(bus, args.id, kt=1.0, motor_type=args.type)

        before = await motor.get_position()
        print(f"  before: {before:+.4f} rad")

        await motor.set_zero_position()

        after = await motor.get_position()
        print(f"  after:  {after:+.4f} rad")
        print("  done")

        if isinstance(motor, DamiaoMotor):
            print("\n  ⚠  Damiao motor — power-cycle required to apply.")


# ---------------------------------------------------------------------------
# Guided mode
# ---------------------------------------------------------------------------

# Motion tolerances used to validate each end-stop press (rad).
_MIN_MOTION_RAD = math.radians(3.0)
_MAGNITUDE_WARN_RAD = math.radians(20.0)


def _prompt(msg: str) -> bool:
    """Block on Enter; return ``False`` on Ctrl-C / EOF."""
    try:
        input(msg)
        return True
    except (EOFError, KeyboardInterrupt):
        print("\n    skipped.")
        return False


def _fmt(rad: float) -> str:
    """Format an angle as ``+1.5708 rad (+90.0°)``."""
    return f"{rad:+.4f} rad ({math.degrees(rad):+.1f}°)"


async def _calibrate_joint(
    motor: Motor, joint: Joint, target_rad: float, expected_sign: int
) -> bool:
    """Zero one joint at its closer end stop. Returns ``True`` on success."""
    target_deg = math.degrees(target_rad)
    direction = "−" if expected_sign < 0 else "+"
    print(f"\n— {joint.name}  →  end stop {target_deg:+.1f}° ({direction} motion) —")

    while True:
        if not _prompt("  1) hold the joint inside its range, then Enter: "):
            return False
        p_start = await motor.get_position()
        print(f"     start: {_fmt(p_start)}")

        if not _prompt(
            f"  2) move to the END STOP at {target_deg:+.1f}°, then Enter: "
        ):
            return False
        p_end = await motor.get_position()
        print(f"     end:   {_fmt(p_end)}")

        delta = p_end - p_start
        print(f"     moved: {_fmt(delta)}")

        if abs(delta) < _MIN_MOTION_RAD:
            print("    ✗ no motion — retry.")
            continue

        if (1 if delta > 0 else -1) != expected_sign:
            print(f"    ✗ wrong direction (expected {direction}) — retry.")
            continue

        mag_diff = abs(abs(delta) - abs(target_rad))
        if mag_diff > _MAGNITUDE_WARN_RAD:
            print(
                f"    ⚠  moved {math.degrees(abs(delta)):.1f}°, expected"
                f" ~{abs(target_deg):.1f}° — make sure you're against the stop."
            )

        if not _prompt(f"    set zero at {_fmt(p_end)}? Enter to confirm: "):
            return False

        await motor.set_zero_position()
        print("    ✓ zeroed.")
        return True


async def _run_guided(args: argparse.Namespace) -> None:
    is_left = args.l
    channel = CAN_LEFT if is_left else CAN_RIGHT
    side = "LEFT" if is_left else "RIGHT"
    print(f"\nset-zero-pos --guided — {side} arm  ({channel})")

    async with CanBus(channel) as bus:
        results: list[tuple[Joint, bool]] = []
        any_damiao = False
        for joint in ARM_JOINTS:
            target, sign = closer_end_stop(joint, is_left)
            motor = Motor(bus, joint)
            try:
                ok = await _calibrate_joint(motor, joint, target, sign)
            except KeyboardInterrupt:
                print("\naborted.")
                break
            results.append((joint, ok))
            if ok and isinstance(motor._driver, DamiaoMotor):
                any_damiao = True

        print("\n— summary —")
        for joint, ok in results:
            print(f"  {joint.name:<12} {'zeroed' if ok else 'skipped'}")

        if any_damiao:
            print(
                "\n⚠  Damiao motors zeroed (WRIST_2 / WRIST_3) — power-cycle required."
            )
