"""
axol motor.set-zero-pos

Set the zero position of a single motor, or use ``--guided`` to zero every
arm joint against its closer end stop. Guided mode first captures an in-range
start reference for all joints at once, then walks each joint to its end stop
and zeros it. The current mechanical position becomes the new zero reference
(persisted to flash).

Note: each motor's encoder zero is calibrated at one of the joint's
mechanical END STOPS, not at the robot's rest position. ``AxolArm`` adds a
per-joint offset so the public API stays in joint frame (``0`` = rest).

Examples:
    axol motor.set-zero-pos --l --id 0x01
    axol motor.set-zero-pos --r --id 0x06
    axol motor.set-zero-pos --l --guided
"""

import argparse
import asyncio
import math
import sys

from ...constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, Joint
from ...motor.bus import CanBus
from ...motor.damiao import DamiaoMotor
from ...motor.motor import Motor, make_driver
from ...robot.axol import closer_end_stop

# Marker prefix a --web-prompts step prints before blocking on stdin (same
# convention as the ROM diagnostics); the dashboard shows a Continue button.
PROMPT_MARKER = "[prompt]"


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motor.set-zero-pos`` subcommand."""
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
    p.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit '[prompt] ...' markers and block on stdin for the guided "
        "steps, so the web dashboard can drive them with a Continue button "
        "(set automatically by the dashboard).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Set a motor's zero position (single ``--id`` or ``--guided`` mode)."""
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


def _prompt(instruction: str, web_prompts: bool) -> bool:
    """Block until the operator confirms a hands-on step.

    ``--web-prompts``: emit a ``[prompt]`` marker the dashboard turns into a
    Continue button, then block until it writes a line to our stdin. Otherwise
    an ordinary tty Enter prompt. Returns ``False`` on Ctrl-C / EOF.
    """
    try:
        if web_prompts:
            print(f"{PROMPT_MARKER} {instruction}", flush=True)
            if sys.stdin.readline() == "":
                raise EOFError
            return True
        input(f"  {instruction} — press Enter: ")
        return True
    except (EOFError, KeyboardInterrupt):
        print("\n    skipped.")
        return False


def _fmt(rad: float) -> str:
    """Format an angle as ``+1.5708 rad (+90.0°)``."""
    return f"{rad:+.4f} rad ({math.degrees(rad):+.1f}°)"


async def _calibrate_joint(
    motor: Motor,
    joint: Joint,
    p_start: float,
    target_rad: float,
    expected_sign: int,
    web_prompts: bool,
) -> bool:
    """Zero one joint at its closer end stop, measured from ``p_start``.

    Returns ``True`` on success.
    """
    target_deg = math.degrees(target_rad)
    direction = "−" if expected_sign < 0 else "+"
    print(f"\n— {joint.name}  →  end stop {target_deg:+.1f}° ({direction} motion) —")
    print(f"     start: {_fmt(p_start)}")

    # Prompts are self-contained (they name the joint) since the dashboard
    # shows them on a Continue button without the surrounding log context.
    while True:
        if not _prompt(
            f"{joint.name}: move to the END STOP at {target_deg:+.1f}°",
            web_prompts,
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
            # Wrong direction: rather than restart, send the user to the
            # correct end stop and zero there on the next Enter.
            print(
                f"    ✗ wrong direction (expected {direction}) — move all the way"
                f" to the OTHER end stop at {target_deg:+.1f}°."
            )
            if not _prompt(
                f"{joint.name}: wrong direction — move all the way to the OTHER "
                f"end stop at {target_deg:+.1f}°, then zero",
                web_prompts,
            ):
                return False
            p_end = await motor.get_position()
            print(f"     end:   {_fmt(p_end)}")
        else:
            mag_diff = abs(abs(delta) - abs(target_rad))
            if mag_diff > _MAGNITUDE_WARN_RAD:
                print(
                    f"    ⚠  moved {math.degrees(abs(delta)):.1f}°, expected"
                    f" ~{abs(target_deg):.1f}° — make sure you're against the stop."
                )

        # Right direction → zero immediately, no extra confirmation.
        await motor.set_zero_position()
        print(f"    ✓ zeroed at {_fmt(p_end)}.")
        return True


async def _run_guided(args: argparse.Namespace) -> None:
    is_left = args.l
    channel = CAN_LEFT if is_left else CAN_RIGHT
    side = "LEFT" if is_left else "RIGHT"
    print(f"\nset-zero-pos --guided — {side} arm  ({channel})")

    async with CanBus(channel) as bus:
        motors = {joint: Motor(bus, joint) for joint in ARM_JOINTS}

        # Capture an in-range start reference for every joint up front, so the
        # operator positions the whole arm once rather than once per joint.
        if not _prompt(
            "hold EVERY joint somewhere inside its range (away from the end stops)",
            args.web_prompts,
        ):
            print("\naborted.")
            return
        starts = dict(
            zip(
                ARM_JOINTS,
                await asyncio.gather(*[motors[j].get_position() for j in ARM_JOINTS]),
            )
        )
        for joint in ARM_JOINTS:
            print(f"  {joint.name:<12} start: {_fmt(starts[joint])}")

        results: list[tuple[Joint, bool]] = []
        any_damiao = False
        for joint in ARM_JOINTS:
            target, sign = closer_end_stop(joint, is_left)
            motor = motors[joint]
            try:
                ok = await _calibrate_joint(
                    motor, joint, starts[joint], target, sign, args.web_prompts
                )
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
