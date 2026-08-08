"""
axol motor.set-zero-pos

Set the zero position of a single motor, or use ``--guided`` to zero every
arm joint against an end stop. Guided mode first captures an in-range start
reference for all joints at once, then walks each joint to its end stop and
zeros it. The current mechanical position becomes the new zero reference
(persisted to flash).

Most joints are zeroed at one specific end stop (the closer one). WRIST_2 and
WRIST_3 accept EITHER end stop — their stops are not laser-aligned, so per
unit one may be better placed than the other; move to whichever you trust and
the runtime detects the side automatically from the encoder reading.

Note: each motor's encoder zero is calibrated at one of the joint's
mechanical END STOPS, not at the robot's rest position. ``AxolArm`` adds a
per-joint offset so the public API stays in joint frame (``0`` = rest).

Examples:
    axol motor.set-zero-pos --l --id 0x01
    axol motor.set-zero-pos --r --id 0x06
    axol motor.set-zero-pos --l --guided
    axol motor.set-zero-pos --l --guided --joints wrist_2,wrist_3
"""

import argparse
import asyncio
import math
import sys

from ...constants import ARM_JOINTS, Joint
from ...motor.bus import CanBus
from ...motor.damiao import DamiaoMotor
from ...motor.motor import Motor, make_driver
from ...robot.axol import EITHER_STOP_JOINTS, closer_end_stop
from . import add_side_and_channel_arguments, resolve_channel

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
    add_side_and_channel_arguments(p)
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
        help="Walk the arm joints, zeroing each at an end stop "
        "(wrist_2/wrist_3 accept either side).",
    )
    p.add_argument(
        "--joints",
        default=None,
        help="Comma-separated subset of arm joints to walk in --guided mode "
        "(e.g. wrist_2,wrist_3). Default: all seven. One of: "
        f"{', '.join(j.value for j in ARM_JOINTS)}.",
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

    if args.joints is not None:
        raise SystemExit("error: --joints only applies to --guided mode.")
    if args.id is None:
        raise SystemExit("error: --id is required (or use --guided).")
    await _run_single(args)


def _parse_arm_joints(spec: str | None) -> list[Joint]:
    """Parse a ``--joints`` spec into an ordered subset of :data:`ARM_JOINTS`.

    ``None`` or empty selects all seven arm joints. Names match the joint enum
    values (e.g. ``shoulder_1``, ``elbow``). The gripper has no zero to set
    (it self-calibrates against its hard stops at enable time), so it is not
    a valid choice here.
    """
    if not spec:
        return list(ARM_JOINTS)
    by_value = {j.value: j for j in ARM_JOINTS}
    selected: set[Joint] = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name == Joint.GRIPPER.value:
            raise SystemExit(
                "The gripper has no zero to set — it self-calibrates against "
                "its hard stops when enabled."
            )
        if name not in by_value:
            valid = ", ".join(by_value)
            raise SystemExit(f"Unknown joint '{name}'. Valid joints: {valid}")
        selected.add(by_value[name])
    return [j for j in ARM_JOINTS if j in selected] or list(ARM_JOINTS)


async def _run_single(args: argparse.Namespace) -> None:
    channel = resolve_channel(args)
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


async def _calibrate_joint_either_stop(
    motor: Motor,
    joint: Joint,
    p_start: float,
    magnitude_rad: float,
    web_prompts: bool,
) -> bool:
    """Zero one joint at whichever of its two end stops the operator chose.

    For :data:`EITHER_STOP_JOINTS` (symmetric limits ±``magnitude_rad``):
    the operator moves from the in-range start reference ``p_start`` to
    either stop; the direction of travel identifies which one, and the
    runtime later re-detects the side from the encoder reading, so no side
    needs to be recorded. Returns ``True`` on success.
    """
    mag_deg = math.degrees(magnitude_rad)
    print(f"\n— {joint.name}  →  EITHER end stop (±{mag_deg:.1f}°) —")
    print(f"     start: {_fmt(p_start)}")

    while True:
        if not _prompt(
            f"{joint.name}: move to EITHER end stop (+{mag_deg:.1f}° or "
            f"-{mag_deg:.1f}° — whichever is better placed)",
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

        mag_diff = abs(abs(delta) - magnitude_rad)
        if mag_diff > _MAGNITUDE_WARN_RAD:
            print(
                f"    ⚠  moved {math.degrees(abs(delta)):.1f}°, expected"
                f" ~{mag_deg:.1f}° — make sure you're against the stop."
            )

        chosen_deg = math.copysign(mag_deg, delta)
        await motor.set_zero_position()
        print(
            f"    ✓ zeroed at the {chosen_deg:+.1f}° end stop at {_fmt(p_end)} "
            f"(side is auto-detected at runtime)."
        )
        return True


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
    channel = resolve_channel(args)
    side = "LEFT" if is_left else "RIGHT"
    joints = _parse_arm_joints(args.joints)
    joints_desc = (
        "all joints"
        if len(joints) == len(ARM_JOINTS)
        else ", ".join(j.value for j in joints)
    )
    print(f"\nset-zero-pos --guided — {side} arm  ({channel})  |  {joints_desc}")

    async with CanBus(channel) as bus:
        motors = {joint: Motor(bus, joint) for joint in joints}

        # Capture an in-range start reference for every selected joint up
        # front, so the operator positions the arm once rather than once per
        # joint.
        hold_what = (
            "EVERY joint somewhere inside its range"
            if len(joints) == len(ARM_JOINTS)
            else f"the selected joints ({joints_desc}) somewhere inside their range"
        )
        if not _prompt(
            f"hold {hold_what} (away from the end stops)",
            args.web_prompts,
        ):
            print("\naborted.")
            return
        starts = dict(
            zip(
                joints,
                await asyncio.gather(*[motors[j].get_position() for j in joints]),
            )
        )
        for joint in joints:
            print(f"  {joint.name:<12} start: {_fmt(starts[joint])}")

        results: list[tuple[Joint, bool]] = []
        any_damiao = False
        for joint in joints:
            target, sign = closer_end_stop(joint, is_left)
            motor = motors[joint]
            try:
                if joint in EITHER_STOP_JOINTS:
                    ok = await _calibrate_joint_either_stop(
                        motor, joint, starts[joint], abs(target), args.web_prompts
                    )
                else:
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
