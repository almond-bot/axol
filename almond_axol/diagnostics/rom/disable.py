"""
rom.disable

Release the item gripped by ``rom.enable`` and power the robot down.

``rom.enable`` finishes (or is Ctrl-C'd) with the arms at home, the grippers
still clamped on the item, and every motor left enabled. This script attaches
to that already-enabled robot, opens each gripper one at a time (right first,
then left) so an operator can catch the item, and finally disables all motors.

It deliberately does NOT call ``Axol.enable()``: that would recalibrate the
grippers (forcing them open and dropping the item) and reset the arm motors.
Instead it only starts the CAN reader loops and talks to the gripper motors
directly in raw motor radians, leaving the arms holding their last command.

If ``rom.enable`` was run on a subset of joints (via ``--joints``), pass the
same ``--joints`` here so only those motors are talked to and disabled. When
the gripper is not among them there is nothing to release, so this just powers
the selected motors down.

Run (right after rom.enable, while the motors are still enabled):
    uv run -m almond_axol.diagnostics.rom.disable
    uv run -m almond_axol.diagnostics.rom.disable --no-left
    uv run -m almond_axol.diagnostics.rom.disable --joints wrist_1,wrist_2,wrist_3
"""

import argparse
import asyncio
import math
import time

from ...constants import Joint
from ...motor import ControlMode
from ...robot.axol import GRIPPER_TRAVEL, Axol, AxolArm

RATE_HZ = 100.0  # Hz
OPEN_SPEED = 0.2 * 2 * math.pi  # rad/s — gradual, so the operator can catch the item
OPEN_MAX_SPEED = 10.0  # rad/s — POSITION_FORCE velocity cap (smoothstep paces it)
OPEN_TORQUE = 2.0  # Nm — POSITION_FORCE output cap while opening


def parse_joints(spec: str | None) -> set[Joint]:
    """Parse a comma-separated joint spec into a set of present :class:`Joint`.

    ``None`` or empty selects every joint. Names match the joint enum values
    (e.g. ``shoulder_1``, ``elbow``, ``gripper``).
    """
    if not spec:
        return set(Joint)
    by_value = {j.value: j for j in Joint}
    selected: set[Joint] = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in by_value:
            valid = ", ".join(by_value)
            raise SystemExit(f"Unknown joint '{name}'. Valid joints: {valid}")
        selected.add(by_value[name])
    return selected or set(Joint)


async def open_gripper(arm: AxolArm, side: str) -> None:
    """Open one gripper to its open hard-stop, gradually, in raw motor frame.

    The robot is assumed already enabled (by ``rom.enable``), so the gripper is
    still holding its grasp in POSITION_FORCE mode. We re-assert ``enable`` on
    just this motor — which only clears errors and reads the motor's
    position/torque limits so scaling is correct, sending no motion — then
    smoothstep the position-force setpoint from the current shaft position
    toward open. Opening is the negative direction, and a full ``GRIPPER_TRAVEL``
    guarantees the jaw reaches the open stop regardless of how far it had closed
    onto the item; the torque cap keeps it gentle against the stop.
    """
    motor = arm.motors[Joint.GRIPPER]
    await motor.enable()
    await motor.set_control_mode(ControlMode.POSITION_FORCE)

    start = await motor.get_position()  # raw motor rad
    target = start - GRIPPER_TRAVEL

    duration = max(GRIPPER_TRAVEL / OPEN_SPEED, 0.1)  # seconds
    dt = 1.0 / RATE_HZ  # seconds
    t0 = time.monotonic()
    print(f"Opening {side} gripper ...")
    while True:
        alpha = min((time.monotonic() - t0) / duration, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        pos = start + (target - start) * smooth
        await motor.set_position_force(pos, OPEN_MAX_SPEED, OPEN_TORQUE)
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)
    print(f"  {side} gripper open.")


async def _disable(axol: Axol, present: set[Joint]) -> None:
    """Disable the motors and close the buses, limited to the present joints.

    A full set delegates to ``Axol.disable`` (its usual all-motor shutdown);
    a subset only disables the motors that are actually on the bus so absent
    ones are not waited on, then closes the buses.
    """
    if present == set(Joint):
        await axol.disable()
        return
    tasks = []
    for arm in (axol.left, axol.right):
        if arm is not None:
            tasks.extend(arm.motors[j].disable() for j in present)
    try:
        await asyncio.gather(*tasks)
    except Exception:
        pass
    finally:
        close_tasks = []
        if axol.left is not None:
            close_tasks.append(axol._left_bus.close())
        if axol.right is not None:
            close_tasks.append(axol._right_bus.close())
        await asyncio.gather(*close_tasks)


async def run(
    no_left: bool, no_right: bool, present: set[Joint], no_prompt: bool = False
) -> None:
    """Open each present gripper sequentially, then disable the present motors."""
    kwargs = {}
    if no_left:
        kwargs["left_channel"] = None
    if no_right:
        kwargs["right_channel"] = None

    axol = Axol(**kwargs)
    has_gripper = Joint.GRIPPER in present

    # The motors are already enabled and holding (rom.enable left them up); only
    # start the CAN reader loops so we can command them. Do NOT call enable().
    bus_tasks = []
    if axol.left is not None:
        bus_tasks.append(axol._left_bus.start())
    if axol.right is not None:
        bus_tasks.append(axol._right_bus.start())
    await asyncio.gather(*bus_tasks)

    try:
        if has_gripper:
            targets: list[tuple[str, AxolArm]] = []
            if axol.right is not None:
                targets.append(("RIGHT", axol.right))
            if axol.left is not None:
                targets.append(("LEFT", axol.left))

            for side, arm in targets:
                message = f"Press Enter to open the {side} gripper ..."
                if no_prompt:
                    # No stdin under the web control panel: give the operator a
                    # moment to get ready to catch the item, then open.
                    print(f"{message} — opening in 10s (--no-prompt)")
                    await asyncio.sleep(10.0)
                else:
                    await asyncio.to_thread(input, message)
                await open_gripper(arm, side)

            print("\nGrippers open — item released.")
        else:
            print("\nNo gripper in this run — nothing to release.")
    finally:
        print("Disabling motors ...")
        await _disable(axol, present)
        print("Motors disabled.")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    valid_joints = [j.value for j in Joint]
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    parser.add_argument(
        "--joints",
        default=None,
        help="Comma-separated joints present on the bus (must match the rom.enable run). "
        f"Only these are talked to and disabled. Default: all. One of: {', '.join(valid_joints)}.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Replace the 'Press Enter' prompts with a 10s countdown "
        "(for headless / web-panel runs).",
    )


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``diag.rom-disable`` subcommand."""
    p = subparsers.add_parser(
        "diag.rom-disable",
        help="Open the grippers left clamped by the ROM test and power down.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> None:
    """Run the gripper release routine from parsed arguments."""
    if args.no_left and args.no_right:
        raise SystemExit("Cannot disable both arms.")
    present = parse_joints(args.joints)
    asyncio.run(
        run(
            no_left=args.no_left,
            no_right=args.no_right,
            present=present,
            no_prompt=args.no_prompt,
        )
    )


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the gripper release routine."""
    parser = argparse.ArgumentParser(
        description="Open the grippers from rom.enable and disable the robot."
    )
    _add_arguments(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
