"""
rom.disable

Release the item gripped by ``rom.enable`` and power the robot down.

``rom.enable`` finishes (or is Ctrl-C'd) with the arms at home, the grippers
still clamped on the item, and every motor left enabled. This script attaches
to that already-enabled robot, opens each gripper one at a time (right first,
then left) so an operator can catch the item, and finally disables all motors.

It deliberately does NOT bring the realtime core up or call ``Axol.enable()``:
that would recalibrate the grippers (forcing them open and dropping the item)
and reset the arm motors. Instead it only opens the maintenance CAN proxies and
talks to the gripper motors directly in raw motor radians, leaving the arms
holding the last command the core left them with. The arms never move here.

Run (right after rom.enable, while the motors are still enabled):
    uv run -m almond_axol.diagnostics.rom.disable
    uv run -m almond_axol.diagnostics.rom.disable --no-left
"""

import argparse
import asyncio
import math
import sys
import time

from ...constants import CAN_LEFT, CAN_RIGHT, Joint
from ...motor import ControlMode
from ...robot.axol import GRIPPER_TRAVEL, Axol, AxolArm

# Marker prefix a --web-prompts step prints before blocking on stdin, matching
# rom.enable; the dashboard turns it into a Continue button.
PROMPT_MARKER = "[prompt]"

RATE_HZ = 100.0  # Hz
OPEN_SPEED = 0.2 * 2 * math.pi  # rad/s — gradual, so the operator can catch the item
OPEN_MAX_SPEED = 10.0  # rad/s — POSITION_FORCE velocity cap (smoothstep paces it)
OPEN_TORQUE = 2.0  # Nm — POSITION_FORCE output cap while opening


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


async def run(
    no_left: bool,
    no_right: bool,
    web_prompts: bool = False,
    left_channel: str = CAN_LEFT,
    right_channel: str = CAN_RIGHT,
) -> None:
    """Open each gripper sequentially, then disable every motor."""
    axol = Axol(
        left_channel=None if no_left else left_channel,
        right_channel=None if no_right else right_channel,
    )

    # The motors are already enabled and holding (rom.enable detached its
    # realtime core with them up); only open the maintenance proxies so we can
    # command them. Do NOT call enable().
    await axol.connect()

    try:
        targets: list[tuple[str, AxolArm]] = []
        if axol.right is not None:
            targets.append(("RIGHT", axol.right))
        if axol.left is not None:
            targets.append(("LEFT", axol.left))

        for side, arm in targets:
            instruction = f"Get ready to catch the item, then open the {side} gripper."
            if web_prompts:
                # The dashboard turns this marker into a Continue button and
                # writes a line to our stdin when the operator clicks.
                print(f"{PROMPT_MARKER} {instruction}", flush=True)
                await asyncio.to_thread(sys.stdin.readline)
            else:
                await asyncio.to_thread(
                    input, f"{instruction} Press Enter to continue ..."
                )
            await open_gripper(arm, side)

        print("\nGrippers open — item released.")
    finally:
        print("Disabling motors ...")
        await axol.disable()
        print("Motors disabled.")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    parser.add_argument(
        "--left-channel",
        default=CAN_LEFT,
        metavar="IFACE",
        help="SocketCAN interface for the left arm, for setups without the "
        "Axol hub CAN adapter (default: %(default)s).",
    )
    parser.add_argument(
        "--right-channel",
        default=CAN_RIGHT,
        metavar="IFACE",
        help="SocketCAN interface for the right arm, for setups without the "
        "Axol hub CAN adapter (default: %(default)s).",
    )
    parser.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit '[prompt] ...' markers and block on stdin for the "
        "gripper-open steps, so the web dashboard can drive them with a "
        "Continue button (set automatically by the dashboard).",
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
    asyncio.run(
        run(
            no_left=args.no_left,
            no_right=args.no_right,
            web_prompts=args.web_prompts,
            left_channel=args.left_channel,
            right_channel=args.right_channel,
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
