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

Run (right after rom.enable, while the motors are still enabled):
    uv run -m almond_axol.diagnostics.rom.disable
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


async def run(no_left: bool, no_right: bool) -> None:
    """Open each gripper sequentially, then disable all motors."""
    kwargs = {}
    if no_left:
        kwargs["left_channel"] = None
    if no_right:
        kwargs["right_channel"] = None

    axol = Axol(**kwargs)

    # The motors are already enabled and holding (rom.enable left them up); only
    # start the CAN reader loops so we can command them. Do NOT call enable().
    bus_tasks = []
    if axol.left is not None:
        bus_tasks.append(axol._left_bus.start())
    if axol.right is not None:
        bus_tasks.append(axol._right_bus.start())
    await asyncio.gather(*bus_tasks)

    try:
        targets: list[tuple[str, AxolArm]] = []
        if axol.right is not None:
            targets.append(("RIGHT", axol.right))
        if axol.left is not None:
            targets.append(("LEFT", axol.left))

        for side, arm in targets:
            await asyncio.to_thread(
                input, f"Press Enter to open the {side} gripper ..."
            )
            await open_gripper(arm, side)

        print("\nBoth grippers open — item released.")
    finally:
        print("Disabling motors ...")
        await axol.disable()
        print("Motors disabled.")


def main() -> None:
    """Parse CLI arguments and run the gripper release routine."""
    parser = argparse.ArgumentParser(
        description="Open the grippers from rom.enable and disable the robot."
    )
    parser.add_argument("--no-left", action="store_true", help="Skip the left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip the right arm.")
    args = parser.parse_args()

    if args.no_left and args.no_right:
        parser.error("Cannot disable both arms.")

    asyncio.run(run(no_left=args.no_left, no_right=args.no_right))


if __name__ == "__main__":
    main()
