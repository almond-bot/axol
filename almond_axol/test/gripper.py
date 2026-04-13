"""Open then close the gripper on each arm using MIT impedance control.

Run directly:
    python -m almond_axol.test.gripper
    python -m almond_axol.test.gripper --no-right
    python -m almond_axol.test.gripper --no-left
"""

import argparse
import asyncio
import math
import time

import numpy as np

from ..robot.axol import LIMITS, Axol
from ..shared import Joint

_SPEED = 0.2 * 2 * math.pi  # rad/s
_RATE_HZ = 100.0

_GRIPPER_IDX = list(Joint).index(Joint.GRIPPER)
_GRIPPER_RANGE = abs(LIMITS[Joint.GRIPPER][0] - LIMITS[Joint.GRIPPER][1])


async def _move_gripper(
    arm, q_hold: np.ndarray, start_norm: float, end_norm: float
) -> None:
    """Interpolate gripper from start_norm to end_norm at _SPEED."""
    dist_rad = abs(end_norm - start_norm) * _GRIPPER_RANGE
    duration = max(dist_rad / _SPEED, 0.1)
    dt = 1.0 / _RATE_HZ
    t0 = time.monotonic()
    q = q_hold.copy()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / duration, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        q[_GRIPPER_IDX] = start_norm + smooth * (end_norm - start_norm)
        await arm.motion_control(q)
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def _run(no_left: bool, no_right: bool) -> None:
    kwargs = {}
    if no_left:
        kwargs["left_channel"] = None
    if no_right:
        kwargs["right_channel"] = None

    async with Axol(**kwargs) as axol:
        arms = []
        if axol.left is not None and not no_left:
            pos = await axol.left.get_positions()
            arms.append(("left", axol.left, pos))
        if axol.right is not None and not no_right:
            pos = await axol.right.get_positions()
            arms.append(("right", axol.right, pos))

        for side, arm, start_pos in arms:
            # get_positions() returns gripper as [0, 1] normalized.
            q_hold = start_pos.copy()
            start_norm = float(np.clip(q_hold[_GRIPPER_IDX], 0.0, 1.0))
            q_hold[_GRIPPER_IDX] = start_norm

            print(f"Opening {side} gripper ...")
            await _move_gripper(arm, q_hold, start_norm, 1.0)

            print(f"Closing {side} gripper ...")
            await _move_gripper(arm, q_hold, 1.0, 0.0)

            print(f"  {side} gripper closed.")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open then close the gripper on each arm."
    )
    parser.add_argument("--no-left", action="store_true", help="Skip left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip right arm.")
    args = parser.parse_args()

    if args.no_left and args.no_right:
        parser.error("Cannot disable both arms.")

    asyncio.run(_run(no_left=args.no_left, no_right=args.no_right))


if __name__ == "__main__":
    main()
