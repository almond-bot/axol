"""Move both arm joints from zero to target, hold 5 s, then return.
Grippers hold a firm position throughout the entire test.

Right arm — zero: 0.0115 rad, target: -1.5919 rad, gripper: 2.6186 rad
Left arm  — zero: -0.0115 rad, target: 1.5919 rad, gripper: 2.6965 rad (mirrored)

Run directly:
    python -m almond_axol.test.testarm
    python -m almond_axol.test.testarm --joint SHOULDER_2
"""

import argparse
import asyncio
import math
import time

import numpy as np

from ..robot.axol import Axol
from ..shared import Joint

_ZERO_POS = 0.0115  # rad — right arm zero (left = negated)
_TARGET_POS = -1.5919  # rad — right arm target (left = negated)
_SPEED = 0.08  # rad/s
_HOLD_SECS = 300  # seconds to hold at target before returning
_RATE_HZ = 100.0
_DEFAULT_JOINT = Joint.SHOULDER_1

# Gripper: raw encoder positions (rad) given by user
_R_GRIPPER_RAW = 2.6186
_L_GRIPPER_RAW = 2.6965

# Close 12.5° extra beyond given positions for a firm grip without over-stressing.
_GRIPPER_FIRM_RAD = 12.5 * math.pi / 180  # ≈ 0.218 rad

# Gripper limits from axol.py: (_GRIPPER_LO, 0.0).  Dividing a positive raw
# value by the negative _GRIPPER_LO flips the sign → negative normalized value.
# motion_control un-flips it on de-normalization, but np.clip then kills it at 0.0.
# Fix: widen _limits_hi[GRIPPER] at runtime (done in _run after enable).
_GRIPPER_LO = -0.8037 * 2 * math.pi  # ≈ -5.049 rad (from axol.py)
_GRIPPER_RAW_MAX = 3.5  # rad — upper clip override for positive-range grippers

# Both grippers close toward larger positive values (+delta = firmer grip).
_R_GRIPPER_NORM = (_R_GRIPPER_RAW + _GRIPPER_FIRM_RAD) / _GRIPPER_LO
_L_GRIPPER_NORM = (_L_GRIPPER_RAW + _GRIPPER_FIRM_RAD) / _GRIPPER_LO

_GRIPPER_IDX = list(Joint).index(Joint.GRIPPER)  # 7


async def _run(joint: Joint) -> None:
    joint_index = list(Joint).index(joint)

    dist = abs(_TARGET_POS - _ZERO_POS)
    duration = max(dist / _SPEED, 1.0)
    dt = 1.0 / _RATE_HZ

    async with Axol() as axol:
        print("Enabling motors (both arms) ...")
        await axol.enable()

        # Allow positive raw gripper positions (axol.py defaults to max=0.0).
        axol.right._limits_hi[_GRIPPER_IDX] = _GRIPPER_RAW_MAX
        axol.left._limits_hi[_GRIPPER_IDX] = _GRIPPER_RAW_MAX

        print("Reading current positions ...")
        cur_l, cur_r = await axol.get_positions()

        # Right arm — set moving joint and lock gripper
        r_start = cur_r.copy()
        r_start[joint_index] = _ZERO_POS
        r_start[_GRIPPER_IDX] = _R_GRIPPER_NORM

        r_target = cur_r.copy()
        r_target[joint_index] = _TARGET_POS
        r_target[_GRIPPER_IDX] = _R_GRIPPER_NORM

        # Left arm — mirrored signs for the moving joint, same firm gripper
        l_start = cur_l.copy()
        l_start[joint_index] = -_ZERO_POS
        l_start[_GRIPPER_IDX] = _L_GRIPPER_NORM

        l_target = cur_l.copy()
        l_target[joint_index] = -_TARGET_POS
        l_target[_GRIPPER_IDX] = _L_GRIPPER_NORM

        async def _sweep(arm, from_pos: np.ndarray, to_pos: np.ndarray) -> None:
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                alpha = min(t / duration, 1.0)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                q = (from_pos * (1.0 - smooth) + to_pos * smooth).astype(np.float32)
                await arm.motion_control(q)
                if alpha >= 1.0:
                    break
                await asyncio.sleep(dt)

        print(
            f"Moving both arms [{joint.value}] over {duration:.1f}s at {_SPEED} rad/s ...\n"
            f"  right: {_ZERO_POS:.4f} → {_TARGET_POS:.4f} rad  |  gripper: {_R_GRIPPER_RAW:.4f} rad (firm)\n"
            f"  left:  {-_ZERO_POS:.4f} → {-_TARGET_POS:.4f} rad  |  gripper: {_L_GRIPPER_RAW:.4f} rad (firm)"
        )
        await asyncio.gather(
            _sweep(axol.right, r_start, r_target),
            _sweep(axol.left, l_start, l_target),
        )

        print(f"Holding for {_HOLD_SECS:.0f}s ...")
        await asyncio.sleep(_HOLD_SECS)

        print("Returning both arms to zero ...")
        await asyncio.gather(
            _sweep(axol.right, r_target, r_start),
            _sweep(axol.left, l_target, l_start),
        )

        print("Both arms back at zero. Disabling motors ...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move both arm joints in sync while grippers hold firm."
    )
    parser.add_argument(
        "--joint",
        default=_DEFAULT_JOINT.name,
        choices=[j.name for j in Joint if j != Joint.GRIPPER],
        help=f"Joint to move on both arms (default: {_DEFAULT_JOINT.name}).",
    )
    args = parser.parse_args()
    asyncio.run(_run(joint=Joint[args.joint]))


if __name__ == "__main__":
    main()
