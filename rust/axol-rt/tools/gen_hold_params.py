"""Generate per-joint hold parameters (gains + gravity t_ff) for `axol-rt hold`.

Read-only: opens the CAN buses, reads the current joint-frame pose, computes
gravity torques with the production MuJoCo model, and writes one line per arm
joint (whitespace-separated, `#` comments):

    <iface> <joint> <motor_id> <kp> <kd> <t_ff_nm>

This is the prototype of the hybrid split's data flow: Python owns the
calibration offsets and the gravity model, the Rust core consumes plain
joint-space numbers.

Usage: uv run python rust/axol-rt/tools/gen_hold_params.py [out_path]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from almond_axol.constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT
from almond_axol.robot.axol import Axol
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import GravityCompensator


async def main(out_path: Path) -> None:
    cfg = AxolConfig().resolved()
    gravity_comp = GravityCompensator(cfg)
    robot = Axol(cfg)
    await robot.connect()
    try:
        lines = ["# iface joint motor_id kp kd t_ff_nm"]
        for iface, arm, arm_cfg, is_left in (
            (CAN_LEFT, robot.left, cfg.left, True),
            (CAN_RIGHT, robot.right, cfg.right, False),
        ):
            assert arm is not None
            q = await arm.get_positions()  # (8,), joint frame
            gravity = gravity_comp.gravity_arm(q[: len(ARM_JOINTS)], is_left=is_left)
            for i, joint in enumerate(ARM_JOINTS):
                jc = getattr(arm_cfg, joint.value)
                lines.append(
                    f"{iface} {joint.value} {i + 1} "
                    f"{jc.kp:.3f} {jc.kd:.3f} {float(gravity[i]):.4f}"
                )
                print(
                    f"{iface} {joint.value:<10} q={float(q[i]):+.3f} rad  "
                    f"kp={jc.kp:.0f} kd={jc.kd:.2f} t_ff={float(gravity[i]):+.2f} Nm"
                )
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out_path}")
    finally:
        await robot.disconnect()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/hold_params.txt")
    asyncio.run(main(out))
