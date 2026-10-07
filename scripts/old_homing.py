"""The calibration homing as it was before the 0xA4 removal (d555009), alone.

Same calls as the old ``tune.factory`` bring-up, ``_home_all`` /
``_ramp_verified`` and teardown: every joint enabled and reset onto its
firmware position loop; homed distal to proximal one at a time (read, one
target at 0.25 rad/s, poll every 0.1 s until within 0.05 rad, one resend,
else abort); then re-home, reset to IMPEDANCE and disable. The one addition:
shoulder_1 is read (0x92, read-only) and printed on every poll.

    uv run python scripts/old_homing.py left
"""

import asyncio
import math
import sys
import time

from almond_axol.cli.tune.friction import rest_target
from almond_axol.constants import ARM_JOINTS
from almond_axol.motor import CanBus, ControlMode, Joint, Motor
from almond_axol.robot.axol import arm_limits
from almond_axol.tuning.joint_frame import joint_frame_motors

ORDER = [
    Joint(j)
    for j in (
        "wrist_3",
        "wrist_2",
        "wrist_1",
        "elbow",
        "shoulder_3",
        "shoulder_2",
        "shoulder_1",
    )
]


async def ramp_verified(m, target, s1, left):
    pos = await m.get_position()
    lo, hi = arm_limits(m.joint, left)
    if not lo - math.radians(15) <= pos <= hi + math.radians(15):
        raise RuntimeError(f"refusing: {m.joint.value} reads {math.degrees(pos):+.1f}°")
    for _attempt in range(2):
        await m.set_position_velocity(target, 0.25)
        pos = await m.get_position()
        t0, timeout = time.monotonic(), abs(pos - target) / 0.25 + 2.0
        while time.monotonic() - t0 < timeout:
            await asyncio.sleep(0.1)
            pos = await m.get_position()
            s1_pos = await s1.get_position()
            print(
                f"{m.joint.value:<11} {math.degrees(pos):+7.2f}°   "
                f"shoulder_1 {math.degrees(s1_pos):+7.2f}°"
            )
            if abs(pos - target) < 0.05:
                return
    raise RuntimeError(f"{m.joint.value} never reached {math.degrees(target):+.1f}°")


async def home_all(motors, left):
    for j in ORDER:
        await ramp_verified(
            motors[j], rest_target(j, left), motors[Joint.SHOULDER_1], left
        )


async def main(arm: str) -> None:
    left = arm == "left"
    async with CanBus(f"can_alm_axol_{arm[0]}") as bus:
        raw = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw.values()])
        motors = await joint_frame_motors(raw, left)
        await asyncio.gather(
            *[
                m.set_control_mode(ControlMode.POSITION_VELOCITY)
                for m in motors.values()
            ]
        )
        try:
            print("Homing all joints to rest (distal to proximal) ...")
            await home_all(motors, left)
        finally:
            print("Returning to rest and disabling ...")
            try:
                await home_all(motors, left)
            except Exception:
                pass
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])


asyncio.run(main(sys.argv[1]))
