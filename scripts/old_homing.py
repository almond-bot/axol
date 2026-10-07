"""The calibration homing as it was before the 0xA4 removal (d555009), alone.

Every joint enabled and reset onto its firmware position loop, then homed
distal to proximal one at a time: one target per joint (0.25 rad/s cap),
polled every 0.1 s until within 0.05 rad, one resend. Prints shoulder_1 on
every poll. Disables at the end.

    uv run python scripts/old_homing.py left
"""

import asyncio
import math
import sys
import time

from almond_axol.cli.tune.friction import rest_target
from almond_axol.constants import ARM_JOINTS
from almond_axol.motor import CanBus, ControlMode, Joint, Motor
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
        s1 = motors[Joint.SHOULDER_1]
        for j in ORDER:
            m, target = motors[j], rest_target(j, left)
            for _attempt in range(2):
                await m.set_position_velocity(target, 0.25)
                pos = await m.get_position()
                t0, timeout = time.monotonic(), abs(pos - target) / 0.25 + 2.0
                while time.monotonic() - t0 < timeout:
                    await asyncio.sleep(0.1)
                    pos = await m.get_position()
                    print(
                        f"{j.value:<11} {math.degrees(pos):+7.2f}°   "
                        f"shoulder_1 {math.degrees(await s1.get_position()):+7.2f}°"
                    )
                    if abs(pos - target) < 0.05:
                        break
                else:
                    continue
                break
        await asyncio.gather(*[m.disable() for m in raw.values()])


asyncio.run(main(sys.argv[1]))
