"""Home one joint on 0xA4 the way the old calibration did: 0x77, 0x76, one target.

uv run python scripts/a4_s1_home.py left shoulder_1 -10   # to -10°, then rest
uv run python scripts/a4_s1_home.py left elbow 10
"""

import asyncio
import math
import sys

from almond_axol.cli.tune.friction import rest_target
from almond_axol.motor import CanBus, ControlMode, Joint, Motor
from almond_axol.tuning.joint_frame import joint_frame_motors


async def main(arm: str, joint: Joint, start_deg: float) -> None:
    async with CanBus(f"can_alm_axol_{arm[0]}") as bus:
        raw = Motor(bus, joint)
        await raw.enable()  # 0x77 brake release
        m = (await joint_frame_motors({joint: raw}, arm == "left"))[joint]
        await m.set_control_mode(ControlMode.POSITION_VELOCITY)  # 0x76 reset
        # Re-read after the reset: refreshes the ±360° boot-wrap correction.
        print(f"start {math.degrees(await m.get_position()):+7.2f}°")
        for target in (math.radians(start_deg), rest_target(joint, arm == "left")):
            await m.set_position_velocity(target, 0.25)
            for _ in range(100):  # 10 s
                print(f"{math.degrees(await m.get_position()):+7.2f}°")
                await asyncio.sleep(0.1)
        await raw.disable()


asyncio.run(main(sys.argv[1], Joint(sys.argv[2]), float(sys.argv[3])))
