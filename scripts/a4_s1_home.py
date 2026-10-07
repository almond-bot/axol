"""Home shoulder_1 on 0xA4 the way the old calibration did: 0x77, 0x76, one target.

uv run python scripts/a4_s1_home.py left -10   # to -10°, then home
"""

import asyncio
import math
import sys

from almond_axol.motor import CanBus, ControlMode, Joint, Motor
from almond_axol.tuning.joint_frame import joint_frame_motors


async def main(arm: str, start_deg: float) -> None:
    async with CanBus(f"can_alm_axol_{arm[0]}") as bus:
        raw = Motor(bus, Joint.SHOULDER_1)
        await raw.enable()  # 0x77 brake release
        m = (await joint_frame_motors({Joint.SHOULDER_1: raw}, arm == "left"))[
            Joint.SHOULDER_1
        ]
        await m.set_control_mode(ControlMode.POSITION_VELOCITY)  # 0x76 reset
        for target in (math.radians(start_deg), 0.0):
            await m.set_position_velocity(target, 0.25)
            for _ in range(100):  # 10 s
                print(f"{math.degrees(await m.get_position()):+7.2f}°")
                await asyncio.sleep(0.1)
        await raw.disable()


asyncio.run(main(sys.argv[1], float(sys.argv[2])))
