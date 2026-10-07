"""Home one joint on 0xA4 the way the old calibration did: 0x77, 0x76, one target.

uv run python scripts/a4_s1_home.py left shoulder_1 -10        # to -10°, then rest
uv run python scripts/a4_s1_home.py left shoulder_1 -3.4 10    # wait 10 s first
uv run python scripts/a4_s1_home.py left shoulder_1 -10 0 hold # others held on
                                                               # their firmware loops
"""

import asyncio
import math
import sys

from almond_axol.cli.tune.friction import rest_target
from almond_axol.constants import ARM_JOINTS
from almond_axol.motor import CanBus, ControlMode, Joint, Motor
from almond_axol.tuning.joint_frame import joint_frame_motors


async def main(arm: str, joint: Joint, start_deg: float, wait_s: float, hold: bool):
    left = arm == "left"
    async with CanBus(f"can_alm_axol_{arm[0]}") as bus:
        joints = list(ARM_JOINTS) if hold else [joint]
        raw = {j: Motor(bus, j) for j in joints}
        await asyncio.gather(*[r.enable() for r in raw.values()])  # 0x77
        motors = await joint_frame_motors(raw, left)
        await asyncio.gather(
            *[
                mm.set_control_mode(ControlMode.POSITION_VELOCITY)
                for mm in motors.values()
            ]
        )  # 0x76 reset
        if hold:  # every other joint holds where it is, on its own firmware loop
            for j, mm in motors.items():
                if j != joint:
                    await mm.set_position_velocity(await mm.get_position(), 0.25)
        m = motors[joint]
        if wait_s:
            print(f"waiting {wait_s:.0f} s — nudge {joint.value} by hand now")
            await asyncio.sleep(wait_s)
        # Re-read after the reset: refreshes the ±360° boot-wrap correction.
        print(f"start {math.degrees(await m.get_position()):+7.2f}°")
        for target in (math.radians(start_deg), rest_target(joint, left)):
            await m.set_position_velocity(target, 0.25)
            for _ in range(100):  # 10 s
                print(f"{math.degrees(await m.get_position()):+7.2f}°")
                await asyncio.sleep(0.1)
        await asyncio.gather(*[r.disable() for r in raw.values()])


a = sys.argv[1:]
asyncio.run(
    main(
        a[0],
        Joint(a[1]),
        float(a[2]),
        float(a[3]) if len(a) > 3 else 0.0,
        len(a) > 4 and a[4] == "hold",
    )
)
