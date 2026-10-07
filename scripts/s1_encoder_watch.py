"""Read-only: poll one joint's multi-turn angle (0x92) and error flags (0x9A)
as fast as the bus allows and flag jumps. Sends no enable and no command —
move the arm by hand while it runs. Ctrl-C to stop.

    uv run python scripts/s1_encoder_watch.py left shoulder_1
"""

import asyncio
import math
import struct
import sys
import time

from almond_axol.motor import CanBus, Joint, Motor


async def main(arm: str, joint: Joint) -> None:
    async with CanBus(f"can_alm_axol_{arm[0]}") as bus:
        m = Motor(bus, joint)
        prev, n, t_print, t0 = None, 0, 0.0, time.monotonic()
        while True:
            deg = math.degrees(await m.get_position())
            flags = struct.unpack_from("<H", await m._driver._get_status1(), 6)[0]
            n += 1
            now = time.monotonic()
            if prev is not None and abs(deg - prev) > 2.0 or flags:
                print(
                    f"!! {now - t0:8.3f}s  {prev:+8.2f}° -> {deg:+8.2f}°  flags {flags:#06x}"
                )
            if now - t_print > 0.5:
                print(
                    f"{now - t0:8.3f}s  motor {deg:+8.2f}°  ({n / (now - t0):.0f} reads/s)"
                )
                t_print = now
            prev = deg


asyncio.run(main(sys.argv[1], Joint(sys.argv[2])))
