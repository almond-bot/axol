"""Live terminal display of all motor positions at 100 Hz.

Run directly:
    python -m almond_axol.test.can
"""

import asyncio
import math

from ..robot.axol import Axol, arm_limits
from ..shared import Joint

_BAR_WIDTH = 24
_TAU = 2 * math.pi


def _bar(value: float, lo: float, hi: float) -> str:
    if math.isclose(lo, hi):
        return "─" * _BAR_WIDTH
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    pos = round(frac * _BAR_WIDTH)
    bar = list("░" * _BAR_WIDTH)
    bar[max(0, min(_BAR_WIDTH - 1, pos))] = "█"
    return "".join(bar)


async def _run() -> None:
    joints = list(Joint)

    async with Axol() as axol:
        await axol.start_telemetry(100)
        await asyncio.sleep(0.1)

        print("\033[?25l", end="")  # hide cursor
        try:
            while True:
                pos_l = axol.left.positions
                pos_r = axol.right.positions

                lines = []
                lines.append("\033[H\033[J")  # home + clear
                lines.append(
                    f"  {'Joint':<12}  {'Left':>8}  {'rev':<{_BAR_WIDTH}}  {'Right':>8}  {'rev':<{_BAR_WIDTH}}"
                )
                lines.append("  " + "─" * (12 + 8 + _BAR_WIDTH + 8 + _BAR_WIDTH + 6))

                for i, joint in enumerate(joints):
                    lo_l, hi_l = arm_limits(joint, is_left=True)
                    lo_r, hi_r = arm_limits(joint, is_left=False)
                    pl, pr = float(pos_l[i]), float(pos_r[i])
                    lines.append(
                        f"  {joint.value:<12}  {pl / _TAU:>+8.4f}  {_bar(pl, lo_l, hi_l)}  {pr / _TAU:>+8.4f}  {_bar(pr, lo_r, hi_r)}"
                    )

                lines.append("")
                lines.append("  ctrl+c to quit")
                print("\n".join(lines), end="", flush=True)

                await asyncio.sleep(1 / 100)

        except KeyboardInterrupt:
            pass
        finally:
            print("\033[?25h")  # restore cursor


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
