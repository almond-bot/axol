"""Live terminal display of all motor positions.

Run directly:
    python -m almond_axol.test.can --l
    python -m almond_axol.test.can --r
    python -m almond_axol.test.can --l --hz 50
"""

import argparse
import asyncio
import math

from ..motor import CanBus
from ..robot.axol import ArmController, arm_limits
from ..robot.config import AxolConfig
from ..shared import CAN_LEFT, CAN_RIGHT, Joint

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


async def _run(is_left: bool, hz: int) -> None:
    joints = list(Joint)
    side = "left" if is_left else "right"
    channel = CAN_LEFT if is_left else CAN_RIGHT

    async with CanBus(channel) as bus:
        arm = ArmController(bus, AxolConfig(), is_left=is_left)
        await arm.start_telemetry(hz)
        await asyncio.sleep(0.1)

        print("\033[?25l", end="")  # hide cursor
        try:
            while True:
                positions = arm.positions

                lines = []
                lines.append("\033[H\033[J")  # home + clear
                lines.append(f"  {side.upper()} ARM")
                lines.append(f"  {'Joint':<12}  {'rev':>8}  {'':^{_BAR_WIDTH}}")
                lines.append("  " + "─" * (12 + 8 + _BAR_WIDTH + 4))

                for i, joint in enumerate(joints):
                    lo, hi = arm_limits(joint, is_left=is_left)
                    p = float(positions[i])
                    lines.append(
                        f"  {joint.value:<12}  {p / _TAU:>+8.4f}  {_bar(p, lo, hi)}"
                    )

                lines.append("")
                lines.append("  ctrl+c to quit")
                print("\n".join(lines), end="", flush=True)

                await asyncio.sleep(1 / hz)

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            print("\033[?25h")  # restore cursor


def main() -> None:
    parser = argparse.ArgumentParser(description="Live motor position display")
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help="Monitor left arm")
    side.add_argument("--r", action="store_true", help="Monitor right arm")
    parser.add_argument(
        "--hz", type=int, default=100, help="Telemetry rate in Hz (default: 100)"
    )
    args = parser.parse_args()

    asyncio.run(_run(is_left=args.l, hz=args.hz))


if __name__ == "__main__":
    main()
