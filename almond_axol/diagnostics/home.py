"""Enable Axol and move all joints to zero.

Run via the CLI or directly:
    axol diag.home
    uv run -m almond_axol.diagnostics.home
    uv run -m almond_axol.diagnostics.home --no-right
    uv run -m almond_axol.diagnostics.home --no-left
"""

import argparse
import asyncio
import math
import time

import numpy as np

from ..robot.axol import Axol

_SPEED = 0.2 * 2 * math.pi  # rad/s
_RATE_HZ = 100.0


async def _run(no_left: bool, no_right: bool) -> None:
    kwargs = {}
    if no_left:
        kwargs["left_channel"] = None
    if no_right:
        kwargs["right_channel"] = None

    async with Axol(**kwargs) as axol:
        print("Enabling motors ...")
        await axol.enable()

        print("Reading current positions ...")
        pos_l, pos_r = await axol.get_positions()

        arms = []
        if axol.left is not None and pos_l is not None:
            arms.append(("left", axol.left, pos_l))
        if axol.right is not None and pos_r is not None:
            arms.append(("right", axol.right, pos_r))

        for side, arm, start_pos in arms:
            # start_pos is shape (8,) — last entry is gripper
            max_dist = float(np.max(np.abs(start_pos[:7])))
            duration = max(max_dist / _SPEED, 1.0)
            print(f"Moving {side} arm to zero over {duration:.1f}s ...")

            dt = 1.0 / _RATE_HZ
            t0 = time.monotonic()
            target = np.zeros(8, dtype=np.float32)
            target[7] = start_pos[7]  # leave gripper unchanged

            while True:
                t = time.monotonic() - t0
                alpha = min(t / duration, 1.0)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                q = (start_pos * (1.0 - smooth) + target * smooth).astype(np.float32)
                await arm.motion_control(q)
                if alpha >= 1.0:
                    break
                await asyncio.sleep(dt)

            print(f"  {side} arm at zero.")

        print("Done. Disabling motors ...")
        await axol.disable()


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-left", action="store_true", help="Skip left arm.")
    parser.add_argument("--no-right", action="store_true", help="Skip right arm.")


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``diag.home`` subcommand."""
    p = subparsers.add_parser(
        "diag.home",
        help="Enable Axol and bring all joints to zero.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the homing routine from parsed arguments."""
    if args.no_left and args.no_right:
        raise SystemExit("Cannot disable both arms.")
    asyncio.run(_run(no_left=args.no_left, no_right=args.no_right))


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the homing routine."""
    parser = argparse.ArgumentParser(
        description="Enable Axol and bring all joints to zero."
    )
    _add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
