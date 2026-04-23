"""Range of motion test (right arm only) — timed teach, then 1-hour loop.

Teach flow (motors OFF — backdrive freely):
  1. HOME      — 2 s to settle, then recorded
  2. FORWARD   — 5 s to reach it, then recorded
  3. HOME      — 5 s to return
  4. SIDEWAYS  — 5 s to reach it, then recorded
  5. HOME      — 5 s to return

Then: 5 s countdown → motors enable → loop for 1 hour.

Run:
    python -m almond_axol.test.rom_test
"""

import asyncio
import time

import numpy as np

from ..robot.axol import Axol
from ..shared import Joint

_SPEED = 0.05
_RATE_HZ = 100.0
_LOOP_SECS = 3600.0

_GRIPPER_IDX = list(Joint).index(Joint.GRIPPER)  # 7
_GRIPPER_RAW_MAX = 3.5  # allows positive raw gripper positions


async def _wait(msg: str, secs: int) -> None:
    print(f"\n{msg}")
    for i in range(secs, 0, -1):
        print(f"  {i}s ...", end="\r", flush=True)
        await asyncio.sleep(1.0)
    print()


def _show(label: str, pos: np.ndarray) -> None:
    print(f"  >>> {label} recorded")
    print(f"      right: {np.round(pos, 4).tolist()}")


async def _sweep(axol, from_r: np.ndarray, to_r: np.ndarray) -> None:
    dist = float(np.max(np.abs(to_r - from_r)))
    dur = max(dist / _SPEED, 1.0)
    dt = 1.0 / _RATE_HZ
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / dur, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        q = (from_r * (1.0 - smooth) + to_r * smooth).astype(np.float32)
        await axol.right.motion_control(q)
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def _run() -> None:
    async with Axol() as axol:
        # ── Teach phase (motors OFF) ──────────────────────────────────────
        print("\n=== TEACH PHASE — motors OFF, move the right arm freely ===")

        await _wait("Take the right arm to HOME position ...", 2)
        _, r_home = await axol.get_positions()
        _show("HOME", r_home)

        await _wait("Move the right arm FORWARD 90° ...", 5)
        _, r_fwd = await axol.get_positions()
        _show("FORWARD", r_fwd)
        await asyncio.sleep(1.0)

        await _wait("Move the right arm back to HOME ...", 5)

        await _wait("Move the right arm SIDEWAYS 90° ...", 5)
        _, r_side = await axol.get_positions()
        _show("SIDEWAYS", r_side)
        await asyncio.sleep(1.0)

        await _wait("Move the right arm back to HOME ...", 5)

        # ── Summary ───────────────────────────────────────────────────────
        print("\n=== ALL WAYPOINTS ===")
        for name, pos in [("Home", r_home), ("Forward", r_fwd), ("Sideways", r_side)]:
            print(f"  {name}: {np.round(pos, 4).tolist()}")

        print("\nPath: home → forward → home → sideways → home  (repeat 1 hour)")

        # ── Enable + execute ──────────────────────────────────────────────
        await _wait("Enabling motors in ...", 5)
        await axol.enable()
        axol.right._limits_hi[_GRIPPER_IDX] = _GRIPPER_RAW_MAX
        print("Motors ON. Starting 1-hour loop ...\n")

        segments = [
            (r_home, r_fwd),  # home → forward
            (r_fwd, r_home),  # forward → home
            (r_home, r_side),  # home → sideways
            (r_side, r_home),  # sideways → home
        ]

        deadline = time.monotonic() + _LOOP_SECS
        cycle = 0
        while time.monotonic() < deadline:
            cycle += 1
            print(
                f"Cycle {cycle}  —  {(deadline - time.monotonic()) / 60:.1f} min left"
            )
            for from_r, to_r in segments:
                if time.monotonic() >= deadline:
                    break
                await _sweep(axol, from_r, to_r)

        print("\n1 hour complete. Returning to home ...")
        _, cur_r = await axol.get_positions()
        await _sweep(axol, cur_r, r_home)
        await axol.disable()
        print("Done.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
