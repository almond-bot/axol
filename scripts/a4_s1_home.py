"""Drive shoulder_1 home on the motor's own 0xA4 position loop and log it.

A reproduction aid for the left shoulder_1 swing (2026-10-07): the old
calibration homing brought each MyActuator joint up exactly this way —
brake release (0x77), reset (0x76), then one 0xA4 target with a speed cap
(0.25 rad/s) — and polled until arrival. Only shoulder_1 is touched: no
other motor is enabled, so the rest of the arm stays on its brakes.

Optionally (``--start-deg``) it first moves to a start angle on 0xA4 so
there is a move to watch; then it homes. Position, speed and torque are
logged at ~50 Hz to a CSV. A trip — speed over ``--trip-dps`` or position
outside the moves' span by more than ``--window-deg`` — disables the motor
(0x80: torque off, brake engaged). Keep a hand on the e-stop anyway.

    uv run python scripts/a4_s1_home.py --arm left --start-deg -10 --csv ~/a4_s1.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path

from almond_axol.cli.tune.friction import rest_target
from almond_axol.constants import Joint
from almond_axol.motor import CanBus, ControlMode, Motor
from almond_axol.tuning.joint_frame import joint_frame_motors

_IFACE = {"left": "can_alm_axol_l", "right": "can_alm_axol_r"}


class Tripped(RuntimeError):
    pass


async def _move(m, target, speed, args, log, t0, label) -> None:
    start = await m.get_position()
    lo = min(start, target) - math.radians(args.window_deg)
    hi = max(start, target) + math.radians(args.window_deg)
    print(
        f"  {label}: {math.degrees(start):+.2f}° -> {math.degrees(target):+.2f}° "
        f"on 0xA4 at {math.degrees(speed):.1f}°/s cap"
    )
    await m.set_position_velocity(target, speed)
    timeout = abs(target - start) / speed + args.settle_s + 2.0
    t_end = time.monotonic() + timeout
    arrived_at = None
    peak = 0.0
    while time.monotonic() < t_end:
        pos = await m.get_position()
        vel = await m.motor.get_velocity()
        tq = await m.motor.get_torque()
        t = time.monotonic() - t0
        log.writerow(
            [
                f"{t:.4f}",
                label,
                f"{math.degrees(pos):.3f}",
                f"{math.degrees(vel):.2f}",
                f"{tq:.3f}",
            ]
        )
        peak = max(peak, abs(math.degrees(vel)))
        if abs(math.degrees(vel)) > args.trip_dps:
            raise Tripped(
                f"speed {math.degrees(vel):+.1f}°/s at {math.degrees(pos):+.2f}°"
            )
        if not lo <= pos <= hi:
            raise Tripped(f"position {math.degrees(pos):+.2f}° left the move's span")
        if abs(pos - target) < math.radians(0.5):
            arrived_at = arrived_at or time.monotonic()
            if time.monotonic() - arrived_at > args.settle_s:
                break
        else:
            arrived_at = None
        await asyncio.sleep(0.02)
    pos = await m.get_position()
    print(
        f"    ended at {math.degrees(pos):+.2f}° (err "
        f"{math.degrees(pos - target):+.2f}°), peak speed {peak:.1f}°/s"
    )


async def main(args: argparse.Namespace) -> None:
    is_left = args.arm == "left"
    home = rest_target(Joint.SHOULDER_1, is_left)
    speed = math.radians(args.speed_dps)
    out = open(Path(args.csv).expanduser(), "w", newline="")
    log = csv.writer(out)
    log.writerow(["t_s", "phase", "joint_deg", "speed_dps", "torque_nm"])
    async with CanBus(_IFACE[args.arm]) as bus:
        raw = Motor(bus, Joint.SHOULDER_1)
        await raw.enable()  # 0x77 brake release, as the old bring-up did
        m = (await joint_frame_motors({Joint.SHOULDER_1: raw}, is_left))[
            Joint.SHOULDER_1
        ]
        await m.set_control_mode(ControlMode.POSITION_VELOCITY)  # 0x76 reset
        pos = await m.get_position()
        print(
            f"{args.arm} shoulder_1 at {math.degrees(pos):+.2f}° (home {math.degrees(home):+.2f}°)"
        )
        targets = []
        if args.start_deg is not None:
            targets.append(("to-start", math.radians(args.start_deg)))
        targets.append(("home", home))
        far = max(abs(t - pos) for _, t in targets)
        if math.degrees(far) > args.max_move_deg:
            raise SystemExit(
                f"refusing: a {math.degrees(far):.1f}° move exceeds --max-move-deg "
                f"{args.max_move_deg}"
            )
        t0 = time.monotonic()
        try:
            for label, target in targets:
                await _move(m, target, speed, args, log, t0, label)
                out.flush()
        except (Tripped, KeyboardInterrupt, asyncio.CancelledError) as exc:
            print(
                f"  ! TRIP: {exc or 'interrupted'} — disabling (torque off, brake on)"
            )
            await raw.disable()
            raise
        finally:
            out.close()
            print(f"  log: {args.csv}")
        final = await m.get_position()
        if abs(final - home) < math.radians(2.0):
            await raw.disable()
            print("  at home: disabled (brake on)")
        else:
            print(
                f"  ! not at home ({math.degrees(final):+.2f}°): left holding on 0xA4"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument(
        "--start-deg",
        type=float,
        default=None,
        help="first move here on 0xA4 (joint frame, deg), then home",
    )
    p.add_argument(
        "--speed-dps",
        type=float,
        default=math.degrees(0.25),
        help="0xA4 speed cap (default: the old homing's 0.25 rad/s)",
    )
    p.add_argument("--max-move-deg", type=float, default=20.0)
    p.add_argument("--trip-dps", type=float, default=40.0)
    p.add_argument("--window-deg", type=float, default=5.0)
    p.add_argument(
        "--settle-s",
        type=float,
        default=3.0,
        help="seconds to watch after arriving (default 3)",
    )
    p.add_argument("--csv", default="~/a4_s1.csv")
    asyncio.run(main(p.parse_args()))
