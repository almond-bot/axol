"""End-to-end smoke test for the hybrid rt path — no VR needed.

Drives the robot through ``RtAxol``: the full ``motion_control`` math runs in
Python at 120 Hz, the Rust core owns the bus at 240 Hz. The motion is a hold
followed by a gentle wrist_3 sinusoid on both arms with a slow gripper
open/close cycle riding along (empty-jaw safe: it sweeps at most 0.35 of
the normalized span toward closed from wherever the jaw starts).

    uv run python rust/axol-rt/tools/rt_smoke.py [--secs 10] [--amp-deg 8]

Reports the measured-vs-commanded tracking error observed through the
passively filled feedback caches, plus the gripper's measured travel.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time

import numpy as np

from almond_axol.robot import Axol
from almond_axol.rt import RtAxol

WRIST_3 = 6  # Joint enum index
GRIPPER = 7
RATE_HZ = 120.0


async def main(secs: float, amp_deg: float, freq_hz: float) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    robot = RtAxol(Axol())
    await robot.enable()
    try:
        pos_l, pos_r = await robot.get_positions()
        assert pos_l is not None and pos_r is not None
        base_l, base_r = pos_l.astype(np.float64), pos_r.astype(np.float64)
        print(f"holding pose; wrist_3 sinusoid ±{amp_deg}° at {freq_hz} Hz for {secs}s")

        amp = math.radians(amp_deg)
        period = 1.0 / RATE_HZ
        n_hold = int(2.0 * RATE_HZ)
        n_move = int(secs * RATE_HZ)
        worst_err = np.zeros(7)
        grip_meas: list[tuple[float, float]] = []
        deadline = time.perf_counter()
        for k in range(n_hold + n_move):
            deadline += period
            if k < n_hold:
                offset = 0.0
                grip_off = 0.0
            else:
                t = (k - n_hold) * period
                # Sine-squared envelope over the first second avoids a
                # velocity step at motion start.
                env = min(1.0, t) ** 2
                offset = env * amp * math.sin(2 * math.pi * freq_hz * t)
                # Raised-cosine gripper cycle: sweeps toward closed and back,
                # starting and ending at the measured jaw position.
                grip_off = -0.35 * env * (1 - math.cos(2 * math.pi * freq_hz * t)) / 2
            ql, qr = base_l.copy(), base_r.copy()
            ql[WRIST_3] += offset
            qr[WRIST_3] += offset
            ql[GRIPPER] = min(1.0, max(0.0, ql[GRIPPER] + grip_off))
            qr[GRIPPER] = min(1.0, max(0.0, qr[GRIPPER] + grip_off))
            await robot.motion_control(left=ql, right=qr)

            meas_l, meas_r = await robot.get_positions()
            if meas_l is not None:
                worst_err = np.maximum(worst_err, np.abs(meas_l[:7] - ql[:7]))
            if meas_r is not None:
                worst_err = np.maximum(worst_err, np.abs(meas_r[:7] - qr[:7]))
            if meas_l is not None and meas_r is not None:
                grip_meas.append((float(meas_l[GRIPPER]), float(meas_r[GRIPPER])))

            now = time.perf_counter()
            if deadline > now:
                await asyncio.sleep(deadline - now)

        print("worst measured-vs-commanded error per joint (deg):")
        joints = ["s1", "s2", "s3", "elbow", "w1", "w2", "w3"]
        for name, err in zip(joints, worst_err):
            print(f"  {name:<6} {math.degrees(err):6.2f}")
        if grip_meas:
            arr = np.asarray(grip_meas)
            print(
                f"gripper measured travel (norm): "
                f"left {arr[:, 0].min():.2f}..{arr[:, 0].max():.2f}  "
                f"right {arr[:, 1].min():.2f}..{arr[:, 1].max():.2f}"
            )
    finally:
        await robot.disable()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--amp-deg", type=float, default=8.0)
    ap.add_argument("--freq-hz", type=float, default=0.25)
    args = ap.parse_args()
    asyncio.run(main(args.secs, args.amp_deg, args.freq_hz))
