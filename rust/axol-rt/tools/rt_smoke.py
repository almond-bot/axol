"""End-to-end smoke test for the hybrid rt path — no VR needed.

Drives the robot through ``RtAxol``: the full ``motion_control`` math runs in
Python at 120 Hz, the Rust core owns the bus at 240 Hz. The motion is a hold
followed by a gentle wrist_3 sinusoid on both arms.

    uv run python rust/axol-rt/tools/rt_smoke.py [--secs 10] [--amp-deg 8]

Reports the measured-vs-commanded tracking error observed through the
passively filled feedback caches.
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
        deadline = time.perf_counter()
        for k in range(n_hold + n_move):
            deadline += period
            if k < n_hold:
                offset = 0.0
            else:
                t = (k - n_hold) * period
                # Sine-squared envelope over the first second avoids a
                # velocity step at motion start.
                env = min(1.0, t) ** 2
                offset = env * amp * math.sin(2 * math.pi * freq_hz * t)
            ql, qr = base_l.copy(), base_r.copy()
            ql[WRIST_3] += offset
            qr[WRIST_3] += offset
            await robot.motion_control(left=ql, right=qr)

            meas_l, meas_r = await robot.get_positions()
            if meas_l is not None:
                worst_err = np.maximum(worst_err, np.abs(meas_l[:7] - ql[:7]))
            if meas_r is not None:
                worst_err = np.maximum(worst_err, np.abs(meas_r[:7] - qr[:7]))

            now = time.perf_counter()
            if deadline > now:
                await asyncio.sleep(deadline - now)

        print("worst measured-vs-commanded error per joint (deg):")
        joints = ["s1", "s2", "s3", "elbow", "w1", "w2", "w3"]
        for name, err in zip(joints, worst_err):
            print(f"  {name:<6} {math.degrees(err):6.2f}")
    finally:
        await robot.disable()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--amp-deg", type=float, default=8.0)
    ap.add_argument("--freq-hz", type=float, default=0.25)
    args = ap.parse_args()
    asyncio.run(main(args.secs, args.amp_deg, args.freq_hz))
