"""Quantify motion smoothness of a startup-like Rust-core ramp.

Plays a min-jerk excursion on the wrists (out 3 s / hold 1 s / back 3 s —
the same speed class as teleop's return-to-rest) and logs commanded vs
measured positions at the 120 Hz command rate. Reports band-passed
(3-30 Hz) measured-velocity RMS per joint — the "felt jitter" band — and
saves the raw streams to an npz.

    uv run python rust/axol-rt/tools/rt_ramp_diag.py

SAFETY: this script commands open-loop joint excursions with no collision
checking (unlike teleop's reset trajectories, which are planned). Only
wrist joints move by default — a shoulder_2/elbow excursion from rest was
observed to contact the robot's base. Do not add proximal-joint motion
without physically verifying the swept envelope is clear.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time

import numpy as np

from almond_axol.robot import Axol

RATE_HZ = 120.0
WRIST_1, WRIST_3 = 4, 6
# Wrist-only: rotations that stay within the arm's own envelope. See the
# SAFETY note in the module docstring before touching this.
EXCURSION = {WRIST_1: math.radians(15.0), WRIST_3: math.radians(20.0)}


def min_jerk(alpha: float) -> float:
    a = min(1.0, max(0.0, alpha))
    return a**3 * (10 - 15 * a + 6 * a * a)


async def main(out: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from almond_axol.rt import RtAxol

    inner = Axol()
    robot = RtAxol(inner)
    await robot.enable()
    try:
        pos_l, pos_r = await robot.get_positions()
        assert pos_l is not None and pos_r is not None
        base_l, base_r = pos_l.astype(np.float64), pos_r.astype(np.float64)

        period = 1.0 / RATE_HZ
        phases = [3.0, 1.0, 3.0]  # out, hold, back
        n_total = int(sum(phases) * RATE_HZ) + int(2 * RATE_HZ)  # + lead-in hold
        t_log: list[float] = []
        cmd_log: list[np.ndarray] = []
        meas_log: list[np.ndarray] = []
        ts_log: list[np.ndarray] = []

        left_arm = inner.left
        assert left_arm is not None
        names = ", ".join(
            f"joint[{j}] {math.degrees(a):+.0f}°" for j, a in EXCURSION.items()
        )
        print(f"Rust core: min-jerk ramp {names}")

        deadline = time.perf_counter()
        t0 = deadline
        for k in range(n_total):
            deadline += period
            t = k * period - 2.0  # lead-in hold of 2 s
            if t < 0:
                blend = 0.0
            elif t < phases[0]:
                blend = min_jerk(t / phases[0])
            elif t < phases[0] + phases[1]:
                blend = 1.0
            else:
                blend = 1.0 - min_jerk((t - phases[0] - phases[1]) / phases[2])
            ql, qr = base_l.copy(), base_r.copy()
            for j, amp in EXCURSION.items():
                ql[j] += blend * amp
                qr[j] -= blend * amp  # mirror for the right arm
            await robot.motion_control(left=ql, right=qr)

            # Measured side straight from the passive caches, with the CAN
            # feedback timestamps so velocity is jitter-free to compute.
            meas = np.full(7, np.nan)
            ts = np.full(7, np.nan)
            from almond_axol.constants import ARM_JOINTS

            for i, j in enumerate(ARM_JOINTS):
                m = left_arm.motors[j]
                try:
                    meas[i] = m.position
                    ts[i] = m.feedback_ts
                except Exception:  # noqa: BLE001 - cache not primed yet
                    pass
            t_log.append(time.perf_counter() - t0)
            cmd_log.append(ql[:7].copy())
            meas_log.append(meas)
            ts_log.append(ts)

            now = time.perf_counter()
            if deadline > now:
                await asyncio.sleep(deadline - now)
    finally:
        await robot.disable()

    cmd = np.asarray(cmd_log)
    meas = np.asarray(meas_log)
    ts = np.asarray(ts_log)
    tt = np.asarray(t_log)
    np.savez(out, t=tt, cmd=cmd, meas=meas, ts=ts, mode="rust")

    # Jitter metric: velocity from measured positions against feedback
    # timestamps, band-passed 3-30 Hz (simple FFT mask), RMS per joint.
    print(f"\n== rust ==   (saved {out})")
    joints = ["s1", "s2", "s3", "elbow", "w1", "w2", "w3"]
    # Offsets differ between frames; compare motion only (offsets constant).
    for i, name in enumerate(joints):
        m = meas[:, i]
        k = ts[:, i]
        good = np.isfinite(m) & np.isfinite(k)
        m, k = m[good], k[good]
        keep = np.concatenate(([True], np.diff(k) > 1e-6))
        m, k = m[keep], k[keep]
        if len(m) < 100:
            print(f"  {name:<6} (no data)")
            continue
        # Resample to a uniform 120 Hz grid before the FFT.
        grid = np.arange(k[0], k[-1], 1.0 / RATE_HZ)
        mu = np.interp(grid, k, m)
        vel = np.gradient(mu, 1.0 / RATE_HZ)
        spec = np.fft.rfft(vel * np.hanning(len(vel)))
        freqs = np.fft.rfftfreq(len(vel), 1.0 / RATE_HZ)
        band = (freqs >= 3.0) & (freqs <= 30.0)
        spec[~band] = 0
        vband = np.fft.irfft(spec, n=len(vel))
        rms = math.degrees(float(np.sqrt(np.mean(vband**2))))
        print(f"  {name:<6} vel 3-30 Hz RMS {rms:7.3f} °/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/rt_ramp_diag.npz")
    args = ap.parse_args()
    asyncio.run(main(args.out))
