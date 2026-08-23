"""Diagnose vibration during a production-path mirrored shoulder sweep.

Reproduces the ROM test's shoulder_1 sweep (both arms mirrored, smoothstep,
1 rad/s) through the production ``Axol`` stack at the ROM stiffness, sampling
per-joint position/torque at ~100 Hz — the in-motion vibration the 5 Hz ROM
capture can't resolve. All non-swept joints hold their measured current pose
(not exact 0), so the base-contact buzz doesn't confound the measurement.

    uv run python scripts/sweep_chatter.py --stiffness 1.0
    uv run python scripts/sweep_chatter.py --stiffness 1.0 --left-only
    uv run python scripts/sweep_chatter.py --stiffness 0.5 --range-deg 60
"""

import argparse
import asyncio
import math
import time

from almond_axol.constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, Joint
from almond_axol.motor import CanBus, Motor, MotorError
from almond_axol.robot.axol import Axol
from almond_axol.robot.config import AxolConfig

RATE_HZ = 100.0
S1 = list(ARM_JOINTS).index(Joint.SHOULDER_1)
ELB = list(ARM_JOINTS).index(Joint.ELBOW)
W2 = list(ARM_JOINTS).index(Joint.WRIST_2)


async def _force_disable(channel: str) -> None:
    async with CanBus(channel) as bus:
        for j in Joint:
            try:
                await Motor(bus, j).disable()
            except MotorError:
                pass


def _smooth(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stiffness", type=float, default=1.0)
    ap.add_argument("--range-deg", type=float, default=90.0)
    ap.add_argument("--speed", type=float, default=1.0, help="rad/s (ROM uses 1.0)")
    ap.add_argument(
        "--left-only",
        action="store_true",
        help="Sweep only the left shoulder_1 (right arm holds) to separate "
        "dual-arm structural coupling from single-joint behaviour.",
    )
    ap.add_argument(
        "--joint",
        choices=("shoulder_1", "wrist_2"),
        default="shoulder_1",
        help="Which ROM sweep to reproduce. wrist_2 first eases the elbows "
        "to ±90° (ROM's wrist pre-pose) then sweeps both wrist_2 through "
        "±range like ROM does.",
    )
    ap.add_argument("--dump-csv", type=str, default=None)
    args = ap.parse_args()

    await _force_disable(CAN_LEFT)
    await _force_disable(CAN_RIGHT)

    cfg = AxolConfig(left_stiffness=args.stiffness, right_stiffness=args.stiffness)
    axol = Axol(config=cfg)
    await axol.enable()

    q_left = await axol.left.get_positions()
    q_right = await axol.right.get_positions()
    l0, r0 = float(q_left[S1]), float(q_right[S1])
    amp = math.radians(args.range_deg)
    duration = amp / args.speed
    print(
        f"mirrored shoulder_1 sweep: s={args.stiffness} range={args.range_deg}° "
        f"speed={args.speed} rad/s ({'left only' if args.left_only else 'both arms'})"
    )

    samples: dict[str, dict[str, list[float]]] = {
        f"{side}:{j.value}": {"tq": [], "pos": [], "t": []}
        for side in ("left", "right")
        for j in ARM_JOINTS
    }

    async def stream(apply_targets, seconds: float) -> None:
        """Stream at RATE_HZ for ``seconds``; ``apply_targets(smooth_progress)``
        mutates ``q_left`` / ``q_right`` in place each cycle."""
        dt = 1.0 / RATE_HZ
        start = time.monotonic()
        while True:
            el = time.monotonic() - start
            p = min(el / seconds, 1.0)
            apply_targets(_smooth(p))
            await axol.motion_control(left=q_left, right=q_right)
            now = time.monotonic()
            for side, arm in (("left", axol.left), ("right", axol.right)):
                for j in ARM_JOINTS:
                    m = arm.motors[j]
                    rec = samples[f"{side}:{j.value}"]
                    try:
                        rec["tq"].append(m.torque)
                        rec["pos"].append(m.position)
                        rec["t"].append(now)
                    except MotorError:
                        pass
            if p >= 1.0:
                return
            await asyncio.sleep(dt)

    try:
        if args.joint == "wrist_2":
            # ROM wrist pre-pose: elbows to ±90°, then sweep wrist_2 through
            # +range → -range → start (same values both arms, like ROM).
            el0, er0 = float(q_left[ELB]), float(q_right[ELB])
            elt, ert = math.pi / 2, -math.pi / 2
            wl0 = float(q_left[W2])

            def elbows(a0l: float, a1l: float, a0r: float, a1r: float):
                def fn(s: float) -> None:
                    q_left[ELB] = a0l + (a1l - a0l) * s
                    q_right[ELB] = a0r + (a1r - a0r) * s

                return fn

            def w2(a: float, b: float):
                def fn(s: float) -> None:
                    q_left[W2] = a + (b - a) * s
                    q_right[W2] = a + (b - a) * s

                return fn

            pre_dur = max(abs(elt - el0), abs(ert - er0)) / args.speed
            await stream(elbows(el0, elt, er0, ert), pre_dur)
            await stream(w2(wl0, amp), max(abs(amp - wl0) / args.speed, 0.1))
            await stream(w2(amp, amp), 2.0)
            await stream(w2(amp, -amp), 2 * amp / args.speed)
            await stream(w2(-amp, -amp), 2.0)
            await stream(w2(-amp, wl0), max(abs(amp + wl0) / args.speed, 0.1))
            await stream(elbows(elt, el0, ert, er0), pre_dur)
        else:
            # out, hold 2 s, back, hold 2 s — mirroring ROM's sweep + pause
            lo_t = l0 + amp
            ro_t = r0 if args.left_only else r0 - amp

            def s1(a0l: float, a1l: float, a0r: float, a1r: float):
                def fn(s: float) -> None:
                    q_left[S1] = a0l + (a1l - a0l) * s
                    q_right[S1] = a0r + (a1r - a0r) * s

                return fn

            await stream(s1(l0, lo_t, r0, ro_t), duration)
            await stream(s1(lo_t, lo_t, ro_t, ro_t), 2.0)
            await stream(s1(lo_t, l0, ro_t, r0), duration)
            await stream(s1(l0, l0, r0, r0), 2.0)
    finally:

        def c2c(xs: list[float]) -> float | None:
            d = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            return math.sqrt(sum(x * x for x in d) / len(d)) if len(d) > 10 else None

        print(f"\n  {'joint':17s} {'tq chatter Nm':>13s}")
        for name, rec in samples.items():
            tq = c2c(rec["tq"])
            print(
                f"  {name:17s} {tq:13.4f}" if tq is not None else f"  {name:17s}  n/a"
            )
        if args.dump_csv:
            import csv

            with open(args.dump_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["joint", "t", "pos", "tq"])
                for name, rec in samples.items():
                    for t, p_, q_ in zip(rec["t"], rec["pos"], rec["tq"]):
                        w.writerow([name, f"{t:.4f}", f"{p_:.6f}", f"{q_:.4f}"])
            print(f"  raw samples written to {args.dump_csv}")
        print("\n  disabling — arms go limp ...")
        await axol.disable()


if __name__ == "__main__":
    asyncio.run(main())
