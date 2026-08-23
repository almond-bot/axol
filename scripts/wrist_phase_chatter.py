"""Reproduce ROM's full wrist phase at 100 Hz and score each joint per phase.

ROM sweeps wrist_1, wrist_2, wrist_3 sequentially through their full limits
(same waypoint values on both arms) with the elbows pre-posed to ±90°. The
5 Hz ROM telemetry can't resolve in-motion vibration, so this streams the
same trajectory through the production ``Axol`` stack (all joints commanded
every cycle, both buses — same bus load as ROM) while sampling per-joint
position/torque at ~100 Hz.

    uv run python scripts/wrist_phase_chatter.py --stiffness 1.0
    uv run python scripts/wrist_phase_chatter.py --stiffness 1.0 --speed 0.6

With ``--grip`` the script prompts you to place an item in each gripper and
closes on it at ROM's raised 2 Nm grasp torque before sweeping — measuring
the loaded wrist modes that an empty-handed run can't see. It prompts again
to release before disabling.
"""

import argparse
import asyncio
import math
import time

import numpy as np

from almond_axol.constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, Joint
from almond_axol.motor import CanBus, Motor, MotorError
from almond_axol.robot.axol import Axol
from almond_axol.robot.config import AxolConfig

RATE_HZ = 100.0
IDX = {j: i for i, j in enumerate(ARM_JOINTS)}
GRIP_I = list(Joint).index(Joint.GRIPPER)
GRIP_TORQUE = 2.0  # Nm — ROM's raised grasp torque cap
# ROM's LIMITS for the wrists (joint frame, rad)
WRIST_WAYPOINTS = {
    Joint.WRIST_1: (2.356, -2.356),
    Joint.WRIST_2: (1.571, -1.571),
    Joint.WRIST_3: (1.571, -1.571),
}
ELBOW_ANGLE = math.pi / 2


async def _force_disable(channel: str) -> None:
    async with CanBus(channel) as bus:
        for j in Joint:
            try:
                await Motor(bus, j).disable()
            except MotorError:
                pass


def _smooth(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _resid_stats(t: list[float], x: list[float]) -> tuple[float, float] | None:
    """(residual RMS, dominant frequency >1 Hz) after removing a 0.4 s trend."""
    ta = np.array(t)
    xa = np.array(x)
    if len(xa) < 80:
        return None
    ta -= ta[0]
    fs = 1.0 / np.median(np.diff(ta))
    k = max(int(0.4 * fs) | 1, 3)
    pad = np.pad(xa, k // 2, mode="edge")
    trend = np.convolve(pad, np.ones(k) / k, mode="valid")
    resid = xa - trend[: len(xa)]
    w = np.hanning(len(resid))
    spec = np.abs(np.fft.rfft(resid * w)) ** 2
    fr = np.fft.rfftfreq(len(resid), 1 / fs)
    m = fr > 1.0
    peak = fr[m][int(np.argmax(spec[m]))] if m.any() else float("nan")
    return float(resid.std()), float(peak)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stiffness", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=0.6, help="rad/s (ROM's AXOL_SPEED)")
    ap.add_argument(
        "--grip",
        action="store_true",
        help="Close the grippers on an item (prompted) at ROM's 2 Nm grasp "
        "torque before sweeping, to measure the loaded wrist modes.",
    )
    ap.add_argument("--dump-csv", type=str, default=None)
    args = ap.parse_args()

    await _force_disable(CAN_LEFT)
    await _force_disable(CAN_RIGHT)
    await asyncio.sleep(1.0)

    cfg = AxolConfig(left_stiffness=args.stiffness, right_stiffness=args.stiffness)
    if args.grip:
        cfg.left.gripper.torque_limit = GRIP_TORQUE
        cfg.right.gripper.torque_limit = GRIP_TORQUE
    axol = Axol(config=cfg)
    await axol.enable()
    await asyncio.sleep(0.5)

    q_left = await axol.left.get_positions()
    q_right = await axol.right.get_positions()
    print(f"wrist phase repro: s={args.stiffness} speed={args.speed} rad/s")

    # samples[phase][side:joint] = (t[], pos[], tq[])
    samples: dict[str, dict[str, tuple[list, list, list]]] = {}
    phase = "init"

    def set_phase(name: str) -> None:
        nonlocal phase
        phase = name
        samples[name] = {
            f"{side}:{j.value}": ([], [], [])
            for side in ("left", "right")
            for j in ARM_JOINTS
        }

    async def stream(apply_targets, seconds: float) -> None:
        dt = 1.0 / RATE_HZ
        start = time.monotonic()
        while True:
            p = min((time.monotonic() - start) / seconds, 1.0)
            apply_targets(_smooth(p))
            await axol.motion_control(left=q_left, right=q_right)
            now = time.monotonic()
            for side, arm in (("left", axol.left), ("right", axol.right)):
                for j in ARM_JOINTS:
                    m = arm.motors[j]
                    t_l, p_l, q_l = samples[phase][f"{side}:{j.value}"]
                    try:
                        q_l.append(m.torque)
                        p_l.append(m.position)
                        t_l.append(now)
                    except MotorError:
                        pass
            if p >= 1.0:
                return
            await asyncio.sleep(dt)

    def move(joint: Joint, a_l: float, b_l: float, a_r: float, b_r: float):
        i = IDX[joint]

        def fn(s: float) -> None:
            q_left[i] = a_l + (b_l - a_l) * s
            q_right[i] = a_r + (b_r - a_r) * s

        return fn

    async def hold_and_prompt(msg: str) -> None:
        """Block on Enter while streaming the hold (keeps host damping live)."""
        done = asyncio.Event()

        async def hold() -> None:
            while not done.is_set():
                await axol.motion_control(left=q_left, right=q_right)
                await asyncio.sleep(1.0 / RATE_HZ)

        task = asyncio.create_task(hold())
        await asyncio.to_thread(input, f"{msg} Press Enter to continue ... ")
        done.set()
        await task

    def grippers(a: float, b: float):
        def fn(s: float) -> None:
            q_left[GRIP_I] = a + (b - a) * s
            q_right[GRIP_I] = a + (b - a) * s

        return fn

    try:
        if args.grip:
            set_phase("grip")
            await stream(grippers(float(q_left[GRIP_I]), 1.0), 1.5)
            await hold_and_prompt("Place an item in EACH gripper.")
            await stream(grippers(1.0, 0.0), 1.5)
            print("grippers closed at 2 Nm grasp torque")

        el0, er0 = float(q_left[IDX[Joint.ELBOW]]), float(q_right[IDX[Joint.ELBOW]])
        set_phase("elbow-prepose")
        await stream(
            move(Joint.ELBOW, el0, ELBOW_ANGLE, er0, -ELBOW_ANGLE),
            max(abs(ELBOW_ANGLE - el0), abs(-ELBOW_ANGLE - er0)) / 0.3,
        )

        for wrist, (hi, lo) in WRIST_WAYPOINTS.items():
            i = IDX[wrist]
            w0_l, w0_r = float(q_left[i]), float(q_right[i])
            # ROM: current → high → (pause) → low → (pause) → 0
            legs = [
                (w0_l, hi, w0_r, hi),
                (hi, hi, hi, hi),
                (hi, lo, hi, lo),
                (lo, lo, lo, lo),
                (lo, 0.0, lo, 0.0),
            ]
            set_phase(wrist.value)
            for a_l, b_l, a_r, b_r in legs:
                if a_l == b_l and a_r == b_r:
                    await stream(move(wrist, a_l, b_l, a_r, b_r), 1.0)
                else:
                    dur = max(abs(b_l - a_l), abs(b_r - a_r)) / args.speed
                    await stream(move(wrist, a_l, b_l, a_r, b_r), max(dur, 0.1))

        set_phase("elbow-return")
        await stream(
            move(Joint.ELBOW, ELBOW_ANGLE, el0, -ELBOW_ANGLE, er0),
            max(abs(ELBOW_ANGLE - el0), abs(-ELBOW_ANGLE - er0)) / 0.3,
        )

        if args.grip:
            set_phase("release")
            await hold_and_prompt("Get ready to catch the items.")
            await stream(grippers(0.0, 1.0), 1.5)
    finally:
        print(f"\n  {'phase':14s} {'joint':16s} {'resid Nm':>9s} {'peak Hz':>8s}")
        for ph, joints in samples.items():
            for name, (t_l, p_l, q_l) in joints.items():
                st = _resid_stats(t_l, q_l)
                if st is None:
                    continue
                rms, peak = st
                if rms > 0.15 or name.split(":")[1] == ph:
                    print(f"  {ph:14s} {name:16s} {rms:9.3f} {peak:8.1f}")
        if args.dump_csv:
            import csv

            with open(args.dump_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["phase", "joint", "t", "pos", "tq"])
                for ph, joints in samples.items():
                    for name, (t_l, p_l, q_l) in joints.items():
                        for t, p_, q_ in zip(t_l, p_l, q_l):
                            w.writerow([ph, name, f"{t:.4f}", f"{p_:.6f}", f"{q_:.4f}"])
            print(f"  raw samples written to {args.dump_csv}")
        print("\n  disabling — arms go limp ...")
        await axol.disable()


if __name__ == "__main__":
    asyncio.run(main())
