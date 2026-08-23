"""Replay one full ROM cycle at 100 Hz and score every joint per phase.

Reproduces ROM's exact sweep sequence (shoulder_1 mirrored, shoulder_2 full
range, shoulder_3 with the arms-forward pre-pose, elbow, then the three
wrists with the elbows at 90°) through the production ``Axol`` stack —
same waypoints, speeds, pauses, and bus load — while sampling every joint's
position/torque at ~100 Hz. ROM's own 5 Hz telemetry aliases the 3–35 Hz
structural vibrations; this capture resolves them.

For each phase it reports any joint whose detrended torque residual exceeds
the floor, its dominant frequency, and position ripple.

    uv run python scripts/rom_cycle_chatter.py --stiffness 1.0
    uv run python scripts/rom_cycle_chatter.py --phases wrists --grip
"""

import argparse
import asyncio
import math
import time

import numpy as np

from almond_axol.constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT, Joint
from almond_axol.motor import CanBus, Motor, MotorError
from almond_axol.robot.axol import (
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_1_LEFT_LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
    Axol,
)
from almond_axol.robot.config import AxolConfig

RATE_HZ = 100.0
IDX = {j: i for i, j in enumerate(ARM_JOINTS)}
GRIP_I = list(Joint).index(Joint.GRIPPER)
GRIP_TORQUE = 2.0  # Nm — ROM's raised grasp torque cap
SPEED = 0.6  # rad/s — ROM's AXOL_SPEED
PRE_POSE_SPEED = 0.3  # rad/s
PAUSE = 1.0  # s — ROM's waypoint pause
ELBOW_ANGLE = math.pi / 2
SHOULDER_PRE_POSE = math.radians(25)


async def _force_disable(channel: str) -> None:
    async with CanBus(channel) as bus:
        for j in Joint:
            try:
                await Motor(bus, j).disable()
            except MotorError:
                pass


def _smooth(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _resid_stats(t: list[float], x: list[float]) -> tuple[float, float, float] | None:
    """(residual RMS, dominant freq >1 Hz, 15–48 Hz band fraction)."""
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
    hi = spec[(fr >= 15) & (fr <= 48)].sum() / max(spec[m].sum(), 1e-12)
    return float(resid.std()), float(peak), float(hi)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stiffness", type=float, default=1.0)
    ap.add_argument(
        "--phases",
        default="all",
        help="Comma-separated subset of: shoulder_1,shoulder_2,shoulder_3,"
        "elbow,wrists. Default: all.",
    )
    ap.add_argument(
        "--grip",
        action="store_true",
        help="Close the grippers on an item (prompted) at ROM's 2 Nm grasp "
        "torque before sweeping, to measure the loaded modes.",
    )
    ap.add_argument("--dump-csv", type=str, default=None)
    args = ap.parse_args()
    wanted = (
        {"shoulder_1", "shoulder_2", "shoulder_3", "elbow", "wrists"}
        if args.phases == "all"
        else set(args.phases.split(","))
    )

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
    print(f"ROM cycle replay: s={args.stiffness} phases={sorted(wanted)}")

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
        print(f"  phase: {name}")

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

    def hold_fn(s: float) -> None:
        pass

    async def step(
        joint: Joint, left_val: float, right_val: float, spd: float = SPEED
    ) -> None:
        """Mirror ROM's ``step``: smoothstep both arms to the joint targets,
        then hold for the waypoint pause."""
        i = IDX[joint]
        a_l, a_r = float(q_left[i]), float(q_right[i])

        def fn(s: float) -> None:
            q_left[i] = a_l + (left_val - a_l) * s
            q_right[i] = a_r + (right_val - a_r) * s

        dur = max(abs(left_val - a_l), abs(right_val - a_r)) / spd
        await stream(fn, max(dur, 0.05))
        await stream(hold_fn, PAUSE)

    async def hold_and_prompt(msg: str) -> None:
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
            print("  grippers closed at 2 Nm grasp torque")

        if "shoulder_1" in wanted:
            s1_low, s1_high = SHOULDER_1_LEFT_LIMITS
            set_phase("shoulder_1")
            for value in (s1_high, s1_low, 0.0):
                await step(Joint.SHOULDER_1, value, -value)

        if "shoulder_2" in wanted:
            _, s2_right_high = SHOULDER_2_RIGHT_LIMITS
            s2_left_low, _ = SHOULDER_2_LEFT_LIMITS
            set_phase("shoulder_2")
            await step(Joint.SHOULDER_2, s2_left_low, s2_right_high)
            await step(Joint.SHOULDER_2, 0.0, 0.0)

        if "shoulder_3" in wanted:
            set_phase("shoulder_3")
            await step(
                Joint.SHOULDER_1, SHOULDER_PRE_POSE, -SHOULDER_PRE_POSE, PRE_POSE_SPEED
            )
            low, high = LIMITS[Joint.SHOULDER_3]
            # ROM runs shoulder3_mirror=False: same values on both arms.
            await step(Joint.SHOULDER_3, -low, -low)
            await step(Joint.SHOULDER_3, -high, -high)
            await step(Joint.SHOULDER_3, 0.0, 0.0)
            await step(Joint.SHOULDER_1, 0.0, 0.0, PRE_POSE_SPEED)

        if "elbow" in wanted:
            _, e_left_high = ELBOW_LEFT_LIMITS
            e_right_low, _ = ELBOW_RIGHT_LIMITS
            set_phase("elbow")
            await step(Joint.ELBOW, e_left_high, e_right_low)
            await step(Joint.ELBOW, 0.0, 0.0)

        if "wrists" in wanted:
            set_phase("elbow-prepose")
            await step(Joint.ELBOW, ELBOW_ANGLE, -ELBOW_ANGLE)
            for wrist in (Joint.WRIST_1, Joint.WRIST_2, Joint.WRIST_3):
                low, high = LIMITS[wrist]
                set_phase(wrist.value)
                for value in (high, low, 0.0):
                    await step(wrist, value, value)
            set_phase("elbow-return")
            await step(Joint.ELBOW, 0.0, 0.0)

        if args.grip:
            set_phase("release")
            await hold_and_prompt("Get ready to catch the items.")
            await stream(grippers(0.0, 1.0), 1.5)
    finally:
        print(
            f"\n  {'phase':14s} {'joint':18s} {'resid Nm':>9s} {'peak Hz':>8s} {'15-48Hz':>8s}"
        )
        for ph, joints in samples.items():
            for name, (t_l, p_l, q_l) in joints.items():
                st = _resid_stats(t_l, q_l)
                if st is None:
                    continue
                rms, peak, hi = st
                if rms > 0.15 or name.split(":")[1] == ph:
                    print(
                        f"  {ph:14s} {name:18s} {rms:9.3f} {peak:8.1f} {100 * hi:7.0f}%"
                    )
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
