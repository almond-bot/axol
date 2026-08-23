"""Diagnose vibration during a production-path static hold.

Enables the full robot through the production ``Axol`` stack (same as the
ROM test and teleop), holds the arms exactly where they are, streams
``motion_control`` at 100 Hz for a few seconds, and reports per-joint
torque chatter and position ripple at full rate — the enable-time
vibration the 5 Hz ROM capture can't resolve.

The robot holds its CURRENT pose: pose it however you want to test
(e.g. the parked pose where enable-time vibration was observed) before
running. On exit (or Ctrl-C) the motors are disabled and the arms go
limp — make sure the pose is safe to release.

    uv run python scripts/hold_chatter.py --stiffness 1.0
    uv run python scripts/hold_chatter.py --stiffness 1.0 --host-scale 0
    uv run python scripts/hold_chatter.py --stiffness 0.5
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


async def _force_disable(channel: str) -> None:
    """Disable every motor on the bus, clearing any stale mode/hold state
    left by a crashed session (enable() refuses to attach to a motor whose
    control mode doesn't match, e.g. a gripper stuck in IMPEDANCE)."""
    async with CanBus(channel) as bus:
        for j in Joint:
            try:
                await Motor(bus, j).disable()
            except MotorError:
                pass


def _diff_rms(xs: list[float]) -> float | None:
    d = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    return math.sqrt(sum(x * x for x in d) / len(d)) if len(d) > 10 else None


def _second_diff_rms(xs: list[float]) -> float | None:
    dd = [xs[i + 2] - 2 * xs[i + 1] + xs[i] for i in range(len(xs) - 2)]
    return math.sqrt(sum(x * x for x in dd) / len(dd)) if len(dd) > 10 else None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stiffness", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument(
        "--host-scale",
        type=float,
        default=1.0,
        help="Scale every joint's kd_host / kd_host_max by this factor "
        "(0 disables host damping entirely) to isolate its contribution.",
    )
    ap.add_argument(
        "--fw-kd-cap",
        type=float,
        default=None,
        help="Cap the firmware kd sent to every joint at this value (e.g. 5, "
        "the pre-retune stiff endpoint) to isolate firmware-kd torque noise.",
    )
    ap.add_argument(
        "--dump-csv",
        type=str,
        default=None,
        help="Write raw per-cycle samples (t + pos/tq per joint) to this path "
        "for offline spectral analysis.",
    )
    ap.add_argument(
        "--gripper",
        type=float,
        default=None,
        help="Hold the grippers at this normalized target (0=closed, 1=open) "
        "instead of their current position — 1.0 reproduces ROM's "
        "held-against-the-open-stop condition.",
    )
    ap.add_argument(
        "--gripper-torque",
        type=float,
        default=None,
        help="Raise the gripper torque cap (Nm) — ROM's grasp runs use 2.0 "
        "instead of the 0.5 default.",
    )
    ap.add_argument(
        "--gripper-cycle",
        action="store_true",
        help="Command the grippers closed (0.0) for the first 3 s, then open "
        "(1.0) for the rest — reproduces ROM's dynamic approach into the "
        "open stop instead of a static hold at it.",
    )
    ap.add_argument(
        "--home",
        action="store_true",
        help="Command exact joint-frame home (all zeros) like the ROM test, "
        "instead of holding the measured current pose. At home shoulder_2 "
        "and wrist_2 rest against the chassis, so any offset between the "
        "true rest and 0 becomes a constant contact force.",
    )
    args = ap.parse_args()

    cfg = AxolConfig(left_stiffness=args.stiffness, right_stiffness=args.stiffness)
    if args.gripper_torque is not None:
        cfg.left.gripper.torque_limit = args.gripper_torque
        cfg.right.gripper.torque_limit = args.gripper_torque
    axol = Axol(config=cfg)
    if args.host_scale != 1.0 or args.fw_kd_cap is not None:
        for arm in (axol.left, axol.right):
            for j in ARM_JOINTS:
                jc = getattr(arm._arm_config, j.value)
                jc.kd_host *= args.host_scale
                jc.kd_host_max *= args.host_scale
                if args.fw_kd_cap is not None:
                    jc.kd = min(jc.kd, args.fw_kd_cap)

    print("Clearing any stale motor state from previous sessions ...")
    await _force_disable(CAN_LEFT)
    await _force_disable(CAN_RIGHT)

    print(
        f"Enabling at stiffness={args.stiffness} host-scale={args.host_scale} "
        f"— arms will stiff-hold their CURRENT pose. Ctrl-C to abort."
    )
    await axol.enable()
    sh = axol.left._arm_config.shoulder_1
    print(
        f"  shoulder_1 effective: kp={sh.kp:.0f} kd={sh.kd:.1f} "
        f"kd_host={sh.kd_host:.1f} kd_host_max={sh.kd_host_max:.1f}"
    )

    q_left = await axol.left.get_positions()
    q_right = await axol.right.get_positions()
    gripper_i = len(ARM_JOINTS)
    if args.home:
        print("  commanding exact joint-frame home (ROM targets); rest offsets:")
        for side, q in (("left", q_left), ("right", q_right)):
            offs = "  ".join(
                f"{j.value}={math.degrees(q[i]):+5.2f}°"
                for i, j in enumerate(ARM_JOINTS)
            )
            print(f"    {side}: {offs}")
        for i in range(len(ARM_JOINTS)):
            q_left[i] = 0.0
            q_right[i] = 0.0
    if args.gripper is not None:
        q_left[gripper_i] = args.gripper
        q_right[gripper_i] = args.gripper
        print(f"  grippers commanded to {args.gripper:.2f} (normalized)")

    sampled_joints = list(ARM_JOINTS) + [Joint.GRIPPER]
    samples: dict[str, dict[str, list[float]]] = {
        f"{side}:{j.value}": {"tq": [], "pos": []}
        for side in ("left", "right")
        for j in sampled_joints
    }
    raw_rows: list[list[float]] = []

    print(f"  holding for {args.seconds:.0f} s at {RATE_HZ:.0f} Hz ...")
    dt = 1.0 / RATE_HZ
    start = time.monotonic()
    n = 0
    try:
        while time.monotonic() - start < args.seconds:
            loop_start = time.monotonic()
            if args.gripper_cycle:
                target = 0.0 if (time.monotonic() - start) < 3.0 else 1.0
                q_left[gripper_i] = target
                q_right[gripper_i] = target
            await axol.motion_control(left=q_left, right=q_right)
            row: list[float] = [time.monotonic()]
            for side, arm in (("left", axol.left), ("right", axol.right)):
                for j in sampled_joints:
                    m = arm.motors[j]
                    rec = samples[f"{side}:{j.value}"]
                    try:
                        rec["tq"].append(m.torque)
                        rec["pos"].append(m.position)
                        row.extend((m.position, m.torque))
                    except MotorError:
                        row.extend((float("nan"), float("nan")))
            raw_rows.append(row)
            n += 1
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)
    except KeyboardInterrupt:
        print("  interrupted")
    finally:
        elapsed = time.monotonic() - start
        print(f"\n  {n} cycles in {elapsed:.1f} s ({n / elapsed:.1f} Hz actual)")
        print(f"  {'joint':17s} {'tq chatter Nm':>13s} {'pos ripple mdeg':>15s}")
        for name, rec in samples.items():
            tq = _diff_rms(rec["tq"])
            rip = _second_diff_rms(rec["pos"])
            tq_s = f"{tq:.4f}" if tq is not None else "n/a"
            rip_s = f"{math.degrees(rip) * 1000:.2f}" if rip is not None else "n/a"
            print(f"  {name:17s} {tq_s:>13s} {rip_s:>15s}")
        if args.dump_csv:
            import csv

            with open(args.dump_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = ["t"]
                for side in ("left", "right"):
                    for j in sampled_joints:
                        header.extend((f"{side}:{j.value}:pos", f"{side}:{j.value}:tq"))
                w.writerow(header)
                w.writerows(raw_rows)
            print(f"  raw samples written to {args.dump_csv}")
        print("\n  disabling — arms go limp ...")
        await axol.disable()


if __name__ == "__main__":
    asyncio.run(main())
