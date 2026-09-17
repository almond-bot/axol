"""
axol tune.breakaway

Measure a joint's **breakaway** (static) torque — the torque at which a
stationary joint first moves — and compare it to the sliding Coulomb torque
``fc`` the friction model already knows.

Why this exists, and why ``tune.friction`` cannot answer it:

* ``tune.friction`` sweeps at fixed velocities and fits
  ``fc·tanh(0.1·k·v) + fv·v``. That model form is monotonic in ``v`` and has
  **no static/kinetic distinction** — it cannot represent breakaway at any
  parameter value.
* Its sweep also bottoms out around 0.13 rad/s, and one least-significant bit
  of reported motor velocity is ~0.022 rad/s. Creep motion at a few
  thousandths of a rad/s is below the resolution of the velocity channel
  entirely, so a "slower sweep" fits torque against a signal that reads zero.

Stick-slip exists precisely *because* breakaway exceeds sliding friction. That
gap is what this measures, and it needs no velocity signal at all — only the
torque at the instant position moves.

Method, per direction and per pose:

1. Park the joint and hold the rest of the arm on the shared sweep-safety
   geometry (the same poses ``tune.friction`` uses).
2. Command the joint with ``kp = 0`` — no position spring — and a small ``kd``
   so it cannot run away once it releases. The only torque is gravity
   feedforward plus a slow, deliberately shaped ramp.
3. Ramp that extra torque up and back down in a triangle, and watch for the
   first motion past ``--move-lsb`` encoder counts.
4. Escalate the ramp's peak over successive trials and stop at the first
   release, so the joint never sees more torque than it took to move it.

What comes out, and what it is for:

* ``fc_break`` per direction, and ``fc_break/fc`` — which is what
  ``--axol.experiments.stiction_gain`` should be set to, instead of a guess.
* ``(fc_break − fc)/kp`` — the predicted stick-slip stair height. If it
  matches the steps seen in a slow replay, the diagnosis is closed.
* With ``--poses``, breakaway against gravity load at several points in the
  range: the slope is ``--axol.experiments.friction_load_gain``.

Examples:
    axol tune.breakaway --l --joint shoulder_1
    axol tune.breakaway --l --joint elbow --trials 5 --csv ~/breakaway-elbow.csv
    axol tune.breakaway --l --joint shoulder_1 --poses -20 0 20   # load slope
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
from pathlib import Path

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...motor import ControlMode, Motor
from ...motor.bus import CanBus
from ...robot.axol import arm_limits
from ...robot.config import AxolConfig
from ...robot.gravity import GravityCompensator
from ...tuning import joint_frame_motors, ramp_stages, sweep_safety
from ...tuning.joint_frame import JointFrameMotor
from ...utils.logquiet import quiet_noisy_loggers

_RATE_HZ = 240.0
_RAMP_SPEED = 0.25  # rad/s, for repositioning between trials
#: Encoder least-significant bit of the MIT feedback frame (rad). Motion has
#: to clear a few of these to count: one LSB is indistinguishable from the
#: position readout ticking over under a joint that has not actually moved.
_FEEDBACK_LSB = 2 * 12.566 / 65535
#: Peak ramp torques to try, as multiples of the joint's fitted ``fc``. The
#: search stops at the first release, so a joint that breaks away near ``fc``
#: never sees the larger ones.
_ESCALATION = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
_NO_FF = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.8)


async def _hold(motor: JointFrameMotor, kp: float, kd: float, q: float, secs: float):
    """Hold ``q`` with a real position spring (between probes)."""
    n = max(2, int(secs * _RATE_HZ))
    await motor.run_experiment(
        kp=kp,
        kd=kd,
        rate_hz=_RATE_HZ,
        samples=[(q, q, 0.0)] * n,
        differentiate=False,
        feedforward=_NO_FF,
    )


async def _ramp_to(motor: JointFrameMotor, kp: float, kd: float, q: float):
    start = await motor.get_position()
    secs = abs(q - start) / _RAMP_SPEED + 1.0
    n = max(2, math.ceil(secs * _RATE_HZ) + 1)
    samples = [(start + i / (n - 1) * (q - start), q, 0.0) for i in range(n)]
    await motor.run_experiment(
        kp=kp,
        kd=kd,
        rate_hz=_RATE_HZ,
        samples=samples,
        differentiate=False,
        feedforward=_NO_FF,
    )


def _triangle(n: int) -> np.ndarray:
    """Up-and-back ramp shape. Coming back down bounds the excursion: past
    breakaway the torque is already falling, so a release cannot run."""
    up = n // 2
    return np.concatenate([np.linspace(0.0, 1.0, up), np.linspace(1.0, 0.0, n - up)])


async def _probe(
    motor: JointFrameMotor,
    hold_q: float,
    gravity_nm: float,
    direction: float,
    peak_nm: float,
    ramp_s: float,
    kd: float,
    move_rad: float,
) -> tuple[float | None, list[dict], np.ndarray]:
    """One triangular torque ramp at ``kp = 0``.

    Returns ``(breakaway_nm, rows, ramp)`` — ``breakaway_nm`` is ``None`` if
    the joint never moved, i.e. this peak was not enough.
    """
    n = max(8, int(ramp_s * _RATE_HZ))
    ramp = direction * peak_nm * _triangle(n)
    # kp = 0: the wire carries damping and this torque only. The per-sample
    # "gravity" field is the core's raw torque injection point (see
    # experiment.rs), and every friction/inertia/damping term is zeroed in
    # `_NO_FF`, so what the joint feels is exactly gravity + ramp.
    samples = [(hold_q, hold_q, gravity_nm + float(r)) for r in ramp]
    rows = await motor.run_experiment(
        kp=0.0,
        kd=kd,
        rate_hz=_RATE_HZ,
        samples=samples,
        differentiate=False,
        feedforward=_NO_FF,
    )
    return detect_release(rows, ramp, move_rad), rows, ramp


def detect_release(rows: list[dict], ramp: np.ndarray, move_rad: float) -> float | None:
    """Injected torque at the first sample past ``move_rad`` of motion.

    ``None`` means the joint never released within this ramp. Dropped
    feedback frames arrive as NaN and are skipped rather than counted as
    motion; the start position is a median of the first samples so one bad
    frame cannot define the origin.
    """
    actual = np.array([r["actual"] for r in rows], dtype=float)
    good = np.isfinite(actual)
    if good.sum() < 4:
        return None
    start = float(np.median(actual[good][:8]))
    moved = np.flatnonzero(good & (np.abs(actual - start) > move_rad))
    if not len(moved):
        return None
    i = int(moved[0])
    return abs(float(ramp[min(i, len(ramp) - 1)]))


async def _measure(
    motor: JointFrameMotor,
    joint: Joint,
    is_left: bool,
    hold_q: float,
    gravity_nm: float,
    fc: float,
    kp_hold: float,
    kd: float,
    trials: int,
    ramp_s: float,
    max_mult: float,
    move_rad: float,
    writer: csv.writer | None,
) -> dict[str, list[float]]:
    """Breakaway in both directions at one pose, ``trials`` times each."""
    out: dict[str, list[float]] = {"+": [], "-": []}
    for direction, key in ((+1.0, "+"), (-1.0, "-")):
        for trial in range(trials):
            await _ramp_to(motor, kp_hold, kd, hold_q)
            await _hold(motor, kp_hold, kd, hold_q, 0.4)
            found = None
            for mult in _ESCALATION:
                if mult > max_mult:
                    break
                peak = mult * fc
                found, rows, ramp = await _probe(
                    motor, hold_q, gravity_nm, direction, peak, ramp_s, kd, move_rad
                )
                if writer is not None:
                    for r, tau in zip(rows, ramp):
                        writer.writerow(
                            [
                                joint.value,
                                "left" if is_left else "right",
                                f"{math.degrees(hold_q):.3f}",
                                key,
                                trial,
                                f"{mult:.2f}",
                                f"{r['t']:.5f}",
                                f"{r['actual']:.6f}",
                                f"{gravity_nm + tau:.4f}",
                                f"{r['torque']:.4f}",
                            ]
                        )
                if found is not None:
                    out[key].append(found)
                    print(
                        f"      {key} trial {trial + 1}: released at "
                        f"{found:.3f} Nm ({found / fc:.2f}x fc, ramp peak {mult:.2f}x)"
                    )
                    break
            if found is None:
                print(
                    f"      {key} trial {trial + 1}: no release up to "
                    f"{max_mult:.2f}x fc — raise --max-torque"
                )
    await _ramp_to(motor, kp_hold, kd, hold_q)
    return out


def _report(
    joint: Joint, kp: float, fc: float, by_pose: list[tuple[float, float, dict]]
) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {joint.value}: breakaway vs sliding friction (fc = {fc:.3f} Nm)\n")
    print(
        f"  {'pose':>8} {'gravity':>9} {'break +':>9} {'break -':>9} "
        f"{'mean':>8} {'/fc':>6} {'stair':>8}"
    )
    rows = []
    for q, g, res in by_pose:
        vals = res["+"] + res["-"]
        if not vals:
            continue
        mp = float(np.mean(res["+"])) if res["+"] else float("nan")
        mm = float(np.mean(res["-"])) if res["-"] else float("nan")
        mean = float(np.mean(vals))
        stair = math.degrees(max(mean - fc, 0.0) / kp)
        rows.append((abs(g), mean))
        print(
            f"  {math.degrees(q):8.1f} {g:9.3f} {mp:9.3f} {mm:9.3f} "
            f"{mean:8.3f} {mean / fc:6.2f} {stair:7.3f}°"
        )
    if not rows:
        print("  (no releases recorded)")
        return
    means = [m for _, m in rows]
    ratio = float(np.mean(means)) / fc
    print(f"\n  breakaway / fc = {ratio:.2f}")
    print(
        f"    -> suggested --axol.experiments.stiction_gain {min(ratio - 1.0, 0.9):.2f}"
    )
    stair = math.degrees(max(float(np.mean(means)) - fc, 0.0) / kp)
    print(
        f"    -> predicted stick-slip stair height {stair:.3f}° "
        f"(compare with the steps in a slow replay)"
    )
    print(
        f"    -> suggested --axol.experiments.stiction_err_deg {max(stair, 0.01):.3f}"
    )
    if len(rows) >= 2:
        g = np.array([x for x, _ in rows])
        m = np.array(means)
        if g.ptp() > 0.2:
            slope = float(np.polyfit(g, m, 1)[0])
            print(
                f"    -> breakaway rises {slope:.3f} Nm per Nm of gravity load"
                f"  =>  --axol.experiments.friction_load_gain {max(slope, 0.0):.3f}"
            )
        else:
            print(
                "    -> poses spanned too little gravity load to fit "
                "friction_load_gain; spread --poses further"
            )
    else:
        print("    -> pass --poses with 2+ angles to fit friction_load_gain")


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "tune.breakaway",
        help="Measure a joint's breakaway (static) torque vs its sliding fc.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help="Left arm")
    side.add_argument("--r", action="store_true", help="Right arm")
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        help="Joint to probe",
    )
    p.add_argument("--channel", help="CAN interface (default: from settings)")
    p.add_argument(
        "--trials", type=int, default=3, help="Releases per direction (default: 3)"
    )
    p.add_argument(
        "--poses",
        type=float,
        nargs="+",
        default=None,
        help="Joint angles (deg) to probe at. Two or more with different "
        "gravity load fits friction_load_gain. Default: the joint's "
        "sweep-safety hold pose only.",
    )
    p.add_argument(
        "--ramp-s",
        type=float,
        default=4.0,
        help="Seconds for one up-and-back torque ramp (default: 4).",
    )
    p.add_argument(
        "--max-torque",
        type=float,
        default=2.5,
        help="Largest ramp peak to try, as a multiple of fc (default: 2.5). "
        "The search stops at the first release.",
    )
    p.add_argument(
        "--move-lsb",
        type=float,
        default=3.0,
        help="Encoder LSBs of motion that count as a release (default: 3).",
    )
    p.add_argument(
        "--kd", type=float, default=1.0, help="Wire damping during the probe (0-5)."
    )
    p.add_argument("--csv", type=Path, default=None, help="Dump every ramp sample here")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    quiet_noisy_loggers(args.log_level)
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    from ..can.driver import resolve_channel

    is_left = bool(args.l)
    joint = Joint(args.joint)
    config = AxolConfig().resolved()
    arm_cfg = config.left if is_left else config.right
    jc = getattr(arm_cfg, joint.value)
    fc, kp_cfg = jc.friction.fc, jc.kp
    if fc <= 0:
        raise SystemExit(f"{joint.value} has fc = 0 — nothing to compare breakaway to")
    move_rad = args.move_lsb * _FEEDBACK_LSB
    gravity = GravityCompensator(config)

    print(f"\nAxol breakaway probe — {'left' if is_left else 'right'} {joint.value}")
    print(f"  fitted fc = {fc:.3f} Nm   kp = {kp_cfg:.0f}")
    print(
        f"  release threshold = {args.move_lsb:.0f} LSB = {math.degrees(move_rad):.3f}°"
    )
    print(
        f"  ramp peaks up to {args.max_torque:.2f}x fc over {args.ramp_s:.1f}s, "
        f"{args.trials} trials per direction"
    )

    writer = csv_file = None
    if args.csv is not None:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "joint",
                "side",
                "pose_deg",
                "dir",
                "trial",
                "peak_mult",
                "t",
                "actual_rad",
                "cmd_torque_nm",
                "meas_torque_nm",
            ]
        )

    channel = resolve_channel(args)
    async with CanBus(channel) as bus:
        raw = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw.values()])
        motors = await joint_frame_motors(raw, is_left)
        await asyncio.gather(
            *[
                m.set_control_mode(ControlMode.POSITION_VELOCITY)
                for m in motors.values()
            ]
        )
        try:
            other_targets, lo_default, hi_default, notes = sweep_safety(joint, is_left)
            for note in notes:
                print(f"  {note}")
            for stage in ramp_stages(other_targets):
                await asyncio.gather(
                    *[
                        _ramp_to(motors[j], getattr(arm_cfg, j.value).kp, args.kd, q)
                        for j, q in stage.items()
                        if j in motors
                    ]
                )
            await motors[joint].set_control_mode(ControlMode.IMPEDANCE)
            await asyncio.sleep(1.0)

            lo, hi = arm_limits(joint, is_left)
            lo = lo_default if lo_default is not None else lo
            hi = hi_default if hi_default is not None else hi
            if args.poses is None:
                poses = [float(np.clip(await motors[joint].get_position(), lo, hi))]
            else:
                poses = [float(np.clip(math.radians(p), lo, hi)) for p in args.poses]

            by_pose = []
            for q in poses:
                q_arm = np.zeros(len(ARM_JOINTS))
                for j, target in other_targets.items():
                    if j in ARM_JOINTS:
                        q_arm[ARM_JOINTS.index(j)] = target
                q_arm[ARM_JOINTS.index(joint)] = q
                g = float(
                    gravity.gravity_arm(q_arm, is_left=is_left)[ARM_JOINTS.index(joint)]
                )
                print(f"\n  pose {math.degrees(q):.1f}°  (gravity {g:+.3f} Nm)")
                res = await _measure(
                    motors[joint],
                    joint,
                    is_left,
                    q,
                    g,
                    fc,
                    kp_cfg,
                    args.kd,
                    args.trials,
                    args.ramp_s,
                    args.max_torque,
                    move_rad,
                    writer,
                )
                by_pose.append((q, g, res))
            _report(joint, kp_cfg, fc, by_pose)
            if args.csv is not None:
                print(f"\n  raw ramp samples -> {args.csv}")
        finally:
            if csv_file is not None:
                csv_file.close()
            await asyncio.gather(
                *[m.disable() for m in raw.values()], return_exceptions=True
            )
