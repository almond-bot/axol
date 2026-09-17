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

Method, per pose:

1. Park every other joint at rest and ramp the test joint to the pose.
2. Command it with ``kp = 0`` — no position spring — and a small ``kd`` so it
   cannot run away once it releases. Gravity feedforward is then the only
   thing holding it.
3. Trim that feedforward until the joint actually stands still. A joint sits
   still for any trim inside its stiction band, so this converges quickly.
   Without it, loaded poses are unmeasurable: at 15 Nm a 2 % gravity-model
   error outweighs the whole breakaway torque.

   The search stops at the first trim that holds, which is a band *edge*,
   not its centre — so the trim on its own is not the gravity residual.
   Adding the half-difference below recovers the centre, and that sum is.
4. Ramp an extra torque up and back down in a triangle and watch for the
   first motion past ``--move-lsb`` encoder counts. Peaks escalate over
   trials and the search stops at the first release, so the joint never sees
   more torque than it took to move it.
5. Repeat in both directions.

What comes out, and what it is for. Probing both directions separates two
things a single average hides — the symmetric half is friction, the
antisymmetric half is whatever standing torque the feedforward missed:

* **breakaway** and ``breakaway/fc``. Above ``fc``, the excess is what
  ``--axol.experiments.stiction_gain`` should be. At or below ``fc``, the
  velocity feedforward is over-compensating at rest and raising
  ``stiction_gain`` or ``friction_k_max`` pushes the wrong way.
* ``(breakaway − fc)/kp`` — the predicted stick-slip stair height. If it
  matches the steps seen in a slow replay, the diagnosis is closed.
* With ``--poses``, breakaway against gravity load: the slope is
  ``--axol.experiments.friction_load_gain``, and the per-pose trim is the
  gravity model's own error curve.

Examples:
    axol tune.breakaway --l --joint shoulder_1
    axol tune.breakaway --l --joint elbow --trials 5 --csv ~/breakaway-elbow.csv
    axol tune.breakaway --l --joint shoulder_1 --poses -20 0 20   # load slope
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
from pathlib import Path

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...motor import ControlMode, Motor
from ...motor.bus import CanBus
from ...robot.axol import arm_limits
from ...robot.config import AxolConfig
from ...robot.gravity import GravityCompensator
from ...tuning import (
    joint_frame_motors,
    ramp_joints_to,
    ramp_others_to_zero,
    sweep_safety,
)
from ...tuning.joint_frame import JointFrameMotor
from ...utils.logquiet import quiet_noisy_loggers
from ..motor import add_side_and_channel_arguments, resolve_channel

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
#: Homing order. Distal first: straightening the wrists and then the elbow
#: means each shoulder later swings a folded arm, so commanding the rest pose
#: is safe from any starting pose.
_HOME_ORDER: tuple[Joint, ...] = (
    Joint.WRIST_3,
    Joint.WRIST_2,
    Joint.WRIST_1,
    Joint.ELBOW,
    Joint.SHOULDER_3,
    Joint.SHOULDER_2,
    Joint.SHOULDER_1,
)


async def _home_all(
    motors: dict[Joint, JointFrameMotor], exclude: Joint | None = None
) -> None:
    """Ramp every joint to rest, one at a time, in :data:`_HOME_ORDER`."""
    for j in _HOME_ORDER:
        if j == exclude or j not in motors:
            continue
        await ramp_joints_to(motors, {j: 0.0})


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


async def _drift_check(
    motor: JointFrameMotor,
    hold_q: float,
    gravity_nm: float,
    secs: float,
    kd: float,
) -> float:
    """Hold at ``kp = 0`` with gravity feedforward only; return drift (rad).

    This is the probe's own precondition: the ramp measures the *extra*
    torque needed to move the joint, which only means something if the joint
    is standing still to begin with. It is also a direct read of the
    gravity-model residual at this pose — at high load a small model error is
    worth more torque than breakaway itself, and then the probe is measuring
    the model, not the friction.
    """
    n = max(8, int(secs * _RATE_HZ))
    rows = await motor.run_experiment(
        kp=0.0,
        kd=kd,
        rate_hz=_RATE_HZ,
        samples=[(hold_q, hold_q, gravity_nm)] * n,
        differentiate=False,
        feedforward=_NO_FF,
    )
    actual = np.array([r["actual"] for r in rows], dtype=float)
    good = actual[np.isfinite(actual)]
    if len(good) < 4:
        return 0.0
    return float(np.max(np.abs(good - np.median(good[:8]))))


async def _null_bias(
    motor: JointFrameMotor,
    hold_q: float,
    gravity_nm: float,
    kd: float,
    move_rad: float,
    fc: float,
    kp_hold: float,
    max_steps: int = 8,
) -> tuple[float, float, bool]:
    """Find a trim torque that holds the joint still at ``kp = 0``.

    The ramp measures the *extra* torque needed to move a stationary joint,
    so the joint has to be stationary first. Under load it often is not:
    gravity feedforward is the only thing holding it, and a small model error
    there is worth more torque than breakaway itself — which is why every
    loaded pose was skipped.

    A joint sits still for any trim within its stiction band, so the band is
    what this searches for: step the trim against the observed drift until
    the motion stops. The trim that results is a direct measurement of the
    gravity-model residual at this pose, and the bidirectional ramp that
    follows still cancels whatever bias remains.

    Returns ``(trim_nm, drift_rad, ok)``.
    """
    trim = 0.0
    step = 0.15 * fc
    drift = await _drift_check(motor, hold_q, gravity_nm, 0.6, kd)
    for _ in range(max_steps):
        if abs(drift) <= move_rad:
            return trim, drift, True
        # Which way did it fall? Re-park, then push back by one step.
        await _ramp_to(motor, kp_hold, kd, hold_q)
        signed = await _signed_drift(motor, hold_q, gravity_nm + trim, 0.6, kd)
        trim -= math.copysign(step, signed)
        if abs(trim) > 3.0 * fc:
            return trim, drift, False
        await _ramp_to(motor, kp_hold, kd, hold_q)
        drift = await _drift_check(motor, hold_q, gravity_nm + trim, 0.6, kd)
    return trim, drift, abs(drift) <= move_rad


async def _signed_drift(
    motor: JointFrameMotor,
    hold_q: float,
    torque_nm: float,
    secs: float,
    kd: float,
) -> float:
    """Drift with its sign — which way the joint falls at ``kp = 0``."""
    n = max(8, int(secs * _RATE_HZ))
    rows = await motor.run_experiment(
        kp=0.0,
        kd=kd,
        rate_hz=_RATE_HZ,
        samples=[(hold_q, hold_q, torque_nm)] * n,
        differentiate=False,
        feedforward=_NO_FF,
    )
    actual = np.array([r["actual"] for r in rows], dtype=float)
    good = actual[np.isfinite(actual)]
    if len(good) < 4:
        return 0.0
    return float(np.median(good[-8:]) - np.median(good[:8]))


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
    await _ramp_to(motor, kp_hold, kd, hold_q)
    trim, drift, ok = await _null_bias(
        motor, hold_q, gravity_nm, kd, move_rad, fc, kp_hold
    )
    out["drift_rad"] = [drift]  # type: ignore[index]
    out["trim_nm"] = [trim]  # type: ignore[index]
    if not ok:
        print(
            f"      ! joint will not hold still at kp=0 even with {trim:+.3f} Nm "
            f"of trim (drift {math.degrees(drift):.3f}° > "
            f"{math.degrees(move_rad):.3f}°) — skipping this pose."
        )
        if writer is not None:
            writer.writerow(
                [
                    joint.value,
                    "left" if is_left else "right",
                    f"{math.degrees(hold_q):.3f}",
                    "skipped",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"{trim:.4f}",
                    f"{math.degrees(drift):.4f}",
                ]
            )
        await _ramp_to(motor, kp_hold, kd, hold_q)
        return out
    if abs(trim) > 1e-9:
        print(
            f"      gravity-model residual at this pose: {trim:+.3f} Nm "
            f"(trimmed out; drift now {math.degrees(drift):.4f}°)"
        )
    else:
        print(f"      drift at kp=0: {math.degrees(drift):.4f}° (no trim needed)")
    gravity_nm = gravity_nm + trim
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
                                f"{trim:.4f}",
                                f"{math.degrees(drift):.4f}",
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
    """Split each pose into its symmetric and antisymmetric parts.

    Probing both directions separates two different things that a single
    average hides, the same way ``tune.friction`` separates gravity from
    friction with its bidirectional sweep:

    * **breakaway** = ``(|release+| + |release-|) / 2`` — symmetric, the
      static friction we came to measure. Writing the stiction band as
      ``[c - b, c + b]`` in trim space and the trim we started from as
      ``t``, the two ramps must cover ``(c + b) - t`` and ``t - (c - b)``,
      whose mean is ``b`` *whatever* ``t`` was. So this number does not
      care where inside the band the trim search happened to stop.
    * **offset** = ``(|release+| - |release-|) / 2`` = ``c - t`` — how far
      the band centre sits from where we started. Added to the trim it
      gives ``c``, the standing torque the gravity feedforward is not
      cancelling at this pose (``residual`` in the table). That is a
      gravity-model error, not friction.
    """
    print(f"\n{'-' * 70}")
    print(f"  {joint.value}: breakaway vs sliding friction (fc = {fc:.3f} Nm)\n")
    print(
        f"  {'pose':>8} {'gravity':>9} {'break+':>8} {'break-':>8} "
        f"{'BREAK':>8} {'/fc':>6} {'trim':>8} {'residual':>9}"
    )
    rows = []
    for q, g, res in by_pose:
        if not (res.get("+") or res.get("-")):
            trim = res.get("trim_nm", [float("nan")])[0]
            print(
                f"  {math.degrees(q):8.1f} {g:9.3f} {'—':>8} {'—':>8} "
                f"{'skipped':>8} {'':>6} {trim:8.3f} {'':>9}"
            )
            continue
        mp = float(np.mean(res["+"])) if res["+"] else float("nan")
        mm = float(np.mean(res["-"])) if res["-"] else float("nan")
        if res["+"] and res["-"]:
            brk, bias = (mp + mm) / 2.0, (mp - mm) / 2.0
        else:
            brk, bias = (mp if res["+"] else mm), float("nan")
        # `bias` here is the offset from wherever the trim search stopped,
        # not an absolute torque; trim + offset is the standing torque the
        # gravity feedforward failed to cancel at this pose.
        trim = res.get("trim_nm", [float("nan")])[0]
        rows.append((abs(g), brk))
        print(
            f"  {math.degrees(q):8.1f} {g:9.3f} {mp:8.3f} {mm:8.3f} "
            f"{brk:8.3f} {brk / fc:6.2f} {trim:8.3f} {trim + bias:9.3f}"
        )
    if not rows:
        print("\n  No usable releases. If every pose was skipped, the gravity")
        print("  model is the thing to fix before friction can be measured here.")
        return

    means = [m for _, m in rows]
    ratio = float(np.mean(means)) / fc
    print(f"\n  breakaway / fc = {ratio:.2f}")
    if ratio >= 1.05:
        # The textbook case: static exceeds sliding, and that gap is the
        # stick-slip stair.
        stair = math.degrees((float(np.mean(means)) - fc) / kp)
        print(
            f"    -> static exceeds sliding by {ratio - 1:.2f}x fc — stick-slip is expected"
        )
        print(f"    -> --axol.experiments.stiction_gain {min(ratio - 1.0, 0.9):.2f}")
        print(
            f"    -> predicted stair height {stair:.3f}° "
            f"(compare with the steps in a slow replay)"
        )
        print(f"    -> --axol.experiments.stiction_err_deg {max(stair, 0.01):.3f}")
    else:
        # Breakaway at or below the fitted fc. Adding stiction compensation
        # here pushes harder against friction that is already over-modelled.
        over = fc - float(np.mean(means))
        print(f"    -> breakaway is at or BELOW the fitted fc, by {over:.3f} Nm.")
        print("       The velocity feedforward is over-compensating at rest: it")
        print(f"       commands {fc:.3f} Nm of Coulomb torque where the joint")
        print(f"       releases at {float(np.mean(means)):.3f} Nm.")
        print("    -> do NOT raise stiction_gain or friction_k_max here; both")
        print("       deliver *more* of an already-too-large fc at low speed.")
        print("    -> the lever is fc itself: try scaling it toward the measured")
        print(
            f"       breakaway, e.g. --axol.<side>.{joint.value}.friction.fc "
            f"{float(np.mean(means)):.3f}"
        )
    if len(rows) >= 2:
        g = np.array([x for x, _ in rows])
        m = np.array(means)
        if float(np.ptp(g)) > 0.2:
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
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        help="Joint to probe",
    )
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
    # `quiet_noisy_loggers` caps the chatty third-party packages; the root
    # level comes from basicConfig, as in the other tune commands.
    logging.basicConfig(level=getattr(logging, args.log_level))
    quiet_noisy_loggers()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nExiting tune.breakaway ...")


async def _run(args: argparse.Namespace) -> None:
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
                "trim_nm",
                "drift_deg",
            ]
        )

    channel = resolve_channel(args)
    async with CanBus(channel) as bus:
        raw = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw.values()])
        motors = await joint_frame_motors(raw, is_left)
        # One reset window, before anything moves. A MyActuator mode switch
        # is a system reset and the joint holds nothing for the ~2 s it takes,
        # so the probed joint takes its impedance mode here rather than after
        # the arm has been parked — same reason tune.gravity/tune.friction
        # assign modes up front (see friction.assign_modes).
        await asyncio.gather(
            *[
                m.set_control_mode(
                    ControlMode.IMPEDANCE
                    if j is joint
                    else ControlMode.POSITION_VELOCITY
                )
                for j, m in motors.items()
            ]
        )
        try:
            # `sweep_safety` supplies this joint's safe range and any notes;
            # the holding itself goes through `ramp_others_to_zero`, which
            # parks every other joint at rest under its own position loop and
            # already knows which ones must not be blindly commanded (their
            # inboard half can meet the base) and which need a clearance pose.
            _, lo_default, hi_default, notes = sweep_safety(joint, is_left)
            for note in notes:
                print(f"  {note}")
            print("  Parking every other joint at rest ...")
            await ramp_others_to_zero(motors, joint, is_left)
            # The test joint starts from rest too, so a probe pose is always
            # reached by a ramp from a known place rather than from wherever
            # the last run left it. It is already in impedance, so it ramps
            # under its own spring rather than a position command.
            await _ramp_to(motors[joint], kp_cfg, args.kd, 0.0)

            lo, hi = arm_limits(joint, is_left)
            lo = lo_default if lo_default is not None else lo
            hi = hi_default if hi_default is not None else hi
            if args.poses is None:
                poses = [float(np.clip(await motors[joint].get_position(), lo, hi))]
            else:
                poses = [float(np.clip(math.radians(p), lo, hi)) for p in args.poses]

            by_pose = []
            for q in poses:
                # Gravity feedforward is the *only* thing holding this joint
                # once kp goes to zero, so it has to come from where the arm
                # actually is. `sweep_safety` has no hold pose for several
                # joints (shoulder_1 among them), and assuming one would put
                # the wrong torque on the wire.
                q_arm = np.zeros(len(ARM_JOINTS))
                for idx, j in enumerate(ARM_JOINTS):
                    if j in motors:
                        q_arm[idx] = await motors[j].get_position()
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
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            if csv_file is not None:
                csv_file.close()
            print("  Returning to rest and disabling ...")
            # The test joint gets an impedance ramp home only if it actually
            # reached IMPEDANCE mode; otherwise `_home_all` covers it like
            # any other joint. Both are best-effort: a failure here must not
            # stop the disable below.
            in_impedance = motors[joint].motor.mode == ControlMode.IMPEDANCE
            if in_impedance:
                try:
                    await _ramp_to(motors[joint], kp_cfg, args.kd, 0.0)
                except Exception:
                    pass
            try:
                await _home_all(motors, exclude=joint if in_impedance else None)
            except Exception:
                pass
            # Homing put every joint in POSITION_VELOCITY. Leaving them there
            # is not cosmetic: the realtime core checks each Damiao's control
            # mode at bring-up and refuses to arm on anything but MIT, so a
            # run that exits without restoring it bricks the next teleop or
            # tune.motion with "control mode 2 (expected 1)". The gripper is
            # not in `motors` and keeps its POSITION_FORCE mode.
            try:
                await asyncio.gather(
                    *[
                        m.set_control_mode(ControlMode.IMPEDANCE)
                        for m in motors.values()
                    ],
                    return_exceptions=True,
                )
            except Exception:
                pass
            await asyncio.gather(
                *[m.disable() for m in raw.values()], return_exceptions=True
            )
