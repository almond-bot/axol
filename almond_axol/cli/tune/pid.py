"""
axol tune.pid

Tune Kp/Kd for a single Axol joint at ~100 Hz.

Tests one or more (Kp, Kd) candidates — pass several values to either flag to
sweep the whole grid in one session — via sinusoidal or step-response
tracking, measures error (RMS, max, overshoot, settling), ranks the
candidates, and can persist the winner to this robot's calibration file
(``~/.almond/calibration.json``, loaded automatically by ``AxolConfig``).

The feedforward path mirrors the production ``motion_control`` loop —
``gravity + friction + j_eff·a_des`` — so gains found here transfer 1:1 to
teleop. Use ``--ff`` to tune against a reduced feedforward instead (e.g.
``--ff none`` for bare PD, ``--ff gravity`` for gravity-only).

Examples:
    axol tune.pid --l --joint elbow                       # config gains, production FF
    axol tune.pid --l --joint elbow --kp 20 30 45 --kd 0.5 1.0   # 6-way sweep
    axol tune.pid --r --joint shoulder_1 --mode step --ff none
    axol tune.pid --l --joint wrist_1 --kp 12 --kd 0.4 --save
"""

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path

import numpy as np

from ...constants import ARM_JOINTS
from ...motor import CanBus, ControlMode, Joint, Motor
from ...robot.axol import arm_limits
from ...robot.calibration import CALIBRATION_PATH, update_joint_calibration
from ...robot.config import ArmConfig, AxolConfig, JointConfig
from ...robot.control import Differentiator, compute_friction
from ...robot.gravity import GravityCompensator
from ..motor import add_side_and_channel_arguments, resolve_channel
from .joint_frame import JointFrameMotor, joint_frame_motors

# Default sine amplitude / step size (rad). 0.175 rad ≈ 10° — well above
# the encoder noise floor and the ``5%`` settling threshold (≈0.5°), well
# clear of the friction-stiction breakaway condition (``kp · amp > Fc``)
# at all typical PID-tuning gains, and small enough to avoid hitting joint
# limits or driving any joint into its high-velocity saturation regime.
_DEFAULT_AMP_RAD = 0.175
_RAMP_SPEED = 0.25  # rad/s

_FF_MODES = ("full", "gravity", "friction", "none")


# Joints whose 0 position physically collides with the robot base. ``run_step``
# frames the test entirely in the safe (outboard) half for these, and
# ``_ramp_others_to_zero`` leaves them in place rather than commanding 0.
_BASE_COLLISION_JOINTS = frozenset({Joint.SHOULDER_2, Joint.WRIST_2})


def _safe_outboard_direction(joint: Joint, is_left: bool) -> int | None:
    """Step direction that swings away from the robot base, or ``None`` if the
    joint has no base-collision constraint."""
    if joint == Joint.SHOULDER_2:
        return -1 if is_left else 1
    if joint == Joint.WRIST_2:
        # mirrored across arms: the outboard (base-free) half is +π/2 on the
        # left arm and −π/2 on the right.
        return 1 if is_left else -1
    return None


def _sine_center(joint: Joint, is_left: bool) -> float:
    lo, hi = arm_limits(joint, is_left)
    if joint == Joint.WRIST_2:
        # wrist_2 midpoint is 0; the inboard half hits the robot base, so
        # center at the midpoint of the outboard half (side-dependent).
        return hi / 2.0 if is_left else lo / 2.0
    return (lo + hi) / 2.0


def _safe_amplitude(
    joint: Joint, is_left: bool, center: float, requested: float | None
) -> float:
    lo, hi = arm_limits(joint, is_left)
    if not (lo <= center <= hi):
        raise ValueError(
            f"Current position {center:.4f} rad is outside [{lo:.4f}, {hi:.4f}] for {joint.value}"
        )
    headroom = min(center - lo, hi - center)
    if headroom < 0.03:  # ~1.7°
        raise ValueError(
            f"{joint.value} center {center:.4f} rad is too close to a limit [{lo:.4f}, {hi:.4f}]. "
            f"Sine test centers on the joint midpoint ({_sine_center(joint, is_left):.4f} rad) — "
            f"move there first, or use --mode step."
        )
    if requested is not None:
        amp = min(requested, headroom)
        if amp < requested:
            print(
                f"  ! requested amp {requested:.4f} rad exceeds headroom; clamped to {amp:.4f} rad"
            )
    else:
        amp = min(_DEFAULT_AMP_RAD, headroom)
    return amp


class FeedForward:
    """Per-run feedforward matching the production ``motion_control`` path.

    ``compute(q_target, q_meas)`` returns ``(v_des, t_ff)`` where::

        t_ff = gravity(q_target) + friction(v_des) + j_eff · a_des
               + host_kd · (v_des − v_meas)

    ``gravity_fn`` evaluates the full-chain URDF model with the *other*
    joints at their real (measured) positions, not an assumed zero pose —
    shoulder_2 / wrist_2 are deliberately never parked at 0 (base
    collision), so assuming zeros there skews the model torque.

    ``host_kd`` adds host-side velocity damping computed from the
    differentiated measured position (the ``kd_soft`` scheme). It exists to
    diagnose joints where the motor's firmware kd underdelivers — on the
    high-inertia shoulders the firmware velocity estimate appears too
    filtered to damp the ~2 Hz closed-loop resonance, while a host-side
    estimate at the command rate still can.

    Construct one instance per candidate run: the differentiators are
    stateful low-pass filters and must not leak between runs. For step
    references pass ``differentiate_target=False`` — differentiating a
    discontinuous target would fire a one-sample velocity/accel spike into
    the motor; production never sees that because ``max_step_rad`` keeps
    commanded steps small.
    """

    def __init__(
        self,
        gravity_fn,
        fc: float,
        k: float,
        fv: float,
        fo: float,
        j_eff: float,
        differentiate_target: bool = True,
        host_kd: float = 0.0,
    ) -> None:
        self.gravity_fn = gravity_fn
        self._fric = (fc, k, fv, fo)
        self._j_eff = j_eff
        self._differentiate_target = differentiate_target
        self._host_kd = host_kd
        self._v_des_diff = Differentiator(1)
        self._a_des_diff = Differentiator(1)
        self._v_meas_diff = Differentiator(1)

    def compute(
        self, q_target: float, q_meas: float | None = None
    ) -> tuple[float, float]:
        if self._differentiate_target:
            v_des = self._v_des_diff.differentiate([q_target])[0]
            a_des = self._a_des_diff.differentiate([v_des])[0]
        else:
            v_des = a_des = 0.0
        t_ff = (
            self.gravity_fn(q_target)
            + compute_friction(v_des, *self._fric)
            + self._j_eff * a_des
        )
        if self._host_kd and q_meas is not None:
            v_meas = self._v_meas_diff.differentiate([q_meas])[0]
            t_ff += self._host_kd * (v_des - v_meas)
        return v_des, t_ff


async def _ramp_impedance(
    motor: JointFrameMotor,
    kp: float,
    kd: float,
    target: float,
    gravity_fn,
    rate_hz: float = 100.0,
    speed: float = _RAMP_SPEED,
) -> None:
    """Ramp one impedance-mode joint to ``target`` (joint frame) at ``speed``, with gravity FF."""
    start = await motor.get_position()
    duration = max(abs(target - start) / speed, 0.5)
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / duration, 1.0)
        q = start + alpha * (target - start)
        await motor.set_impedance(q, 0.0, kp, kd, gravity_fn(q))
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def _ramp_others_to_zero(
    motors: dict[Joint, JointFrameMotor],
    exclude: Joint,
) -> None:
    """Send non-test joints to rest (joint-frame 0) and poll until arrival.

    Joints listed in ``_BASE_COLLISION_JOINTS`` are also skipped: 0 physically
    collides with the robot base (the URDF limits don't capture this), and the
    rest of the workflow keeps them safely outboard — ``run_step`` repositions
    before testing, and ``run_sine`` centers them in the safe half. The user is
    responsible for initially posing those joints outside the danger zone.
    """
    skip = {exclude} | _BASE_COLLISION_JOINTS
    joints = [j for j in ARM_JOINTS if j not in skip]
    pos_vals = await asyncio.gather(*[motors[j].get_position() for j in joints])
    max_dist = max((abs(p) for p in pos_vals), default=0.0)
    await asyncio.gather(
        *[motors[j].set_position_velocity(0.0, _RAMP_SPEED) for j in joints]
    )
    timeout = max_dist / _RAMP_SPEED + 2.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.1)
        positions = await asyncio.gather(*[motors[j].get_position() for j in joints])
        if all(abs(p) < 0.05 for p in positions):
            break


async def run_sine(
    motors: dict[Joint, JointFrameMotor],
    joint: Joint,
    kp: float,
    kd: float,
    freq: float,
    requested_amp: float | None,
    duration: float,
    rate_hz: float,
    is_left: bool,
    ff: FeedForward,
) -> tuple[list[dict], float]:
    """Track a sine reference on ``joint`` and log target/actual error.

    Returns the per-sample log and the amplitude actually used (clamped
    to the joint's headroom).
    """
    test_motor = motors[joint]
    lo, hi = arm_limits(joint, is_left)
    center = _sine_center(joint, is_left)
    amp = _safe_amplitude(joint, is_left, center, requested_amp)
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"amp=±{amp:.4f} rad  freq={freq:.2f} Hz"
    )

    print("  moving to center ...")
    await _ramp_impedance(test_motor, kp, kd, center, ff.gravity_fn, rate_hz)
    await asyncio.sleep(1.0)

    print(f"  running {duration:.1f} s at {rate_hz:.0f} Hz ...")
    dt = 1.0 / rate_hz
    log: list[dict] = []
    q_meas = await test_motor.get_position()
    start = time.monotonic()

    while True:
        t = time.monotonic() - start
        if t >= duration:
            break
        loop_start = time.monotonic()

        target = center + amp * math.sin(2 * math.pi * freq * t)
        v_des, t_ff = ff.compute(target, q_meas)
        await test_motor.set_impedance(target, v_des, kp, kd, t_ff)
        actual = await test_motor.get_position()
        q_meas = actual
        t_read = time.monotonic() - start
        target_at_read = center + amp * math.sin(2 * math.pi * freq * t_read)
        log.append(
            {
                "t": round(t_read, 5),
                "target": target_at_read,
                "actual": actual,
                "error": actual - target_at_read,
            }
        )

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    return log, amp


async def run_step(
    motors: dict[Joint, JointFrameMotor],
    joint: Joint,
    kp: float,
    kd: float,
    requested_amp: float | None,
    hold: float,
    rate_hz: float,
    is_left: bool,
    ff: FeedForward,
) -> tuple[list[dict], float]:
    """Drive a step on ``joint`` and log the step-response error.

    Returns the per-sample log and the amplitude actually used (clamped
    to the joint's safe headroom).
    """
    test_motor = motors[joint]
    current = await test_motor.get_position()
    lo, hi = arm_limits(joint, is_left)

    safe_dir = _safe_outboard_direction(joint, is_left)
    if safe_dir is not None:
        # 0 physically collides with the robot base; frame the whole test in
        # the safe half so that center *and* step_target stay outboard. amp
        # goes from 0 → safe-limit/2 (room for a 2× swing).
        direction = safe_dir
        outboard_limit = lo if direction < 0 else hi
        max_safe_amp = abs(outboard_limit) / 2.0
        amp = min(
            requested_amp if requested_amp is not None else _DEFAULT_AMP_RAD,
            max_safe_amp,
        )
        center = direction * amp
        step_target = direction * 2.0 * amp
        if requested_amp is not None and amp < requested_amp:
            print(
                f"  ! requested amp {requested_amp:.4f} rad would push past the safe half; clamped to {amp:.4f} rad"
            )
    else:
        center = current
        headroom_up = hi - center
        headroom_down = center - lo
        if headroom_up < 0.03 and headroom_down < 0.03:
            raise ValueError(
                f"{joint.value} at {center:.4f} rad has no headroom within [{lo:.4f}, {hi:.4f}]."
            )
        if headroom_up >= headroom_down:
            direction, headroom = 1, headroom_up
        else:
            direction, headroom = -1, headroom_down

        if requested_amp is not None:
            amp = min(requested_amp, headroom)
            if amp < requested_amp:
                print(
                    f"  ! requested amp {requested_amp:.4f} rad exceeds headroom; clamped to {amp:.4f} rad"
                )
        else:
            amp = min(_DEFAULT_AMP_RAD, headroom)
        step_target = center + direction * amp

    sign_str = f"+{amp:.4f}" if direction == 1 else f"-{amp:.4f}"
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"step={sign_str} rad  hold={hold:.1f} s  rate={rate_hz:.0f} Hz"
    )

    if abs(current - center) > 0.01:
        print(f"  moving to step center ({center:.4f} rad) ...")
        await _ramp_impedance(test_motor, kp, kd, center, ff.gravity_fn, rate_hz)
        await asyncio.sleep(0.5)

    dt = 1.0 / rate_hz
    log: list[dict] = []
    q_meas = await test_motor.get_position()
    start = time.monotonic()

    for phase_target in [step_target, center]:
        phase_start = time.monotonic()
        while time.monotonic() - phase_start < hold:
            loop_start = time.monotonic()
            t = time.monotonic() - start
            _, t_ff = ff.compute(phase_target, q_meas)
            await test_motor.set_impedance(phase_target, 0.0, kp, kd, t_ff)
            actual = await test_motor.get_position()
            q_meas = actual
            log.append(
                {
                    "t": round(t, 5),
                    "target": phase_target,
                    "actual": actual,
                    "error": actual - phase_target,
                }
            )
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)

    return log, amp


def _sine_metrics(log: list[dict]) -> dict[str, float]:
    errors = [r["error"] for r in log]
    rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
    max_err = max(abs(e) for e in errors)
    elapsed = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 1.0
    actual_hz = len(log) / elapsed if elapsed > 0 else 0.0
    # Score: tracking RMS dominates, with a small penalty on the worst
    # excursion so two equal-RMS candidates prefer the one without spikes.
    return {
        "rms": rms,
        "max": max_err,
        "hz": actual_hz,
        "score": rms + 0.2 * max_err,
    }


def _step_metrics(log: list[dict], amp: float, hold: float) -> dict[str, float | None]:
    targets = list(dict.fromkeys(r["target"] for r in log))
    step_target = targets[0]
    step_rows = [r for r in log if r["target"] == step_target]
    return_target = targets[1] if len(targets) > 1 else step_target - amp
    direction = 1 if step_target > return_target else -1
    overshoot = max(
        0.0, max(direction * (r["actual"] - step_target) for r in step_rows)
    )

    threshold = 0.05 * amp
    t_step_start = step_rows[0]["t"]
    settling_s = None
    for i, r in enumerate(step_rows):
        if abs(r["error"]) < threshold:
            future = step_rows[i : i + 10]
            if len(future) == 10 and all(abs(fr["error"]) < threshold for fr in future):
                settling_s = r["t"] - t_step_start
                break

    settled = step_rows[len(step_rows) // 2 :]
    ss_rms = (
        math.sqrt(sum(r["error"] ** 2 for r in settled) / len(settled))
        if settled
        else 0.0
    )
    elapsed = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 1.0
    actual_hz = len(log) / elapsed if elapsed > 0 else 0.0
    overshoot_frac = overshoot / amp if amp > 0 else 0.0
    # Score (heuristic, lower is better): settling time in seconds, plus
    # 0.5 s of penalty per 10% overshoot, plus steady-state RMS weighted so
    # 0.01 rad ≈ 0.1 s. A candidate that never settles is charged 2× hold.
    score = (
        (settling_s if settling_s is not None else 2.0 * hold)
        + 5.0 * overshoot_frac
        + 10.0 * ss_rms
    )
    return {
        "settling_s": settling_s,
        "overshoot": overshoot,
        "overshoot_frac": overshoot_frac,
        "ss_rms": ss_rms,
        "hz": actual_hz,
        "score": score,
    }


def _print_stats_sine(m: dict, n: int, kp: float, kd: float) -> None:
    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {n}  ({m['hz']:.1f} Hz actual)")
    print(f"  RMS error:  {m['rms']:.5f} rad  ({math.degrees(m['rms']):.3f}°)")
    print(f"  Max error:  {m['max']:.5f} rad  ({math.degrees(m['max']):.3f}°)")
    print(f"{'─' * 40}")


def _print_stats_step(m: dict, n: int, kp: float, kd: float) -> None:
    settling = (
        f"{m['settling_s'] * 1000:.0f} ms"
        if m["settling_s"] is not None
        else ">hold time"
    )
    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {n}  ({m['hz']:.1f} Hz actual)")
    print(f"  Settling:   {settling}  (5% threshold)")
    print(
        f"  Overshoot:  {math.degrees(m['overshoot']):.3f}°  "
        f"({m['overshoot_frac'] * 100:.1f}% of step)"
    )
    print(f"  SS RMS:     {m['ss_rms']:.5f} rad  ({math.degrees(m['ss_rms']):.3f}°)")
    print(f"{'─' * 40}")


def _print_ranking(mode: str, results: list[dict]) -> None:
    """Comparison table over all candidates, best first."""
    ranked = sorted(results, key=lambda r: r["metrics"]["score"])
    print(f"\n{'═' * 64}")
    print(f"  Candidate ranking ({mode}, lower score is better)")
    if mode == "sine":
        print(f"  {'Kp':>7}  {'Kd':>6}  {'RMS °':>8}  {'Max °':>8}  {'score':>8}")
        for r in ranked:
            m = r["metrics"]
            print(
                f"  {r['kp']:>7.1f}  {r['kd']:>6.2f}  "
                f"{math.degrees(m['rms']):>8.3f}  {math.degrees(m['max']):>8.3f}  "
                f"{m['score']:>8.5f}"
            )
    else:
        print(
            f"  {'Kp':>7}  {'Kd':>6}  {'settle':>8}  {'ovsh %':>7}  "
            f"{'SS °':>7}  {'score':>7}"
        )
        for r in ranked:
            m = r["metrics"]
            settling = (
                f"{m['settling_s'] * 1000:.0f}ms"
                if m["settling_s"] is not None
                else ">hold"
            )
            print(
                f"  {r['kp']:>7.1f}  {r['kd']:>6.2f}  {settling:>8}  "
                f"{m['overshoot_frac'] * 100:>7.1f}  "
                f"{math.degrees(m['ss_rms']):>7.3f}  {m['score']:>7.3f}"
            )
    best = ranked[0]
    print(f"\n  Best: Kp={best['kp']}  Kd={best['kd']}")
    print(f"{'═' * 64}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.pid`` subcommand."""
    p = subparsers.add_parser(
        "tune.pid",
        help="Tune Kp/Kd for a single Axol joint (sweeps candidate gains).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to tune: {', '.join(j.value for j in ARM_JOINTS)}",
    )
    p.add_argument(
        "--kp",
        type=float,
        nargs="+",
        default=None,
        metavar="KP",
        help="Proportional gain candidate(s); several values sweep the grid "
        "(default: this joint's configured kp)",
    )
    p.add_argument(
        "--kd",
        type=float,
        nargs="+",
        default=None,
        metavar="KD",
        help="Derivative gain candidate(s); several values sweep the grid "
        "(default: this joint's configured kd)",
    )
    p.add_argument(
        "--ff",
        choices=list(_FF_MODES),
        default="full",
        help="Feedforward during the test: full (gravity + friction + inertia "
        "— matches production; default), gravity, friction, or none (bare PD)",
    )
    p.add_argument(
        "--mode",
        choices=["sine", "step"],
        default="sine",
        help="sine (default): continuous tracking; step: step response",
    )
    p.add_argument(
        "--host-kd",
        type=float,
        default=None,
        metavar="KD",
        help="Host-side damping (Nm·s/rad) applied through t_ff from the "
        "differentiated measured position — matches the production kd_host "
        "term, needed on the high-inertia shoulders where firmware kd "
        "underdelivers (default: this joint's configured kd_host)",
    )
    p.add_argument(
        "--amp",
        type=float,
        default=None,
        help="Motion amplitude in rad (default: 0.175 rad ≈ 10°, clamped to joint headroom)",
    )
    p.add_argument(
        "--freq", type=float, default=1.0, help="[sine] Frequency in Hz (default: 1.0)"
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="[sine] Duration in seconds (default: 5.0)",
    )
    p.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="[step] Hold time per phase in seconds (default: 2.0)",
    )
    p.add_argument(
        "--rate", type=float, default=100.0, help="Command rate in Hz (default: 100)"
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save the best candidate's Kp/Kd to this robot's calibration file "
        f"({CALIBRATION_PATH}); it then overrides the shared defaults on this "
        "machine",
    )
    p.add_argument(
        "--dump-csv",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help="Write per-sample (kp, kd, t, target, actual, error) rows to a "
        "CSV for offline plotting. Pass without a value to auto-name as "
        "logs/pid_<side>_<joint>_<timestamp>.csv, or pass an explicit path.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the PID tuning session for the selected joint."""
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side_str = "left" if is_left else "right"
    lo, hi = arm_limits(joint, is_left)

    arm_cfg: ArmConfig = AxolConfig().left if is_left else AxolConfig().right
    jc: JointConfig = getattr(arm_cfg, joint.value)
    kps: list[float] = args.kp if args.kp else [jc.kp]
    kds: list[float] = args.kd if args.kd else [jc.kd]
    candidates = [(kp, kd) for kp in kps for kd in kds]

    use_gravity = args.ff in ("full", "gravity")
    use_friction = args.ff in ("full", "friction")
    f = jc.friction
    fric = (f.fc, f.k, f.fv, f.fo) if use_friction else (0.0, 0.0, 0.0, 0.0)
    j_eff = jc.j_eff if args.ff == "full" else 0.0
    host_kd = args.host_kd if args.host_kd is not None else jc.kd_host

    gravity_comp = GravityCompensator() if use_gravity else None
    test_idx = ARM_JOINTS.index(joint)
    arm_q_buf = np.zeros(len(ARM_JOINTS), dtype=np.float32)

    def gravity_fn(q: float) -> float:
        if gravity_comp is None:
            return 0.0
        arm_q_buf[test_idx] = q
        return float(gravity_comp.gravity_arm(arm_q_buf, is_left=is_left)[test_idx])

    dump_csv: Path | None = None
    if args.dump_csv == "__auto__":
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_csv = Path("logs") / f"pid_{side_str}_{joint.value}_{timestamp}.csv"
    elif args.dump_csv is not None:
        dump_csv = Path(args.dump_csv)
    if dump_csv is not None:
        dump_csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nAxol PID tuner — {side_str} {joint.value}  limits=[{lo:.4f}, {hi:.4f}] rad"
    )
    print(f"  mode={args.mode}  ff={args.ff}  candidates={len(candidates)}")
    # The configured kp/kd are the compliant (s=0) endpoint; production blends
    # them toward the stiff endpoint by the robot's stiffness setting. Print
    # both so the operator knows what the tested values relate to.
    resolved_jc: JointConfig = getattr(
        AxolConfig().resolved().left if is_left else AxolConfig().resolved().right,
        joint.value,
    )
    print(
        f"  config kp={jc.kp:.1f} kd={jc.kd:.2f} (compliant endpoint)  |  "
        f"production at default stiffness: kp={resolved_jc.kp:.1f} kd={resolved_jc.kd:.2f}"
    )
    if use_friction:
        print(f"  friction  Fc={f.fc}  k={f.k}  Fv={f.fv}  Fo={f.fo}")
    if args.ff == "full":
        print(f"  inertia  j_eff={j_eff}")
    if host_kd:
        print(f"  host-kd  {host_kd} (host-side damping via t_ff, kd_host)")

    channel = resolve_channel(args)

    csv_file = None
    csv_writer = None
    if dump_csv is not None:
        csv_file = dump_csv.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["kp", "kd", "mode", "t", "target", "actual", "error"])
        print(f"  dumping per-sample rows to {dump_csv}")

    results: list[dict] = []
    ref_kp, ref_kd = candidates[0]

    async with CanBus(channel) as bus:
        raw_motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw_motors.values()])
        # Motor encoders are zeroed at end stops; everything below (limits,
        # centers, gravity poses) is joint frame (0 = rest), so wrap the
        # motors in the frame conversion before any position I/O.
        motors = await joint_frame_motors(raw_motors, is_left)
        await asyncio.gather(
            *[
                motors[j].set_control_mode(
                    ControlMode.IMPEDANCE
                    if j == joint
                    else ControlMode.POSITION_VELOCITY
                )
                for j in motors
            ]
        )

        try:
            print("  ramping other joints to rest (joint-frame 0) ...")
            await _ramp_others_to_zero(motors, joint)

            # Fill the gravity pose with the *measured* positions of the
            # other joints: base-collision joints (shoulder_2, wrist_2) are
            # deliberately left off-zero, and assuming 0 there would bias
            # the gravity feedforward for every candidate.
            positions = await asyncio.gather(
                *[motors[j].get_position() for j in ARM_JOINTS]
            )
            arm_q_buf[:] = positions

            for i, (kp, kd) in enumerate(candidates):
                if len(candidates) > 1:
                    print(f"\n[{i + 1}/{len(candidates)}] testing Kp={kp}  Kd={kd}")
                else:
                    print(f"\n  testing Kp={kp}  Kd={kd}")
                ff = FeedForward(
                    gravity_fn,
                    *fric,
                    j_eff=j_eff,
                    differentiate_target=(args.mode == "sine"),
                    host_kd=host_kd,
                )
                if args.mode == "sine":
                    log, amp = await run_sine(
                        motors,
                        joint,
                        kp,
                        kd,
                        args.freq,
                        args.amp,
                        args.duration,
                        args.rate,
                        is_left,
                        ff,
                    )
                    metrics = _sine_metrics(log)
                    _print_stats_sine(metrics, len(log), kp, kd)
                else:
                    log, amp = await run_step(
                        motors,
                        joint,
                        kp,
                        kd,
                        args.amp,
                        args.hold,
                        args.rate,
                        is_left,
                        ff,
                    )
                    metrics = _step_metrics(log, amp, args.hold)
                    _print_stats_step(metrics, len(log), kp, kd)

                results.append({"kp": kp, "kd": kd, "metrics": metrics})
                if csv_writer is not None:
                    for r in log:
                        csv_writer.writerow(
                            [
                                kp,
                                kd,
                                args.mode,
                                f"{r['t']:.5f}",
                                f"{r['target']:.6f}",
                                f"{r['actual']:.6f}",
                                f"{r['error']:.6f}",
                            ]
                        )
                    csv_file.flush()
                await asyncio.sleep(0.5)

            if len(results) > 1:
                _print_ranking(args.mode, results)

            if args.save and results:
                best = min(results, key=lambda r: r["metrics"]["score"])
                # Persist kd_host only when the operator set it explicitly:
                # the gains were validated with that damping active, so the
                # two must be saved (and later loaded) together.
                path = update_joint_calibration(
                    side_str,
                    joint.value,
                    kp=best["kp"],
                    kd=best["kd"],
                    kd_host=args.host_kd,
                )
                saved = f"Kp={best['kp']}  Kd={best['kd']}"
                if args.host_kd is not None:
                    saved += f"  kd_host={args.host_kd}"
                print(f"\n  Saved {saved} to {path}")
                print(
                    "  (loaded automatically by AxolConfig on this machine; "
                    "these are the compliant s=0 endpoint of the stiffness "
                    "blend, like the shared defaults they replace)"
                )

        except KeyboardInterrupt:
            print("\n  interrupted")
        finally:
            if csv_file is not None:
                csv_file.close()
            # Slow controlled ramp to rest — for shoulder_2 this is the *safe*
            # way to reach the base side: the danger was a fast mid-step
            # return-to-center, not the gentle approach at _RAMP_SPEED.
            print("  returning to rest ...")
            try:
                await _ramp_impedance(
                    motors[joint], ref_kp, ref_kd, 0.0, gravity_fn, args.rate
                )
            except Exception:
                pass
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])
