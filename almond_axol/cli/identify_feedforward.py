"""
almond-axol identify-feedforward

Identify all six feedforward parameters (gravity + friction) for an Axol joint
in a single bidirectional sweep.

Sweeps the full joint range at multiple velocities, both forward and backward.
Bidirectional averaging at the same position separates gravity from friction:

    avg(τ_fwd, τ_bwd) at same q     →  ga·cos(q) + gb·sin(q) + Fo
    half(τ_fwd - τ_bwd) at same q   →  Fc·tanh(0.1·k·v) + Fv·v

Fits six parameters at once:
    ga, gb  — gravity model (τ_grav = ga·cos(q) + gb·sin(q))
    Fc, k   — Coulomb friction magnitude and tanh sharpness
    Fv      — viscous friction coefficient
    Fo      — constant torque offset (from gravity fit)

At runtime:
    tff(q, v) = ga·cos(q) + gb·sin(q) + Fc·tanh(0.1·k·v) + Fv·v + Fo

Examples:
    almond-axol identify-feedforward --l --joint shoulder_1 --kp 30 --kd 0.8
    almond-axol identify-feedforward --r --joint elbow --kp 20 --kd 0.6
    almond-axol identify-feedforward --l --joint wrist_1 --velocities 0.2 0.6 1.0
"""

import argparse
import asyncio
import math
import time

import numpy as np
from scipy.optimize import curve_fit

from ..motor import CanBus, ControlMode, Joint, Motor
from ..robot.axol import arm_limits
from ..robot.config import AxolConfig
from ..shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT

_TAU = 2 * math.pi
_RAMP_SPEED = 0.25  # rad/s
_SWEEP_MARGIN = 0.05  # rad — don't sweep all the way to hard limits
_WARMUP_FRACTION = 0.15  # skip first 15% of each pass for motor settling
_RATE_HZ = 100.0
_N_BINS = 40  # position bins for matching fwd/bwd samples

# Default velocity sweep in rad/s (~0.02, 0.05, 0.1, 0.15, 0.2 rev/s)
DEFAULT_VELOCITIES = [v * _TAU for v in [0.02, 0.05, 0.1, 0.15, 0.2]]


async def _ramp_to(
    motor: Motor,
    kp: float,
    kd: float,
    target: float,
    duration: float = 2.0,
) -> None:
    start_pos = await motor.get_position()
    dt = 1.0 / _RATE_HZ
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / duration, 1.0)
        await motor.motion_control(
            start_pos + alpha * (target - start_pos), 0.0, kp, kd, 0.0
        )
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def _ramp_others(
    motors: dict[Joint, Motor],
    exclude: Joint,
    targets: dict[Joint, float] | None = None,
) -> None:
    """Move all joints except `exclude` to their target positions (default 0)."""
    joints = [j for j in ARM_JOINTS if j != exclude]
    t = targets or {}
    pos_vals = await asyncio.gather(*[motors[j].get_position() for j in joints])
    max_dist = max(
        (abs(pos - t.get(j, 0.0)) for j, pos in zip(joints, pos_vals)), default=0.0
    )
    await asyncio.gather(
        *[motors[j].set_position(t.get(j, 0.0), _RAMP_SPEED) for j in joints]
    )
    timeout = max_dist / _RAMP_SPEED + 2.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.1)
        positions = await asyncio.gather(*[motors[j].get_position() for j in joints])
        if all(abs(pos - t.get(j, 0.0)) < 0.05 for j, pos in zip(joints, positions)):
            break


async def _run_sweep_raw(
    motor: Motor,
    kp: float,
    kd: float,
    start_pos: float,
    velocity_rad_s: float,
    end_pos: float,
) -> list[tuple[float, float, float]]:
    """Sweep from start_pos to end_pos at constant velocity.

    Returns list of (q_actual, v_actual, tau_est). No friction subtraction.
    The first WARMUP_FRACTION of travel is discarded for motor settling.
    """
    travel = abs(end_pos - start_pos)
    if travel < 0.02:
        return []
    total_time = travel / abs(velocity_rad_s)
    warmup_time = total_time * _WARMUP_FRACTION
    dt = 1.0 / _RATE_HZ

    samples: list[tuple[float, float, float]] = []
    pos_prev: float | None = None
    t_prev: float | None = None

    t0 = time.monotonic()
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= total_time:
            break
        loop_start = now

        target = start_pos + velocity_rad_s * t
        await motor.motion_control(target, velocity_rad_s, kp, kd, 0.0)
        q_actual = await motor.get_position()

        if t >= warmup_time and pos_prev is not None and t_prev is not None:
            pos_err = target - q_actual
            dt_actual = now - t_prev
            v_actual = (q_actual - pos_prev) / dt_actual if dt_actual > 0 else 0.0
            vel_err = velocity_rad_s - v_actual
            tau_est = kp * pos_err + kd * vel_err
            samples.append((q_actual, v_actual, tau_est))

        pos_prev = q_actual
        t_prev = now

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    return samples


def _bin_by_position(
    samples: list[tuple[float, float, float]],
    sweep_lo: float,
    sweep_hi: float,
    n_bins: int = _N_BINS,
) -> dict[float, float]:
    """Return {bin_center: mean_tau} from (q, v, tau) samples."""
    span = sweep_hi - sweep_lo
    if span <= 0:
        return {}
    bin_width = span / n_bins
    buckets: dict[int, list[float]] = {}
    for q, _v, tau in samples:
        idx = int((q - sweep_lo) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        buckets.setdefault(idx, []).append(tau)
    return {
        sweep_lo + (idx + 0.5) * bin_width: float(np.mean(taus))
        for idx, taus in buckets.items()
        if len(taus) >= 2
    }


def _fit_gravity_with_offset(
    avg_samples: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Linear fit: tau_avg = ga*cos(q) + gb*sin(q) + Fo.

    Returns (ga, gb, Fo) or None.
    """
    if len(avg_samples) < 10:
        print("  ! Too few avg samples to fit gravity.")
        return None
    q_arr = np.array([s[0] for s in avg_samples])
    t_arr = np.array([s[1] for s in avg_samples])
    A = np.column_stack([np.cos(q_arr), np.sin(q_arr), np.ones_like(q_arr)])
    coeffs, *_ = np.linalg.lstsq(A, t_arr, rcond=None)
    ga, gb, Fo = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    residual_rms = float(np.sqrt(np.mean((t_arr - A @ coeffs) ** 2)))
    amplitude = float(np.sqrt(ga**2 + gb**2))
    print(
        f"  Gravity fit residual RMS: {residual_rms:.4f} Nm  (amplitude: {amplitude:.4f} Nm)"
    )
    return ga, gb, Fo


def _tanh_friction(v: np.ndarray, Fc: float, k: float, Fv: float) -> np.ndarray:
    return Fc * np.tanh(0.1 * k * v) + Fv * v


def _fit_friction_halfdiff(
    halfdiff_samples: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Fit Fc*tanh(0.1*k*v) + Fv*v to half-difference samples.

    Returns (Fc, k, Fv) or None.
    """
    if len(halfdiff_samples) < 5:
        print("  ! Too few half-diff samples to fit friction.")
        return None
    v_arr = np.array([s[0] for s in halfdiff_samples])
    t_arr = np.array([s[1] for s in halfdiff_samples])

    # Half-differences should be positive; clamp noise-driven negatives
    t_arr = np.maximum(t_arr, 0.0)

    Fc_guess = float(np.mean(t_arr))
    try:
        popt, _ = curve_fit(
            _tanh_friction,
            v_arr,
            t_arr,
            p0=[Fc_guess, 10.0, 0.02],
            bounds=([0, 0.1, 0], [10.0, 1000.0, 5.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception as e:
        print(f"  ! Friction fit failed: {e}")
        return None


async def _identify_joint(
    motor: Motor,
    joint: Joint,
    kp: float,
    kd: float,
    is_left: bool,
    velocities: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Run bidirectional multi-velocity sweep over the full joint range.

    Returns:
        avg_samples:      (q, tau_avg)  — for gravity+Fo fitting
        halfdiff_samples: (v, tau_half) — for Fc/k/Fv fitting
    """
    lo, hi = arm_limits(joint, is_left)
    sweep_lo = lo + _SWEEP_MARGIN
    sweep_hi = hi - _SWEEP_MARGIN

    print(f"\n  Joint limits: [{lo:.4f}, {hi:.4f}] rad")
    print(f"  Sweep range:  [{sweep_lo:.4f}, {sweep_hi:.4f}] rad")
    print(f"  Kp={kp}  Kd={kd}")

    if sweep_hi - sweep_lo < 0.1:
        print("  ! Joint range too small to sweep.")
        return [], []

    all_avg: list[tuple[float, float]] = []
    all_halfdiff: list[tuple[float, float]] = []

    for v in velocities:
        print(f"\n  v = {v:.3f} rad/s ...")

        # Ramp to sweep start with time proportional to distance
        cur = await motor.get_position()
        ramp_dur = abs(sweep_lo - cur) / _RAMP_SPEED + 1.0
        await _ramp_to(motor, kp, kd, sweep_lo, duration=ramp_dur)
        await asyncio.sleep(0.3)

        # Forward sweep: sweep_lo → sweep_hi
        fwd = await _run_sweep_raw(motor, kp, kd, sweep_lo, +v, sweep_hi)
        cur = await motor.get_position()
        print(f"    fwd: {len(fwd)} samples")

        # Hold at turnaround to damp velocity before reversing
        await _ramp_to(motor, kp, kd, cur, duration=2.0)

        # Backward sweep: sweep_hi → sweep_lo
        bwd = await _run_sweep_raw(motor, kp, kd, cur, -v, sweep_lo)
        print(f"    bwd: {len(bwd)} samples")

        # Bin by position and match fwd/bwd
        fwd_bins = _bin_by_position(fwd, sweep_lo, sweep_hi)
        bwd_bins = _bin_by_position(bwd, sweep_lo, sweep_hi)
        matched = sum(1 for q in fwd_bins if q in bwd_bins)
        print(f"    {matched}/{_N_BINS} position bins matched")

        for q_center, tau_f in fwd_bins.items():
            if q_center in bwd_bins:
                tau_b = bwd_bins[q_center]
                all_avg.append((q_center, (tau_f + tau_b) / 2.0))
                all_halfdiff.append((v, (tau_f - tau_b) / 2.0))

        await asyncio.sleep(0.2)

    return all_avg, all_halfdiff


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "identify-feedforward",
        help="Identify all feedforward parameters (gravity + friction) in one pass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true")
    side.add_argument("--r", action="store_true")
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to identify: {', '.join(j.value for j in ARM_JOINTS)}",
    )
    p.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Proportional gain (default: from config)",
    )
    p.add_argument(
        "--kd", type=float, default=None, help="Derivative gain (default: from config)"
    )
    p.add_argument(
        "--velocities",
        type=float,
        nargs="+",
        default=DEFAULT_VELOCITIES,
        metavar="V",
        help="Velocity setpoints in rad/s (default: ~0.1 0.3 0.6 0.9 1.3 rad/s)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side_str = "left" if is_left else "right"
    config_gains = getattr(AxolConfig(), joint.value)
    kp = args.kp if args.kp is not None else config_gains.kp
    kd = args.kd if args.kd is not None else config_gains.kd

    print(f"\nAxol feedforward identification — {side_str} {joint.value}")
    print(f"  Velocity sweep: {[round(v, 3) for v in args.velocities]} rad/s")
    print(f"  Kp={kp}  Kd={kd}")

    channel = CAN_LEFT if is_left else CAN_RIGHT

    async with CanBus(channel) as bus:
        motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in motors.values()])
        await asyncio.gather(
            *[
                motors[j].set_control_mode(
                    ControlMode.MIT if j == joint else ControlMode.POS_VEL
                )
                for j in ARM_JOINTS
            ]
        )

        try:
            # wrist_2: elbow at midpoint of its range so wrist_2 can sweep
            # its full ±range without the forearm hitting the robot base.
            other_targets: dict[Joint, float] = {}
            if joint == Joint.WRIST_2:
                elbow_lo, elbow_hi = arm_limits(Joint.ELBOW, is_left)
                other_targets[Joint.ELBOW] = (elbow_lo + elbow_hi) / 2.0
                print(
                    f"  Moving elbow to {other_targets[Joint.ELBOW]:.3f} rad (midpoint of range) for wrist_2 clearance."
                )
            print("  Ramping other joints to start position ...")
            await _ramp_others(motors, joint, other_targets)
            await asyncio.sleep(1.0)

            avg_samples, halfdiff_samples = await _identify_joint(
                motors[joint], joint, kp, kd, is_left, args.velocities
            )

            if not avg_samples and not halfdiff_samples:
                print("\nNo samples collected.")
                return

            print(f"\n{'─' * 50}")
            print(f"  Avg samples:       {len(avg_samples)}")
            print(f"  Half-diff samples: {len(halfdiff_samples)}")

            gravity_result = _fit_gravity_with_offset(avg_samples)
            friction_result = _fit_friction_halfdiff(halfdiff_samples)

            ga_out = gb_out = Fo_out = Fc_out = k_out = Fv_out = 0.0

            if gravity_result is not None:
                ga_out, gb_out, Fo_out = gravity_result
                amplitude = math.sqrt(ga_out**2 + gb_out**2)
                phase_deg = math.degrees(math.atan2(gb_out, ga_out))
                print("\n  Fitted gravity model: τ = ga·cos(q) + gb·sin(q)")
                print(f"    ga = {ga_out:.4f} Nm")
                print(f"    gb = {gb_out:.4f} Nm")
                print(f"    amplitude = {amplitude:.4f} Nm  phase = {phase_deg:.1f}°")
                print(f"    Fo = {Fo_out:.4f} Nm  (offset, from avg fit)")

            if friction_result is not None:
                Fc_out, k_out, Fv_out = friction_result
                print("\n  Fitted friction model: τ = Fc·tanh(0.1·k·v) + Fv·v + Fo")
                print(f"    Fc = {Fc_out:.4f} Nm  (Coulomb)")
                print(f"    k  = {k_out:.2f}      (tanh steepness)")
                print(f"    Fv = {Fv_out:.4f} Nm·s/rad  (viscous)")

            if gravity_result is not None or friction_result is not None:
                print(f"\n  Add to config.py JointGains for {joint.value}:")
                print(
                    f"    fc={Fc_out:.4f}, k={k_out:.2f}, fv={Fv_out:.4f}, fo={Fo_out:.4f},"
                )
                print(f"    ga={ga_out:.4f}, gb={gb_out:.4f}")

            print(f"{'─' * 50}")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to 0 and disabling ...")
            try:
                await _ramp_to(motors[joint], kp, kd, 0.0, duration=4.0)
            except Exception:
                pass
            try:
                await _ramp_others(motors, joint)
            except Exception:
                pass
            await asyncio.gather(*[m.disable() for m in motors.values()])
