"""
axol tune.friction

Identify the four friction-model parameters (Fc, k, Fv, Fo) for an Axol joint
in a single bidirectional sweep.

Sweeps the full joint range at multiple velocities, both forward and backward.
Bidirectional averaging at the same position separates gravity from friction:

    avg(τ_fwd, τ_bwd) at same q     →  gravity(q) + Fo
    half(τ_fwd - τ_bwd) at same q   →  Fc·tanh(0.1·k·v) + Fv·v

The half-difference (the part that flips sign with velocity) is fit to the
friction model and yields ``Fc``, ``k``, and ``Fv``. The average is then
compared against the URDF gravity model (see
:class:`almond_axol.robot.gravity.GravityCompensator`) — the constant residual
becomes ``Fo`` and the shape residual is reported as a sanity check on the
``mass`` / ``com`` values in :class:`JointConfig`.

At runtime:
    tff(q, v) = gravity_model(q) + Fc·tanh(0.1·k·v) + Fv·v + Fo

Examples:
    axol tune.friction --l --joint shoulder_1 --kp 30 --kd 0.8
    axol tune.friction --r --joint elbow --kp 20 --kd 0.6
    axol tune.friction --l --joint wrist_1 --velocities 0.2 0.6 1.0
"""

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from ...motor import CanBus, ControlMode, Joint, Motor
from ...robot.axol import arm_limits
from ...robot.config import ArmConfig, AxolConfig
from ...robot.gravity import GravityCompensator
from ...shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT

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
        await motor.set_impedance(
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
        *[motors[j].set_position_velocity(t.get(j, 0.0), _RAMP_SPEED) for j in joints]
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
) -> list[tuple[float, float]]:
    """Sweep from start_pos to end_pos at constant velocity.

    Returns list of ``(q_actual, tau_measured)``. The first
    ``WARMUP_FRACTION`` of travel is discarded for motor settling.
    ``tau_measured`` is read from the motor's feedback frame (motor-side
    torque estimate), **not** computed from host-side ``kp·pos_err +
    kd·vel_err`` — host-side numerical differentiation of position is
    too noisy at the velocities we sweep, and the motor's own estimate is
    already what we want.
    """
    travel = abs(end_pos - start_pos)
    if travel < 0.02:
        return []
    total_time = travel / abs(velocity_rad_s)
    warmup_time = total_time * _WARMUP_FRACTION
    dt = 1.0 / _RATE_HZ

    samples: list[tuple[float, float]] = []

    t0 = time.monotonic()
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= total_time:
            break
        loop_start = now

        target = start_pos + velocity_rad_s * t
        # set_impedance returns a feedback frame, which updates
        # motor.position / motor.torque via the driver _on_feedback hook.
        await motor.set_impedance(target, velocity_rad_s, kp, kd, 0.0)

        if t >= warmup_time:
            samples.append((motor.position, motor.torque))

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    return samples


def _bin_by_position(
    samples: list[tuple[float, float]],
    sweep_lo: float,
    sweep_hi: float,
    n_bins: int = _N_BINS,
) -> dict[float, float]:
    """Return {bin_center: mean_tau} from (q, tau) samples."""
    span = sweep_hi - sweep_lo
    if span <= 0:
        return {}
    bin_width = span / n_bins
    buckets: dict[int, list[float]] = {}
    for q, tau in samples:
        idx = int((q - sweep_lo) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        buckets.setdefault(idx, []).append(tau)
    return {
        sweep_lo + (idx + 0.5) * bin_width: float(np.mean(taus))
        for idx, taus in buckets.items()
        if len(taus) >= 2
    }


def _compare_to_gravity_model(
    avg_samples: list[tuple[float, float]],
    joint: Joint,
    is_left: bool,
    other_targets: dict[Joint, float],
) -> float | None:
    """Compare ``tau_avg`` to the URDF gravity model and return ``Fo``.

    For each ``(q, tau_avg)`` sample, predicts gravity at the pose used during
    the sweep — test joint at ``q``, other arm joints at the targets they
    were ramped to — and reports the residual. The mean residual is the
    constant friction offset ``Fo``; the residual after removing the bias is
    a sanity check on the per-link ``mass`` / ``com`` values.
    """
    if len(avg_samples) < 5:
        print("  ! Too few avg samples to compare against gravity model.")
        return None

    gc = GravityCompensator()
    test_idx = ARM_JOINTS.index(joint)
    arm_q_buf = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for j, target in other_targets.items():
        if j in ARM_JOINTS and j != joint:
            arm_q_buf[ARM_JOINTS.index(j)] = float(target)

    measured = np.array([s[1] for s in avg_samples], dtype=np.float64)
    predicted = np.empty_like(measured)
    for i, (q, _) in enumerate(avg_samples):
        arm_q_buf[test_idx] = float(q)
        predicted[i] = float(gc.gravity_arm(arm_q_buf, is_left=is_left)[test_idx])

    residual = measured - predicted
    Fo = float(np.mean(residual))
    rms_after_bias = float(np.sqrt(np.mean((residual - Fo) ** 2)))
    rms_total = float(np.sqrt(np.mean(residual**2)))

    pred_rms = float(np.sqrt(np.mean(predicted**2)))
    pred_peak = float(np.max(np.abs(predicted)))

    print("\n  URDF gravity check (mass/com from JointConfig):")
    print(f"    Predicted gravity : {pred_rms:.4f} Nm RMS  (peak {pred_peak:.4f} Nm)")
    print(f"    Fo                : {Fo:+.4f} Nm  (mean residual → FrictionParams.fo)")
    # Joints with a vertical rotation axis (e.g. shoulder_3, wrist_3 at q≈0)
    # see no gravity moment, so the relative-error metric is meaningless;
    # report the absolute residual without a percentage or warning.
    if pred_rms < 0.1:
        print(
            f"    Shape residual    : {rms_after_bias:.4f} Nm RMS  "
            "(no gravity dependence at this pose — abs residual is just noise)"
        )
        print(f"    Total residual    : {rms_total:.4f} Nm RMS")
    else:
        pct = rms_after_bias / pred_rms * 100.0
        print(
            f"    Shape residual    : {rms_after_bias:.4f} Nm RMS  ({pct:.1f}% of predicted)"
        )
        print(f"    Total residual    : {rms_total:.4f} Nm RMS")
        if pct > 20.0 and rms_after_bias > 0.1:
            print(
                "    ! Large shape residual: URDF mass/com likely off for this "
                "joint or its children. Verify with `axol gravity-comp`."
            )
    return Fo


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
    lo_override: float | None = None,
    hi_override: float | None = None,
    dump_csv: Path | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Run bidirectional multi-velocity sweep over the full joint range.

    Returns:
        avg_samples:      (q, tau_avg)  — for gravity+Fo fitting
        halfdiff_samples: (v, tau_half) — for Fc/k/Fv fitting

    If ``dump_csv`` is given, every matched (fwd, bwd) bin is also written to
    a CSV with the per-velocity, per-position torque values. Useful for
    plotting the raw friction-vs-velocity curve and comparing arms.
    """
    lo, hi = arm_limits(joint, is_left)
    if lo_override is not None:
        lo = lo_override
    if hi_override is not None:
        hi = hi_override
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

    csv_file = None
    csv_writer = None
    if dump_csv is not None:
        csv_file = dump_csv.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "joint",
                "side",
                "v_rad_s",
                "q_rad",
                "tau_fwd_nm",
                "tau_bwd_nm",
                "tau_avg_nm",
                "tau_halfdiff_nm",
            ]
        )
        print(f"  Dumping per-bin samples to {dump_csv}")

    try:
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
                    tau_avg = (tau_f + tau_b) / 2.0
                    tau_half = (tau_f - tau_b) / 2.0
                    all_avg.append((q_center, tau_avg))
                    all_halfdiff.append((v, tau_half))
                    if csv_writer is not None:
                        csv_writer.writerow(
                            [
                                joint.value,
                                "left" if is_left else "right",
                                f"{v:.6f}",
                                f"{q_center:.6f}",
                                f"{tau_f:.6f}",
                                f"{tau_b:.6f}",
                                f"{tau_avg:.6f}",
                                f"{tau_half:.6f}",
                            ]
                        )

            if csv_file is not None:
                csv_file.flush()

            await asyncio.sleep(0.2)
    finally:
        if csv_file is not None:
            csv_file.close()

    return all_avg, all_halfdiff


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "tune.friction",
        help="Identify the friction-model parameters (Fc, k, Fv, Fo) for one joint.",
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
    p.add_argument(
        "--lo",
        type=float,
        default=None,
        metavar="RAD",
        help="Override lower joint limit for the sweep (rad)",
    )
    p.add_argument(
        "--hi",
        type=float,
        default=None,
        metavar="RAD",
        help="Override upper joint limit for the sweep (rad)",
    )
    p.add_argument(
        "--dump-csv",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help="Write per-bin (v, q, tau_fwd, tau_bwd, tau_avg, tau_halfdiff) "
        "rows to a CSV for offline plotting / arm-vs-arm comparison. Pass "
        "without a value to auto-name as "
        "logs/friction_<side>_<joint>_<timestamp>.csv, or pass an explicit "
        "path.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side_str = "left" if is_left else "right"
    arm_cfg: ArmConfig = AxolConfig().left if is_left else AxolConfig().right
    config_gains = getattr(arm_cfg, joint.value)
    kp = args.kp if args.kp is not None else config_gains.kp
    kd = args.kd if args.kd is not None else config_gains.kd

    dump_csv: Path | None = None
    if args.dump_csv == "__auto__":
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_csv = Path("logs") / f"friction_{side_str}_{joint.value}_{timestamp}.csv"
    elif args.dump_csv is not None:
        dump_csv = Path(args.dump_csv)
    if dump_csv is not None:
        dump_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nAxol friction identification — {side_str} {joint.value}")
    print(f"  Velocity sweep: {[round(v, 3) for v in args.velocities]} rad/s")
    print(f"  Kp={kp}  Kd={kd}")

    channel = CAN_LEFT if is_left else CAN_RIGHT

    async with CanBus(channel) as bus:
        motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in motors.values()])
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

            # shoulder_2 swings into the robot base on the inboard side; cap
            # the sweep at 0 so it stays on the safe half of its range.
            lo_default = hi_default = None
            if joint == Joint.SHOULDER_2:
                if is_left:
                    hi_default = 0.0
                else:
                    lo_default = 0.0
                print("  Capping shoulder_2 sweep at 0 rad to avoid the base.")

            avg_samples, halfdiff_samples = await _identify_joint(
                motors[joint],
                joint,
                kp,
                kd,
                is_left,
                args.velocities,
                lo_override=args.lo if args.lo is not None else lo_default,
                hi_override=args.hi if args.hi is not None else hi_default,
                dump_csv=dump_csv,
            )

            if not avg_samples and not halfdiff_samples:
                print("\nNo samples collected.")
                return

            print(f"\n{'─' * 50}")
            print(f"  Avg samples:       {len(avg_samples)}")
            print(f"  Half-diff samples: {len(halfdiff_samples)}")

            Fo_result = _compare_to_gravity_model(
                avg_samples, joint, is_left, other_targets
            )
            friction_result = _fit_friction_halfdiff(halfdiff_samples)

            Fo_out = Fo_result if Fo_result is not None else 0.0
            Fc_out = k_out = Fv_out = 0.0

            if friction_result is not None:
                Fc_out, k_out, Fv_out = friction_result
                print("\n  Fitted friction model: τ = Fc·tanh(0.1·k·v) + Fv·v + Fo")
                print(f"    Fc = {Fc_out:.4f} Nm  (Coulomb)")
                print(f"    k  = {k_out:.2f}      (tanh steepness)")
                print(f"    Fv = {Fv_out:.4f} Nm·s/rad  (viscous)")

            if friction_result is not None or Fo_result is not None:
                print(f"\n  Add to config.py JointConfig.{joint.value}.friction:")
                print(
                    f"    FrictionParams(fc={Fc_out:.4f}, k={k_out:.2f}, "
                    f"fv={Fv_out:.4f}, fo={Fo_out:.4f}),"
                )

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
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])
