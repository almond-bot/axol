"""
almond-axol identify-friction

Identify friction parameters for an Axol arm joint.

Sweeps the joint through constant velocities in both directions at position=0
(gravity-free point). Estimates friction from the PD tracking error:

    τ_friction = Kp * (pos_cmd - pos_actual) + Kd * (vel_cmd - vel_actual)

Then fits the tanh model: τ = Fc*tanh(0.1*k*v) + Fv*v + Fo

Bidirectional averaging cancels gravity bias in Fo for joints that can travel
both directions from 0. Elbow [0, ~2.58 rad] can only travel forward; its Fo
will include any residual gravity at the measurement positions.

Run tune-pid first to find good Kp/Kd values before using this command.

Examples:
    almond-axol identify-friction --left  --joint shoulder_1 --kp 30 --kd 0.8
    almond-axol identify-friction --right --joint elbow      --kp 20 --kd 0.6
    almond-axol identify-friction --left  --joint wrist_2    --kp 10 --kd 0.4 --velocities 0.1 0.3 0.6 1.2
"""

import argparse
import asyncio
import math
import time

import numpy as np
from scipy.optimize import curve_fit

from ..motor import CanBus, Joint, Motor
from ..robot.axol import arm_limits
from ..shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT

_TAU = 2 * math.pi

# Default velocity sweep in rad/s (~0.02, 0.05, 0.1, 0.15, 0.2 rev/s)
DEFAULT_VELOCITIES = [v * _TAU for v in [0.02, 0.05, 0.1, 0.15, 0.2]]

# Max travel per pass in rad (≈ 0.15 rev), keeps joint near 0 for gravity-free measurement
MAX_TRAVEL_RAD = 0.15 * _TAU

# Fraction of each pass to skip before collecting samples (motor settling)
WARMUP_FRACTION = 0.35

RATE_HZ = 100.0


async def _ramp_to(
    motor: Motor,
    kp: float,
    kd: float,
    target: float,
    rate_hz: float = RATE_HZ,
    duration: float = 2.0,
) -> None:
    start_pos = await motor.get_position()
    dt = 1.0 / rate_hz
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


async def _run_constant_velocity_pass(
    motor: Motor,
    kp: float,
    kd: float,
    start_pos: float,
    velocity_rad_s: float,
    travel_rad: float,
    rate_hz: float,
) -> list[tuple[float, float]]:
    """
    Command a constant-velocity ramp and collect (vel_actual_rad_s, torque_est_Nm) samples.

    Friction is estimated from the PD correction:
        τ_friction = Kp * pos_err + Kd * vel_err

    The first WARMUP_FRACTION of travel is discarded for motor settling.
    """
    dt = 1.0 / rate_hz
    total_time = travel_rad / abs(velocity_rad_s)
    warmup_time = total_time * WARMUP_FRACTION

    samples: list[tuple[float, float]] = []
    pos_actual_prev: float | None = None
    t_prev: float | None = None

    t0 = time.monotonic()
    while True:
        now = time.monotonic()
        t = now - t0
        if t >= total_time:
            break
        loop_start = now

        target = start_pos + velocity_rad_s * t
        await motor.motion_control(target, 0.0, kp, kd, 0.0)
        pos_actual = await motor.get_position()

        if t >= warmup_time and pos_actual_prev is not None and t_prev is not None:
            pos_err = target - pos_actual
            dt_actual = now - t_prev
            vel_actual = (
                (pos_actual - pos_actual_prev) / dt_actual if dt_actual > 0 else 0.0
            )
            vel_err = velocity_rad_s - vel_actual
            torque_est = kp * pos_err + kd * vel_err
            samples.append((vel_actual, torque_est))

        pos_actual_prev = pos_actual
        t_prev = now

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    return samples


def _tanh_model(v: np.ndarray, Fc: float, k: float, Fv: float, Fo: float) -> np.ndarray:
    return Fc * np.tanh(0.1 * k * v) + Fv * v + Fo


def _fit_friction_model(
    samples: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if len(samples) < 10:
        print("  ! Too few samples to fit model.")
        return None

    v_arr = np.array([s[0] for s in samples])
    t_arr = np.array([s[1] for s in samples])

    pos_mask = v_arr > 0
    neg_mask = v_arr < 0
    Fo_guess = float(np.mean(t_arr))
    Fc_guess = 0.1
    Fv_guess = 0.02
    k_guess = 10.0

    if pos_mask.any() and neg_mask.any():
        Fo_guess = float((np.mean(t_arr[pos_mask]) + np.mean(t_arr[neg_mask])) / 2)
        Fc_guess = float((np.mean(t_arr[pos_mask]) - np.mean(t_arr[neg_mask])) / 2)

    try:
        popt, _ = curve_fit(
            _tanh_model,
            v_arr,
            t_arr,
            p0=[Fc_guess, k_guess, Fv_guess, Fo_guess],
            bounds=([0, 0.1, 0, -1.0], [10.0, 1000.0, 5.0, 1.0]),
            maxfev=10000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    except Exception as e:
        print(f"  ! Fit failed: {e}")
        return None


async def _identify_joint(
    motor: Motor,
    joint: Joint,
    kp: float,
    kd: float,
    is_left: bool,
    velocities: list[float],
) -> list[tuple[float, float]]:
    lo, hi = arm_limits(joint, is_left)
    headroom_fwd = hi - 0.0
    headroom_bwd = 0.0 - lo

    print(f"\n  Joint limits: [{lo:.4f}, {hi:.4f}] rad")
    print(f"  Headroom from 0: +{headroom_fwd:.4f} / -{headroom_bwd:.4f} rad")
    print(f"  Kp={kp}  Kd={kd}")
    if headroom_bwd < 0.06:
        print(
            "  ! No backward headroom from 0 — Fo will include residual gravity bias."
        )

    print("\n  Ramping to position 0 ...")
    await _ramp_to(motor, kp, kd, 0.0)
    await asyncio.sleep(0.5)

    all_samples: list[tuple[float, float]] = []

    for v in velocities:
        travel_fwd = min(MAX_TRAVEL_RAD, headroom_fwd - 0.12)
        travel_bwd = min(MAX_TRAVEL_RAD, headroom_bwd - 0.12)

        if travel_fwd < 0.06:
            print(f"  v={v:.3f} rad/s: no forward headroom, skipping")
            continue

        print(f"\n  v = ±{v:.3f} rad/s ...")

        fwd_samples = await _run_constant_velocity_pass(
            motor,
            kp,
            kd,
            start_pos=0.0,
            velocity_rad_s=+v,
            travel_rad=travel_fwd,
            rate_hz=RATE_HZ,
        )
        all_samples.extend(fwd_samples)
        fwd_pos = await motor.get_position()

        if travel_bwd > 0.06:
            bwd_samples = await _run_constant_velocity_pass(
                motor,
                kp,
                kd,
                start_pos=fwd_pos,
                velocity_rad_s=-v,
                travel_rad=fwd_pos + travel_bwd,
                rate_hz=RATE_HZ,
            )
            all_samples.extend(bwd_samples)
            await _ramp_to(motor, kp, kd, 0.0, duration=1.5)
        else:
            await _ramp_to(motor, kp, kd, 0.0, duration=1.5)

        await asyncio.sleep(0.3)
        n_fwd = len(fwd_samples)
        print(
            f"    {n_fwd} fwd samples, {len(all_samples) - n_fwd} total bwd samples so far"
        )

    print("\n  Returning to 0 ...")
    await _ramp_to(motor, kp, kd, 0.0)

    return all_samples


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "identify-friction",
        help="Identify friction parameters for an Axol joint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--left", action="store_true")
    side.add_argument("--right", action="store_true")
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to identify: {', '.join(j.value for j in ARM_JOINTS)}",
    )
    p.add_argument(
        "--kp", type=float, required=True, help="Proportional gain (from tune-pid)"
    )
    p.add_argument(
        "--kd", type=float, required=True, help="Derivative gain (from tune-pid)"
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
    is_left = args.left
    side_str = "left" if is_left else "right"
    kp, kd = args.kp, args.kd

    print(f"\nAxol friction identification — {side_str} {joint.value}")
    print(f"  Velocity sweep: {[round(v, 3) for v in args.velocities]} rad/s")
    print(f"  Max travel per pass: {MAX_TRAVEL_RAD:.4f} rad")
    print(f"  Kp={kp}  Kd={kd}")
    print("  Gravity is zero at position=0 for this joint.")

    channel = CAN_LEFT if is_left else CAN_RIGHT

    async with CanBus(channel) as bus:
        motor = Motor(bus, joint)
        await motor.enable()

        try:
            samples = await _identify_joint(
                motor, joint, kp, kd, is_left, args.velocities
            )

            if not samples:
                print("\nNo samples collected.")
                return

            print(f"\n{'─' * 50}")
            print(f"  Total samples: {len(samples)}")

            v_all = [s[0] for s in samples]
            t_all = [s[1] for s in samples]
            print(f"  Velocity range: [{min(v_all):.3f}, {max(v_all):.3f}] rad/s")
            print(f"  Torque range:   [{min(t_all):.4f}, {max(t_all):.4f}] Nm")

            result = _fit_friction_model(samples)

            if result is not None:
                Fc, k, Fv, Fo = result
                print("\n  Fitted friction model:")
                print(f"    Fc = {Fc:.4f} Nm  (Coulomb)")
                print(f"    k  = {k:.2f}      (tanh steepness)")
                print(f"    Fv = {Fv:.4f} Nm·s/rad  (viscous)")
                print(f"    Fo = {Fo:.4f} Nm  (offset bias)")
                print(
                    f"\n  Model: τ = {Fc:.4f}·tanh(0.1·{k:.2f}·v) + {Fv:.4f}·v + {Fo:.4f}"
                )
            else:
                print("\n  Fitting failed — raw data sample:")
                for v, t in samples[:: max(1, len(samples) // 20)]:
                    print(f"    v={v:+.3f} rad/s  τ={t:+.4f} Nm")

            print(f"{'─' * 50}")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to 0 and disabling ...")
            try:
                await _ramp_to(motor, kp, kd, 0.0, duration=1.5)
            except Exception:
                pass
            await motor.disable()
