"""
almond-axol tune-pid

Tune Kp/Kd for a single Axol joint at ~100 Hz.

Tests gains via sinusoidal or step-response tracking and measures error (RMS, max,
overshoot). Results are printed to stdout.

Examples:
    almond-axol tune-pid --left  --joint elbow      --kp 25 --kd 0.6
    almond-axol tune-pid --right --joint shoulder_1 --kp 35 --kd 1.2 --mode step
    almond-axol tune-pid --left  --joint wrist_1    --kp 12 --kd 0.4 --freq 2
    almond-axol tune-pid --left  --joint wrist_2    --kp 10 --kd 0.3 --mode step
"""

import argparse
import asyncio
import math
import time

import numpy as np

from ..motor import CanBus, Joint, Motor
from ..robot.axol import arm_limits
from ..robot.config import AxolConfig
from ..robot.control import compute_friction
from ..shared import ARM_JOINTS, CAN_LEFT, CAN_RIGHT

_DEFAULT_AMP_FRACTION = 0.3
_RAMP_MAX_SPEED = (
    4 * math.pi
)  # rad/s — generous cap; actual speed is interpolation-limited


def _sine_center(joint: Joint, is_left: bool) -> float:
    lo, hi = arm_limits(joint, is_left)
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
    default_amp = headroom * _DEFAULT_AMP_FRACTION
    if requested is not None:
        amp = min(requested, headroom)
        if amp < requested:
            print(
                f"  ! requested amp {requested:.4f} rad exceeds headroom; clamped to {amp:.4f} rad"
            )
    else:
        amp = default_amp
    return amp


async def _ramp_all(
    motors: dict[Joint, Motor],
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    duration: float,
    rate_hz: float,
) -> None:
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / duration, 1.0)
        targets = from_pos + alpha * (to_pos - from_pos)
        await asyncio.gather(
            *[
                motors[j].set_position(float(targets[i]), _RAMP_MAX_SPEED)
                for i, j in enumerate(ARM_JOINTS)
            ]
        )
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def run_sine(
    motors: dict[Joint, Motor],
    joint: Joint,
    kp: float,
    kd: float,
    freq: float,
    requested_amp: float | None,
    duration: float,
    rate_hz: float,
    is_left: bool,
    fc: float = 0.0,
    k: float = 0.0,
    fv: float = 0.0,
    fo: float = 0.0,
) -> tuple[list[dict], float]:
    test_motor = motors[joint]
    lo, hi = arm_limits(joint, is_left)
    center = _sine_center(joint, is_left)
    amp = _safe_amplitude(joint, is_left, center, requested_amp)
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"amp=±{amp:.4f} rad  freq={freq:.2f} Hz"
    )

    print("  moving to center ...")
    start_rad = await test_motor.get_position()
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / 2.0, 1.0)
        await test_motor.motion_control(
            start_rad + alpha * (center - start_rad), 0.0, kp, kd, 0.0
        )
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)
    await asyncio.sleep(1.0)

    print(f"  running {duration:.1f} s at {rate_hz:.0f} Hz ...")
    dt = 1.0 / rate_hz
    log: list[dict] = []
    start = time.monotonic()

    while True:
        t = time.monotonic() - start
        if t >= duration:
            break
        loop_start = time.monotonic()

        v_des = amp * 2 * math.pi * freq * math.cos(2 * math.pi * freq * t)
        target = center + amp * math.sin(2 * math.pi * freq * t)
        tff = compute_friction(v_des, fc, k, fv, fo)
        await test_motor.motion_control(target, v_des, kp, kd, tff)
        actual = await test_motor.get_position()
        log.append(
            {
                "t": round(t, 5),
                "target": target,
                "actual": actual,
                "error": actual - target,
            }
        )

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    return log, amp


async def run_step(
    motors: dict[Joint, Motor],
    joint: Joint,
    kp: float,
    kd: float,
    requested_amp: float | None,
    hold: float,
    rate_hz: float,
    is_left: bool,
    fc: float = 0.0,
    k: float = 0.0,
    fv: float = 0.0,
    fo: float = 0.0,
) -> tuple[list[dict], float]:
    test_motor = motors[joint]
    center = await test_motor.get_position()
    lo, hi = arm_limits(joint, is_left)

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
        amp = headroom * _DEFAULT_AMP_FRACTION

    step_target = center + direction * amp
    sign_str = f"+{amp:.4f}" if direction == 1 else f"-{amp:.4f}"
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"step={sign_str} rad  hold={hold:.1f} s  rate={rate_hz:.0f} Hz"
    )

    dt = 1.0 / rate_hz
    log: list[dict] = []
    start = time.monotonic()

    for phase_target in [step_target, center]:
        phase_start = time.monotonic()
        while time.monotonic() - phase_start < hold:
            loop_start = time.monotonic()
            t = time.monotonic() - start
            tff = compute_friction(0.0, fc, k, fv, fo)
            await test_motor.motion_control(phase_target, 0.0, kp, kd, tff)
            actual = await test_motor.get_position()
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


def _print_stats_sine(log: list[dict], kp: float, kd: float) -> None:
    errors = [r["error"] for r in log]
    rms = math.sqrt(sum(e**2 for e in errors) / len(errors))
    max_err = max(abs(e) for e in errors)
    elapsed = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 1.0
    actual_hz = len(log) / elapsed if elapsed > 0 else 0
    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {len(log)}  ({actual_hz:.1f} Hz actual)")
    print(f"  RMS error:  {rms:.5f} rad  ({math.degrees(rms):.3f}°)")
    print(f"  Max error:  {max_err:.5f} rad  ({math.degrees(max_err):.3f}°)")
    print(f"{'─' * 40}")


def _print_stats_step(log: list[dict], amp: float, kp: float, kd: float) -> None:
    targets = list(dict.fromkeys(r["target"] for r in log))
    step_target = targets[0]
    step_rows = [r for r in log if r["target"] == step_target]
    direction = 1 if step_target > targets[1] else -1
    real_overshoot = max(
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
    actual_hz = len(log) / elapsed if elapsed > 0 else 0
    settling = f"{settling_s * 1000:.0f} ms" if settling_s is not None else ">hold time"

    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {len(log)}  ({actual_hz:.1f} Hz actual)")
    print(f"  Settling:   {settling}  (5% threshold)")
    print(
        f"  Overshoot:  {math.degrees(real_overshoot):.3f}°  "
        f"({real_overshoot / amp * 100 if amp > 0 else 0:.1f}% of step)"
    )
    print(f"  SS RMS:     {ss_rms:.5f} rad  ({math.degrees(ss_rms):.3f}°)")
    print(f"{'─' * 40}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "tune-pid",
        help="Tune Kp/Kd for a single Axol joint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    side = p.add_mutually_exclusive_group(required=True)
    side.add_argument("--left", action="store_true", help="Left arm")
    side.add_argument("--right", action="store_true", help="Right arm")
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to tune: {', '.join(j.value for j in ARM_JOINTS)}",
    )
    p.add_argument("--kp", type=float, required=True, help="Proportional gain to test")
    p.add_argument("--kd", type=float, required=True, help="Derivative gain to test")
    p.add_argument(
        "--friction",
        action="store_true",
        help="Apply friction feedforward from AxolConfig",
    )
    p.add_argument(
        "--mode",
        choices=["sine", "step"],
        default="sine",
        help="sine (default): continuous tracking; step: step response",
    )
    p.add_argument(
        "--amp",
        type=float,
        default=None,
        help="Motion amplitude in rad (default: joint safe value)",
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
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.left
    side_str = "left" if is_left else "right"
    lo, hi = arm_limits(joint, is_left)

    joint_gains = getattr(AxolConfig(), joint.value)
    fc, k, fv, fo = (
        (joint_gains.fc, joint_gains.k, joint_gains.fv, joint_gains.fo)
        if args.friction
        else (0.0, 0.0, 0.0, 0.0)
    )

    print(
        f"\nAxol PID tuner — {side_str} {joint.value}  limits=[{lo:.4f}, {hi:.4f}] rad"
    )
    print(f"  testing  Kp={args.kp}  Kd={args.kd}  mode={args.mode}")
    if args.friction:
        print(f"  friction  Fc={fc}  k={k}  Fv={fv}  Fo={fo}")

    channel = CAN_LEFT if is_left else CAN_RIGHT

    async with CanBus(channel) as bus:
        motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in motors.values()])

        try:
            print("  ramping all joints to 0 ...")
            pos_vals = await asyncio.gather(
                *[motors[j].get_position() for j in ARM_JOINTS]
            )
            start_positions = np.array(pos_vals, dtype=float)
            await _ramp_all(
                motors,
                start_positions,
                np.zeros_like(start_positions),
                duration=2.0,
                rate_hz=args.rate,
            )
            await asyncio.sleep(1.0)

            if args.mode == "sine":
                log, amp = await run_sine(
                    motors,
                    joint,
                    args.kp,
                    args.kd,
                    args.freq,
                    args.amp,
                    args.duration,
                    args.rate,
                    is_left,
                    fc=fc,
                    k=k,
                    fv=fv,
                    fo=fo,
                )
                _print_stats_sine(log, args.kp, args.kd)
            else:
                log, amp = await run_step(
                    motors,
                    joint,
                    args.kp,
                    args.kd,
                    args.amp,
                    args.hold,
                    args.rate,
                    is_left,
                    fc=fc,
                    k=k,
                    fv=fv,
                    fo=fo,
                )
                _print_stats_step(log, amp, args.kp, args.kd)

        except KeyboardInterrupt:
            print("\n  interrupted")
        finally:
            print("  returning to 0 ...")
            test_motor = motors[joint]
            try:
                start_rad = await test_motor.get_position()
                dt = 1.0 / 100.0
                t0 = time.monotonic()
                while True:
                    t = time.monotonic() - t0
                    alpha = min(t / 2.0, 1.0)
                    loop_start = time.monotonic()
                    await test_motor.motion_control(
                        start_rad * (1.0 - alpha), 0.0, args.kp, args.kd, 0.0
                    )
                    if alpha >= 1.0:
                        break
                    spent = time.monotonic() - loop_start
                    if spent < dt:
                        await asyncio.sleep(dt - spent)
                await asyncio.sleep(1.0)
            except Exception:
                pass
            await asyncio.gather(*[m.disable() for m in motors.values()])
