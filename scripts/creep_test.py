"""Creep test: does a MyActuator joint move smoothly at slow speed under its
*own* position loop (0xA4 absolute position closed-loop), where the MIT
impedance frame stick-slips?

Feasibility probe for moving the X8-P20 shoulders onto the firmware loop.
Streams a constant-velocity 0xA4 target at ``--rate`` Hz while reading the
fine multi-turn position (0x92, 0.01 deg/LSB) every cycle, then scores each
constant-speed segment: velocity ripple, lag, 1-4 Hz error band, and the
fraction of 50 ms windows in which the joint did not move at all (the
stick phases). No realtime core, no impedance control: the other joints
are parked in their firmware position holds exactly as the tuning probes
park them, and the arm is homed before and after.

The comparable MIT numbers (right shoulder_1 at ~0.05 rad/s, extended,
2026-09-18): velocity swinging 0 to 0.15 rad/s at ~2 Hz, 1-4 Hz error band
0.05-0.11 deg, stairs 0.1-0.6 deg.

Usage:
    uv run python scripts/creep_test.py --r --joint shoulder_1
    uv run python scripts/creep_test.py --r --joint shoulder_1 --center -35 --amp 10 --speeds 2 3 6 --csv ~/creep-s1.csv

The joint under test runs the whole time in firmware position control with
a speed cap (``--cap``, dps): it holds position stiffly and will push back
against contact up to the motor's limit. Keep the workspace clear.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import struct
import time
from pathlib import Path

import numpy as np

from almond_axol.cli.motor import add_side_and_channel_arguments, resolve_channel
from almond_axol.cli.tune.friction import _home_all, _ramp_verified
from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.motor import CanBus, ControlMode, Motor
from almond_axol.motor.myactuator import MyActuatorMotor
from almond_axol.tuning import joint_frame_motors, safe_limits

_MA_POS_CONTROL = 0xA4
_MA_MULTI_TURN_ANGLE = 0x92
_MA_READ_ACCEL = 0x42
_MA_WRITE_ACCEL = 0x43  # RAM + ROM; 0x00 = position-plan accel, 0x01 = decel
#: The firmware needs a moment after a 0x43 flash write before it answers again.
_ACCEL_WRITE_SETTLE_S = 0.3


async def _read_plan_accel(driver: MyActuatorMotor) -> tuple[int, int]:
    """(acceleration, deceleration) of the position planner, dps/s."""
    out = []
    for kind in (0x00, 0x01):
        resp = await driver._request(bytes([_MA_READ_ACCEL, kind, 0, 0, 0, 0, 0, 0]))
        out.append(struct.unpack_from("<i", resp, 4)[0])
    return out[0], out[1]


async def _write_plan_accel(driver: MyActuatorMotor, dps_s2: int) -> tuple[int, int]:
    """Write the position planner's accel and decel (raw, so 0 = direct
    tracking can be asked for) and return what the motor reads back."""
    for kind in (0x00, 0x01):
        await driver._request(
            bytes([_MA_WRITE_ACCEL, kind, 0, 0]) + struct.pack("<I", int(dps_s2))
        )
        await asyncio.sleep(_ACCEL_WRITE_SETTLE_S)
    return await _read_plan_accel(driver)


def _a4_frame(position_rad: float, cap_dps: float) -> bytes:
    """0xA4: uint16 speed cap (dps) + int32 target (0.01 deg)."""
    return (
        bytes([_MA_POS_CONTROL, 0x00])
        + struct.pack("<H", int(cap_dps))
        + struct.pack("<i", int(round(position_rad * 18000.0 / math.pi)))
    )


def _decode_a4_reply(resp: bytes) -> tuple[int, float, float, float]:
    """(temp C, iq A, speed rad/s, position rad) from a 0x240 control reply.
    Position is the coarse int16 degrees the reply carries (1 deg/LSB)."""
    temp = struct.unpack_from("<b", resp, 1)[0]
    iq = struct.unpack_from("<h", resp, 2)[0] * 0.01
    speed = math.radians(struct.unpack_from("<h", resp, 4)[0])
    pos = math.radians(struct.unpack_from("<h", resp, 6)[0])
    return temp, iq, speed, pos


def segment_metrics(
    t: np.ndarray, target: np.ndarray, pos: np.ndarray, v_cmd: float
) -> dict:
    """Score one constant-velocity cruise (joint frame, rad)."""
    if len(t) < 20:
        return {}
    dt = float(np.median(np.diff(t)))
    err = pos - target
    # Velocity from fine position, 50 ms box-averaged.
    win = max(1, int(round(0.05 / dt)))
    kern = np.ones(win) / win
    v = np.convolve(np.gradient(pos, t), kern, mode="same")
    core = slice(win, -win) if len(t) > 3 * win else slice(0, len(t))
    v_core = v[core]
    # Stick fraction: windows with < 0.03 deg of travel while the command moves.
    n_win = len(pos) // win
    moved = np.abs(
        np.diff(pos[: n_win * win].reshape(n_win, win)[:, [0, -1]], axis=1)
    ).ravel()
    stuck = float(np.mean(moved < math.radians(0.03))) if n_win else math.nan
    expected = abs(v_cmd) * win * dt
    stairs = float(np.degrees(np.max(moved) - expected)) if n_win else math.nan
    e = err - err.mean()
    n = len(e)
    F = np.abs(np.fft.rfft(e * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, dt)
    band = math.sqrt(F[(f >= 1) & (f <= 4)].sum() / max(F.sum(), 1e-30)) * float(
        e.std()
    )
    return {
        "v_cmd_dps": math.degrees(abs(v_cmd)),
        "lag_ms": float(-err.mean() / v_cmd * 1000.0) if v_cmd else math.nan,
        "err_rms_deg": float(np.degrees(err.std())),
        "band_1_4_deg": float(np.degrees(band)),
        "v_ripple": float(v_core.std() / max(abs(v_cmd), 1e-9)),
        "v_min_frac": float(np.min(v_core * np.sign(v_cmd)) / max(abs(v_cmd), 1e-9)),
        "stuck_frac": stuck,
        "max_stair_deg": stairs,
    }


async def _stream(
    driver: MyActuatorMotor,
    offset: float,
    start: float,
    end: float,
    v: float,
    cap_dps: float,
    rate: float,
    log: list,
    seg: int,
) -> None:
    """Stream a constant-velocity 0xA4 trajectory (joint frame) with a fine
    position read every cycle."""
    period = 1.0 / rate
    duration = abs(end - start) / v
    sign = 1.0 if end > start else -1.0
    t0 = time.perf_counter()
    deadline = t0
    while True:
        t = time.perf_counter() - t0
        if t > duration + 0.5:
            return
        deadline += period
        target = start + sign * v * min(t, duration)
        resp = await driver._request(_a4_frame(target - offset, cap_dps))
        temp, iq, speed, pos_coarse = _decode_a4_reply(resp)
        fine = await driver._request(bytes([_MA_MULTI_TURN_ANGLE, 0, 0, 0, 0, 0, 0, 0]))
        pos = struct.unpack_from("<i", fine, 4)[0] * (0.01 * math.pi / 180.0) + offset
        log.append(
            (
                time.perf_counter(),
                seg,
                sign * v if t <= duration else 0.0,
                target,
                pos,
                pos_coarse + offset,
                speed,
                iq,
                temp,
            )
        )
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    lo, hi = safe_limits(joint, is_left)
    center = math.radians(args.center)
    half = math.radians(args.amp) / 2.0
    margin = math.radians(3.0)
    if not (lo + margin <= center - half and center + half <= hi - margin):
        raise SystemExit(
            f"range {args.center - args.amp / 2:+.1f}..{args.center + args.amp / 2:+.1f}° "
            f"is outside the safe range [{math.degrees(lo) + 3:.1f}, {math.degrees(hi) - 3:.1f}]° "
            f"for {joint.value}"
        )
    speeds = [math.radians(s) for s in args.speeds]
    channel = resolve_channel(args)
    log: list = []
    print(
        f"\nCreep test — {'left' if is_left else 'right'} {joint.value}: 0xA4 direct tracking"
    )
    print(
        f"  range {args.center - args.amp / 2:+.1f}..{args.center + args.amp / 2:+.1f}°, "
        f"speeds {args.speeds} deg/s, cap {args.cap:g} dps, {args.rate:g} Hz"
    )
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
        driver = motors[joint].motor._driver
        if not isinstance(driver, MyActuatorMotor):
            raise SystemExit(f"{joint.value} is not a MyActuator joint")
        offset = motors[joint].offset
        original_accel: tuple[int, int] | None = None
        try:
            acc, dec = await _read_plan_accel(driver)
            print(f"  position planner: accel {acc} dps/s, decel {dec} dps/s (stored)")
            if args.accel is not None and (acc, dec) != (args.accel, args.accel):
                original_accel = (acc, dec)
                got = await _write_plan_accel(driver, args.accel)
                mode = "direct PI tracking" if got[0] == 0 else "velocity-profiled hops"
                print(
                    f"  position planner now: accel {got[0]}, decel {got[1]} dps/s → {mode}"
                )
                if got[0] != args.accel:
                    print(
                        f"  ! the motor did not accept {args.accel}; running with {got[0]}"
                    )
            elif acc == 0:
                print("  mode: direct PI tracking")
            else:
                print("  mode: velocity-profiled hops between streamed targets")
            print("  Homing all joints to rest ...")
            await _home_all(motors)
            start = center - half
            print(f"  Ramping {joint.value} to {math.degrees(start):+.1f}° ...")
            await _ramp_verified(motors, {joint: start})
            await asyncio.sleep(0.5)
            seg = 0
            here = start
            for v in speeds:
                for _ in range(args.passes):
                    there = center + half if here < center else center - half
                    print(
                        f"  segment {seg}: {math.degrees(here):+.1f} → {math.degrees(there):+.1f}° at {math.degrees(v):.1f} deg/s"
                    )
                    await _stream(
                        driver, offset, here, there, v, args.cap, args.rate, log, seg
                    )
                    here = there
                    seg += 1
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to rest and disabling ...")
            try:
                await _ramp_verified(motors, {joint: 0.0})
                await _home_all(motors)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            if original_accel is not None:
                try:
                    got = await _write_plan_accel(driver, original_accel[0])
                    print(
                        f"  position planner restored: accel {got[0]}, decel {got[1]} dps/s"
                    )
                except Exception:  # noqa: BLE001 - report, the value is in ROM
                    print(
                        f"  ! could not restore the planner acceleration ({original_accel[0]} "
                        "dps/s) — set it with `axol motor.set-config` before the next session"
                    )
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in raw.values()])

    if not log:
        print("No samples.")
        return
    arr = np.array(log, dtype=float)
    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "t",
                    "seg",
                    "v_cmd",
                    "target",
                    "pos_fine",
                    "pos_coarse",
                    "speed_motor",
                    "iq_a",
                    "temp_c",
                ]
            )
            w.writerows(arr.tolist())
        print(f"  samples → {args.csv}")
    print(
        f"\n{'seg':>3s} {'v dps':>6s} {'lag ms':>7s} {'err°':>6s} {'band1-4°':>8s} {'v ripple':>8s} {'v min':>6s} {'stuck':>6s} {'stair°':>7s}"
    )
    for seg in np.unique(arr[:, 1]).astype(int):
        s = (arr[:, 1] == seg) & (arr[:, 2] != 0.0)
        t = arr[s, 0]
        if len(t) < 20:
            continue
        cruise = t > t[0] + 0.3
        m = segment_metrics(
            t[cruise], arr[s, 3][cruise], arr[s, 4][cruise], float(arr[s, 2][0])
        )
        if m:
            print(
                f"{seg:3d} {m['v_cmd_dps']:6.1f} {m['lag_ms']:7.1f} {m['err_rms_deg']:6.3f} "
                f"{m['band_1_4_deg']:8.3f} {m['v_ripple']:8.2f} {m['v_min_frac']:6.2f} "
                f"{m['stuck_frac']:6.2f} {m['max_stair_deg']:7.3f}"
            )
    print(
        "\n  v ripple = std(velocity)/|v_cmd| (MIT stick-slip ≈ 1.0, smooth < 0.2); "
        "v min = slowest 50 ms window over |v_cmd| (0 = it stopped);\n"
        "  stuck = fraction of 50 ms windows with < 0.03° travel; stair = largest window travel beyond the commanded amount."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint", default="shoulder_1", choices=[j.value for j in ARM_JOINTS]
    )
    p.add_argument(
        "--center",
        type=float,
        default=-35.0,
        help="Centre of the creep range, joint-frame degrees (default: -35)",
    )
    p.add_argument(
        "--amp",
        type=float,
        default=10.0,
        help="Total travel per pass, degrees (default: 10)",
    )
    p.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 6.0],
        help="Creep speeds, deg/s (default: 2 3 6)",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Passes per speed, alternating direction (default: 2)",
    )
    p.add_argument(
        "--cap", type=float, default=45.0, help="0xA4 speed cap, dps (default: 45)"
    )
    p.add_argument(
        "--rate", type=float, default=200.0, help="Command rate, Hz (default: 200)"
    )
    p.add_argument(
        "--accel",
        type=int,
        default=None,
        help="Position-planner acceleration (dps/s) to run with: 0 asks for direct PI "
        "tracking, otherwise velocity-profiled hops. Written to the motor for the run "
        "and restored afterwards (default: leave the stored value)",
    )
    p.add_argument("--csv", type=Path, default=None, help="Write every sample here")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
