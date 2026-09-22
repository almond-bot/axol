"""
axol tune.a4

Tune a MyActuator joint's **firmware position loop** (0xA4 absolute position
closed-loop) — the loop the realtime core drives when a joint's
``wire_mode`` is ``a4``. Streams a sine or constant-speed triangle target at
``--rate`` Hz with the firmware gains, speed cap and planner acceleration you
choose, reads the fine 0.01° position (0x92) every cycle, scores tracking and
smoothness, and saves the run for the diagnostics dashboard.

Planner acceleration: ``0`` is the documented direct-tracking mode, and the
protocol maximum ``60000`` makes the planner finish each 200 Hz step inside
the tick — on the X6-P20 elbow the latter tracked a 3 deg/s triangle to
0.02° RMS with 4 ms lag against 0.23° / 74 ms for direct tracking. Values in
between re-plan every target and the joint barely moves. The value is
written *before* the mode-switch reset: on the elbow's 2025070202 firmware
a 0 written into a running position loop is silently ignored (the joint
holds and executes nothing), while the same 0 applied through the reset
works; non-zero values apply live on every firmware seen.

Why a separate tool: ``tune.pid`` tunes the MIT impedance frame, whose gains
live in the host. Under 0xA4 the whole controller is the motor's own
position PI → speed PI → current loop, so the knobs are the firmware gains
(``position_kp/ki/kd``, ``speed_kp/ki``, ``current_kp/ki``), the 0xA4 speed
cap, and the position planner's acceleration (0 = direct PI tracking of the
stream; anything else re-plans every target and will not follow a stream).

Safety:

* Gains are written to **RAM** (0x31) unless ``--persist`` is given, and the
  pre-run values are written back when the run ends (``--keep`` skips that
  so a winner stays). The tuning flows reset a motor when they switch its
  control mode, which also reloads ROM gains — so the gains are written
  *after* the mode switch and homing, right before the wave.
* A buzz guard watches the fine position for high-frequency motion and the
  reply current; past ``--buzz-abort`` degrees of >10 Hz content or
  ``--iq-abort`` amps it restores the previous gains at once, holds, and
  ends the run. Size the current limit for the pose: a loaded X8 shoulder
  draws ~10 A holding gravity alone at -55°. Shoulder_1 at speed_kp 0.1 (3× stock) vibrated immediately
  on 2026-09-18; start every sweep from the stock values in small steps.
* The joint under test holds position stiffly in this mode and will push
  back against contact up to motor torque. Keep the workspace clear.

Examples:
    axol tune.a4 --r --joint shoulder_1 --center -35 --amp 10 --mode triangle --speed 3
    axol tune.a4 --r --joint shoulder_1 --mode sine --freq 0.3 --speed-kp 0.05 --speed-ki 0.0005
    axol tune.a4 --r --joint shoulder_1 --accel 0 --position-kp 0.02 --save-run --label "pkp 0.02"
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import time
from collections import deque
from typing import Any

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...motor import CanBus, ControlMode, Motor
from ...motor.myactuator import _MA_PID_IDX, MyActuatorMotor
from ...tuning import (
    JointFrameMotor,
    joint_frame_motors,
    log_to_series,
    ramp_stages,
    safe_limits,
    save_run,
    sine_metrics,
    sweep_safety,
)
from ...tuning.runner import LiveStream, report_achieved_rate
from ..motor import add_side_and_channel_arguments, resolve_channel
from .friction import _home_all, _ramp_verified, _safe_torque_off

_MA_POS_CONTROL = 0xA4
_MA_MULTI_TURN_ANGLE = 0x92
_MA_READ_ACCEL = 0x42
_MA_WRITE_ACCEL = 0x43
_MA_READ_GAIN = 0x30
_MA_WRITE_GAIN_RAM = 0x31
_MA_WRITE_GAIN_ROM = 0x32
_FLASH_SETTLE_S = 0.3

#: Planner acceleration (dps/s, the protocol maximum) at which the firmware
#: completes each 200 Hz step's plan inside the tick, so a re-planning
#: position loop follows the stream instead of stalling on it. On the right
#: elbow (X6-P20) this tracked a 3 deg/s triangle to 0.02° RMS with 4 ms lag
#: — better than direct tracking (accel 0: 0.23°, 74 ms) — while 5000 dps/s
#: never finished a plan before the next target and the joint barely moved.
_ACCEL_STEP_FOLLOW = 60000

GAIN_NAMES: tuple[str, ...] = tuple(_MA_PID_IDX)

#: Position error (rad) past which the wave is abandoned — the loop is not
#: following at all (planner in profiled mode, or a runaway).
_ERR_ABORT = math.radians(20.0)
#: Window (s) the buzz guard evaluates high-frequency motion over.
_BUZZ_WINDOW_S = 0.1


# ---------------------------------------------------------------------------
# Pure pieces (unit-tested)
# ---------------------------------------------------------------------------


def waveform(
    mode: str,
    center: float,
    amp: float,
    duration: float,
    rate: float,
    *,
    freq: float = 0.5,
    speed: float = 0.05,
) -> list[tuple[float, float, float]]:
    """``(t, target, v_cmd)`` samples of a sine (``freq`` Hz) or a constant-speed
    triangle (``speed`` rad/s, ``amp`` half-travel) about ``center`` (rad).

    Both start at ``center`` with zero velocity so the first command is the
    hold pose the joint was ramped to. The triangle is the stick-slip probe:
    the whole pass runs at one speed, so creep behaviour is not confined to
    the sine's turnarounds.
    """
    n = max(1, math.ceil(duration * rate))
    out: list[tuple[float, float, float]] = []
    if mode == "sine":
        w = 2.0 * math.pi * freed(freq)
        for k in range(n):
            t = k / rate
            out.append((t, center + amp * math.sin(w * t), amp * w * math.cos(w * t)))
        return out
    if mode != "triangle":
        raise ValueError(f"unknown mode {mode!r}")
    v = abs(speed)
    if v <= 0.0 or amp <= 0.0:
        return [(k / rate, center, 0.0) for k in range(n)]
    # Triangle: centre → +amp → −amp → +amp …, each leg at constant speed.
    for k in range(n):
        t = k / rate
        s = v * t  # distance travelled along the zig-zag
        # Fold onto a 4·amp period: 0..amp up, amp..3amp down, 3amp..4amp up.
        phase = math.fmod(s, 4.0 * amp)
        if phase < amp:
            q, vs = center + phase, +v
        elif phase < 3.0 * amp:
            q, vs = center + amp - (phase - amp), -v
        else:
            q, vs = center - amp + (phase - 3.0 * amp), +v
        out.append((t, q, vs))
    return out


def freed(freq: float) -> float:
    """Positive frequency (a zero or negative request becomes a hold)."""
    return max(freq, 0.0)


class BuzzGuard:
    """Abort detector: high-frequency position motion or excess current.

    Feed one ``(position rad, iq A)`` sample per cycle. High-frequency motion
    is the RMS of the position about its mean over the last
    :data:`_BUZZ_WINDOW_S`; a commanded wave contributes little to that at
    creep speeds, a limit cycle or resonance a lot.
    """

    def __init__(self, rate: float, buzz_rad: float, iq_abort: float) -> None:
        self._n = max(4, int(round(_BUZZ_WINDOW_S * rate)))
        self._pos: deque[float] = deque(maxlen=self._n)
        self.buzz_rad = buzz_rad
        self.iq_abort = iq_abort
        self.peak_buzz = 0.0
        self.peak_iq = 0.0

    def feed(self, pos: float, iq: float) -> str | None:
        self._pos.append(pos)
        self.peak_iq = max(self.peak_iq, abs(iq))
        if self.iq_abort > 0.0 and abs(iq) > self.iq_abort:
            return f"current {iq:+.2f} A past the {self.iq_abort:g} A limit"
        if len(self._pos) < self._n:
            return None
        arr = np.asarray(self._pos)
        # Remove the wave itself (a straight line over 0.1 s) before scoring.
        x = np.arange(self._n)
        coef = np.polyfit(x, arr, 1)
        resid = arr - np.polyval(coef, x)
        buzz = float(np.sqrt(np.mean(resid * resid)))
        self.peak_buzz = max(self.peak_buzz, buzz)
        if self.buzz_rad > 0.0 and buzz > self.buzz_rad:
            return f"{math.degrees(buzz):.2f}° of high-frequency motion (limit {math.degrees(self.buzz_rad):.2f}°)"
        return None


def a4_metrics(log: list[dict], rate: float) -> dict[str, Any]:
    """Score a firmware-loop run: the sine scorecard plus creep smoothness."""
    m: dict[str, Any] = sine_metrics(log)
    if len(log) < 20:
        return m
    t = np.array([r["t"] for r in log])
    target = np.array([r["target"] for r in log])
    actual = np.array([r["actual"] for r in log])
    v_cmd = np.array([r["v_cmd"] for r in log])
    iq = np.array([r["iq"] for r in log])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / rate
    err = actual - target
    # Lag: how far behind the command the joint runs, from the error while
    # moving (error against the direction of travel over the speed). Works
    # for a ramp or triangle, where a correlation shift would be swallowed
    # by an offset, and for a sine, where the sign flips each half-cycle.
    moving_now = np.abs(v_cmd) > 0.02
    if moving_now.any():
        behind = float(np.mean(-err[moving_now] * np.sign(v_cmd[moving_now])))
        m["lag_ms"] = behind / float(np.mean(np.abs(v_cmd[moving_now]))) * 1000.0
    else:
        m["lag_ms"] = math.nan
    # 1-4 Hz band of the error (the stick-slip band).
    e = err - err.mean()
    n = len(e)
    F = np.abs(np.fft.rfft(e * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, dt)
    tot = float(F.sum())
    m["band_1_4"] = (
        float(math.sqrt(F[(f >= 1) & (f <= 4)].sum() / tot) * e.std())
        if tot > 0
        else 0.0
    )
    # Velocity ripple relative to the command, and stuck windows, while the
    # command is actually moving.
    win = max(2, int(round(0.05 / dt)))
    kern = np.ones(win) / win
    v_meas = np.convolve(np.gradient(actual, t), kern, mode="same")
    moving = np.abs(v_cmd) > 0.02
    v_ref = float(np.mean(np.abs(v_cmd[moving]))) if moving.any() else 0.0
    m["v_ripple"] = (
        float(np.std((v_meas - v_cmd)[moving]) / v_ref) if v_ref > 0 else math.nan
    )
    nwin = len(actual) // win
    if nwin and moving.any():
        blocks = actual[: nwin * win].reshape(nwin, win)
        mv_blocks = moving[: nwin * win].reshape(nwin, win).all(axis=1)
        travel = np.abs(blocks[:, -1] - blocks[:, 0])
        m["stuck_frac"] = (
            float(np.mean(travel[mv_blocks] < math.radians(0.03)))
            if mv_blocks.any()
            else math.nan
        )
    else:
        m["stuck_frac"] = math.nan
    # >10 Hz position content: buzz/resonance.
    m["buzz"] = float(math.sqrt(F[f >= 10].sum() / tot) * e.std()) if tot > 0 else 0.0
    m["iq_rms"] = float(np.sqrt(np.nanmean(iq * iq)))
    m["iq_max"] = float(np.nanmax(np.abs(iq)))
    # Current *variation* — what the operator feels. The mean current is the
    # gravity hold and says nothing about smoothness; its spread does, and
    # the 3-8 Hz band of it is the position loop's own mode (~5 Hz on the
    # X8-P20 shoulders), the shudder a 12 deg/s triangle's reversals kick up
    # (1.9 A at position_kp 0.7 against 0.2 A at 3 deg/s) that neither the
    # >10 Hz position "buzz" nor the >20 Hz current band track.
    iq_c = iq - np.nanmean(iq)
    m["iq_sd"] = float(np.nanstd(iq))
    Fi = np.abs(np.fft.rfft(np.nan_to_num(iq_c) * np.hanning(n))) ** 2
    tot_i = float(Fi.sum())
    m["iq_mode"] = (
        float(math.sqrt(Fi[(f >= 3) & (f <= 8)].sum() / tot_i) * m["iq_sd"])
        if tot_i > 0
        else 0.0
    )
    return m


# ---------------------------------------------------------------------------
# Motor access
# ---------------------------------------------------------------------------


def _a4_frame(position_rad: float, cap_dps: float) -> bytes:
    cap = int(max(0.0, min(65535.0, round(cap_dps))))
    return (
        bytes([_MA_POS_CONTROL, 0x00])
        + struct.pack("<H", cap)
        + struct.pack("<i", int(round(position_rad * 18000.0 / math.pi)))
    )


def speed_cap(
    v_cmd_rad_s: float, cap_dps: float, track: float, floor_dps: float
) -> float:
    """The 0xA4 speed cap (deg/s) for one streamed sample.

    ``track <= 0``: the fixed ``cap_dps``. Otherwise the cap follows the
    commanded speed — ``track × |v_cmd|``, floored at ``floor_dps`` so a
    stationary or reversing target can still be corrected, and never above
    ``cap_dps``.

    Why: with the planner at its maximum (60000 dps/s) and a fixed 60 dps
    cap, each 200 Hz target is a 0.015° step at 3 deg/s that the planner
    covers in ~0.5 ms at the cap and then idles for the remaining 4.5 ms —
    the joint moves in bursts at twenty times the commanded speed with a
    5 % duty cycle. On the right elbow that was 0.019° RMS tracking with
    four times the current spread of direct tracking (1.28 A vs 0.33 A,
    0.76 A above 20 Hz, 68–82 Hz velocity content). A cap of ~1.1–1.2× the
    commanded speed lets the planner run continuously at about that speed
    and arrive just before the next target instead.
    """
    if track <= 0.0:
        return cap_dps
    want = track * abs(math.degrees(v_cmd_rad_s))
    return min(cap_dps, max(floor_dps, want))


def _decode_a4_reply(resp: bytes) -> tuple[float, float]:
    """(iq A, speed rad/s) from a 0xA4 reply."""
    iq = struct.unpack_from("<h", resp, 2)[0] * 0.01
    speed = math.radians(struct.unpack_from("<h", resp, 4)[0])
    return iq, speed


async def _read_gains(driver: MyActuatorMotor) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, index in _MA_PID_IDX.items():
        resp = await driver._request(bytes([_MA_READ_GAIN, index, 0, 0, 0, 0, 0, 0]))
        out[name] = float(struct.unpack_from("<f", resp, 4)[0])
    return out


async def _write_gains(
    driver: MyActuatorMotor, gains: dict[str, float], persist: bool
) -> None:
    cmd = _MA_WRITE_GAIN_ROM if persist else _MA_WRITE_GAIN_RAM
    for name, value in gains.items():
        await driver._request(
            bytes([cmd, _MA_PID_IDX[name], 0, 0]) + struct.pack("<f", float(value))
        )
        await asyncio.sleep(_FLASH_SETTLE_S if persist else 0.02)


async def _read_accel(driver: MyActuatorMotor) -> tuple[int, int]:
    out = []
    for kind in (0x00, 0x01):
        resp = await driver._request(bytes([_MA_READ_ACCEL, kind, 0, 0, 0, 0, 0, 0]))
        out.append(int(struct.unpack_from("<i", resp, 4)[0]))
    return out[0], out[1]


async def _write_accel(driver: MyActuatorMotor, acc: int, dec: int) -> tuple[int, int]:
    for kind, value in ((0x00, acc), (0x01, dec)):
        await driver._request(
            bytes([_MA_WRITE_ACCEL, kind, 0, 0]) + struct.pack("<I", int(max(0, value)))
        )
        await asyncio.sleep(_FLASH_SETTLE_S)
    return await _read_accel(driver)


async def _stream(
    motor: JointFrameMotor,
    driver: MyActuatorMotor,
    samples: list[tuple[float, float, float]],
    cap_dps: float,
    rate: float,
    guard: BuzzGuard,
    live: LiveStream,
    cap_track: float = 0.0,
    cap_floor_dps: float = 1.0,
) -> tuple[list[dict], str | None]:
    """Stream the wave; returns the log and the abort reason, if any."""
    offset = motor.offset
    period = 1.0 / rate
    log: list[dict] = []
    t0 = time.perf_counter()
    deadline = t0
    for _t_nominal, target, v_cmd in samples:
        deadline += period
        cap = speed_cap(v_cmd, cap_dps, cap_track, cap_floor_dps)
        resp = await driver._request(_a4_frame(target - offset, cap))
        iq, speed = _decode_a4_reply(resp)
        fine = await driver._request(bytes([_MA_MULTI_TURN_ANGLE, 0, 0, 0, 0, 0, 0, 0]))
        pos = struct.unpack_from("<i", fine, 4)[0] * (0.01 * math.pi / 180.0) + offset
        now = time.perf_counter() - t0
        log.append(
            {
                "t": now,
                "target": target,
                "actual": pos,
                "error": pos - target,
                "torque": math.nan,
                "speed": speed,
                "iq": iq,
                "v_cmd": v_cmd,
            }
        )
        live.add(now, target, pos)
        reason = guard.feed(pos, iq)
        if reason is None and abs(pos - target) > _ERR_ABORT:
            reason = f"tracking error {math.degrees(pos - target):+.1f}° — the loop is not following"
        if reason is not None:
            return log, reason
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
    return log, None


async def _hold(
    driver: MyActuatorMotor,
    motor: JointFrameMotor,
    pose: float,
    cap_dps: float,
    seconds: float,
) -> None:
    period = 0.01
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        await driver._request(_a4_frame(pose - motor.offset, cap_dps))
        await asyncio.sleep(period)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.a4`` subcommand."""
    p = subparsers.add_parser(
        "tune.a4",
        help="Tune a MyActuator joint's firmware position loop (0xA4) with a sine or triangle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        help="Joint to drive (MyActuator joints: shoulder_1 … wrist_1)",
    )
    p.add_argument(
        "--mode",
        choices=["sine", "triangle"],
        default="triangle",
        help="Wave shape: sine (--freq) or constant-speed triangle (--speed) (default: triangle)",
    )
    p.add_argument(
        "--center",
        type=float,
        default=None,
        help="Centre, joint-frame degrees (default: joint midpoint of the safe range)",
    )
    p.add_argument(
        "--amp", type=float, default=10.0, help="Half-travel, degrees (default: 10)"
    )
    p.add_argument(
        "--freq", type=float, default=0.3, help="[sine] frequency, Hz (default: 0.3)"
    )
    p.add_argument(
        "--speed",
        type=float,
        default=3.0,
        help="[triangle] pass speed, deg/s (default: 3)",
    )
    p.add_argument(
        "--duration", type=float, default=12.0, help="Seconds of wave (default: 12)"
    )
    p.add_argument(
        "--rate", type=float, default=200.0, help="Command rate, Hz (default: 200)"
    )
    p.add_argument(
        "--cap", type=float, default=60.0, help="0xA4 speed cap, deg/s (default: 60)"
    )
    p.add_argument(
        "--cap-track",
        type=float,
        default=0.0,
        help="Make the per-command speed cap follow the wave: cap = this × |commanded "
        "speed| (floored at --cap-floor, never above --cap). 0 (default) = fixed --cap. "
        f"With --accel {_ACCEL_STEP_FOLLOW} a fixed cap lets the planner burst through "
        "each 200 Hz step at the cap and idle the rest of the tick (4x the current "
        "spread on the elbow); 1.1-1.2 keeps it moving continuously at about the "
        "commanded speed.",
    )
    p.add_argument(
        "--cap-floor",
        type=float,
        default=1.0,
        help="Lowest cap --cap-track may set, deg/s, so a stationary or reversing "
        "target can still be corrected (default: 1)",
    )
    p.add_argument(
        "--accel",
        type=int,
        default=None,
        help="Position-planner acceleration (dps/s) for the run, written to ROM before the "
        "mode-switch reset and restored afterwards unless --keep; default: leave the stored "
        f"value. 0 = direct PI tracking of the stream; {_ACCEL_STEP_FOLLOW} (the protocol "
        "maximum) = the planner completes each 200 Hz step within the tick, which tracked the "
        "X6-P20 elbow better than 0 (0.02° vs 0.23° RMS). Anything in between re-plans every "
        "target and will not follow the wave.",
    )
    for name in GAIN_NAMES:
        p.add_argument(
            f"--{name.replace('_', '-')}",
            dest=name,
            type=float,
            default=None,
            help=f"Firmware {name} for this run (default: leave as is)",
        )
    p.add_argument(
        "--persist",
        action="store_true",
        help="Write gains to ROM (0x32) instead of RAM (0x31)",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Leave the run's gains and planner acceleration in the motor afterwards",
    )
    p.add_argument(
        "--buzz-abort",
        type=float,
        default=0.3,
        help="Abort past this much >10 Hz position motion, degrees RMS over 0.1 s (default: 0.3; 0 off)",
    )
    p.add_argument(
        "--iq-abort",
        type=float,
        default=30.0,
        help="Abort past this reply current, amps (default: 30; 0 off). A loaded X8 "
        "shoulder draws ~10 A just holding gravity at -55°, so keep this well above "
        "the pose's static current",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help="Persist the run for the diagnostics dashboard",
    )
    p.add_argument("--label", default=None, help="Free-form note stored on the run")
    p.add_argument(
        "--group", default=None, help="Shared id linking the runs of one sweep"
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side = "left" if is_left else "right"
    lo, hi = safe_limits(joint, is_left)
    amp = math.radians(args.amp)
    center = math.radians(args.center) if args.center is not None else (lo + hi) / 2.0
    margin = math.radians(2.0)
    if not (lo + margin <= center - amp and center + amp <= hi - margin):
        raise SystemExit(
            f"{joint.value}: {math.degrees(center - amp):+.1f}..{math.degrees(center + amp):+.1f}° "
            f"is outside the safe range [{math.degrees(lo) + 2:.1f}, {math.degrees(hi) - 2:.1f}]°"
        )
    requested = {
        n: getattr(args, n) for n in GAIN_NAMES if getattr(args, n) is not None
    }
    samples = waveform(
        args.mode,
        center,
        amp,
        args.duration,
        args.rate,
        freq=args.freq,
        speed=math.radians(args.speed),
    )
    print(f"\ntune.a4 — {side} {joint.value}: firmware position loop (0xA4)")
    print(
        f"  {args.mode} about {math.degrees(center):+.1f}° ±{args.amp:g}°, "
        + (f"{args.freq:g} Hz" if args.mode == "sine" else f"{args.speed:g} deg/s")
        + f", {args.duration:g} s at {args.rate:g} Hz, speed cap {args.cap:g} dps"
        + (
            f" tracking {args.cap_track:g}× commanded speed (floor {args.cap_floor:g}"
            ", planner permitting)"
            if args.cap_track > 0
            else ""
        )
    )

    channel = resolve_channel(args)
    async with CanBus(channel) as bus:
        raw = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw.values()])
        driver = raw[joint]._driver
        if not isinstance(driver, MyActuatorMotor):
            raise SystemExit(f"{joint.value} is not a MyActuator joint")
        before_gains: dict[str, float] | None = None
        before_accel: tuple[int, int] | None = None
        log: list[dict] = []
        reason: str | None = None
        used_gains: dict[str, float] = {}
        accel_used: tuple[int, int] | None = None

        # Planner acceleration goes in *before* the mode switch below: that
        # switch is a 0x76 reset, and the reset is what makes a planner value
        # of 0 take effect. On the X6-P20 elbow (firmware 2025070202) a 0
        # written into a running position loop is ignored — the joint held
        # its target and executed nothing for a whole run (2026-09-21) while
        # the same 0 stored before the reset gave the documented direct
        # tracking. Non-zero values do apply live on that firmware (5000 →
        # 60000 took effect mid-session); the X8-P20 shoulders (2026042402)
        # apply 0 live as well. Writing first is right for every one of them.
        stored_accel = await _read_accel(driver)
        print(
            f"  planner accel/decel stored: {stored_accel[0]}/{stored_accel[1]} dps/s"
        )
        if args.accel is not None and stored_accel != (args.accel, args.accel):
            before_accel = stored_accel
            accel_used = await _write_accel(driver, args.accel, args.accel)
            print(
                f"  planner accel/decel {stored_accel[0]}/{stored_accel[1]} → {accel_used[0]}/{accel_used[1]} dps/s"
            )
        else:
            accel_used = stored_accel
        if accel_used[0] not in (0, _ACCEL_STEP_FOLLOW):
            print(
                f"  ! planner acceleration is {accel_used[0]} dps/s: the firmware re-plans "
                "every streamed target and will not follow the wave — pass --accel 0 "
                f"(direct PI tracking) or --accel {_ACCEL_STEP_FOLLOW} (planner completes "
                "each step within the tick)"
            )

        cap_track = args.cap_track
        if accel_used[0] == 0 and cap_track > 0:
            # Under direct tracking the cap is a hard limit on the PI output:
            # pinned near the commanded speed the loop can never catch up
            # (right elbow, pKp 0.5, cap-track 1.1: 1.8° RMS, 480 ms lag).
            # The knob exists for the planner's per-tick bursts, which
            # direct tracking does not have.
            print(
                f"  ! --cap-track {cap_track:g} ignored: the planner is at 0 (direct "
                "PI tracking), where the cap would only throttle the loop"
            )
            cap_track = 0.0

        motors = await joint_frame_motors(raw, is_left)
        await asyncio.gather(
            *[
                m.set_control_mode(ControlMode.POSITION_VELOCITY)
                for m in motors.values()
            ]
        )
        motor = motors[joint]
        try:
            print("  Homing all joints to rest ...")
            await _home_all(motors)
            other_targets, _lo, _hi, notes = sweep_safety(joint, is_left)
            for note in notes:
                print(f"  {note}")
            for stage in ramp_stages(other_targets):
                await _ramp_verified(motors, stage)
            print(f"  Ramping {joint.value} to {math.degrees(center):+.1f}° ...")
            await _ramp_verified(motors, {joint: center})
            await asyncio.sleep(0.3)

            # Gains are written here, after the mode switch and homing: RAM
            # gains (0x31) do not survive the reset those perform.
            stock = await _read_gains(driver)
            used_gains = {**stock, **requested}
            if requested:
                before_gains = stock
                await _write_gains(driver, requested, args.persist)
                used_gains = await _read_gains(driver)
                for n in requested:
                    flag = (
                        ""
                        if abs(used_gains[n] - requested[n])
                        <= 1e-6 * max(1.0, abs(requested[n]))
                        else "  (! not accepted)"
                    )
                    print(f"  {n:12s} {stock[n]:.6g} → {used_gains[n]:.6g}{flag}")
            else:
                print("  gains: " + ", ".join(f"{n}={v:.6g}" for n, v in stock.items()))

            guard = BuzzGuard(args.rate, math.radians(args.buzz_abort), args.iq_abort)
            live = LiveStream("sine", joint)
            print("  Running ...")
            log, reason = await _stream(
                motor,
                driver,
                samples,
                args.cap,
                args.rate,
                guard,
                live,
                cap_track=cap_track,
                cap_floor_dps=args.cap_floor,
            )
            live.flush()
            if reason is not None:
                print(f"\n  ! aborted: {reason}")
                if before_gains is not None:
                    await _write_gains(driver, before_gains, args.persist)
                    print("  previous gains restored")
                    before_gains = None
            # The raw 0xA4 stream never fills the driver's position cache;
            # read the joint explicitly before holding it where it stopped.
            here = await motor.get_position()
            await _hold(driver, motor, here, args.cap, 0.5)
            report_achieved_rate(log, args.rate)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to rest ...")
            # Home *before* restoring the firmware gains: the run's gains are
            # the stiffer set, and the stock position loop has been seen to
            # stall short of rest on a gravity-loaded elbow. The planner is
            # restored first only when the run left it at 0, because a
            # direct-tracking joint would otherwise execute the homing target
            # at the speed cap.
            homed = False
            try:
                if (
                    before_accel is not None
                    and not args.keep
                    and accel_used
                    and accel_used[0] == 0
                ):
                    got = await _write_accel(driver, before_accel[0], before_accel[1])
                    print(f"  planner accel/decel restored to {got[0]}/{got[1]} dps/s")
                    before_accel = None
                await _ramp_verified(motors, {joint: 0.0})
                await _home_all(motors)
                homed = True
            except Exception as exc:  # noqa: BLE001 - reported below, arm keeps holding
                print(f"  ! return to rest did not complete: {exc}")
            try:
                if before_gains is not None and not args.keep:
                    await _write_gains(driver, before_gains, args.persist)
                    print("  previous gains restored")
                elif before_gains is not None:
                    print("  gains kept (--keep)")
                if before_accel is not None and not args.keep:
                    got = await _write_accel(driver, before_accel[0], before_accel[1])
                    print(f"  planner accel/decel restored to {got[0]}/{got[1]} dps/s")
                elif before_accel is not None and args.keep:
                    print(
                        f"  planner left at {accel_used[0]}/{accel_used[1]} dps/s (--keep) — a direct-tracking joint executes a stored target on wake"
                    )
            except Exception as exc:  # noqa: BLE001 - report, then keep tearing down
                print(f"  ! restore failed: {exc}")
            if not homed:
                print("  (torque-off will be refused unless every joint is at rest)")
            await _safe_torque_off(motors, raw)

    if len(log) < 20:
        print("\nToo few samples to score.")
        return
    metrics = a4_metrics(log, args.rate)
    metrics["aborted"] = reason is not None
    print(f"\n{'─' * 66}")
    print(
        f"  tracking RMS {math.degrees(metrics['rms']):.3f}°   max {math.degrees(metrics['max']):.3f}°   lag {metrics['lag_ms']:.0f} ms"
    )
    print(
        f"  1-4 Hz band {math.degrees(metrics['band_1_4']):.3f}°   >10 Hz buzz {math.degrees(metrics['buzz']):.3f}°"
    )
    print(
        f"  velocity ripple {metrics['v_ripple']:.2f} (MIT stick-slip ≈ 0.8, smooth < 0.2)   stuck windows {metrics['stuck_frac']:.2f}"
    )
    print(
        f"  current RMS {metrics['iq_rms']:.2f} A   peak {metrics['iq_max']:.2f} A   "
        f"spread {metrics['iq_sd']:.2f} A   3-8 Hz mode {metrics['iq_mode']:.2f} A   "
        f"loop {metrics['hz']:.0f} Hz"
    )
    print(f"{'─' * 66}")
    if args.save_run:
        params = {
            "wire": "a4",
            "mode": args.mode,
            "center_deg": math.degrees(center),
            "amp_deg": args.amp,
            "freq_hz": args.freq if args.mode == "sine" else None,
            "speed_dps": args.speed if args.mode == "triangle" else None,
            "duration_s": args.duration,
            "rate_hz": args.rate,
            "cap_dps": args.cap,
            "cap_track": cap_track,
            "cap_floor_dps": args.cap_floor,
            "accel": list(accel_used) if accel_used else None,
            "persist": args.persist,
        }
        run_id = save_run(
            "sine",
            log_to_series(log),
            metrics,
            side=side,
            joint=joint.value,
            gains=used_gains,
            params=params,
            label=args.label,
            group=args.group,
        )
        print(f"\nSaved tuning run {run_id} (kind=sine, wire=a4)")
