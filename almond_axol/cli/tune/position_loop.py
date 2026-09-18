"""
axol tune.position-loop

Tune a MyActuator's **internal position-loop PID** — the gains the firmware
uses for the 0xA4 / 0xA9 position commands — so that a commanded position is
actually held under gravity.

Why this exists. Every MyActuator on this fleet ships with the position loop
effectively switched off: ``position_kp`` 0.008-0.06 and ``position_ki``
**exactly 0**. A proportional-only loop with no integral cannot hold a gravity
load at all — it sags until the P term balances the weight and stays there.
That has two consequences that look unrelated until you measure the gains:

* the ``wire_mode`` experiments (``a4`` / ``a9``) do not actuate, because the
  firmware loop they hand control to produces almost no torque; and
* every single-joint tuning tool that parks its *other* joints with
  ``set_position_velocity`` — ``tune.gravity``, ``tune.friction``,
  ``tune.breakaway`` — is holding them with that same dead loop. Lightly
  loaded they hold; loaded they sag, silently, and the fit is then computed
  against a pose the arm was never in.

None of this touches MIT-mode impedance control: ``set_impedance`` carries its
gains in every frame and ignores the stored ones, so teleop and the realtime
core are unaffected by anything this command writes.

Method, per joint:

1. Hold every *other* joint with streamed MIT impedance and gravity
   feedforward — deliberately not the position servo this command exists to
   fix, so the holders are trustworthy while the test joint is not.
2. Put the test joint in position mode, command it to hold where it is, and
   measure how far it sags over a settle window.
3. Step the gains up (``0x31``, **RAM only** — a power cycle undoes every
   trial) and repeat, watching both the sag and the position ripple, until
   the sag is inside tolerance or the joint starts to oscillate.
4. Report the best pair. ``--save`` commits it with ``0x32`` to ROM.

Back the motors up first: ``axol motor.dump-config --r --out before.json``
records the loop gains, and ``axol motor.restore-config`` writes them back.

Examples:
    axol tune.position-loop --r --joint elbow
    axol tune.position-loop --r --joint shoulder_1 --center 30 --save
    axol tune.position-loop --r --joint elbow --kp 0.5 2 8 32 --ki 0 0.01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from dataclasses import replace

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...motor import ControlMode, Motor
from ...motor.bus import CanBus
from ...robot.axol import arm_limits
from ...robot.config import AxolConfig
from ...robot.gravity import GravityCompensator
from ...tuning import joint_frame_motors, save_run
from ...tuning.holders import HOLD_HZ as _HOLD_HZ  # noqa: F401
from ...tuning.holders import read_position
from ...tuning.holders import ImpedanceHolders as _Holders
from ...tuning.joint_frame import JointFrameMotor
from ...utils.logquiet import quiet_noisy_loggers
from ..motor import add_side_and_channel_arguments, resolve_channel

#: Homing order, distal first: straightening the wrists and elbow before the
#: shoulders means each shoulder later swings a folded arm.
_HOME_ORDER: tuple[Joint, ...] = (
    Joint.WRIST_3,
    Joint.WRIST_2,
    Joint.WRIST_1,
    Joint.ELBOW,
    Joint.SHOULDER_3,
    Joint.SHOULDER_2,
    Joint.SHOULDER_1,
)
_HOME_SPEED = 0.25  # rad/s
#: How long the tracking test will chase the sine's start before giving up
#: and measuring anyway (saying so, rather than silently reporting the
#: approach as tracking error).
_APPROACH_MAX_S = 20.0
_APPROACH_TOL_RAD = math.radians(0.5)
#: Multipliers applied to the joint's current `position_kp`, low to high. The
#: search stops at the first value that holds, so a joint that is nearly right
#: never sees the large ones.
_KP_STEPS = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)


_HOME_TARGET_SPEED = math.radians(20.0)


async def _home_test_joint(
    test: JointFrameMotor, joint: Joint, label: str, timeout_s: float = 20.0
) -> bool:
    """Drive the swept joint to rest under 0xA4 and *watch it get there*.

    Bench: after the planner and stock gains were restored, a single 0xA4 to
    rest was acknowledged and then ignored -- the elbow sat at its last target
    for 15 s at constant torque while the tool waited in silence. So the
    target is re-sent continuously, progress is printed every 2 s (a joint
    that is not moving is visible within seconds, in the dashboard log too),
    and 3 s without motion is a failed strategy, not something to wait out.
    """
    start = None
    t_start = time.monotonic()
    last_log = t_start
    last_pos = None
    last_move = t_start
    while time.monotonic() - t_start < timeout_s:
        try:
            await test.set_position_velocity(0.0, _HOME_TARGET_SPEED)
        except Exception as e:
            print(f"    ({label}) command failed: {e}")
        try:
            pos = await read_position(test)
        except Exception:
            await asyncio.sleep(0.5)
            continue
        if start is None:
            start = last_pos = pos
        if abs(pos) < math.radians(2.0):
            print(f"    ({label}) {joint.value} at rest ({math.degrees(pos):+.1f}°)")
            return True
        if last_pos is not None and abs(pos - last_pos) > math.radians(0.5):
            last_move = time.monotonic()
        last_pos = pos
        now = time.monotonic()
        if now - last_log >= 2.0:
            print(f"    ({label}) {joint.value} at {math.degrees(pos):+.1f}° ...")
            last_log = now
        if now - last_move > 3.0:
            print(
                f"    ({label}) {joint.value} has not moved for 3 s at "
                f"{math.degrees(pos):+.1f}° -- the motor is not acting on 0xA4"
            )
            return False
        await asyncio.sleep(0.5)
    return False


async def _home_test_with_fallback(
    test: JointFrameMotor,
    joint: Joint,
    original,
    accel_before: float | None,
    kp_hint: float = 0.0,
) -> bool:
    """Home on the motor's own profiled planner; if it will not move, fall
    back to direct tracking (accel 0) at a moderate gain -- the mode that
    homed this joint all day -- and put the planner and gains back after."""
    if await _home_test_joint(test, joint, "profiled"):
        return True
    await test.motor.clear_errors()
    try:
        await test.motor._driver.set_acceleration(0.0, allow_zero=True)
        kp_home = max(original.position_kp * 4.0, kp_hint)
        await test.motor.set_gains(
            replace(original, position_kp=kp_home, position_ki=0.0), persist=False
        )
        print(f"    falling back to direct tracking at position_kp={kp_home:.4f}")
    except Exception as e:
        print(f"    fallback setup failed: {e}")
    ok = await _home_test_joint(test, joint, "direct")
    try:
        if accel_before is not None:
            await test.motor._driver.set_acceleration(accel_before, allow_zero=True)
        await test.motor.set_gains(original, persist=False)
    except Exception as e:
        print(f"    could not restore after fallback: {e}")
    return ok


class _MotorFaulted(Exception):
    """The motor's own protection latched during a gain point."""


#: Sag (deg) at or below which the joint counts as holding.
_SAG_OK_DEG = 0.3
#: Position ripple (deg rms) above which a candidate is called oscillating and
#: the search stops rather than escalating into a louder instability.
_RIPPLE_LIMIT_DEG = 0.25
#: Default ripple limit for the *tracking* sweep. Deliberately permissive.
#: There is no calibration for this number: on hardware an operator heard
#: nothing at a gain measuring 0.088° and heard clear vibration at a higher
#: one that was never measured. A tight guess simply truncates the sweep
#: before it can produce the data that would calibrate it, so the default
#: lets the sweep run and prints the column. The relative jump below is the
#: detector that does not need a calibrated absolute level, and
#: ``--ripple-limit`` sets this once the joint's own numbers are known.
_TRACK_RIPPLE_LIMIT_DEG = 0.25
#: Ripple growth against the previous gain that counts as the onset of
#: oscillation regardless of the absolute level.
_TRACK_RIPPLE_JUMP = 3.0
#: The same test on q-axis current, which is the channel that actually sees a
#: limit cycle. Tighter than the position jump because current carries the
#: cycle at full amplitude instead of aliased down to nothing, and because a
#: smooth gain increase moves it far less than a cycle does.
_TRACK_BUZZ_JUMP = 2.0
#: position_ki ladder for the tracking sweep, as fractions of the winning
#: position_kp. A P-only position loop is type 1: it cannot follow a ramp
#: without a standing error proportional to velocity, which is the whole
#: ``a/kp`` term the sweep measures. The integral is what removes it, and it
#: does so without moving position_kp toward the stability cliff. Geometric
#: and ascending, because the firmware's integral units are not documented
#: against a loop rate, so the decade is the unit of ignorance here.
_KI_STEPS: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1)
_SETTLE_S = 2.0
_MEASURE_S = 2.0


async def _track(
    motor: JointFrameMotor,
    center: float,
    amp: float,
    freq: float,
    secs: float,
    max_speed: float,
    rate_hz: float,
) -> tuple[float, float, float, float, float]:
    """Stream 0xA4 along a sine.

    Returns ``(rms, max, lag_ms, ripple)`` in degrees, ``buzz`` (the
    detrended q-axis current in amps), and the ``series`` dict of the pass
    (``t``/``target``/``actual`` in rad, ``current`` in A) so a run can be
    persisted for the dashboard.

    This is the test that matters for ``wire_mode``: a *held* position says
    nothing about whether the firmware loop can follow a target that keeps
    moving. In profiled-motion mode it cannot — every frame restarts a ramp —
    so a joint that holds perfectly can still track nothing at all.
    """
    # Drive a *cosine* from one extreme, so the sweep asks for zero velocity
    # at the instant it starts timing. A sine about the centre starts at peak
    # velocity, which after the approach below means peak velocity demanded
    # from a standstill: the joint has to break stiction before it can track,
    # and that costs far more than the loop's own settling. Measured on the
    # right elbow at position_kp 0.96 it was worth 23 % of the reported rms
    # (0.126° -> 0.155°) -- and it scaled inversely with gain, so it flattered
    # high gains and tilted the whole sweep. Starting at rest at a turning
    # point matches the state the approach leaves behind, so there is no step
    # to recover from.
    start = center - amp

    # Reach that start before timing anything. Without this the first entry of
    # a sweep measures the joint travelling there from wherever it was parked
    # — tens of degrees over the whole window — and reports it as a tracking
    # failure. Every later entry then starts on target and looks fine, so the
    # artefact lands on whichever gain happens to be measured first.
    approach_deadline = time.monotonic() + _APPROACH_MAX_S
    while time.monotonic() < approach_deadline:
        await motor.set_position_velocity(start, max_speed)
        try:
            if abs(await motor.get_position() - start) < _APPROACH_TOL_RAD:
                break
        except Exception:
            pass
        await asyncio.sleep(0.02)
    else:
        # Do not measure anyway. On hardware an unloaded pose that never
        # reached the start still printed 4.86° and 6.66° into the results
        # table, with a lag of -344 ms — the joint leading its own target by
        # a third of a second, which is not a tracking figure at all. A row
        # that looks like data outranks a warning above it.
        print(
            f"    !! never reached {math.degrees(start):.1f}° in "
            f"{_APPROACH_MAX_S:.0f}s — no result for this pass. The joint is "
            f"not following 0xA4 here; check the gain, the load and the pose."
        )
        return (float("nan"),) * 5 + ({},)
    await asyncio.sleep(0.3)

    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    tt: list[float] = []
    tgt: list[float] = []
    act: list[float] = []
    amps: list[float] = []
    while True:
        now = time.monotonic() - t0
        if now >= secs:
            break
        target = center - amp * math.cos(2.0 * math.pi * freq * now)
        # The q-axis current rides the command's own reply, so it costs no
        # extra round trip -- and it is the only channel here that can see a
        # limit cycle. A cycle the motor runs at tens of Hz is a fraction of
        # an encoder count in position and this loop samples position at
        # ~50 Hz, so it aliases away: an elbow measured 0.0615° of position
        # ripple while visibly oscillating. The same cycle swings amps.
        try:
            amps.append((await motor.set_position_velocity_reply(target, max_speed))[2])
        except AttributeError:
            await motor.set_position_velocity(target, max_speed)
            amps.append(float("nan"))
        # An explicit 0x92 read, not the cached `position`: that cache is fed
        # by MIT impedance replies, and 0xA4 answers on the 0x240 frame
        # instead, so it is never populated here. The 0xA4 reply does carry a
        # position but only at 1 deg/LSB — useless for a tracking error this
        # size — hence the extra round trip.
        try:
            pos = await motor.get_position()
        except Exception:
            pos = float("nan")
        tt.append(now)
        tgt.append(target)
        act.append(pos)
        spent = time.monotonic() - t0 - now
        if spent < dt:
            await asyncio.sleep(dt - spent)
    # Discard the first full cycle anyway. The cosine start removes the step,
    # but the joint still has to break away from a dead stop once, and a
    # steady-state figure should not carry it.
    a = np.array(act)
    g = np.array(tgt)
    i_a = np.array(amps)
    tt_a = np.array(tt)
    warm = tt_a >= min(1.0 / max(freq, 1e-6), 0.5 * secs)
    a, g, i_a, tt_a = a[warm], g[warm], i_a[warm], tt_a[warm]
    tt = list(tt_a)
    good = np.isfinite(a)
    achieved = len(tt) / max(tt[-1] - tt[0], 1e-9) if len(tt) > 1 else 0.0
    if achieved < 0.8 * rate_hz:
        print(
            f"    (achieved {achieved:.0f} Hz of the requested {rate_hz:.0f} — two "
            f"CAN round trips per sample; lower --rate for a clean cadence)"
        )
    if good.sum() < 20:
        print(f"    ({good.sum()} of {len(a)} position reads succeeded)")
        return (float("nan"),) * 5 + ({},)
    err = np.degrees(g[good] - a[good])
    v = np.gradient(g[good], tt_a[good])
    m = np.abs(v) > 1e-3
    lag = (
        1e3 * np.polyfit(np.abs(v[m]), np.abs(np.radians(err[m])), 1)[0]
        if m.sum() > 20
        else float("nan")
    )
    # Ripple: what is left after removing the smooth following error. A
    # raised gain buys tracking accuracy and eventually spends it on
    # oscillation, and rms alone will happily keep falling while the joint
    # buzzes — an operator hears that long before the mean error notices.
    # A fixed 0.15 s window: the drive sine passes through it essentially
    # unchanged, so it leaves no residue, while anything above ~7 Hz is
    # retained in full. A window sized as a fraction of the record instead
    # distorts the sine and reports its own smoothing error as ripple.
    k = max(3, int(0.15 * rate_hz) | 1)

    def detrended_std(x: np.ndarray) -> float:
        if len(x) <= 3 * k or not np.isfinite(x).all():
            return float("nan")
        return float((x - np.convolve(x, np.ones(k) / k, mode="same"))[k:-k].std())

    ripple = detrended_std(err)
    # Same detrending on the q-axis current. Units are amps, and the absolute
    # level means nothing without a per-motor torque constant this driver does
    # not measure -- but the *growth* against the previous gain does not need
    # one, and it is the growth that marks the onset of a limit cycle.
    buzz = detrended_std(i_a[good])
    series = {
        "t": tt_a[good] - tt_a[good][0],
        "target": g[good],
        "actual": a[good],
        "current": i_a[good],
    }
    return float(err.std()), float(np.abs(err).max()), float(lag), ripple, buzz, series


async def _measure(
    motor: JointFrameMotor, target: float, max_speed: float
) -> tuple[float, float]:
    """Command ``target`` with 0xA4 and return ``(sag_deg, ripple_deg_rms)``."""
    deadline = time.monotonic() + _SETTLE_S
    while time.monotonic() < deadline:
        await motor.set_position_velocity(target, max_speed)
        await asyncio.sleep(0.02)
    samples: list[float] = []
    deadline = time.monotonic() + _MEASURE_S
    while time.monotonic() < deadline:
        await motor.set_position_velocity(target, max_speed)
        await asyncio.sleep(0.02)
        samples.append(await motor.get_position())
    if not samples:
        return float("nan"), float("nan")
    a = np.array(samples)
    return math.degrees(abs(target - a.mean())), math.degrees(a.std())


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "tune.position-loop",
        help="Tune a MyActuator's internal position-loop PID (0xA4 hold).",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS if j is not Joint.GRIPPER],
        help="Joint to tune (MyActuator: shoulder_1 .. wrist_1)",
    )
    p.add_argument(
        "--center",
        type=float,
        default=None,
        help="Angle (deg) to hold at. Pick a loaded pose — an unloaded joint "
        "holds at any gain and teaches nothing. Default: current position.",
    )
    p.add_argument(
        "--kp",
        type=float,
        nargs="+",
        help="position_kp values to sweep, ascending. This is the stiffness of "
        "the motor's own position loop: how much speed it asks for per degree "
        "of error. Higher means less lag and tighter tracking until the joint "
        "buzzes -- the sweep stops at the first value that does. Default: the "
        "motor's current value times 1, 4, 16 ...",
    )
    p.add_argument(
        "--speed-kp",
        type=float,
        nargs="+",
        default=None,
        help="[track] Inner speed-loop kp. One value: fixed override for the "
        "run. Several: swept FIRST, ascending, at the configured position "
        "gains (innermost loop first -- the position loop's stability cliff "
        "is bounded by the speed loop's phase margin); the winner is used by "
        "the position stages. RAM-only like every other write here.",
    )
    p.add_argument(
        "--speed-ki",
        type=float,
        nargs="+",
        default=None,
        help="[track] Inner speed-loop ki; same one-value / sweep semantics "
        "as --speed-kp. Current-loop gains are printed, never written: they "
        "are tuned to the motor's electrical constants and mistuning them "
        "heats or faults the motor rather than making a visible step.",
    )
    p.add_argument(
        "--ki",
        type=float,
        nargs="+",
        default=None,
        help="position_ki values to try at the winning kp. [hold] default "
        "kp/100, which removes the last of the sag. [track] default is the "
        "ladder kp x (0.001, 0.003, 0.01, 0.03, 0.1): a P-only position loop "
        "carries a velocity-following error it cannot remove, and the "
        "integral removes it without pushing kp toward the stability cliff.",
    )
    p.add_argument(
        "--max-speed",
        type=float,
        default=30.0,
        help="Speed limit (deg/s) carried by the 0xA4 frame (default: 30)",
    )
    p.add_argument(
        "--sag-deg",
        type=float,
        default=_SAG_OK_DEG,
        help=f"Sag counted as holding (default: {_SAG_OK_DEG})",
    )
    p.add_argument(
        "--accel",
        type=float,
        default=None,
        help="Write this position-planning acceleration (deg/s^2) before "
        "testing. **0 selects direct tracking mode**, which is what a "
        "streamed target needs; anything else is profiled motion, where "
        "every frame restarts its own ramp. 0 is outside the documented "
        "100-60000 range and the motor may refuse it — the value is read "
        "back and reported. Restored afterwards unless --save.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="[track] passes per gain, reporting mean and spread (default 1). "
        "Two hardware runs at position_kp 0.96 with different preceding gains "
        "read 0.1540° and 0.1562°, so a single pass resolves ~1 % and repeats "
        "are only worth their wall time when a difference looks that small.",
    )
    p.add_argument(
        "--ripple-limit",
        type=float,
        default=_TRACK_RIPPLE_LIMIT_DEG,
        help=f"[track] ripple (deg) that stops the sweep (default "
        f"{_TRACK_RIPPLE_LIMIT_DEG}). The sweep also stops on a "
        f"{_TRACK_RIPPLE_JUMP:.0f}x jump against the previous gain, which "
        f"needs no calibration. Read the column and trust your ears: a joint "
        f"buys tracking accuracy with vibration, and rms falls right through "
        f"the point where it becomes audible.",
    )
    p.add_argument(
        "--mode",
        choices=("hold", "track"),
        default="hold",
        help="hold: measure sag at a fixed target (default). track: stream a "
        "sine and measure following error — the test that matters for "
        "wire_mode, since a joint can hold perfectly and track nothing.",
    )
    p.add_argument("--amp", type=float, default=10.0, help="[track] amplitude (deg)")
    p.add_argument("--freq", type=float, default=0.2, help="[track] frequency (Hz)")
    p.add_argument(
        "--duration", type=float, default=15.0, help="[track] seconds (default 15)"
    )
    p.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="[track] command rate Hz (default 100)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Commit the winner to ROM (0x32). Without this every write is "
        "RAM-only (0x31) and a power cycle restores the motor.",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help="[track] Persist every gain point as a dashboard tuning run "
        "(kind position_loop): the target/actual pass plus its scorecard, "
        "all points of one sweep sharing a group. This is how the tuning "
        "workbench drives the command.",
    )
    p.add_argument("--label", default=None, help="Free-form note stored on saved runs.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    quiet_noisy_loggers()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nExiting tune.position-loop ...")


async def _run(args: argparse.Namespace) -> None:
    is_left = bool(args.l)
    joint = Joint(args.joint)
    config = AxolConfig().resolved()
    channel = resolve_channel(args)
    max_speed = math.radians(args.max_speed)

    print(f"\nPosition-loop tuning — {'left' if is_left else 'right'} {joint.value}")
    print("  Holders run MIT impedance, not the position servo being tuned.")
    print(
        "  Writes are RAM-only (0x31) unless --save; a power cycle restores ROM."
        if not args.save
        else "  --save: the winner will be committed to ROM (0x32)."
    )

    async with CanBus(channel) as bus:
        raw = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw.values()])
        motors = await joint_frame_motors(raw, is_left)
        test = motors[joint]
        original = await test.motor.get_gains()
        print(
            f"  current gains: position kp={original.position_kp:.4f} "
            f"ki={original.position_ki:.4f} kd={original.position_kd} | "
            f"speed kp={original.speed_kp:.4f} ki={original.speed_ki:.4f} | "
            f"current kp={original.current_kp} ki={original.current_ki} (not tuned here)"
        )
        # The inner speed loop, as the position stages will run it. One value
        # on the flag is a fixed override; several are swept first (see the
        # speed stage below), and the winner lands here.
        inner = {
            "speed_kp": (args.speed_kp[0] if args.speed_kp else original.speed_kp),
            "speed_ki": (args.speed_ki[0] if args.speed_ki else original.speed_ki),
        }

        # Holders to impedance, test joint to position mode. Both are mode
        # switches (a ~2 s reset each on MyActuator), so they happen once.
        await asyncio.gather(
            *[
                m.set_control_mode(ControlMode.IMPEDANCE)
                for j, m in motors.items()
                if j is not joint
            ]
        )
        await test.set_control_mode(ControlMode.POSITION_VELOCITY)
        await asyncio.sleep(1.0)

        # Read the planner setting before anything moves (read-only), so a
        # homing fallback can always put it back.
        accel_before: float | None = None
        try:
            accel_before = await test.motor._driver.get_acceleration()
        except Exception as e:
            print(f"  ! could not read the position-planning acceleration: {e}")

        # Every joint to rest before the sweep, distal to proximal, so a run
        # starts from a known pose rather than wherever the last run stopped
        # -- the holders otherwise hold whatever they were snapshotted at. The
        # swept joint is homed on the motor's stock planner with the observable
        # homing (fallback to direct tracking if it will not move).
        holders = _Holders(motors, joint, is_left, config)
        await holders.start()
        await asyncio.sleep(0.5)
        print("  Homing all joints to rest (distal to proximal) ...")
        for j in _HOME_ORDER:
            if j is joint:
                if not await _home_test_with_fallback(
                    test, joint, original, accel_before
                ):
                    raise RuntimeError(
                        f"{joint.value} would not come to rest before the sweep; "
                        "not sweeping from an unknown pose"
                    )
            elif j in motors:
                await holders.ramp_to(j, 0.0, _HOME_SPEED)

        if args.accel is not None:
            try:
                print(
                    f"  position-planning accel: "
                    f"{math.degrees(accel_before) if accel_before is not None else float('nan'):.0f} dps/s "
                    f"-> writing {args.accel:.0f}"
                )
                await test.motor._driver.set_acceleration(
                    math.radians(args.accel), allow_zero=True
                )
                readback = await test.motor._driver.get_acceleration()
                print(f"  read back: {math.degrees(readback):.0f} dps/s", end="")
                if abs(math.degrees(readback) - args.accel) > 1.0:
                    print(
                        "  ! the motor did not accept it — it clamps or refuses "
                        "out-of-range values, so direct tracking is not reachable "
                        "over CAN on this firmware."
                    )
                else:
                    print("  (accepted)")
                    if args.accel == 0.0:
                        print(
                            "  -> direct tracking mode: 0xA4 now chases the target "
                            "through its PI loop under the frame's speed limit."
                        )
            except Exception as e:
                print(f"  ! could not set acceleration: {e}")

        best: tuple[float, float, float, float] | None = None
        try:
            lo, hi = arm_limits(joint, is_left)
            target = (
                float(np.clip(math.radians(args.center), lo, hi))
                if args.center is not None
                else await test.get_position()
            )
            grav = GravityCompensator(config)
            q = np.zeros(len(ARM_JOINTS), dtype=np.float32)
            q[ARM_JOINTS.index(joint)] = target
            load = float(grav.gravity_arm(q, is_left=is_left)[ARM_JOINTS.index(joint)])
            print(f"  holding {math.degrees(target):.1f}° under {load:+.2f} Nm\n")
            if abs(load) < 0.3:
                print(
                    "  ! this pose is nearly unloaded — the joint will hold at "
                    "any gain. Pass --center for a loaded one.\n"
                )

            if args.mode == "track":
                print(
                    f"  streaming a {args.amp:.0f}° {args.freq:.2f} Hz sine at "
                    f"{args.rate:.0f} Hz for {args.duration:.0f}s\n"
                )
                print(
                    f"  {'position_kp':>12} {'position_ki':>12} {'rms err':>10} "
                    f"{'+/-':>9} {'max err':>10} {'lag':>9} {'ripple':>9} "
                    f"{'buzz':>8} {'holder pk/rms':>14}"
                )

                sweep_group = f"position-loop-{int(time.time())}-{joint.value}"

                async def point(kp: float, ki: float):
                    """One position-gain pair at the current inner-loop gains."""
                    await test.motor.set_gains(
                        replace(
                            original,
                            position_kp=kp,
                            position_ki=ki,
                            speed_kp=inner["speed_kp"],
                            speed_ki=inner["speed_ki"],
                        ),
                        persist=False,
                    )
                    # Per-gain holder drift. The holders run MIT impedance and
                    # are not rigid: at -90° the right elbow carries 5.46 Nm
                    # against 3.86 at -45°, a 1.41x load, but its stability
                    # cliff fell from 0.84-0.96 to 0.24-0.36, about 3.5x. That
                    # is far too steep for load-dependent friction, and the
                    # obvious candidate is the arm moving under the joint:
                    # a compliant base feeds the joint's own reaction torque
                    # back into its loop. If wobble tracks ripple, the limit
                    # is structural and no gain on this joint will fix it.
                    holders.reset_wobble()
                    trials = [
                        await _track(
                            test,
                            target,
                            math.radians(args.amp),
                            args.freq,
                            args.duration,
                            max_speed,
                            args.rate,
                        )
                        for _ in range(args.repeat)
                    ]
                    worst, wobble, wobble_rms = holders.reset_wobble()
                    # The motor's own protection is the last word. Its stall
                    # bit latches after STALL_TIME_LIMIT (1.5 s on this elbow)
                    # of being driven without moving -- a limit cycle at the
                    # cliff can do that -- and while it is set the motor ignores
                    # 0xA4, so every later point would measure a dead motor.
                    try:
                        status = await test.motor.get_error_code()
                    except Exception:
                        status = None
                    if status is not None and status.name not in ("OK", "NORMAL"):
                        print(
                            f"\n  !! motor reports {status.name} at position_kp={kp:.4f} "
                            f"position_ki={ki:.4f} -- its protection tripped. Stopping "
                            "the sweep; the flag is cleared once the arm is back at rest."
                        )
                        raise _MotorFaulted(status.name)
                    ok = [t for t in trials if math.isfinite(t[0])]
                    if not ok:
                        print(f"  {kp:12.4f} {ki:12.4f}   (no usable pass)")
                        return None
                    return (
                        float(np.mean([t[0] for t in ok])),
                        float(np.std([t[0] for t in ok])),
                        float(np.mean([t[1] for t in ok])),
                        float(np.mean([t[2] for t in ok])),
                        float(np.mean([t[3] for t in ok])),
                        float(np.mean([t[4] for t in ok])),
                        wobble,
                        wobble_rms,
                        worst,
                        ok[-1][5],
                    )

                def persist(kp: float, ki: float, m, noisy: bool) -> None:
                    """One dashboard run per gain point (kind ``position_loop``).

                    The sweep table above is what the operator reads on the
                    terminal; this is the same point as the workbench sees it:
                    a chartable target/actual pass plus the scorecard, linked
                    to the other points of the sweep by ``group``.
                    """
                    if not args.save_run:
                        return
                    rms, spread, mx, lag, ripple, buzz, wobble, w_rms, worst, series = m
                    run_id = save_run(
                        "position_loop",
                        series,
                        {
                            "rms": rms,
                            "rms_spread": spread,
                            "max": mx,
                            "lag_ms": lag,
                            "pos_ripple": ripple,
                            "buzz_a": buzz,
                            "holder_peak_deg": wobble,
                            "holder_rms_deg": w_rms,
                            "holder_joint": worst.value if worst is not None else None,
                            "oscillating": bool(noisy),
                        },
                        side="left" if is_left else "right",
                        joint=joint.value,
                        gains={
                            "position_kp": kp,
                            "position_ki": ki,
                            "speed_kp": inner["speed_kp"],
                            "speed_ki": inner["speed_ki"],
                        },
                        params={
                            "mode": "track",
                            "center_deg": math.degrees(target),
                            "amp_deg": args.amp,
                            "freq_hz": args.freq,
                            "duration_s": args.duration,
                            "rate_hz": args.rate,
                            "repeat": args.repeat,
                            "accel": args.accel,
                            "wire": "0xA4 direct tracking",
                        },
                        label=args.label,
                        group=sweep_group,
                    )
                    print(f"      saved run {run_id}")

                def row(kp: float, ki: float, m, noisy: bool) -> None:
                    (
                        rms,
                        spread,
                        mx,
                        lag,
                        ripple,
                        buzz,
                        wobble,
                        w_rms,
                        worst,
                        _series,
                    ) = m
                    tag = "  <- oscillating" if noisy else ""
                    # Name the holder only when it moved enough to matter: a
                    # joint's own loop cannot be blamed for a base that is
                    # moving as far as the error being measured.
                    if worst is not None and wobble > rms:
                        # Peak vs rms says which kind of movement it is: a
                        # holder that settles once and sits there has rms near
                        # its peak, a holder genuinely wobbling has rms well
                        # below it. Only the second can destabilise the joint.
                        kind = "sagged" if w_rms > 0.7 * wobble else "wobbled"
                        tag += f"  ({worst.value} base {kind} {wobble:.2f}°)"
                    print(
                        f"  {kp:12.4f} {ki:12.4f} {rms:9.4f}° {spread:8.4f}° "
                        f"{mx:9.4f}° {lag:8.1f}ms {ripple:8.4f}° {buzz:7.3f}A "
                        f"{wobble:6.2f}/{w_rms:.2f}°" + tag
                    )

                def oscillating(m, prev: tuple[float, float] | None) -> bool:
                    """Has this gain started a limit cycle?

                    Current first, position second. On the right elbow at
                    kp=0.36, adding position_ki=0.00036 took position ripple
                    *down* (0.0664 -> 0.0615°) while the joint visibly
                    oscillated: the cycle is faster than this loop's ~50 Hz
                    position sampling and aliases away. The q-axis current
                    sees it, because a cycle that is a fraction of an encoder
                    count still swings amps.

                    Both jump tests are relative -- neither channel has a
                    calibrated absolute level. A permissive absolute ceiling
                    stays on position so a runaway still stops the sweep.
                    """
                    ripple, buzz = m[4], m[5]
                    if ripple > args.ripple_limit:
                        return True
                    if prev is None:
                        return False
                    p_ripple, p_buzz = prev
                    if (
                        np.isfinite(buzz)
                        and np.isfinite(p_buzz)
                        and buzz > _TRACK_BUZZ_JUMP * max(p_buzz, 1e-3)
                    ):
                        return True
                    return ripple > _TRACK_RIPPLE_JUMP * max(p_ripple, 1e-4)

                # Stage 0: the inner speed loop, innermost first. The
                # position loop's stability cliff is bounded by the phase
                # margin of the speed loop under it -- on the right elbow
                # position_kp fell off between 0.84-0.96 at -45° and 0.24-0.36
                # at -90° with the speed loop at factory, and no outer gain can
                # buy back margin the inner loop does not have. Swept at the
                # configured position gains, ascending, same buzz/ripple stop;
                # the winner is what every later stage runs on.
                for name in ("speed_kp", "speed_ki"):
                    values = getattr(args, name)
                    if not values or len(values) < 2:
                        continue
                    print(
                        f"\n  speed-loop stage: sweeping {name} at position kp={original.position_kp:.4f}"
                    )
                    best_inner: tuple[float, float] | None = None
                    prev_inner: tuple[float, float] | None = None
                    for v in values:
                        inner[name] = v
                        m = await point(original.position_kp, original.position_ki)
                        if m is None:
                            continue
                        noisy = oscillating(m, prev_inner)
                        print(f"    {name}={v:.4f}", end="")
                        row(original.position_kp, original.position_ki, m, noisy)
                        persist(original.position_kp, original.position_ki, m, noisy)
                        if noisy:
                            print(
                                f"    stopping: the speed loop is buzzing at {name}={v:.4f}."
                            )
                            break
                        prev_inner = (m[4], m[5])
                        if best_inner is None or m[0] < best_inner[1]:
                            best_inner = (v, m[0])
                    if best_inner is not None:
                        inner[name] = best_inner[0]
                        print(
                            f"  -> {name}={best_inner[0]:.4f} (tracking rms {best_inner[1]:.4f}°)"
                        )
                    else:
                        inner[name] = values[0]
                print(
                    f"\n  position stages run with speed kp={inner['speed_kp']:.4f} ki={inner['speed_ki']:.4f}"
                )

                kps = args.kp or [original.position_kp * m for m in _KP_STEPS]
                prev_ripple: tuple[float, float] | None = None
                for kp in kps:
                    try:
                        m = await point(kp, 0.0)
                    except _MotorFaulted:
                        break
                    if m is None:
                        continue
                    noisy = oscillating(m, prev_ripple)
                    row(kp, 0.0, m, noisy)
                    persist(kp, 0.0, m, noisy)
                    if noisy:
                        print(
                            "    stopping: past here the joint buys tracking "
                            "accuracy with vibration."
                        )
                        break
                    prev_ripple = (m[4], m[5])
                    if best is None or m[0] < best[2]:
                        best = (kp, 0.0, m[0], m[2])

                # Then the integral, at the kp that won. This is the term that
                # removes the velocity-following error a P-only loop cannot
                # avoid -- and it does it without pushing position_kp toward
                # the cliff, which on the right elbow moved from above 0.84 at
                # -45° to below 0.72 at -90° as gravity rose 3.86 -> 5.46 Nm.
                # Ascending and stopping on ripple, because an integral
                # winding up against stiction limit-cycles rather than
                # diverging, and this joint's friction rises with that load.
                if best is not None:
                    kis = (
                        args.ki
                        if args.ki is not None
                        else [best[0] * s for s in _KI_STEPS]
                    )
                    prev_ripple = None
                    for ki in kis:
                        try:
                            m = await point(best[0], ki)
                        except _MotorFaulted:
                            break
                        if m is None:
                            continue
                        noisy = oscillating(m, prev_ripple)
                        row(best[0], ki, m, noisy)
                        persist(best[0], ki, m, noisy)
                        if noisy:
                            print(
                                "    stopping: the integral is winding up "
                                "against friction faster than it is helping."
                            )
                            break
                        prev_ripple = (m[4], m[5])
                        if m[0] < best[2]:
                            best = (best[0], ki, m[0], m[2])

                    print(
                        f"\n  best: position_kp={best[0]:.4f} "
                        f"position_ki={best[1]:.4f} "
                        f"(speed kp={inner['speed_kp']:.4f} ki={inner['speed_ki']:.4f}) "
                        f"-> tracking rms {best[2]:.4f}°"
                    )
                    if best[2] > 1.0:
                        print(
                            "  ! still not tracking. If the acceleration read back "
                            "non-zero above, that is why: profiled motion cannot "
                            "follow a stream at any gain."
                        )
                return

            kps = args.kp or [original.position_kp * m for m in _KP_STEPS]
            print(f"  {'position_kp':>12} {'position_ki':>12} {'sag':>9} {'ripple':>9}")
            for kp in kps:
                await test.motor.set_gains(
                    replace(original, position_kp=kp, position_ki=0.0), persist=False
                )
                sag, ripple = await _measure(test, target, max_speed)
                flag = ""
                if ripple > _RIPPLE_LIMIT_DEG:
                    flag = "  <- oscillating, stopping"
                elif sag <= args.sag_deg:
                    flag = "  <- holds"
                print(f"  {kp:12.4f} {0.0:12.4f} {sag:8.3f}° {ripple:8.3f}°{flag}")
                if ripple > _RIPPLE_LIMIT_DEG:
                    break
                if best is None or sag < best[2]:
                    best = (kp, 0.0, sag, ripple)
                if sag <= args.sag_deg:
                    break

            if best is not None and best[2] > args.sag_deg:
                # P alone left a droop; that is what the integral is for.
                kis = args.ki if args.ki is not None else [best[0] / 100.0]
                for ki in kis:
                    await test.motor.set_gains(
                        replace(original, position_kp=best[0], position_ki=ki),
                        persist=False,
                    )
                    sag, ripple = await _measure(test, target, max_speed)
                    flag = "  <- oscillating" if ripple > _RIPPLE_LIMIT_DEG else ""
                    print(
                        f"  {best[0]:12.4f} {ki:12.4f} {sag:8.3f}° {ripple:8.3f}°{flag}"
                    )
                    if ripple <= _RIPPLE_LIMIT_DEG and sag < best[2]:
                        best = (best[0], ki, sag, ripple)

            print(
                "\n  holder peak wobble: "
                + ", ".join(
                    f"{j.value} {v:.2f}°" for j, v in holders.peak_wobble.items()
                )
            )
            if best is None:
                print("  No usable candidate.")
            else:
                kp, ki, sag, ripple = best
                print(
                    f"\n  best: position_kp={kp:.4f} position_ki={ki:.4f} "
                    f"-> sag {sag:.3f}° ripple {ripple:.3f}°"
                )
                if sag > args.sag_deg:
                    print(
                        "  ! still sagging past tolerance — raise --kp further or "
                        "add --ki, but check the ripple column before trusting it."
                    )
                if args.save:
                    await test.motor.set_gains(
                        replace(original, position_kp=kp, position_ki=ki), persist=True
                    )
                    print("  committed to ROM (0x32).")
                else:
                    print("  not saved; re-run with --save to commit.")
        finally:
            # Home before disabling. The holders keep streaming impedance
            # while their targets are walked to rest, so nothing is
            # unsupported mid-move; the test joint rides its own position
            # command. Distal to proximal, so each shoulder swings a folded
            # arm. Best-effort: a failure here must not skip the disable.
            # Order matters, and it used to be wrong. Homing ran FIRST, under
            # whatever gains the sweep had just left in RAM -- often the ones
            # it stopped at for buzzing -- and in direct-tracking mode, then
            # the stored gains and planner were restored. A loaded elbow asked
            # to travel 90 deg on a buzzing or too-weak loop stalls; the motor's
            # stall protection latches after STALL_TIME_LIMIT (1.5 s) and the
            # motor then ignores 0xA4, so it never arrived, the code fell
            # through to disable, and the next run met a faulted motor.
            #
            # Now: restore the stored gains and planner first, so homing runs
            # on the profiled-motion planner the motor ships with; home and
            # VERIFY; clear the protection flag once at rest (0x9B, best
            # effort); only then release.
            if accel_before is not None and not args.save:
                try:
                    await test.motor._driver.set_acceleration(
                        accel_before, allow_zero=True
                    )
                    print(
                        f"  acceleration restored to "
                        f"{math.degrees(accel_before):.0f} dps/s."
                    )
                except Exception as e:
                    print(
                        f"  ! could not restore acceleration ({e}) — it was "
                        f"{math.degrees(accel_before):.0f} dps/s."
                    )
            if not args.save:
                try:
                    await test.motor.set_gains(original, persist=False)
                    print("  RAM gains restored to the values found at start.")
                except Exception as e:
                    print(
                        f"  ! could not restore RAM gains ({e}) — power-cycle to reset."
                    )
            # Homing must be OBSERVABLE and self-correcting. Bench: after the
            # planner and stock gains were restored, a 0xA4 to rest was
            # acknowledged and then ignored -- the elbow sat at its last
            # target for 15 s at constant torque while the tool waited.
            # Yesterday's teardown homed under direct tracking (accel 0) and
            # never failed, so that is the fallback. The target is re-sent
            # continuously, progress is logged every 2 s, and a joint that has
            # not moved in 3 s switches strategy instead of waiting.
            print("  Returning to rest ...")
            at_rest = False
            stop_requested = False

            def _on_term(*_: object) -> None:
                nonlocal stop_requested
                stop_requested = True

            import signal
            import sys

            loop_ = asyncio.get_running_loop()
            try:
                loop_.add_signal_handler(signal.SIGTERM, _on_term)
            except (NotImplementedError, RuntimeError):
                pass
            try:
                for j in _HOME_ORDER:
                    if j is joint:
                        at_rest = await _home_test_with_fallback(
                            test,
                            joint,
                            original,
                            accel_before,
                            best[0] if best else 0.0,
                        )
                    elif j in motors:
                        await holders.ramp_to(j, 0.0, _HOME_SPEED)
            except Exception as e:
                print(f"  ! could not return to rest ({e}) — support the arm.")
            interactive = sys.stdin is not None and sys.stdin.isatty()
            while not at_rest:
                try:
                    status = (await test.motor.get_error_code()).name
                except Exception:
                    status = "unreadable"
                print(
                    f"\n  !! {joint.value} is NOT at rest (motor status {status}) -- "
                    "holders are streaming, not releasing."
                )
                if interactive:
                    answer = await asyncio.to_thread(
                        input,
                        "  Support the arm, then press Enter to retry homing "
                        "(or type 'drop' to release anyway): ",
                    )
                    if answer.strip().lower() == "drop":
                        print("  releasing on operator request.")
                        break
                else:
                    # Launched from the dashboard: nobody can type. Keep
                    # holding and retry; Stop from the UI ends the wait.
                    if stop_requested:
                        print(
                            "  stop requested from the dashboard; one last homing "
                            "attempt, then releasing -- SUPPORT THE ARM."
                        )
                        at_rest = await _home_test_with_fallback(
                            test,
                            joint,
                            original,
                            accel_before,
                            best[0] if best else 0.0,
                        )
                        break
                    print(
                        "  retrying homing in 5 s (press Stop in the dashboard to end)"
                    )
                    await asyncio.sleep(5.0)
                    if stop_requested:
                        continue
                await test.motor.clear_errors()
                at_rest = await _home_test_with_fallback(
                    test, joint, original, accel_before, best[0] if best else 0.0
                )
            # At rest and unloaded the stall condition is gone: clear the flag
            # so the next run does not start on a faulted motor, and say what
            # the motor reports either way.
            await test.motor.clear_errors()
            try:
                print(
                    f"  motor status at rest: {(await test.motor.get_error_code()).name}"
                )
            except Exception:
                pass
            await holders.stop()
            await asyncio.gather(
                *[m.disable() for m in raw.values()], return_exceptions=True
            )
