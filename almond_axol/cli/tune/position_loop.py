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
from ...tuning import joint_frame_motors
from ...tuning.joint_frame import JointFrameMotor
from ...utils.logquiet import quiet_noisy_loggers
from ..motor import add_side_and_channel_arguments, resolve_channel

_HOLD_HZ = 100.0
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
_SETTLE_S = 2.0
_MEASURE_S = 2.0


class _Holders:
    """Streams MIT impedance + gravity feedforward to the non-test joints.

    The point of the whole command is that the firmware position servo cannot
    hold a loaded joint, so the holders must not use it. Impedance carries its
    gains in every frame and is unaffected by the stored gains being searched.
    """

    def __init__(self, motors, exclude: Joint, is_left: bool, config: AxolConfig):
        self._motors = motors
        self._exclude = exclude
        self._is_left = is_left
        self._arm = config.left if is_left else config.right
        self._gravity = GravityCompensator(config)
        self._hold: dict[Joint, float] = {}
        self._task: asyncio.Task | None = None
        self.peak_wobble: dict[Joint, float] = {}

    async def start(self) -> None:
        for j, m in self._motors.items():
            if j is self._exclude:
                continue
            self._hold[j] = await m.get_position()
        self.peak_wobble = {j: 0.0 for j in self._hold}
        self._task = asyncio.create_task(self._loop())

    async def ramp_to(self, joint: Joint, target: float, speed: float) -> None:
        """Walk one holder's target to ``target`` while it keeps streaming.

        The hold loop keeps commanding throughout, so the joint is under
        impedance the whole way — unlike a one-shot position command, which
        is what leaves an arm unsupported.
        """
        if joint not in self._hold:
            return
        start = self._hold[joint]
        dist = abs(target - start)
        if dist < 1e-4:
            return
        secs = dist / max(speed, 1e-3)
        t0 = time.monotonic()
        while True:
            frac = (time.monotonic() - t0) / secs
            if frac >= 1.0:
                break
            self._hold[joint] = start + (target - start) * frac
            await asyncio.sleep(0.01)
        self._hold[joint] = target
        await asyncio.sleep(0.3)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        dt = 1.0 / _HOLD_HZ
        q = np.zeros(len(ARM_JOINTS), dtype=np.float32)
        while True:
            t0 = time.monotonic()
            for i, j in enumerate(ARM_JOINTS):
                if j in self._hold:
                    q[i] = self._hold[j]
            grav = self._gravity.gravity_arm(q, is_left=self._is_left)
            sends = []
            for i, j in enumerate(ARM_JOINTS):
                if j not in self._hold:
                    continue
                jc = getattr(self._arm, j.value)
                sends.append(
                    self._motors[j].set_impedance(
                        self._hold[j], 0.0, jc.kp, jc.kd, float(grav[i])
                    )
                )
            await asyncio.gather(*sends, return_exceptions=True)
            for j in self._hold:
                try:
                    drift = abs(self._motors[j].position - self._hold[j])
                except Exception:
                    continue
                self.peak_wobble[j] = max(self.peak_wobble[j], math.degrees(drift))
            spent = time.monotonic() - t0
            if spent < dt:
                await asyncio.sleep(dt - spent)


async def _track(
    motor: JointFrameMotor,
    center: float,
    amp: float,
    freq: float,
    secs: float,
    max_speed: float,
    rate_hz: float,
) -> tuple[float, float, float, float]:
    """Stream 0xA4 along a sine; return ``(rms, max, lag_ms, ripple)`` in deg.

    This is the test that matters for ``wire_mode``: a *held* position says
    nothing about whether the firmware loop can follow a target that keeps
    moving. In profiled-motion mode it cannot — every frame restarts a ramp —
    so a joint that holds perfectly can still track nothing at all.
    """
    # Reach the sine's start before timing anything. Without this the first
    # entry of a sweep measures the joint travelling to `center` from
    # wherever it was parked — tens of degrees over the whole window — and
    # reports it as a tracking failure. Every later entry then starts on
    # target and looks fine, so the artefact lands on whichever gain happens
    # to be measured first.
    approach_deadline = time.monotonic() + _APPROACH_MAX_S
    while time.monotonic() < approach_deadline:
        await motor.set_position_velocity(center, max_speed)
        try:
            if abs(await motor.get_position() - center) < _APPROACH_TOL_RAD:
                break
        except Exception:
            pass
        await asyncio.sleep(0.02)
    else:
        print(
            f"    (did not reach {math.degrees(center):.1f}° within "
            f"{_APPROACH_MAX_S:.0f}s — the result below includes the approach)"
        )
    await asyncio.sleep(0.3)

    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    tt: list[float] = []
    tgt: list[float] = []
    act: list[float] = []
    while True:
        now = time.monotonic() - t0
        if now >= secs:
            break
        target = center + amp * math.sin(2.0 * math.pi * freq * now)
        await motor.set_position_velocity(target, max_speed)
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
    a = np.array(act)
    g = np.array(tgt)
    good = np.isfinite(a)
    achieved = len(tt) / max(tt[-1] - tt[0], 1e-9) if len(tt) > 1 else 0.0
    if achieved < 0.8 * rate_hz:
        print(
            f"    (achieved {achieved:.0f} Hz of the requested {rate_hz:.0f} — two "
            f"CAN round trips per sample; lower --rate for a clean cadence)"
        )
    if good.sum() < 20:
        print(f"    ({good.sum()} of {len(a)} position reads succeeded)")
        return float("nan"), float("nan"), float("nan")
    err = np.degrees(g[good] - a[good])
    v = np.gradient(g[good], np.array(tt)[good])
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
    smooth = np.convolve(err, np.ones(k) / k, mode="same")
    ripple = float((err - smooth)[k:-k].std()) if len(err) > 3 * k else float("nan")
    return float(err.std()), float(np.abs(err).max()), float(lag), ripple


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
    p.add_argument("--kp", type=float, nargs="+", help="Explicit position_kp values")
    p.add_argument(
        "--ki",
        type=float,
        nargs="+",
        default=None,
        help="position_ki values to try at the winning kp (default: 0 and "
        "kp/100 — an integral is what removes the last of the sag)",
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
            f"  current gains: kp={original.position_kp:.4f} "
            f"ki={original.position_ki:.4f} kd={original.position_kd}"
        )

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

        accel_before: float | None = None
        if args.accel is not None:
            try:
                accel_before = await test.motor._driver.get_acceleration()
                print(
                    f"  position-planning accel: {math.degrees(accel_before):.0f} dps/s "
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

        holders = _Holders(motors, joint, is_left, config)
        await holders.start()
        await asyncio.sleep(0.5)

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
                    f"  {'position_kp':>12} {'rms err':>10} {'max err':>10} "
                    f"{'lag':>9} {'ripple':>9}"
                )
                kps = args.kp or [original.position_kp * m for m in _KP_STEPS]
                prev_ripple: float | None = None
                for kp in kps:
                    await test.motor.set_gains(
                        replace(original, position_kp=kp), persist=False
                    )
                    rms, mx, lag, ripple = await _track(
                        test,
                        target,
                        math.radians(args.amp),
                        args.freq,
                        args.duration,
                        max_speed,
                        args.rate,
                    )
                    noisy = ripple > args.ripple_limit or (
                        prev_ripple is not None
                        and ripple > _TRACK_RIPPLE_JUMP * max(prev_ripple, 1e-4)
                    )
                    print(
                        f"  {kp:12.4f} {rms:9.4f}° {mx:9.4f}° {lag:8.1f}ms "
                        f"{ripple:8.4f}°" + ("  <- oscillating" if noisy else "")
                    )
                    if noisy:
                        print(
                            "    stopping: past here the joint buys tracking "
                            "accuracy with vibration."
                        )
                        break
                    prev_ripple = ripple
                    if best is None or rms < best[2]:
                        best = (kp, 0.0, rms, mx)
                if best is not None:
                    print(
                        f"\n  best: position_kp={best[0]:.4f} -> tracking rms "
                        f"{best[2]:.4f}°"
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
            try:
                print("  Returning to rest ...")
                for j in _HOME_ORDER:
                    if j is joint:
                        await test.set_position_velocity(0.0, math.radians(20.0))
                        for _ in range(60):
                            if abs(await test.get_position()) < 0.02:
                                break
                            await asyncio.sleep(0.1)
                    elif j in motors:
                        await holders.ramp_to(j, 0.0, _HOME_SPEED)
            except Exception as e:
                print(f"  ! could not return to rest ({e}) — support the arm.")
            await holders.stop()
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
            await asyncio.gather(
                *[m.disable() for m in raw.values()], return_exceptions=True
            )
