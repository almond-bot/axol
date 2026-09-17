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
#: Multipliers applied to the joint's current `position_kp`, low to high. The
#: search stops at the first value that holds, so a joint that is nearly right
#: never sees the large ones.
_KP_STEPS = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)
#: Sag (deg) at or below which the joint counts as holding.
_SAG_OK_DEG = 0.3
#: Position ripple (deg rms) above which a candidate is called oscillating and
#: the search stops rather than escalating into a louder instability.
_RIPPLE_LIMIT_DEG = 0.25
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
            await holders.stop()
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
