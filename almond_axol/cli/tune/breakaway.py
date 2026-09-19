"""
axol tune.breakaway

Measure a joint's **breakaway** (static) torque — the torque at which a
stationary joint first moves — and compare it to the sliding Coulomb torque
``fc`` the friction model already knows.

Why this exists, and why ``tune.friction`` cannot answer it:

* ``tune.friction`` sweeps at fixed velocities and fits
  ``fc·tanh(0.1·k·v) + fv·v``. That model is monotonic in ``v`` and has no
  static/kinetic distinction — it cannot represent breakaway at any
  parameter value.
* Its sweep bottoms out around 0.13 rad/s, and one LSB of reported motor
  velocity is ~0.022 rad/s. Creep at a few thousandths of a rad/s is below
  the velocity channel entirely, so a "slower sweep" fits torque against a
  signal that reads zero.

Stick-slip exists precisely *because* breakaway exceeds sliding friction: a
joint tracking a slow target sticks until ``kp·err`` makes up the gap, lets
go, overshoots, and re-sticks — the 2 Hz, ~0.5° stairs on the X8-P20
shoulders. That gap is what this measures, and it needs no velocity signal at
all — only the torque at the instant position moves.

Method, per pose:

1. Home the arm, apply the shared sweep-safety clearances, and ramp the test
   joint to the pose under impedance control.
2. Drop to ``kp = 0`` — no position spring — with a small ``kd`` so the joint
   cannot run away once it releases. Gravity feedforward is then the only
   thing holding it.
3. Trim that feedforward until the joint stands still. A joint sits still for
   any trim inside its stiction band, so this converges quickly; without it
   loaded poses are unmeasurable (at 15 Nm a 2 % gravity-model error
   outweighs the whole breakaway torque).
4. Ramp an extra torque up and back down in a triangle and watch for the
   first motion past ``--move-deg``. Peaks escalate over attempts and the
   search stops at the first release, so the joint never sees more torque
   than it took to move it; the instant it moves it is caught under ``kp``
   and returned to the pose.
5. Repeat in both directions, ``--trials`` times each.

Probing both directions separates two things one average hides — the
symmetric half is friction, the antisymmetric half is whatever standing
torque the trimmed feedforward still missed:

* **F_static** and ``F_static / fc``. The excess over ``fc`` is the torque a
  pure velocity feedforward can never supply while the joint is stuck, and
  ``(F_static − fc) / kp`` is the predicted stick-slip stair height. If it
  matches the stairs a slow ``tune.motion`` replay shows, the diagnosis is
  closed.
* ``stiction_gain`` (see ``JointConfig``) should stay **below**
  ``F_static / fc − 1`` … the compensation must not exceed the friction it
  is compensating, or the joint hunts around the target at rest.

Examples:
    axol tune.breakaway --r --joint shoulder_1
    axol tune.breakaway --r --joint shoulder_2 --trials 5 --csv ~/breakaway-s2.csv
    axol tune.breakaway --r --joint shoulder_1 --poses -30 0 30   # load dependence
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...motor import CanBus, ControlMode, Motor, MotorError
from ...robot.config import ArmConfig, AxolConfig
from ...robot.gravity import GravityCompensator
from ...tuning import (
    JointFrameMotor,
    joint_frame_motors,
    ramp_impedance,
    ramp_stages,
    safe_limits,
    safe_outboard_direction,
    sweep_safety,
)
from ..motor import add_side_and_channel_arguments, resolve_channel
from .friction import _home_all, _ramp_verified

_RATE_HZ = 100.0
#: Drift (rad) over one trim hold below which the joint counts as standing
#: still: ~1.4 LSB of the 16-bit MIT position, i.e. the noise floor.
_HOLD_TOL = math.radians(0.03)
#: Duration of one kp = 0 trim hold, and the settle time skipped at its start.
_TRIM_HOLD_S = 1.0
_TRIM_SETTLE_S = 0.25
#: Displacement (rad) from the pose at which a kp = 0 phase is abandoned and
#: the joint is caught under kp — the "it let go" guard for the trim search
#: and the backstop for a release the ramp somehow missed.
_CATCH = math.radians(3.0)
#: Joints with a base-collision boundary at 0 are probed at least this far
#: outboard so a release toward the boundary cannot cross it.
_BOUNDARY_MARGIN = math.radians(5.0)
#: Floor on the friction level the escalation schedule is scaled by, so the
#: low-friction Damiao wrists (fc ≈ 0.1 Nm) still get a usable ramp.
_FC_FLOOR_NM = 0.2
#: Escalating ramp peaks as multiples of the (floored) fc, cut at --max-torque.
_PEAK_MULTIPLES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def peak_schedule(fc: float, max_multiple: float) -> list[float]:
    """Ramp peaks (Nm) to try in order: ``_PEAK_MULTIPLES × max(fc, floor)``,
    keeping every peak at or below ``max_multiple × max(fc, floor)``."""
    ref = max(fc, _FC_FLOOR_NM)
    return [m * ref for m in _PEAK_MULTIPLES if m <= max_multiple + 1e-9]


def triangle(t: float, ramp_s: float) -> float:
    """Unit triangle: 0 → 1 at ``ramp_s / 2`` → 0 at ``ramp_s``; 0 outside."""
    if t <= 0.0 or t >= ramp_s:
        return 0.0
    half = ramp_s / 2.0
    return t / half if t <= half else (ramp_s - t) / half


def split_breakaway(plus: list[float], minus: list[float]) -> tuple[float, float]:
    """``(F_static, bias)`` from the released torques in each direction.

    With the joint held by ``g_model + trim`` and the true standing torque
    ``g_true``, the residual ``r = g_model + trim − g_true`` helps one
    direction and hinders the other: ``b+ = F_static − r``,
    ``b− = F_static + r``. So the symmetric half is friction and the
    antisymmetric half is the residual the trim search left inside the
    stiction band (``r`` is what the feedforward *over*-supplies in the +
    direction).
    """
    bp = float(np.mean(plus))
    bm = float(np.mean(minus))
    return (bp + bm) / 2.0, (bm - bp) / 2.0


def predicted_stair_deg(f_static: float, fc: float, kp: float) -> float:
    """Stick-slip stair height (deg) a pure velocity feedforward leaves:
    the joint sticks until ``kp·err`` covers what ``fc`` did not."""
    if kp <= 0.0:
        return math.nan
    return math.degrees(max(f_static - fc, 0.0) / kp)


@dataclass
class TrimSearch:
    """Bisection-style search for the feedforward trim that holds the joint.

    Feed one drift measurement (rad over a hold) per step; :attr:`trim` is the
    correction to apply next. The step shrinks whenever the drift changes
    sign, so the search closes on an edge of the stiction band. ``done`` is
    set on a hold that stood still; ``failed`` when the trim runs away
    (gravity model far off, or the joint is not actually free).
    """

    step: float
    max_trim: float
    trim: float = 0.0
    done: bool = False
    failed: bool = False
    _last_sign: int = 0
    steps: int = field(default=0)

    def update(self, drift: float) -> float:
        if abs(drift) < _HOLD_TOL:
            self.done = True
            return self.trim
        sign = 1 if drift > 0 else -1
        if self._last_sign and sign != self._last_sign:
            self.step /= 2.0
        self._last_sign = sign
        # Drifting + means the applied torque exceeds the standing load.
        self.trim -= sign * self.step
        self.steps += 1
        if abs(self.trim) > self.max_trim or self.steps > 16:
            self.failed = True
        return self.trim


async def _hold(
    motor: JointFrameMotor,
    pose: float,
    kp: float,
    kd: float,
    t_ff: float,
    duration: float,
) -> None:
    """Command ``(pose, kp, kd, t_ff)`` at the probe rate for ``duration``."""
    period = 1.0 / _RATE_HZ
    deadline = time.perf_counter()
    end = deadline + duration
    while deadline < end:
        deadline += period
        await motor.set_impedance(pose, 0.0, kp, kd, t_ff)
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))


async def _catch(
    motor: JointFrameMotor,
    pose: float,
    kp: float,
    kd: float,
    gravity_fn,
) -> None:
    """Take a released joint back under ``kp`` and return it to ``pose``."""
    here = motor.position
    await _hold(motor, here, kp, kd, gravity_fn(here), 0.4)
    await ramp_impedance(motor, kp, kd, pose, gravity_fn, rate_hz=_RATE_HZ)
    await _hold(motor, pose, kp, kd, gravity_fn(pose), 0.5)


async def _trim(
    motor: JointFrameMotor,
    pose: float,
    kp: float,
    kd: float,
    kd_probe: float,
    gravity_fn,
    fc_ref: float,
    trim0: float = 0.0,
) -> float | None:
    """Find the feedforward trim (Nm) that holds the joint still at kp = 0."""
    search = TrimSearch(step=0.15 * fc_ref, max_trim=3.0 * fc_ref, trim=trim0)
    g = gravity_fn(pose)
    while True:
        period = 1.0 / _RATE_HZ
        deadline = time.perf_counter()
        t0 = deadline
        p_start: float | None = None
        caught = False
        while time.perf_counter() - t0 < _TRIM_HOLD_S:
            deadline += period
            await motor.set_impedance(pose, 0.0, 0.0, kd_probe, g + search.trim)
            pos = motor.position
            if abs(pos - pose) > _CATCH:
                caught = True
                break
            if p_start is None and time.perf_counter() - t0 >= _TRIM_SETTLE_S:
                p_start = pos
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        if caught:
            drift = motor.position - pose
            await _catch(motor, pose, kp, kd, gravity_fn)
        else:
            drift = motor.position - (p_start if p_start is not None else pose)
        trim = search.update(drift)
        print(
            f"      trim {trim:+.3f} Nm  drift {math.degrees(drift):+.3f}°"
            f"{'  (let go — caught)' if caught else ''}"
        )
        if search.done:
            return trim
        if search.failed:
            return None


async def _release(
    motor: JointFrameMotor,
    pose: float,
    direction: int,
    peak: float,
    ramp_s: float,
    kd_probe: float,
    hold_ff: float,
    move_thr: float,
    writer: csv.writer | None,
    tag: tuple,
) -> float | None:
    """One triangle ramp of ``direction × peak``; the extra torque at first
    motion past ``move_thr``, or ``None`` if the joint never moved."""
    period = 1.0 / _RATE_HZ
    t0 = time.perf_counter()
    deadline = t0
    pos0 = motor.position
    while True:
        t = time.perf_counter() - t0
        if t >= ramp_s:
            return None
        deadline += period
        extra = direction * peak * triangle(t, ramp_s)
        await motor.set_impedance(pose, 0.0, 0.0, kd_probe, hold_ff + extra)
        pos = motor.position
        moved = pos - pos0
        try:
            tau = motor.torque
        except MotorError:
            tau = math.nan
        if writer is not None:
            writer.writerow(
                [
                    *tag,
                    f"{t:.4f}",
                    f"{extra:.5f}",
                    f"{math.degrees(pos):.5f}",
                    f"{tau:.4f}",
                ]
            )
        if abs(moved) > move_thr:
            return abs(extra)
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.breakaway`` subcommand."""
    p = subparsers.add_parser(
        "tune.breakaway",
        help="Measure a joint's static (breakaway) friction against its sliding fc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        help="Joint to probe",
    )
    p.add_argument(
        "--poses",
        type=float,
        nargs="+",
        default=None,
        metavar="DEG",
        help="Joint-frame poses (degrees, 0 = rest) to probe at (default: 0; "
        "base-collision joints are pushed 5° outboard of their boundary)",
    )
    p.add_argument(
        "--trials", type=int, default=3, help="Releases per direction (default: 3)"
    )
    p.add_argument(
        "--ramp-s",
        type=float,
        default=4.0,
        help="Seconds for one up-and-back torque ramp (default: 4)",
    )
    p.add_argument(
        "--max-torque",
        type=float,
        default=3.0,
        help="Largest ramp peak to try, as a multiple of the joint's fc "
        "(floored at 0.2 Nm; default: 3.0)",
    )
    p.add_argument(
        "--move-deg",
        type=float,
        default=0.06,
        help="Motion (degrees) that counts as a release — about 3 LSB of the "
        "16-bit MIT position (default: 0.06)",
    )
    p.add_argument(
        "--kd",
        type=float,
        default=1.0,
        help="Firmware damping during the kp = 0 phases, so a released joint "
        "creeps rather than runs (default: 1.0)",
    )
    p.add_argument(
        "--csv", type=Path, default=None, help="Dump every ramp sample to this CSV"
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(_run(args))


def _probe_poses(
    joint: Joint, is_left: bool, requested: list[float] | None
) -> list[float]:
    """Requested poses (rad), clamped inside the safe range with a margin, and
    pushed outboard of a base-collision boundary."""
    lo, hi = safe_limits(joint, is_left)
    margin = _CATCH + math.radians(2.0)
    lo_ok, hi_ok = lo + margin, hi - margin
    outboard = safe_outboard_direction(joint, is_left)
    if outboard is not None:
        if outboard > 0:
            lo_ok = max(lo_ok, _BOUNDARY_MARGIN)
        else:
            hi_ok = min(hi_ok, -_BOUNDARY_MARGIN)
    poses = [math.radians(d) for d in (requested if requested is not None else [0.0])]
    out = []
    for q in poses:
        clamped = min(max(q, lo_ok), hi_ok)
        if abs(clamped - q) > 1e-9:
            print(
                f"  ! pose {math.degrees(q):+.1f}° moved to {math.degrees(clamped):+.1f}° "
                "(safe range with catch margin)"
            )
        out.append(clamped)
    return out


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side = "left" if is_left else "right"
    resolved = AxolConfig().resolved()
    arm_cfg: ArmConfig = resolved.left if is_left else resolved.right
    gains = getattr(arm_cfg, joint.value)
    kp, kd = gains.kp, gains.kd
    fc = gains.friction.fc
    fc_ref = max(fc, _FC_FLOOR_NM)
    peaks = peak_schedule(fc, args.max_torque)
    move_thr = math.radians(args.move_deg)
    poses = _probe_poses(joint, is_left, args.poses)
    joint_index = ARM_JOINTS.index(joint)

    print(f"\nAxol breakaway (static friction) probe — {side} {joint.value}")
    print(
        f"  config: kp={kp:g} kd={kd:g}  fc={fc:.3f} Nm  "
        f"stiction_gain={gains.stiction_gain:g}"
    )
    print(
        f"  ramp peaks: {', '.join(f'{p:.2f}' for p in peaks)} Nm over {args.ramp_s:g} s; "
        f"release at {args.move_deg:g}°; probe kd={args.kd:g}"
    )

    gravity_comp = GravityCompensator(resolved)
    other_targets, _lo, _hi, notes = sweep_safety(joint, is_left)
    hold_q = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for j, q in other_targets.items():
        hold_q[ARM_JOINTS.index(j)] = q

    def gravity_fn(q: float) -> float:
        arm_q = hold_q.copy()
        arm_q[joint_index] = q
        return float(gravity_comp.gravity_arm(arm_q, is_left=is_left)[joint_index])

    writer = csv_file = None
    if args.csv is not None:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "joint",
                "side",
                "pose_deg",
                "direction",
                "trial",
                "peak_nm",
                "t_s",
                "extra_nm",
                "pos_deg",
                "tau_nm",
            ]
        )

    results: list[dict] = []
    channel = resolve_channel(args)
    async with CanBus(channel) as bus:
        raw_motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw_motors.values()])
        motors = await joint_frame_motors(raw_motors, is_left)
        await asyncio.gather(
            *[
                m.set_control_mode(ControlMode.POSITION_VELOCITY)
                for m in motors.values()
            ]
        )
        motor = motors[joint]
        try:
            print("  Homing all joints to rest (distal to proximal) ...")
            await _home_all(motors)
            for note in notes:
                print(f"  {note}")
            for stage in ramp_stages(other_targets):
                await _ramp_verified(motors, stage)
            await motor.set_control_mode(ControlMode.IMPEDANCE)
            await asyncio.sleep(1.0)
            # Prime the feedback cache at the current pose before any kp = 0.
            here = await motor.get_position()
            await _hold(motor, here, kp, kd, gravity_fn(here), 0.3)

            for pose in poses:
                print(
                    f"\n  pose {math.degrees(pose):+.1f}°  (gravity model {gravity_fn(pose):+.2f} Nm)"
                )
                await ramp_impedance(motor, kp, kd, pose, gravity_fn, rate_hz=_RATE_HZ)
                await _hold(motor, pose, kp, kd, gravity_fn(pose), 0.5)
                print("    trimming the hold (kp = 0) ...")
                trim = await _trim(motor, pose, kp, kd, args.kd, gravity_fn, fc_ref)
                if trim is None:
                    print(
                        "    ! could not trim the joint still — gravity model too far off here; skipping pose"
                    )
                    await _catch(motor, pose, kp, kd, gravity_fn)
                    continue
                hold_ff = gravity_fn(pose) + trim
                found: dict[int, list[float]] = {+1: [], -1: []}
                for direction in (+1, -1):
                    start = 0
                    for trial in range(args.trials):
                        released: float | None = None
                        for i in range(start, len(peaks)):
                            peak = peaks[i]
                            tag = (
                                joint.value,
                                side,
                                f"{math.degrees(pose):.2f}",
                                direction,
                                trial,
                                f"{peak:.3f}",
                            )
                            released = await _release(
                                motor,
                                pose,
                                direction,
                                peak,
                                args.ramp_s,
                                args.kd,
                                hold_ff,
                                move_thr,
                                writer,
                                tag,
                            )
                            if released is not None:
                                print(
                                    f"    {'+' if direction > 0 else '-'} trial {trial + 1}: "
                                    f"released at {released:.3f} Nm (ramp peak {peak:.2f})"
                                )
                                found[direction].append(released)
                                start = max(0, i - 1)
                                await _catch(motor, pose, kp, kd, gravity_fn)
                                # The catch may have landed on a different
                                # edge of the stiction band; re-verify the trim.
                                trim = await _trim(
                                    motor,
                                    pose,
                                    kp,
                                    kd,
                                    args.kd,
                                    gravity_fn,
                                    fc_ref,
                                    trim0=trim,
                                )
                                if trim is None:
                                    raise RuntimeError("lost the hold trim mid-probe")
                                hold_ff = gravity_fn(pose) + trim
                                break
                            await _hold(motor, pose, 0.0, args.kd, hold_ff, 0.3)
                        if released is None:
                            print(
                                f"    {'+' if direction > 0 else '-'} trial {trial + 1}: "
                                f"no release up to {peaks[-1]:.2f} Nm"
                            )
                if found[+1] and found[-1]:
                    f_static, bias = split_breakaway(found[+1], found[-1])
                    stair = predicted_stair_deg(f_static, fc, kp)
                    results.append(
                        {
                            "pose_deg": math.degrees(pose),
                            "trim_nm": trim,
                            "plus": found[+1],
                            "minus": found[-1],
                            "f_static": f_static,
                            "bias": bias,
                            "ratio": f_static / fc if fc > 0 else math.nan,
                            "stair_deg": stair,
                        }
                    )
                else:
                    results.append(
                        {
                            "pose_deg": math.degrees(pose),
                            "trim_nm": trim,
                            "plus": found[+1],
                            "minus": found[-1],
                        }
                    )
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            if csv_file is not None:
                csv_file.close()
            print("  Returning to rest and disabling ...")
            in_impedance = motor.motor.mode == ControlMode.IMPEDANCE
            if in_impedance:
                try:
                    here = motor.position
                    await _hold(motor, here, kp, kd, gravity_fn(here), 0.3)
                    await ramp_impedance(
                        motor, kp, kd, 0.0, gravity_fn, rate_hz=_RATE_HZ
                    )
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
            try:
                await _home_all(motors, exclude=joint if in_impedance else None)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])

    _report(results, fc, kp, gains.stiction_gain)


def _report(results: list[dict], fc: float, kp: float, stiction_gain: float) -> None:
    print(f"\n{'─' * 72}")
    if not results:
        print("  No poses probed.")
        return
    print(
        f"  {'pose':>7s} {'trim':>7s} {'F_static':>9s} {'bias':>7s} {'F_s/fc':>7s} {'stair':>7s}   releases + / -"
    )
    for r in results:
        if "f_static" not in r:
            print(
                f"  {r['pose_deg']:+7.1f} {r['trim_nm']:+7.3f}  (incomplete: + {r['plus']}  - {r['minus']})"
            )
            continue
        print(
            f"  {r['pose_deg']:+7.1f} {r['trim_nm']:+7.3f} {r['f_static']:9.3f} {r['bias']:+7.3f} "
            f"{r['ratio']:7.2f} {r['stair_deg']:6.3f}°   "
            f"{' '.join(f'{v:.2f}' for v in r['plus'])} / {' '.join(f'{v:.2f}' for v in r['minus'])}"
        )
    complete = [r for r in results if "f_static" in r]
    if complete:
        f_static = float(np.mean([r["f_static"] for r in complete]))
        ratio = f_static / fc if fc > 0 else math.nan
        print(
            f"\n  F_static ≈ {f_static:.3f} Nm vs sliding fc {fc:.3f} Nm  (ratio {ratio:.2f})"
        )
        print(
            f"  predicted stick-slip stair at kp={kp:g}: "
            f"{predicted_stair_deg(f_static, fc, kp):.3f}° per stick"
        )
        if fc > 0 and ratio > 1.05:
            print(
                f"  stiction_gain ceiling (F_static/fc − 1): {ratio - 1:.2f}; "
                f"currently {stiction_gain:g}. Try 60-80 % of the ceiling with "
                "tune.motion --gain <side>.<joint>.stiction_gain=…"
            )
        elif fc > 0:
            print(
                "  breakaway is at or below fc — the velocity feedforward already "
                "covers static friction here; stiction_gain would over-compensate."
            )
    print(f"{'─' * 72}")
