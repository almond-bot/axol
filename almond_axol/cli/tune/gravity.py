"""
axol tune.gravity

Identify one link's real centre of mass from the motors instead of trusting
the CAD constants, and fix the gravity feedforward with it.

Why: gravity feedforward comes from the URDF masses/CoMs, which are CAD
values shared by every robot. If the real link is a few percent off (cables,
end-effector variance, manufacturing spread), the kp spring has to make up
the missing torque and the joint parks with a static droop under load
(droop = unmodeled torque / kp — e.g. 0.12° at 80° on a shoulder_1 with
kp 350 is ~0.7 Nm of unmodeled gravity). No amount of kp/kd tuning fixes
that cleanly; correcting the model does, at every angle at once.

How: the same bidirectional constant-velocity sweep as ``tune.friction`` —
averaging the forward and backward torque at the same position cancels
friction exactly, leaving ``gravity(q) + Fo``. The residual against the
current gravity model is then fit to a shift of this link's centre of mass
(ridge-regularized least squares straight through the MuJoCo gravity model,
so the correction generalizes to every arm pose, not just the sweep's).
Mass stays at CAD: gravity torque only depends on the first moment m·c, so
a CoM shift with fixed mass spans every identifiable error. The ridge pulls
directions the sweep barely observes toward the current value — a
single-joint sweep is a 1-D pose slice and can never see all three CoM
components equally — and a shift beyond the plausibility cap is rejected
as bad data.

The sweep runs at a *loaded* pose: gravity has zero moment about a vertical
axis, so joints that hang axis-vertical at rest (shoulder_3, wrist_1, and
wrist_2 once the elbow is raised for base clearance) are swept with other
joints posed to tilt/load them (see ``sweep_safety``). A sweep the model
says is still unloaded is refused as unobservable rather than fit to noise.

Run distal→proximal (wrist_3 → … → shoulder_1): a proximal joint's sweep
carries every distal link, so distal CoMs must be corrected first — a bad
or missing distal calibration lumps into the proximal fit.

``--save`` writes the fitted CoM to this robot's calibration file
(``~/.almond/calibration.json``), where ``AxolConfig`` overlays it like
friction and gains — the gravity compensator, teleop, and every tuning tool
pick it up automatically. The friction offset ``Fo`` is refit against the
corrected model at the same time (the two are coupled: what the old Fo
absorbed as a constant may really have been gravity shape).

Examples:
    axol tune.gravity --l --joint shoulder_1
    axol tune.gravity --l --joint elbow --save
    axol tune.gravity --r --joint wrist_1 --velocity 25 --save
"""

import argparse
import asyncio
import math
from dataclasses import replace

import numpy as np

from ...constants import ARM_JOINTS
from ...motor import CanBus, ControlMode, Joint, Motor
from ...robot.calibration import (
    CALIBRATION_PATH,
    load_calibration,
    update_joint_calibration,
)
from ...robot.config import AxolConfig
from ...robot.gravity import GravityCompensator
from ...tuning import joint_frame_motors, ramp_stages, save_run, sweep_safety
from ..motor import add_side_and_channel_arguments, resolve_channel
from .friction import (
    _home_all,
    _identify_joint,
    _ramp_to,
    _ramp_verified,
)

# Central-difference step for the CoM sensitivity columns (metres). Gravity
# torque is exactly linear in the CoM, so any small step gives the exact
# Jacobian up to float noise; 5 mm keeps the difference well above it.
_FD_STEP = 0.005
# Per-bin torque noise scale (Nm): MIT-feedback quantization plus the
# residual imbalance the fwd/bwd average leaves. Sets both the ridge weight
# and the observability gate below.
_TAU_NOISE_NM = 0.05
# Prior scale (m) on the CoM correction: genuine CAD-vs-build differences
# (cables, end-effector variance) are centimetre-scale. The ridge weight
# (_TAU_NOISE_NM / _COM_PRIOR_M)² keeps directions the sweep barely
# observes at their current value instead of letting them absorb torque
# noise with a huge lever arm — a single-joint sweep is a 1-D slice of pose
# space and always leaves one CoM direction weakly measured or exactly
# invisible (the unconstrained fit once moved a forearm CoM 240 mm on
# 0.06 Nm of on-sweep improvement, wrecking the model everywhere else).
_COM_PRIOR_M = 0.020
# Hard cap on the fitted shift (m). With the loaded sweep poses and the
# distal→proximal ordering, a genuine correction never needs more than
# this; beyond it the sweep data is suspect (collision, something touching
# the arm, distal links not yet calibrated).
_MAX_SHIFT_M = 0.060
_DEFAULT_VELOCITY_DEG = 18.0


def _with_com(
    cfg: AxolConfig, is_left: bool, joint: Joint, com: tuple[float, float, float]
) -> AxolConfig:
    """Return a config copy with one joint's link CoM replaced."""
    arm = cfg.left if is_left else cfg.right
    new_arm = replace(arm, **{joint.value: replace(getattr(arm, joint.value), com=com)})
    return replace(cfg, **{"left" if is_left else "right": new_arm})


def _model_torques(
    cfg: AxolConfig,
    joint: Joint,
    is_left: bool,
    q_bins: np.ndarray,
    other_targets: dict[Joint, float],
) -> np.ndarray:
    """Predicted gravity torque on ``joint`` at each sweep angle."""
    gc = GravityCompensator(cfg)
    test_idx = ARM_JOINTS.index(joint)
    arm_q = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for j, target in other_targets.items():
        if j in ARM_JOINTS and j != joint:
            arm_q[ARM_JOINTS.index(j)] = float(target)
    out = np.empty(len(q_bins))
    for i, q in enumerate(q_bins):
        arm_q[test_idx] = float(q)
        out[i] = float(gc.gravity_arm(arm_q, is_left=is_left)[test_idx])
    return out


def fit_com(
    q_bins: np.ndarray,
    tau_meas: np.ndarray,
    joint: Joint,
    is_left: bool,
    other_targets: dict[Joint, float],
) -> tuple[tuple[float, float, float], float, np.ndarray, np.ndarray] | None:
    """Fit this link's CoM (and a constant offset) to the measured torques.

    Returns ``(com_fit, offset, tau_model_before, tau_model_after)``, or
    ``None`` when the sweep cannot observe this link's CoM. The offset is
    the friction ``Fo`` re-estimated against the corrected model.

    The design matrix is built by central differences of the full MuJoCo
    gravity model around the current (calibrated) CoM — torque is linear in
    the CoM, so this is exact. Observability is judged from the *model*,
    not the measurement: gravity has zero moment about a vertical axis no
    matter where the mass sits, so when every sensitivity column is ~zero
    (the joint is unloaded at this pose) any measured torque variation is
    noise by construction and the fit refuses rather than chase it.

    The solve is ridge-regularized toward the current CoM — a Gaussian
    prior of scale ``_COM_PRIOR_M`` on the shift, given ``_TAU_NOISE_NM``
    of per-bin noise — so directions the sweep barely observes stay put
    instead of soaking up sensor junk with a giant lever arm. The constant
    (Fo) column is never penalized. A fit still beyond ``_MAX_SHIFT_M``
    raises: with the loaded sweep poses that is bad data, not build spread.
    """
    cfg = AxolConfig()
    jc = getattr(cfg.left if is_left else cfg.right, joint.value)
    com0 = np.array(jc.com, dtype=float)

    tau_before = _model_torques(cfg, joint, is_left, q_bins, other_targets)
    residual = tau_meas - tau_before

    # Columns: dτ/d(com_x,y,z) at every bin, plus a constant (→ Fo).
    cols = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = _FD_STEP
        hi = _model_torques(
            _with_com(cfg, is_left, joint, tuple(com0 + step)),
            joint,
            is_left,
            q_bins,
            other_targets,
        )
        lo = _model_torques(
            _with_com(cfg, is_left, joint, tuple(com0 - step)),
            joint,
            is_left,
            q_bins,
            other_targets,
        )
        cols.append((hi - lo) / (2 * _FD_STEP))

    # Observability gate: the smallest CoM shift that would rise above one
    # sigma of torque noise anywhere in the sweep. If even a cap-sized
    # shift could not, the pose leaves this CoM invisible.
    max_sens = max(float(np.linalg.norm(c)) for c in cols)
    if max_sens == 0.0 or _TAU_NOISE_NM / max_sens > _MAX_SHIFT_M:
        return None

    cols.append(np.ones(len(q_bins)))
    design = np.column_stack(cols)

    # Ridge solve: (AᵀA + λI₃)x = Aᵀr with λ = (noise/prior)², identity on
    # the CoM block only. Exactly-unobservable directions (zero columns)
    # come out as exactly zero shift; weak ones shrink toward the prior.
    lam = (_TAU_NOISE_NM / _COM_PRIOR_M) ** 2
    reg = np.zeros((4, 4))
    reg[:3, :3] = lam * np.eye(3)
    solution = np.linalg.solve(design.T @ design + reg, design.T @ residual)

    delta = solution[:3]
    shift = float(np.linalg.norm(delta))
    if shift > _MAX_SHIFT_M:
        raise RuntimeError(
            f"fitted CoM shift {shift * 1000:.0f} mm exceeds the "
            f"{_MAX_SHIFT_M * 1000:.0f} mm plausibility cap. The sweep data "
            "is suspect (collision, something touching the arm, dropped "
            "feedback, or distal links not yet calibrated — run distal → "
            "proximal); not applying it"
        )
    com_fit = tuple(float(v) for v in com0 + delta)
    offset = float(solution[3])
    tau_after = _model_torques(
        _with_com(cfg, is_left, joint, com_fit),
        joint,
        is_left,
        q_bins,
        other_targets,
    )
    return com_fit, offset, tau_before, tau_after


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.gravity`` subcommand."""
    p = subparsers.add_parser(
        "tune.gravity",
        help="Fit one link's real centre of mass from a torque sweep, "
        "correcting the gravity feedforward.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to identify: {', '.join(j.value for j in ARM_JOINTS)}. "
        "Run distal→proximal (wrist_3 first, shoulder_1 last) so proximal "
        "sweeps see already-corrected distal links.",
    )
    p.add_argument(
        "--velocity",
        type=float,
        default=_DEFAULT_VELOCITY_DEG,
        metavar="DEG_S",
        help=f"Sweep velocity in deg/s (default: {_DEFAULT_VELOCITY_DEG:g}). "
        "Slow enough that the shoulder torque telemetry stays clean; "
        "friction cancels in the fwd/bwd average regardless.",
    )
    p.add_argument(
        "--lo",
        type=float,
        default=None,
        metavar="DEG",
        help="Override lower sweep limit (degrees)",
    )
    p.add_argument(
        "--hi",
        type=float,
        default=None,
        metavar="DEG",
        help="Override upper sweep limit (degrees)",
    )
    p.add_argument(
        "--kp",
        type=float,
        default=None,
        help="Sweep-hold proportional gain (default: from config)",
    )
    p.add_argument(
        "--kd",
        type=float,
        default=None,
        help="Sweep-hold derivative gain (default: from config)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save the fitted CoM (and the refit friction Fo) to this "
        f"robot's calibration file ({CALIBRATION_PATH}); the gravity model "
        "then uses it everywhere on this machine",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help="Persist the sweep (measured vs model torque per angle) as a "
        "tuning run for the diagnostics dashboard charts",
    )
    p.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-form note stored with the saved run",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the gravity-identification session for the selected joint."""
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side_str = "left" if is_left else "right"
    resolved = AxolConfig().resolved()
    jc = getattr(resolved.left if is_left else resolved.right, joint.value)
    kp = args.kp if args.kp is not None else jc.kp
    kd = args.kd if args.kd is not None else jc.kd

    print(f"\nAxol gravity identification — {side_str} {joint.value}")
    print(f"  Sweep velocity: {args.velocity:g} deg/s   Kp={kp}  Kd={kd}")
    cal_side = load_calibration()[side_str]
    cal = cal_side.get(joint.value, {})
    if "com" in cal:
        print(f"  Current CoM is already calibrated: {cal['com']} (refining it)")
    # A proximal sweep rotates every distal link with it, so distal CoM
    # errors are indistinguishable from this link's and get lumped into it.
    # Fine at the rest pose the sweep runs at, but wrong once the elbow /
    # wrists bend away from it — hence the distal→proximal order.
    distal_uncal = [
        j.value
        for j in ARM_JOINTS[ARM_JOINTS.index(joint) + 1 :]
        if "com" not in cal_side.get(j.value, {})
    ]
    if distal_uncal:
        print(
            f"  ! Distal links not yet gravity-calibrated: "
            f"{', '.join(distal_uncal)}. Their errors will be lumped into "
            f"{joint.value}'s CoM — exact at the sweep pose, approximate "
            "once those joints bend. For clean attribution run them first "
            "(distal → proximal)."
        )

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
        try:
            print("  Homing all joints to rest (distal to proximal) ...")
            await _home_all(motors)

            # Shared sweep-safety geometry (see sweep_safety): base-collision
            # caps, camera clearance, and the gravity-load poses that tilt
            # axis-vertical joints so their sweep actually carries a CoM
            # signal. The clearance targets also feed the gravity-model
            # predictions, so the fit is computed at the pose the sweep
            # actually ran at. Staged ramps: proximal joints settle before
            # the wrists rotate to their holds.
            other_targets, lo_default, hi_default, notes = sweep_safety(joint, is_left)
            for note in notes:
                print(f"  {note}")
            for stage in ramp_stages(other_targets):
                await _ramp_verified(motors, stage)

            await motors[joint].set_control_mode(ControlMode.IMPEDANCE)
            await asyncio.sleep(1.0)

            avg_samples, _halfdiff = await _identify_joint(
                motors[joint],
                joint,
                kp,
                kd,
                is_left,
                [math.radians(args.velocity)],
                lo_override=math.radians(args.lo)
                if args.lo is not None
                else lo_default,
                hi_override=math.radians(args.hi)
                if args.hi is not None
                else hi_default,
            )
            if len(avg_samples) < 8:
                print("\n  ! Too few matched fwd/bwd bins to fit anything.")
                return

            q_bins = np.array([s[0] for s in avg_samples])
            tau_meas = np.array([s[1] for s in avg_samples])
            order = np.argsort(q_bins)
            q_bins, tau_meas = q_bins[order], tau_meas[order]

            try:
                fit = fit_com(q_bins, tau_meas, joint, is_left, other_targets)
            except RuntimeError as exc:
                print(f"\n  ! Gravity fit rejected: {exc}")
                return
            _report_and_save(
                args, joint, side_str, jc, q_bins, tau_meas, fit, other_targets
            )

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to rest and disabling ...")
            in_impedance = motors[joint].motor.mode == ControlMode.IMPEDANCE
            if in_impedance:
                try:
                    await _ramp_to(motors[joint], kp, kd, 0.0, duration=4.0)
                except Exception:
                    pass
            try:
                await _home_all(motors, exclude=joint if in_impedance else None)
            except Exception:
                pass
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])


def _report_and_save(
    args: argparse.Namespace,
    joint: Joint,
    side_str: str,
    jc,
    q_bins: np.ndarray,
    tau_meas: np.ndarray,
    fit: tuple[tuple[float, float, float], float, np.ndarray, np.ndarray] | None,
    clearance: dict[Joint, float],
) -> None:
    print(f"\n{'─' * 50}")
    if fit is None:
        print(
            f"  No gravity observability on {joint.value} at this pose — "
            "the joint axis is (near-)parallel to gravity here, which "
            "produces zero gravity moment for *any* mass placement, so the "
            "sweep cannot see the CoM and any measured torque variation is "
            "noise. Current value kept. (The loaded sweep poses should "
            "prevent this; check the clearance ramps completed.)"
        )
        return

    com_fit, offset, tau_before, tau_after = fit
    res_before = tau_meas - tau_before
    res_after = tau_meas - tau_after - offset
    rms_before = float(np.sqrt(np.mean((res_before - np.mean(res_before)) ** 2)))
    rms_after = float(np.sqrt(np.mean(res_after**2)))
    delta_mm = [(f - c) * 1000 for f, c in zip(com_fit, jc.com)]

    # The droop the kp spring shows for the worst remaining/removed torque
    # error — the user-visible payoff (parked error = torque error / kp).
    droop_before = math.degrees(float(np.max(np.abs(res_before))) / jc.kp)
    droop_after = math.degrees(float(np.max(np.abs(res_after))) / jc.kp)

    print("  Fitted link CoM (URDF link frame, metres):")
    print(f"    CAD    : ({jc.com[0]:+.4f}, {jc.com[1]:+.4f}, {jc.com[2]:+.4f})")
    print(
        f"    Fitted : ({com_fit[0]:+.4f}, {com_fit[1]:+.4f}, {com_fit[2]:+.4f})"
        f"   (shift {delta_mm[0]:+.1f}, {delta_mm[1]:+.1f}, {delta_mm[2]:+.1f} mm)"
    )
    print(f"    Fo     : {offset:+.4f} Nm  (friction offset refit to match)")
    print(
        f"  Shape residual: {rms_before:.4f} → {rms_after:.4f} Nm RMS "
        f"({rms_after / rms_before * 100:.0f}% of before)"
        if rms_before > 0
        else f"  Shape residual: {rms_after:.4f} Nm RMS"
    )
    print(
        f"  Worst parked droop at kp={jc.kp:g}: "
        f"{droop_before:.3f}° → {droop_after:.3f}°"
    )
    if rms_before > 0.1 and rms_after > 0.6 * rms_before:
        print(
            "  ! The fit explains less than half the residual shape — a CoM "
            "error can't produce this profile. Suspect torque telemetry or "
            "something touching the arm during the sweep; treat the values "
            "with caution."
        )

    if args.save_run:
        run_id = save_run(
            "gravity",
            {
                "q": q_bins,
                "measured": tau_meas,
                "model_before": tau_before,
                "model_after": tau_after + offset,
            },
            {
                "rms_before": rms_before,
                "rms_after": rms_after,
                "droop_before_deg": droop_before,
                "droop_after_deg": droop_after,
                "fo": offset,
                "com_shift_mm": delta_mm,
            },
            side=side_str,
            joint=joint.value,
            params={
                "velocity_deg_s": args.velocity,
                "com_cad": list(jc.com),
                "com_fit": list(com_fit),
                "saved": bool(args.save),
                "clearance_deg": {
                    j.value: round(math.degrees(v), 1) for j, v in clearance.items()
                },
            },
            label=args.label,
        )
        print(f"  Saved run {run_id}")

    if not args.save:
        print("\n  Re-run with --save to write the fitted CoM to this robot's")
        print(f"  calibration ({CALIBRATION_PATH}).")
        print(f"{'─' * 50}")
        return

    friction_cal = load_calibration()[side_str].get(joint.value, {}).get("friction")
    friction_update = None
    if friction_cal is not None:
        friction_update = {**friction_cal, "fo": round(offset, 4)}
    path = update_joint_calibration(
        side_str,
        joint.value,
        com=tuple(round(v, 5) for v in com_fit),
        friction=friction_update,
    )
    print(f"\n  Saved to {path}")
    if friction_update is not None:
        print(f"  (friction fo refit to {offset:+.4f} Nm against the new model)")
    else:
        print(
            "  ! No friction calibration for this joint yet — Fo not saved. "
            "Run axol tune.friction --save after this."
        )
    print(f"{'─' * 50}")
