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
from ...motor import CanBus, Joint, Motor
from ...robot.calibration import (
    CALIBRATION_PATH,
    load_calibration,
    update_joint_calibration,
)
from ...robot.config import AxolConfig
from ...robot.gravity import GravityCompensator
from ...robot.identity import hub_serial
from ...tuning import joint_frame_motors, ramp_stages, save_run, sweep_safety
from ..motor import add_side_and_channel_arguments, resolve_channel
from .friction import (
    _home_all,
    _identify_joint,
    _ramp_verified,
    assign_modes,
    safe_return_to_rest,
)

# Central-difference step for the CoM sensitivity columns (metres). Gravity
# torque is exactly linear in the CoM, so any small step gives the exact
# Jacobian up to float noise; 5 mm keeps the difference well above it.
_FD_STEP = 0.005
# Holder drift (rad) worth reporting before the fit. The holders run on the
# motors' own position loops with no host feedback, so this is the only place
# a sagging hold becomes visible instead of quietly biasing the CoM.
_HOLDER_DRIFT_WARN = math.radians(1.0)
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
#: Prior scale and plausibility cap on the fitted link MASS, as fractions of
#: the configured mass. A CoM-only fit cannot represent a mass error: on the
#: right elbow the link weighs 0.25 kg in the model and the sweep demanded a
#: 93 mm CoM shift -- 0.23 Nm of torque the light link can only produce with
#: an absurd lever, i.e. ~78 g of unmodelled mass at the hand. Fit mass and
#: CoM together so a few-percent mass error lands on the mass, where it is
#: a small number, instead of on a lever, where it is a rejected one.
#: A sweep about one axis observes only m·r: the link's mass and its CoM
#: *along the lever* produce identical torque curves, so the two are
#: collinear and only a prior can split them. The rule here is: along the
#: lever goes to MASS, perpendicular goes to CoM. Physically that is the
#: right attribution -- a light link's own CoM does not move 93 mm, but a
#: harness or hand carrying ~80 g at 0.3 m looks to the elbow exactly like
#: +0.3 kg on its link, and as mass it propagates to the shoulders with
#: nearly the right moment. ``_COM_PRIOR_ALONG_M`` pins the along-lever
#: CoM component; ``_MASS_PRIOR_FRAC`` is loose so mass takes that content.
_COM_PRIOR_ALONG_M = 0.002
_MASS_PRIOR_FRAC = 1.0
#: Mass plausibility cap, absolute: a harness's worth. It is deliberately
#: NOT a fraction of the link mass. Bench: with a 50 % cap, shoulder_3
#: (3.75 kg) accepted -0.77 kg while its residual barely moved -- that was
#: the uncalibrated elbow's error, lumped in and booked as mass. A heavy
#: proximal link's mass is well known; the only unmodelled mass a sweep
#: should be allowed to discover is cabling and hand-side hardware, which
#: is the same few hundred grams whatever link it is attributed to.
_MAX_MASS_ABS_KG = 0.35
_DEFAULT_VELOCITY_DEG = 18.0


def _with_com(
    cfg: AxolConfig, is_left: bool, joint: Joint, com: tuple[float, float, float]
) -> AxolConfig:
    """Return a config copy with one joint's link CoM replaced."""
    arm = cfg.left if is_left else cfg.right
    new_arm = replace(arm, **{joint.value: replace(getattr(arm, joint.value), com=com)})
    return replace(cfg, **{"left" if is_left else "right": new_arm})


def _with_mass(cfg: AxolConfig, is_left: bool, joint: Joint, mass: float) -> AxolConfig:
    """Return a config copy with one joint's link mass replaced."""
    arm = cfg.left if is_left else cfg.right
    new_arm = replace(
        arm, **{joint.value: replace(getattr(arm, joint.value), mass=mass)}
    )
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
) -> tuple[tuple[float, float, float], float, np.ndarray, np.ndarray, float] | None:
    """Fit this link's CoM, its mass, and a constant offset to the torques.

    Returns ``(com_fit, offset, tau_model_before, tau_model_after,
    mass_fit)``, or ``None`` when the sweep cannot observe this link's CoM.
    The offset is the friction ``Fo`` re-estimated against the corrected
    model. Mass is fitted alongside the CoM because a CoM-only fit cannot
    express a mass error at a light link except as an implausible lever
    (see ``_MASS_PRIOR_FRAC``); its prior is tighter than the CoM's so a
    sweep that a CoM shift explains equally well still lands on the CoM.

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
    mass0 = float(jc.mass)

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

    # dτ/d(mass): torque is linear in the link mass, so the central
    # difference is exact for any step. Column in Nm per kg.
    dm = max(0.05 * mass0, 1e-3)
    hi_m = _model_torques(
        _with_mass(cfg, is_left, joint, mass0 + dm),
        joint,
        is_left,
        q_bins,
        other_targets,
    )
    lo_m = _model_torques(
        _with_mass(cfg, is_left, joint, mass0 - dm),
        joint,
        is_left,
        q_bins,
        other_targets,
    )
    cols.append((hi_m - lo_m) / (2 * dm))

    cols.append(np.ones(len(q_bins)))
    design = np.column_stack(cols)

    # Ridge solve: (AᵀA + Λ)x = Aᵀr with λ = (noise/prior)² per block —
    # CoM block on a metres prior, mass on a kg prior. Exactly-unobservable
    # directions (zero columns) come out as exactly zero; weak ones shrink
    # toward the configured value. The constant (Fo) column is never
    # penalised.
    reg = np.zeros((5, 5))
    # The direction degenerate with mass is the CoM's projection onto the
    # plane the sweep observes: components along the joint axis have zero
    # torque sensitivity (their columns are ~0) and take no part in it.
    observable = np.array([np.linalg.norm(c) > 1e-9 for c in cols[:3]])
    com_plane = np.where(observable, com0, 0.0)
    r0 = np.linalg.norm(com_plane)
    if r0 > 1e-6:
        u = com_plane / r0
        perp = np.eye(3) - np.outer(u, u)
        reg[:3, :3] = (_TAU_NOISE_NM / _COM_PRIOR_M) ** 2 * perp + (
            _TAU_NOISE_NM / _COM_PRIOR_ALONG_M
        ) ** 2 * np.outer(u, u)
    else:
        reg[:3, :3] = (_TAU_NOISE_NM / _COM_PRIOR_M) ** 2 * np.eye(3)
    reg[3, 3] = (_TAU_NOISE_NM / max(_MASS_PRIOR_FRAC * mass0, 1e-6)) ** 2
    solution = np.linalg.solve(design.T @ design + reg, design.T @ residual)

    delta = solution[:3]
    dmass = float(solution[3])
    shift = float(np.linalg.norm(delta))
    if shift > _MAX_SHIFT_M:
        raise RuntimeError(
            f"fitted CoM shift {shift * 1000:.0f} mm exceeds the "
            f"{_MAX_SHIFT_M * 1000:.0f} mm plausibility cap. The sweep data "
            "is suspect (collision, something touching the arm, dropped "
            "feedback, or distal links not yet calibrated — run distal → "
            "proximal); not applying it"
        )
    if abs(dmass) > _MAX_MASS_ABS_KG:
        raise RuntimeError(
            f"fitted mass change {dmass:+.3f} kg exceeds the "
            f"{_MAX_MASS_ABS_KG:.2f} kg plausibility cap (a harness's worth). "
            f"On a {mass0:.3f} kg link that much unmodelled mass is a payload, "
            "a wrong link model, or a distal link's error lumped in -- run "
            "distal to proximal; not applying it"
        )
    com_fit = tuple(float(v) for v in com0 + delta)
    mass_fit = mass0 + dmass
    offset = float(solution[4])
    tau_after = _model_torques(
        _with_mass(_with_com(cfg, is_left, joint, com_fit), is_left, joint, mass_fit),
        joint,
        is_left,
        q_bins,
        other_targets,
    )
    return com_fit, offset, tau_before, tau_after, mass_fit


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
    serial = hub_serial()
    resolved = AxolConfig().resolved()
    jc = getattr(resolved.left if is_left else resolved.right, joint.value)
    kp = args.kp if args.kp is not None else jc.kp
    kd = args.kd if args.kd is not None else jc.kd

    print(f"\nAxol gravity identification — {side_str} {joint.value}")
    print(f"  Sweep velocity: {args.velocity:g} deg/s   Kp={kp}  Kd={kd}")
    cal_side = load_calibration(expected_hub_serial=serial)[side_str] if serial else {}
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
        # Modes are decided here, at rest, and never changed again: a
        # MyActuator mode switch is a system reset and the joint is limp for
        # it (see assign_modes). Switching the swept joint to impedance after
        # the arm was posed is what dropped a loaded wrist.
        holders = await assign_modes(
            motors, impedance=joint, kp=kp, kd=kd, is_left=is_left, config=resolved
        )
        try:
            print("  Homing all joints to rest (distal to proximal) ...")
            await _home_all(motors, holders, impedance=joint, kp=kp, kd=kd)

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
                await _ramp_verified(motors, stage, holders)

            # Fit against the pose the arm is actually in, not the one it was
            # told to reach. The holders sit on their own firmware position
            # loop at whatever position_kp the motor shipped with, and a
            # loaded one sags: an elbow posed to its midpoint carries several
            # Nm there. Feeding the *commanded* pose to the model makes the
            # fitted CoM absorb that sag, which is a silent wrong answer
            # rather than a visible failure.
            held = {}
            for j, m in motors.items():
                if j is joint:
                    continue
                try:
                    held[j] = await m.get_position()
                except Exception:
                    held[j] = other_targets.get(j, 0.0)
            # Sort on the drift alone: a tie would otherwise fall through to
            # comparing Joint enums, which have no ordering (bench traceback).
            drift = sorted(
                ((abs(held[j] - other_targets.get(j, 0.0)), j) for j in held),
                key=lambda d: d[0],
                reverse=True,
            )
            if drift and drift[0][0] > _HOLDER_DRIFT_WARN:
                print("  ! holders are not where they were put:")
                for d, j in drift:
                    if d <= _HOLDER_DRIFT_WARN:
                        break
                    print(
                        f"      {j.value}: {math.degrees(held[j]):+.2f}° "
                        f"(asked {math.degrees(other_targets.get(j, 0.0)):+.2f}°, "
                        f"off by {math.degrees(d):.2f}°)"
                    )
                print(
                    "    Fitting at the measured pose. A large sag means the "
                    "holder's firmware position loop is too soft for the load "
                    "— see axol tune.position-loop --mode hold."
                )

            # The swept joint has been under impedance since before homing,
            # so there is no mode switch here and nothing to settle from.
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
                fit = fit_com(q_bins, tau_meas, joint, is_left, held)
            except RuntimeError as exc:
                print(f"\n  ! Gravity fit rejected: {exc}")
                return
            _report_and_save(
                args,
                joint,
                side_str,
                jc,
                q_bins,
                tau_meas,
                fit,
                held,
                serial,
                cal_side,
            )

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            print("  Returning to rest and disabling ...")
            await safe_return_to_rest(motors, holders, joint, kp, kd)


def _report_and_save(
    args: argparse.Namespace,
    joint: Joint,
    side_str: str,
    jc,
    q_bins: np.ndarray,
    tau_meas: np.ndarray,
    fit: tuple[tuple[float, float, float], float, np.ndarray, np.ndarray] | None,
    clearance: dict[Joint, float],  # measured, not commanded — see _run
    hub_serial: str | None,
    cal_side: dict[str, dict],
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

    com_fit, offset, tau_before, tau_after, mass_fit = fit
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
    print(
        f"    Mass   : {jc.mass:.3f} -> {mass_fit:.3f} kg "
        f"({100 * (mass_fit / jc.mass - 1):+.1f}%)"
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
                "mass_fit": mass_fit,
                "saved": bool(args.save),
                # The pose the fit was computed at, as measured after the
                # ramps — a sagging holder makes this differ from what was
                # commanded, and the fit is only meaningful alongside it.
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

    friction_cal = cal_side.get(joint.value, {}).get("friction")
    friction_update = None
    if friction_cal is not None:
        friction_update = {**friction_cal, "fo": round(offset, 4)}
    path = update_joint_calibration(
        side_str,
        joint.value,
        com=tuple(round(v, 5) for v in com_fit),
        mass=round(mass_fit, 4),
        friction=friction_update,
        hub_serial=hub_serial,
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
