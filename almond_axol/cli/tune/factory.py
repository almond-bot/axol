"""
axol tune.factory

Full-robot factory calibration: friction and gravity for all 14 joints (both
arms x 7) in one command, saved to this machine's calibration file and
uploaded to the cloud keyed by the Axol hub adapter's serial.

One bidirectional multi-velocity sweep per joint yields both fits (the same
sweep ``tune.friction`` and ``tune.gravity`` run individually):

  - half(t_fwd - t_bwd) at matched positions -> Fc / k / Fv (friction)
  - avg(t_fwd, t_bwd) at matched positions   -> gravity(q), fit to the
    link's real centre of mass, with the friction offset Fo refit against
    the corrected model

Each arm runs distal -> proximal (wrist_3 ... shoulder_1) and every joint's
fit is saved before the next joint sweeps, so proximal sweeps see the
already-corrected distal links — the ordering ``tune.gravity`` asks the
operator to keep by hand, kept automatically here.

Results land in ``~/.almond/calibration.json`` (this machine uses them
immediately) and, when the Supabase write key is configured
(``AXOL_SUPABASE_KEY`` in the environment or a ``.env``), the same document
is upserted to the public ``axol-calibrations`` storage bucket keyed by the
hub adapter serial — any later machine fetches it keylessly with ``axol
calibration.pull`` (or automatically during ``axol can.setup``). Without
the key the upload is skipped and the local calibration still happens.

Examples:
    axol tune.factory                       # both arms, default velocities
    axol tune.factory --arms left           # one arm only
    axol tune.factory --velocities 18 36    # quicker sweep (fewer velocities)
"""

import argparse
import asyncio
import math
from typing import Any

import numpy as np

from ...constants import ARM_JOINTS, CAN_LEFT, CAN_RIGHT
from ...motor import CanBus, ControlMode, Joint, Motor
from ...robot.calibration import CALIBRATION_PATH, update_joint_calibration
from ...robot.calibration_cloud import (
    fetch_calibration,
    push_calibration,
    supabase_credentials,
)
from ...robot.config import AxolConfig
from ...tuning import JointFrameMotor, joint_frame_motors, sweep_safety
from ..can.setup import hub_serial
from .friction import (
    DEFAULT_VELOCITIES_DEG,
    _compare_to_gravity_model,
    _fit_friction_halfdiff,
    _home_all,
    _identify_joint,
    _ramp_to,
    _ramp_verified,
)
from .gravity import fit_com

# Distal -> proximal: proximal sweeps carry every distal link, so distal CoMs
# must be corrected first for clean attribution (see tune.gravity).
_CAL_ORDER: tuple[Joint, ...] = (
    Joint.WRIST_3,
    Joint.WRIST_2,
    Joint.WRIST_1,
    Joint.ELBOW,
    Joint.SHOULDER_3,
    Joint.SHOULDER_2,
    Joint.SHOULDER_1,
)
# MyActuator mode switches are a ~2 s reset that silently drops commands.
_MODE_SWITCH_SETTLE_S = 2.5


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.factory`` subcommand."""
    p = subparsers.add_parser(
        "tune.factory",
        help="Factory calibration: friction + gravity for all 14 joints, "
        "saved locally and uploaded to the cloud by hub serial.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--arms",
        choices=["both", "left", "right"],
        default="both",
        help="Which arm(s) to calibrate (default: both)",
    )
    p.add_argument(
        "--left-channel",
        default=CAN_LEFT,
        metavar="IFACE",
        help=f"Left arm CAN interface (default: {CAN_LEFT})",
    )
    p.add_argument(
        "--right-channel",
        default=CAN_RIGHT,
        metavar="IFACE",
        help=f"Right arm CAN interface (default: {CAN_RIGHT})",
    )
    p.add_argument(
        "--velocities",
        type=float,
        nargs="+",
        default=DEFAULT_VELOCITIES_DEG,
        metavar="DEG_S",
        help="Friction velocity sweep in deg/s (default: 7.2 18 36 54 72). "
        "The gravity fit uses the same sweeps' averaged torques.",
    )
    p.add_argument(
        "--hub-serial",
        default=None,
        metavar="SERIAL",
        help="Robot identity for the cloud upload (default: the attached "
        "Axol hub adapter's USB serial)",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the full factory-calibration session."""
    asyncio.run(_run(args))


async def _calibrate_joint(
    motors: dict[Joint, JointFrameMotor],
    joint: Joint,
    is_left: bool,
    velocities_rad: list[float],
) -> dict[str, Any] | None:
    """Sweep one joint, fit friction + CoM, save both, return the entry.

    The config is re-read per joint so each fit sees the CoMs already saved
    by the more-distal joints of this same session.
    """
    side_str = "left" if is_left else "right"
    resolved = AxolConfig().resolved()
    jc = getattr(resolved.left if is_left else resolved.right, joint.value)
    kp, kd = jc.kp, jc.kd
    print(f"\n{'=' * 60}")
    print(f"  {side_str} {joint.value}  (Kp={kp:g}  Kd={kd:g})")
    print(f"{'=' * 60}")

    # Shared sweep-safety geometry: base-collision caps for shoulder_2 /
    # wrist_2, elbow raised for wrist_2, shoulder_2 held outboard for the
    # camera-adjacent joints. The clearance pose also feeds the gravity
    # model so the fit matches the pose the sweep ran at.
    other_targets, lo_default, hi_default, notes = sweep_safety(joint, is_left)
    for note in notes:
        print(f"  {note}")
    if other_targets:
        await _ramp_verified(motors, other_targets)

    await motors[joint].set_control_mode(ControlMode.IMPEDANCE)
    await asyncio.sleep(1.0)
    try:
        avg_samples, halfdiff_samples = await _identify_joint(
            motors[joint],
            joint,
            kp,
            kd,
            is_left,
            velocities_rad,
            lo_override=lo_default,
            hi_override=hi_default,
        )
    finally:
        # Park and hand the joint back to POSITION_VELOCITY so the next
        # joint's homing/clearance ramps can drive it.
        try:
            await _ramp_to(motors[joint], kp, kd, 0.0, duration=4.0)
        except Exception:
            pass
        await motors[joint].set_control_mode(ControlMode.POSITION_VELOCITY)
        await asyncio.sleep(_MODE_SWITCH_SETTLE_S)
        # Re-verify the whole arm at rest (returns the clearance joints too).
        await _home_all(motors)

    if len(avg_samples) < 8:
        print(f"  ! Too few samples on {joint.value} — skipping its fits.")
        return None

    friction_fit = _fit_friction_halfdiff(halfdiff_samples)

    q_bins = np.array([s[0] for s in avg_samples])
    tau_meas = np.array([s[1] for s in avg_samples])
    order = np.argsort(q_bins)
    q_bins, tau_meas = q_bins[order], tau_meas[order]
    try:
        gravity_fit = fit_com(q_bins, tau_meas, joint, is_left, other_targets)
    except RuntimeError as exc:
        # An implausible fit means bad sweep data — keep the CAD CoM but
        # don't lose the friction fit over it.
        print(f"  ! Gravity fit rejected: {exc}")
        gravity_fit = None

    com_fit = None
    if gravity_fit is not None:
        com_fit, fo, tau_before, tau_after = gravity_fit
        res_before = tau_meas - tau_before
        res_after = tau_meas - tau_after - fo
        rms_b = float(np.sqrt(np.mean((res_before - np.mean(res_before)) ** 2)))
        rms_a = float(np.sqrt(np.mean(res_after**2)))
        delta_mm = [(f - c) * 1000 for f, c in zip(com_fit, jc.com)]
        print(
            f"  CoM shift ({delta_mm[0]:+.1f}, {delta_mm[1]:+.1f}, "
            f"{delta_mm[2]:+.1f}) mm — shape residual "
            f"{rms_b:.4f} -> {rms_a:.4f} Nm RMS"
        )
    else:
        # No gravity signal (axis parallel to gravity) or rejected fit: CAD
        # CoM stays, Fo is the mean residual against the current model.
        fo_result = _compare_to_gravity_model(
            avg_samples, joint, is_left, other_targets
        )
        fo = fo_result if fo_result is not None else 0.0

    entry: dict[str, Any] = {}
    if friction_fit is not None:
        fc, k, fv = friction_fit
        entry["friction"] = {
            "fc": round(fc, 4),
            "k": round(k, 2),
            "fv": round(fv, 4),
            "fo": round(fo, 4),
        }
        print(
            f"  Friction: Fc={fc:.4f} Nm  k={k:.2f}  Fv={fv:.4f} Nm·s/rad  "
            f"Fo={fo:+.4f} Nm"
        )
    else:
        print(f"  ! Friction fit failed on {joint.value} — not saving friction.")
    if com_fit is not None:
        entry["com"] = [round(v, 5) for v in com_fit]

    if not entry:
        return None
    update_joint_calibration(
        side_str,
        joint.value,
        friction=entry.get("friction"),
        com=tuple(entry["com"]) if "com" in entry else None,
    )
    return entry


async def _calibrate_arm(
    channel: str,
    is_left: bool,
    velocities_rad: list[float],
    results: dict[str, dict[str, Any]],
) -> None:
    """Home the arm, calibrate all 7 joints distal->proximal, park, disable.

    Populates ``results`` in place per completed joint so an interrupt
    mid-arm still leaves everything finished so far in the document (and
    already saved to the local calibration file).
    """
    side_str = "left" if is_left else "right"
    print(f"\n### {side_str.upper()} ARM ({channel}) ###")
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
            for joint in _CAL_ORDER:
                entry = await _calibrate_joint(motors, joint, is_left, velocities_rad)
                if entry:
                    results[joint.value] = entry
        finally:
            print("  Returning to rest and disabling ...")
            try:
                await _home_all(motors)
            except Exception:
                pass
            await asyncio.gather(
                *[m.set_control_mode(ControlMode.IMPEDANCE) for m in motors.values()]
            )
            await asyncio.gather(*[m.disable() for m in motors.values()])


async def _run(args: argparse.Namespace) -> None:
    velocities_rad = [math.radians(v) for v in args.velocities]
    sides = {
        "both": [("left", args.left_channel), ("right", args.right_channel)],
        "left": [("left", args.left_channel)],
        "right": [("right", args.right_channel)],
    }[args.arms]

    creds = supabase_credentials()
    serial = args.hub_serial or hub_serial()
    print("\nAxol factory calibration — friction + gravity, all joints")
    print(f"  Arms: {', '.join(s for s, _ in sides)}")
    print(f"  Velocities: {[round(v, 1) for v in args.velocities]} deg/s")
    print(f"  Robot id (hub serial): {serial or 'not detected'}")
    if creds is None:
        print(
            "  Supabase: no write key (AXOL_SUPABASE_KEY, plus "
            "AXOL_SUPABASE_URL if not baked in) — results will be saved "
            "locally only."
        )

    document: dict[str, Any] = {"version": 1}
    try:
        for side_str, channel in sides:
            side_results: dict[str, dict[str, Any]] = {}
            document[side_str] = side_results
            await _calibrate_arm(
                channel, side_str == "left", velocities_rad, side_results
            )
    except KeyboardInterrupt:
        print("\n  Interrupted — keeping what completed.")

    n = sum(len(document.get(s, {})) for s in ("left", "right"))
    print(f"\n{'=' * 60}")
    print(f"  Calibrated {n} joints; saved to {CALIBRATION_PATH}")

    if creds is None:
        print("  Skipped cloud upload (no Supabase credentials).")
        return
    if serial is None:
        print(
            "  ! No Axol hub adapter serial detected — cannot key the cloud "
            "upload. Re-run the upload with --hub-serial once known."
        )
        return
    if n == 0:
        print("  Nothing calibrated — skipping cloud upload.")
        return
    # Merge over what the cloud already has, so a one-arm run doesn't wipe
    # the other arm's stored data.
    try:
        existing = fetch_calibration(serial) or {}
        merged: dict[str, Any] = {"version": 1}
        for s in ("left", "right"):
            old = existing.get(s)
            new = document.get(s)
            side_doc = dict(old) if isinstance(old, dict) else {}
            if isinstance(new, dict):
                side_doc.update(new)
            if side_doc:
                merged[s] = side_doc
        push_calibration(creds, serial, merged)
        print(f"  Uploaded to Supabase as {serial}.")
    except RuntimeError as exc:
        print(f"  ! Cloud upload failed: {exc}")
        print("    The local calibration is saved; re-run the upload later.")
