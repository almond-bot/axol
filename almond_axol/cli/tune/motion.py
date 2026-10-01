"""
axol tune.motion

Replay a committed reference motion through the production ``motion_control``
path and score tracking accuracy and smoothness per joint — the closest
thing to a repeatable teleop session.

The motion (see ``axol motion.list`` / ``motion.build``) streams to both
arms at its stored rate with absolute-deadline pacing, exactly like teleop
drives the robot: impedance gains, gravity/friction/inertia feedforward, and
host-side damping all come from the same ``AxolConfig`` production uses.
Override individual gains per run with ``--gain`` and compare runs on the
identical motion — the deterministic A/B loop that ad-hoc teleop testing
can't give you.

With ``--ik`` the run exercises the full Cartesian pipeline instead of raw
joint replay: every waypoint is converted to its two end-effector poses
(plus elbow hints) by FK, and the IK solver re-solves the chain exactly
like teleop's pose->joints loop — the arms execute the *solver's* output,
scored against the clean reference, so IK reconstruction error and
controller tracking show up together (the charts overlay reference, solved,
and measured).

Every run is persisted as a tuning-run artifact (full per-joint time series
+ metrics, ``~/.almond/diagnostics/tuning/``) for charting and side-by-side
comparison in the diagnostics UI; ``--no-save-run`` disables that.

Safety: the arm moves to the motion's start (and back to rest at the end)
on collision-aware planned trajectories; a contact watchdog aborts playback
if a sustained torque residual says the arm is pushing on something that
isn't in the plan.

Examples:
    axol tune.motion --motion reach-and-place
    axol tune.motion --motion reach-and-place --gain left.elbow.kd=4.5
    axol tune.motion --motion reach-and-place --gain shoulder_3.kd_host=8 --label "s3 damp"
    axol tune.motion --motion reach-and-place --stiffness 0.8
    axol tune.motion --motion reach-and-place --ik   # drive through the IK solver
    axol tune.motion --motion slow_osc --arms right  # one arm only
    axol tune.motion --motion slow_osc --controller position  # firmware loops, 400 Hz
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...robot import Axol
from ...robot.axol import arm_limits
from ...robot.config import (
    CONTROLLERS,
    FAST_IMPEDANCE_HZ,
    IMPEDANCE_LOOP_HZ,
    IMPEDANCE_RATES,
    AxolConfig,
    check_firmware_extras,
    check_loop_hz,
    fast_impedance_joints,
)
from ...robot.control import ContactWatchdog
from ...tuning import save_run, tracking_metrics
from ...tuning.imu_damping import EncoderTipDamper, GyroFlexDamper, TorqueProbe
from ...tuning.learning import LEARN_BAND, CommandLearner
from ...tuning.motion import ReferenceMotion, list_motions, load_motion
from ...tuning.runs import load_run
from ...tuning.wrist_imu import WristImu, format_imu
from ...utils.logquiet import quiet_noisy_loggers

_PLAN_SPEED = 0.1 * np.pi  # rad/s — approach/return trajectory speed

#: A firmware-loop joint (0xA4 / pv) this far (rad) from its command has left
#: its target: the replay stops and returns to rest. Firmware-loop joints
#: carry no torque telemetry for the contact watchdog, and the core has no
#: position-deviation abort (a pushed impedance joint is normal), so nothing
#: else catches one — right elbow on the 0xA4 planner ended 117° from its
#: command (2026-09-22). ``tune.a4``'s own abort is the same 20°; normal lag
#: is ~1° at the approach speed.
_FW_DEVIATION_ABORT = math.radians(20.0)
_PLAN_MIN_DURATION = 1.5  # s

_GAIN_FIELDS = (
    "kp",
    "kd",
    "kd_host",
    "kd_host_hz",
    "kd_host_q",
    "j_eff",
    "stiction_gain",
    "stiction_load_gain",
    "stiction_err_deg",
    "dither_nm",
    "dither_hz",
    "stribeck_gain",
    "stribeck_dfs",
    "stribeck_load_gain",
    "stribeck_vs",
    "stribeck_pole",
    # Friction model, addressed as ``joint.friction.fc`` etc. — the sliding
    # friction feedforward is the other half of every stick-slip A/B.
    "friction.fc",
    "friction.k",
    "friction.fv",
    "friction.fo",
    "friction.fl",
    # Firmware position-loop gains (``firmware.*`` on JointConfig), for A/B
    # runs of the position controller: written to the motors' ROM at enable
    # like the config values they replace (so a run leaves them there).
    "firmware.position_kp",
    "firmware.position_ki",
    "firmware.position_kd",
    "firmware.speed_kp",
    "firmware.speed_ki",
    "firmware.profile_acc",
    # MyActuator 0xA4: the position planner (0 direct / 60000) and the
    # core's per-tick speed-cap tracking that the planner wants.
    "firmware.planner_accel",
    "firmware.cap_track",
    "firmware.planner_lead_ms",
    # MyActuator 0x73: the rated current that scales the torque feedforward
    # (set = the joint's position command carries gravity + inertia +
    # cogging; unset = plain 0xA4).
    "firmware.tf_rated_current_a",
    # The cogging ("osc") cancellation's share of the calibrated series.
    "cogging_gain",
    # The gravity model's per-link inertials (the body this joint drives):
    # mass (kg) and centre of mass (m, URDF link frame) — for trying a gravity
    # correction before committing it to calibration.
    "mass",
    "com.x",
    "com.y",
    "com.z",
)

# Column names of a 14-wide motion row: left arm then right arm.
_COLUMNS = [f"left.{j.value}" for j in ARM_JOINTS] + [
    f"right.{j.value}" for j in ARM_JOINTS
]


def _parse_holds(specs: list[str]) -> dict[int, float | None]:
    """``--hold SIDE.JOINT[=DEG]`` → ``{column: angle rad, or None}``.

    ``None`` holds the joint at the motion's own first-row angle; an angle
    must sit inside the arm's joint limits.
    """
    out: dict[int, float | None] = {}
    for spec in specs:
        name, eq, deg = spec.partition("=")
        if name not in _COLUMNS:
            raise SystemExit(
                f"--hold wants SIDE.JOINT[=DEG] with an arm joint, got {spec!r}"
            )
        angle: float | None = None
        if eq:
            try:
                angle = math.radians(float(deg))
            except ValueError:
                raise SystemExit(f"--hold: bad angle in {spec!r}") from None
            side, joint = name.split(".")
            lo, hi = arm_limits(Joint(joint), side == "left")
            if not lo <= angle <= hi:
                raise SystemExit(
                    f"--hold: {name}={deg}° is outside "
                    f"[{math.degrees(lo):.0f}, {math.degrees(hi):.0f}]° for that arm"
                )
        out[_COLUMNS.index(name)] = angle
    return out


def _apply_holds(
    rows: np.ndarray, holds: dict[int, float | None], first: np.ndarray
) -> np.ndarray:
    """A copy of ``rows`` with each held column constant (``first`` = row 0)."""
    out = np.array(rows, dtype=float, copy=True)
    for col, angle in holds.items():
        out[:, col] = first[col] if angle is None else angle
    return out


#: A joint further than this (rad, ~3°) from the motion's first row after the
#: approach move did not follow it, and playback must not start from there.
_START_POSE_TOL = 0.05


def retime_measurements(
    t: np.ndarray, offsets: np.ndarray, actual: np.ndarray, torque: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Put cache reads back on the command clock.

    ``offsets[k, i]`` is how long before log time ``t[k]`` joint ``i``'s
    cached sample was really taken (≤ 0; ``0`` when unknown). Each column
    is interpolated from its true sample times ``t + offsets`` back onto
    ``t``, so the scorecard compares the target with the measurement at the
    same instant instead of with a sample up to one core tick old — the
    sawtooth in that age is what a 400 Hz core read at 240 Hz shows as an
    80 Hz buzz on every joint. Duplicate samples (the same tick read twice)
    collapse to one point; NaN columns (absent arm) pass through.
    """
    if len(t) < 2 or offsets.shape != actual.shape:
        return actual, torque
    out_a = actual.copy()
    out_q = torque.copy()
    for i in range(actual.shape[1]):
        col = actual[:, i]
        if not np.any(np.isfinite(col)) or not np.any(offsets[:, i] != 0.0):
            continue
        ts = t + offsets[:, i]
        keep = np.concatenate([[True], np.diff(ts) > 0])
        keep &= np.isfinite(col)
        if keep.sum() < 2:
            continue
        out_a[:, i] = np.interp(
            t, ts[keep], col[keep], left=col[keep][0], right=col[keep][-1]
        )
        tq = torque[:, i]
        if np.any(np.isfinite(tq)):
            kq = keep & np.isfinite(tq)
            if kq.sum() >= 2:
                out_q[:, i] = np.interp(
                    t, ts[kq], tq[kq], left=tq[kq][0], right=tq[kq][-1]
                )
    return out_a, out_q


def start_pose_stragglers(
    q_now: np.ndarray,
    q_start: np.ndarray,
    arms: list[tuple[str, np.ndarray]],
    tol: float = _START_POSE_TOL,
) -> list[tuple[str, float]]:
    """Joints not at the motion start pose: ``[(column, error_deg), ...]``.

    ``arms`` lists the arms actually driven, ``(side, full-N indices)`` —
    an arm left off with ``--arms`` reads as rest and must not be judged.

    The approach move is streamed, not verified — a joint that will not
    follow the stream (a ``--a4`` joint whose stored planner acceleration
    is neither 0 nor 60000 barely moves, see ``tune.a4``) is silently left
    at rest, and replaying from there scores garbage for that joint and
    swings the others around a pose the motion never planned for.
    """
    out: list[tuple[str, float]] = []
    for side, indices in arms:
        for j, idx in zip(ARM_JOINTS, indices):
            err = float(q_now[idx] - q_start[idx])
            if abs(err) > tol:
                out.append((f"{side}.{j.value}", math.degrees(err)))
    return out


def _parse_gain_overrides(specs: list[str]) -> dict[tuple[str, str, str], float]:
    """Parse ``--gain [side.]joint.field=value`` into ``{(side, joint, field): v}``.

    Omitting the side applies the override to both arms.
    """
    joints = {j.value for j in ARM_JOINTS}
    out: dict[tuple[str, str, str], float] = {}
    for spec in specs:
        path, _, raw = spec.partition("=")
        parts = path.split(".")
        try:
            value = float(raw)
        except ValueError:
            raise SystemExit(f"--gain: bad value in {spec!r} (want PATH=NUMBER)")
        # ``[side.]joint.friction.fc`` / ``joint.firmware.speed_kp``: fold the
        # sub-field back into one token.
        if len(parts) >= 2 and parts[-2] in ("friction", "firmware", "com"):
            parts = parts[:-2] + [f"{parts[-2]}.{parts[-1]}"]
        if len(parts) == 3:
            sides, joint, fld = [parts[0]], parts[1], parts[2]
            if sides[0] not in ("left", "right"):
                raise SystemExit(f"--gain: side must be left or right in {spec!r}")
        elif len(parts) == 2:
            sides, joint, fld = ["left", "right"], parts[0], parts[1]
        else:
            raise SystemExit(f"--gain: want [side.]joint.field=value, got {spec!r}")
        if joint not in joints:
            raise SystemExit(f"--gain: unknown joint {joint!r} in {spec!r}")
        if fld not in _GAIN_FIELDS:
            raise SystemExit(
                f"--gain: unknown field {fld!r} in {spec!r} "
                f"(one of {', '.join(_GAIN_FIELDS)})"
            )
        if fld == "firmware.tf_rated_current_a":
            if joint in ("wrist_2", "wrist_3"):
                raise SystemExit(
                    f"--gain: {fld} scales the MyActuator 0x73 feedforward; {joint} "
                    "is a Damiao wrist"
                )
            if not (math.isfinite(value) and value > 0.0):
                raise SystemExit(f"--gain {spec}: the rated current in amps, > 0")
        if fld in (
            "firmware.planner_accel",
            "firmware.cap_track",
            "firmware.planner_lead_ms",
        ):
            if joint in ("wrist_2", "wrist_3"):
                raise SystemExit(
                    f"--gain: {fld} is the MyActuator 0xA4 planner's; {joint} is a "
                    "Damiao wrist (its profiler is firmware.profile_acc)"
                )
            try:
                check_firmware_extras(
                    value if fld == "firmware.planner_accel" else None,
                    value if fld == "firmware.cap_track" else None,
                    value if fld == "firmware.planner_lead_ms" else None,
                )
            except ValueError as exc:
                raise SystemExit(f"--gain {spec}: {exc}") from None
        for side in sides:
            out[(side, joint, fld)] = value
    return out


def _apply_gain_overrides(
    config: AxolConfig, overrides: dict[tuple[str, str, str], float]
) -> None:
    """Set each ``(side, joint, field)`` override on ``config``, that joint only.

    The ``friction`` / ``firmware`` blocks are shared instances across the
    joints of a motor type (shoulder_1 + shoulder_2, both elbows, ...), so
    those are replaced with this joint's own copy, never mutated in place —
    an in-place set once carried a shoulder_1 planner override onto
    shoulder_2 (2026-09-22).
    """
    for (side, joint, fld), value in overrides.items():
        target = getattr(getattr(config, side), joint)
        if fld.startswith("friction."):
            target.friction = replace(target.friction, **{fld.split(".", 1)[1]: value})
        elif fld.startswith("firmware."):
            target.firmware = replace(target.firmware, **{fld.split(".", 1)[1]: value})
        elif fld.startswith("com."):
            com = list(target.com)
            com["xyz".index(fld[-1])] = value
            target.com = tuple(com)
        else:
            setattr(target, fld, value)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.motion`` subcommand."""
    p = subparsers.add_parser(
        "tune.motion",
        help="Replay a reference motion through motion_control and score tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--motion",
        required=True,
        help="Committed motion name (axol motion.list) or a path to a motion .npz",
    )
    p.add_argument(
        "--gain",
        action="append",
        default=None,
        metavar="[SIDE.]JOINT.FIELD=VALUE",
        help="Override one gain for this run, e.g. left.elbow.kd=4.5 or "
        "shoulder_3.kd_host=8 (no side = both arms). Fields: "
        f"{', '.join(_GAIN_FIELDS)}. Repeatable. Overrides are the tuned "
        "s=1 anchors, like the config defaults they replace.",
    )
    p.add_argument(
        "--stiffness",
        type=float,
        default=1.0,
        help="Stiffness-slider position in [0, 1] applied to both arms "
        "(default: 1.0, the production default — the tuned gains, where "
        "gain overrides land exactly; lower only adds compliance)",
    )
    p.add_argument(
        "--noise",
        choices=("none", "network", "ik", "combined"),
        default="none",
        help="Corrupt the motion before streaming it, at the noise source's "
        "real pipeline entry point (see tune.filter): 'network' = "
        "jitter/outliers/stalls, 'ik' = solver churn/jumps, 'combined' = "
        "both. Deterministic per --seed. Default: none (clean playback).",
    )
    p.add_argument(
        "--filter",
        action="store_true",
        help="Replay the (possibly noise-corrupted) command stream through "
        "the production teleop filter stack (pose low-pass -> EMA -> "
        "trapezoid) before streaming — the hardware version of tune.filter: "
        "the arm physically shows what the stack removes and what it costs "
        "in lag. Off: the stream is sent as-is.",
    )
    p.add_argument(
        "--ik",
        action="store_true",
        help="Drive the run through the IK solver: each waypoint's "
        "end-effector poses (FK of the reference, with elbow hints) are "
        "re-solved to joints exactly like teleop's pose->joints loop, and "
        "the arms execute the solver's output — still scored against the "
        "clean reference, so IK reconstruction error and tracking error "
        "show up together. Composes after --noise/--filter.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --noise — identical seed, identical corrupted "
        "stream (default: 0)",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Free-form note stored on the run artifact (shows up in "
        "listings and the UI)",
    )
    p.add_argument(
        "--torque-threshold",
        type=float,
        default=8.0,
        help="Contact watchdog: a joint torque residual (measured minus "
        "modeled gravity, Nm) sustained above this aborts playback "
        "(default: 8.0; 0 disables)",
    )
    p.add_argument(
        "--no-save-run",
        action="store_true",
        help="Don't persist the run artifact (dry run)",
    )
    p.add_argument(
        "--a4",
        action="append",
        default=[],
        metavar="SIDE.JOINT",
        help="Drive this MyActuator joint with the firmware's own position loop "
        "(0xA4 absolute position closed-loop) instead of the MIT impedance frame, "
        "e.g. right.shoulder_1. Repeatable. The joint then has no compliance, no "
        "host feedforward and NaN torque telemetry (contact watchdog blind on it); "
        "everything else about the replay is unchanged, so runs compare directly.",
    )
    p.add_argument(
        "--hold",
        action="append",
        default=[],
        metavar="SIDE.JOINT[=DEG]",
        help="Hold this joint steady for the replay instead of following the "
        "motion, e.g. right.elbow (at the motion's own start angle) or "
        "right.elbow=-75 (at that joint-frame angle, degrees; the approach goes "
        "there). Repeatable. The joint keeps its controller and gains, commanded "
        "to one pose, and is scored as a parked joint (buzz / chatter only). "
        "Only the approach is collision-checked: a joint frozen while the others "
        "move can bring links closer than the recording ever did.",
    )
    p.add_argument(
        "--loop-hz",
        type=float,
        default=None,
        help="Realtime-core tick rate override. Default follows the wire modes: "
        "240 Hz all impedance, 400 Hz all firmware loops (--controller "
        "position), 480 Hz mixed (--a4 joints every tick, impedance joints on "
        "alternate ticks). Impedance (MIT) is commanded at 240 Hz only, so with "
        "any arm joint on it only 240 or 480 is accepted. For A/B runs: "
        "--controller position --loop-hz 240 separates the rate from the "
        "controller. Above 300 Hz the core thins the bus schedule.",
    )
    p.add_argument(
        "--no-imu",
        action="store_true",
        help="Do not record the wrist cameras' IMUs (by default each driven arm's "
        "wrist ZED X One IMU is recorded and the run gets an 'imu' shake score: "
        "1-15 Hz displacement p2p in mm, what the encoders cannot see).",
    )
    p.add_argument(
        "--fast-impedance",
        action="append",
        default=[],
        metavar="SIDE.JOINT",
        help="Run this impedance joint at 480 Hz — every tick of a 480 Hz core "
        "loop, its host feedforward, damping and tracker stepped at 480 — while "
        "every other impedance joint stays at 240 Hz on alternate ticks, e.g. "
        "--fast-impedance right.shoulder_1 --fast-impedance right.elbow. "
        "Repeatable. An experiment: the gains were tuned at 240.",
    )
    p.add_argument(
        "--impedance-hz",
        type=float,
        choices=IMPEDANCE_RATES,
        default=None,
        help="Command rate of the MyActuator impedance joints for this run: 240 "
        "(the config default, verified) or 480 — every tick of a 480 Hz core "
        "loop with the Damiao wrists staying at 240 Hz on alternate ticks. An "
        "experiment: the impedance gains, host damping and feedforward were "
        "tuned at 240.",
    )
    p.add_argument(
        "--record",
        metavar="PREFIX",
        default=None,
        help="Flight-recorder prefix, as teleop's --teleop.record: the replay's "
        "measured joints go to PREFIX_meas.npz and the realtime core's per-tick "
        "trace (target, command, measured position, motor speed, feed-forward "
        "terms) to PREFIX_rt.npz for `axol diag.teleop-jitter` or offline "
        "analysis. A bare name lands in the recordings directory.",
    )
    p.add_argument(
        "--controller",
        choices=CONTROLLERS,
        default=None,
        help="Which control law the core runs the arms on for this run. "
        "'impedance' (the config default) is the production MIT frame at 240 Hz "
        "with the host feedforward; 'position' puts every joint on its motor's "
        "own position loop (MyActuator 0xA4, Damiao position-velocity; the "
        "firmware.* gains) streamed at 400 Hz — stiff, no host feedforward, "
        "NaN torque on the MyActuator joints. --a4 still adds single joints "
        "inside the impedance controller.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Replay the motion N times back to back in one session (default 1; "
        "0 = until Ctrl-C): no homing in between — each pass after the first "
        "starts with a planned move back to the start pose if the motion does "
        "not end there. Each pass is scored and saved as its own run (label "
        "suffixed [k/N]) and a one-line-per-pass summary closes the session; "
        "--record captures the whole session in one trace. For soak runs and "
        "catching an intermittent buzz.",
    )
    p.add_argument(
        "--learn",
        type=int,
        default=0,
        metavar="N",
        help="Iterative learning over N passes (in place of --repeat): pass 1 "
        "runs uncorrected, and after each pass a per-joint position offset on "
        "the streamed motion is updated from that pass's tracking error in the "
        "--learn-band — first the error advanced by the joint's lag, then the "
        "inverse of each joint's command-to-position response measured from "
        "the passes themselves. A pass that comes out worse rolls back to the "
        "best offset with the gain halved. Each pass is saved with the offset "
        "it flew ('correction'), for --correction. The offset is specific to "
        "this motion: it measures what each joint needed.",
    )
    p.add_argument(
        "--learn-imu",
        action="store_true",
        help="With --learn: the error the learning cancels is the tool's "
        "vertical deviation from its commanded path as the wrist IMU measures "
        "it — including flex the joint encoders cannot see — spread over the "
        "learned joints by their lever arms, instead of each joint's encoder "
        "error. Tests whether joint commands can cancel that motion at all "
        "(single driven arm with its wrist camera)",
    )
    p.add_argument(
        "--learn-gain",
        type=float,
        default=0.7,
        help="Step size of the model-based learning updates (default 0.7; "
        "0.9 converges faster where passes repeat closely)",
    )
    p.add_argument(
        "--learn-band",
        type=float,
        nargs=2,
        default=list(LEARN_BAND),
        metavar=("LO", "HI"),
        help=f"Band (Hz) the learning corrects (default {LEARN_BAND[0]:g} "
        f"{LEARN_BAND[1]:g}): the shake, above the deliberate motion and its lag",
    )
    p.add_argument(
        "--learn-max-deg",
        type=float,
        default=1.5,
        help="Clamp on the learned offset per joint (degrees, default 1.5)",
    )
    p.add_argument(
        "--learn-joint",
        action="append",
        default=[],
        metavar="SIDE.JOINT",
        help="Learn only these joints (repeatable; default: every joint of the "
        "driven arms that moves at least 1° and is not --hold)",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Pre-compensate the streamed motion with each joint's tracking "
        "model (axol tune.tf --save): the reference through the inverse of the "
        "measured closed-loop response, so the lag and resonance cancel. "
        "Joints without a model stream as before; scored against the clean "
        "reference",
    )
    p.add_argument(
        "--notch",
        type=float,
        action="append",
        default=[],
        metavar="HZ",
        help="Notch the streamed motion at HZ (repeatable) with teleop's "
        "command notch (VRTeleopConfig.command_notch_hz: causal biquads "
        "between the IK EMA and the tracker) — for structural modes the joint "
        "loops cannot damp; scored against the clean reference",
    )
    p.add_argument(
        "--notch-q",
        type=float,
        default=2.0,
        help="Quality factor of the --notch filters (default 2.0; bandwidth f0/Q)",
    )
    p.add_argument(
        "--imu-damp",
        type=float,
        default=0.0,
        metavar="C",
        help="Damp the tool's vertical shake from the wrist IMU: a force of "
        "-C x the flex velocity (N·s/m) — the IMU's band vertical velocity "
        "less what the joint encoders account for — at the tool, applied "
        "through --imu-damp-joint by the measured-pose Jacobian (see "
        "almond_axol.tuning.imu_damping). Start at 5; the simulation damped "
        "a hidden 2 Hz mode 31/44/52%% at 5/10/20 and diverged by 80. The "
        "damper switches itself off if the shake it measures runs away",
    )
    p.add_argument(
        "--imu-damp-joint",
        action="append",
        default=[],
        metavar="SIDE.JOINT[=SCALE]",
        help="Joints that apply the IMU damping, optionally with a scale on "
        "--imu-damp for that joint (repeatable; default the driven arm's "
        "shoulder_1, shoulder_2 and elbow)",
    )
    p.add_argument(
        "--imu-damp-max",
        type=float,
        default=0.5,
        metavar="NM",
        help="Per-joint clamp on the IMU damping torque (default 0.5 Nm)",
    )
    p.add_argument(
        "--imu-damp-hp",
        type=float,
        default=1.0,
        metavar="HZ",
        help="Low edge of the damped band (default 1.0). The high-pass leads "
        "by 90° at its own corner, so below ~2× this the force is more spring "
        "than damper; 0.5 halves the lead at 2 Hz and lets ~4 mm/s of the "
        "commanded motion through on slow_osc",
    )
    p.add_argument(
        "--gyro-damp",
        type=float,
        default=0.0,
        metavar="C",
        help="Damp the flex past the joint encoders from the wrist gyro: "
        "each --gyro-damp-joint gets a torque of -C x the gyro's rotation "
        "rate less the encoder-implied rate, about that joint's axis "
        "(N·m·s/rad; see almond_axol.tuning.imu_damping.GyroFlexDamper). "
        "Needs the camera mount (--gyro-mount). Shares --imu-damp-max and "
        "--imu-damp-alternate",
    )
    p.add_argument(
        "--gyro-damp-joint",
        action="append",
        default=[],
        metavar="SIDE.JOINT[=C]",
        help="Joints that apply the gyro damping, optionally with their own "
        "gain (repeatable; default the driven arm's shoulder_1 and elbow at "
        "--gyro-damp)",
    )
    p.add_argument(
        "--gyro-mount",
        default=str(Path.home() / ".almond" / "wrist_imu_mount.json"),
        metavar="PATH",
        help="The wrist cameras' rotation against the gripper mount, per side "
        "(default ~/.almond/wrist_imu_mount.json; write it with "
        "--gyro-mount-fit)",
    )
    p.add_argument(
        "--torque-probe",
        action="append",
        default=[],
        metavar="SIDE.JOINT=NM",
        help="Add a known multisine torque (0.5-15 Hz, NM peak) to this joint "
        "(repeatable; add --imu-damp-alternate to probe every second pass) and "
        "log the gyro flex estimate, to "
        "measure the torque → flex response --gyro-damp closes its loop "
        "through. Needs --gyro-mount",
    )
    p.add_argument(
        "--enc2",
        action="append",
        default=[],
        metavar="SIDE.JOINT",
        help="Read this MyActuator joint's output-side encoder (0x60) every "
        "few ticks into the realtime core's trace (columns enc2_p / enc2_t of "
        "PREFIX_rt.npz; implies --record when none is given) — to compare "
        "it with the motor-side position under load. Repeatable; the core "
        "reads one listed joint per tick",
    )
    p.add_argument(
        "--gyro-mount-fit",
        metavar="RUN_ID",
        help="Fit the wrist camera's mount rotation from a saved tune.motion "
        "run with the IMU recorded, write it to --gyro-mount, and exit",
    )
    p.add_argument(
        "--imu-damp-lp",
        type=float,
        default=15.0,
        metavar="HZ",
        help="High edge of the damped band (default 15; first-order). On "
        "jelly's shoulder_1 the damping's loop phase wraps near 7 Hz: at 200 "
        "N·s/m against the command height it cut the 1-3 Hz sway 18%% and "
        "drove a 7 Hz mode, so a high gain needs this below it",
    )
    p.add_argument(
        "--imu-damp-lead",
        type=float,
        default=0.0,
        metavar="HZ",
        help="Centre of a lead-lag stage on the damped velocity (+37° phase "
        "there; default 0 = none) — for the frequency where the damping's "
        "loop phase runs out",
    )
    p.add_argument(
        "--imu-damp-notch",
        type=float,
        default=0.0,
        metavar="HZ",
        help="Notch the damping force at HZ (Q --imu-damp-notch-q; default 0 "
        "= none), for a mode the damping drives once its loop phase has "
        "wrapped",
    )
    p.add_argument(
        "--imu-damp-notch-q",
        type=float,
        default=1.0,
        help="Quality factor of --imu-damp-notch (default 1.0)",
    )
    p.add_argument(
        "--imu-damp-source",
        choices=("imu", "encoder"),
        default="imu",
        help="What --imu-damp measures the tool's velocity with: the wrist IMU "
        "(default) or FK of the joint encoders (encoder: no camera needed and "
        "no IMU latency, but blind to flex past the joints; always against "
        "the commanded height)",
    )
    p.add_argument(
        "--imu-damp-ref",
        choices=("encoder", "command"),
        default="encoder",
        help="What --imu-damp measures the tool's velocity against: the "
        "height the joint encoders give (encoder, the default: damp only the "
        "flex past the encoders) or the commanded height (command: damp the "
        "whole deviation from the path, encoder-visible wobble included — on "
        "jelly a shoulder_1 torque probe moved the 1-3 Hz tool height "
        "coherently while barely moving the hidden flex)",
    )
    p.add_argument(
        "--imu-damp-alternate",
        action="store_true",
        help="IMU damping on every second pass only (passes 2, 4, ...), for "
        "an A/B inside one session",
    )
    p.add_argument(
        "--correction",
        metavar="RUN_ID",
        help="Fly the offset a --learn run saved (that pass's 'correction') on "
        "every pass, without learning — to check a learned correction holds up",
    )
    p.add_argument(
        "--arms",
        choices=("both", "left", "right"),
        default="both",
        help="Which arm(s) to bring up and drive (default: both). The other "
        "arm's channel is left untouched, so a single-arm bench or an "
        "unpowered arm does not block the run.",
    )
    p.add_argument(
        "--no-gripper",
        action="store_true",
        help="Run on the gripperless SKU (the gripper motor is never "
        "enabled or calibrated)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Replay the selected reference motion and score tracking per joint."""
    logging.basicConfig(level=getattr(logging, args.log_level))
    quiet_noisy_loggers()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nExiting tune.motion ...")


def _print_metrics_table(per_joint: dict[str, dict[str, float]]) -> None:
    """Tracking/smoothness scorecard, one row per scored joint.

    Joints that stayed parked show "-" in the tracking columns (they track
    trivially well) but still carry buzz / chatter, which are meaningful —
    and matter — at rest.
    """

    def fmt(m: dict[str, float], key: str, digits: int, deg: bool = False) -> str:
        v = m.get(key, math.nan)
        if not math.isfinite(v):
            return "-"
        return f"{math.degrees(v) if deg else v:.{digits}f}"

    print(f"\n{'═' * 78}")
    print(
        f"  {'joint':<18} {'RMS °':>7} {'lagfree °':>9} {'lag ms':>7} "
        f"{'jitter °':>8} {'amp':>5} {'trq HF':>7} {'buzz °':>7} {'@Hz':>4}"
    )
    for name, m in per_joint.items():
        print(
            f"  {name:<18} {fmt(m, 'rms_err', 3, deg=True):>7} "
            f"{fmt(m, 'rms_err_lagfree', 3, deg=True):>9} {fmt(m, 'lag_ms', 0):>7} "
            f"{fmt(m, 'err_band_mid', 3, deg=True):>8} {fmt(m, 'amplification', 2):>5} "
            f"{fmt(m, 'torque_hf', 3):>7} {fmt(m, 'buzz', 3, deg=True):>7} "
            f"{fmt(m, 'buzz_hz', 0):>4}"
        )
    print(f"{'═' * 78}")
    print(
        "  RMS = tracking error vs the reference; lagfree = after removing\n"
        "  the measured command->measurement delay; jitter = 3-15 Hz band of\n"
        "  the error (what the operator feels); amp = measured/commanded\n"
        "  mid-band motion (>1 rings, <1 filters); trq HF = torque chatter (Nm);\n"
        "  buzz = sustained >=20 Hz motion (what you hear) at its frequency —\n"
        "  healthy joints sit near 0.005 deg, an audible limit cycle 2-5x that;\n"
        "  parked joints show '-' tracking but are still scored for buzz/chatter."
    )


def _prepare_stream(
    args: argparse.Namespace, motion: ReferenceMotion
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The stream actually sent to the arms: motion -> noise? -> filter?.

    Returns ``(t, sent, ref, info)`` on one uniform grid at the motion's
    rate: ``sent`` is what gets streamed, ``ref`` is the clean reference the
    run is scored against (identical to ``sent`` for a clean playback), and
    ``info`` records what was injected/filtered for the run artifact.
    """
    t_ref = motion.times()
    clean = np.asarray(motion.q, dtype=float)
    info: dict = {"noise": args.noise, "filter": bool(args.filter), "seed": args.seed}
    if args.noise == "none" and not args.filter:
        return t_ref, clean, clean, info

    from ...tuning.filtering import inject_ik_noise, inject_noise, replay_filter_stack

    with_network = args.noise in ("network", "combined")
    with_ik = args.noise in ("ik", "combined")
    # tune.filter's offline defaults — the same insult, now on hardware.
    noisy, events = inject_noise(
        t_ref,
        clean,
        jitter_rms=math.radians(0.3) if with_network else 0.0,
        outlier_rate=0.5 if with_network else 0.0,
        outlier_amp=math.radians(10.0),
        stall_rate=0.5 if with_network else 0.0,
        stall_ms=150.0,
        seed=args.seed,
    )
    ik_noise = None
    ik_events = {"ik_jumps": 0}
    if with_ik:
        ik_noise, ik_events = inject_ik_noise(
            t_ref,
            clean.shape[1],
            churn_rms=math.radians(0.2),
            jump_rate=0.2,
            jump_amp=math.radians(3.0),
            seed=args.seed,
        )
    info.update(events)
    info.update(ik_events)

    if args.filter:
        from ...teleop.config import VRTeleopConfig

        # The stack's control-rate stages run on the playback grid itself
        # (frequency = the motion's rate), so the output streams 1:1.
        t_out, sent, _ = replay_filter_stack(
            t_ref,
            noisy,
            config=VRTeleopConfig(frequency=motion.rate),
            post_lp_noise=ik_noise,
        )
        ref = np.stack(
            [np.interp(t_out, t_ref, clean[:, i]) for i in range(clean.shape[1])],
            axis=1,
        )
        return t_out, sent, ref, info

    sent = noisy if ik_noise is None else noisy + ik_noise
    if args.noise != "none":
        print(
            "  ! streaming raw noise with the filter stack OFF — outlier "
            "teleports go to the arms unsoftened (the contact watchdog "
            "stays active). Compare against a --filter run."
        )
    return t_ref, sent, clean, info


def _ik_stream(solver, sent: np.ndarray, to_full, info: dict) -> np.ndarray:
    """Re-solve a joint stream through the IK solver, like teleop does.

    Each 14-wide row is converted to its two end-effector poses and elbow
    positions by FK, then handed to :meth:`KinematicsSolver.ik`
    warm-started from the previous solution — the same Cartesian
    pose->joints chain teleop runs, so the arms execute what the solver
    produces rather than the recorded joints. The chain is solved before
    playback starts (the pose path is fully known in advance), so a slow
    solve can't stretch the command interval and corrupt the tracking
    scores with pacing jitter; per-solve wall times are still measured and
    stored on the run (``ik_solve_ms_*``) so solver latency stays visible.
    """
    n = len(sent)
    out = np.empty_like(sent)
    solve_ms = np.empty(n)
    q_full = to_full(sent[0])
    t_note = time.perf_counter()
    for k in range(n):
        q_ref = to_full(sent[k])
        left_pose, right_pose = solver.fk(q_ref)
        left_elbow, right_elbow = solver.elbow_positions(q_ref)
        t0 = time.perf_counter()
        q_full = solver.ik(
            q_full,
            left_pose=left_pose,
            right_pose=right_pose,
            left_elbow_pos=left_elbow,
            right_elbow_pos=right_elbow,
        )
        solve_ms[k] = (time.perf_counter() - t0) * 1e3
        out[k] = np.concatenate(
            [q_full[solver.left_indices], q_full[solver.right_indices]]
        )
        if time.perf_counter() - t_note > 5.0:
            print(f"  ... {k + 1}/{n} waypoints solved")
            t_note = time.perf_counter()
    dev = out - sent
    # The first solve absorbs the JIT trace of this call signature (the
    # planner's warm-up doesn't cover the elbow-hint variant) — seconds, not
    # milliseconds — so it would swamp the mean of an otherwise steady chain.
    steady = solve_ms[1:] if n > 1 else solve_ms
    info.update(
        ik=True,
        ik_solve_ms_mean=float(steady.mean()),
        ik_solve_ms_p99=float(np.percentile(steady, 99)),
        ik_dev_rms_deg=float(math.degrees(np.sqrt(np.mean(dev**2)))),
        ik_dev_max_deg=float(math.degrees(np.max(np.abs(dev)))),
    )
    print(
        f"  IK re-solve: solve {info['ik_solve_ms_mean']:.2f} ms mean / "
        f"{info['ik_solve_ms_p99']:.2f} ms p99; solved joints deviate from "
        f"the reference by {info['ik_dev_rms_deg']:.3f}° RMS "
        f"({info['ik_dev_max_deg']:.2f}° max) — that deviation is part of "
        "what the run scores"
    )
    return out


#: The height Jacobian is refreshed every this many ticks (the pose moves
#: slowly against the shake; one batched FK of 4 poses per refresh).
_JAC_EVERY = 4


def _imu_dampers(args: argparse.Namespace) -> dict[str, Any]:
    """One :class:`TipDamper` per driven side, or none (``--imu-damp 0``)."""
    from ...tuning.imu_damping import (
        EncoderTipDamper,
        EncoderVelocity,
        TipDamper,
        VerticalVelocity,
    )

    if args.imu_damp <= 0:
        return {}
    by_encoder = args.imu_damp_source == "encoder"
    if args.no_imu and not by_encoder:
        raise SystemExit("tune.motion: --imu-damp needs the wrist IMU (drop --no-imu)")
    names = [j.value for j in ARM_JOINTS]
    sides = ["left", "right"] if args.arms == "both" else [args.arms]
    out = {}
    for side in sides:
        specs = args.imu_damp_joint or [
            f"{side}.shoulder_1",
            f"{side}.shoulder_2",
            f"{side}.elbow",
        ]
        cols = []
        weights: dict[int, float] = {}
        for spec in specs:
            name, _, scale = spec.partition("=")
            s_side, _, joint = name.partition(".")
            if joint not in names or s_side not in ("left", "right"):
                raise SystemExit(
                    f"--imu-damp-joint wants SIDE.JOINT[=SCALE], got {spec!r}"
                )
            if s_side == side:
                cols.append(names.index(joint))
                if scale:
                    weights[names.index(joint)] = float(scale)
        if cols:
            kind = EncoderTipDamper if by_encoder else TipDamper
            extra = (
                {
                    "command": EncoderVelocity(
                        hp_hz=args.imu_damp_hp, lp_hz=args.imu_damp_lp
                    )
                }
                if by_encoder
                else {}
            )
            out[side] = kind(
                **extra,
                gain=args.imu_damp,
                columns=tuple(cols),
                max_torque=args.imu_damp_max,
                estimator=VerticalVelocity(
                    hp_hz=args.imu_damp_hp, lp_hz=args.imu_damp_lp
                ),
                encoder=EncoderVelocity(hp_hz=args.imu_damp_hp, lp_hz=args.imu_damp_lp),
                lead_hz=args.imu_damp_lead,
                weights=weights,
                notch_hz=args.imu_damp_notch,
                notch_q=args.imu_damp_notch_q,
            )
            print(
                f"  {'encoder' if by_encoder else 'IMU'} damping ({side}): "
                + f"{args.imu_damp:g} N·s/m at the tool through "
                + ", ".join(
                    names[c] + (f" ×{weights[c]:g}" if c in weights else "")
                    for c in cols
                )
                + f" (clamp {args.imu_damp_max:g} Nm, band {args.imu_damp_hp:g}-"
                + f"{args.imu_damp_lp:g} Hz, "
                + (
                    "encoder height against the command"
                    if by_encoder
                    else f"against the {args.imu_damp_ref} height"
                )
                + (", alternate passes)" if args.imu_damp_alternate else ")")
            )
    return out


def _gyro_dampers(args: argparse.Namespace) -> dict[str, Any]:
    """One :class:`GyroFlexDamper` per driven side, or none (``--gyro-damp 0``)."""
    import json

    from ...tuning.imu_damping import GyroFlexDamper

    if args.torque_probe:
        return _torque_probes(args)
    if args.gyro_damp <= 0:
        return {}
    if args.imu_damp > 0:
        raise SystemExit("tune.motion: --gyro-damp and --imu-damp are exclusive")
    if args.no_imu:
        raise SystemExit("tune.motion: --gyro-damp needs the wrist IMU (drop --no-imu)")
    path = Path(args.gyro_mount).expanduser()
    try:
        mounts = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"tune.motion: --gyro-damp needs the camera mount ({path}: {e}); "
            "fit it with --gyro-mount-fit RUN_ID"
        ) from e
    names = [j.value for j in ARM_JOINTS]
    sides = ["left", "right"] if args.arms == "both" else [args.arms]
    out = {}
    for side in sides:
        specs = args.gyro_damp_joint or [f"{side}.shoulder_1", f"{side}.elbow"]
        gains: dict[int, float] = {}
        for spec in specs:
            name, _, c = spec.partition("=")
            s_side, _, joint = name.partition(".")
            if joint not in names or s_side not in ("left", "right"):
                raise SystemExit(
                    f"--gyro-damp-joint wants SIDE.JOINT[=C], got {spec!r}"
                )
            if s_side == side:
                gains[names.index(joint)] = float(c) if c else args.gyro_damp
        if not gains:
            continue
        if side not in mounts:
            raise SystemExit(
                f"tune.motion: no {side} camera mount in {path}; "
                "fit it with --gyro-mount-fit RUN_ID"
            )
        out[side] = GyroFlexDamper(
            gains=gains,
            mount=np.asarray(mounts[side]["rotation"], dtype=float),
            max_torque=args.imu_damp_max,
        )
        print(
            f"  gyro flex damping ({side}): "
            + ", ".join(f"{names[j]} {c:g}" for j, c in sorted(gains.items()))
            + f" N·m·s/rad (clamp {args.imu_damp_max:g} Nm"
            + (", alternate passes)" if args.imu_damp_alternate else ")")
        )
    return out


def _load_gyro_mounts(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    import json

    path = Path(args.gyro_mount).expanduser()
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"tune.motion: gyro damping needs the camera mount ({path}: {e}); "
            "fit it with --gyro-mount-fit RUN_ID"
        ) from e


def _torque_probes(args: argparse.Namespace) -> dict[str, Any]:
    """``--torque-probe``: one :class:`TorqueProbe` per side it names."""
    from ...tuning.imu_damping import TorqueProbe

    if args.no_imu:
        raise SystemExit("tune.motion: --torque-probe needs the wrist IMU")
    path, mounts = _load_gyro_mounts(args)
    names = [j.value for j in ARM_JOINTS]
    amps: dict[str, dict[int, float]] = {}
    for spec in args.torque_probe:
        name, _, a = spec.partition("=")
        side, _, joint = name.partition(".")
        if joint not in names or side not in ("left", "right") or not a:
            raise SystemExit(f"--torque-probe wants SIDE.JOINT=NM, got {spec!r}")
        if abs(float(a)) > 1.0:
            raise SystemExit("--torque-probe: keep the probe at or under 1 Nm")
        amps.setdefault(side, {})[names.index(joint)] = float(a)
    out = {}
    for side, a in amps.items():
        if side not in mounts:
            raise SystemExit(f"tune.motion: no {side} camera mount in {path}")
        out[side] = TorqueProbe(
            amplitudes=a, mount=np.asarray(mounts[side]["rotation"], dtype=float)
        )
        print(
            f"  torque probe ({side}): "
            + ", ".join(f"{names[j]} {v:g} Nm" for j, v in sorted(a.items()))
            + " peak, 0.5-15 Hz multisine"
            + (" on alternate passes" if args.imu_damp_alternate else "")
        )
    return out


def _enable_enc2(args: argparse.Namespace) -> None:
    """``--enc2``: tell the realtime core which joints' output encoder to
    read (``AXOL_RT_ENC2``, by motor id) and make sure its trace is kept."""
    import os

    # CAN ids follow the joint order: shoulder_1 = 1 ... wrist_1 = 5, the
    # MyActuator joints (the Damiao wrists have no second encoder).
    names = [j.value for j in ARM_JOINTS[:5]]
    ids: set[int] = set()
    for spec in args.enc2:
        side, _, joint = spec.partition(".")
        if side not in ("left", "right") or joint not in names:
            raise SystemExit(
                f"--enc2 wants SIDE.JOINT for a MyActuator joint "
                f"({', '.join(names)}), got {spec!r}"
            )
        ids.add(names.index(joint) + 1)
    os.environ["AXOL_RT_ENC2"] = ",".join(str(i) for i in sorted(ids))
    if args.record is None:
        args.record = time.strftime("enc2_%Y%m%d-%H%M%S")
    print(
        f"  output encoder (0x60): {', '.join(args.enc2)} into the "
        f"{args.record}_rt.npz trace"
    )


def _fit_gyro_mount(args: argparse.Namespace) -> None:
    """``--gyro-mount-fit``: the camera mount from a saved run, to --gyro-mount."""
    import json

    from ...kinematics.solver import KinematicsSolver
    from ...tuning.imu_damping import fit_mount

    _, series = load_run(args.gyro_mount_fit)
    solver = KinematicsSolver()
    path = Path(args.gyro_mount).expanduser()
    try:
        mounts = json.loads(path.read_text())
    except (OSError, ValueError):
        mounts = {}
    done = False
    for side, base in (("left", 0), ("right", 7)):
        if f"imu_{side}_t" not in series:
            continue
        q = np.asarray(series["actual"], dtype=float)
        rows = np.zeros((len(q), solver.num_joints), dtype=np.float32)
        rows[:, solver.left_indices] = q[:, :7]
        rows[:, solver.right_indices] = q[:, 7:]
        if not np.all(np.isfinite(q[:, base : base + 7])):
            continue
        rl, rr = solver.ee_rotations(rows)
        mount, r2 = fit_mount(
            series["t"],
            rl if side == "left" else rr,
            series[f"imu_{side}_t"],
            series[f"imu_{side}_gyro"],
        )
        mounts[side] = {
            "rotation": mount.tolist(),
            "fit_r2": r2,
            "run": args.gyro_mount_fit,
            "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"  {side} camera mount: fit R² {r2:.3f} from {args.gyro_mount_fit}")
        if r2 < 0.9:
            print("  ! poor fit: record a run with more wrist rotation")
        done = True
    if not done:
        raise SystemExit(f"--gyro-mount-fit: {args.gyro_mount_fit} has no wrist IMU")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mounts, indent=2))
    print(f"Wrote {path}")


def _joint_axes(
    solver: Any, q_full: np.ndarray, side: str, columns: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """The gripper mount's rotation (world ← mount) at ``q_full`` and the
    world-frame unit axes (7, 3) of ``columns`` (zero rows elsewhere)."""
    h = 1e-3
    idx = solver.left_indices if side == "left" else solver.right_indices
    rows = [np.asarray(q_full, dtype=np.float32)]
    for c in columns:
        r = np.array(q_full, dtype=np.float32, copy=True)
        r[idx[c]] += h
        rows.append(r)
    rl, rr = solver.ee_rotations(np.stack(rows))
    rots = rl if side == "left" else rr
    axes = np.zeros((7, 3))
    for n, c in enumerate(columns):
        d = rots[1 + n] @ rots[0].T
        a = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
        axes[c] = a / max(float(np.linalg.norm(a)), 1e-12)
    return rots[0], axes


def _ee_rotation(solver: Any, q_full: np.ndarray, side: str) -> np.ndarray:
    """The gripper mount's rotation (world ← mount) at ``q_full``."""
    rl, rr = solver.ee_rotations(np.asarray(q_full, dtype=np.float32)[None])
    return (rl if side == "left" else rr)[0]


def _height(solver: Any, q_full: np.ndarray, side: str) -> float:
    """The gripper mount's height (m) at ``q_full``."""
    left, right = solver.ee_positions(np.asarray(q_full, dtype=np.float32)[None])
    return float((left if side == "left" else right)[0, 2])


def _height_jacobian(
    solver: Any, q_full: np.ndarray, side: str, columns: tuple[int, ...]
) -> tuple[float, np.ndarray]:
    """The gripper mount's height (m) at ``q_full`` and ∂height/∂q for the
    arm's 7 joints (m/rad; zero outside ``columns``, central differences)."""
    h = 1e-3
    idx = solver.left_indices if side == "left" else solver.right_indices
    rows = [q_full]
    for c in columns:
        for sign in (1.0, -1.0):
            r = np.array(q_full, dtype=np.float32, copy=True)
            r[idx[c]] += sign * h
            rows.append(r)
    left, right = solver.ee_positions(np.stack(rows))
    z = (left if side == "left" else right)[:, 2]
    jac = np.zeros(7)
    for n, c in enumerate(columns):
        jac[c] = (z[1 + 2 * n] - z[2 + 2 * n]) / (2 * h)
    return float(z[0]), jac


def _clear_extra_torque(axol: Any) -> None:
    for arm in (getattr(axol, "left", None), getattr(axol, "right", None)):
        if arm is not None:
            arm.extra_torque = None


def _invert_stream(
    sent: Any,
    ref: Any,
    holds: dict[int, Any],
    overrides: dict[tuple[str, str, str], float],
    rate: float,
    arms: str,
) -> np.ndarray:
    """``sent`` with every modelled, moving, unheld joint pre-compensated
    (``--invert``)."""
    from ...tuning.tracking_model import (
        TRACKING_MODELS_PATH,
        invert_reference,
        load_models,
    )

    models = load_models()
    if not models:
        raise SystemExit(
            f"--invert: no tracking models in {TRACKING_MODELS_PATH} — "
            "run axol motion.chirp / tune.motion / tune.tf --save first"
        )
    out = np.array(sent, dtype=float, copy=True)
    ref_arr = np.asarray(ref, dtype=float)
    current = {f"{s}.{j}.{f}": v for (s, j, f), v in overrides.items()}
    for i, name in enumerate(_COLUMNS):
        model = models.get(name)
        side = name.split(".")[0]
        if model is None or i in holds or arms not in ("both", side):
            continue
        if np.ptp(ref_arr[:, i]) < math.radians(1.0):
            continue
        if model.fn_hz > 1.2 * model.f_hi:
            print(
                f"  invert: {name} skipped — its model's resonance "
                f"({model.fn_hz:.1f} Hz) is outside the band it was measured on"
            )
            continue
        mine = {k: v for k, v in current.items() if k.startswith(name + ".")}
        theirs = {
            k: v for k, v in (model.gains or {}).items() if k.startswith(name + ".")
        }
        if mine != theirs:
            print(
                f"  ! invert: {name}'s model was measured with {theirs or 'config gains'}, "
                f"this run has {mine or 'config gains'} — re-measure if the loop changed"
            )
        out[:, i] = invert_reference(out[:, i], rate, model)
        change = np.degrees(
            np.abs(out[:, i] - np.asarray(sent, dtype=float)[:, i]).max()
        )
        print(
            f"  invert: {name} through its {model.fn_hz:.2f} Hz / ζ {model.zeta:.2f} / "
            f"{model.tau * 1e3:.0f} ms model (peak change {change:.3f}°)"
        )
    return out


def _learn_columns(
    args: argparse.Namespace, ref: np.ndarray, holds: dict[int, Any]
) -> np.ndarray:
    """Which of the 14 columns --learn corrects: --learn-joint, or every
    joint of the driven arms that moves at least 1° and is not held."""
    if args.learn_joint:
        mask = np.zeros(len(_COLUMNS), dtype=bool)
        for spec in args.learn_joint:
            if spec not in _COLUMNS:
                raise SystemExit(f"--learn-joint wants SIDE.JOINT, got {spec!r}")
            mask[_COLUMNS.index(spec)] = True
    else:
        mask = np.ptp(ref, axis=0) >= math.radians(1.0)
    for i, name in enumerate(_COLUMNS):
        side = name.split(".")[0]
        if args.arms not in ("both", side) or i in holds:
            mask[i] = False
    return mask


def _load_correction(run_id: str, n: int) -> np.ndarray:
    """The ``correction`` a --learn pass flew, checked against this motion."""
    loaded = load_run(run_id)
    if loaded is None:
        raise SystemExit(f"--correction: no tuning run {run_id!r}")
    _, series = loaded
    corr = series.get("correction")
    if corr is None:
        raise SystemExit(f"--correction: run {run_id} has no learned correction")
    corr = np.asarray(corr, dtype=float)
    if corr.shape != (n, len(_COLUMNS)):
        raise SystemExit(
            f"--correction: run {run_id}'s correction is {corr.shape}, this "
            f"motion streams {(n, len(_COLUMNS))} — a different motion?"
        )
    return corr


def _learn_step(
    learner: CommandLearner,
    k: int,
    t: np.ndarray,
    meas_offset: np.ndarray,
    actual: np.ndarray,
    torque: np.ndarray,
    ref: np.ndarray,
) -> None:
    """Feed a finished pass to the learner and print what it did."""
    if len(t) < learner.n:
        print(f"  learning: pass {k + 1} cut short — offset left unchanged")
        return
    actual, _ = retime_measurements(t, meas_offset, actual, torque)
    rep = learner.update(ref, actual)
    cols = np.where(learner.columns)[0]
    joints = "  ".join(
        f"{_COLUMNS[i].split('.')[1]} {math.degrees(rep.band_rms[i]) * 1e3:.0f}"
        for i in cols
    )
    action = {
        "first": "first step (error advanced by the lag)",
        "model": f"model step (gain {rep.gain:g})",
        "rollback": f"worse than the best pass — rolled back, gain now {rep.gain:g}",
    }[rep.step]
    print(
        f"  learning: pass {k + 1} band error {math.degrees(rep.total) * 1e3:.1f} mdeg "
        f"rms [{joints}] → {action}; next offset peak "
        f"{math.degrees(float(np.abs(learner.offset).max())):.3f}°"
    )


def _tool_error_as_joints(
    t: np.ndarray,
    imu_t: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro: np.ndarray | None,
    ref: np.ndarray,
    columns: np.ndarray,
    side: str,
    to_full: Any,
    solver: Any,
    band: tuple[float, float],
) -> tuple[np.ndarray, float]:
    """The wrist IMU's tool-height error mapped onto the learned joints.

    The IMU's vertical displacement in ``band`` (gravity-projected, twice
    integrated) against the commanded tool height through the same band; the
    difference is spread over ``columns`` by the height Jacobian's
    minimum-norm inverse (each joint takes its lever's share). Returns the
    ``(N, 14)`` joint-space error (command minus actual convention) and the
    tool error's RMS (m).
    """
    from ...tuning.learning import band_limit
    from ...tuning.wrist_imu import _band_integrate

    fs_i = 1.0 / float(np.median(np.diff(imu_t)))
    grid = np.arange(imu_t[0], imu_t[-1], 1.0 / fs_i)
    acc = np.stack([np.interp(grid, imu_t, imu_acc[:, i]) for i in range(3)], 1)
    # Vertical acceleration with gravity carried through the camera's
    # rotation by the gyro: the wrist turns tens of degrees in a motion, and
    # a fixed "up" leaks that tilt in as fake vertical motion at the band's
    # low edge, where double integration amplifies it most.
    up = acc[: int(fs_i)].mean(axis=0)
    a_v = np.empty(len(grid))
    gyro = (
        np.radians(
            np.stack([np.interp(grid, imu_t, imu_gyro[:, i]) for i in range(3)], 1)
        )
        if imu_gyro is not None
        else None
    )
    dt = 1.0 / fs_i
    for n in range(len(grid)):
        if gyro is not None:
            up = up - np.cross(gyro[n], up) * dt
        up = up + (acc[n] - up) * (dt / 2.0)
        gn = float(np.linalg.norm(up)) or 9.80665
        a_v[n] = float(acc[n] @ up) / gn - gn
    z_imu = np.interp(t, grid, _band_integrate(a_v[:, None], fs_i, band)[:, 0])
    fs = 1.0 / float(np.median(np.diff(t)))
    full = np.stack([to_full(r) for r in ref]).astype(np.float32)
    left, right = solver.ee_positions(full)
    z_cmd = band_limit((left if side == "left" else right)[:, 2], fs, band)
    tool_err = z_cmd - z_imu  # positive: the tool is below its path
    idx = solver.left_indices if side == "left" else solver.right_indices
    base = 0 if side == "left" else 7
    cols = [c for c in np.where(columns)[0] if base <= c < base + 7]
    step = 12
    sub = np.arange(0, len(full), step)
    jac = np.zeros((len(sub), len(cols)))
    h = 1e-3
    for n, c in enumerate(cols):
        pert = full[sub].copy()
        pert[:, idx[c - base]] += h
        lp, rp = solver.ee_positions(pert)
        jac[:, n] = (
            (lp if side == "left" else rp)[:, 2]
            - (left if side == "left" else right)[sub, 2]
        ) / h
    jac_full = np.stack(
        [np.interp(np.arange(len(full)), sub, jac[:, n]) for n in range(len(cols))], 1
    )
    norm = np.maximum(np.sum(jac_full**2, axis=1), 1e-6)
    err = np.zeros((len(ref), len(columns)))
    for n, c in enumerate(cols):
        err[:, c] = jac_full[:, n] * tool_err / norm
    return err, float(np.std(tool_err))


def _learn_step_imu(
    learner: CommandLearner,
    k: int,
    imu: Any,
    arms: str,
    t: np.ndarray,
    t_abs: np.ndarray,
    ref: np.ndarray,
    to_full: Any,
    solver: Any,
) -> None:
    """``--learn-imu``: feed the learner the IMU-measured tool error."""
    side = arms if arms in ("left", "right") else "right"
    if len(t) < learner.n:
        print(f"  learning: pass {k + 1} cut short — offset left unchanged")
        return
    imu.flush()
    w = imu.window(
        side, float(t_abs[0]), float(t_abs[-1]), origin=float(t_abs[0] - t[0])
    )
    if w is None or len(w["t"]) < 100:
        print(
            f"  learning: pass {k + 1} has no {side} wrist IMU data — offset left unchanged"
        )
        return
    err, tool_rms = _tool_error_as_joints(
        t[: learner.n],
        w["t"],
        w["acc"],
        w.get("gyro"),
        ref[: learner.n],
        learner.columns,
        side,
        to_full,
        solver,
        learner.band,
    )
    # The learner cancels ``ref - actual``; hand it the tool error that way.
    rep = learner.update(ref[: learner.n], ref[: learner.n] - err)
    action = {
        "first": "first step (error advanced by the lag)",
        "model": f"model step (gain {rep.gain:g})",
        "rollback": f"worse than the best pass — rolled back, gain now {rep.gain:g}",
    }[rep.step]
    print(
        f"  learning (IMU): pass {k + 1} tool error {tool_rms * 1e3:.2f} mm rms "
        f"({learner.band[0]:g}-{learner.band[1]:g} Hz) → {action}; next offset "
        f"peak {math.degrees(float(np.abs(learner.offset).max())):.3f}°"
    )


async def _run(args: argparse.Namespace) -> None:
    if args.gyro_mount_fit:
        _fit_gyro_mount(args)
        return
    if args.enc2:
        _enable_enc2(args)
    motion = _load_motion_or_exit(args.motion)
    overrides = _parse_gain_overrides(args.gain or [])

    print(
        f"Reference motion {motion.name!r}: {len(motion.q)} waypoints, "
        f"{motion.duration:.1f} s at {motion.rate:.0f} Hz"
    )
    _, sent, ref, stream_info = _prepare_stream(args, motion)
    stream_differs = args.noise != "none" or args.filter
    if stream_differs:
        print(
            f"  stream: noise={args.noise}, "
            f"filter={'on' if args.filter else 'off'} (seed {args.seed}) — "
            "scored against the clean reference"
        )

    if not 0.0 <= args.stiffness <= 1.0:
        raise SystemExit("--stiffness must be in [0, 1]")
    config = AxolConfig(
        left_stiffness=args.stiffness,
        right_stiffness=args.stiffness,
        has_gripper=not args.no_gripper,
    )
    _apply_gain_overrides(config, overrides)
    for (side, joint, fld), value in overrides.items():
        print(f"  gain override: {side}.{joint}.{fld} = {value}")
    for spec in args.a4:
        parts = spec.split(".")
        if len(parts) != 2 or parts[0] not in ("left", "right"):
            raise SystemExit(f"--a4 wants SIDE.JOINT, got {spec!r}")
        side, joint = parts
        if joint not in {j.value for j in ARM_JOINTS}:
            raise SystemExit(f"--a4: unknown joint {joint!r}")
        getattr(getattr(config, side), joint).wire_mode = "a4"
        print(f"  wire mode: {side}.{joint} = a4 (firmware position loop)")
    holds = _parse_holds(args.hold)
    if args.controller is not None:
        config.controller = args.controller
    if args.impedance_hz is not None:
        config.impedance_hz = args.impedance_hz
    for spec in args.fast_impedance:
        parts = spec.split(".")
        if len(parts) != 2 or parts[0] not in ("left", "right"):
            raise SystemExit(f"--fast-impedance wants SIDE.JOINT, got {spec!r}")
        side, joint = parts
        if joint not in {j.value for j in ARM_JOINTS}:
            raise SystemExit(f"--fast-impedance: unknown joint {joint!r}")
        getattr(getattr(config, side), joint).impedance_hz = FAST_IMPEDANCE_HZ
        print(f"  impedance rate: {side}.{joint} = {FAST_IMPEDANCE_HZ:.0f} Hz")
    if args.repeat < 0:
        raise SystemExit("tune.motion: --repeat must be 0 (until Ctrl-C) or more")
    if args.learn_imu and not args.learn:
        raise SystemExit("tune.motion: --learn-imu goes with --learn N")
    if args.learn_imu and (args.no_imu or args.arms == "both"):
        raise SystemExit(
            "tune.motion: --learn-imu needs one driven arm and its wrist IMU"
        )
    if args.learn:
        if args.learn < 2:
            raise SystemExit("tune.motion: --learn needs at least 2 passes")
        if args.correction:
            raise SystemExit("tune.motion: --learn and --correction are exclusive")
        args.repeat = args.learn
    try:
        # Before anything touches the bus: impedance runs at 240 Hz only.
        check_loop_hz(config, args.loop_hz or config.loop_hz)
    except ValueError as exc:
        raise SystemExit(f"tune.motion: {exc}") from None
    core_hz = args.loop_hz or config.loop_hz
    fast = bool(fast_impedance_joints(config))
    mixed = config.controller != "position" and core_hz > IMPEDANCE_LOOP_HZ
    print(
        f"  controller: {config.controller} "
        f"({core_hz:.0f} Hz core loop"
        + (
            ", every joint on its firmware position loop)"
            if config.controller == "position"
            else (
                f", {', '.join(fast_impedance_joints(config))} every tick, the other "
                f"impedance joints at {IMPEDANCE_LOOP_HZ:.0f} Hz on alternate ticks)"
                if fast
                else (
                    f", impedance joints on alternate ticks at {IMPEDANCE_LOOP_HZ:.0f} Hz)"
                    if mixed
                    else ")"
                )
            )
        )
    )

    # The kinematics stack plans the collision-aware approach/return moves.
    print("Loading kinematics solver (JIT compile may take a few seconds) ...")
    from ...kinematics.solver import KinematicsSolver
    from ...teleop.config import VRTeleopConfig
    from ...teleop.trajectory import plan_collision_aware_trajectory

    solver = KinematicsSolver()
    rest_cfg = VRTeleopConfig()
    q_rest = np.zeros(solver.num_joints, dtype=np.float32)
    q_rest[solver.left_indices] = rest_cfg.rest_pose_left
    q_rest[solver.right_indices] = rest_cfg.rest_pose_right

    def to_full(row: np.ndarray) -> np.ndarray:
        q = q_rest.copy()
        q[solver.left_indices] = row[:7]
        q[solver.right_indices] = row[7:]
        return q

    def plan(q_from: np.ndarray, q_to: np.ndarray) -> list[np.ndarray]:
        return plan_collision_aware_trajectory(
            solver,
            q_from,
            q_to,
            speed=_PLAN_SPEED,
            rate=motion.rate,
            min_duration=_PLAN_MIN_DURATION,
        )

    def snapshot(axol: Axol) -> np.ndarray:
        q = q_rest.copy()
        if axol.left is not None:
            q[solver.left_indices] = axol.left.positions[:7]
        if axol.right is not None:
            q[solver.right_indices] = axol.right.positions[:7]
        return q

    if args.ik:
        print(f"Re-solving {len(sent)} waypoints through the IK solver ...")
        sent = _ik_stream(solver, sent, to_full, stream_info)
        stream_differs = True
    if holds:
        # After any IK re-solve, so the solver cannot move a held joint back.
        # The scoring reference is held the same way: the joint is scored as
        # parked (buzz / chatter), not against the motion it no longer runs.
        first = np.asarray(ref[0], dtype=float)
        sent = _apply_holds(sent, holds, np.asarray(sent[0], dtype=float))
        ref = _apply_holds(ref, holds, first)
        for col, angle in holds.items():
            at = (
                f"{math.degrees(angle):+.1f}°"
                if angle is not None
                else f"{math.degrees(first[col]):+.1f}° (the motion's start)"
            )
            print(f"  hold: {_COLUMNS[col]} at {at}")
        print(
            "  ! held joints: only the approach is collision-checked — a frozen "
            "joint can bring links closer than the recording did; watch the first pass"
        )

    if args.invert:
        sent = _invert_stream(sent, ref, holds, overrides, motion.rate, args.arms)
        stream_differs = True
    if args.notch:
        from ...teleop.filter import NotchFilter

        notch = NotchFilter(args.notch, args.notch_q, motion.rate)
        rows = np.asarray(sent, dtype=float)
        notch.reset(rows[0])
        sent = np.stack([notch.update(r) for r in rows]).astype(float)
        for col in holds:
            sent[:, col] = rows[:, col]
        stream_differs = True
        print(
            "  notch: "
            + ", ".join(f"{f:g} Hz" for f in args.notch)
            + f" (Q {args.notch_q:g}) on the streamed motion"
        )
    n_wp = len(sent)
    ref_arr = np.asarray(ref, dtype=float)
    learner: CommandLearner | None = None
    fixed_offset: np.ndarray | None = None
    if args.learn:
        columns = _learn_columns(args, ref_arr, holds)
        if not columns.any():
            raise SystemExit("tune.motion: --learn has no joint to learn")
        learner = CommandLearner(
            n_wp,
            motion.rate,
            columns,
            band=(float(args.learn_band[0]), float(args.learn_band[1])),
            gain=args.learn_gain,
            max_rad=math.radians(args.learn_max_deg),
            # The IMU's tool error is noisier pass to pass than the encoders'.
            worse_ratio=1.5 if args.learn_imu else 1.15,
            min_gain=0.2 if args.learn_imu else 0.0,
        )
        print(
            f"  learning: {args.learn} passes on "
            + ", ".join(_COLUMNS[i] for i in np.where(columns)[0])
            + f" ({args.learn_band[0]:g}-{args.learn_band[1]:g} Hz, gain "
            f"{args.learn_gain:g}, clamp {args.learn_max_deg:g}°)"
        )
    elif args.correction:
        fixed_offset = _load_correction(args.correction, n_wp)
        for col in holds:
            fixed_offset[:, col] = 0.0
        print(
            f"  correction: {args.correction} (peak "
            f"{math.degrees(float(np.abs(fixed_offset).max())):.3f}°)"
        )
    # The offset each pass flew, alongside passes_run.
    pass_offsets: list[np.ndarray | None] = []
    dampers = _imu_dampers(args) | _gyro_dampers(args)
    # Per pass: whether IMU damping flew it; per sample: the damper's view.
    pass_damped: list[bool] = []
    pass_damp_rows: list[tuple[int, int]] = []
    log_damp: list[np.ndarray] = []

    watchdog = ContactWatchdog(args.torque_threshold)
    # Firmware-loop joints, as (side, index in the arm's 7, name), for the
    # deviation guard in execute().
    resolved_cfg = config.resolved()
    fw_joints = [
        (side, i, f"{side}.{j.value}")
        for side in ("left", "right")
        for i, j in enumerate(ARM_JOINTS)
        if str(getattr(getattr(resolved_cfg, side), j.value).wire_mode).lower()
        in ("a4", "pv")
    ]
    log_t: list[float] = []
    # The same samples on the absolute perf_counter clock: log_t restarts at
    # 0 with every execute(), the wrist IMU record does not.
    log_abs: list[float] = []
    log_target: list[np.ndarray] = []
    log_sent: list[np.ndarray] = []
    log_actual: list[np.ndarray] = []
    log_torque: list[np.ndarray] = []
    # When each measured sample was actually taken on the wire (seconds
    # relative to the sample's own log time, ≤ 0): the core refreshes the
    # caches at its tick rate, this loop reads them at the motion rate, and
    # the varying cache age between the two clocks is a sawtooth that a
    # 400 Hz core sampled at 240 Hz turns into an 80 Hz "buzz" on every
    # joint. Re-timing each sample removes it.
    log_meas_offset: list[np.ndarray] = []
    # [start, end) of each pass's samples in the logs above (--repeat).
    passes_run: list[tuple[int, int]] = []

    def _feedback_offsets(arm: Any, now_wall: float) -> np.ndarray:
        out = np.zeros(7, dtype=np.float64)
        for i, j in enumerate(ARM_JOINTS):
            motor = arm.motors.get(j)
            ts = getattr(motor, "_feedback_ts", None) if motor is not None else None
            if ts is not None:
                out[i] = min(0.0, ts - now_wall)
        return out

    async def execute(
        axol: Axol,
        waypoints: list[np.ndarray] | np.ndarray,
        record: bool = False,
        refs: np.ndarray | None = None,
        damp: bool = False,
    ) -> tuple[str, float] | None:
        """Stream full-N waypoints at the motion rate with deadline pacing.

        Absolute deadlines: a late wakeup is corrected on the next cycle
        instead of stretching the command interval — interval jitter would
        otherwise land in motion_control's differentiated feedforward as
        torque jitter. ``refs`` (``(N, 14)``, optional) is the clean
        reference logged as the scoring target when the streamed waypoints
        are a corrupted/filtered version of it; the streamed rows are then
        logged separately as ``sent``. Returns the watchdog trip or ``None``.
        """
        period = 1.0 / motion.rate
        left = np.zeros(8, dtype=np.float32)
        right = np.zeros(8, dtype=np.float32)
        t0 = time.perf_counter()
        deadline = t0
        active = {
            side: d
            for side, d in dampers.items()
            if damp and (side in imu.sides or isinstance(d, EncoderTipDamper))
        }
        for side, d in active.items():
            imu.poll(side)  # drop what queued between passes
            d.start(t0)
        jac: dict[str, np.ndarray] = {}
        for k, q in enumerate(waypoints):
            deadline += period
            left[:7] = q[solver.left_indices]
            right[:7] = q[solver.right_indices]
            for side, d in active.items():
                arm = axol.left if side == "left" else axol.right
                if arm is None:
                    continue
                q_meas = snapshot(axol)
                if isinstance(d, (GyroFlexDamper, TorqueProbe)):
                    if k % _JAC_EVERY == 0 or side not in jac:
                        rot, jac[side] = _joint_axes(solver, q_meas, side, d.columns)
                    else:
                        rot = _ee_rotation(solver, q_meas, side)
                    now = time.perf_counter()
                    d.feed(imu.poll(side))
                    d.feed_pose(now, rot, jac[side])
                    tau = d.torque(now)
                else:
                    if k % _JAC_EVERY == 0 or side not in jac:
                        height, jac[side] = _height_jacobian(
                            solver, q_meas, side, d.columns
                        )
                    else:
                        height = _height(solver, q_meas, side)
                    now = time.perf_counter()
                    if isinstance(d, EncoderTipDamper):
                        d.feed_command(now, _height(solver, np.asarray(q), side))
                    elif args.imu_damp_ref == "command":
                        # The tool height this waypoint commands: the
                        # damper then sees the whole deviation from the path.
                        height = _height(solver, np.asarray(q), side)
                    d.feed_height(now, height)
                    d.feed(imu.poll(side))
                    tau = d.torque(now, jac[side])
                arm.extra_torque = tau
                if record:
                    log_damp.append(np.concatenate([[d.flex, float(d.tripped)], tau]))
            await axol.motion_control(
                left=left if axol.left is not None else None,
                right=right if axol.right is not None else None,
            )
            for side, i, name in fw_joints:
                arm = axol.left if side == "left" else axol.right
                if arm is None:
                    continue
                cmd = float((left if side == "left" else right)[i])
                off = float(arm.positions[i]) - cmd
                if abs(off) > _FW_DEVIATION_ABORT:
                    raise _Runaway(name, math.degrees(off))
            if record:
                row_a = np.full(14, np.nan, dtype=np.float32)
                row_tq = np.full(14, np.nan, dtype=np.float32)
                row_off = np.zeros(14, dtype=np.float64)
                now_wall = time.time()
                if axol.left is not None:
                    row_a[:7] = axol.left.positions[:7]
                    row_tq[:7] = axol.left.torques[:7]
                    row_off[:7] = _feedback_offsets(axol.left, now_wall)
                if axol.right is not None:
                    row_a[7:] = axol.right.positions[:7]
                    row_tq[7:] = axol.right.torques[:7]
                    row_off[7:] = _feedback_offsets(axol.right, now_wall)
                now = time.perf_counter()
                log_t.append(now - t0)
                log_abs.append(now)
                log_meas_offset.append(row_off)
                row_cmd = np.concatenate(
                    [q[solver.left_indices], q[solver.right_indices]]
                ).astype(np.float32)
                if refs is not None:
                    log_target.append(refs[k].astype(np.float32))
                    log_sent.append(row_cmd)
                else:
                    log_target.append(row_cmd)
                log_actual.append(row_a)
                log_torque.append(row_tq)
            tripped = watchdog.update(
                (
                    axol.left.torque_residuals() if axol.left is not None else None,
                    axol.right.torque_residuals() if axol.right is not None else None,
                )
            )
            if tripped is not None:
                _clear_extra_torque(axol)
                return tripped
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        _clear_extra_torque(axol)
        for side, d in active.items():
            if d.tripped:
                if isinstance(d, GyroFlexDamper):
                    print(
                        f"  ! gyro damping ({side}) switched itself off: the flex "
                        f"rate passed {math.degrees(d.trip_rate):.1f}°/s for "
                        f"{d.trip_s:g} s — lower --gyro-damp"
                    )
                    continue
                print(
                    f"  ! IMU damping ({side}) switched itself off: the flex "
                    f"velocity passed {d.trip_speed * 1e3:.0f} mm/s for "
                    f"{d.trip_s:g} s — lower --imu-damp"
                )
        return None

    print("Planning approach and return trajectories ...")
    q_start = to_full(sent[0])
    traj_playback = [to_full(row) for row in sent]

    # Production playback always runs through the Rust core, matching teleop.
    arm_channels: dict[str, None] = {}
    if args.arms == "right":
        arm_channels["left_channel"] = None
    elif args.arms == "left":
        arm_channels["right_channel"] = None
    robot = Axol(
        config=config, record=args.record, loop_hz=args.loop_hz, **arm_channels
    )

    # The wrist cameras' IMUs see what the encoders cannot (backlash, flex,
    # the gripper itself). Opened before bring-up, so a camera that is slow
    # to open costs time while nothing moves; stopped after return to rest.
    imu = WristImu(
        ["left", "right"] if args.arms == "both" else [args.arms],
        enabled=not args.no_imu,
        live=(args.imu_damp > 0 and args.imu_damp_source == "imu")
        or args.gyro_damp > 0
        or bool(args.torque_probe),
    )
    imu.start()
    needs_imu = (
        args.learn_imu
        or (args.imu_damp > 0 and args.imu_damp_source == "imu")
        or args.gyro_damp > 0
        or bool(args.torque_probe)
    )
    if needs_imu and not imu.sides:
        imu.stop()
        raise SystemExit(
            "tune.motion: the wrist IMU did not start (camera did not open) and "
            + (
                "--learn-imu"
                if args.learn_imu
                else "--gyro-damp"
                if args.gyro_damp > 0
                else "--imu-damp"
            )
            + " needs it — nothing moved. If the ZED stack is wedged, restart "
            "it (sudo systemctl restart zed_x_daemon) and try again."
        )

    async with robot as axol:
        contact: tuple[str, float] | None = None
        try:
            q_now = snapshot(axol)
            if float(np.max(np.abs(q_now - q_start))) > 0.02:
                print("Moving to the motion start pose ...")
                contact = await execute(axol, plan(q_now, q_start))
                if contact is not None:
                    raise _Contact(contact)
                await asyncio.sleep(0.5)
            driven = [
                (side, idx)
                for side, arm, idx in (
                    ("left", axol.left, solver.left_indices),
                    ("right", axol.right, solver.right_indices),
                )
                if arm is not None
            ]
            stragglers = start_pose_stragglers(snapshot(axol), q_start, driven)
            if stragglers:
                raise _NotAtStart(stragglers)

            # The flight recorder captures the replay segment only, like
            # teleop's engage→disengage — one segment for the whole session
            # when repeating (each new segment truncates the last), so a
            # buzz on the move back to the start is in the trace too.
            axol.set_recording_engaged(True)
            try:
                passes = itertools.count() if args.repeat == 0 else range(args.repeat)
                total = "∞" if args.repeat == 0 else str(args.repeat)
                for k in passes:
                    if k > 0:
                        q_now = snapshot(axol)
                        if float(np.max(np.abs(q_now - q_start))) > 0.02:
                            print("Back to the motion start pose ...")
                            contact = await execute(axol, plan(q_now, q_start))
                            if contact is not None:
                                raise _Contact(contact)
                        stragglers = start_pose_stragglers(
                            snapshot(axol), q_start, driven
                        )
                        if stragglers:
                            raise _NotAtStart(stragglers)
                    print(
                        f"Replaying {motion.duration:.1f} s of motion"
                        + (f" (pass {k + 1}/{total})" if args.repeat != 1 else "")
                        + " ..."
                    )
                    pass_start = len(log_t)
                    passes_run.append((pass_start, pass_start))
                    offset = (
                        learner.offset.copy() if learner is not None else fixed_offset
                    )
                    pass_offsets.append(offset)
                    damp_this = bool(dampers) and (
                        not args.imu_damp_alternate or k % 2 == 1
                    )
                    pass_damped.append(damp_this)
                    if dampers:
                        print(f"  IMU damping {'ON' if damp_this else 'off'} this pass")
                    damp_start = len(log_damp)
                    playback = traj_playback
                    if offset is not None and np.any(offset):
                        playback = [
                            to_full(np.asarray(row, dtype=float) + offset[i])
                            for i, row in enumerate(sent)
                        ]
                    try:
                        contact = await execute(
                            axol,
                            playback,
                            record=True,
                            refs=ref if stream_differs or offset is not None else None,
                            damp=damp_this,
                        )
                    finally:
                        passes_run[-1] = (pass_start, len(log_t))
                        pass_damp_rows.append((damp_start, len(log_damp)))
                    if contact is not None:
                        raise _Contact(contact)
                    if learner is not None and k + 1 < args.learn and args.learn_imu:
                        _learn_step_imu(
                            learner,
                            k,
                            imu,
                            args.arms,
                            np.asarray(log_t[pass_start:]),
                            np.asarray(log_abs[pass_start:]),
                            ref_arr,
                            to_full,
                            solver,
                        )
                    elif learner is not None and k + 1 < args.learn:
                        _learn_step(
                            learner,
                            k,
                            np.asarray(log_t[pass_start:]),
                            np.stack(log_meas_offset[pass_start:]),
                            np.stack(log_actual[pass_start:]),
                            np.stack(log_torque[pass_start:]),
                            ref_arr,
                        )
            finally:
                axol.set_recording_engaged(False)
        except _NotAtStart as exc:
            print(
                "\n  ! not at the motion start pose after the approach — playback "
                "skipped: "
                + ", ".join(f"{name} {err:+.1f}° off" for name, err in exc.stragglers)
            )
            if args.a4:
                print(
                    "    a --a4 joint that did not follow the approach: check its stored "
                    "planner acceleration (scripts/fw_gains.py --id <id>; 0 or 60000 "
                    "follow a stream, anything in between barely moves) and that the "
                    "realtime core is built from a checkout that holds a4 joints on the "
                    "position frame (the X6-P20's 2025-07 firmware ignores 0xA4 after "
                    "an MIT frame until reset)"
                )
        except _Runaway as exc:
            print(
                f"\n  ! {exc.joint} is {exc.deg:+.1f}° from its command on the "
                f"firmware loop (limit {math.degrees(_FW_DEVIATION_ABORT):.0f}°) — it "
                "has left its target; playback aborted, returning to rest"
            )
        except _Contact as exc:
            joint, residual = exc.trip
            print(
                f"\n  ! contact: {joint} torque residual {residual:.1f} Nm "
                f"exceeded {args.torque_threshold:.1f} — playback aborted"
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  interrupted — returning to rest before disabling ...")
        finally:
            current = asyncio.current_task()
            if current is not None:
                while current.cancelling():
                    current.uncancel()
            # Always finish at rest, re-planned from wherever we stopped.
            # On a contact trip the watchdog already said we're pushing on
            # something — return slowly and let the (still active) operator
            # Ctrl-C if the environment needs clearing first.
            try:
                q_now = snapshot(axol)
                if float(np.max(np.abs(q_now - q_rest))) > 0.02:
                    print("Returning to rest ...")
                    await execute(axol, plan(q_now, q_rest))
            except Exception:  # noqa: BLE001 - best-effort teardown
                logging.getLogger(__name__).warning(
                    "return-to-rest failed", exc_info=True
                )

    # A daemon subprocess: an exception out of the block above ends it with
    # the process; here it hands over its samples.
    imu.stop()
    if not log_t:
        print("No playback samples recorded — nothing to score.")
        return
    if not passes_run:
        passes_run.append((0, len(log_t)))

    # Tracking quality is only scored for joints that actually moved (> ~1°
    # of commanded travel) — a joint parked at rest tracks meaninglessly
    # well. But a parked joint can still buzz or chatter at hold (that's how
    # the wrist limit cycle presents), so stationary joints keep their row
    # with only the rest-meaningful columns; the tracking numbers go NaN and
    # render as "–".
    _TRACKING_KEYS = (
        "rms_err",
        "rms_err_lagfree",
        "lag_ms",
        "err_band_mid",
        "peak_hz",
        "amplification",
    )

    def score_pass(
        a: int, b: int, tag: str, pass_index: int = 0
    ) -> dict[str, Any] | None:
        """Score and save one pass's slice of the logs; its summary, or None.

        ``tag`` ("[k/N] ", empty for a single pass) heads the scorecard and is
        appended to the saved run's label.
        """
        if b - a < 2:
            return None
        t = np.asarray(log_t[a:b])
        target = np.stack(log_target[a:b])
        actual = np.stack(log_actual[a:b])
        torque = np.stack(log_torque[a:b])
        actual, torque = retime_measurements(
            t, np.stack(log_meas_offset[a:b]), actual, torque
        )
        per_joint: dict[str, dict[str, float]] = {}
        moved: dict[str, dict[str, float]] = {}
        for i, name in enumerate(_COLUMNS):
            if np.isnan(actual[:, i]).all():
                continue
            m = tracking_metrics(t, target[:, i], actual[:, i], torque[:, i])
            if float(np.ptp(target[:, i])) >= math.radians(1.0):
                moved[name] = m
            else:
                for key in _TRACKING_KEYS:
                    m[key] = math.nan
            per_joint[name] = m
        if tag:
            print(f"\n{tag.strip()}")
        _print_metrics_table(per_joint)

        summary: dict[str, Any] = {
            "per_joint": per_joint,
            "completed": bool(b - a >= len(sent)),
        }
        if pass_index < len(pass_damped):
            summary["imu_damp"] = args.imu_damp if pass_damped[pass_index] else 0.0
            summary["gyro_damp"] = args.gyro_damp if pass_damped[pass_index] else 0.0
        if moved:
            worst = max(moved.items(), key=lambda kv: kv[1]["rms_err"])
            summary["worst_joint"] = worst[0]
            summary["mean_rms_err"] = float(
                np.mean([m["rms_err"] for m in moved.values()])
            )
            summary["mean_jitter"] = float(
                np.mean([m["err_band_mid"] for m in moved.values()])
            )
        else:
            # A hold (the ``hold`` motion): no tracking to score, but the
            # buzz columns and the wrist IMU's floor under the running
            # controller are the point of it.
            print(f"{tag}No joint moved more than 1° — hold: buzz and IMU only.")
        # The pass on the wrist IMUs' clock: its log origin is the execute()
        # start, so the IMU series shares the run's time axis.
        origin = log_abs[a] - log_t[a]
        imu_metrics, imu_series = imu.run_blocks(
            origin + float(t[0]), origin + float(t[-1]), origin
        )
        if imu_metrics:
            summary["imu"] = imu_metrics
            for line in format_imu(imu_metrics):
                print(line)

        if not args.no_save_run:
            series = {"t": t, "target": target, "actual": actual, "torque": torque}
            series.update(imu_series)
            if log_sent:
                series["sent"] = np.stack(log_sent[a:b])
            if pass_index < len(pass_damp_rows):
                d0, d1 = pass_damp_rows[pass_index]
                if d1 > d0:
                    # Per sample: band vertical velocity (m/s), tripped, τ (7).
                    series["imu_damp"] = np.stack(log_damp[d0:d1]).astype(np.float32)
            if pass_index < len(pass_offsets) and pass_offsets[pass_index] is not None:
                series["correction"] = pass_offsets[pass_index].astype(np.float32)
            label = " ".join(x for x in (args.label, tag.strip()) if x) or None
            run_id = save_run(
                "motion",
                series,
                summary,
                gains={f"{s}.{j}.{f}": v for (s, j, f), v in overrides.items()},
                params={
                    "motion": motion.name,
                    "rate": motion.rate,
                    "stiffness": args.stiffness,
                    "columns": _COLUMNS,
                    # Joints driven on the firmware position loop (--a4) for
                    # this run, so the dashboard can re-arm the same split.
                    "a4": list(args.a4),
                    # Joints held steady instead of following the motion.
                    "hold": list(args.hold),
                    "arms": args.arms,
                    # The control law the whole run ran on (impedance at
                    # 240 Hz or the firmware position loops at 400 Hz).
                    "controller": config.controller,
                    "loop_hz": args.loop_hz or config.loop_hz,
                    "impedance_hz": config.impedance_hz,
                    # Joints commanded at 480 Hz (--fast-impedance / the
                    # config-wide 480).
                    "fast_impedance": fast_impedance_joints(config),
                    "record": args.record,
                    **stream_info,
                },
                label=label,
            )
            print(f"\nSaved tuning run {run_id} (kind=motion, motion={motion.name!r})")
        return summary

    many = len(passes_run) > 1
    summaries = [
        score_pass(a, b, f"[{k + 1}/{len(passes_run)}] " if many else "", k)
        for k, (a, b) in enumerate(passes_run)
    ]
    if many:
        # One line per pass: the intermittent faults (a buzz on one pass in
        # five) are what repeating is for.
        print(f"\n{'─' * 78}\n  passes: worst buzz / mean jitter / worst joint")
        for k, sm in enumerate(summaries):
            if sm is None:
                print(f"    [{k + 1}] too short to score")
                continue
            name, m = max(
                sm["per_joint"].items(),
                key=lambda kv: (
                    kv[1]["buzz"] if math.isfinite(kv[1].get("buzz", math.nan)) else 0.0
                ),
            )
            print(
                f"    [{k + 1}] {math.degrees(m.get('buzz', math.nan)):.3f}° on {name} "
                f"@ {m.get('buzz_hz', math.nan):.0f} Hz / "
                f"{math.degrees(sm.get('mean_jitter', math.nan)):.3f}° / "
                f"{sm.get('worst_joint', 'hold')}"
                + ("" if sm["completed"] else "  (cut short)")
            )


class _Runaway(Exception):
    """Internal: a firmware-loop joint left its target (``_FW_DEVIATION_ABORT``)."""

    def __init__(self, joint: str, deg: float) -> None:
        self.joint = joint
        self.deg = deg


class _Contact(Exception):
    """Internal: unwind playback on a contact-watchdog trip."""

    def __init__(self, trip: tuple[str, float]) -> None:
        self.trip = trip


class _NotAtStart(Exception):
    """Internal: the approach left joints off the start pose; skip playback."""

    def __init__(self, stragglers: list[tuple[str, float]]) -> None:
        self.stragglers = stragglers


def _load_motion_or_exit(name: str) -> ReferenceMotion:
    try:
        return load_motion(name)
    except FileNotFoundError as exc:
        known = ", ".join(m.name for m in list_motions()) or "(none committed)"
        raise SystemExit(f"{exc}\nKnown motions: {known}")
