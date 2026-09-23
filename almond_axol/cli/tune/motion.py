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
from typing import Any

import numpy as np

from ...constants import ARM_JOINTS
from ...robot import Axol
from ...robot.config import (
    CONTROLLERS,
    IMPEDANCE_LOOP_HZ,
    AxolConfig,
    check_loop_hz,
)
from ...robot.control import ContactWatchdog
from ...tuning import save_run, tracking_metrics
from ...tuning.motion import ReferenceMotion, list_motions, load_motion
from ...utils.logquiet import quiet_noisy_loggers

_PLAN_SPEED = 0.1 * np.pi  # rad/s — approach/return trajectory speed
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
)

# Column names of a 14-wide motion row: left arm then right arm.
_COLUMNS = [f"left.{j.value}" for j in ARM_JOINTS] + [
    f"right.{j.value}" for j in ARM_JOINTS
]


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
        if len(parts) >= 2 and parts[-2] in ("friction", "firmware"):
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
        for side in sides:
            out[(side, joint, fld)] = value
    return out


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


async def _run(args: argparse.Namespace) -> None:
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
    for (side, joint, fld), value in overrides.items():
        target = getattr(getattr(config, side), joint)
        if fld.startswith("friction."):
            setattr(target.friction, fld.split(".", 1)[1], value)
        elif fld.startswith("firmware."):
            setattr(target.firmware, fld.split(".", 1)[1], value)
        else:
            setattr(target, fld, value)
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
    if args.controller is not None:
        config.controller = args.controller
    if args.repeat < 0:
        raise SystemExit("tune.motion: --repeat must be 0 (until Ctrl-C) or more")
    try:
        # Before anything touches the bus: impedance runs at 240 Hz only.
        check_loop_hz(config, args.loop_hz or config.loop_hz)
    except ValueError as exc:
        raise SystemExit(f"tune.motion: {exc}") from None
    core_hz = args.loop_hz or config.loop_hz
    mixed = config.controller != "position" and core_hz > IMPEDANCE_LOOP_HZ
    print(
        f"  controller: {config.controller} "
        f"({core_hz:.0f} Hz core loop"
        + (
            ", every joint on its firmware position loop)"
            if config.controller == "position"
            else (
                f", impedance joints on alternate ticks at {IMPEDANCE_LOOP_HZ:.0f} Hz)"
                if mixed
                else ")"
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

    watchdog = ContactWatchdog(args.torque_threshold)
    log_t: list[float] = []
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
        for k, q in enumerate(waypoints):
            deadline += period
            left[:7] = q[solver.left_indices]
            right[:7] = q[solver.right_indices]
            await axol.motion_control(
                left=left if axol.left is not None else None,
                right=right if axol.right is not None else None,
            )
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
                log_t.append(time.perf_counter() - t0)
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
                return tripped
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
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
                    try:
                        contact = await execute(
                            axol,
                            traj_playback,
                            record=True,
                            refs=ref if stream_differs else None,
                        )
                    finally:
                        passes_run[-1] = (pass_start, len(log_t))
                    if contact is not None:
                        raise _Contact(contact)
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

    def score_pass(a: int, b: int, tag: str) -> dict[str, Any] | None:
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
        if not moved:
            print(f"{tag}No joint moved more than 1° — nothing to score.")
            return None

        if tag:
            print(f"\n{tag.strip()}")
        _print_metrics_table(per_joint)

        worst = max(moved.items(), key=lambda kv: kv[1]["rms_err"])
        summary = {
            "per_joint": per_joint,
            "worst_joint": worst[0],
            "mean_rms_err": float(np.mean([m["rms_err"] for m in moved.values()])),
            "mean_jitter": float(np.mean([m["err_band_mid"] for m in moved.values()])),
            "completed": bool(b - a >= len(sent)),
        }

        if not args.no_save_run:
            series = {"t": t, "target": target, "actual": actual, "torque": torque}
            if log_sent:
                series["sent"] = np.stack(log_sent[a:b])
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
                    # The control law the whole run ran on (impedance at
                    # 240 Hz or the firmware position loops at 400 Hz).
                    "controller": config.controller,
                    "loop_hz": args.loop_hz or config.loop_hz,
                    "record": args.record,
                    **stream_info,
                },
                label=label,
            )
            print(f"\nSaved tuning run {run_id} (kind=motion, motion={motion.name!r})")
        return summary

    many = len(passes_run) > 1
    summaries = [
        score_pass(a, b, f"[{k + 1}/{len(passes_run)}] " if many else "")
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
                f"{math.degrees(sm['mean_jitter']):.3f}° / {sm['worst_joint']}"
                + ("" if sm["completed"] else "  (cut short)")
            )


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
