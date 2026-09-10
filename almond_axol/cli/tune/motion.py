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
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

import numpy as np

from ...constants import ARM_JOINTS
from ...robot import Axol
from ...robot.config import AxolConfig
from ...robot.control import ContactWatchdog
from ...tuning import save_run, tracking_metrics
from ...tuning.motion import ReferenceMotion, list_motions, load_motion

if TYPE_CHECKING:
    from ...rt import RtAxol

_PLAN_SPEED = 0.1 * np.pi  # rad/s — approach/return trajectory speed
_PLAN_MIN_DURATION = 1.5  # s

_GAIN_FIELDS = (
    "kp",
    "kd",
    "kd_host",
    "kd_host_hz",
    "kd_host_q",
    "j_eff",
)

# Column names of a 14-wide motion row: left arm then right arm.
_COLUMNS = [f"left.{j.value}" for j in ARM_JOINTS] + [
    f"right.{j.value}" for j in ARM_JOINTS
]


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
        setattr(getattr(getattr(config, side), joint), fld, value)
        print(f"  gain override: {side}.{joint}.{fld} = {value}")

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
                if axol.left is not None:
                    row_a[:7] = axol.left.positions[:7]
                    row_tq[:7] = axol.left.torques[:7]
                if axol.right is not None:
                    row_a[7:] = axol.right.positions[:7]
                    row_tq[7:] = axol.right.torques[:7]
                log_t.append(time.perf_counter() - t0)
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
    from ...rt import RtAxol as _RtAxol

    robot: RtAxol = _RtAxol(Axol(config=config))

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

            print(f"Replaying {motion.duration:.1f} s of motion ...")
            contact = await execute(
                axol,
                traj_playback,
                record=True,
                refs=ref if stream_differs else None,
            )
            if contact is not None:
                raise _Contact(contact)
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

    t = np.asarray(log_t)
    target = np.stack(log_target)
    actual = np.stack(log_actual)
    torque = np.stack(log_torque)

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
        print("No joint moved more than 1° — nothing to score.")
        return

    _print_metrics_table(per_joint)

    worst = max(moved.items(), key=lambda kv: kv[1]["rms_err"])
    summary = {
        "per_joint": per_joint,
        "worst_joint": worst[0],
        "mean_rms_err": float(np.mean([m["rms_err"] for m in moved.values()])),
        "mean_jitter": float(np.mean([m["err_band_mid"] for m in moved.values()])),
        "completed": bool(len(log_t) >= len(sent)),
    }

    if not args.no_save_run:
        series = {"t": t, "target": target, "actual": actual, "torque": torque}
        if log_sent:
            series["sent"] = np.stack(log_sent)
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
                **stream_info,
            },
            label=args.label,
        )
        print(f"\nSaved tuning run {run_id} (kind=motion, motion={motion.name!r})")


class _Contact(Exception):
    """Internal: unwind playback on a contact-watchdog trip."""

    def __init__(self, trip: tuple[str, float]) -> None:
        self.trip = trip


def _load_motion_or_exit(name: str) -> ReferenceMotion:
    try:
        return load_motion(name)
    except FileNotFoundError as exc:
        known = ", ".join(m.name for m in list_motions()) or "(none committed)"
        raise SystemExit(f"{exc}\nKnown motions: {known}")
