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
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time

import numpy as np

from ...constants import ARM_JOINTS
from ...robot import Axol
from ...robot.config import AxolConfig
from ...robot.control import ContactWatchdog
from ...tuning import save_run, tracking_metrics
from ...tuning.motion import ReferenceMotion, list_motions, load_motion

_PLAN_SPEED = 0.1 * np.pi  # rad/s — approach/return trajectory speed
_PLAN_MIN_DURATION = 1.5  # s

_GAIN_FIELDS = ("kp", "kd", "kd_host", "kd_host_max", "kd_host_hz", "j_eff")

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
        "s=0.5 midpoint anchors, like the config defaults they replace.",
    )
    p.add_argument(
        "--stiffness",
        type=float,
        default=0.5,
        help="Stiffness-slider position in [0, 1] applied to both arms "
        "(default: 0.5, the teleop default — gain overrides land exactly "
        "at that midpoint)",
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
    """Tracking/smoothness scorecard, one row per joint that actually moved."""
    print(f"\n{'═' * 78}")
    print(
        f"  {'joint':<18} {'RMS °':>7} {'lagfree °':>9} {'lag ms':>7} "
        f"{'jitter °':>8} {'amp':>5} {'trq HF':>7}"
    )
    for name, m in per_joint.items():
        amp = f"{m['amplification']:.2f}" if math.isfinite(m["amplification"]) else "-"
        lag = f"{m['lag_ms']:.0f}" if math.isfinite(m["lag_ms"]) else "-"
        print(
            f"  {name:<18} {math.degrees(m['rms_err']):>7.3f} "
            f"{math.degrees(m['rms_err_lagfree']):>9.3f} {lag:>7} "
            f"{math.degrees(m['err_band_mid']):>8.3f} {amp:>5} "
            f"{m['torque_hf']:>7.3f}"
        )
    print(f"{'═' * 78}")
    print(
        "  RMS = tracking error vs the reference; lagfree = after removing\n"
        "  the measured command->measurement delay; jitter = 3-15 Hz band of\n"
        "  the error (what the operator feels); amp = measured/commanded\n"
        "  mid-band motion (>1 rings, <1 filters); trq HF = torque chatter (Nm)."
    )


async def _run(args: argparse.Namespace) -> None:
    motion = _load_motion_or_exit(args.motion)
    overrides = _parse_gain_overrides(args.gain or [])

    print(
        f"Reference motion {motion.name!r}: {len(motion.q)} waypoints, "
        f"{motion.duration:.1f} s at {motion.rate:.0f} Hz"
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

    watchdog = ContactWatchdog(args.torque_threshold)
    log_t: list[float] = []
    log_target: list[np.ndarray] = []
    log_actual: list[np.ndarray] = []
    log_torque: list[np.ndarray] = []

    async def execute(
        axol: Axol, waypoints: list[np.ndarray] | np.ndarray, record: bool = False
    ) -> tuple[str, float] | None:
        """Stream full-N waypoints at the motion rate with deadline pacing.

        Absolute deadlines: a late wakeup is corrected on the next cycle
        instead of stretching the command interval — interval jitter would
        otherwise land in motion_control's differentiated feedforward as
        torque jitter. Returns the watchdog trip, or ``None``.
        """
        period = 1.0 / motion.rate
        left = np.zeros(8, dtype=np.float32)
        right = np.zeros(8, dtype=np.float32)
        t0 = time.perf_counter()
        deadline = t0
        for q in waypoints:
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
                log_target.append(
                    np.concatenate(
                        [q[solver.left_indices], q[solver.right_indices]]
                    ).astype(np.float32)
                )
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
    q_start = to_full(motion.q[0])
    traj_playback = [to_full(row) for row in motion.q]

    async with Axol(config=config) as axol:
        await axol.start_telemetry(500)
        await axol.wait_for_telemetry()

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
            contact = await execute(axol, traj_playback, record=True)
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

    # Score only joints that actually moved (> ~1° of commanded travel) —
    # a joint parked at rest scores meaninglessly well.
    per_joint: dict[str, dict[str, float]] = {}
    for i, name in enumerate(_COLUMNS):
        if np.isnan(actual[:, i]).all():
            continue
        travel = float(np.ptp(target[:, i]))
        if travel < math.radians(1.0):
            continue
        per_joint[name] = tracking_metrics(t, target[:, i], actual[:, i], torque[:, i])
    if not per_joint:
        print("No joint moved more than 1° — nothing to score.")
        return

    _print_metrics_table(per_joint)

    worst = max(per_joint.items(), key=lambda kv: kv[1]["rms_err"])
    summary = {
        "per_joint": per_joint,
        "worst_joint": worst[0],
        "mean_rms_err": float(np.mean([m["rms_err"] for m in per_joint.values()])),
        "mean_jitter": float(np.mean([m["err_band_mid"] for m in per_joint.values()])),
        "completed": bool(len(log_t) >= len(motion.q)),
    }

    if not args.no_save_run:
        run_id = save_run(
            "motion",
            {"t": t, "target": target, "actual": actual, "torque": torque},
            summary,
            gains={f"{s}.{j}.{f}": v for (s, j, f), v in overrides.items()},
            params={
                "motion": motion.name,
                "rate": motion.rate,
                "stiffness": args.stiffness,
                "columns": _COLUMNS,
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
