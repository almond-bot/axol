"""
axol tune.repeatability

Repeatability test for the Axol **left arm**: move rest → A once, then
bounce between two hard-coded poses (A → B → A → B …) for the requested
number of cycles, returning to rest only at the end (or on Ctrl-C). Each
leg is planned in joint space with pyroki + the URDF so the body never
clips the torso.

Poses A and B live in :data:`_POSE_A_LEFT` / :data:`_POSE_B_LEFT` below, in
:data:`ARM_JOINTS` order (shoulder_1 … wrist_3). They are the *raw motor-frame*
angles printed by ``axol motor.info`` (motor ids 0x01…0x07) — pose the arm by
hand, run ``axol motor.info --l``, and paste the numbers in. They are
converted to joint frame internally via :func:`closer_end_stop`.

The motion is a pure joint-space interpolation between the poses — no
per-waypoint IK — so the playback is smooth and perfectly repeatable. The
right arm is left untouched throughout; only the left arm actuates. The arm
runs at ``--stiffness`` (default 0.5).

Examples:
    axol tune.repeatability               # bounce A↔B forever
    axol tune.repeatability --cycles 5    # five A→B→A round trips, then rest
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np

from ...constants import ARM_JOINTS, Joint
from ...kinematics.solver import KinematicsSolver
from ...robot import Axol, closer_end_stop
from ...robot.config import AxolConfig
from ...teleop.config import VRTeleopConfig
from ...teleop.trajectory import plan_collision_aware_trajectory

_RATE_HZ = (
    250.0  # waypoint density — high for smooth playback (speed is set by --speed)
)
_PLAN_SPEED = 0.1 * np.pi  # rad/s — joint-space speed for planned trajectories
_PLAN_MIN_DURATION = 0.5  # seconds — floor on planned trajectory duration


# ---------------------------------------------------------------------------
# Waypoint poses (raw motor-frame angles, as printed by ``axol motor.info``)
# ---------------------------------------------------------------------------

# Left-arm poses in :data:`ARM_JOINTS` order (motor ids 0x01…0x07), copied
# straight from ``axol motor.info --l``. These are *motor-frame* readings
# (encoder zero at the calibration end stop); :func:`_build_pose` converts
# them to joint frame (0 = rest) before planning. Re-pose and paste to retune.
_POSE_A_LEFT: dict[Joint, float] = {
    Joint.SHOULDER_1: 0.2752,
    Joint.SHOULDER_2: -0.4388,
    Joint.SHOULDER_3: 2.4007,
    Joint.ELBOW: -0.3112,
    Joint.WRIST_1: 2.4147,
    Joint.WRIST_2: 1.6943,
    Joint.WRIST_3: 2.1761,
}
_POSE_B_LEFT: dict[Joint, float] = {
    Joint.SHOULDER_1: 1.9953,
    Joint.SHOULDER_2: -0.4185,
    Joint.SHOULDER_3: 2.3180,
    Joint.ELBOW: -1.4659,
    Joint.WRIST_1: 2.4560,
    Joint.WRIST_2: 1.5921,
    Joint.WRIST_3: 1.6711,
}


def _build_pose(
    solver: KinematicsSolver, q_rest: np.ndarray, motor_pose: dict[Joint, float]
) -> np.ndarray:
    """Pack a motor-frame left-arm pose into a full-N joint-frame solver vector.

    ``axol motor.info`` reports raw motor-frame angles (encoder zero at the
    calibration end stop). The control stack works in joint frame (0 = rest),
    where ``joint = motor + closer_end_stop(j, is_left)[0]``. The right arm is
    held at its rest configuration so only the left arm differs from rest.
    """
    q = q_rest.copy()
    for i, j in enumerate(ARM_JOINTS):
        offset = closer_end_stop(j, is_left=True)[0]
        q[solver.left_indices[i]] = motor_pose[j] + offset
    return q


# ---------------------------------------------------------------------------
# Trajectory planning
# ---------------------------------------------------------------------------


def _plan_joint_trajectory(
    solver: KinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    speed_rad_s: float,
    rate_hz: float,
) -> list[np.ndarray]:
    """Collision-aware joint-space slerp from ``q_from`` to ``q_to``.

    Thin wrapper around :func:`plan_collision_aware_trajectory`. Smoothsteps
    a linear interpolation in joint space and projects each waypoint with
    limit + self-collision costs so the body never clips the torso during
    the arc. Returns one full ``(N,)`` joint vector per control tick.
    """
    return plan_collision_aware_trajectory(
        solver.robot,
        solver.robot_coll,
        q_from,
        q_to,
        speed=speed_rad_s,
        rate=rate_hz,
        min_duration=_PLAN_MIN_DURATION,
    )


# ---------------------------------------------------------------------------
# Joint-vector marshalling between solver and motion_control
# ---------------------------------------------------------------------------


def _make_motion_command(
    q_full: np.ndarray, solver: KinematicsSolver
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a full-N solver vector into per-arm ``(8,)`` ``motion_control`` arrays.

    ``solver.left_indices`` / ``right_indices`` are already in
    :data:`ARM_JOINTS` order, matching the first 7 entries of the
    ``motion_control`` vector. The eighth slot is the gripper, normalised —
    held at ``0.0`` (closed).
    """
    left = np.zeros(8, dtype=np.float32)
    right = np.zeros(8, dtype=np.float32)
    left[:7] = q_full[solver.left_indices]
    right[:7] = q_full[solver.right_indices]
    return left, right


def _snapshot_q(
    axol: Axol, solver: KinematicsSolver, q_default: np.ndarray
) -> np.ndarray:
    """Read the *cached* arm positions into a full-N solver vector.

    ``axol.get_positions()`` polls the bus directly and is rejected once
    telemetry is running; the per-arm ``positions`` properties read the
    telemetry cache instead. Joints belonging to a disabled / not-yet-
    populated arm fall back to ``q_default`` so the solver vector is always
    well-defined.
    """
    q = q_default.copy()
    if axol.left is not None:
        q[solver.left_indices] = axol.left.positions[:7]
    if axol.right is not None:
        q[solver.right_indices] = axol.right.positions[:7]
    return q


async def _execute(
    axol: Axol,
    solver: KinematicsSolver,
    trajectory: list[np.ndarray],
    rate_hz: float,
) -> None:
    """Send the planned trajectory at ``rate_hz``."""
    dt = 1.0 / rate_hz
    for q in trajectory:
        loop_start = time.monotonic()
        left, right = _make_motion_command(q, solver)
        await axol.motion_control(
            left=left if axol.left is not None else None,
            right=right if axol.right is not None else None,
        )
        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.repeatability`` subcommand."""
    p = subparsers.add_parser(
        "tune.repeatability",
        help="Drive the left arm back and forth between rest and a target pose.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of A→B→A round trips. 0 (default) = run until Ctrl-C.",
    )
    p.add_argument(
        "--gripper-torque-limit",
        type=float,
        default=0.3,
        help="Gripper closing torque limit (Nm). Default 0.3.",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="Seconds to hold at each waypoint (except B). Default 0.5.",
    )
    p.add_argument(
        "--dwell-b",
        type=float,
        default=3.0,
        help="Seconds to hold at pose B before returning to A. Default 3.0.",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=_RATE_HZ,
        help=(
            "Control loop / waypoint rate in Hz. Higher = finer steps and "
            "smoother playback at the same speed. Default 250."
        ),
    )
    p.add_argument(
        "--speed",
        type=float,
        default=_PLAN_SPEED,
        help=(
            "Average joint speed (rad/s) of the worst-case joint; peak is "
            "1.5x this. Lower = slower, gentler motion. Default ~0.31."
        ),
    )
    p.add_argument(
        "--stiffness",
        type=float,
        default=0.5,
        help=(
            "Arm stiffness scale (0-1). Lower softens the motors and reduces "
            "tracking jitter. Default 0.5."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the left-arm repeatability test (rest ↔ target cycles)."""
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Inner ``_run`` already returns gently to rest before disabling —
        # this only catches a second Ctrl-C while the cleanup itself is
        # still in flight.
        print("\nExiting tune.repeatability ...")


async def _run(args: argparse.Namespace) -> None:
    rest_cfg = VRTeleopConfig()

    print("Loading kinematics solver (JIT compile may take a few seconds) ...")
    solver = KinematicsSolver()

    # Full-N rest vector. ``solver`` is only used for joint-index marshalling,
    # FK reporting, and collision-aware path planning.
    q_rest = np.zeros(solver.num_joints, dtype=np.float32)
    for i, gi in enumerate(solver.left_indices):
        q_rest[gi] = rest_cfg.rest_pose_left[i]
    for i, gi in enumerate(solver.right_indices):
        q_rest[gi] = rest_cfg.rest_pose_right[i]
    q_a = _build_pose(solver, q_rest, _POSE_A_LEFT)
    q_b = _build_pose(solver, q_rest, _POSE_B_LEFT)

    # Report the FK gripper position at each waypoint so the operator can
    # eyeball the geometry before any motors move.
    for name, q_wp in (("rest", q_rest), ("A", q_a), ("B", q_b)):
        se3, _ = solver.fk(q_wp)
        p = np.asarray(se3.translation())
        print(f"  left gripper @ {name:4} → ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")

    # The arm moves rest → A once, then bounces A → B → A → B … for every
    # cycle, returning to rest only at the very end (or on Ctrl-C). Each leg
    # is a collision-aware joint-space slerp with a ``--dwell`` pause at each
    # waypoint. Pure interpolation — no IK in the motion path.
    print("Planning rest → A and A ↔ B legs ...")
    traj_rest_a = _plan_joint_trajectory(solver, q_rest, q_a, args.speed, args.rate)
    traj_a_b = _plan_joint_trajectory(solver, q_a, q_b, args.speed, args.rate)
    traj_b_a = _plan_joint_trajectory(solver, q_b, q_a, args.speed, args.rate)
    print(
        f"  rest → A: {len(traj_rest_a)} wp  A → B: {len(traj_a_b)} wp  "
        f"B → A: {len(traj_b_a)} wp  ({len(traj_a_b) / args.rate:.2f} s per A↔B leg)"
    )

    # Only the left arm actuates — disable the right channel entirely.
    axol_kwargs: dict = {"right_channel": None}
    axol_config = AxolConfig(
        left_stiffness=args.stiffness, right_stiffness=args.stiffness
    )
    axol_config.left.gripper.torque_limit = args.gripper_torque_limit
    axol_config.right.gripper.torque_limit = args.gripper_torque_limit

    print(
        f"Repeatability run: "
        f"{'∞' if args.cycles == 0 else args.cycles} cycle(s), "
        f"rate={args.rate:.0f} Hz. Press Ctrl-C to stop."
    )

    async with Axol(config=axol_config, **axol_kwargs) as axol:
        await axol.start_telemetry(500)
        # Settle the telemetry cache before driving (mirrors gravity_comp).
        await axol.wait_for_telemetry()

        # Always begin from the planned rest pose. If the operator parked the
        # arm anywhere else, sneak there with a one-off collision-aware plan
        # so the first cycle doesn't snap.
        q_start = _snapshot_q(axol, solver, q_rest)
        if float(np.max(np.abs(q_start - q_rest))) > 0.02:
            print("Moving from current pose to rest ...")
            traj_init = _plan_joint_trajectory(
                solver, q_start, q_rest, args.speed, args.rate
            )
            await _execute(axol, solver, traj_init, args.rate)

        try:
            # Enter the A↔B oscillation from rest, once.
            print("  rest → A")
            await _execute(axol, solver, traj_rest_a, args.rate)
            await asyncio.sleep(args.dwell)

            cycle = 0
            while args.cycles == 0 or cycle < args.cycles:
                cycle += 1
                print(f"  cycle {cycle}: A → B")
                await _execute(axol, solver, traj_a_b, args.rate)
                await asyncio.sleep(args.dwell_b)
                print(f"  cycle {cycle}: B → A")
                await _execute(axol, solver, traj_b_a, args.rate)
                await asyncio.sleep(args.dwell)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  interrupted — returning to rest before disabling ...")
        finally:
            # Python 3.11+ asyncio.run cancels the task on SIGINT, raising
            # CancelledError at the next ``await`` instead of leaking a
            # KeyboardInterrupt. Without ``uncancel`` here, every cleanup
            # ``await`` below would re-raise CancelledError immediately and
            # the arm would skip the return-to-rest motion.
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()

            # Always finish at rest. Re-plan from the *current* commanded pose
            # since we may have bailed mid-trajectory.
            q_now = _snapshot_q(axol, solver, q_rest)
            if float(np.max(np.abs(q_now - q_rest))) > 0.02:
                traj_back = _plan_joint_trajectory(
                    solver, q_now, q_rest, args.speed, args.rate
                )
                await _execute(axol, solver, traj_back, args.rate)
