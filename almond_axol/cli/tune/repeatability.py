"""
axol tune.repeatability

Repeatability test for the Axol arms: drive between the rest pose and a
crossed-arms tips-touching pose, planning each leg with pyroki + the URDF
so the body never clips the torso during the long arc.

The touching pose is hard-coded from a hand-posed measurement (see
:data:`_TOUCH_LEFT` / :data:`_TOUCH_RIGHT` below) — easier and far more
reliable than IK once you've physically dialled in the contact point.
The gripper is held closed throughout so the finger tips can actually
touch.

Useful for measuring how reliably the grippers return to the same
physical contact point after a long sequence of motions.

Examples:
    axol tune.repeatability                    # forever, max compliance
    axol tune.repeatability --stiffness 0.5    # blend partway to stiff gains
    axol tune.repeatability --cycles 5         # five touch-and-return cycles
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import time

import jax
import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk

from ...kinematics.solver import KinematicsSolver
from ...robot import Axol
from ...robot.config import AxolConfig
from ...shared import ARM_JOINTS, Joint
from ...teleop.config import VRTeleopConfig

_RATE_HZ = 100.0
_PLAN_SPEED = 0.1 * np.pi  # rad/s — joint-space speed for the planned trajectory


# ---------------------------------------------------------------------------
# Touch pose
# ---------------------------------------------------------------------------

# Hand-posed crossed-arms tips-touching configuration, in :data:`ARM_JOINTS`
# order (shoulder_1, shoulder_2, shoulder_3, elbow, wrist_1, wrist_2,
# wrist_3) — read off ``axol motor.info`` while the operator held the arms
# in the desired contact pose. Edit these in place to re-calibrate.
_TOUCH_LEFT: dict[Joint, float] = {
    Joint.SHOULDER_1: 0.1194,
    Joint.SHOULDER_2: -0.0515,
    Joint.SHOULDER_3: -0.3430,
    Joint.ELBOW: 1.3842,
    Joint.WRIST_1: -0.2252,
    Joint.WRIST_2: -0.1582,
    Joint.WRIST_3: 0.3352,
}
_TOUCH_RIGHT: dict[Joint, float] = {
    Joint.SHOULDER_1: -0.1827,
    Joint.SHOULDER_2: -0.0716,
    Joint.SHOULDER_3: 0.1444,
    Joint.ELBOW: -1.3580,
    Joint.WRIST_1: 0.3029,
    Joint.WRIST_2: 0.5780,
    Joint.WRIST_3: -0.3291,
}


def _build_q_touch(solver: KinematicsSolver) -> np.ndarray:
    """Pack :data:`_TOUCH_LEFT` / :data:`_TOUCH_RIGHT` into the full-N solver vector."""
    q = np.zeros(solver.num_joints, dtype=np.float32)
    for i, j in enumerate(ARM_JOINTS):
        q[solver.left_indices[i]] = _TOUCH_LEFT[j]
        q[solver.right_indices[i]] = _TOUCH_RIGHT[j]
    return q


# ---------------------------------------------------------------------------
# Collision-aware joint-space slerp
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def _solve_path_step(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    q_interp: jax.Array,
    q_current: jax.Array,
    rest_weight: float,
    limit_weight: float,
    collision_margin: float,
    collision_weight: float,
    max_iterations: int,
) -> jax.Array:
    """One IK step toward ``q_interp`` with limit + self-collision costs.

    Mirrors :func:`almond_axol.teleop.worker._solve_reset_step` — the
    trajectory here is generated the same way (smoothstep slerp through
    pyroki) but as a one-shot offline plan rather than a live feedback
    loop.
    """
    JointVar = robot.joint_var_cls
    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_interp, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=collision_margin,
            weight=collision_weight,
        ),
    ]
    var_joints = JointVar(jnp.array([0]))
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(q_current[jnp.newaxis, :])]
    )
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    solution_vals = problem.analyze().solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=1e-2,
        ),
    )
    return solution_vals[var_joints][0]


def _plan_trajectory(
    solver: KinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    speed_rad_s: float,
    rate_hz: float,
) -> list[np.ndarray]:
    """Collision-aware joint-space slerp from ``q_from`` to ``q_to``.

    Smoothsteps a linear interpolation in joint space and projects each
    waypoint with limit + self-collision costs so the body never clips
    the torso during the arc. Returns one full ``(N,)`` joint vector per
    control tick at ``rate_hz``.
    """
    max_dist = float(np.max(np.abs(q_from - q_to)))
    duration = max(max_dist / speed_rad_s, 0.5)
    n_steps = max(2, round(duration * rate_hz))

    rest_weight = 50.0
    limit_weight = 100.0
    collision_margin = 0.025
    collision_weight = 100.0
    max_iters = 10

    traj: list[np.ndarray] = []
    q = q_from.astype(np.float32).copy()
    for i in range(n_steps):
        t = (i + 1) / n_steps
        alpha = t * t * (3.0 - 2.0 * t)
        q_interp = (q_from * (1.0 - alpha) + q_to * alpha).astype(np.float32)
        q_jax = _solve_path_step(
            solver.robot,
            solver.robot_coll,
            jnp.asarray(q_interp),
            jnp.asarray(q),
            rest_weight,
            limit_weight,
            collision_margin,
            collision_weight,
            max_iters,
        )
        q = np.asarray(q_jax, dtype=np.float32)
        traj.append(q.copy())
    return traj


# ---------------------------------------------------------------------------
# Joint-vector marshalling between solver and motion_control
# ---------------------------------------------------------------------------


def _make_motion_command(
    q_full: np.ndarray, solver: KinematicsSolver
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a full-N solver vector into per-arm ``(8,)`` ``motion_control`` arrays.

    ``solver.left_indices`` / ``right_indices`` are already in
    :data:`ARM_JOINTS` order (see :func:`urdf_arm_joint_names`), which
    matches the first 7 entries of the ``motion_control`` vector. The
    eighth slot is the gripper, normalised — held at ``0.0`` (closed)
    so the finger tips can touch.
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
    telemetry is running (see ``MotorError`` in ``Motor.get_position``);
    the per-arm ``positions`` properties read the telemetry cache
    instead. Joints belonging to a disabled / not-yet-populated arm fall
    back to ``q_default`` so the solver vector is always well-defined.
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
    p = subparsers.add_parser(
        "tune.repeatability",
        help="Drive between rest pose and a crossed-arms tips-touching pose.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--stiffness",
        type=float,
        default=0.0,
        help=(
            "Compliance ↔ stiffness blend in [0, 1]. 0 (default) is fully "
            "compliant; 1 restores the pre-tuning industrial gains. See "
            "AxolConfig.stiffness."
        ),
    )
    p.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of touch-and-return cycles. 0 (default) = run until Ctrl-C.",
    )
    p.add_argument(
        "--gripper-torque-limit",
        type=float,
        default=0.3,
        help=(
            "Gripper closing torque limit (Nm). Lower keeps the close gentle "
            "when the tips collide. Default 0.3."
        ),
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="Seconds to hold each end of the cycle. Default 0.5.",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=_RATE_HZ,
        help="Control loop rate in Hz. Default 100.",
    )
    p.add_argument(
        "--no-left",
        action="store_true",
        help="Disable the left arm (no touching — left arm stays at rest).",
    )
    p.add_argument(
        "--no-right",
        action="store_true",
        help="Disable the right arm (no touching — right arm stays at rest).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Inner ``_run`` already returns gently to rest before disabling —
        # this only catches a second Ctrl-C while the cleanup itself is
        # still in flight.
        print("\nExiting tune.repeatability ...")


async def _run(args: argparse.Namespace) -> None:
    if args.no_left and args.no_right:
        raise SystemExit("Both arms disabled — nothing to test.")
    if not 0.0 <= args.stiffness <= 1.0:
        raise SystemExit(f"--stiffness must be in [0, 1], got {args.stiffness}")

    rest_cfg = VRTeleopConfig()

    print("Loading kinematics solver (JIT compile may take a few seconds) ...")
    solver = KinematicsSolver()

    # Build the full-N rest and touch vectors. ``q_touch`` is the hand-posed
    # measurement at the top of this file; ``solver`` is only used for
    # joint-index marshalling, FK reporting, and collision-aware path
    # planning.
    q_rest = np.zeros(solver.num_joints, dtype=np.float32)
    for i, gi in enumerate(solver.left_indices):
        q_rest[gi] = rest_cfg.rest_pose_left[i]
    for i, gi in enumerate(solver.right_indices):
        q_rest[gi] = rest_cfg.rest_pose_right[i]
    q_touch = _build_q_touch(solver)

    # Report the FK-computed tip positions so the operator can eyeball the
    # touch geometry before any motors move.
    L_se3, R_se3 = solver.fk(q_touch)
    L_pos = np.asarray(L_se3.translation())
    R_pos = np.asarray(R_se3.translation())
    gap_mm = float(np.linalg.norm(L_pos - R_pos)) * 1000.0
    print(
        f"Touch pose (FK on final joint angles):\n"
        f"  left  gripper → ({L_pos[0]:+.3f}, {L_pos[1]:+.3f}, "
        f"{L_pos[2]:+.3f}) m\n"
        f"  right gripper → ({R_pos[0]:+.3f}, {R_pos[1]:+.3f}, "
        f"{R_pos[2]:+.3f}) m\n"
        f"  gripper-link separation: {gap_mm:.1f} mm"
    )

    print("Planning rest ↔ touch trajectories ...")
    traj_to_touch = _plan_trajectory(solver, q_rest, q_touch, _PLAN_SPEED, args.rate)
    traj_to_rest = _plan_trajectory(solver, q_touch, q_rest, _PLAN_SPEED, args.rate)
    print(
        f"  → touch: {len(traj_to_touch)} waypoints  "
        f"→ rest: {len(traj_to_rest)} waypoints  "
        f"({len(traj_to_touch) / args.rate:.2f} s each)"
    )

    axol_kwargs: dict = {}
    if args.no_left:
        axol_kwargs["left_channel"] = None
    if args.no_right:
        axol_kwargs["right_channel"] = None
    axol_config = AxolConfig(stiffness=args.stiffness)
    axol_config.left.gripper.torque_limit = args.gripper_torque_limit
    axol_config.right.gripper.torque_limit = args.gripper_torque_limit

    print(
        f"Repeatability run: stiffness={args.stiffness:.2f}, "
        f"{'∞' if args.cycles == 0 else args.cycles} cycle(s), "
        f"rate={args.rate:.0f} Hz. Press Ctrl-C to stop."
    )

    async with Axol(config=axol_config, **axol_kwargs) as axol:
        await axol.start_telemetry(500)
        # Settle the telemetry cache before driving (mirrors gravity_comp).
        await asyncio.sleep(0.05)

        # Always begin from the planned rest pose. If the operator parked the
        # arms anywhere else, sneak there with a one-off collision-aware plan
        # so the first cycle doesn't snap. Read the cached positions —
        # ``axol.get_positions()`` polls the bus directly and is rejected
        # while telemetry is active.
        q_start = _snapshot_q(axol, solver, q_rest)
        if float(np.max(np.abs(q_start - q_rest))) > 0.02:
            print("Moving from current pose to rest ...")
            traj_init = _plan_trajectory(
                solver, q_start, q_rest, _PLAN_SPEED, args.rate
            )
            await _execute(axol, solver, traj_init, args.rate)

        try:
            cycle = 0
            while args.cycles == 0 or cycle < args.cycles:
                cycle += 1
                print(f"  cycle {cycle}: rest → touch")
                await _execute(axol, solver, traj_to_touch, args.rate)
                await asyncio.sleep(args.dwell)
                print(f"  cycle {cycle}: touch → rest")
                await _execute(axol, solver, traj_to_rest, args.rate)
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

            # Always finish at rest. Re-plan from the *current* commanded
            # pose since we may have bailed mid-trajectory. Read the cached
            # positions — ``axol.get_positions()`` polls the bus directly
            # and is rejected while telemetry is active.
            q_now = _snapshot_q(axol, solver, q_rest)
            if float(np.max(np.abs(q_now - q_rest))) > 0.02:
                traj_back = _plan_trajectory(
                    solver, q_now, q_rest, _PLAN_SPEED, args.rate
                )
                await _execute(axol, solver, traj_back, args.rate)
