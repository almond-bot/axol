"""Straight-line Cartesian path planning between taught joint configurations.

Where :mod:`almond_axol.teleop.trajectory` interpolates in *joint* space (the
return-to-rest planner: shortest joint path, arbitrary end-effector arc), this
module interpolates in *Cartesian* space: each gripper travels a straight line
in world coordinates while its orientation slerps, and every sample along the
way is resolved back to joint angles with :class:`~.solver.KinematicsSolver`.

Used by ``axol waypoints`` to replay hand-taught waypoints. The whole path is
solved before any motor moves, so an unreachable segment raises
:class:`PathPlanningError` instead of stalling the arm mid-move.

Redundancy (7 DOF for a 6-DOF task) is resolved by the *taught* posture: the
solver's posture attractor is swept from the start waypoint's joint vector to
the end waypoint's, so the elbow follows the pose the operator hand-guided the
arm through rather than whatever null-space configuration the solver drifts
into.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import jaxlie
import numpy as np

from .solver import KinematicsSolver

_logger = logging.getLogger(__name__)

# Below this rotation magnitude (rad) the Rodrigues expansion is replaced by
# its first-order term, which is exact to float32 precision at that scale.
_SMALL_ANGLE = 1e-8

Pose = tuple[np.ndarray, np.ndarray]
"""End-effector pose as ``(position (3,), rotation (3, 3))``, world frame."""


class PathPlanningError(RuntimeError):
    """A Cartesian segment could not be tracked within tolerance.

    Raised before any motion is commanded — the arm never leaves the pose it
    was in when planning started.
    """


def ee_poses(solver: KinematicsSolver, q: np.ndarray) -> tuple[Pose, Pose]:
    """Return ``(left, right)`` end-effector poses for a full joint vector."""
    left, right = solver.fk(np.asarray(q, dtype=np.float32))
    return _to_pose(left), _to_pose(right)


def _to_pose(se3: jaxlie.SE3) -> Pose:
    return (
        np.asarray(se3.translation(), dtype=np.float32),
        np.asarray(se3.rotation().as_matrix(), dtype=np.float32),
    )


def _rot_log(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """Axis-angle vector rotating ``r_from`` onto ``r_to`` (in ``r_from``'s frame)."""
    rel = jaxlie.SO3.from_matrix(jnp.asarray(r_from.T @ r_to, dtype=jnp.float32))
    return np.asarray(rel.log(), dtype=np.float32)


def _rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for the axis-angle vector ``w`` (pure NumPy)."""
    theta = float(np.linalg.norm(w))
    k = np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=np.float32
    )
    if theta < _SMALL_ANGLE:
        return (np.eye(3, dtype=np.float32) + k).astype(np.float32)
    k = k / theta
    return (
        np.eye(3, dtype=np.float32)
        + np.sin(theta) * k
        + (1.0 - np.cos(theta)) * (k @ k)
    ).astype(np.float32)


def _pose_error(actual: Pose, target: Pose) -> tuple[float, float]:
    """Return ``(position error in m, orientation error in rad)``."""
    pos_err = float(np.linalg.norm(actual[0] - target[0]))
    ori_err = float(np.linalg.norm(_rot_log(actual[1], target[1])))
    return pos_err, ori_err


def plan_linear_segment(
    solver: KinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    speed: float,
    ang_speed: float,
    rate: float,
    pos_tolerance: float = 0.01,
    ori_tolerance: float = 0.15,
    min_duration: float = 0.25,
    max_settle_iters: int = 8,
    tracking_weight_scale: float = 10.0,
    label: str = "segment",
) -> list[np.ndarray]:
    """Plan a straight-line Cartesian move between two taught joint vectors.

    Both grippers travel a straight world-frame line from their pose at
    ``q_from`` to their pose at ``q_to`` while their orientations slerp, on a
    smoothstep velocity profile. Each sample is resolved to joint angles with
    :meth:`KinematicsSolver.ik`, seeded from the previous sample's solution.

    The segment duration is whichever is slower across both arms and both
    modalities — ``distance / speed`` or ``angle / ang_speed`` — floored at
    ``min_duration``, so a short segment does not snap.

    Args:
        solver:          Kinematics solver (its posture attractor is swept
                         during planning and restored before returning).
        q_from:          Starting joint configuration, full ``(N,)`` vector.
        q_to:            Target joint configuration, full ``(N,)`` vector.
        speed:           Cartesian speed (m/s) of the faster-travelling
                         gripper, averaged over the segment.
        ang_speed:       Angular speed (rad/s) applied the same way.
        rate:            Sample rate (Hz) the waypoints will be played at.
        pos_tolerance:   Maximum tolerated position error (m) at any sample.
        ori_tolerance:   Maximum tolerated orientation error (rad).
        min_duration:    Floor on the segment duration (s).
        max_settle_iters: Extra solver calls allowed per sample.
            :meth:`KinematicsSolver.ik` clamps each call to
            ``KinematicsConfig.max_joint_delta``, so a sample needing a larger
            joint step (near a wrist flip, say) is re-solved against the same
            target until it converges or this budget runs out.
        tracking_weight_scale: Factor applied to the solver's pose-cost
            weights while planning. The configured weights are balanced for
            live teleop, where the null-space terms usefully damp a noisy
            hand-tracked target; against a target that is already exact they
            just pull the gripper off the line (measured ~9 mm, versus under
            1 mm at 10x). Restored before returning.
        label:           Segment name used in error messages.

    Returns:
        One full ``(N,)`` joint vector per control tick at ``rate``, ending at
        the configuration that reaches ``q_to``'s end-effector poses.

    Raises:
        PathPlanningError: A sample could not be resolved within tolerance —
            the straight line leaves the workspace, or the solver cannot get
            there without violating a joint limit or self-collision cost.
    """
    q_from = np.asarray(q_from, dtype=np.float32)
    q_to = np.asarray(q_to, dtype=np.float32)

    start = ee_poses(solver, q_from)
    goal = ee_poses(solver, q_to)
    # Axis-angle deltas, precomputed per arm so each sample is a cheap
    # Rodrigues evaluation rather than a fresh matrix logarithm.
    logs = [_rot_log(s[1], g[1]) for s, g in zip(start, goal)]

    duration = min_duration
    for (p_start, _), (p_goal, _), w in zip(start, goal, logs):
        duration = max(
            duration,
            float(np.linalg.norm(p_goal - p_start)) / speed,
            float(np.linalg.norm(w)) / ang_speed,
        )
    n_steps = max(2, round(duration * rate))

    trajectory: list[np.ndarray] = []
    posture_before = solver.posture_pose
    weights_before = (solver.config.pos_weight, solver.config.ori_weight)
    solver.config.pos_weight = weights_before[0] * tracking_weight_scale
    solver.config.ori_weight = weights_before[1] * tracking_weight_scale
    q = q_from.copy()
    try:
        for i in range(n_steps):
            t = (i + 1) / n_steps
            alpha = t * t * (3.0 - 2.0 * t)
            targets = tuple(
                (
                    (1.0 - alpha) * p_start + alpha * p_goal,
                    r_start @ _rodrigues(alpha * w),
                )
                for (p_start, r_start), (p_goal, _), w in zip(start, goal, logs)
            )
            # Sweep the null-space attractor along the taught postures so the
            # elbow tracks how the arm was posed by hand.
            solver.set_posture_pose(q_from * (1.0 - alpha) + q_to * alpha)

            for _ in range(max_settle_iters + 1):
                q = solver.ik(q, left_pose=targets[0], right_pose=targets[1])
                errors = [
                    _pose_error(actual, target)
                    for actual, target in zip(ee_poses(solver, q), targets)
                ]
                if all(
                    pos <= pos_tolerance and ori <= ori_tolerance for pos, ori in errors
                ):
                    break
            else:
                worst = max(errors, key=lambda e: e[0])
                raise PathPlanningError(
                    f"{label}: cannot reach {alpha:.0%} along the straight line "
                    f"(off by {worst[0] * 1e3:.0f} mm / {np.degrees(worst[1]):.0f}°). "
                    "Move the waypoints closer together or into the arm's reach."
                )
            trajectory.append(q.copy())
    finally:
        solver.set_posture_pose(posture_before)
        solver.config.pos_weight, solver.config.ori_weight = weights_before

    drift = float(np.max(np.abs(trajectory[-1] - q_to)))
    if drift > 0.2:
        _logger.debug(
            "%s: solved arrival differs from the taught pose by %.2f rad on the "
            "worst joint (the arm reaches the same gripper pose a different way)",
            label,
            drift,
        )
    return trajectory
