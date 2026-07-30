"""Straight-line Cartesian path planning between taught joint configurations.

Where :mod:`almond_axol.teleop.trajectory` interpolates in *joint* space (the
return-to-rest planner: shortest joint path, arbitrary end-effector arc), this
module interpolates in *Cartesian* space: each gripper tip travels a straight
line in world coordinates while its orientation slerps, and every sample along
the way is resolved back to joint angles with :class:`~.solver.KinematicsSolver`.

Used by ``axol waypoints`` to replay hand-taught waypoints. The whole path is
solved before any motor moves, so an unreachable segment raises
:class:`PathPlanningError` instead of stalling the arm mid-move.

Two details make the line straight *where it matters* and quick to compute:

- **The tip, not the flange.** The solver's target frame is the gripper mount
  link; the fingertips are :data:`~almond_axol.constants.GRIPPER_TIP_OFFSET`
  beyond it. Holding the mount to a straight line swings the tip through an
  arc as the wrist reorients, so the line is planned for the tip and converted
  back to a mount target for each solve.
- **Solve coarse, emit fine.** IK runs at :data:`DEFAULT_PLAN_RATE`, well
  under the control rate, and the solved joint vectors are interpolated up to
  one per tick. Consecutive solves are milliseconds and a millimetre apart, so
  the interpolation is invisible while the solve count drops several-fold.

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

from ..constants import GRIPPER_TIP_OFFSET
from .solver import KinematicsSolver

_logger = logging.getLogger(__name__)

# Solves per second of motion. Above roughly this the extra solves only
# re-derive what interpolation already gets right, at full IK cost apiece.
DEFAULT_PLAN_RATE = 50.0

# Below this rotation magnitude (rad) the Rodrigues expansion is replaced by
# its first-order term, which is exact to float32 precision at that scale.
_SMALL_ANGLE = 1e-8

Pose = tuple[np.ndarray, np.ndarray]
"""A frame as ``(position (3,), rotation (3, 3))`` in the world frame."""

Offset = np.ndarray | tuple[float, float, float]
"""A tool point in the gripper link frame."""


class PathPlanningError(RuntimeError):
    """A Cartesian segment could not be tracked within tolerance.

    Raised before any motion is commanded — the arm never leaves the pose it
    was in when planning started.
    """


def ee_poses(solver: KinematicsSolver, q: np.ndarray) -> tuple[Pose, Pose]:
    """Return ``(left, right)`` gripper-mount poses for a full joint vector.

    This is the frame :meth:`KinematicsSolver.ik` targets. For the point the
    fingers close on, see :func:`tip_poses`.
    """
    left, right = solver.fk(np.asarray(q, dtype=np.float32))
    return _to_pose(left), _to_pose(right)


def tip_poses(
    solver: KinematicsSolver,
    q: np.ndarray,
    tool_offset: Offset = GRIPPER_TIP_OFFSET,
) -> tuple[Pose, Pose]:
    """Return ``(left, right)`` gripper-tip poses for a full joint vector.

    Same orientation as the mount, translated out to the fingertips.
    """
    offset = np.asarray(tool_offset, dtype=np.float32)
    left, right = ee_poses(solver, q)
    return _apply_offset(left, offset), _apply_offset(right, offset)


def _apply_offset(pose: Pose, offset: np.ndarray) -> Pose:
    position, rotation = pose
    return (position + rotation @ offset, rotation)


def _remove_offset(pose: Pose, offset: np.ndarray) -> Pose:
    """Invert :func:`_apply_offset`: the mount pose that puts the tip here."""
    position, rotation = pose
    return (position - rotation @ offset, rotation)


def _to_pose(se3: jaxlie.SE3) -> Pose:
    return (
        np.asarray(se3.translation(), dtype=np.float32),
        np.asarray(se3.rotation().as_matrix(), dtype=np.float32),
    )


def _rot_log(r_from: np.ndarray, r_to: np.ndarray) -> np.ndarray:
    """Axis-angle vector rotating ``r_from`` onto ``r_to`` (in ``r_from``'s frame)."""
    rel = jaxlie.SO3.from_matrix(jnp.asarray(r_from.T @ r_to, dtype=jnp.float32))
    return np.asarray(rel.log(), dtype=np.float32)


def _rot_angle(r_from: np.ndarray, r_to: np.ndarray) -> float:
    """Angle (rad) between two rotations, without building the axis.

    The per-sample tolerance check runs this thousands of times per segment;
    the trace identity keeps it in NumPy instead of paying JAX dispatch for a
    matrix logarithm whose axis is thrown away.
    """
    cos = (float(np.trace(r_from.T @ r_to)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


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


def _resample(
    solved: list[np.ndarray], q_start: np.ndarray, n_out: int
) -> list[np.ndarray]:
    """Stretch ``solved`` to ``n_out`` evenly spaced joint vectors.

    ``solved`` covers the segment at the (coarser) planning rate, sample ``i``
    landing at time ``(i + 1) / len(solved)``. ``q_start`` anchors time zero so
    the first output tick interpolates from where the arm actually is; the last
    output tick lands exactly on the final solved pose.
    """
    if n_out <= len(solved):
        return solved
    knots = np.linspace(0.0, 1.0, len(solved) + 1)
    values = np.asarray([q_start, *solved], dtype=np.float32)
    ticks = np.arange(1, n_out + 1, dtype=np.float32) / n_out
    out = np.empty((n_out, values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(ticks, knots, values[:, j])
    return list(out)


def warmup(
    solver: KinematicsSolver,
    tool_offset: Offset = GRIPPER_TIP_OFFSET,
    tracking_weight_scale: float = 10.0,
) -> None:
    """Compile the JAX kernels :func:`plan_linear_segment` will use.

    :class:`KinematicsSolver` warms up the call teleop makes — with elbow
    hints. Planning passes none, which is a different trace and so a fresh
    compile: a couple of seconds, landing on whatever triggers the first plan.
    Planning a degenerate zero-length segment here pays that cost up front,
    where it can overlap something slower (the operator teaching a path).
    """
    q = np.zeros(solver.num_joints, dtype=np.float32)
    try:
        plan_linear_segment(
            solver,
            q,
            q,
            speed=1.0,
            ang_speed=1.0,
            rate=8.0,
            plan_rate=8.0,
            tool_offset=tool_offset,
            tracking_weight_scale=tracking_weight_scale,
            label="warmup",
        )
    except PathPlanningError:
        # The zero pose is only a vehicle for the compile; if the solver can't
        # sit still there, real segments will still plan (and report) normally.
        _logger.debug("planner warmup did not converge", exc_info=True)


def plan_linear_segment(
    solver: KinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    speed: float,
    ang_speed: float,
    rate: float,
    plan_rate: float = DEFAULT_PLAN_RATE,
    tool_offset: Offset = GRIPPER_TIP_OFFSET,
    pos_tolerance: float = 0.01,
    ori_tolerance: float = 0.15,
    min_duration: float = 0.25,
    max_settle_iters: int = 8,
    tracking_weight_scale: float = 10.0,
    label: str = "segment",
) -> list[np.ndarray]:
    """Plan a straight-line Cartesian move between two taught joint vectors.

    Both gripper tips travel a straight world-frame line from their pose at
    ``q_from`` to their pose at ``q_to`` while their orientations slerp, on a
    smoothstep velocity profile. Samples are resolved to joint angles with
    :meth:`KinematicsSolver.ik`, each seeded from the previous solution, then
    interpolated up to one joint vector per control tick.

    The segment duration is whichever is slower across both arms and both
    modalities — ``distance / speed`` or ``angle / ang_speed`` — floored at
    ``min_duration``, so a short segment does not snap.

    Args:
        solver:          Kinematics solver. Its posture attractor and pose-cost
                         weights are swept during planning and restored before
                         returning.
        q_from:          Starting joint configuration, full ``(N,)`` vector.
        q_to:            Target joint configuration, full ``(N,)`` vector.
        speed:           Cartesian speed (m/s) of the faster-travelling gripper
                         tip, averaged over the segment.
        ang_speed:       Angular speed (rad/s) applied the same way.
        rate:            Tick rate (Hz) the result will be played back at.
        plan_rate:       Samples actually solved per second of motion. Capped
                         at ``rate``; the solved vectors are interpolated up to
                         ``rate`` afterwards.
        tool_offset:     Tip of the tool in the gripper link frame — the point
                         held to the straight line. Pass zeros to hold the
                         gripper mount itself (a gripperless arm).
        pos_tolerance:   Maximum tolerated tip position error (m) at any sample.
        ori_tolerance:   Maximum tolerated orientation error (rad).
        min_duration:    Floor on the segment duration (s).
        max_settle_iters: Extra solver calls allowed per sample.
            :meth:`KinematicsSolver.ik` clamps each call to
            ``KinematicsConfig.max_joint_delta``, so a sample needing a larger
            joint step (near a wrist flip, say) is re-solved against the same
            target until it converges or this budget runs out.
        tracking_weight_scale: Factor applied to the solver's pose-cost weights
            while planning. The configured weights are balanced for live
            teleop, where the null-space terms usefully damp a noisy
            hand-tracked target; against a target that is already exact they
            just pull the gripper off the line (measured ~9 mm, versus under
            1 mm at 10x). Restored before returning.
        label:           Segment name used in error messages.

    Returns:
        One full ``(N,)`` joint vector per control tick at ``rate``, ending at
        the configuration that reaches ``q_to``'s gripper poses.

    Raises:
        PathPlanningError: A sample could not be resolved within tolerance —
            the straight line leaves the workspace, or the solver cannot get
            there without violating a joint limit or self-collision cost.
    """
    q_from = np.asarray(q_from, dtype=np.float32)
    q_to = np.asarray(q_to, dtype=np.float32)
    tool_offset = np.asarray(tool_offset, dtype=np.float32)

    start = tip_poses(solver, q_from, tool_offset)
    goal = tip_poses(solver, q_to, tool_offset)
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
    n_solve = max(2, round(duration * min(plan_rate, rate)))

    solved: list[np.ndarray] = []
    posture_before = solver.posture_pose
    weights_before = (solver.config.pos_weight, solver.config.ori_weight)
    solver.config.pos_weight = weights_before[0] * tracking_weight_scale
    solver.config.ori_weight = weights_before[1] * tracking_weight_scale
    q = q_from.copy()
    try:
        for i in range(n_solve):
            t = (i + 1) / n_solve
            alpha = t * t * (3.0 - 2.0 * t)
            tips = [
                (
                    (1.0 - alpha) * p_start + alpha * p_goal,
                    r_start @ _rodrigues(alpha * w),
                )
                for (p_start, r_start), (p_goal, _), w in zip(start, goal, logs)
            ]
            # The solver aims the mount, so ask it for the mount pose that
            # puts the tip on the line.
            targets = [_remove_offset(tip, tool_offset) for tip in tips]
            # Sweep the null-space attractor along the taught postures so the
            # elbow tracks how the arm was posed by hand.
            solver.set_posture_pose(q_from * (1.0 - alpha) + q_to * alpha)

            for _ in range(max_settle_iters + 1):
                q = solver.ik(q, left_pose=targets[0], right_pose=targets[1])
                errors = [
                    (
                        float(np.linalg.norm(actual[0] - target[0])),
                        _rot_angle(actual[1], target[1]),
                    )
                    for actual, target in zip(tip_poses(solver, q, tool_offset), tips)
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
            solved.append(q.copy())
    finally:
        solver.set_posture_pose(posture_before)
        solver.config.pos_weight, solver.config.ori_weight = weights_before

    drift = float(np.max(np.abs(solved[-1] - q_to)))
    if drift > 0.2:
        _logger.debug(
            "%s: solved arrival differs from the taught pose by %.2f rad on the "
            "worst joint (the arm reaches the same gripper pose a different way)",
            label,
            drift,
        )
    return _resample(solved, q_from, max(2, round(duration * rate)))
