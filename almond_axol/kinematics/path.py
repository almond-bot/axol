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
- **Solve coarse, spline fine.** IK runs a couple of dozen times a second, far
  under the control rate, and a cubic spline fills in the ticks between. This
  is not only cheaper: solving every tick commands the solver's own
  sample-to-sample noise straight into the arm, and interpolating between
  sparse solves leaves it behind.
- **Minimum-jerk timing.** Legs start and stop with zero acceleration as well
  as zero velocity (:func:`ease`), so there is no impulse at a departure or an
  arrival.
- **An arm that isn't going anywhere is pinned.** Below ``min_travel`` the
  taught difference is gravity-compensation drift, not intent, and tracking it
  makes a 7-DOF arm hunt through its null space: degrees of elbow swing to
  chase a millimetre of tip.

Redundancy (7 DOF for a 6-DOF task) is resolved by the *taught* posture: the
solver's posture attractor is swept from the start waypoint's joint vector to
the end waypoint's, so the elbow follows the pose the operator hand-guided the
arm through rather than whatever null-space configuration the solver drifts
into.
"""

from __future__ import annotations

import logging
from math import ceil

import jax.numpy as jnp
import jaxlie
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

from ..constants import GRIPPER_TIP_OFFSET
from .solver import KinematicsSolver

_logger = logging.getLogger(__name__)

# Solves per second of motion — a floor under the spacing caps below, and the
# only one that binds on a slow leg, where the caps are met long before the
# leg is over. Extra solves past this only re-derive what interpolation
# already gets right, and they actively hurt: IK lands each sample a fraction
# of a milliradian off its neighbours' trend (the flat direction of the cost,
# not early stopping — more iterations do not help), and commanding those
# samples directly turns that wobble into acceleration noise. Solving sparsely
# and splining between leaves the wobble behind. Measured peak tip
# acceleration on a 35 cm leg at 0.03 m/s: 0.073 m/s² at 20 Hz, 0.022 at 10 —
# against 0.015 for the ideal profile.
DEFAULT_PLAN_RATE = 10.0

# Ceiling on how far the tip may travel, and how far it may turn, between two
# solved samples. Sample spacing has to follow the path's geometry as well as
# its duration: a fast leg covers more ground per second, and past roughly
# 15 mm a sample the solver starts running into its own per-call joint-step
# clamp and falls behind the line.
MAX_STEP_M = 0.010
MAX_STEP_RAD = 0.10

# Samples are spaced evenly in time, and :func:`ease` runs at 1.875x the
# average speed mid-segment, so that is the spacing the caps have to hold at.
_PEAK_SPEED_FACTOR = 1.875

# Below this rotation magnitude (rad) the Rodrigues expansion is replaced by
# its first-order term, which is exact to float32 precision at that scale.
_SMALL_ANGLE = 1e-8

Pose = tuple[np.ndarray, np.ndarray]
"""A frame as ``(position (3,), rotation (3, 3))`` in the world frame."""

Offset = np.ndarray | tuple[float, float, float]
"""A tool point in the gripper link frame."""


def ease(t: ArrayLike) -> np.ndarray:
    """Minimum-jerk easing: ``6t⁵ - 15t⁴ + 10t³`` over ``t`` in ``[0, 1]``.

    Both the first *and* second derivatives vanish at each end, so a move
    starting or stopping on this curve does so with no step in acceleration.
    The obvious cheaper choice, smoothstep, gets velocity right but steps
    acceleration from zero to its peak the instant a leg begins — a jerk
    impulse at every departure and arrival, which is exactly where a stiff
    arm is felt to jolt (measured 6.8 m/s³ against 0.3 here).

    The cost is a peakier profile for the same duration: this tops out at
    1.875x the average speed where smoothstep reaches 1.5x.
    """
    t = np.asarray(t, dtype=np.float64)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


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


def _resample(solved: list[np.ndarray], n_out: int) -> list[np.ndarray]:
    """Stretch ``solved`` to ``n_out`` evenly spaced joint vectors.

    ``solved`` covers the segment at the (coarser) planning rate, sample ``i``
    landing at time ``(i + 1) / len(solved)``; the last output tick therefore
    lands exactly on the final solved pose. Ticks before the first solved
    sample hold at it rather than ramping in from the caller's start pose:
    that ramp would have to absorb the whole difference between the taught
    configuration and the one IK settles into within a single planning
    interval, which is a velocity spike. Blending that difference out over a
    sensible time is the caller's job.

    The interpolation is a clamped cubic spline rather than straight lines
    between samples. Joining the samples with lines leaves a corner at each
    one — a step in velocity, at the planning rate, that a stiff arm tracks
    faithfully enough to feel. The spline is continuous in acceleration, and
    clamping it holds the segment's ends at rest to match the motion profile.
    """
    if n_out <= len(solved):
        return solved
    knots = np.arange(1, len(solved) + 1, dtype=np.float64) / len(solved)
    values = np.asarray(solved, dtype=np.float64)
    ticks = np.arange(1, n_out + 1, dtype=np.float64) / n_out
    spline = CubicSpline(knots, values, axis=0, bc_type="clamped")
    # Ticks before the first knot would be extrapolated; hold them instead.
    out = spline(np.clip(ticks, knots[0], knots[-1])).astype(np.float32)
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
    min_travel: float = 0.01,
    min_rotation: float = 0.05,
    pos_tolerance: float = 0.01,
    ori_tolerance: float = 0.15,
    min_duration: float = 0.25,
    settle_fraction: float = 0.05,
    max_settle_iters: int = 8,
    tracking_weight_scale: float = 10.0,
    label: str = "segment",
) -> list[np.ndarray]:
    """Plan a straight-line Cartesian move between two taught joint vectors.

    Both gripper tips travel a straight world-frame line from their pose at
    ``q_from`` to their pose at ``q_to`` while their orientations slerp, on the
    minimum-jerk profile of :func:`ease`. Samples are resolved to joint angles with
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
                         tip, averaged over the segment. The minimum-jerk
                         profile peaks at 1.875x this mid-segment.
        ang_speed:       Angular speed (rad/s) applied the same way.
        rate:            Tick rate (Hz) the result will be played back at.
        plan_rate:       Floor on samples solved per second of motion, capped
            at ``rate``. :data:`MAX_STEP_M` and :data:`MAX_STEP_RAD` raise the
            count further when the tip covers ground quickly; the solved
            vectors are splined up to ``rate`` afterwards.
        tool_offset:     Tip of the tool in the gripper link frame — the point
                         held to the straight line. Pass zeros to hold the
                         gripper mount itself (a gripperless arm).
        min_travel:      An arm whose tip moves less than this (m) between the
            two configurations, and turns less than ``min_rotation``, is left
            out of the solve and pinned at ``q_from`` instead. An arm the
            operator never touched still records a millimetre or two of
            gravity-compensation drift per waypoint, and asking IK to track
            that has it hunting through the null space — the tip creeps a
            millimetre while the elbow swings degrees. Pinning is also what
            you want physically: an arm nobody taught should not move.
        min_rotation:    Rotation counterpart to ``min_travel`` (rad).
        pos_tolerance:   Tip position error (m) at which a sample is declared
                         unreachable and the whole segment fails.
        ori_tolerance:   Orientation counterpart to ``pos_tolerance`` (rad).
        min_duration:    Floor on the segment duration (s).
        settle_fraction: Fraction of the tolerances a sample is re-solved down
            to before moving on. Giving up as soon as a sample is merely
            *acceptable* leaves each one a different distance behind its
            target, and that difference — millimetres, varying sample to
            sample — becomes acceleration noise on playback. The two numbers
            answer different questions: how well a reachable sample should be
            solved, and when to conclude one is not reachable at all.
        max_settle_iters: Extra solver calls allowed per sample.
            :meth:`KinematicsSolver.ik` clamps each call to
            ``KinematicsConfig.max_joint_delta``, so a sample needing a larger
            joint step (near a wrist flip, say) is re-solved against the same
            target until it converges, stops improving, or this budget runs out.
        tracking_weight_scale: Factor applied to the solver's pose-cost weights
            while planning. The configured weights are balanced for live
            teleop, where the null-space terms usefully damp a noisy
            hand-tracked target; against a target that is already exact they
            just pull the gripper off the line (measured ~9 mm, versus under
            1 mm at 10x). Restored before returning.
        label:           Segment name used in error messages.

    Returns:
        One full ``(N,)`` joint vector per control tick at ``rate``, ending at
        the configuration that reaches ``q_to``'s gripper poses. The first
        vector is the configuration IK settles into at the start of the line,
        which is near ``q_from`` but generally not equal to it: the caller is
        expected to blend into it (see ``ease_in`` in ``axol waypoints``).

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
    arm_indices = (solver.left_indices, solver.right_indices)

    pinned = [
        float(np.linalg.norm(g[0] - s[0])) < min_travel
        and float(np.linalg.norm(w)) < min_rotation
        for s, g, w in zip(start, goal, logs)
    ]
    for side, is_pinned in zip(("left", "right"), pinned):
        if is_pinned:
            _logger.debug(
                "%s: %s arm holds still (below the travel floor)", label, side
            )

    duration = min_duration
    n_solve = 2
    for (p_start, _), (p_goal, _), w, is_pinned in zip(start, goal, logs, pinned):
        if is_pinned:
            continue
        travel = float(np.linalg.norm(p_goal - p_start))
        turn = float(np.linalg.norm(w))
        duration = max(duration, travel / speed, turn / ang_speed)
        n_solve = max(
            n_solve,
            ceil(_PEAK_SPEED_FACTOR * travel / MAX_STEP_M),
            ceil(_PEAK_SPEED_FACTOR * turn / MAX_STEP_RAD),
        )
    n_solve = max(n_solve, round(duration * min(plan_rate, rate)))

    solved: list[np.ndarray] = []
    posture_before = solver.posture_pose
    weights_before = (solver.config.pos_weight, solver.config.ori_weight)
    solver.config.pos_weight = weights_before[0] * tracking_weight_scale
    solver.config.ori_weight = weights_before[1] * tracking_weight_scale

    def tips_at(t: float) -> list[Pose]:
        """Where both tips belong at time ``t`` in [0, 1] along the line.

        A pinned arm keeps its starting pose rather than dropping out of the
        call: passing ``None`` there would be a different JAX trace and cost a
        fresh multi-second compile.
        """
        alpha = float(ease(t))
        return [
            (p_start, r_start)
            if is_pinned
            else (
                (1.0 - alpha) * p_start + alpha * p_goal,
                r_start @ _rodrigues(alpha * w),
            )
            for (p_start, r_start), (p_goal, _), w, is_pinned in zip(
                start, goal, logs, pinned
            )
        ]

    q = q_from.copy()
    try:
        for i in range(n_solve):
            t = (i + 1) / n_solve
            alpha = float(ease(t))
            tips = tips_at(t)
            # The solver aims the mount, so ask it for the mount pose that
            # puts the tip on the line.
            targets = [_remove_offset(tip, tool_offset) for tip in tips]
            # Sweep the null-space attractor along the taught postures so the
            # elbow tracks how the arm was posed by hand.
            posture = q_from * (1.0 - alpha) + q_to * alpha
            for idx, is_pinned in zip(arm_indices, pinned):
                if is_pinned:
                    posture[idx] = q_from[idx]
            solver.set_posture_pose(posture)

            previous = np.inf
            for _ in range(max_settle_iters + 1):
                q = solver.ik(q, left_pose=targets[0], right_pose=targets[1])
                # Discard whatever the solve did to a pinned arm — with its
                # target already met, any movement is null-space wander.
                for idx, is_pinned in zip(arm_indices, pinned):
                    if is_pinned:
                        q[idx] = q_from[idx]
                errors = [
                    (
                        float(np.linalg.norm(actual[0] - target[0])),
                        _rot_angle(actual[1], target[1]),
                    )
                    for actual, target in zip(tip_poses(solver, q, tool_offset), tips)
                ]
                worst_pos = max(pos for pos, _ in errors)
                if worst_pos <= pos_tolerance * settle_fraction and all(
                    ori <= ori_tolerance * settle_fraction for _, ori in errors
                ):
                    break
                if worst_pos > previous * 0.95:
                    break  # as close as this target is going to get
                previous = worst_pos
            if any(pos > pos_tolerance or ori > ori_tolerance for pos, ori in errors):
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

    n_out = max(2, round(duration * rate))
    out = _resample(solved, n_out)
    # Only the solved samples have been checked so far. Between them the path
    # is the spline's guess, so check there too — halfway between neighbouring
    # samples, where a chord deviates most — and let a caller who has asked for
    # more speed than the sampling supports hear about it before anything moves.
    for i in range(len(solved) - 1):
        t = (i + 1.5) / len(solved)
        k = min(n_out - 1, max(0, round(t * n_out) - 1))
        for actual, target in zip(tip_poses(solver, out[k], tool_offset), tips_at(t)):
            error = float(np.linalg.norm(actual[0] - target[0]))
            if error > pos_tolerance:
                raise PathPlanningError(
                    f"{label}: the path bows {error * 1e3:.0f} mm off the straight "
                    f"line {t:.0%} of the way along, between solved samples. "
                    "Lower the speed, or raise plan_rate."
                )
    return out
