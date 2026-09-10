"""Collision-aware joint-space trajectory planner.

Single source of truth for "return to rest" / "go-to-pose" trajectories
used by :class:`almond_axol.teleop.worker.IKWorker` (teleop, collect-data,
run-policy via the IK subprocess) and by
:mod:`almond_axol.cli.tune.repeatability`.

Each waypoint is a smoothstep-interpolated joint vector projected onto
the joint-limit / self-collision manifold with a small pyroki least-
squares solve. Duration is governed by ``speed`` (peak joint velocity is
``1.5 * speed`` on the worst-case joint) with a hard ``min_duration``
floor so very-short returns do not snap.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk

from ..kinematics.model import collision_cost_params
from ..kinematics.solver import KinematicsSolver, _self_collision_cost


@functools.partial(jax.jit, static_argnames=("max_iterations", "use_self_collision"))
def solve_path_step(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision | None,
    q_interp: jax.Array,
    q_current: jax.Array,
    rest_weight: float,
    limit_weight: float,
    collision_start: jax.Array,
    collision_ramp: jax.Array,
    collision_weight: float,
    max_iterations: int,
    use_self_collision: bool = True,
) -> jax.Array:
    """One IK step toward ``q_interp`` with limit + self-collision costs only.

    ``q_interp`` is the smoothstep target for this waypoint; ``q_current``
    is the previous waypoint (or the starting pose for the first step).
    Returns the projected joint configuration. With ``use_self_collision``
    false (the solver was built without a collision model) only the joint
    limits constrain the projection.
    """
    JointVar = robot.joint_var_cls
    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_interp, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight),
    ]
    if use_self_collision:
        assert robot_coll is not None
        costs.append(
            _self_collision_cost(
                robot,
                robot_coll,
                JointVar(0),
                activation_start=collision_start,
                ramp_width=collision_ramp,
                weight=collision_weight,
            )
        )
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


def plan_collision_aware_trajectory(
    solver: KinematicsSolver,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    speed: float,
    rate: float,
    min_duration: float,
    rest_weight: float = 50.0,
    limit_weight: float = 100.0,
    collision_margin: float = 0.025,
    collision_weight: float = 100.0,
    max_iterations: int = 10,
) -> list[np.ndarray]:
    """Plan a collision-aware joint-space trajectory from ``q_from`` to ``q_to``.

    The motion is smoothstepped in joint space, then projected onto the
    joint-limit / self-collision manifold by :func:`solve_path_step`. The
    duration scales with the worst-case joint deviation so the peak joint
    velocity is ``1.5 * speed`` (the smoothstep peak), but is clamped from
    below by ``min_duration`` so near-rest starts do not snap home in a
    handful of frames.

    Args:
        solver:           :class:`KinematicsSolver` providing the shared robot
                          and collision models. Joint vectors use its public
                          ordering (left then right arm, ARM_JOINTS order).
        q_from:           Starting joint configuration, shape ``(N,)``.
        q_to:             Target joint configuration, shape ``(N,)``.
        speed:            Average joint velocity (rad/s) for the worst-case
                          joint; peak velocity is ``1.5 * speed``.
        rate:             Sample rate (Hz) at which waypoints will be played.
        min_duration:     Floor on the trajectory duration in seconds.
        rest_weight:      Cost weight pulling each waypoint toward the
                          smoothstep target.
        limit_weight:     Cost weight on joint-limit violation.
        collision_margin: Default clearance (m) below which the collision
                          cost activates; each pair's actual activation is
                          derived from its home-pose clearance (see
                          :func:`almond_axol.kinematics.model.collision_cost_params`),
                          matching the interactive solver's behavior.
        collision_weight: Cost weight on self-collision penalty.
        max_iterations:   IK solver iterations per waypoint.

    Returns:
        A list of full ``(N,)`` joint vectors, one per control tick at
        ``rate`` Hz. Always at least two waypoints long.
    """
    # The projection costs (joint limits, self-collision) index joints the
    # way the pyroki robot does, so the solve runs in that order and the
    # returned waypoints are converted back.
    q_from = solver.to_pyroki_order(q_from)
    q_to = solver.to_pyroki_order(q_to)
    max_dist = float(np.max(np.abs(q_from - q_to)))
    duration = max(max_dist / speed, min_duration)
    n_steps = max(2, round(duration * rate))

    use_collision = solver.robot_coll is not None
    if use_collision:
        starts, widths = collision_cost_params(
            solver.robot, solver.robot_coll, collision_margin
        )
    else:
        starts = widths = np.zeros(0, dtype=np.float32)
    starts_jax, widths_jax = jnp.asarray(starts), jnp.asarray(widths)

    trajectory: list[np.ndarray] = []
    q = q_from.copy()
    for i in range(n_steps):
        t = (i + 1) / n_steps
        alpha = t * t * (3.0 - 2.0 * t)
        q_interp = (q_from * (1.0 - alpha) + q_to * alpha).astype(np.float32)
        result = solve_path_step(
            solver.robot,
            solver.robot_coll,
            jnp.asarray(q_interp),
            jnp.asarray(q),
            rest_weight,
            limit_weight,
            starts_jax,
            widths_jax,
            collision_weight,
            max_iterations,
            use_self_collision=use_collision,
        )
        q = np.asarray(result, dtype=np.float32)
        trajectory.append(solver.from_pyroki_order(q))
    return trajectory
