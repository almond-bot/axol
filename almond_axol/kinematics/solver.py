"""
Standalone bimanual IK solver for the Axol robot.

Uses pyroki + jaxls to solve for joint positions given absolute Cartesian
end-effector poses in the robot's world frame (FLU: +x forward, +y left,
+z up).

All public joint vectors are laid out as the robot itself orders them: the
left arm's 7 joints in :data:`~almond_axol.constants.ARM_JOINTS` order
(shoulder_1 … wrist_3) followed by the right arm's — so ``q[:7]`` /
``q[7:]`` are exactly the arrays ``motion_control`` takes. pyroki orders
the actuated joints its own way internally; the permutation is applied at
every public boundary and callers never see it.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import math

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

from ..constants import (
    ARM_JOINTS,
    AxolModel,
    Joint,
    upper_arm_guard_links,
    urdf_arm_joint_names,
    urdf_body_name,
    urdf_joint_name,
    urdf_path,
)
from ..settings import resolve_gripper, resolve_robot_model
from .aot import ExportedJit
from .collision import ArmBodyCollision
from .config import KinematicsConfig
from .jax_cache import enable_persistent_compilation_cache
from .model import (
    collision_cost_params,
    finger_slide,
    ik_urdf_xml,
    lift_link_shift,
    shared_robot,
    shared_robot_collision,
    shared_urdf,
    upper_arm_guard_floor,
)

_logger = logging.getLogger(__name__)


Pose = tuple[np.ndarray, np.ndarray]
"""A frame as ``(position (3,), rotation (3, 3))`` numpy arrays, world frame
(FLU: +x forward, +y left, +z up).

The format :meth:`KinematicsSolver.fk` returns and :meth:`KinematicsSolver.ik`
takes, so a pose can be read, edited, and solved for without conversion.
"""


@dataclasses.dataclass(frozen=True)
class ElbowSwivel:
    """One arm's elbow swivel (:meth:`KinematicsSolver.elbow_swivel`).

    Attributes:
        shoulder: Shoulder centre, world frame — where the three shoulder
            axes meet; fixed.
        elbow: Elbow position, world frame (as
            :meth:`KinematicsSolver.elbow_positions`).
        wrist_in_mount: Wrist centre — where the forearm roll meets
            ``wrist_2`` — in the gripper *mount* frame at these joints, to
            place it on a target pose.
        direction: Unit world-frame direction the elbow moves in as the arm
            swings with its gripper mount held (either sense; zeros at a
            singular pose).
    """

    shoulder: np.ndarray
    elbow: np.ndarray
    wrist_in_mount: np.ndarray
    direction: np.ndarray


# Convenience aliases for URDF link / joint names. The single source of
# truth for these strings lives in :mod:`almond_axol.constants`; the helpers
# below just compose ``"left_*"`` / ``"right_*"`` from a side-agnostic
# suffix table so renaming the URDF only requires editing one place.
_LEFT_EE = urdf_body_name(Joint.GRIPPER, is_left=True)
_RIGHT_EE = urdf_body_name(Joint.GRIPPER, is_left=False)
_LEFT_ELBOW = urdf_body_name(Joint.ELBOW, is_left=True)
_RIGHT_ELBOW = urdf_body_name(Joint.ELBOW, is_left=False)
_LEFT_SHOULDER = urdf_body_name(Joint.SHOULDER_1, is_left=True)
_RIGHT_SHOULDER = urdf_body_name(Joint.SHOULDER_1, is_left=False)

# Actuated joint names in ARM_JOINTS order (shoulder_1 … wrist_3). Must
# match the ordering assumed by rest_pose / motion_control.
_LEFT_JOINT_NAMES = urdf_arm_joint_names(is_left=True)
_RIGHT_JOINT_NAMES = urdf_arm_joint_names(is_left=False)


# ---------------------------------------------------------------------------
# Bounded manipulability cost
# ---------------------------------------------------------------------------

# Saturation point of the reciprocal manipulability barrier. pyroki's
# manipulability_residual uses 1e-6, which lets the residual's Jacobian grow
# like 1/manip^2 ~ 1e12 near a singular pose (e.g. the all-zero straight-arm
# pose, where the Yoshikawa index is ~1e-16). Rows that large make the float32
# Gauss-Newton normal matrix indefinite after rounding, and cuSolver's
# Cholesky then returns NaN — every LM step is rejected and ik() silently
# returns its seed (CPU LAPACK only survives the same matrix by rounding
# luck). 1e-2 caps the Jacobian factor at 1e4, keeping the worst-case rows
# the same order as the pose cost, while barely changing the cost away from
# singularities (residual 15.4 -> 13.3 at a typical healthy index of 0.065).
_MANIP_BARRIER_EPS = 1e-2


def _manip_yoshikawa(
    cfg: jax.Array, robot: pk.Robot, target_link_index: jax.Array
) -> jax.Array:
    """Yoshikawa manipulability index of one link's translation Jacobian."""
    jacobian = jax.jacfwd(
        lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
    )(cfg)[target_link_index]
    return jnp.sqrt(jnp.maximum(0.0, jnp.linalg.det(jacobian @ jacobian.T)))


def _bounded_manipulability_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    target_link_indices: jax.Array,
    weight: jax.Array | float,
) -> jax.Array:
    """pyroki's manipulability residual with the barrier bounded near singularities."""
    cfg = vals[joint_var]
    manip = jax.vmap(_manip_yoshikawa, in_axes=(None, None, 0))(
        cfg, robot, target_link_indices
    )
    return (weight / (manip + _MANIP_BARRIER_EPS)).flatten()


_bounded_manipulability_cost = jaxls.Cost.factory(_bounded_manipulability_residual)


@dataclasses.dataclass(frozen=True)
class _ArmSwivel:
    """One arm's fixed geometry for :meth:`KinematicsSolver.elbow_swivel`."""

    axes: tuple[tuple[int, np.ndarray], ...]  # (child link, local axis) per joint
    shoulder: np.ndarray  # shoulder centre, world
    wrist_link: int
    wrist_local: np.ndarray  # wrist centre in wrist_link's frame
    elbow_joints: int  # joints upstream of the elbow link


def _pose_np(wxyz_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rotation (3, 3), position (3,))`` of a jaxlie ``wxyz_xyz`` link pose."""
    w, x, y, z = wxyz_xyz[:4]
    rot = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )
    return rot, np.asarray(wxyz_xyz[4:7], np.float64)


def _axis_line(
    poses: np.ndarray, link: int, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World ``(point, direction)`` of a joint axis given its child link's pose."""
    rot, pos = _pose_np(poses[link])
    return pos, rot @ axis


def _axes_meet(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    """Midpoint of the closest approach of two lines (point, direction)."""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        return p1.copy()
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


# ---------------------------------------------------------------------------
# Self-collision cost with per-pair (possibly negative) activation starts
# ---------------------------------------------------------------------------


def shift_links(link_poses: jax.Array, link_shift: jax.Array) -> jax.Array:
    """``link_poses`` (``(*batch, num_links, 7)``, wxyz_xyz) with each link
    moved by its row of ``link_shift``: columns 0-2 a displacement in the
    torso frame (the live lift height, :func:`.model.lift_link_shift`),
    columns 3-5 (when present) one in the link's own frame (the live gripper
    opening, :func:`.model.finger_slide`)."""
    move = link_shift[..., :3]
    if link_shift.shape[-1] == 6:
        move = move + jaxlie.SO3(link_poses[..., :4]).apply(link_shift[..., 3:])
    return link_poses.at[..., 4:].add(move)


def body_collision_distance(
    robot: pk.Robot,
    robot_coll: ArmBodyCollision,
    cfg: jax.Array,
    link_shift: jax.Array | None = None,
) -> jax.Array:
    """Signed distance of each active arm/body pair at ``cfg``.

    ``robot_coll.compute_self_collision_distance``, with each link first
    moved by its row of ``link_shift`` (see :func:`shift_links`): where the
    live lift height puts the parts the lift leaves behind, and the live
    gripper opening the fingers. ``None``: the model as frozen.
    """
    Ts = robot.forward_kinematics(cfg)
    if link_shift is not None:
        Ts = shift_links(Ts, link_shift)
    return robot_coll.distances(Ts)


def _self_collision_residual(
    vals: jaxls.VarValues,
    robot: pk.Robot,
    robot_coll: ArmBodyCollision,
    joint_var: jaxls.Var[jax.Array],
    activation_start: jax.Array,
    ramp_width: jax.Array,
    weight: jax.Array | float,
    link_shift: jax.Array | None = None,
) -> jax.Array:
    """pyroki's self-collision residual with decoupled activation and ramp.

    Per pair: zero for clearance above ``activation_start``, a quartic ramp
    over the ``ramp_width`` below it (C2 at both ends), then the full linear
    slope. The ramp's derivative is smoothstep ``3u² - 2u³``: unlike the old
    quadratic residual, its curvature also reaches zero at the shell edges.
    This prevents the collision boundary itself from injecting a joint-
    acceleration impulse while the arm approaches the base.

    The activation/ramp decoupling lets pairs whose fitted shapes already
    interpenetrate at the home pose use a *negative* activation start
    (activate only when the pose gets closer than home). Pyroki's stock cost
    bottoms out at zero, which would keep such a pair's gradient permanently
    active during ordinary front-of-torso work (see
    :func:`.model.collision_cost_params`). ``link_shift``: see
    :func:`body_collision_distance`.
    """
    cfg = vals[joint_var]
    d = body_collision_distance(robot, robot_coll, cfg, link_shift)
    s = activation_start - d  # violation depth beyond the activation shell
    # Integral of smoothstep(clip(s / width, 0, 1)). At u=1 this is
    # width/2 with unit slope and zero curvature, matching the linear branch
    # exactly through the second derivative.
    u = jnp.clip(s / ramp_width, 0.0, 1.0)
    smooth_ramp = ramp_width * (u**3 - 0.5 * u**4)
    r = jnp.where(
        s < ramp_width,
        smooth_ramp,
        s - 0.5 * ramp_width,
    )
    return (r * weight).flatten()


_self_collision_cost = jaxls.Cost.factory(_self_collision_residual)


@jax.jit
def _base_collision_safe_step(
    robot: pk.Robot,
    robot_coll: ArmBodyCollision,
    q_from: jax.Array,
    q_to: jax.Array,
    clearance_floor: jax.Array,
    max_joint_delta: jax.Array,
    link_shift: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Project a rate-limited joint step along the base-clearance boundary.

    The smooth collision residual steers IK away early, but it remains a soft
    least-squares task. Check the proposed endpoint independently. If it would
    cross a pair's calibrated floor, remove only the inward component of the
    joint step using the clearance Jacobian; tangential motion survives, so
    the arm can slide/reconfigure around the base rather than freezing. A
    negative floor is valid for fitted shapes that overlap, or come within
    the guard's slack, at the physically safe home pose. Both the proposed step and any clearance
    correction are scaled along their full joint-space direction to stay
    within the per-call delta budget and the robot's joint limits, after
    dropping any component that pushes a joint already on its limit outward.
    ``link_shift``: see :func:`body_collision_distance`.
    """

    guard_buffer = jnp.array(5e-4, dtype=q_from.dtype)
    limit_tol = jnp.array(1e-5, dtype=q_from.dtype)

    def bound_step(q_candidate: jax.Array) -> jax.Array:
        """Project outward boundary directions, then shorten the step."""
        delta = q_candidate - q_from

        # A solver proposal can point outside a joint limit when q_from is
        # on that boundary. Remove only that infeasible component so it
        # cannot make the global limit scale zero for every other joint.
        # "On" is within a tolerance: a step scaled to reach a limit lands a
        # rounding error short of it (float32 ulp is ~2.4e-7 near pi), and an
        # exact comparison would leave that sliver of room driving the scale.
        outward = jnp.logical_or(
            jnp.logical_and(
                q_from - robot.joints.lower_limits <= limit_tol,
                delta < 0.0,
            ),
            jnp.logical_and(
                robot.joints.upper_limits - q_from <= limit_tol,
                delta > 0.0,
            ),
        )
        delta = jnp.where(outward, 0.0, delta)

        max_abs = jnp.max(jnp.abs(delta))
        rate_scale = jnp.minimum(
            1.0,
            jnp.maximum(max_joint_delta, 0.0) / jnp.maximum(max_abs, 1e-12),
        )
        delta = delta * rate_scale

        # Find the largest scalar that keeps every component inside its hard
        # joint interval. Scaling the whole vector preserves the tangent
        # projection; clipping individual joints could point it back into the
        # base after the clearance calculation.
        room = jnp.where(
            delta > 0.0,
            robot.joints.upper_limits - q_from,
            q_from - robot.joints.lower_limits,
        )
        limit_scale = jnp.where(
            delta == 0.0,
            1.0,
            jnp.maximum(room, 0.0) / jnp.maximum(jnp.abs(delta), 1e-12),
        )
        delta = delta * jnp.clip(jnp.min(limit_scale), 0.0, 1.0)
        return q_from + delta

    # The caller already rate-limits q_to, but applying the constraints at
    # this boundary makes them part of the projection's contract and also
    # enforces the solver's otherwise-soft joint limits.
    q_to = bound_step(q_to)

    def margin(q: jax.Array) -> jax.Array:
        distances = body_collision_distance(robot, robot_coll, q, link_shift)
        return jnp.min(distances - clearance_floor)

    to_margin = margin(q_to)

    def project_unsafe(_unused: None) -> jax.Array:
        from_margin = margin(q_from)
        delta = q_to - q_from
        gradient = jax.grad(margin)(q_from)
        gradient_sq = jnp.dot(gradient, gradient) + 1e-12
        inward_step = jnp.dot(gradient, delta)
        allowed_step = guard_buffer - from_margin
        correction = jnp.maximum(allowed_step - inward_step, 0.0) / gradient_sq
        q_slide = bound_step(q_from + delta + correction * gradient)
        slide_margin = margin(q_slide)

        def recover_existing(_unused: None) -> jax.Array:
            improving = slide_margin > from_margin
            return jnp.where(improving, q_slide, q_from)

        def keep_safe(_unused: None) -> jax.Array:
            def accept_slide(_unused: None) -> jax.Array:
                return q_slide

            def find_boundary(_unused: None) -> jax.Array:
                # Clearance is mildly nonlinear in joint space. If the
                # first-order tangent projection is still just inside the
                # floor, retain its direction and shorten it to the boundary.
                slide_delta = q_slide - q_from

                def refine(_i: int, bounds: tuple[jax.Array, jax.Array]):
                    low, high = bounds
                    mid = 0.5 * (low + high)
                    q_mid = q_from + mid * slide_delta
                    mid_safe = margin(q_mid) >= 0.0
                    return (
                        jnp.where(mid_safe, mid, low),
                        jnp.where(mid_safe, high, mid),
                    )

                lo, _ = jax.lax.fori_loop(
                    0,
                    8,
                    refine,
                    (
                        jnp.array(0.0, dtype=q_from.dtype),
                        jnp.array(1.0, dtype=q_from.dtype),
                    ),
                )
                return q_from + lo * slide_delta

            return jax.lax.cond(
                slide_margin >= 0.0, accept_slide, find_boundary, operand=None
            )

        return jax.lax.cond(
            from_margin < 0.0, recover_existing, keep_safe, operand=None
        )

    guard_active = to_margin < 0.0
    q_safe = jax.lax.cond(
        guard_active,
        project_unsafe,
        lambda _unused: q_to,
        operand=None,
    )
    return q_safe, guard_active


# ---------------------------------------------------------------------------
# JIT-compiled core solve
# ---------------------------------------------------------------------------


# Full limit-approach damping applies when a joint moves toward its nearby
# limit faster than this many radians per solve step (~0.24 rad/s at 120 Hz);
# slower approaches get proportionally less.
_LIMIT_GATE_STEP = 2e-3


def _project_elbow(
    elbow: jaxlie.SE3, shoulder: jax.Array, current_elbow_pos: jax.Array
) -> jaxlie.SE3:
    """Project an elbow target onto the robot's reachable elbow sphere.

    The operator's elbow (different arm proportions, position-multiplier
    scaling) generally lies off the sphere the robot's elbow actually lives on
    (|shoulder->elbow| at the current configuration, measured from FK). The
    radial component of the raw target is unreachable by construction and
    would inject a permanent residual that fights the EE cost through the
    shared shoulder joints; only the swivel direction is kept. ``shoulder``
    is the shoulder's centre, where its three axes meet: the radius is fixed
    about it, and box mode's swivel hints (already on the sphere) pass
    through unchanged.
    """
    r = jnp.linalg.norm(current_elbow_pos - shoulder)
    d = elbow.translation() - shoulder
    n = jnp.linalg.norm(d)
    proj = shoulder + d * (r / jnp.maximum(n, 1e-6))
    return jaxlie.SE3.from_rotation_and_translation(elbow.rotation(), proj)


def _clamp_to_limits_np(
    robot: pk.Robot, q_from: np.ndarray, q_to: np.ndarray
) -> np.ndarray:
    """Clip the step ``q_from -> q_to`` so every joint stays inside its limits.

    Enforces the solver's otherwise-soft joint limits on the collision-free
    path (with the collision guard on, ``bound_step`` inside
    :func:`_base_collision_safe_step` does this). Joints are clipped
    *individually*: a joint that has reached its limit parks there while the
    others keep moving. Scaling the whole vector instead — as ``bound_step``
    must, to keep its clearance projection tangent — would freeze the entire
    arm the moment one joint touches a limit, since the soft limit cost keeps
    nudging that joint outward by a hair on every solve.
    """
    lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
    upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
    # A joint that starts outside its interval (hand-moved, measured offset)
    # isn't yanked back in one step: clip to the interval widened to include
    # q_from, so it can only move back toward its range, never further out.
    lo = np.minimum(lower, q_from)
    hi = np.maximum(upper, q_from)
    return np.clip(q_to, lo, hi).astype(np.float32)


@functools.partial(jax.jit, static_argnames=("max_iterations", "use_self_collision"))
def _solve_ik(
    robot: pk.Robot,
    robot_coll: ArmBodyCollision | None,
    target_L: jaxlie.SE3 | None,
    target_R: jaxlie.SE3 | None,
    L_ee_idx: jax.Array,
    R_ee_idx: jax.Array,
    elbow_L: jaxlie.SE3 | None,
    elbow_R: jaxlie.SE3 | None,
    L_elbow_idx: jax.Array,
    R_elbow_idx: jax.Array,
    q_current: jax.Array,
    q_prev: jax.Array,
    posture_pose: jax.Array,
    left_joint_idx: jax.Array,
    right_joint_idx: jax.Array,
    shoulder_L: jax.Array,
    shoulder_R: jax.Array,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
    posture_weight: float,
    manipulability_weight: float,
    limit_weight: float,
    self_collision_start: jax.Array,
    self_collision_ramp: jax.Array,
    self_collision_weight: float,
    elbow_weight_L: float,
    elbow_weight_R: float,
    manip_damping_threshold: float,
    manip_damping_boost: float,
    limit_damping_margin: float,
    max_iterations: int,
    cost_tolerance: float,
    lambda_initial: float,
    lambda_factor: float,
    use_self_collision: bool,
    link_shift: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    JointVar = robot.joint_var_cls

    # Adaptive damping (Chiaverini-style damped least squares): as an arm's
    # translational manipulability at the seed drops toward the singular
    # boundary, its joints' rest-cost weight ramps up quadratically, so the
    # solver takes small conservative steps instead of thrashing between
    # rank-deficient directions. Fully off above the threshold, so normal
    # teleop is unaffected.
    def _ramp(m: jax.Array) -> jax.Array:
        return jnp.square(
            jnp.clip(1.0 - m / jnp.maximum(manip_damping_threshold, 1e-9), 0.0, 1.0)
        )

    manip_L = _manip_yoshikawa(q_current, robot, L_ee_idx)
    manip_R = _manip_yoshikawa(q_current, robot, R_ee_idx)
    rest_w = jnp.full(q_current.shape, rest_weight, dtype=jnp.float32)
    rest_w = rest_w.at[left_joint_idx].add(manip_damping_boost * _ramp(manip_L))
    rest_w = rest_w.at[right_joint_idx].add(manip_damping_boost * _ramp(manip_R))

    # Per-joint damping near joint limits, gated on approach direction: the
    # limit cost is a hinge (e.g. shoulder_2 at -pi with the arm overhead, or
    # the elbow's 0-rad limit at full extension), and an undamped approach
    # slams into it. Damping is applied only to a joint *moving toward* its
    # nearby limit — a joint parked against a limit by task pressure (common
    # for a wrist during ordinary manipulation) is left free, otherwise the
    # damping fights the task and rails the step limiter.
    dist_lo = q_current - robot.joints.lower_limits
    dist_hi = robot.joints.upper_limits - q_current
    dist_to_limit = jnp.minimum(dist_lo, dist_hi)
    limit_ramp = jnp.square(
        jnp.clip(
            1.0 - dist_to_limit / jnp.maximum(limit_damping_margin, 1e-9), 0.0, 1.0
        )
    )
    vel = q_current - q_prev  # rad per solve step
    toward = jnp.where(dist_lo < dist_hi, -vel, vel)
    gate = jnp.clip(toward / _LIMIT_GATE_STEP, 0.0, 1.0)
    rest_w = rest_w + manip_damping_boost * limit_ramp * gate

    # Elbow targets: keep only the swivel direction (see _project_elbow).
    if elbow_L is not None or elbow_R is not None:
        fk_cur = robot.forward_kinematics(q_current)
        if elbow_L is not None:
            elbow_L = _project_elbow(
                elbow_L, shoulder_L, jaxlie.SE3(fk_cur[L_elbow_idx]).translation()
            )
        if elbow_R is not None:
            elbow_R = _project_elbow(
                elbow_R, shoulder_R, jaxlie.SE3(fk_cur[R_elbow_idx]).translation()
            )

    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_current, weight=rest_w),
        pk.costs.rest_cost(JointVar(0), rest_pose=posture_pose, weight=posture_weight),
        _bounded_manipulability_cost(
            robot,
            JointVar(0),
            jnp.array([L_ee_idx, R_ee_idx], dtype=jnp.int32),
            weight=manipulability_weight,
        ),
    ]

    if target_L is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                target_L,
                jnp.array(L_ee_idx, dtype=jnp.int32),
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            )
        )

    if target_R is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                target_R,
                jnp.array(R_ee_idx, dtype=jnp.int32),
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            )
        )

    if elbow_L is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                elbow_L,
                jnp.array(L_elbow_idx, dtype=jnp.int32),
                pos_weight=elbow_weight_L,
                ori_weight=0.0,
            )
        )

    if elbow_R is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                elbow_R,
                jnp.array(R_elbow_idx, dtype=jnp.int32),
                pos_weight=elbow_weight_R,
                ori_weight=0.0,
            )
        )

    costs.append(pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight))
    if use_self_collision:
        assert robot_coll is not None
        costs.append(
            _self_collision_cost(
                robot,
                robot_coll,
                JointVar(0),
                activation_start=self_collision_start,
                ramp_width=self_collision_ramp,
                weight=self_collision_weight,
                link_shift=link_shift,
            )
        )

    var_joints = JointVar(jnp.array([0]))
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(q_current[jnp.newaxis, :])]
    )
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    analyzed = problem.analyze()
    solution_vals, summary = analyzed.solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=lambda_initial,
            lambda_factor=lambda_factor,
        ),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=cost_tolerance,
        ),
        return_summary=True,
    )
    # cost_history holds each iteration's proposed cost; a NaN entry means a
    # step proposal evaluated to NaN (numerically degenerate problem), which
    # jaxls silently rejects. Surfaced so ik() can warn instead of freezing.
    return solution_vals[var_joints][0], summary.cost_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_reach(
    pos: np.ndarray, center: np.ndarray, soft_start: float, cap: float
) -> np.ndarray:
    """Softly saturate an EE target's distance from ``center`` (the shoulder).

    Identity below ``soft_start``; beyond it the radial distance follows a
    tanh saturation that approaches (never reaches) ``cap``. The mapping is
    C1-smooth — unit slope at the knee — so an operator sweeping through the
    boundary produces no velocity discontinuity in the target, unlike a hard
    radial projection. Keeping the cap below the straight-elbow distance means
    the solver is never dragged onto the extension singularity (where the
    elbow's 0-rad joint limit, the Jacobian rank drop, and the 1/manipulability
    barrier all coincide).
    """
    d = pos - center
    dist = float(np.linalg.norm(d))
    if dist <= soft_start:
        return pos
    span = max(cap - soft_start, 1e-6)
    new_dist = soft_start + span * math.tanh((dist - soft_start) / span)
    return (center + d * (new_dist / dist)).astype(np.float32)


def _rot_3x3_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → unit quaternion (w, x, y, z), float32.

    Pure NumPy (Shepperd method) — avoids JAX dispatch overhead outside JIT.
    """
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        r = np.sqrt(t + 1.0)
        s = 0.5 / r
        return np.array(
            [
                0.5 * r,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ],
            np.float32,
        )
    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        r = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (R[2, 1] - R[1, 2]) * s,
                0.5 * r,
                (R[0, 1] + R[1, 0]) * s,
                (R[0, 2] + R[2, 0]) * s,
            ],
            np.float32,
        )
    if R[1, 1] >= R[2, 2]:
        r = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (R[0, 2] - R[2, 0]) * s,
                (R[0, 1] + R[1, 0]) * s,
                0.5 * r,
                (R[1, 2] + R[2, 1]) * s,
            ],
            np.float32,
        )
    r = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    s = 0.5 / r
    return np.array(
        [
            (R[1, 0] - R[0, 1]) * s,
            (R[0, 2] + R[2, 0]) * s,
            (R[1, 2] + R[2, 1]) * s,
            0.5 * r,
        ],
        np.float32,
    )


def _np_to_se3(pos: np.ndarray, rot_3x3: np.ndarray) -> jaxlie.SE3:
    """Construct SE3 from numpy pos + rot_3x3 at the JAX boundary."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=jnp.asarray(_rot_3x3_to_wxyz(rot_3x3))),
        jnp.asarray(pos, dtype=jnp.float32),
    )


def _pos3_to_se3(pos: np.ndarray) -> jaxlie.SE3:
    """Convert a (3,) position array to an identity-rotation SE3."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)),
        jnp.asarray(pos, dtype=jnp.float32),
    )


def _se3_to_pose(se3: jaxlie.SE3) -> Pose:
    """Convert an SE3 to the numpy ``(pos, rot_3x3)`` format :meth:`KinematicsSolver.ik` takes."""
    return (
        np.asarray(se3.translation(), dtype=np.float32),
        np.asarray(se3.rotation().as_matrix(), dtype=np.float32),
    )


def _quat_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    """``(M, 4)`` unit quaternions (w, x, y, z) → ``(M, 3, 3)`` rotations."""
    w, x, y, z = (np.asarray(wxyz, dtype=np.float64)[:, i] for i in range(4))
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        axis=1,
    )


# The jitted solve and safe step, exported once per graph shape so a new
# process deserializes them instead of re-tracing (see .aot).
_SOLVE_IK = ExportedJit(
    _solve_ik, "solve_ik", static_argnames=("max_iterations", "use_self_collision")
)
_SAFE_STEP = ExportedJit(_base_collision_safe_step, "safe_step")
_COLLISION_DISTANCE = jax.jit(body_collision_distance)


@jax.jit
def _arm_shift_jacobian(
    robot: pk.Robot,
    robot_coll: ArmBodyCollision,
    cfg: jax.Array,
    link_shift: jax.Array,
    arm_links: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Each pair's signed distance at ``cfg`` and its derivative with respect
    to moving each arm as a rigid whole (``(num_pairs,)`` and
    ``(num_pairs, 2, 3)``): ``arm_links`` is ``(2, num_links)``, 1 on the
    links of the left and the right arm. The derivative of a pair is the
    direction (torso frame) that takes that arm out of it."""

    def distances(move: jax.Array) -> jax.Array:
        shift = link_shift.at[:, :3].add(arm_links.T @ move)
        return body_collision_distance(robot, robot_coll, cfg, shift)

    zero = jnp.zeros((2, 3), dtype=cfg.dtype)
    return distances(zero), jax.jacfwd(distances)(zero)


# How far outside a pair's soft-cost activation the proximity cue starts
# (see KinematicsSolver.collision_proximity). Capped per pair at its home
# clearance, so a pose as clear as home never reads as near.
_PROXIMITY_LEAD = 0.02
# Narrowest distance (m) over which a pair's proximity cue rises from 0 to 1
# (see KinematicsSolver.set_proximity_reference).
_PROXIMITY_SPAN_MIN = 0.01
# Lift clearance (see lift_clear_step): how far (m) past the soft-cost
# activation shell an arm is moved, and the most the deck may rise (m) per
# step of lift_clear_pose's replay of the lift lowering.
_LIFT_CLEAR_MARGIN = 0.003
_LIFT_REPLAY_STEP = 0.01


# ---------------------------------------------------------------------------
# KinematicsSolver
# ---------------------------------------------------------------------------


class KinematicsSolver:
    """Bimanual IK solver for the Axol robot.

    Loads the bundled URDF, builds a pyroki + jaxls solver, and resolves
    absolute Cartesian end-effector poses (world frame, FLU: +x forward,
    +y left, +z up) to joint angles. JIT compilation is triggered during
    ``__init__`` so the first call to :meth:`ik` is fast.

    Every joint vector this class takes or returns is laid out left arm
    then right arm, each in :data:`~almond_axol.constants.ARM_JOINTS` order
    (shoulder_1 … wrist_3) — the same order ``motion_control`` and
    ``rest_pose`` use, so ``q[:7]`` is the left arm and ``q[7:]`` the right.

    Args:
        config: Solver cost weights and parameters.

    Example::

        solver = KinematicsSolver()
        q = np.zeros(solver.num_joints, dtype=np.float32)
        pos = np.array([0.3, 0.2, 0.4], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)
        q = solver.ik(q, left_pose=(pos, rot))
        left_arm, right_arm = q[:7], q[7:]  # ARM_JOINTS order

        # fk returns the same (pos, rot_3x3) format ik takes, so nudging a
        # pose is a round trip:
        (l_pos, l_rot), _ = solver.fk(q)
        q = solver.ik(q, left_pose=(l_pos + np.array([0.0, 0.0, 0.05]), l_rot))
    """

    def __init__(self, config: KinematicsConfig = KinematicsConfig()) -> None:
        """Build (or reuse) the robot model and warm up JIT.

        The URDF, pyroki robot, and collision model are built once per process
        and shared across instances, so only the first solver pays for them —
        and for the JAX trace + jaxls problem analysis its warm-up solve
        triggers. Each instance resolves link and joint indices, computes fixed
        shoulder positions, and runs a dummy solve so the first real call to
        :meth:`ik` is fast.

        Args:
            config: Cost weights and solver parameters. ``config.robot_model``
                picks the Axol version's URDF (mobile when Jelly is enabled,
                else ``config.mount``, if ``None``).
        """
        self.config = config
        self.robot_model = resolve_robot_model(config.robot_model, mount=config.mount)
        # The fitted gripper (the mobile Axol has a URDF per gripper).
        self.gripper = (
            resolve_gripper(config.gripper)
            if self.robot_model is AxolModel.MOBILE
            else "parallel"
        )

        enable_persistent_compilation_cache()

        # One robot model per process (see .model): reusing the same pytree —
        # including pyroki's per-instance JointVar class, a static field —
        # lets every solver after the first hit the in-memory jit cache
        # instead of re-tracing and re-running jaxls analysis.
        self.robot = shared_robot(self.robot_model, gripper=self.gripper)
        # What the exported programs were traced for (see .aot): the model
        # IK loads, beyond what its arrays' shapes say.
        self._export_fingerprint = hashlib.sha256(
            ik_urdf_xml(urdf_path(self.robot_model, gripper=self.gripper)).encode()
        ).hexdigest()
        # Base collision detection is opt-in (config.self_collision): when
        # off, no collision model is built at all — the soft cost is left out
        # of the solve graph and the hard base guard below is skipped, so the
        # only geometric constraint on the arms is their joint limits.
        self.robot_coll: ArmBodyCollision | None = (
            shared_robot_collision(self.robot_model, gripper=self.gripper)
            if config.self_collision
            else None
        )
        # The lift (see set_lift): IK freezes it fully raised; the collision
        # checks move what it leaves behind by its live height. The fingers
        # (see set_gripper): IK freezes their joints at zero; the checks move
        # them by the live opening, open until the first call.
        self._lift_shift, self._lift_travel = lift_link_shift(
            self.robot_model, gripper=self.gripper
        )
        self._lift_lowered = 0.0
        (
            self._finger_axis,
            self._finger_side,
            self._finger_closed,
            self._finger_stroke,
        ) = finger_slide(self.robot_model, gripper=self.gripper)
        self._grip = (1.0, 1.0)
        self._link_shift = self._make_link_shift()
        if self.robot_coll is not None:
            starts, widths = self.collision_params(config.self_collision_margin)
        else:
            _logger.info("Self-collision (arm/base) detection is off.")
            starts = np.zeros(0, dtype=np.float32)
            widths = np.zeros(0, dtype=np.float32)
        self._collision_starts = jnp.asarray(starts)
        self._collision_ramps = jnp.asarray(widths)
        # A soft cost normally turns every arm/base pair away. Independently
        # hard-stop the physical contact pair found in the recorded cross-body
        # run: either upper arm (e1) against the body (see
        # upper_arm_guard_floor for the threshold).
        clearance_floor = np.full_like(starts, -np.inf)
        if self.robot_coll is not None:
            q_home = jnp.zeros(self.robot.joints.num_actuated_joints)
            # The home clearance over the fingers closed and open, as in
            # collision_params (the frozen finger pose counts too).
            home_distances = np.asarray(
                self.robot_coll.compute_self_collision_distance(self.robot, q_home)
            )
            for grip in ((0.0, 0.0), (1.0, 1.0)):
                home_distances = np.minimum(
                    home_distances,
                    np.asarray(
                        _COLLISION_DISTANCE(
                            self.robot,
                            self.robot_coll,
                            q_home,
                            self._make_link_shift(0.0, grip),
                        )
                    ),
                )
            torso = upper_arm_guard_links(self.robot_model)
            for k, (i, j) in enumerate(
                zip(
                    np.asarray(self.robot_coll.active_idx_i),
                    np.asarray(self.robot_coll.active_idx_j),
                )
            ):
                a = self.robot_coll.link_names[int(i)]
                b = self.robot_coll.link_names[int(j)]
                arm_link = b if a in torso else a if b in torso else ""
                if arm_link.endswith("_e1"):
                    clearance_floor[k] = upper_arm_guard_floor(home_distances[k])
        self._base_clearance_floor = jnp.asarray(clearance_floor)
        self._base_collision_guard_active = False
        # Proximity cue (see collision_proximity): each pair reads 0 at or
        # beyond its onset and 1 once the soft cost reaches full slope. The
        # onset sits _PROXIMITY_LEAD outside the activation start, but never
        # outside the home clearance — many base pairs activate just 2 mm
        # inside home, and the arms hanging at home must read as clear (the
        # teleop rest pose is added by set_proximity_reference).
        self._proximity_onset = np.zeros(0, dtype=np.float32)
        self._proximity_full = np.zeros(0, dtype=np.float32)
        self._proximity_side: dict[str, np.ndarray] = {}
        if self.robot_coll is not None:
            self._proximity_onset = np.minimum(
                starts + _PROXIMITY_LEAD, home_distances
            ).astype(np.float32)
            self._proximity_full = (starts - widths).astype(np.float32)
            pair_links = [
                (self.robot_coll.link_names[int(i)], self.robot_coll.link_names[int(j)])
                for i, j in zip(
                    self.robot_coll.active_idx_i, self.robot_coll.active_idx_j
                )
            ]
            self._proximity_side = {
                side: np.array(
                    [
                        a.startswith(f"{side}_") or b.startswith(f"{side}_")
                        for a, b in pair_links
                    ],
                    dtype=bool,
                )
                for side in ("left", "right")
            }

        # Lift contact (see lift_contact): the pairs between an arm and a
        # body part the lift leaves behind (Jelly's deck, base and lower
        # stage), and each arm's links.
        self._lift_pairs = np.zeros(len(starts), dtype=bool)
        self._lift_pair_side: dict[str, np.ndarray] = {}
        self._lift_starts = np.asarray(starts, dtype=np.float32)
        self._arm_links = jnp.zeros((2, len(self.robot.links.names)), jnp.float32)
        if self.robot_coll is not None and self._lift_travel > 0.0:
            moves = np.abs(np.asarray(self._lift_shift)).max(axis=1) > 1e-6
            link_names = self.robot_coll.link_names
            idx_i = np.asarray(self.robot_coll.active_idx_i)
            idx_j = np.asarray(self.robot_coll.active_idx_j)
            self._lift_pairs = moves[idx_i] | moves[idx_j]
            for side in ("left", "right"):
                on_side = np.array(
                    [
                        link_names[int(i)].startswith(f"{side}_")
                        or link_names[int(j)].startswith(f"{side}_")
                        for i, j in zip(idx_i, idx_j)
                    ],
                    dtype=bool,
                )
                self._lift_pair_side[side] = self._lift_pairs & on_side
            self._arm_links = jnp.asarray(
                [
                    [n.startswith(f"{side}_") for n in self.robot.links.names]
                    for side in ("left", "right")
                ],
                dtype=jnp.float32,
            )

        names = self.robot.links.names
        self.l_ee_idx = names.index(_LEFT_EE)
        self.r_ee_idx = names.index(_RIGHT_EE)
        self.l_elbow_idx = names.index(_LEFT_ELBOW)
        self.r_elbow_idx = names.index(_RIGHT_ELBOW)

        self._l_ee_idx_jax = jnp.asarray(self.l_ee_idx, dtype=jnp.int32)
        self._r_ee_idx_jax = jnp.asarray(self.r_ee_idx, dtype=jnp.int32)
        self._l_elbow_idx_jax = jnp.asarray(self.l_elbow_idx, dtype=jnp.int32)
        self._r_elbow_idx_jax = jnp.asarray(self.r_elbow_idx, dtype=jnp.int32)

        # Shoulder positions are fixed in world frame (independent of joint angles)
        L_sh_idx = names.index(_LEFT_SHOULDER)
        R_sh_idx = names.index(_RIGHT_SHOULDER)
        fk0 = self.robot.forward_kinematics(
            jnp.zeros(self.robot.joints.num_actuated_joints)
        )
        self._left_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[L_sh_idx]).translation(), dtype=np.float32
        )
        self._right_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[R_sh_idx]).translation(), dtype=np.float32
        )
        self._left_shoulder_jax = jnp.asarray(self._left_shoulder_pos)
        self._right_shoulder_jax = jnp.asarray(self._right_shoulder_pos)

        # The elbow's swivel (see elbow_swivel): each arm's joint axes (the
        # child link and the axis in it, ARM_JOINTS order), its shoulder
        # centre — the three shoulder axes meet at one point, fixed in the
        # world — and its wrist centre, where the forearm roll (wrist_1)
        # meets wrist_2, fixed in wrist_2's link.
        urdf = shared_urdf(self.robot_model, gripper=self.gripper)
        fk0_np = np.asarray(fk0, np.float64)
        self._swivel: dict[str, _ArmSwivel] = {}
        for side, is_left in (("left", True), ("right", False)):
            axes = []
            for joint in ARM_JOINTS:
                j = urdf.joint_map[urdf_joint_name(joint, is_left=is_left)]
                axes.append((names.index(j.child), np.asarray(j.axis, np.float64)))
            lines = [_axis_line(fk0_np, link, axis) for link, axis in axes]
            shoulder = _axes_meet(*lines[0], *lines[1])
            wrist = _axes_meet(*lines[4], *lines[5])
            w2_link = axes[5][0]
            rot, pos = _pose_np(fk0_np[w2_link])
            self._swivel[side] = _ArmSwivel(
                axes=tuple(axes),
                shoulder=shoulder,
                wrist_link=w2_link,
                wrist_local=rot.T @ (wrist - pos),
                elbow_joints=ARM_JOINTS.index(Joint.ELBOW) + 1,
            )
        self._left_swivel_jax = jnp.asarray(self._swivel["left"].shoulder, jnp.float32)
        self._right_swivel_jax = jnp.asarray(
            self._swivel["right"].shoulder, jnp.float32
        )

        # Public joint vectors are left-then-right in ARM_JOINTS order (the
        # robot's own ordering); pyroki reorders the actuated joints
        # internally (topologically — alphabetical for this URDF). The
        # permutation between the two is applied at every public boundary,
        # so callers never deal with pyroki's ordering.
        canonical = _LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES
        actuated = list(self.robot.joints.actuated_names)
        self._pyroki_index = np.array(
            [actuated.index(n) for n in canonical], dtype=np.intp
        )

        # Per-arm joint indices in pyroki's order, for the internal costs.
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self._left_idx_jax = jnp.asarray(
            [name_to_idx[n] for n in _LEFT_JOINT_NAMES], dtype=jnp.int32
        )
        self._right_idx_jax = jnp.asarray(
            [name_to_idx[n] for n in _RIGHT_JOINT_NAMES], dtype=jnp.int32
        )

        # The public layout makes these trivial; kept because splitting a
        # full vector into per-arm arrays reads better through them.
        n_arm = len(_LEFT_JOINT_NAMES)
        self.left_indices = list(range(n_arm))
        self.right_indices = list(range(n_arm, 2 * n_arm))

        self._posture_pose = jnp.zeros(
            self.robot.joints.num_actuated_joints, dtype=jnp.float32
        )
        self._last_solve_had_nan = False

        # Seed of the previous ik() call: per-joint velocity estimate for the
        # limit-approach damping gate. None until the first call (gate off).
        self._q_prev: np.ndarray | None = None

        self._warmup()
        self._base_collision_guard_active = False

    def set_posture_pose(self, q: np.ndarray) -> None:
        """Set the global preferred posture used as a persistent attractor.

        Args:
            q: Full ``(N,)`` joint array in radians (same ordering as :meth:`ik`).
        """
        self._posture_pose = jnp.asarray(self.to_pyroki_order(q), dtype=jnp.float32)

    def set_lift(self, raised: float) -> None:
        """Place the parts the lift leaves behind for the live lift height.

        IK freezes the lift fully raised and solves in the torso frame, which
        rides the lift with the arms; lowering it brings the deck and the
        lower column up toward them. ``raised`` is the height as a fraction
        of the lift's travel (1 = fully raised, 0 = fully lowered, as the
        Jelly lift reports it). Until the first call the lift is taken as
        fully raised. No effect on a version without a lift.
        """
        if self._lift_travel <= 0.0:
            return
        fraction = min(max(float(raised), 0.0), 1.0)
        lowered = (1.0 - fraction) * self._lift_travel
        if abs(lowered - self._lift_lowered) < 1e-4:
            return
        self._lift_lowered = lowered
        self._link_shift = self._make_link_shift()

    def set_gripper(self, left: float, right: float) -> None:
        """Place the fingers for the live gripper opening.

        IK freezes the finger joints at zero; each finger slides along its
        joint axis to where the opening puts it (:func:`.model.finger_slide`).
        ``left`` / ``right`` are the openings as fractions (0 = closed, 1 =
        fully open, as teleop commands them). Until the
        first call the grippers are taken as fully open. No effect on a
        gripper without sliding fingers (the parcel gripper).
        """
        if self._finger_stroke == 0.0:
            return
        grip = tuple(min(max(float(g), 0.0), 1.0) for g in (left, right))
        if max(abs(a - b) for a, b in zip(grip, self._grip)) < 1e-3:
            return
        self._grip = grip
        self._link_shift = self._make_link_shift()

    def collision_params(self, margin: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-pair ``(activation_start, ramp_width)`` of the collision cost
        for a default ``margin`` (:func:`.model.collision_cost_params`), with
        the home clearance taken over the fingers closed and open: whichever
        the gripper holds at rest, no pair is active there."""
        assert self.robot_coll is not None
        return collision_cost_params(
            self.robot,
            self.robot_coll,
            margin,
            home_shifts=(
                self._make_link_shift(0.0, (0.0, 0.0)),
                self._make_link_shift(0.0, (1.0, 1.0)),
            ),
        )

    def _make_link_shift(
        self, lowered: float | None = None, grip: tuple[float, float] | None = None
    ) -> jax.Array:
        """The ``(num_links, 6)`` link shift (see :func:`shift_links`) for a
        lift lowered by ``lowered`` (m) and gripper openings ``grip``
        (fractions); the live values by default."""
        lowered = self._lift_lowered if lowered is None else lowered
        grip = self._grip if grip is None else grip
        opening = np.where(
            self._finger_side >= 0,
            np.asarray(grip, dtype=np.float32)[np.maximum(self._finger_side, 0)],
            0.0,
        )
        q = self._finger_closed + opening * self._finger_stroke
        local = self._finger_axis * np.where(self._finger_side >= 0, q, 0.0)[:, None]
        return jnp.asarray(
            np.concatenate([self._lift_shift * lowered, local], axis=1),
            dtype=jnp.float32,
        )

    @property
    def lift_lowered(self) -> float:
        """How far below fully raised (m) the collision checks place the lift."""
        return self._lift_lowered

    @property
    def grip(self) -> tuple[float, float]:
        """The gripper openings (fractions) the collision checks place the
        fingers at (see :meth:`set_gripper`)."""
        return self._grip

    @property
    def link_shift(self) -> jax.Array:
        """Per-link displacement (``(num_links, 6)``, see :func:`shift_links`)
        the live lift height and gripper opening put on the collision model
        (see :meth:`set_lift` and :meth:`set_gripper`)."""
        return self._link_shift

    @property
    def posture_pose(self) -> np.ndarray:
        """The global preferred posture, as set by :meth:`set_posture_pose`.

        Lets a caller that sweeps the attractor (e.g. Cartesian path planning
        in :mod:`almond_axol.kinematics.path`) restore what it found.
        """
        return self.from_pyroki_order(np.asarray(self._posture_pose, dtype=np.float32))

    # -- Properties ----------------------------------------------------------

    @property
    def shoulder_positions(self) -> dict[str, np.ndarray]:
        """Base-frame shoulder origins the reach soft-clamp is centred on.

        Fixed by the URDF (the shoulder body does not move with the arm
        joints); the same centres :meth:`ik` clamps its targets around.
        """
        return {
            "left": np.asarray(self._left_shoulder_pos, dtype=np.float64),
            "right": np.asarray(self._right_shoulder_pos, dtype=np.float64),
        }

    @property
    def joint_names(self) -> list[str]:
        """All actuated joint names — left arm then right arm, ARM_JOINTS order."""
        return list(_LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES)

    @property
    def num_joints(self) -> int:
        """Total number of actuated joints across both arms."""
        return int(self._pyroki_index.size)

    # -- Joint-order conversion ----------------------------------------------

    def to_pyroki_order(self, q: np.ndarray) -> np.ndarray:
        """Reorder a public joint vector into the order ``self.robot`` uses.

        Only needed when driving the pyroki ``self.robot`` /
        ``self.robot_coll`` objects directly (their joint limits, forward
        kinematics, and collision pairs are indexed in pyroki's own actuated
        order); every method on this class converts internally.

        """
        q = np.asarray(q, dtype=np.float32)
        out = np.empty_like(q)
        out[self._pyroki_index] = q
        return out

    def from_pyroki_order(self, q: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_pyroki_order`."""
        return np.asarray(q, dtype=np.float32)[self._pyroki_index]

    # -- Public interface ----------------------------------------------------

    def fk(self, q: np.ndarray) -> tuple[Pose, Pose]:
        """Compute end-effector poses from joint positions.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Tuple ``(left_pose, right_pose)``, each a ``(pos (3,), rot (3, 3))``
            pair of numpy arrays in the robot's world frame (FLU: +x forward,
            +y left, +z up) — the same format :meth:`ik` takes, so a pose can
            be read, nudged, and solved for directly::

                (l_pos, l_rot), _ = solver.fk(q)
                q = solver.ik(q, left_pose=(l_pos + delta, l_rot))

        The returned pose is the gripper *mount* frame — for the point the
        fingers close on, see :func:`almond_axol.kinematics.path.tip_poses`.
        """
        fk = self.robot.forward_kinematics(jnp.asarray(self.to_pyroki_order(q)))
        return (
            _se3_to_pose(jaxlie.SE3(fk[self.l_ee_idx])),
            _se3_to_pose(jaxlie.SE3(fk[self.r_ee_idx])),
        )

    def ee_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched :meth:`fk` positions: ``(M, N)`` joint rows → the left and
        right gripper-mount positions, ``(M, 3)`` each (world frame). Jitted
        once per solver, so a few rows per control tick are cheap."""
        fk = self.__dict__.get("_fk_batch")
        if fk is None:
            fk = self._fk_batch = jax.jit(jax.vmap(self.robot.forward_kinematics))
        rows = np.asarray(q, dtype=np.float32).reshape(-1, len(self._pyroki_index))
        pyroki = np.empty_like(rows)
        pyroki[:, self._pyroki_index] = rows  # to_pyroki_order, per row
        poses = np.asarray(fk(jnp.asarray(pyroki)))
        # jaxlie SE3 parameters: (qw, qx, qy, qz, x, y, z).
        return poses[:, self.l_ee_idx, 4:], poses[:, self.r_ee_idx, 4:]

    def ee_rotations(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched :meth:`fk` orientations: ``(M, N)`` joint rows → the left
        and right gripper-mount rotations (world ← mount), ``(M, 3, 3)``
        each. Shares :meth:`ee_positions`' jitted FK."""
        fk = self.__dict__.get("_fk_batch")
        if fk is None:
            fk = self._fk_batch = jax.jit(jax.vmap(self.robot.forward_kinematics))
        rows = np.asarray(q, dtype=np.float32).reshape(-1, len(self._pyroki_index))
        pyroki = np.empty_like(rows)
        pyroki[:, self._pyroki_index] = rows
        poses = np.asarray(fk(jnp.asarray(pyroki)))
        return (
            _quat_to_matrix(poses[:, self.l_ee_idx, :4]),
            _quat_to_matrix(poses[:, self.r_ee_idx, :4]),
        )

    def elbow_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-frame elbow positions from joint positions.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Tuple ``(left, right)`` of ``(3,)`` numpy arrays — the elbow link
            origins in the robot's world frame.
        """
        fk = self.robot.forward_kinematics(jnp.asarray(self.to_pyroki_order(q)))
        return (
            np.asarray(jaxlie.SE3(fk[self.l_elbow_idx]).translation(), np.float32),
            np.asarray(jaxlie.SE3(fk[self.r_elbow_idx]).translation(), np.float32),
        )

    def elbow_swivel(self, q: np.ndarray) -> dict[str, ElbowSwivel]:
        """Each arm's elbow swivel at joint positions ``q``.

        With its gripper mount held, a 7-joint arm still has one free
        motion: the elbow swinging round. Each arm's shoulder axes meet at
        one point and its forearm roll meets ``wrist_2`` at another, and
        the swing is roughly a circle about the line between the two — but
        only roughly, since ``wrist_3`` is offset from that wrist centre
        and has to turn with it, so the exact motion is
        :attr:`ElbowSwivel.direction`, from the arm's Jacobian. A wrist
        turned far from straight (box mode's angled grasp) bends the motion
        well off the circle.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Per side (``"left"``, ``"right"``), an :class:`ElbowSwivel`.
        """
        poses = np.asarray(
            self.robot.forward_kinematics(jnp.asarray(self.to_pyroki_order(q))),
            np.float64,
        )
        out: dict[str, ElbowSwivel] = {}
        for side, ee, elbow_idx in (
            ("left", self.l_ee_idx, self.l_elbow_idx),
            ("right", self.r_ee_idx, self.r_elbow_idx),
        ):
            geo = self._swivel[side]
            ee_rot, ee_pos = _pose_np(poses[ee])
            elbow = _pose_np(poses[elbow_idx])[1]
            w_rot, w_pos = _pose_np(poses[geo.wrist_link])
            wrist = w_pos + w_rot @ geo.wrist_local
            jac = np.zeros((6, len(geo.axes)))
            elbow_jac = np.zeros((3, len(geo.axes)))
            for i, (link, axis) in enumerate(geo.axes):
                origin, a = _axis_line(poses, link, axis)
                jac[:3, i] = np.cross(a, ee_pos - origin)
                jac[3:, i] = a
                if i < geo.elbow_joints:
                    elbow_jac[:, i] = np.cross(a, elbow - origin)
            null = np.linalg.svd(jac)[2][-1]
            direction = elbow_jac @ null
            norm = float(np.linalg.norm(direction))
            out[side] = ElbowSwivel(
                shoulder=geo.shoulder.copy(),
                elbow=elbow,
                wrist_in_mount=ee_rot.T @ (wrist - ee_pos),
                direction=direction / norm if norm > 1e-9 else np.zeros(3),
            )
        return out

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: Pose | None = None,
        right_pose: Pose | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
        delta_scale: float = 1.0,
        elbow_weight: float | None = None,
        pose_weight_scale: tuple[float, float] = (1.0, 1.0),
    ) -> np.ndarray:
        """Compute joint positions for absolute Cartesian end-effector targets.

        All positions and orientations must be expressed in the robot's world
        frame (FLU: +x forward, +y left, +z up). End-effector targets are
        soft-clamped between
        ``config.reach_soft_start`` and ``config.max_reach`` from each shoulder
        before solving; elbow hints are projected onto the robot's reachable
        elbow sphere and faded out for overhead targets; joint steps are
        rate-limited to ``config.max_joint_delta`` per call with a
        direction-preserving scale.

        Args:
            q_current: Full ``(N,)`` joint array in radians used as the solver
                seed and rest-cost target.
            left_pose: ``(pos, rot_3x3)`` numpy tuple for the left end-effector,
                or ``None`` to skip the left arm.
            right_pose: Same as ``left_pose`` for the right end-effector.
            left_elbow_pos: ``(3,)`` optional left elbow position hint in world frame.
            right_elbow_pos: ``(3,)`` optional right elbow position hint in world frame.
            delta_scale: Multiplier on ``config.max_joint_delta`` for this
                call. The per-call clamp is an implicit velocity limit at the
                nominal solve cadence; when a solve arrives late the caller
                should scale the budget by the actual elapsed time, otherwise
                the effective joint speed silently collapses (a 30 ms solve
                under the default clamp allows only ~1.1 rad/s instead of
                4 rad/s) and the accumulated error releases as a lurch.
            elbow_weight: Per-call weight on the elbow hints, replacing
                ``config.elbow_weight`` for this solve and applied *without*
                the overhead fade — for hints the caller synthesises (box
                mode's outward elbow bias, see
                :func:`almond_axol.teleop.box.elbow_swivel_hint`) rather than
                the headset's inferred elbow the fade exists for. ``None``
                keeps the configured behaviour.
            pose_weight_scale: ``(position, orientation)`` multipliers on
                ``config.pos_weight`` / ``config.ori_weight`` for this solve.
                The other costs (manipulability, an elbow hint) settle where
                they balance a small pose error; a caller that needs the
                pose held to a fraction of a degree raises these.

        Returns:
            Updated full ``(N,)`` joint array in radians.
        """
        if left_pose is None and right_pose is None:
            return q_current

        # Everything below runs in pyroki's joint order; converted back at
        # the return.
        q_current = self.to_pyroki_order(q_current)

        cfg = self.config

        target_L: jaxlie.SE3 | None = None
        lp: np.ndarray | None = None
        if left_pose is not None:
            lp, lr = left_pose
            lp = _clamp_reach(
                np.asarray(lp, dtype=np.float32),
                self._left_shoulder_pos,
                cfg.reach_soft_start,
                cfg.max_reach,
            )
            target_L = _np_to_se3(lp, np.asarray(lr, dtype=np.float32))

        target_R: jaxlie.SE3 | None = None
        rp: np.ndarray | None = None
        if right_pose is not None:
            rp, rr = right_pose
            rp = _clamp_reach(
                np.asarray(rp, dtype=np.float32),
                self._right_shoulder_pos,
                cfg.reach_soft_start,
                cfg.max_reach,
            )
            target_R = _np_to_se3(rp, np.asarray(rr, dtype=np.float32))

        elbow_L = (
            _pos3_to_se3(np.asarray(left_elbow_pos))
            if left_elbow_pos is not None
            else None
        )
        elbow_R = (
            _pos3_to_se3(np.asarray(right_elbow_pos))
            if right_elbow_pos is not None
            else None
        )

        # Fade the elbow hint out for overhead work: the headset's body-model
        # elbow is inferred rather than tracked and degrades sharply once the
        # hand rises to shoulder height, exactly where the shoulder nears its
        # joint limits and bad swivel targets do the most damage.
        if elbow_weight is not None:
            elbow_w_l = elbow_w_r = float(elbow_weight)
        else:
            elbow_w_l = cfg.elbow_weight * self._elbow_fade(lp, self._left_shoulder_pos)
            elbow_w_r = cfg.elbow_weight * self._elbow_fade(
                rp, self._right_shoulder_pos
            )

        q_prev = self._q_prev if self._q_prev is not None else q_current
        self._q_prev = np.asarray(q_current, dtype=np.float32).copy()

        q_result, cost_history = _SOLVE_IK(
            self.robot,
            self.robot_coll,
            target_L,
            target_R,
            self._l_ee_idx_jax,
            self._r_ee_idx_jax,
            elbow_L,
            elbow_R,
            self._l_elbow_idx_jax,
            self._r_elbow_idx_jax,
            jnp.asarray(q_current, dtype=jnp.float32),
            jnp.asarray(q_prev, dtype=jnp.float32),
            self._posture_pose,
            self._left_idx_jax,
            self._right_idx_jax,
            self._left_swivel_jax,
            self._right_swivel_jax,
            cfg.pos_weight * float(pose_weight_scale[0]),
            cfg.ori_weight * float(pose_weight_scale[1]),
            cfg.rest_weight,
            cfg.posture_weight,
            cfg.manipulability_weight,
            cfg.limit_weight,
            self._collision_starts,
            self._collision_ramps,
            cfg.self_collision_weight,
            elbow_w_l,
            elbow_w_r,
            cfg.manip_damping_threshold,
            cfg.manip_damping_boost,
            cfg.limit_damping_margin,
            cfg.max_iterations,
            cfg.cost_tolerance,
            cfg.lambda_initial,
            cfg.lambda_factor,
            self.robot_coll is not None,
            self._link_shift,
            _fingerprint=self._export_fingerprint,
        )
        q_result_np = np.asarray(q_result, dtype=np.float32)

        # NaN step proposals are rejected inside the solve, so they never show
        # up in the output — a solve where they happened can only return the
        # seed (or whatever earlier iterations reached). Historically this
        # froze teleop with no trace; warn on the transition into that state.
        nan_proposals = bool(np.isnan(np.asarray(cost_history)).any())
        if nan_proposals and not self._last_solve_had_nan:
            made_progress = not np.array_equal(
                q_result_np, np.asarray(q_current, dtype=np.float32)
            )
            _logger.warning(
                "IK solve hit NaN step proposals (numerically degenerate problem, "
                "e.g. a seed at a straight-limb singularity or a non-finite "
                "target); %s.",
                "later iterations recovered"
                if made_progress
                else "solver made no progress and returned the seed",
            )
        self._last_solve_had_nan = nan_proposals

        # Direction-preserving rate limit: scale the whole step so its largest
        # component hits max_joint_delta, rather than clipping per joint. Per-
        # joint clipping distorted the step direction (each joint saturated
        # differently), pushing the commanded pose off the solver's path and
        # releasing with a velocity discontinuity.
        delta = q_result_np - q_current
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        delta_budget = cfg.max_joint_delta * delta_scale
        if max_abs > delta_budget:
            delta = delta * (delta_budget / max_abs)
        q_out = (q_current + delta).astype(np.float32)
        if self.robot_coll is None:
            # Collision detection off: the guard's projection is skipped, but
            # the joint-limit clamp it also provided still applies.
            q_out = _clamp_to_limits_np(self.robot, q_current, q_out)
            return self.from_pyroki_order(q_out)
        q_out, guard_active_jax = _SAFE_STEP(
            self.robot,
            self.robot_coll,
            jnp.asarray(q_current, dtype=jnp.float32),
            jnp.asarray(q_out, dtype=jnp.float32),
            self._base_clearance_floor,
            jnp.asarray(delta_budget, dtype=jnp.float32),
            self._link_shift,
            _fingerprint=self._export_fingerprint,
        )
        guard_active = bool(guard_active_jax)
        if guard_active and not self._base_collision_guard_active:
            _logger.warning(
                "IK base-collision guard redirected a commanded step along "
                "the arm/base clearance boundary"
            )
        elif not guard_active and self._base_collision_guard_active:
            _logger.info("IK base-collision guard released")
        self._base_collision_guard_active = guard_active
        q_out = np.asarray(q_out, dtype=np.float32)
        return self.from_pyroki_order(q_out)

    def set_proximity_reference(self, q: np.ndarray) -> None:
        """Make ``q`` read as clear in :meth:`collision_proximity`.

        Each pair's onset moves in to its clearance at ``q`` where that is
        closer, and its full level to at least ``_PROXIMITY_SPAN_MIN`` past
        the onset. For the teleop rest pose: on a mount whose column or cart
        top sits near the arms, rest is already inside the lead (or the soft
        cost), and an operator holding the arms near it must feel nothing.
        Only ever tightens; a no-op with collision detection off.
        """
        if self.robot_coll is None:
            return
        d = np.asarray(
            _COLLISION_DISTANCE(
                self.robot,
                self.robot_coll,
                jnp.asarray(self.to_pyroki_order(q), dtype=jnp.float32),
                self._link_shift,
            )
        )
        self._proximity_onset = np.minimum(self._proximity_onset, d).astype(np.float32)
        self._proximity_full = np.minimum(
            self._proximity_full, self._proximity_onset - _PROXIMITY_SPAN_MIN
        ).astype(np.float32)

    def collision_proximity(self, q: np.ndarray) -> dict[str, float] | None:
        """How close each arm is to the arm/base collision boundary at ``q``.

        ``{"left": p, "right": p}`` with ``p`` in ``[0, 1]``: 0 while every
        pair of that arm is clear of its soft-cost activation by
        ``_PROXIMITY_LEAD`` (or is at least as clear as at home and at the
        :meth:`set_proximity_reference` pose), rising to 1
        where the cost reaches full slope — the region where IK holds the
        arm off the body instead of following the target. The worst pair of
        each arm wins. ``None`` when collision detection is off.
        """
        if self.robot_coll is None:
            return None
        d = np.asarray(
            _COLLISION_DISTANCE(
                self.robot,
                self.robot_coll,
                jnp.asarray(self.to_pyroki_order(q), dtype=jnp.float32),
                self._link_shift,
            )
        )
        span = self._proximity_onset - self._proximity_full
        level = np.clip((self._proximity_onset - d) / span, 0.0, 1.0)
        return {
            side: float(level[mask].max()) if mask.any() else 0.0
            for side, mask in self._proximity_side.items()
        }

    @property
    def has_lift_contact(self) -> bool:
        """Whether :meth:`lift_contact` can report anything: a version with
        a lift (the mobile Axol) and collision detection on."""
        return bool(self._lift_pairs.any())

    def lift_contact(self, q: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
        """How far each arm is inside the parts the lift leaves behind.

        Those parts are Jelly's deck, base and lower lift stage, placed by
        the live lift height (:meth:`set_lift`). For each arm with a pair
        inside its soft-cost activation shell at ``q``: ``(depth, normal)``,
        the worst pair's depth inside the shell (m, positive) and the unit
        direction (world frame) that moves the whole arm out of it, up for
        the deck's top. Arms clear of them are left out, and so is
        everything on a version without a lift or with collision
        detection off.
        """
        if not self.has_lift_contact:
            return {}
        d, jac = _arm_shift_jacobian(
            self.robot,
            self.robot_coll,
            jnp.asarray(self.to_pyroki_order(q), dtype=jnp.float32),
            self._link_shift,
            self._arm_links,
        )
        depth = self._lift_starts - np.asarray(d)
        jac = np.asarray(jac)
        out: dict[str, tuple[float, np.ndarray]] = {}
        for k, side in enumerate(("left", "right")):
            mask = self._lift_pair_side.get(side)
            if mask is None or not mask.any():
                continue
            pairs = np.flatnonzero(mask)
            worst = int(pairs[np.argmax(depth[pairs])])
            if depth[worst] <= 0.0:
                continue
            normal = jac[worst, k].astype(np.float64)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                continue
            out[side] = (float(depth[worst]), (normal / norm).astype(np.float32))
        return out

    def lift_clear_step(
        self,
        q: np.ndarray,
        *,
        max_step: float | None = None,
        sides: tuple[str, ...] = ("left", "right"),
        contact: dict[str, tuple[float, np.ndarray]] | None = None,
    ) -> np.ndarray | None:
        """One IK step that moves each arm in :meth:`lift_contact` out of it.

        Each of ``sides`` in contact gets its gripper target moved out along
        the contact normal (keeping its orientation) by its depth plus a few
        millimetres, at most ``max_step`` (m); the other arm keeps its joints.
        The arm reshapes as the solve finds best: with the posture
        attractor at rest, hands hanging over the deck come up and forward
        as it rises. ``contact`` is
        :meth:`lift_contact` at ``q`` if the caller already has it. Returns
        ``None`` when no arm of ``sides`` is in contact.
        """
        if contact is None:
            contact = self.lift_contact(q)
        moving = [side for side in sides if side in contact]
        if not moving:
            return None
        left, right = self.fk(q)
        poses = {"left": left, "right": right}
        for side in moving:
            depth, normal = contact[side]
            step = depth + _LIFT_CLEAR_MARGIN
            if max_step is not None:
                step = min(step, max_step)
            pos, rot = poses[side]
            poses[side] = ((pos + normal * step).astype(np.float32), rot)
        q_new = np.asarray(
            self.ik(q, left_pose=poses["left"], right_pose=poses["right"]),
            dtype=np.float32,
        ).copy()
        q = np.asarray(q, dtype=np.float32)
        before = {"left": left[0], "right": right[0]}
        after = dict(zip(("left", "right"), (pose[0] for pose in self.fk(q_new))))
        for side, indices in (
            ("left", self.left_indices),
            ("right", self.right_indices),
        ):
            if side not in moving:
                q_new[indices] = q[indices]
                continue
            # The collision cost adds its own push to the target's: keep the
            # gripper's actual move within max_step.
            travel = float(
                np.linalg.norm(np.asarray(after[side]) - np.asarray(before[side]))
            )
            if max_step is not None and travel > max_step > 0.0:
                q_new[indices] = q[indices] + (q_new[indices] - q[indices]) * (
                    max_step / travel
                )
        return q_new

    def lift_clear_pose(self, q: np.ndarray, iterations: int = 3) -> np.ndarray:
        """``q`` moved clear of the parts the lift leaves behind, at the live
        lift height.

        Replays the lift lowering from fully raised to the live height in
        small steps with :meth:`lift_clear_step` at each, so the arms end
        where they would have been pushed had they held ``q`` while it
        lowered. From deep inside the deck the way out is ambiguous; from
        fully raised it never is. For the return-to-rest goal on the mobile
        Axol. ``q`` itself when nothing is in contact at the live height.
        The solver's posture attractor and step history are restored.
        Takes about a second with the lift fully lowered.
        """
        q = np.asarray(q, dtype=np.float32)
        if not self.lift_contact(q):
            return q
        lowered = self._lift_lowered
        posture = self._posture_pose
        q_prev = self._q_prev
        guard = self._base_collision_guard_active
        rise = float(np.max(np.abs(self._lift_shift))) * lowered
        steps = max(1, int(np.ceil(rise / _LIFT_REPLAY_STEP)))
        try:
            # Held at q, the posture attractor keeps the upper arms down: the
            # forearms come up and forward over the deck.
            self.set_posture_pose(q)
            for k in range(1, steps + 1):
                self._lift_lowered = lowered * k / steps
                self._link_shift = self._make_link_shift()
                # Settle at the live height until the arms are clear.
                for _ in range(iterations if k < steps else 10 * iterations):
                    q_next = self.lift_clear_step(q)
                    if q_next is None:
                        break
                    q = q_next
        finally:
            self._lift_lowered = lowered
            self._link_shift = self._make_link_shift()
            self._posture_pose = posture
            self._q_prev = q_prev
            self._base_collision_guard_active = guard
        return q

    def _elbow_fade(
        self, ee_target: np.ndarray | None, shoulder_pos: np.ndarray
    ) -> float:
        """Smoothstep fade factor (1 -> 0) for the elbow hint as the EE target
        rises from shoulder height through ``config.elbow_fade_band`` above it."""
        band = self.config.elbow_fade_band
        if ee_target is None or band <= 0.0:
            return 1.0
        t = (float(ee_target[2]) - float(shoulder_pos[2])) / band
        t = min(max(t, 0.0), 1.0)
        return 1.0 - t * t * (3.0 - 2.0 * t)

    # -- Internal ------------------------------------------------------------

    def _warmup(self) -> None:
        """Trigger JIT compilation with dummy solves.

        Both graph shapes are compiled — with and without elbow hints — so
        neither the plain teleop solve nor box mode's first frame (which adds
        synthetic elbow hints, see ``ik(elbow_weight=...)``) stalls on a
        compile mid-session. The persistent compilation cache
        (:mod:`almond_axol.kinematics.jax_cache`) makes the second one a
        disk load on every run but the first.
        """
        _logger.info("Warming up IK solver (JIT compile)...")
        dummy_q = np.zeros(self.num_joints, dtype=np.float32)
        dummy_pos = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        dummy_rot = np.eye(3, dtype=np.float32)
        dummy_pose = (dummy_pos, dummy_rot)
        dummy_elbow = np.array([0.0, 0.2, 0.3], dtype=np.float32)
        # Box mode passes explicit elbow hints (and an ``elbow_weight``
        # override) regardless of the configured weight, so compile both the
        # with- and without-elbow variants up front.
        for with_elbows in (False, True):
            kwargs: dict = dict(
                q_current=dummy_q, left_pose=dummy_pose, right_pose=dummy_pose
            )
            if with_elbows:
                kwargs["left_elbow_pos"] = dummy_elbow
                kwargs["right_elbow_pos"] = dummy_elbow
            # Compilation/runtime failures here mean the worker cannot safely
            # serve the first real pose. Propagate them instead of announcing a
            # false-ready solver and leaving teleop frozen on its seed pose.
            self.ik(**kwargs)
        # The haptic proximity check runs beside every teleop solve.
        self.collision_proximity(dummy_q)
        # So does the lift contact check, on the mobile Axol.
        self.lift_contact(dummy_q)
        _logger.info("IK solver ready.")
