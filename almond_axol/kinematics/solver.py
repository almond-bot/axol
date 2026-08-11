"""
Standalone bimanual IK solver for the Axol robot.

Uses pyroki + jaxls to solve for joint positions given absolute Cartesian
end-effector poses in the robot's world frame (FLU).
"""

from __future__ import annotations

import functools
import logging
import math

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

from ..constants import (
    Joint,
    urdf_arm_joint_names,
    urdf_body_name,
)
from .config import KinematicsConfig
from .jax_cache import enable_persistent_compilation_cache
from .model import (
    collision_activation_margins,
    shared_robot,
    shared_robot_collision,
)

_logger = logging.getLogger(__name__)


Pose = tuple[np.ndarray, np.ndarray]
"""A frame as ``(position (3,), rotation (3, 3))`` numpy arrays, world frame (FLU).

The format :meth:`KinematicsSolver.fk` returns and :meth:`KinematicsSolver.ik`
takes, so a pose can be read, edited, and solved for without conversion.
"""

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
    (|shoulder->elbow| at the current configuration — the radius varies a few
    cm with shoulder pose, so it is measured from FK rather than fixed). The
    radial component of the raw target is unreachable by construction and
    would inject a permanent residual that fights the EE cost through the
    shared shoulder joints; only the swivel direction is kept.
    """
    r = jnp.linalg.norm(current_elbow_pos - shoulder)
    d = elbow.translation() - shoulder
    n = jnp.linalg.norm(d)
    proj = shoulder + d * (r / jnp.maximum(n, 1e-6))
    return jaxlie.SE3.from_rotation_and_translation(elbow.rotation(), proj)


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def _solve_ik(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
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
    self_collision_margin: jax.Array,
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
    costs.append(
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=self_collision_margin,
            weight=self_collision_weight,
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


# ---------------------------------------------------------------------------
# KinematicsSolver
# ---------------------------------------------------------------------------


class KinematicsSolver:
    """Bimanual IK solver for the Axol robot.

    Loads the bundled URDF, builds a pyroki + jaxls solver, and resolves
    absolute Cartesian end-effector poses (world frame, FLU) to joint angles.
    JIT compilation is triggered during ``__init__`` so the first call to
    :meth:`ik` is fast.

    Args:
        config: Solver cost weights and parameters.

    Example::

        solver = KinematicsSolver()
        q = np.zeros(solver.num_joints, dtype=np.float32)
        pos = np.array([0.3, 0.2, 0.4], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)
        q = solver.ik(q, left_pose=(pos, rot))

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
            config: Cost weights and solver parameters.
        """
        self.config = config

        enable_persistent_compilation_cache()

        # One robot model per process (see .model): reusing the same pytree —
        # including pyroki's per-instance JointVar class, a static field —
        # lets every solver after the first hit the in-memory jit cache
        # instead of re-tracing and re-running jaxls analysis.
        self.robot = shared_robot()
        self.robot_coll = shared_robot_collision()
        self._collision_margins = jnp.asarray(
            collision_activation_margins(
                self.robot, self.robot_coll, config.self_collision_margin
            )
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

        # Determine left/right joint indices in ARM_JOINTS order so that
        # q[left_indices] / q[right_indices] align with rest_pose and motion_control.
        actuated = list(self.robot.joints.actuated_names)
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self.left_indices = [name_to_idx[n] for n in _LEFT_JOINT_NAMES]
        self.right_indices = [name_to_idx[n] for n in _RIGHT_JOINT_NAMES]
        self._left_idx_jax = jnp.asarray(self.left_indices, dtype=jnp.int32)
        self._right_idx_jax = jnp.asarray(self.right_indices, dtype=jnp.int32)

        self._posture_pose = jnp.zeros(
            self.robot.joints.num_actuated_joints, dtype=jnp.float32
        )
        self._last_solve_had_nan = False

        # Seed of the previous ik() call: per-joint velocity estimate for the
        # limit-approach damping gate. None until the first call (gate off).
        self._q_prev: np.ndarray | None = None

        self._warmup()

    def set_posture_pose(self, q: np.ndarray) -> None:
        """Set the global preferred posture used as a persistent attractor.

        Args:
            q: Full ``(N,)`` joint array in radians (same ordering as :meth:`ik`).
        """
        self._posture_pose = jnp.asarray(q, dtype=jnp.float32)

    @property
    def posture_pose(self) -> np.ndarray:
        """The global preferred posture, as set by :meth:`set_posture_pose`.

        Lets a caller that sweeps the attractor (e.g. Cartesian path planning
        in :mod:`almond_axol.kinematics.path`) restore what it found.
        """
        return np.asarray(self._posture_pose, dtype=np.float32)

    # -- Properties ----------------------------------------------------------

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of all actuated joint names (left arm then right arm)."""
        return list(self.robot.joints.actuated_names)

    @property
    def num_joints(self) -> int:
        """Total number of actuated joints across both arms."""
        return self.robot.joints.num_actuated_joints

    # -- Public interface ----------------------------------------------------

    def fk(self, q: np.ndarray) -> tuple[Pose, Pose]:
        """Compute end-effector poses from joint positions.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Tuple ``(left_pose, right_pose)``, each a ``(pos (3,), rot (3, 3))``
            pair of numpy arrays in the robot's world frame (FLU) — the same
            format :meth:`ik` takes, so a pose can be read, nudged, and solved
            for directly::

                (l_pos, l_rot), _ = solver.fk(q)
                q = solver.ik(q, left_pose=(l_pos + delta, l_rot))

        The returned pose is the gripper *mount* frame — for the point the
        fingers close on, see :func:`almond_axol.kinematics.path.tip_poses`.
        """
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return (
            _se3_to_pose(jaxlie.SE3(fk[self.l_ee_idx])),
            _se3_to_pose(jaxlie.SE3(fk[self.r_ee_idx])),
        )

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: Pose | None = None,
        right_pose: Pose | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint positions for absolute Cartesian end-effector targets.

        All positions and orientations must be expressed in the robot's world
        frame (FLU). End-effector targets are soft-clamped between
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

        Returns:
            Updated full ``(N,)`` joint array in radians.
        """
        if left_pose is None and right_pose is None:
            return q_current

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
        elbow_w_l = cfg.elbow_weight * self._elbow_fade(lp, self._left_shoulder_pos)
        elbow_w_r = cfg.elbow_weight * self._elbow_fade(rp, self._right_shoulder_pos)

        q_prev = self._q_prev if self._q_prev is not None else q_current
        self._q_prev = np.asarray(q_current, dtype=np.float32).copy()

        q_result, cost_history = _solve_ik(
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
            self._left_shoulder_jax,
            self._right_shoulder_jax,
            cfg.pos_weight,
            cfg.ori_weight,
            cfg.rest_weight,
            cfg.posture_weight,
            cfg.manipulability_weight,
            cfg.limit_weight,
            self._collision_margins,
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
        if max_abs > cfg.max_joint_delta:
            delta = delta * (cfg.max_joint_delta / max_abs)
        q_out = (q_current + delta).astype(np.float32)
        return q_out

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
        """Trigger JIT compilation with a dummy solve."""
        _logger.info("Warming up IK solver (JIT compile)...")
        dummy_q = np.zeros(self.num_joints, dtype=np.float32)
        dummy_pos = np.array([0.0, 0.0, 0.3], dtype=np.float32)
        dummy_rot = np.eye(3, dtype=np.float32)
        dummy_pose = (dummy_pos, dummy_rot)
        kwargs: dict = dict(
            q_current=dummy_q, left_pose=dummy_pose, right_pose=dummy_pose
        )
        if self.config.elbow_weight > 0:
            dummy_elbow = np.array([0.0, 0.2, 0.3], dtype=np.float32)
            kwargs["left_elbow_pos"] = dummy_elbow
            kwargs["right_elbow_pos"] = dummy_elbow
        try:
            self.ik(**kwargs)
        except Exception:
            pass
        _logger.info("IK solver ready.")
