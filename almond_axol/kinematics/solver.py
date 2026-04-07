"""
Standalone bimanual IK solver for the Axol robot.

Uses pyroki + jaxls to solve for joint positions given absolute Cartesian
end-effector poses in the robot's world frame (FLU).
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from ..shared import URDF_PATH
from .config import KinematicsConfig

_logger = logging.getLogger(__name__)

# Link names in axol.urdf
_LEFT_EE = "left_gripper"
_RIGHT_EE = "right_gripper"
_LEFT_ELBOW = "left_e1"
_RIGHT_ELBOW = "right_e1"
_LEFT_SHOULDER = "left_s2"
_RIGHT_SHOULDER = "right_s2"

# Actuated joint names in ARM_JOINTS order (shoulder_1 … wrist_3).
# Must match the ordering assumed by rest_pose / motion_control.
_LEFT_JOINT_NAMES = [
    "left_s1_0",  # SHOULDER_1
    "left_s2_0",  # SHOULDER_2
    "left_s3_0",  # SHOULDER_3
    "left_e1_0",  # ELBOW
    "left_e2_0",  # WRIST_1
    "left_w1_0",  # WRIST_2
    "left_w2_0",  # WRIST_3
]
_RIGHT_JOINT_NAMES = [
    "right_s1_0",  # SHOULDER_1
    "right_s2_0",  # SHOULDER_2
    "right_s3_0",  # SHOULDER_3
    "right_e1_0",  # ELBOW
    "right_e2_0",  # WRIST_1
    "right_w1_0",  # WRIST_2
    "right_w2_0",  # WRIST_3
]


# ---------------------------------------------------------------------------
# JIT-compiled core solve
# ---------------------------------------------------------------------------


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
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
    manipulability_weight: float,
    limit_weight: float,
    self_collision_margin: float,
    self_collision_weight: float,
    elbow_weight: float,
    max_iterations: int,
    cost_tolerance: float,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_current, weight=rest_weight),
        pk.costs.manipulability_cost(
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
                pos_weight=elbow_weight,
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
                pos_weight=elbow_weight,
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
    solution_vals = analyzed.solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=cost_tolerance,
        ),
    )
    return solution_vals[var_joints][0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_reach(pos: np.ndarray, center: np.ndarray, max_reach: float) -> np.ndarray:
    """Clamp EE target position to within max_reach of center (shoulder position)."""
    d = pos - center
    dist = np.linalg.norm(d)
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    return pos


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
    """

    def __init__(self, config: KinematicsConfig = KinematicsConfig()) -> None:
        self.config = config

        _logger.info("Loading Axol URDF...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

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

        # Determine left/right joint indices in ARM_JOINTS order so that
        # q[left_indices] / q[right_indices] align with rest_pose and motion_control.
        actuated = list(self.robot.joints.actuated_names)
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self.left_indices = [name_to_idx[n] for n in _LEFT_JOINT_NAMES]
        self.right_indices = [name_to_idx[n] for n in _RIGHT_JOINT_NAMES]

        self._warmup()

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

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        """Compute end-effector poses from joint positions.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Tuple ``(left_pose, right_pose)`` as :class:`jaxlie.SE3` transforms
            in the robot's world frame (FLU).
        """
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint positions for absolute Cartesian end-effector targets.

        All positions and orientations must be expressed in the robot's world
        frame (FLU). End-effector targets are clamped to ``config.max_reach``
        from each shoulder before solving, and joint changes are clamped to
        ``config.max_joint_delta`` per call.

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
        if left_pose is not None:
            lp, lr = left_pose
            lp = _clamp_reach(
                np.asarray(lp, dtype=np.float32), self._left_shoulder_pos, cfg.max_reach
            )
            target_L = _np_to_se3(lp, np.asarray(lr, dtype=np.float32))

        target_R: jaxlie.SE3 | None = None
        if right_pose is not None:
            rp, rr = right_pose
            rp = _clamp_reach(
                np.asarray(rp, dtype=np.float32),
                self._right_shoulder_pos,
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

        q_result = _solve_ik(
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
            cfg.pos_weight,
            cfg.ori_weight,
            cfg.rest_weight,
            cfg.manipulability_weight,
            cfg.limit_weight,
            cfg.self_collision_margin,
            cfg.self_collision_weight,
            cfg.elbow_weight,
            cfg.max_iterations,
            cfg.cost_tolerance,
        )
        q_result_np = np.asarray(q_result, dtype=np.float32)
        delta = np.clip(
            q_result_np - q_current, -cfg.max_joint_delta, cfg.max_joint_delta
        )
        q_out = (q_current + delta).astype(np.float32)
        return q_out

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
