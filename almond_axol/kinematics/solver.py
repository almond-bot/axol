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

from ..motor import JointValues
from ..shared import ARM_JOINTS, URDF_PATH, rad_to_rev, rev_to_rad
from .config import KinematicsConfig

_logger = logging.getLogger(__name__)

# Link names in axol.urdf
_LEFT_EE = "left_ee_link"
_RIGHT_EE = "right_ee_link"
_LEFT_ELBOW = "left_elbow_link"
_RIGHT_ELBOW = "right_elbow_link"
_LEFT_SHOULDER = "left_shoulder_1_link"
_RIGHT_SHOULDER = "right_shoulder_1_link"


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


def _pos3_to_se3(pos: np.ndarray) -> jaxlie.SE3:
    """Convert a (3,) position array to an identity-rotation SE3."""
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=identity), jnp.asarray(pos, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# KinematicsSolver
# ---------------------------------------------------------------------------


class KinematicsSolver:
    """Bimanual IK solver for the Axol robot.

    Loads the bundled URDF, builds a pyroki + jaxls solver, and resolves
    absolute Cartesian end-effector poses (world frame, FLU) to joint angles.
    JIT compilation is triggered during ``__init__`` so the first call to
    :meth:`solve` is fast.

    Args:
        config: Solver cost weights and parameters.

    Example::

        solver = KinematicsSolver()
        target = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.3, 0.2, 0.4])
        )
        left_q, right_q = solver.ik(left_pose=target, right_pose=target)
    """

    def __init__(self, config: KinematicsConfig = KinematicsConfig()) -> None:
        self.config = config

        _logger.info("Loading Axol URDF...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir="")
        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        names = self.robot.links.names
        self._L_ee_idx = names.index(_LEFT_EE)
        self._R_ee_idx = names.index(_RIGHT_EE)
        self._L_elbow_idx = names.index(_LEFT_ELBOW)
        self._R_elbow_idx = names.index(_RIGHT_ELBOW)

        self._L_ee_idx_jax = jnp.asarray(self._L_ee_idx, dtype=jnp.int32)
        self._R_ee_idx_jax = jnp.asarray(self._R_ee_idx, dtype=jnp.int32)
        self._L_elbow_idx_jax = jnp.asarray(self._L_elbow_idx, dtype=jnp.int32)
        self._R_elbow_idx_jax = jnp.asarray(self._R_elbow_idx, dtype=jnp.int32)

        # Shoulder positions are fixed in world frame (independent of joint angles)
        L_sh_idx = names.index(_LEFT_SHOULDER)
        R_sh_idx = names.index(_RIGHT_SHOULDER)
        fk0 = self.robot.forward_kinematics(
            jnp.zeros(self.robot.joints.num_actuated_joints)
        )
        self._L_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[L_sh_idx]).translation(), dtype=np.float32
        )
        self._R_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[R_sh_idx]).translation(), dtype=np.float32
        )

        # Determine left/right joint split indices into the full actuated vector
        actuated = list(self.robot.joints.actuated_names)
        self._left_indices = [
            i for i, n in enumerate(actuated) if n.startswith("left_")
        ]
        self._right_indices = [
            i for i, n in enumerate(actuated) if n.startswith("right_")
        ]

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

    def fk(
        self,
        q_left: JointValues | None = None,
        q_right: JointValues | None = None,
    ) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        """Compute end-effector poses from joint positions.

        Args:
            q_left: Left arm joint positions in revolutions. Defaults to zeros.
            q_right: Right arm joint positions in revolutions. Defaults to zeros.

        Returns:
            Tuple ``(left_pose, right_pose)`` as :class:`jaxlie.SE3` transforms
            in the robot's world frame (FLU).
        """
        q_full = np.zeros(self.num_joints, dtype=np.float32)
        if q_left is not None:
            for joint, gi in zip(ARM_JOINTS, self._left_indices):
                q_full[gi] = rev_to_rad(q_left.get(joint, 0.0))
        if q_right is not None:
            for joint, gi in zip(ARM_JOINTS, self._right_indices):
                q_full[gi] = rev_to_rad(q_right.get(joint, 0.0))

        fk = self.robot.forward_kinematics(jnp.asarray(q_full))
        return jaxlie.SE3(fk[self._L_ee_idx]), jaxlie.SE3(fk[self._R_ee_idx])

    def ik(
        self,
        left_pose: jaxlie.SE3 | None = None,
        right_pose: jaxlie.SE3 | None = None,
        q_current_left: JointValues | None = None,
        q_current_right: JointValues | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> tuple[JointValues, JointValues]:
        """Compute joint positions for absolute Cartesian end-effector targets.

        All positions and orientations must be expressed in the robot's world
        frame (FLU). End-effector targets are clamped to ``config.max_reach``
        from each shoulder before solving, and joint changes are clamped to
        ``config.max_joint_delta`` per call.

        Args:
            left_pose: :class:`jaxlie.SE3` target for the left end-effector,
                or ``None`` to skip the left arm.
            right_pose: :class:`jaxlie.SE3` target for the right end-effector,
                or ``None`` to skip the right arm.
            q_current_left: Current left arm joint positions in revolutions.
                Defaults to zeros. Used as the solver seed and rest-cost target.
            q_current_right: Current right arm joint positions in revolutions.
                Defaults to zeros. Used as the solver seed and rest-cost target.
            left_elbow_pos: ``(3,)`` optional left elbow position hint in world
                frame. Improves solutions in kinematically ambiguous configurations.
            right_elbow_pos: ``(3,)`` optional right elbow position hint in world
                frame. Improves solutions in kinematically ambiguous configurations.

        Returns:
            Tuple ``(left_joints, right_joints)`` as :class:`JointValues` dicts
            in revolutions. If an arm's pose is ``None``, that arm's output
            equals its ``q_current`` input.
        """
        n_left = len(self._left_indices)
        n_right = len(self._right_indices)

        q_left = (
            np.zeros(n_left, dtype=np.float32)
            if q_current_left is None
            else np.array(
                [rev_to_rad(q_current_left.get(j, 0.0)) for j in ARM_JOINTS],
                dtype=np.float32,
            )
        )
        q_right = (
            np.zeros(n_right, dtype=np.float32)
            if q_current_right is None
            else np.array(
                [rev_to_rad(q_current_right.get(j, 0.0)) for j in ARM_JOINTS],
                dtype=np.float32,
            )
        )

        def _to_jv(q_rad: np.ndarray) -> JointValues:
            return {j: rad_to_rev(float(q_rad[i])) for i, j in enumerate(ARM_JOINTS)}

        if left_pose is None and right_pose is None:
            return _to_jv(q_left), _to_jv(q_right)

        q_full = np.zeros(self.num_joints, dtype=np.float32)
        for i, gi in enumerate(self._left_indices):
            q_full[gi] = q_left[i]
        for i, gi in enumerate(self._right_indices):
            q_full[gi] = q_right[i]

        cfg = self.config

        def _clamp_pose(pose: jaxlie.SE3, shoulder_pos: np.ndarray) -> jaxlie.SE3:
            pos_clamped = _clamp_reach(
                np.asarray(pose.translation(), dtype=np.float32),
                shoulder_pos,
                cfg.max_reach,
            )
            return jaxlie.SE3.from_rotation_and_translation(
                pose.rotation(), jnp.asarray(pos_clamped, dtype=jnp.float32)
            )

        target_L = (
            _clamp_pose(left_pose, self._L_shoulder_pos)
            if left_pose is not None
            else None
        )
        target_R = (
            _clamp_pose(right_pose, self._R_shoulder_pos)
            if right_pose is not None
            else None
        )
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
            self._L_ee_idx_jax,
            self._R_ee_idx_jax,
            elbow_L,
            elbow_R,
            self._L_elbow_idx_jax,
            self._R_elbow_idx_jax,
            jnp.asarray(q_full, dtype=jnp.float32),
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

        # Clamp per-joint delta to max_joint_delta (revolutions → radians)
        max_delta_rad = rev_to_rad(cfg.max_joint_delta)
        delta = np.clip(q_result_np - q_full, -max_delta_rad, max_delta_rad)
        q_result_np = q_full + delta

        out_left = q_left if left_pose is None else q_result_np[self._left_indices]
        out_right = q_right if right_pose is None else q_result_np[self._right_indices]
        return _to_jv(out_left), _to_jv(out_right)

    # -- Internal ------------------------------------------------------------

    def _warmup(self) -> None:
        """Trigger JIT compilation with a dummy solve."""
        _logger.info("Warming up IK solver (JIT compile)...")
        dummy_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([0.0, 0.0, 0.3])
        )
        dummy_elbow = np.array([0.0, 0.2, 0.3], dtype=np.float32)
        try:
            # Compile both the no-elbow path (pose-only) and the full path
            # (pose + elbow hints) so neither recompiles during live teleop.
            self.ik(left_pose=dummy_pose, right_pose=dummy_pose)
            self.ik(
                left_pose=dummy_pose,
                right_pose=dummy_pose,
                left_elbow_pos=dummy_elbow,
                right_elbow_pos=dummy_elbow,
            )
            q = np.zeros(self.num_joints, dtype=np.float32)
            self.robot.forward_kinematics(jnp.asarray(q))
        except Exception:
            pass
        _logger.info("IK solver ready.")
