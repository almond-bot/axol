"""``pyroki-diff`` backend: single damped Gauss-Newton step per tick.

Keeps the existing pyroki/JAX kinematic model but replaces the per-frame
Levenberg-Marquardt run-to-convergence solve (whose trust region can reject
steps and freeze the output) with a fixed damped Gauss-Newton step per tick:

    dq = -(J^T J + λI)^{-1} J^T r,   clamped to ±v_max·dt, then joint limits.

There is no accept/reject logic, so the output can never freeze; the damping
λ bounds joint velocities through singularities, and the per-joint clamp +
hard limit clip replace the soft limit costs. This is the lowest-churn
differential candidate (same model and dependencies as today).
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk

from ..base import FKFrames, IKBackend, frame_body_names
from ..config import KinematicsConfig
from ..jax_cache import enable_persistent_compilation_cache
from ..pyroki_model import canonical_to_pyroki, load_pyroki_model

_logger = logging.getLogger(__name__)


@functools.partial(jax.jit, static_argnames=("l_ee", "r_ee", "l_el", "r_el"))
def _gn_step(
    robot: pk.Robot,
    q: jax.Array,
    l_ee: int,
    r_ee: int,
    l_el: int,
    r_el: int,
    target_l: jax.Array,
    target_r: jax.Array,
    elbow_l: jax.Array,
    elbow_r: jax.Array,
    posture: jax.Array,
    w_pos: float,
    w_ori: float,
    w_elbow: float,
    w_posture: float,
    damping: float,
    max_step: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    """One damped Gauss-Newton step toward the targets (all in pyroki order)."""

    def residual(q_: jax.Array) -> jax.Array:
        fk = robot.forward_kinematics(q_)

        def pose_err(idx: int, target: jax.Array) -> jax.Array:
            T = jaxlie.SE3(fk[idx])
            T_t = jaxlie.SE3(target)
            p = w_pos * (T.translation() - T_t.translation())
            r = w_ori * (T_t.rotation().inverse() @ T.rotation()).log()
            return jnp.concatenate([p, r])

        def elbow_err(idx: int, target: jax.Array) -> jax.Array:
            return w_elbow * (jaxlie.SE3(fk[idx]).translation() - target)

        return jnp.concatenate(
            [
                pose_err(l_ee, target_l),
                pose_err(r_ee, target_r),
                elbow_err(l_el, elbow_l),
                elbow_err(r_el, elbow_r),
                w_posture * (q_ - posture),
            ]
        )

    r = residual(q)
    J = jax.jacfwd(residual)(q)
    n = q.shape[0]
    H = J.T @ J + damping * jnp.eye(n)
    dq = -jnp.linalg.solve(H, J.T @ r)
    dq = jnp.clip(dq, -max_step, max_step)
    return jnp.clip(q + dq, lower, upper)


class PyrokiDiffBackend(IKBackend):
    """Velocity-level differential IK on the existing pyroki/JAX model."""

    name = "pyroki-diff"

    def __init__(self, config: KinematicsConfig, dt: float) -> None:
        self._config = config
        self._dt = dt

        enable_persistent_compilation_cache()
        _, robot, _ = load_pyroki_model()
        self._robot = robot
        self._perm = canonical_to_pyroki(robot)

        names = frame_body_names()
        link_names = robot.links.names
        self._l_ee_idx = link_names.index(names["left_ee"])
        self._r_ee_idx = link_names.index(names["right_ee"])
        self._l_elbow_idx = link_names.index(names["left_elbow"])
        self._r_elbow_idx = link_names.index(names["right_elbow"])
        l_sh_idx = link_names.index(names["left_shoulder"])
        r_sh_idx = link_names.index(names["right_shoulder"])

        self._lower = jnp.asarray(robot.joints.lower_limits)
        self._upper = jnp.asarray(robot.joints.upper_limits)
        n_iters = max(1, config.diff_iters)
        self._n_iters = n_iters
        self._max_step = jnp.full(
            robot.joints.num_actuated_joints,
            config.diff_max_joint_vel * dt / n_iters,
        )

        fk0 = robot.forward_kinematics(jnp.zeros(robot.joints.num_actuated_joints))
        self.left_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[l_sh_idx]).translation(), dtype=np.float32
        )
        self.right_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[r_sh_idx]).translation(), dtype=np.float32
        )

        self._posture = jnp.zeros(robot.joints.num_actuated_joints)

        # Warm-up solve (compiles the jitted step).
        q0 = np.zeros(self.num_joints, dtype=np.float32)
        frames = self.fk_frames(q0)
        self.ik(q0, left_pose=frames.left_ee, right_pose=frames.right_ee)
        _logger.info("pyroki-diff backend ready.")

    # -- Internal helpers -----------------------------------------------------

    def _to_pk(self, q_canonical: np.ndarray) -> np.ndarray:
        q = np.empty_like(np.asarray(q_canonical, dtype=np.float32))
        q[self._perm] = np.asarray(q_canonical, dtype=np.float32)
        return q

    @staticmethod
    def _wxyz_xyz(pos: np.ndarray, rot: np.ndarray) -> jax.Array:
        se3 = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(jnp.asarray(rot, dtype=jnp.float32)),
            jnp.asarray(pos, dtype=jnp.float32),
        )
        return se3.wxyz_xyz

    # -- IKBackend interface ----------------------------------------------------

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        if left_pose is None and right_pose is None:
            return np.asarray(q_current, dtype=np.float32).copy()

        cfg = self._config
        frames = self.fk_frames(q_current)
        lp, lr = left_pose if left_pose is not None else frames.left_ee
        rp, rr = right_pose if right_pose is not None else frames.right_ee
        el = left_elbow_pos if left_elbow_pos is not None else frames.left_elbow
        er = right_elbow_pos if right_elbow_pos is not None else frames.right_elbow

        target_l = self._wxyz_xyz(lp, lr)
        target_r = self._wxyz_xyz(rp, rr)
        q = jnp.asarray(self._to_pk(q_current))
        for _ in range(self._n_iters):
            q = _gn_step(
                self._robot,
                q,
                self._l_ee_idx,
                self._r_ee_idx,
                self._l_elbow_idx,
                self._r_elbow_idx,
                target_l,
                target_r,
                jnp.asarray(el, dtype=jnp.float32),
                jnp.asarray(er, dtype=jnp.float32),
                self._posture,
                cfg.diff_position_cost,
                cfg.diff_orientation_cost,
                cfg.diff_elbow_cost,
                cfg.diff_posture_cost,
                cfg.diff_damping,
                self._max_step,
                self._lower,
                self._upper,
            )

        q_pk = np.asarray(q, dtype=np.float32)
        return q_pk[self._perm]

    def fk_frames(self, q: np.ndarray) -> FKFrames:
        fk = self._robot.forward_kinematics(jnp.asarray(self._to_pk(q)))

        def _pose(idx: int) -> tuple[np.ndarray, np.ndarray]:
            T = jaxlie.SE3(fk[idx])
            return (
                np.asarray(T.translation(), dtype=np.float32),
                np.asarray(T.rotation().as_matrix(), dtype=np.float32),
            )

        return FKFrames(
            left_ee=_pose(self._l_ee_idx),
            right_ee=_pose(self._r_ee_idx),
            left_elbow=_pose(self._l_elbow_idx)[0],
            right_elbow=_pose(self._r_elbow_idx)[0],
        )

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture = jnp.asarray(self._to_pk(q))
