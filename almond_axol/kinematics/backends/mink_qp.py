"""``mink-qp`` backend: MuJoCo QP differential IK (the OpenArm reference stack).

Velocity-level IK solved as one quadratic program per tick: end-effector /
elbow / posture tasks in the objective, joint position limits and joint
velocity limits as *hard* inequality constraints, plus an optional
arm<->torso collision-clearance constraint (off by default — the coarse STL
convex hulls overlap the nominal workspace, and the bake-off showed the
constraint chattering against them; see
``KinematicsConfig.diff_collision_margin``). Integrating the solved velocity
from the current configuration makes the output inherently smooth — there is
no trust region to reject steps (no freeze/lurch) and Tikhonov damping keeps
behaviour graceful through singularities.
"""

from __future__ import annotations

import logging

import mink
import mujoco
import numpy as np

from ..base import CANONICAL_JOINT_NAMES, FKFrames, IKBackend, frame_body_names
from ..config import KinematicsConfig
from ..mujoco_model import arm_torso_geom_pairs, canonical_qpos_indices, load_mj_model

_logger = logging.getLogger(__name__)

_QP_SOLVER = "daqp"


class MinkBackend(IKBackend):
    """Differential IK on MuJoCo via mink, both arms in a single QP."""

    name = "mink-qp"

    def __init__(self, config: KinematicsConfig, dt: float) -> None:
        self._config = config
        self._dt = dt

        model = load_mj_model(with_meshes=True)
        self._model = model
        self._qidx = canonical_qpos_indices(model)
        self._configuration = mink.Configuration(model)
        self._fk_data = mujoco.MjData(model)

        names = frame_body_names()
        self._body_names = names
        self._l_ee_task = mink.FrameTask(
            frame_name=names["left_ee"],
            frame_type="body",
            position_cost=config.diff_position_cost,
            orientation_cost=config.diff_orientation_cost,
            lm_damping=config.diff_lm_damping,
        )
        self._r_ee_task = mink.FrameTask(
            frame_name=names["right_ee"],
            frame_type="body",
            position_cost=config.diff_position_cost,
            orientation_cost=config.diff_orientation_cost,
            lm_damping=config.diff_lm_damping,
        )
        self._l_elbow_task = mink.FrameTask(
            frame_name=names["left_elbow"],
            frame_type="body",
            position_cost=config.diff_elbow_cost,
            orientation_cost=0.0,
            lm_damping=config.diff_lm_damping,
        )
        self._r_elbow_task = mink.FrameTask(
            frame_name=names["right_elbow"],
            frame_type="body",
            position_cost=config.diff_elbow_cost,
            orientation_cost=0.0,
            lm_damping=config.diff_lm_damping,
        )
        self._posture_task = mink.PostureTask(
            model=model, cost=config.diff_posture_cost
        )
        self._tasks = [
            self._l_ee_task,
            self._r_ee_task,
            self._l_elbow_task,
            self._r_elbow_task,
            self._posture_task,
        ]

        limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(
                model,
                {n: config.diff_max_joint_vel for n in CANONICAL_JOINT_NAMES},
            ),
        ]
        if config.diff_collision_margin > 0.0:
            pairs = arm_torso_geom_pairs(model, margin=config.diff_collision_margin)
            if pairs:
                limits.append(
                    mink.CollisionAvoidanceLimit(
                        model,
                        geom_pairs=pairs,
                        minimum_distance_from_collisions=config.diff_collision_margin,
                        collision_detection_distance=config.diff_collision_margin
                        + 0.05,
                    )
                )
        self._limits = limits

        mujoco.mj_kinematics(model, self._fk_data)
        self.left_shoulder_pos = self._body_pos(names["left_shoulder"]).copy()
        self.right_shoulder_pos = self._body_pos(names["right_shoulder"]).copy()

        self.set_posture_pose(np.zeros(self.num_joints, dtype=np.float32))
        self._warmup()

    # -- Internal helpers -----------------------------------------------------

    def _body_pos(self, name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.asarray(self._fk_data.xpos[bid], dtype=np.float32)

    def _to_mj(self, q_canonical: np.ndarray) -> np.ndarray:
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[self._qidx] = np.asarray(q_canonical, dtype=np.float64)
        return q

    @staticmethod
    def _se3(pos: np.ndarray, rot: np.ndarray | None) -> mink.SE3:
        so3 = (
            mink.SO3.identity()
            if rot is None
            else mink.SO3.from_matrix(np.asarray(rot, dtype=np.float64))
        )
        return mink.SE3.from_rotation_and_translation(
            so3, np.asarray(pos, dtype=np.float64)
        )

    def _warmup(self) -> None:
        q = np.zeros(self.num_joints, dtype=np.float32)
        frames = self.fk_frames(q)
        self.ik(
            q,
            left_pose=frames.left_ee,
            right_pose=frames.right_ee,
            left_elbow_pos=frames.left_elbow,
            right_elbow_pos=frames.right_elbow,
        )
        _logger.info("mink-qp backend ready.")

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
        self._configuration.update(self._to_mj(q_current))

        if left_pose is not None:
            self._l_ee_task.set_target(self._se3(left_pose[0], left_pose[1]))
        else:
            self._l_ee_task.set_target_from_configuration(self._configuration)
        if right_pose is not None:
            self._r_ee_task.set_target(self._se3(right_pose[0], right_pose[1]))
        else:
            self._r_ee_task.set_target_from_configuration(self._configuration)
        if left_elbow_pos is not None:
            self._l_elbow_task.set_target(self._se3(left_elbow_pos, None))
        else:
            self._l_elbow_task.set_target_from_configuration(self._configuration)
        if right_elbow_pos is not None:
            self._r_elbow_task.set_target(self._se3(right_elbow_pos, None))
        else:
            self._r_elbow_task.set_target_from_configuration(self._configuration)

        n_iters = max(1, cfg.diff_iters)
        sub_dt = self._dt / n_iters
        for _ in range(n_iters):
            vel = mink.solve_ik(
                self._configuration,
                self._tasks,
                sub_dt,
                _QP_SOLVER,
                damping=cfg.diff_damping,
                limits=self._limits,
            )
            self._configuration.integrate_inplace(vel, sub_dt)

        return np.asarray(self._configuration.q[self._qidx], dtype=np.float32)

    def fk_frames(self, q: np.ndarray) -> FKFrames:
        self._fk_data.qpos[:] = self._to_mj(q)
        mujoco.mj_kinematics(self._model, self._fk_data)
        names = self._body_names

        def _pose(name: str) -> tuple[np.ndarray, np.ndarray]:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = np.asarray(self._fk_data.xpos[bid], dtype=np.float32)
            rot = np.asarray(self._fk_data.xmat[bid].reshape(3, 3), dtype=np.float32)
            return pos, rot

        l_ee = _pose(names["left_ee"])
        r_ee = _pose(names["right_ee"])
        return FKFrames(
            left_ee=l_ee,
            right_ee=r_ee,
            left_elbow=_pose(names["left_elbow"])[0],
            right_elbow=_pose(names["right_elbow"])[0],
        )

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture_task.set_target(self._to_mj(q))
