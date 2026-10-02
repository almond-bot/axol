"""Teleop interface to the Mink tracker used by Cartesian policies.

Public poses use the Axol world frame (FLU). This adapter converts poses
to the bundled model frame, freezes inactive arms and exposes the teleop
kinematics interface. Joint vectors contain seven arm joints per side,
left followed by right, without grippers.
"""

from __future__ import annotations

import math
from collections.abc import Collection

import mink
import mujoco
import numpy as np

from ..constants import Joint, urdf_body_name
from ..policy.mink_ik import MinkIK, MinkIKConfig
from ..policy.mink_ik.frames import _ROOT_ORIGIN, _WORLD_TO_MODEL_R
from .config import KinematicsConfig

Pose = tuple[np.ndarray, np.ndarray]


def _world_position(position: np.ndarray) -> np.ndarray:
    return _ROOT_ORIGIN + _WORLD_TO_MODEL_R.T @ (position - _ROOT_ORIGIN)


def _model_pose(pose: Pose | None) -> Pose | None:
    if pose is None:
        return None
    position, rotation = (np.asarray(value, dtype=np.float64) for value in pose)
    if (
        position.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("Mink targets must be finite (position[3], rotation[3,3])")
    return (
        (_ROOT_ORIGIN + _WORLD_TO_MODEL_R @ (position - _ROOT_ORIGIN)).astype(
            np.float32
        ),
        (_WORLD_TO_MODEL_R @ rotation).astype(np.float32),
    )


class MinkKinematicsSolver:
    """KinematicsSolver's teleop API with policy-compatible Mink tracking.

    Mink uses its serving cost/collision profile, not JAX's soft-cost weights.
    ``max_joint_delta`` sets the same per-call QP bound used by policies;
    ``solve_hz`` supplies the actual cadence to the posture speed gate.
    Elbow hints are unsupported by the policy-compatible profile and must be
    disabled. Reset trajectories use a separate collision-constrained Mink
    planner so reset queries cannot mutate this tracker's state.
    Mutable model/tracker state belongs to one worker thread.
    """

    num_joints = 14

    def __init__(self, config: KinematicsConfig, *, solve_hz: float = 30.0) -> None:
        if config.elbow_weight != 0:
            raise ValueError("Mink teleop requires elbow_weight=0")
        if not math.isfinite(solve_hz) or solve_hz <= 0:
            raise ValueError("Mink solve_hz must be finite and positive")
        if not math.isfinite(config.max_joint_delta) or config.max_joint_delta <= 0:
            raise ValueError("Mink max_joint_delta must be finite and positive")
        self.config = config
        self.left_indices = list(range(7))
        self.right_indices = list(range(7, 14))
        self._ik = MinkIK(
            MinkIKConfig(
                mink_solve_hz=solve_hz,
                max_joint_delta=config.max_joint_delta,
                mink_max_joint_delta=config.max_joint_delta,
            )
        )
        self._posture_pose = np.zeros(self.num_joints, dtype=np.float32)
        model = self._ik.model
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._ik.joint_names
        ]
        self._qpos_indices = model.jnt_qposadr[joint_ids]
        self._fk_data = mujoco.MjData(model)
        self._elbow_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.ELBOW, is_left=is_left),
            )
            for is_left in (True, False)
        ]
        # Freeze inactive arms inside the QP, so their freedom cannot help
        # satisfy a constraint before a later output splice undoes that move.
        base_constraints = list(self._ik.tracker._constraints)
        self._constraints = {frozenset(("left", "right")): base_constraints}
        for side, inactive in (
            ("left", self.right_indices),
            ("right", self.left_indices),
        ):
            self._constraints[frozenset((side,))] = [
                *base_constraints,
                mink.DofFreezingTask(
                    model,
                    dof_indices=[int(model.jnt_dofadr[joint_ids[i]]) for i in inactive],
                ),
            ]

    @staticmethod
    def _q(q: np.ndarray) -> np.ndarray:
        result = np.asarray(q, dtype=np.float32)
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError("Mink joints must be 14 finite arm angles")
        return result

    @property
    def joint_names(self) -> list[str]:
        return self._ik.joint_names

    @property
    def posture_pose(self) -> np.ndarray:
        return self._posture_pose.copy()

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture_pose = self._q(q).copy()
        self._ik.set_rest_posture(self._posture_pose)

    def reset_tracking_state(self) -> None:
        self._ik.reset_tracking_state()

    @property
    def shoulder_positions(self) -> dict[str, np.ndarray]:
        return {
            side: _world_position(position).astype(np.float32)
            for side, position in self._ik.shoulder_positions.items()
        }

    def fk(self, q: np.ndarray) -> tuple[Pose, Pose]:
        poses = tuple(
            (
                _world_position(position).astype(np.float32),
                (_WORLD_TO_MODEL_R.T @ rotation).astype(np.float32),
            )
            for position, rotation in self._ik.fk(self._q(q))
        )
        return poses[0], poses[1]

    def elbow_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data = self._fk_data
        data.qpos[:] = 0.0
        data.qpos[self._qpos_indices] = self._q(q)
        mujoco.mj_kinematics(self._ik.model, data)
        positions = [
            _world_position(data.xpos[body]).astype(np.float32)
            for body in self._elbow_ids
        ]
        return positions[0], positions[1]

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: Pose | None = None,
        right_pose: Pose | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
        delta_scale: float = 1.0,
        *,
        active_sides: Collection[str] | None = None,
    ) -> np.ndarray:
        """Solve world-frame targets while keeping inactive arms bitwise fixed.

        The QP retains its fixed per-call safety budget when ``delta_scale``
        exceeds one; a delayed VR frame cannot enlarge the policy-compatible
        step bound. A smaller positive scale tightens the returned step.
        """
        del left_elbow_pos, right_elbow_pos  # profile has no elbow task
        q = self._q(q_current)
        if not math.isfinite(delta_scale) or delta_scale <= 0:
            raise ValueError("delta_scale must be finite and positive")
        active = set(("left", "right") if active_sides is None else active_sides)
        if not active <= {"left", "right"}:
            raise ValueError("active_sides may only contain 'left' and 'right'")
        targets = {"left": left_pose, "right": right_pose}
        active = frozenset(side for side in active if targets[side] is not None)
        if not active:
            return q.copy()
        tracker = self._ik.tracker
        previous_constraints = tracker._constraints
        tracker._constraints = self._constraints[active]
        try:
            result = self._ik.solve(
                q,
                _model_pose(left_pose) if "left" in active else None,
                _model_pose(right_pose) if "right" in active else None,
            )
        finally:
            tracker._constraints = previous_constraints
        result = self._q(result).copy()
        if delta_scale < 1:
            delta = result - q
            maximum = float(np.max(np.abs(delta)))
            limit = self.config.max_joint_delta * delta_scale
            if maximum > limit:
                result = q + delta * (limit / maximum)
        for side, indices in (
            ("left", self.left_indices),
            ("right", self.right_indices),
        ):
            if side not in active:
                result[indices] = q[indices]
        return result
