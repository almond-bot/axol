"""``dls`` backend: custom damped-least-squares differential IK in NumPy.

Fully self-contained velocity-level solver with zero new dependencies:
MuJoCo (already a core dependency) provides forward kinematics and geometric
Jacobians; the solve itself is a weighted damped least squares with

- **SVD-adaptive damping**: the damping factor grows as the smallest task-
  space singular value approaches zero, so joint velocities stay bounded
  through singularities (Nakamura & Hanafusa's classic scheme),
- **null-space posture**: the preferred-posture attractor acts through the
  projector ``N = I - J⁺J`` so it can never disturb tracking,
- **joint-limit velocity damper + hard clip**: motion toward a limit slows
  linearly inside an influence zone and is clipped at the bound, so limits
  behave like a smooth detent instead of a soft cost the solver can fight.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..base import CANONICAL_JOINT_NAMES, FKFrames, IKBackend, frame_body_names
from ..config import KinematicsConfig
from ..mujoco_model import load_mj_model

_logger = logging.getLogger(__name__)

_LIMIT_INFLUENCE = 0.15
"""Joint-limit velocity-damper influence zone (rad)."""

_SIGMA_TH_SCALE = 0.02
"""Singular-value threshold as a fraction of the position task weight."""


class DLSBackend(IKBackend):
    """Adaptive damped least squares with null-space posture, in NumPy."""

    name = "dls"

    def __init__(self, config: KinematicsConfig, dt: float) -> None:
        self._config = config
        self._dt = dt

        model = load_mj_model(with_meshes=False)
        self._model = model
        self._data = mujoco.MjData(model)

        qidx: list[int] = []
        didx: list[int] = []
        for name in CANONICAL_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint {name!r} not found in MuJoCo model")
            qidx.append(int(model.jnt_qposadr[jid]))
            didx.append(int(model.jnt_dofadr[jid]))
        self._qidx = np.array(qidx, dtype=np.int64)
        self._didx = np.array(didx, dtype=np.int64)

        jnt_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in CANONICAL_JOINT_NAMES
        ]
        self._lower = model.jnt_range[jnt_ids, 0].astype(np.float64)
        self._upper = model.jnt_range[jnt_ids, 1].astype(np.float64)

        names = frame_body_names()
        self._bids = {
            key: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for key, name in names.items()
        }

        self._posture = np.zeros(self.num_joints, dtype=np.float64)

        self._fk(np.zeros(self.num_joints, dtype=np.float32))
        self.left_shoulder_pos = np.asarray(
            self._data.xpos[self._bids["left_shoulder"]], dtype=np.float32
        ).copy()
        self.right_shoulder_pos = np.asarray(
            self._data.xpos[self._bids["right_shoulder"]], dtype=np.float32
        ).copy()
        _logger.info("dls backend ready.")

    # -- Internal helpers -----------------------------------------------------

    def _fk(self, q_canonical: np.ndarray) -> None:
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qidx] = np.asarray(q_canonical, dtype=np.float64)
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

    def _body_jac(self, bid: int) -> tuple[np.ndarray, np.ndarray]:
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, bid)
        return jacp[:, self._didx], jacr[:, self._didx]

    def _pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._bids[key]
        pos = np.asarray(self._data.xpos[bid], dtype=np.float64)
        rot = np.asarray(self._data.xmat[bid].reshape(3, 3), dtype=np.float64)
        return pos, rot

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
        n = self.num_joints
        n_iters = max(1, cfg.diff_iters)
        sub_dt = self._dt / n_iters
        max_step = cfg.diff_max_joint_vel * sub_dt

        q = np.asarray(q_current, dtype=np.float64).copy()
        for _ in range(n_iters):
            self._fk(q)

            rows: list[np.ndarray] = []
            errs: list[np.ndarray] = []

            def add_pose_task(key: str, target: tuple[np.ndarray, np.ndarray]) -> None:
                pos, rot = self._pose(key)
                jacp, jacr = self._body_jac(self._bids[key])
                t_pos = np.asarray(target[0], dtype=np.float64)
                t_rot = np.asarray(target[1], dtype=np.float64)
                rows.append(cfg.diff_position_cost * jacp)
                errs.append(cfg.diff_position_cost * (t_pos - pos))
                rot_err = Rotation.from_matrix(t_rot @ rot.T).as_rotvec()
                rows.append(cfg.diff_orientation_cost * jacr)
                errs.append(cfg.diff_orientation_cost * rot_err)

            def add_point_task(key: str, target: np.ndarray) -> None:
                pos, _ = self._pose(key)
                jacp, _ = self._body_jac(self._bids[key])
                t = np.asarray(target, dtype=np.float64)
                rows.append(cfg.diff_elbow_cost * jacp)
                errs.append(cfg.diff_elbow_cost * (t - pos))

            if left_pose is not None:
                add_pose_task("left_ee", left_pose)
            if right_pose is not None:
                add_pose_task("right_ee", right_pose)
            if left_elbow_pos is not None:
                add_point_task("left_elbow", left_elbow_pos)
            if right_elbow_pos is not None:
                add_point_task("right_elbow", right_elbow_pos)

            J = np.vstack(rows)
            e = np.concatenate(errs)

            # SVD-adaptive damping: raise damping as the smallest singular
            # value approaches zero (Nakamura & Hanafusa).
            U, S, Vt = np.linalg.svd(J, full_matrices=False)
            sigma_th = _SIGMA_TH_SCALE * cfg.diff_position_cost
            lam_max = 0.5 * sigma_th
            s_min = S[-1]
            lam2 = cfg.diff_damping
            if s_min < sigma_th:
                lam2 = lam2 + (1.0 - (s_min / sigma_th) ** 2) * lam_max**2
            gains = S / (S**2 + lam2)
            dq = Vt.T @ (gains * (U.T @ e))

            # Null-space posture attractor: project through N = I - J⁺J so it
            # cannot disturb the task.
            jjp = Vt.T @ np.diag(S**2 / (S**2 + lam2)) @ Vt
            nullspace = np.eye(n) - jjp
            v_post = cfg.diff_posture_cost * (self._posture - q)
            dq = dq + nullspace @ (v_post * sub_dt)

            # Joint-limit velocity damper: scale motion toward a limit down
            # linearly inside the influence zone, then hard-clip at the bound.
            dist_up = np.maximum(self._upper - q, 0.0)
            dist_lo = np.maximum(q - self._lower, 0.0)
            up_scale = np.minimum(dist_up / _LIMIT_INFLUENCE, 1.0)
            lo_scale = np.minimum(dist_lo / _LIMIT_INFLUENCE, 1.0)
            dq = np.where(dq > 0, dq * up_scale, dq * lo_scale)

            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, self._lower, self._upper)

        return q.astype(np.float32)

    def fk_frames(self, q: np.ndarray) -> FKFrames:
        self._fk(q)

        def _cast(pose: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            return (
                pose[0].astype(np.float32).copy(),
                pose[1].astype(np.float32).copy(),
            )

        return FKFrames(
            left_ee=_cast(self._pose("left_ee")),
            right_ee=_cast(self._pose("right_ee")),
            left_elbow=self._pose("left_elbow")[0].astype(np.float32).copy(),
            right_elbow=self._pose("right_elbow")[0].astype(np.float32).copy(),
        )

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture = np.asarray(q, dtype=np.float64).copy()
