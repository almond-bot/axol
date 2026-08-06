"""``pyroki-lm`` backend: the original per-frame Levenberg-Marquardt solver.

Thin adapter over :class:`almond_axol.kinematics.solver.KinematicsSolver`
that translates between the canonical joint order and pyroki's internal
(topologically sorted) actuated order. This is the bake-off baseline.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxlie
import numpy as np

from ..base import FKFrames, IKBackend
from ..config import KinematicsConfig
from ..pyroki_model import canonical_to_pyroki, to_canonical_order, to_pyroki_order
from ..solver import KinematicsSolver


class PyrokiLMBackend(IKBackend):
    """Adapter exposing :class:`KinematicsSolver` through the backend interface."""

    name = "pyroki-lm"

    def __init__(self, config: KinematicsConfig) -> None:
        self._solver = KinematicsSolver(config)
        self._perm = canonical_to_pyroki(self._solver.robot)
        self.left_shoulder_pos = self._solver._left_shoulder_pos.copy()
        self.right_shoulder_pos = self._solver._right_shoulder_pos.copy()

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        q_pk = to_pyroki_order(np.asarray(q_current, dtype=np.float32), self._perm)
        q_new = self._solver.ik(
            q_pk,
            left_pose=left_pose,
            right_pose=right_pose,
            left_elbow_pos=left_elbow_pos,
            right_elbow_pos=right_elbow_pos,
        )
        return to_canonical_order(q_new, self._perm)

    def fk_frames(self, q: np.ndarray) -> FKFrames:
        q_pk = to_pyroki_order(np.asarray(q, dtype=np.float32), self._perm)
        fk = self._solver.robot.forward_kinematics(jnp.asarray(q_pk))

        def _pose(idx: int) -> tuple[np.ndarray, np.ndarray]:
            T = jaxlie.SE3(fk[idx])
            return (
                np.asarray(T.translation(), dtype=np.float32),
                np.asarray(T.rotation().as_matrix(), dtype=np.float32),
            )

        def _pos(idx: int) -> np.ndarray:
            return np.asarray(jaxlie.SE3(fk[idx]).translation(), dtype=np.float32)

        return FKFrames(
            left_ee=_pose(self._solver.l_ee_idx),
            right_ee=_pose(self._solver.r_ee_idx),
            left_elbow=_pos(self._solver.l_elbow_idx),
            right_elbow=_pos(self._solver.r_elbow_idx),
        )

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._solver.set_posture_pose(
            to_pyroki_order(np.asarray(q, dtype=np.float32), self._perm)
        )

    def settle_rest_pose(self, q_rest: np.ndarray) -> np.ndarray:
        """Iterate the full teleop IK to the manipulability-balanced rest pose.

        EE and elbow targets are the configured rest pose's own FK, and posture
        is pinned to the current iterate, so all costs except manipulability
        have zero gradient at the starting q. The remaining manipulability
        gradient drives q in the EE null space until it stops changing — the
        same conditions the rising-edge posture pin in the teleop worker
        produces at engage time.
        """
        max_iterations, tol = 200, 1e-5
        q = np.asarray(q_rest, dtype=np.float32).copy()
        frames = self.fk_frames(q)
        l_pose, r_pose = frames.left_ee, frames.right_ee
        l_elbow, r_elbow = frames.left_elbow, frames.right_elbow

        for _ in range(max_iterations):
            self.set_posture_pose(q)
            q_new = self.ik(
                q,
                left_pose=l_pose,
                right_pose=r_pose,
                left_elbow_pos=l_elbow,
                right_elbow_pos=r_elbow,
            )
            if float(np.max(np.abs(q_new - q))) < tol:
                return q_new
            q = q_new
        return q
