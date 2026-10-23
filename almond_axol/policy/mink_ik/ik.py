"""Mink inverse kinematics for the 14 Axol arm joints.

Each solve applies a radial reach clamp, a velocity-QP tracking step and a
componentwise joint-displacement backstop. Joint vectors contain the left
arm followed by the right arm in ``urdf_arm_joint_names`` order, excluding
grippers. The bundled MuJoCo model supplies both tracking geometry and FK.

NumPy pose conversion and MuJoCo kinematics keep this module independent of
JAX. The numerical reference tests cover recurrent solves, frame conversion
and FK. One :class:`MinkIK` belongs to one control thread: ``solve`` and
``fk`` share mutable model data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import mujoco
import numpy as np

from almond_axol.constants import Joint, urdf_arm_joint_names, urdf_body_name

from .ik_config import PINNED_URDF, MinkIKConfig
from .ik_mink_backend import MinkTracker
from .ik_mujoco_model import qpos_indices
from .vendor_io import aa2rotm

_logger = logging.getLogger(__name__)

N_ARM_JOINTS = 14


def _clamp_reach(pos: np.ndarray, center: np.ndarray, max_reach: float) -> np.ndarray:
    """Clamp an EE target to the sphere about its shoulder center.

    An out-of-reach target becomes the nearest point on the reachable sphere,
    avoiding unnecessary QP error against joint limits.
    """
    d = pos - center
    dist = np.linalg.norm(d)
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    return pos


def pose6_to_pos_rot_np(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``[x, y, z, rx, ry, rz]`` into position and rotation matrix.

    The NumPy Rodrigues formula avoids compilation on the control path.
    Returns float32 position ``(3,)`` and rotation ``(3, 3)`` arrays. Coercion
    and operation order are covered by the numerical reference tests.
    """
    pose6 = np.asarray(pose6, dtype=np.float32)
    if pose6.shape != (6,):
        raise ValueError(f"pose6 must have shape (6,), got {pose6.shape}")
    pos = pose6[:3].copy()
    rot = np.asarray(aa2rotm(pose6[3:6]), dtype=np.float32)
    return pos, rot


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or a.samefile(b)
    except OSError:
        return False


class MinkIK:
    """Velocity-QP inverse kinematics on 14 arm joints.

    Args:
        config: Solver weights, cadence, collision settings and step limits.
        urdf_path: Must be the bundled ``PINNED_URDF``. The tracker loads that
            asset, so another file would disagree with its geometry or frame.
        warmup: Run one zero-error solve at construction to initialize solver
            code paths before control begins, then clear tracking memory.

    Joint vectors follow ``joint_names``: left s1..w2, then right s1..w2.
    The two grippers are excluded.
    """

    def __init__(
        self,
        config: MinkIKConfig = MinkIKConfig(),
        urdf_path: Path | str = PINNED_URDF,
        *,
        warmup: bool = True,
    ) -> None:
        urdf_path = Path(urdf_path)
        if not _same_file(urdf_path, PINNED_URDF):
            raise ValueError(
                f"MinkIK requires the bundled model ({PINNED_URDF}); "
                f"got {urdf_path}. The tracker loads this asset to keep "
                "FK and collision geometry consistent."
            )
        self._config = config
        names = list(urdf_arm_joint_names(is_left=True)) + list(
            urdf_arm_joint_names(is_left=False)
        )
        assert len(names) == N_ARM_JOINTS, names
        self._joint_names = names
        ee_bodies = (
            urdf_body_name(Joint.GRIPPER, is_left=True),
            urdf_body_name(Joint.GRIPPER, is_left=False),
        )
        elbow_bodies = (
            urdf_body_name(Joint.ELBOW, is_left=True),
            urdf_body_name(Joint.ELBOW, is_left=False),
        )
        # Track the 14 arm joints while freezing finger sliders at zero.
        # Joint-name mapping keeps the input vector independent of qpos order.
        self._tracker = MinkTracker(
            config,
            joint_names=list(names),
            arm_joint_names=list(names),
            ee_bodies=ee_bodies,
            elbow_bodies=elbow_bodies,
        )
        # FK, reach-clamp centers and QP constraints share one model.
        self._model: mujoco.MjModel = self._tracker._model
        self._qadr = qpos_indices(self._model, names)
        self._data = mujoco.MjData(self._model)
        self._ee_bid = {
            side: mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
            for side, body in zip(("left", "right"), ee_bodies, strict=True)
        }
        # Shoulder centers are fixed by the model at qpos=0. Store float32
        # copies to retain the reach clamp's tested arithmetic precision.
        self._data.qpos[:] = 0.0
        mujoco.mj_kinematics(self._model, self._data)
        self._shoulder: dict[str, np.ndarray] = {}
        for side, is_left in (("left", True), ("right", False)):
            sid = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.SHOULDER_1, is_left=is_left),
            )
            self._shoulder[side] = np.asarray(
                self._data.xpos[sid], dtype=np.float32
            ).copy()
        self._last_solve_ms = 0.0
        if warmup:
            self.warmup()

    # -- Properties ----------------------------------------------------------

    @property
    def config(self) -> MinkIKConfig:
        return self._config

    @property
    def joint_names(self) -> list[str]:
        """The 14 arm joint names (left then right) — the ``q`` order."""
        return list(self._joint_names)

    @property
    def model(self) -> mujoco.MjModel:
        """The tracker's MuJoCo model, including fixed end-effector frames."""
        return self._model

    @property
    def tracker(self) -> MinkTracker:
        return self._tracker

    @property
    def shoulder_positions(self) -> dict[str, np.ndarray]:
        """Reach-clamp centres (``left``/``right`` shoulder-1 body at q=0)."""
        return {k: v.copy() for k, v in self._shoulder.items()}

    @property
    def per_call_step_bound(self) -> float:
        """Per-call joint displacement bound (rad): ``mink_max_joint_delta``."""
        return float(self._config.mink_max_joint_delta)

    @property
    def fail_count(self) -> int:
        """QP solves that exhausted both attempts and returned the seed unchanged."""
        return self._tracker.fail_count

    @property
    def last_solve_ms(self) -> float:
        """Wall time of the last ``tracker.solve`` (ms) — the ``ik_ms`` source."""
        return self._last_solve_ms

    # -- Posture / state -----------------------------------------------------

    def set_rest_posture(self, q14: np.ndarray) -> None:
        """Pin the posture attractor to a settled rest configuration.

        With ``mink_posture_target="rest"``, engage-edge ``set_posture`` calls do
        not replace this target.
        """
        self._tracker.set_rest_posture(self._as_q14(q14))

    def reset_tracking_state(self) -> None:
        """Clear posture speed-gate memory at a control discontinuity."""
        self._tracker.reset_tracking_state()

    def warmup(self) -> None:
        """One zero-error solve at the range midpoints, then flush gate memory.

        Pages in the mink/daqp code paths before the first real tick. A HOLD
        here means the QP stack itself is broken (a solver/mujoco mismatch) —
        better to die at construction than to freeze the arms at TICK0.
        """
        model = self._model
        jids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._joint_names
        ]
        q_mid = ((model.jnt_range[jids, 0] + model.jnt_range[jids, 1]) / 2.0).astype(
            np.float32
        )
        left, right = self.fk(q_mid)
        fails_before = self.fail_count
        self.solve(q_mid, left, right)
        if self.fail_count != fails_before:
            raise RuntimeError(
                "mink warm-up solve HELD (QP infeasible on a zero-error target); "
                "the mink/daqp/mujoco stack is not usable"
            )
        self.reset_tracking_state()

    # -- Kinematics ----------------------------------------------------------

    def fk(
        self, q14: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Return gripper-body poses ``((pos_l, R_l), (pos_r, R_r))``.

        Both poses are float64 copies in the bundled model frame, matching the
        solve targets. This helper also supplies warmup and reference checks.
        """
        q = self._as_q14(q14)
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qadr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self._model, self._data)
        out = []
        for side in ("left", "right"):
            bid = self._ee_bid[side]
            out.append(
                (self._data.xpos[bid].copy(), self._data.xmat[bid].reshape(3, 3).copy())
            )
        return out[0], out[1]

    def solve(
        self,
        q14: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None,
        right_pose: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Solve both arm targets with reach and joint-displacement limits.

        Args:
            q14: Seed and displacement-clamp anchor. Callers may reuse the previous
                solution while keeping it close to measured joints.
            left_pose / right_pose: Model-frame ``(position[3], rotation[3,3])``
                targets, or None to omit that arm's end-effector task.

        Returns:
            A float32 joint vector of shape ``(14,)``. Targets are reach-clamped
            before tracking, and the result is bounded by ``mink_max_joint_delta``.
            The QP reserves two percent of that budget, leaving the final
            componentwise clamp as a backstop.
        """
        q_current = self._as_q14(q14)
        if left_pose is None and right_pose is None:
            return q_current

        cfg = self._config

        lp = lr = rp = rr = None
        if left_pose is not None:
            lp, lr = left_pose
            lp = _clamp_reach(
                np.asarray(lp, dtype=np.float32), self._shoulder["left"], cfg.max_reach
            )
        if right_pose is not None:
            rp, rr = right_pose
            rp = _clamp_reach(
                np.asarray(rp, dtype=np.float32),
                self._shoulder["right"],
                cfg.max_reach,
            )

        t0 = time.perf_counter()
        q_result_np = self._tracker.solve(
            np.asarray(q_current, dtype=np.float32),
            None if lp is None else (lp, np.asarray(lr, dtype=np.float32)),
            None if rp is None else (rp, np.asarray(rr, dtype=np.float32)),
            None,
            None,
        )
        self._last_solve_ms = (time.perf_counter() - t0) * 1e3
        max_delta = cfg.mink_max_joint_delta
        delta = np.clip(q_result_np - q_current, -max_delta, max_delta)
        q_out = (q_current + delta).astype(np.float32)
        return q_out

    # -- Internal ------------------------------------------------------------

    @staticmethod
    def _as_q14(q: np.ndarray) -> np.ndarray:
        """Validate joint-vector shape while preserving the caller's dtype.

        Conversion occurs at the arithmetic boundaries; converting earlier would
        change the rounding covered by numerical reference tests.
        """
        arr = np.asarray(q)
        if arr.shape != (N_ARM_JOINTS,):
            raise ValueError(
                f"q must be the 14 arm joints (left s1..w2, right s1..w2), got shape {arr.shape}"
            )
        return arr
