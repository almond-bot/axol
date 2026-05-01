"""MuJoCo-based gravity compensation for the Axol robot.

Computes per-joint gravity torques that account for the full link-chain mass
distribution (i.e. each joint sees the gravity torque from every link distal
of it). This replaces the simplified per-joint ``ga·cos(q) + gb·sin(q)``
model, which only modelled a single mass on each link and ignored the
contribution of all child links.

Gravity is evaluated by setting the joint positions on a MuJoCo model loaded
from the bundled URDF and reading ``qfrc_bias`` with ``qvel=0`` (which equals
the gravitational generalized force vector — Coriolis terms drop out).

The bundled URDF has placeholder Onshape masses on every link (sub-gram
values that are basically zero). The mass and centre-of-mass of each link is
therefore overridden at load time from each :class:`JointConfig`'s ``mass``
and ``com`` fields on an :class:`AxolConfig`. Tune those values to match the
hardware as closely as possible.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mujoco
import numpy as np

from ..shared import (
    ARM_JOINTS,
    URDF_PATH,
    urdf_arm_joint_names,
    urdf_body_name,
)
from .config import AxolConfig

_logger = logging.getLogger(__name__)


__all__ = ["GravityCompensator"]


def _load_urdf_text(urdf_path: Path = URDF_PATH) -> str:
    """Return URDF XML stripped of ``<visual>`` and ``<collision>`` blocks.

    Static gravity comp does not need meshes; stripping them avoids having to
    resolve ``package://`` references when MuJoCo loads the URDF.
    """
    text = urdf_path.read_text()
    text = re.sub(r"<visual>[\s\S]*?</visual>", "", text)
    text = re.sub(r"<collision>[\s\S]*?</collision>", "", text)
    return text


def _body_inertials_from_config(
    config: AxolConfig,
) -> dict[str, tuple[float, tuple[float, float, float]]]:
    """Flatten an AxolConfig into a ``{body_name: (mass, com)}`` dict.

    Each arm joint drives exactly one URDF body (see
    :func:`almond_axol.shared.urdf_body_name`); the mass and CoM are pulled
    straight off the corresponding :class:`JointConfig`.
    """
    out: dict[str, tuple[float, tuple[float, float, float]]] = {}
    for arm, is_left in ((config.left, True), (config.right, False)):
        for joint in ARM_JOINTS:
            jc = getattr(arm, joint.value)
            out[urdf_body_name(joint, is_left=is_left)] = (jc.mass, jc.com)
    return out


def _build_model(config: AxolConfig) -> mujoco.MjModel:
    """Load the Axol URDF into MuJoCo and apply per-body inertial overrides.

    Each override sets the body's mass and CoM in the body's URDF link frame.
    ``body_iquat`` is reset to identity so that the supplied CoM is interpreted
    directly (MuJoCo's URDF importer otherwise expresses CoMs in the
    principal-inertia-axes frame, which is rotated from the link frame whenever
    the URDF inertia tensor is non-isotropic). The inertia tensor is replaced
    with a small isotropic placeholder; this is irrelevant for ``qvel=0``
    gravity but keeps the dynamics well-posed.
    """
    model = mujoco.MjModel.from_xml_string(_load_urdf_text())

    for body_name, (mass, com) in _body_inertials_from_config(config).items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            _logger.warning("Body %r not found in URDF; skipping override.", body_name)
            continue
        model.body_mass[bid] = mass
        model.body_ipos[bid] = com
        model.body_iquat[bid] = (1.0, 0.0, 0.0, 0.0)
        model.body_inertia[bid] = (1e-3, 1e-3, 1e-3)

    return model


class GravityCompensator:
    """Compute per-joint gravity torques for both Axol arms via MuJoCo.

    A single ``MjModel`` / ``MjData`` pair is shared between both arms: the
    arms are independent kinematic chains rooted at the (fixed) trunk, so the
    gravity acting on one arm depends only on its own joint positions. Calls
    are synchronous (no ``await``) so concurrent calls from multiple
    coroutines are safe under asyncio's single-thread execution model.

    Args:
        config: Axol configuration whose per-joint ``mass`` / ``com`` fields
            are written into the MuJoCo model. Defaults to ``AxolConfig()``.
    """

    def __init__(self, config: AxolConfig | None = None) -> None:
        self._model = _build_model(config if config is not None else AxolConfig())
        self._data = mujoco.MjData(self._model)
        self._left_qpos_idx, self._left_dof_idx = self._joint_indices(
            urdf_arm_joint_names(is_left=True)
        )
        self._right_qpos_idx, self._right_dof_idx = self._joint_indices(
            urdf_arm_joint_names(is_left=False)
        )

    def _joint_indices(self, names: list[str]) -> tuple[list[int], list[int]]:
        qpos_idx: list[int] = []
        dof_idx: list[int] = []
        for n in names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise RuntimeError(f"Joint {n!r} not found in URDF")
            qpos_idx.append(int(self._model.jnt_qposadr[jid]))
            dof_idx.append(int(self._model.jnt_dofadr[jid]))
        return qpos_idx, dof_idx

    def gravity(
        self,
        left_q: np.ndarray | None = None,
        right_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(left_gravity, right_gravity)`` torques (Nm) for the 7 arm joints.

        Each input is a ``(7,)`` array of joint positions in radians, in
        :data:`almond_axol.shared.ARM_JOINTS` order (``SHOULDER_1`` →
        ``WRIST_3``); pass ``None`` to skip an arm. Gripper position is
        irrelevant — the gripper joint is fixed in the URDF and its mass is
        already lumped into ``left_w2`` / ``right_w2``.
        """
        if left_q is not None:
            for i, qi in enumerate(self._left_qpos_idx):
                self._data.qpos[qi] = float(left_q[i])
        if right_q is not None:
            for i, qi in enumerate(self._right_qpos_idx):
                self._data.qpos[qi] = float(right_q[i])
        self._data.qvel[:] = 0.0

        mujoco.mj_fwdPosition(self._model, self._data)
        mujoco.mj_fwdVelocity(self._model, self._data)

        left_g = (
            np.array(
                [self._data.qfrc_bias[i] for i in self._left_dof_idx],
                dtype=np.float32,
            )
            if left_q is not None
            else None
        )
        right_g = (
            np.array(
                [self._data.qfrc_bias[i] for i in self._right_dof_idx],
                dtype=np.float32,
            )
            if right_q is not None
            else None
        )
        return left_g, right_g

    def gravity_arm(self, arm_q: np.ndarray, *, is_left: bool) -> np.ndarray:
        """Return gravity torques for a single arm; convenience for per-arm callers.

        Args:
            arm_q: ``(7,)`` array of joint positions in radians,
                :data:`ARM_JOINTS` order.
            is_left: ``True`` to compute gravity for the left arm, ``False``
                for the right.
        """
        if len(arm_q) != len(ARM_JOINTS):
            raise ValueError(
                f"arm_q must have {len(ARM_JOINTS)} elements, got {len(arm_q)}"
            )
        if is_left:
            left, _ = self.gravity(left_q=arm_q)
            assert left is not None
            return left
        _, right = self.gravity(right_q=arm_q)
        assert right is not None
        return right
