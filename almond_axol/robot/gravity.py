"""MuJoCo-based gravity compensation for the Axol robot.

Computes per-joint gravity torques that account for the full link-chain mass
distribution (i.e. each joint sees the gravity torque from every link distal
of it). This replaces the simplified per-joint ``ga·cos(q) + gb·sin(q)``
model, which only modelled a single mass on each link and ignored the
contribution of all child links.

Gravity is evaluated by setting the joint positions on a MuJoCo model loaded
from the bundled URDF and reading ``qfrc_bias`` with ``qvel=0`` (which equals
the gravitational generalized force vector — Coriolis terms drop out).

The model is always loaded from the classic URDF: every Axol version shares
the same arm link frames, joint axes and (placeholder) inertials — the mobile
URDF copies them verbatim, and ``tests/test_urdf_contract.py`` checks it — so
gravity torques, and the ``mass`` / ``com`` parameters tuned for them, are the
same on both versions.

The bundled URDF has placeholder Onshape masses on every link (sub-gram
values that are basically zero). The mass and centre-of-mass of each link is
therefore overridden at load time from each :class:`JointConfig`'s ``mass``
and ``com`` fields on an :class:`AxolConfig`. Tune those values to match the
hardware as closely as possible.

A *payload* — whatever the gripper is holding right now — is layered on top
at runtime with :meth:`GravityCompensator.set_payload`: it is lumped into the
``wrist_3`` body (which already carries the gripper) as a point mass, so every
joint's gravity feedforward accounts for it without touching the calibrated
link inertials underneath.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from ..constants import (
    ARM_JOINTS,
    URDF_PATH,
    Joint,
    urdf_arm_joint_names,
    urdf_body_name,
)
from .config import AxolConfig

_logger = logging.getLogger(__name__)


__all__ = [
    "MAX_PAYLOAD_KG",
    "GravityCompensator",
    "PayloadSide",
    "payload_sides",
    "validate_payload",
]

PayloadSide = Literal["left", "right", "both"]

# Upper bound accepted by ``set_payload``. A guard against unit slips — a
# payload given in grams would otherwise become a feedforward large enough to
# throw the arm upward — not a rated payload.
MAX_PAYLOAD_KG = 5.0

# Largest accepted distance (m) from the gripper mount to a payload's CoM —
# roughly 3x the fingertip distance, again only a typo guard.
_MAX_PAYLOAD_REACH = 0.5


def _load_urdf_text(urdf_path: Path = URDF_PATH) -> str:
    """Return URDF XML stripped of ``<visual>`` and ``<collision>`` blocks.

    Static gravity comp does not need meshes; stripping them avoids having to
    resolve ``package://`` references when MuJoCo loads the URDF.
    """
    text = urdf_path.read_text()
    text = re.sub(r"<visual>[\s\S]*?</visual>", "", text)
    text = re.sub(r"<collision>[\s\S]*?</collision>", "", text)
    return text


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw as a rotation matrix (``Rz · Ry · Rx``)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _gripper_mount(urdf_text: str, *, is_left: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pose ``(R, p)`` of the gripper link in its parent (``wrist_3``) body frame.

    The gripper hangs off ``wrist_3``'s body on a fixed joint, which MuJoCo's
    URDF importer fuses away, so the transform is read from the URDF itself.
    """
    parent = urdf_body_name(Joint.WRIST_3, is_left=is_left)
    child = urdf_body_name(Joint.GRIPPER, is_left=is_left)
    for joint in ET.fromstring(urdf_text).iter("joint"):
        p_el, c_el = joint.find("parent"), joint.find("child")
        if (
            p_el is None
            or c_el is None
            or p_el.get("link") != parent
            or c_el.get("link") != child
        ):
            continue
        origin = joint.find("origin")
        attrs = origin.attrib if origin is not None else {}
        xyz = [float(v) for v in attrs.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in attrs.get("rpy", "0 0 0").split()]
        return _rpy_matrix(*rpy), np.array(xyz)
    raise RuntimeError(f"No joint {parent!r} -> {child!r} in the URDF")


def validate_payload(mass: float, com: Sequence[float]) -> tuple[float, np.ndarray]:
    """Check a payload's mass (kg) and CoM (m), returning them normalised.

    Raises:
        ValueError: ``mass`` is negative, non-finite or above
            :data:`MAX_PAYLOAD_KG`, or ``com`` is not three finite numbers
            within reach of the gripper.
    """
    if isinstance(mass, bool) or not isinstance(mass, numbers.Real):
        raise ValueError(f"payload mass must be a number of kg, got {mass!r}")
    mass = float(mass)
    if not (math.isfinite(mass) and 0.0 <= mass <= MAX_PAYLOAD_KG):
        raise ValueError(
            f"payload mass must be between 0 and {MAX_PAYLOAD_KG:g} kg, got {mass!r}"
        )
    com_arr = np.asarray(com, dtype=np.float64)
    if com_arr.shape != (3,) or not np.all(np.isfinite(com_arr)):
        raise ValueError(f"payload com must be three finite numbers (m), got {com!r}")
    if float(np.linalg.norm(com_arr)) > _MAX_PAYLOAD_REACH:
        raise ValueError(
            f"payload com {com_arr.tolist()} is more than {_MAX_PAYLOAD_REACH:g} m "
            "from the gripper mount; it is in metres in the gripper link frame"
        )
    return mass, com_arr


def payload_sides(side: str) -> tuple[bool, ...]:
    """The ``is_left`` flags a ``set_payload`` ``side`` argument names."""
    sides = {"left": (True,), "right": (False,), "both": (True, False)}
    if side not in sides:
        raise ValueError(f"side must be 'left', 'right' or 'both', got {side!r}")
    return sides[side]


def _body_inertials_from_config(
    config: AxolConfig,
) -> dict[str, tuple[float, tuple[float, float, float]]]:
    """Flatten an AxolConfig into a ``{body_name: (mass, com)}`` dict.

    Each arm joint drives exactly one URDF body (see
    :func:`almond_axol.constants.urdf_body_name`); the mass and CoM are pulled
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
        # Payload bookkeeping, per arm (keyed by is_left): the wrist_3 body
        # the payload is lumped into, that body's configured inertial (what
        # a zero payload restores), the gripper mount pose the payload CoM is
        # given relative to, and the payload currently applied.
        urdf_text = _load_urdf_text()
        self._wrist_bid: dict[bool, int] = {}
        self._wrist_base: dict[bool, tuple[float, np.ndarray]] = {}
        self._mount: dict[bool, tuple[np.ndarray, np.ndarray]] = {}
        self._payload: dict[bool, tuple[float, np.ndarray]] = {}
        for is_left in (True, False):
            name = urdf_body_name(Joint.WRIST_3, is_left=is_left)
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise RuntimeError(f"Body {name!r} not found in URDF")
            self._wrist_bid[is_left] = bid
            self._wrist_base[is_left] = (
                float(self._model.body_mass[bid]),
                np.array(self._model.body_ipos[bid], dtype=np.float64),
            )
            self._mount[is_left] = _gripper_mount(urdf_text, is_left=is_left)
            self._payload[is_left] = (0.0, np.zeros(3))
        self._left_qpos_idx, self._left_dof_idx = self._joint_indices(
            urdf_arm_joint_names(is_left=True)
        )
        self._right_qpos_idx, self._right_dof_idx = self._joint_indices(
            urdf_arm_joint_names(is_left=False)
        )
        # Scratch buffer for expanding MuJoCo's sparse qM into a dense matrix
        # (see gravity_and_inertia_arm). nv is small (14), so this is cheap.
        self._m_full = np.zeros((self._model.nv, self._model.nv))

    def set_payload(
        self, mass: float, com: Sequence[float] = (0.0, 0.0, 0.0), *, is_left: bool
    ) -> None:
        """Model a payload held by one arm's gripper.

        The payload is a point mass lumped into the ``wrist_3`` body (which
        already carries the gripper), so every joint's gravity torque — and
        the reflected inertia used for damping scheduling — includes it from
        the next evaluation on. Replaces any payload set before; ``mass=0``
        removes it, restoring the configured link inertial exactly.

        Args:
            mass: Payload mass (kg), ``0`` to :data:`MAX_PAYLOAD_KG`.
            com: Payload centre of mass (m) in the gripper link frame — the
                end-effector frame forward kinematics reports. The
                fingertips are :data:`~almond_axol.constants.GRIPPER_TIP_OFFSET`.
            is_left: ``True`` for the left arm, ``False`` for the right.

        Raises:
            ValueError: see :func:`validate_payload`.
        """
        mass, com_grip = validate_payload(mass, com)
        rot, pos = self._mount[is_left]
        com_body = rot @ com_grip + pos
        base_mass, base_com = self._wrist_base[is_left]
        total = base_mass + mass
        bid = self._wrist_bid[is_left]
        self._model.body_mass[bid] = total
        self._model.body_ipos[bid] = (
            (base_mass * base_com + mass * com_body) / total
            if total > 0.0
            else base_com
        )
        self._payload[is_left] = (mass, com_grip)
        _logger.info(
            "%s payload: %.3f kg at %s m (gripper frame)",
            "left" if is_left else "right",
            mass,
            np.round(com_grip, 4).tolist(),
        )

    def payload(self, *, is_left: bool) -> tuple[float, np.ndarray]:
        """The payload set on one arm: ``(mass_kg, com_m)`` in the gripper frame."""
        mass, com = self._payload[is_left]
        return mass, com.copy()

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
        :data:`almond_axol.constants.ARM_JOINTS` order (``SHOULDER_1`` →
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

    def gravity_and_inertia_arm(
        self, arm_q: np.ndarray, *, is_left: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gravity torques *and* reflected inertia for one arm, in one pass.

        The second array is the diagonal of the joint-space mass matrix
        (kg·m², one entry per arm joint): how much inertia each joint
        actually "sees" at this configuration. It varies strongly with pose —
        e.g. shoulder_1's entry collapses toward zero when the arm is raised
        to the side (the mass then lies along its axis) and is maximal with
        the arm hanging. The production controller uses it to schedule
        host-side damping (see ``AxolArm.motion_control``).

        Both arrays come from the same MuJoCo forward pass as
        :meth:`gravity_arm`, so calling this instead costs almost nothing
        extra. Off-diagonal coupling terms are ignored — for damping
        scheduling only the joint's own reflected inertia matters.
        """
        gravity = self.gravity_arm(arm_q, is_left=is_left)
        # mj_fwdPosition (run inside gravity()) already computed the sparse
        # factorized mass matrix; expand it and pull this arm's diagonal.
        # MuJoCo 3.10 changes this call's signature and 3.11 drops qM: the
        # pyproject `mujoco<3.10` bound is what keeps this valid.
        mujoco.mj_fullM(self._model, self._m_full, self._data.qM)
        dof_idx = self._left_dof_idx if is_left else self._right_dof_idx
        inertia = np.array([self._m_full[i, i] for i in dof_idx], dtype=np.float32)
        return gravity, inertia
