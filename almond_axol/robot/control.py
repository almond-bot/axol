"""Motor control utilities: friction model, velocity differentiator, gravity compensation."""

from __future__ import annotations

import math
import re
import time

import mujoco

from ..shared import URDF_PATH

# Matches CUTOFF_FREQUENCY in openarm_constants.hpp
CUTOFF_FREQ = 90.0

# MuJoCo joint names for the 7-DOF arm (shoulder_1 → wrist_3)
_LEFT_JOINT_NAMES = [f"openarm_left_joint{i}" for i in range(1, 8)]
_RIGHT_JOINT_NAMES = [f"openarm_right_joint{i}" for i in range(1, 8)]


def compute_friction(
    velocity: float, Fc: float, k: float, Fv: float, Fo: float
) -> float:
    """Tanh friction model: τ = Fc * tanh(0.1 * k * v) + Fv * v + Fo"""
    return Fc * math.tanh(0.1 * k * velocity) + Fv * velocity + Fo


class Differentiator:
    """
    First-order low-pass differentiator, matching C++ Differentiator::Differentiate.

    For each channel:
        a = 1 / (1 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ
        vel[i] = vel_prev[i] * a + b * (pos[i] - pos_prev[i])
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._vel_prev = [0.0] * n
        self._pos_prev: list[float | None] = [None] * n
        self._last_time: float | None = None

    def differentiate(self, positions: list[float]) -> list[float]:
        now = time.perf_counter()

        if self._last_time is None or any(p is None for p in self._pos_prev):
            self._last_time = now
            self._pos_prev = list(positions)
            return [0.0] * self._n

        Ts = now - self._last_time
        self._last_time = now

        if Ts <= 0:
            return list(self._vel_prev)

        a = 1.0 / (1.0 + Ts * CUTOFF_FREQ)
        b = a * CUTOFF_FREQ

        velocities: list[float] = []
        for i in range(self._n):
            vel = self._vel_prev[i] * a + b * (positions[i] - self._pos_prev[i])  # type: ignore[operator]
            self._vel_prev[i] = vel
            self._pos_prev[i] = positions[i]
            velocities.append(vel)

        return velocities


class GravityCompensator:
    """
    Computes per-joint gravity torques for one 7-DOF arm using MuJoCo.

    Loads the bundled bimanual URDF and extracts the relevant arm joints (1–7).
    With qvel=0, MuJoCo's qfrc_bias equals pure gravity (Coriolis term is zero),
    matching C++ Dynamics::GetGravity.

    Joint coverage: joints 0–6 (shoulder_1 → wrist_3). The gripper is excluded.
    """

    def __init__(self, is_left: bool) -> None:
        joint_names = _LEFT_JOINT_NAMES if is_left else _RIGHT_JOINT_NAMES

        urdf_content = URDF_PATH.read_text()

        # The URDF references .glb visual meshes; redirect to the .stl equivalents
        # that are bundled alongside the URDF.
        urdf_content = urdf_content.replace('.glb"', '.stl"')

        # Load mesh assets relative to the URDF directory so MuJoCo can find them.
        assets: dict[str, bytes] = {}
        urdf_dir = URDF_PATH.parent
        for m in re.finditer(r'filename="([^"]+)"', urdf_content):
            p = urdf_dir / m.group(1)
            if p.is_file():
                assets[p.name] = p.read_bytes()

        self._model = mujoco.MjModel.from_xml_string(urdf_content, assets)  # type: ignore[call-arg]
        self._data = mujoco.MjData(self._model)  # type: ignore[call-arg]

        self._qpos_indices: list[int] = []
        self._dof_indices: list[int] = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore[call-arg]
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in URDF")
            self._qpos_indices.append(int(self._model.jnt_qposadr[jid]))
            self._dof_indices.append(int(self._model.jnt_dofadr[jid]))

    def get_gravity(self, joint_positions_rad: list[float]) -> list[float]:
        """Return gravity torques [Nm] for the 7 arm joints given their positions [rad]."""
        for qpos_idx, pos in zip(self._qpos_indices, joint_positions_rad):
            self._data.qpos[qpos_idx] = pos
        self._data.qvel[:] = 0.0

        mujoco.mj_fwdPosition(self._model, self._data)  # type: ignore[call-arg]
        mujoco.mj_fwdVelocity(self._model, self._data)  # type: ignore[call-arg]

        return [float(self._data.qfrc_bias[dof_idx]) for dof_idx in self._dof_indices]
