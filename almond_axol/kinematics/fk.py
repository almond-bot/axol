"""Lightweight forward kinematics for recording Cartesian end-effector poses.

The full :class:`~almond_axol.kinematics.solver.KinematicsSolver` builds a
self-collision model and JIT-compiles the whole IK graph on init — overkill
when all that's needed is the forward pass that turns cached joint angles into
end-effector poses for the observation. :class:`AxolForwardKinematics` loads
the same URDF and pyroki robot and nothing else, exposing only
:meth:`ee_poses`.

It is built in the control process (where the IK solver otherwise runs
out-of-process in a ``spawn`` child) only when ``observe_cartesian`` is set, so
the caller calls :func:`pin_jax_to_cpu` before the first FK op to stay off the
GPU that the camera relay / policy server need.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy

from ..constants import (
    URDF_PATH,
    Joint,
    urdf_arm_joint_names,
    urdf_body_name,
)
from .jax_cache import enable_persistent_compilation_cache

_logger = logging.getLogger(__name__)


def pin_jax_to_cpu() -> None:
    """Pin in-process JAX (the FK/IK used for Cartesian mode) to the CPU backend.

    The CLI imports JAX early — ``cli.config`` pulls in ``KinematicsConfig``,
    which loads ``kinematics.__init__`` -> ``solver`` -> ``import jax`` — and JAX
    reads ``JAX_PLATFORMS`` into ``jax.config`` at import time. So setting that
    env var at ``connect()`` time is too late: the value is already latched. We
    update ``jax.config`` directly instead, before any backend is initialized
    (i.e. before the first FK/IK op), so the forward/inverse kinematics stay off
    the GPU that the camera relay (NVENC) and policy server need. An explicit
    operator ``JAX_PLATFORMS`` (already reflected in the config) is honored.
    """
    import jax

    if not jax.config.jax_platforms:
        jax.config.update("jax_platforms", "cpu")


# 6-axis end-effector pose layout: Cartesian position (metres) followed by an
# axis-angle rotation vector (radians), both in the robot's world frame (FLU).
# The order matches the per-arm vector returned by :meth:`AxolForwardKinematics.ee_poses`.
EE_AXES: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz")


def _pose6(transform: jaxlie.SE3) -> np.ndarray:
    """Flatten an SE3 to ``[x, y, z, rx, ry, rz]`` (position + rotation vector)."""
    pos = np.asarray(transform.translation(), dtype=np.float32)
    rotvec = np.asarray(transform.rotation().log(), dtype=np.float32)
    return np.concatenate([pos, rotvec])


def pose6_to_pos_rot(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a 6-axis pose into ``(pos, rot_3x3)`` — the inverse of :func:`_pose6`.

    Turns the ``[x, y, z, rx, ry, rz]`` layout (see :data:`EE_AXES`) used for
    Cartesian observations/actions back into the ``(position, rotation matrix)``
    form :meth:`KinematicsSolver.ik` expects for its end-effector targets.
    """
    pose6 = np.asarray(pose6, dtype=np.float32)
    pos = pose6[:3].copy()
    rot = np.asarray(
        jaxlie.SO3.exp(jnp.asarray(pose6[3:6])).as_matrix(), dtype=np.float32
    )
    return pos, rot


class AxolForwardKinematics:
    """Forward kinematics for the Axol end-effectors, no IK or collision model.

    Loads the bundled URDF, builds the pyroki robot, and caches the EE link and
    per-arm joint indices. A single warm-up call JIT-compiles the forward pass
    so the first :meth:`ee_poses` in the observation loop isn't slow.
    """

    def __init__(self) -> None:
        enable_persistent_compilation_cache()
        _logger.info("Loading Axol URDF for forward kinematics...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        self.robot = pk.Robot.from_urdf(urdf)

        names = self.robot.links.names
        self._l_ee_idx = names.index(urdf_body_name(Joint.GRIPPER, is_left=True))
        self._r_ee_idx = names.index(urdf_body_name(Joint.GRIPPER, is_left=False))

        # Map the 7 arm joints (ARM_JOINTS order, gripper excluded) into the full
        # actuated-joint vector by name — pyroki may reorder joints internally.
        actuated = list(self.robot.joints.actuated_names)
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self._left_indices = [
            name_to_idx[n] for n in urdf_arm_joint_names(is_left=True)
        ]
        self._right_indices = [
            name_to_idx[n] for n in urdf_arm_joint_names(is_left=False)
        ]
        self._num_joints = self.robot.joints.num_actuated_joints

        # Warm up the JIT'd forward pass so the first observation isn't slow.
        zeros = np.zeros(len(self._left_indices), dtype=np.float32)
        self.ee_poses(zeros, zeros)
        _logger.info("Forward kinematics ready.")

    def ee_poses(
        self, left_pos: np.ndarray, right_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute both end-effector poses from cached arm joint positions.

        Args:
            left_pos:  Left arm joint positions (radians) in ``Joint`` order. A
                       trailing gripper entry (the robot's 8-vector) is ignored.
            right_pos: Right arm joint positions, same convention.

        Returns:
            ``(left_pose, right_pose)``, each a ``(6,)`` array of
            ``[x, y, z, rx, ry, rz]`` (position + axis-angle rotation vector) in
            the robot's world frame (FLU). See :data:`EE_AXES`.
        """
        n = len(self._left_indices)
        q = np.zeros(self._num_joints, dtype=np.float32)
        q[self._left_indices] = np.asarray(left_pos, dtype=np.float32)[:n]
        q[self._right_indices] = np.asarray(right_pos, dtype=np.float32)[:n]
        fk = self.robot.forward_kinematics(jnp.asarray(q))
        return (
            _pose6(jaxlie.SE3(fk[self._l_ee_idx])),
            _pose6(jaxlie.SE3(fk[self._r_ee_idx])),
        )
