"""Lightweight forward kinematics over the bundled Axol URDF.

A thin wrapper around the pyroki robot model that computes per-side gripper
poses from 8-value joint vectors (7 arm joints + gripper, Joint enum order)
without constructing the full :class:`KinematicsSolver` — no jaxls problem
build, no IK JIT warmup. Used by the hand-eye calibration sweep and the
policy executor, both of which need FK only.
"""

from __future__ import annotations

import numpy as np


class ArmFK:
    """Per-side gripper-frame FK for the Axol.

    Example::

        fk = ArmFK()
        rot, pos = fk.gripper_pose("left", q_left_8)
    """

    def __init__(self) -> None:
        import jax.numpy as jnp  # noqa: F401 - ensure jax initialised before pyroki
        import pyroki as pk
        import yourdfpy

        from ..constants import (
            URDF_PATH,
            Joint,
            urdf_arm_joint_names,
            urdf_body_name,
        )

        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        self._robot = pk.Robot.from_urdf(urdf)
        link_names = self._robot.links.names
        self._ee_idx = {
            "left": link_names.index(urdf_body_name(Joint.GRIPPER, is_left=True)),
            "right": link_names.index(urdf_body_name(Joint.GRIPPER, is_left=False)),
        }
        actuated = list(self._robot.joints.actuated_names)
        name_to_idx = {n: i for i, n in enumerate(actuated)}
        self._joint_indices = {
            "left": [name_to_idx[n] for n in urdf_arm_joint_names(is_left=True)],
            "right": [name_to_idx[n] for n in urdf_arm_joint_names(is_left=False)],
        }
        self._num_joints = self._robot.joints.num_actuated_joints

    def gripper_pose(
        self, side: str, q_arm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base-frame gripper pose ``(R_3x3, t_3)`` for one side.

        Args:
            side: ``"left"`` or ``"right"``.
            q_arm: ``(>=7,)`` joint positions (rad) in Joint enum order; only
                the 7 arm joints are used.
        """
        import jax.numpy as jnp
        import jaxlie

        q = np.zeros(self._num_joints, dtype=np.float32)
        for i, gi in enumerate(self._joint_indices[side]):
            q[gi] = float(q_arm[i])
        fk = self._robot.forward_kinematics(jnp.asarray(q))
        T = jaxlie.SE3(fk[self._ee_idx[side]])
        return (
            np.asarray(T.rotation().as_matrix(), dtype=np.float64),
            np.asarray(T.translation(), dtype=np.float64),
        )
