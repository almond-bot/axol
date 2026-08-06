"""IK backend interface.

Every solver backend (pink-qp, pyroki-lm, pyroki-diff, mink-qp, dls) exposes
the same joint-vector convention and call signature so
:class:`almond_axol.teleop.worker.IKWorker` (and the offline benchmark) can
swap them freely.

Canonical joint order
---------------------
All ``q`` vectors crossing the backend boundary are ``(14,)`` arrays in
**canonical order**: the 7 left-arm joints in ``ARM_JOINTS`` order
(shoulder_1 … wrist_3) followed by the 7 right-arm joints. Backends whose
underlying model orders joints differently (pyroki sorts topologically)
permute internally.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from ..constants import Joint, urdf_arm_joint_names, urdf_body_name

CANONICAL_JOINT_NAMES: tuple[str, ...] = tuple(
    urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
)
"""URDF joint names in canonical order (left arm then right arm)."""


def frame_body_names() -> dict[str, str]:
    """URDF body names for the frames the teleop layer tracks."""
    return {
        "left_ee": urdf_body_name(Joint.GRIPPER, is_left=True),
        "right_ee": urdf_body_name(Joint.GRIPPER, is_left=False),
        "left_elbow": urdf_body_name(Joint.ELBOW, is_left=True),
        "right_elbow": urdf_body_name(Joint.ELBOW, is_left=False),
        "left_shoulder": urdf_body_name(Joint.SHOULDER_1, is_left=True),
        "right_shoulder": urdf_body_name(Joint.SHOULDER_1, is_left=False),
    }


NUM_JOINTS: int = len(CANONICAL_JOINT_NAMES)

LEFT_INDICES: list[int] = list(range(7))
RIGHT_INDICES: list[int] = list(range(7, 14))


@dataclass
class FKFrames:
    """Poses of the frames the teleop layer cares about, in world frame (FLU).

    Attributes:
        left_ee / right_ee: ``(pos_3, rot_3x3)`` gripper poses.
        left_elbow / right_elbow: ``(3,)`` elbow (forearm-root) positions.
    """

    left_ee: tuple[np.ndarray, np.ndarray]
    right_ee: tuple[np.ndarray, np.ndarray]
    left_elbow: np.ndarray
    right_elbow: np.ndarray


class IKBackend(abc.ABC):
    """Common interface every teleop IK solver implements.

    Attributes:
        left_indices / right_indices: Per-arm indices into the canonical
            ``(14,)`` joint vector, in ``ARM_JOINTS`` order. Constant across
            backends (``0..6`` / ``7..13``) but kept as attributes because the
            teleop handshake ships them to the control process.
        left_shoulder_pos / right_shoulder_pos: Fixed world-frame shoulder
            positions (m), used by the target-conditioning layer.
    """

    name: str = ""

    left_indices: list[int] = LEFT_INDICES
    right_indices: list[int] = RIGHT_INDICES

    left_shoulder_pos: np.ndarray
    right_shoulder_pos: np.ndarray

    @property
    def num_joints(self) -> int:
        """Length of the canonical joint vector (14)."""
        return NUM_JOINTS

    @abc.abstractmethod
    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the joint solution one tick toward the Cartesian targets.

        Args:
            q_current: Canonical ``(14,)`` joint vector (radians), the seed /
                integration state.
            left_pose: ``(pos_3, rot_3x3)`` world-frame target for the left
                gripper, or ``None`` to leave that arm unconstrained.
            right_pose: Same for the right gripper.
            left_elbow_pos: Optional ``(3,)`` world-frame elbow position hint.
            right_elbow_pos: Same for the right elbow.

        Returns:
            Updated canonical ``(14,)`` joint vector.
        """

    @abc.abstractmethod
    def fk_frames(self, q: np.ndarray) -> FKFrames:
        """Compute gripper poses + elbow positions for a canonical ``q``."""

    @abc.abstractmethod
    def set_posture_pose(self, q: np.ndarray) -> None:
        """Set the preferred-posture attractor (canonical ``(14,)``, radians)."""

    def settle_rest_pose(self, q_rest: np.ndarray) -> np.ndarray:
        """Return the solver's fixed point nearest to ``q_rest``.

        Differential backends are already at a fixed point when posture ==
        rest, so the default is the identity. The pyroki-LM backend overrides
        this to bake in its manipulability-cost null-space settling.
        """
        return np.asarray(q_rest, dtype=np.float32).copy()
