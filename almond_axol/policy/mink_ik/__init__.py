"""Mink IK for Axol, independent of JAX and hardware access.

The solver owns mutable MuJoCo data and belongs to one control thread.
Its poses use the bundled model frame; callers using Axol world poses
convert them at the boundary with ``pose6_world_to_model``. See
``PROVENANCE.md`` for numerical reference tests and runtime requirements.
"""

from .frames import pose6_world_to_model
from .ik import MinkIK, pose6_to_pos_rot_np
from .ik_config import PINNED_URDF, MinkIKConfig

__all__ = [
    "PINNED_URDF",
    "MinkIK",
    "MinkIKConfig",
    "pose6_world_to_model",
    "pose6_to_pos_rot_np",
]
