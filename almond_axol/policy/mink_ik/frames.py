"""Convert Axol world poses to the bundled Mink model frame."""

from __future__ import annotations

import numpy as np

from .ik import pose6_to_pos_rot_np

# Root-joint origin shared by the Axol world frame and the Mink model.
# The world-to-model transform removes the world frame's +pi/2 yaw.
_ROOT_ORIGIN = np.array([-2.77556e-17, -6.93889e-18, 0.86], dtype=np.float64)
_WORLD_TO_MODEL_R = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def pose6_world_to_model(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode a world-frame pose6 into model-frame ``(position, rotation)``.

    The Axol world frame has a +90 degree root yaw relative to the bundled
    Mink model. Invert that rotation about the root's translated origin for
    positions and apply the same rotation to orientations. Outputs retain
    float32 precision; conversion does not mutate solver state.
    """
    position, rotation = pose6_to_pos_rot_np(pose6)
    model_position = _ROOT_ORIGIN + _WORLD_TO_MODEL_R @ (position - _ROOT_ORIGIN)
    model_rotation = _WORLD_TO_MODEL_R @ rotation
    return model_position.astype(np.float32), model_rotation.astype(np.float32)
