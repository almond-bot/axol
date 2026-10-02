"""Rodrigues axis-angle conversion used by Mink IK.

Preserve float32 coercion and expression order for reproducible pose decoding.
Third-party attribution and license are retained in the adjacent NOTICE and
LICENSE files.
"""

import numpy as np


def aa2rotm(axis_angle) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / (angle + 1e-10)
    x, y, z = axis.tolist()
    axis_hat = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    eye = np.identity(3, dtype=np.float32)
    return eye + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * axis_hat @ axis_hat
