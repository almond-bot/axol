"""Small numpy rotation conversions for the UMI recording path.

The training-time relative-EE math lives in :mod:`.processor` (torch, inside
the policy's processor pipeline). This module only holds the conversions the
recorder needs to express tracked poses in the dataset's Cartesian layout
(position + rotation vector, the robot's ``pose6`` convention).
"""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from (..., 4) xyzw quaternion(s), normalized first."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector(s) from (..., 4) xyzw quaternion(s).

    The layout the robot's Cartesian state/actions use (``pose6``: position +
    rotation vector).
    """
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Force w >= 0 so the angle is in [0, pi] (shortest representation).
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    # sin(theta/2) ~= theta/2 near zero: rotvec ~= 2 v.
    scale = np.where(s > 1e-9, angle / np.where(s > 1e-9, s, 1.0), 2.0)
    return v * scale
