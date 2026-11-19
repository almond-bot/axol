"""Small rigid-transform helpers for navigation (4x4 matrices, planar poses)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Pose2D:
    """A planar pose: position in metres and heading in radians (CCW from +x)."""

    x: float
    y: float
    yaw: float

    def matrix(self, z: float = 0.0) -> np.ndarray:
        return planar_matrix(self.x, self.y, self.yaw, z)

    def distance_to(self, other: Pose2D) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def heading_error_to(self, other: Pose2D) -> float:
        return wrap_angle(other.yaw - self.yaw)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def planar_matrix(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    out[:3, 3] = (x, y, z)
    return out


def pose2d(t: np.ndarray) -> Pose2D:
    """Project a 4x4 pose onto the floor plane (heading from the x axis)."""
    return Pose2D(
        float(t[0, 3]), float(t[1, 3]), math.atan2(float(t[1, 0]), float(t[0, 0]))
    )


def inverse(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    r = t[:3, :3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t[:3, 3]
    return out


def transform_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ t[:3, :3].T + t[:3, 3]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation matrix taking direction ``a`` onto direction ``b``."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180°: any axis perpendicular to a.
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(math.pi * perp / np.linalg.norm(perp)).as_matrix()
    return Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()


def interpolate_motion(
    delta: np.ndarray, fractions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotations and translations at ``fractions`` (0..1) of a relative motion.

    Returns ``(rotations (n,3,3), translations (n,3))`` for a constant-velocity
    path from identity to ``delta``. Used to undo the motion within a scan.
    """
    rotvec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    rots = Rotation.from_rotvec(fractions[:, None] * rotvec[None, :]).as_matrix()
    trans = fractions[:, None] * delta[:3, 3][None, :]
    return rots, trans
