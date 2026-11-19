"""Measure how the lidar is mounted: its height above the floor and its tilt.

With the robot standing still on a level floor, the lidar's accelerometer
gives its roll and pitch, and the floor shows up as the lowest dense layer of
points once the scan is levelled. The position along the floor (x, y) and
the heading (yaw) can't be measured this way; they come from the operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .stream import ImuSample, LidarFrame


@dataclass(frozen=True)
class MountMeasurement:
    height: float  # lidar above the floor (m)
    roll: float  # degrees
    pitch: float  # degrees
    floor_points: int


def _tilt(accel: np.ndarray) -> np.ndarray:
    a = accel / np.linalg.norm(accel)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, z)
    s, c = np.linalg.norm(axis), float(np.dot(a, z))
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()


def measure_mount(
    frames: list[LidarFrame], imu: list[ImuSample], *, yaw_deg: float = 0.0
) -> MountMeasurement:
    """Height and roll/pitch of the lidar, from a still robot on a level floor."""
    if len(imu) < 20:
        raise ValueError("Not enough IMU samples to measure the tilt.")
    accel = np.mean([s.accel for s in imu], axis=0)
    if abs(np.linalg.norm(accel) - 9.81) > 0.5:
        raise ValueError("The robot seems to be moving. Keep it still and try again.")
    tilt = _tilt(accel)
    pts = np.concatenate([f.xyz for f in frames]).astype(np.float64) @ tilt.T
    near = pts[np.hypot(pts[:, 0], pts[:, 1]) < 8.0]
    below = near[near[:, 2] < -0.15, 2]
    if len(below) < 200:
        raise ValueError(
            "Can't see the floor. The lidar must be at least 0.15 m above it, "
            "with floor in view a few metres away."
        )
    lo = float(np.percentile(below, 0.5))
    bins = np.arange(lo - 0.02, lo + 0.30, 0.02)
    counts, edges = np.histogram(below, bins=bins)
    i = int(np.argmax(counts))
    on_floor = below[(below >= edges[i] - 0.02) & (below < edges[i + 1] + 0.02)]
    height = -float(np.median(on_floor))
    r = Rotation.from_matrix(
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ tilt
    )
    _yaw, pitch, roll = r.as_euler("ZYX", degrees=True)
    return MountMeasurement(height, float(roll), float(pitch), len(on_floor))
