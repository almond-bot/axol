"""Lidar configuration: where the lidar is, how it is mounted, how to filter it.

``axol lidar.setup`` writes the detected network values to
``~/.almond/lidar.json``; every lidar and navigation command reads that file
for its defaults, so the robot is configured once. The mount (where the lidar
sits on the robot) is set with ``axol lidar.setup --mount X Y Z ROLL PITCH
YAW`` or by editing the file.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from ..utils.paths import almond_path
from ..utils.state_files import secure_atomic_write_json, secure_read_text
from . import livox

CONFIG_FILENAME = "lidar.json"


def config_path() -> Path:
    return almond_path(CONFIG_FILENAME)


@dataclass
class LidarMount:
    """Pose of the lidar in the robot's base frame (+x forward, +y left, +z up).

    Translation in metres, rotation as roll/pitch/yaw in degrees (applied
    yaw, then pitch, then roll, the usual ``Rz·Ry·Rx``). The base frame's
    origin is the centre of the base on the floor.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def matrix(self) -> np.ndarray:
        """4x4 ``T_base_lidar``: maps lidar-frame points into the base frame."""
        return pose_matrix(
            self.x,
            self.y,
            self.z,
            math.radians(self.roll),
            math.radians(self.pitch),
            math.radians(self.yaw),
        )


def pose_matrix(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = (x, y, z)
    return out


@dataclass
class LidarConfig:
    """How to reach and read the lidar.

    Attributes:
        lidar_ip:    The lidar's address. Empty means "discover it".
        host_ip:     This computer's address on the lidar's port. The lidar
                     sends its point and IMU streams here.
        iface:       Network interface the lidar is plugged into.
        serial:      Serial number recorded at setup (informational).
        point_port:  UDP port this computer receives points on.
        imu_port:    UDP port this computer receives IMU samples on.
        frame_period: Seconds of points per frame. The Mid-360 covers its
                     full field of view in about 0.1 s.
        min_range:   Drop points closer than this (m). The Mid-360 can't
                     measure under 0.1 m.
        max_range:   Drop points farther than this (m).
        mount:       Where the lidar sits on the robot.
    """

    lidar_ip: str = ""
    host_ip: str = ""
    iface: str = ""
    serial: str = ""
    point_port: int = livox.HOST_POINT_PORT
    imu_port: int = livox.HOST_IMU_PORT
    frame_period: float = 0.1
    min_range: float = 0.15
    max_range: float = 40.0
    mount: LidarMount = field(default_factory=LidarMount)

    @classmethod
    def load(cls, path: str | Path | None = None) -> LidarConfig:
        """Read the saved config; a missing file gives the defaults."""
        p = Path(path).expanduser() if path is not None else config_path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(secure_read_text(p))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> LidarConfig:
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known and k != "mount"}
        mount_data = data.get("mount") or {}
        mount_known = {f.name for f in fields(LidarMount)}
        mount = LidarMount(
            **{k: float(v) for k, v in mount_data.items() if k in mount_known}
        )
        return cls(**values, mount=mount)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path).expanduser() if path is not None else config_path()
        secure_atomic_write_json(p, asdict(self), sort_keys=False)
        return p
