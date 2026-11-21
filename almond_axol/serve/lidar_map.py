"""A live 3D map for the control panel's Sensors page.

:class:`LiveMap` turns lidar frames into the robot's pose and a map to draw
it in, in one of two ways:

- **live** (``source=None``): lidar odometry from where the robot is when
  the map starts. The map grows as the robot drives, like ``axol nav.map``,
  but nothing is saved.
- **saved map** (``source="office"``): localizes on a map made with
  ``axol nav.map``, like ``axol nav.run``, and draws that map.

The map is kept as a set of occupied voxels (:data:`VOXEL` apart), so the
panel only needs the voxels that are new since the last update. Viewers get
it as binary messages (see :func:`encode_map_chunk`); :class:`~.lidar_link.LidarHub`
runs this on its own thread and sends them.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import Any

import numpy as np

from ..lidar.stream import ImuSample, LidarFrame

MAP_MAGIC = b"LDM1"
_MAP_HEADER = struct.Struct("<4sIII")
# Flag on a map chunk: drop everything you have first (a full snapshot).
REPLACE = 1

VOXEL = 0.06
MAX_VOXELS = 1_500_000
# The live map adds a scan after the robot moves or turns this much, and
# every scan for the first few seconds while the lidar fills in its view.
_INSERT_DISTANCE = 0.2
_INSERT_ANGLE = math.radians(8.0)
_INSERT_FIRST = 30
_TRAIL_STEP = 0.1
_TRAIL_MAX = 20_000
# Frames between attempts to find the robot again once it is lost.
_RELOCALIZE_EVERY = 50
_BIAS = 1 << 20
_MASK = (1 << 21) - 1


def encode_map_chunk(epoch: int, points: np.ndarray, *, replace: bool) -> bytes:
    """Pack map points (metres, map frame) as int16 centimetres."""
    cm = np.round(np.asarray(points, np.float64) * 100.0)
    cm = cm[np.all(np.abs(cm) < 32_000, axis=1)].astype("<i2")
    flags = REPLACE if replace else 0
    return (
        _MAP_HEADER.pack(MAP_MAGIC, epoch & 0xFFFFFFFF, flags, len(cm)) + cm.tobytes()
    )


def decode_map_chunk(buf: bytes) -> tuple[int, bool, np.ndarray]:
    """Inverse of :func:`encode_map_chunk` (for tests and tools)."""
    magic, epoch, flags, n = _MAP_HEADER.unpack_from(buf)
    if magic != MAP_MAGIC:
        raise ValueError("not a map chunk")
    pts = np.frombuffer(buf, "<i2", n * 3, _MAP_HEADER.size).reshape(n, 3)
    return epoch, bool(flags & REPLACE), pts.astype(np.float32) / 100.0


def _yaw(pose: np.ndarray) -> float:
    return math.atan2(pose[1, 0], pose[0, 0])


class LiveMap:
    """The robot's pose and a voxel map, from lidar frames. See the module docs.

    Args:
        t_base_lidar: The lidar mount (4x4 ``T_base_lidar``).
        source: A saved map's name to localize on, or None to map live.
        voxel: Spacing of the drawn map (m).
        odometry: Builds the odometry (tests swap it).
        localizer: Builds the localizer from a loaded map (tests swap it).
        load_map: Loads a saved map by name.
    """

    def __init__(
        self,
        t_base_lidar: np.ndarray,
        source: str | None = None,
        *,
        voxel: float = VOXEL,
        max_voxels: int = MAX_VOXELS,
        odometry: Callable[[np.ndarray], Any] | None = None,
        localizer: Callable[[Any, np.ndarray], Any] | None = None,
        load_map: Callable[[str], Any] | None = None,
    ) -> None:
        self.source = source
        self.voxel = voxel
        self.max_voxels = max_voxels
        self.pose: np.ndarray | None = None
        self.state = "starting"
        self.match: float | None = None
        self.frames = 0
        self.distance = 0.0
        self.full = False
        self.trail: list[tuple[float, float, float]] = []
        self._keys: set[int] = set()
        self._pending: list[np.ndarray] = []
        self._last_insert: np.ndarray | None = None
        self._lost_frames = 0
        self._odometry = None
        self._localizer = None
        if source is None:
            if odometry is None:
                from ..nav.odometry import LidarOdometry

                odometry = LidarOdometry
            self._odometry = odometry(t_base_lidar)
        else:
            if load_map is None:
                from ..nav.mapping import Map

                load_map = Map.load
            if localizer is None:
                from ..nav.localization import Localizer

                localizer = Localizer
            nav_map = load_map(source)
            self._localizer = localizer(nav_map, t_base_lidar)
            self._add(np.asarray(nav_map.cloud, np.float64))

    # -- voxels ----------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._keys)

    def _add(self, points_map: np.ndarray) -> None:
        if self.full or not len(points_map):
            return
        idx = np.floor(points_map / self.voxel).astype(np.int64)
        idx = idx[np.all(np.abs(idx) < _BIAS - 1, axis=1)] + _BIAS
        keys = np.unique((idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2])
        have = self._keys
        new = [k for k in keys.tolist() if k not in have]
        room = self.max_voxels - len(have)
        if len(new) > room:
            new = new[: max(room, 0)]
            self.full = True
        if not new:
            return
        have.update(new)
        self._pending.append(self._centers(np.array(new, np.int64)))

    def _centers(self, keys: np.ndarray) -> np.ndarray:
        idx = np.stack([(keys >> 42) & _MASK, (keys >> 21) & _MASK, keys & _MASK], 1)
        return ((idx - _BIAS + 0.5) * self.voxel).astype(np.float32)

    def take_new(self) -> np.ndarray:
        """Voxels added since the last call (map frame, ``(n, 3)``)."""
        if not self._pending:
            return np.zeros((0, 3), np.float32)
        out = np.concatenate(self._pending)
        self._pending = []
        return out

    def all_points(self) -> np.ndarray:
        """Every voxel in the map."""
        if not self._keys:
            return np.zeros((0, 3), np.float32)
        return self._centers(np.fromiter(self._keys, np.int64, len(self._keys)))

    # -- frames ----------------------------------------------------------

    def update(self, frame: LidarFrame, imu: list[ImuSample] | None = None) -> None:
        """Track the robot through one frame and grow the map."""
        self.frames += 1
        if self._localizer is not None:
            pose = self._localize(frame, imu)
        else:
            result = self._odometry.update(frame, imu)
            pose = result.pose
            self.state = "mapping"
            self.match = float(result.inlier_ratio)
            if self._should_insert(pose):
                self._last_insert = pose.copy()
                pts = np.asarray(result.points, np.float64)
                self._add(pts @ pose[:3, :3].T + pose[:3, 3])
        self._track(pose)

    def _localize(self, frame: LidarFrame, imu: list[ImuSample] | None) -> np.ndarray:
        from ..nav.geometry import pose2d

        loc = self._localizer
        if loc.last is None:
            fix = loc.initialize(frame, imu)
        elif loc.state == "lost":
            self._lost_frames += 1
            if self._lost_frames % _RELOCALIZE_EVERY == 0:
                fix = loc.initialize(frame, imu, guess=pose2d(loc.last.pose))
            else:
                fix = loc.update(frame, imu)
        else:
            self._lost_frames = 0
            fix = loc.update(frame, imu)
        self.state = fix.state
        self.match = float(fix.match_ratio)
        return fix.pose

    def _should_insert(self, pose: np.ndarray) -> bool:
        last = self._last_insert
        if last is None or self.frames <= _INSERT_FIRST:
            return True
        moved = float(np.linalg.norm(pose[:3, 3] - last[:3, 3]))
        turned = abs(math.remainder(_yaw(pose) - _yaw(last), 2 * math.pi))
        return moved >= _INSERT_DISTANCE or turned >= _INSERT_ANGLE

    def _track(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, float)
        here = tuple(float(v) for v in self.pose[:3, 3])
        if self.trail:
            step = math.dist(here[:2], self.trail[-1][:2])
            if step < _TRAIL_STEP:
                return
            # A jump (relocalized) isn't driving.
            if step < 2.0:
                self.distance += step
        self.trail.append(here)
        if len(self.trail) > _TRAIL_MAX:
            del self.trail[: len(self.trail) - _TRAIL_MAX]

    # -- messages --------------------------------------------------------

    def pose_message(self, epoch: int) -> dict[str, Any] | None:
        if self.pose is None:
            return None
        return {
            "type": "pose",
            "epoch": epoch,
            # Rows of T_map_base without the last (0, 0, 0, 1).
            "m": [round(float(v), 4) for v in self.pose[:3].ravel()],
            "state": self.state,
            "match": None if self.match is None else round(self.match, 3),
        }

    def trail_message(self, epoch: int) -> dict[str, Any]:
        step = max(1, len(self.trail) // 5000)
        return {
            "type": "trail",
            "epoch": epoch,
            "points": [[round(v, 3) for v in p] for p in self.trail[::step]],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "points": self.size,
            "distance": round(self.distance, 2),
            "full": self.full,
        }
