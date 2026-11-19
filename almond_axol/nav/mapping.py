"""Build, save and load maps.

A map has two layers, both in the *map frame* (the base's pose when mapping
started, +z up, floor at z = 0):

- a 3D point cloud (``cloud``, thinned to :attr:`MapConfig.cloud_voxel`) that
  :mod:`.localization` registers live scans against;
- a 2D occupancy grid (:class:`OccupancyGrid`) that :mod:`.planner` plans on.
  A cell is *occupied* when lidar points between :attr:`MapConfig.band_low`
  and :attr:`MapConfig.band_high` above the floor land in it (anything the
  robot could hit), *free* when a scan ray crossed it or the robot drove
  over it, and *unknown* otherwise.

Maps live in ``~/.almond/maps/<name>/``: ``map.npz`` (both layers),
``map.json`` (metadata), ``map.png`` (a picture of the grid) and
``waypoints.json`` (see :mod:`.waypoints`).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..utils.paths import almond_path
from ..utils.state_files import (
    secure_atomic_write_bytes,
    secure_atomic_write_json,
    secure_read_text,
)
from .geometry import Pose2D, inverse, pose2d, transform_points

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def maps_dir() -> Path:
    return almond_path("maps")


def map_path(name: str) -> Path:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Map name {name!r} isn't allowed. Use letters, digits, '-', '_' and '.'."
        )
    return maps_dir() / name


def list_maps() -> list[str]:
    root = maps_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "map.npz").exists())


@dataclass
class MapConfig:
    """How a map is built.

    Attributes:
        resolution:   Grid cell size (m).
        cloud_voxel:  Point spacing (m) of the saved 3D cloud.
        band_low:     Points at least this high above the floor are obstacles (m).
        band_high:    Points above this are overhead and ignored (m). Set it to
                      the robot's height.
        keyframe_distance: Add a keyframe after moving this far (m).
        keyframe_angle: ...or turning this much (deg).
        footprint_radius: Cells within this distance of the driven path are
                      marked free: the robot was there (m).
        min_hits:     Keyframes that must see something in a cell for it to
                      count as occupied.
        keyframe_frames: Frames merged into a keyframe while the robot stays
                      near it. The Mid-360 fills in its view over about a
                      second, so this makes keyframes denser.
    """

    resolution: float = 0.05
    cloud_voxel: float = 0.1
    band_low: float = 0.10
    band_high: float = 1.90
    keyframe_distance: float = 0.25
    keyframe_angle: float = 10.0
    footprint_radius: float = 0.25
    min_hits: int = 1
    keyframe_frames: int = 10


@dataclass
class OccupancyGrid:
    """A 2D grid: ``data[row, col]`` covers the cell whose lower-left corner is
    ``(origin_x + col * resolution, origin_y + row * resolution)``."""

    data: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def to_cell(self, x: float | np.ndarray, y: float | np.ndarray):
        col = np.floor((np.asarray(x) - self.origin_x) / self.resolution).astype(int)
        row = np.floor((np.asarray(y) - self.origin_y) / self.resolution).astype(int)
        return row, col

    def to_world(self, row, col):
        x = self.origin_x + (np.asarray(col) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(row) + 0.5) * self.resolution
        return x, y

    def inside(self, row, col) -> np.ndarray:
        h, w = self.shape
        return (
            (np.asarray(row) >= 0)
            & (np.asarray(row) < h)
            & (np.asarray(col) >= 0)
            & (np.asarray(col) < w)
        )

    def value_at(self, x: float, y: float) -> int:
        r, c = self.to_cell(x, y)
        if not self.inside(r, c):
            return UNKNOWN
        return int(self.data[r, c])

    def image(self) -> np.ndarray:
        """RGB picture (row 0 at the top, so +y points up)."""
        palette = np.array([[128, 128, 128], [250, 250, 250], [20, 20, 20]], np.uint8)
        return palette[np.flipud(self.data)]


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """One point (the centroid) per occupied voxel."""
    if len(pts) == 0:
        return pts.reshape(0, 3)
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    return (out / counts[:, None]).astype(np.float32)


@dataclass
class Keyframe:
    pose: np.ndarray  # T_map_base
    points: np.ndarray  # base frame, float32
    frames: int = 1


def _virtual_scan(
    pts_base: np.ndarray, band: tuple[float, float], bins: int = 720
) -> tuple[np.ndarray, np.ndarray]:
    """Per azimuth bin: range to the nearest obstacle, and the farthest
    range seen clear (floor hits or the obstacle itself)."""
    lo, hi = band
    az = np.arctan2(pts_base[:, 1], pts_base[:, 0])
    r = np.hypot(pts_base[:, 0], pts_base[:, 1])
    b = ((az + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    z = pts_base[:, 2]
    obstacle = (z >= lo) & (z <= hi)
    near = np.full(bins, np.inf)
    np.minimum.at(near, b[obstacle], r[obstacle])
    floor = z < lo
    clear = np.zeros(bins)
    np.maximum.at(clear, b[floor], r[floor])
    return near, clear


class MapBuilder:
    """Accumulate registered scans into a map (see the module docs)."""

    def __init__(self, config: MapConfig | None = None) -> None:
        self.config = config or MapConfig()
        self.keyframes: list[Keyframe] = []
        self.trajectory: list[np.ndarray] = []
        self._cloud: list[np.ndarray] = []
        self._cloud_points = 0
        self._merged: np.ndarray = np.zeros((0, 3), np.float32)

    def add(self, pose: np.ndarray, points_base: np.ndarray) -> bool:
        """Offer a registered scan; returns True when it became a keyframe."""
        self.trajectory.append(pose[:3, 3].copy())
        if self.keyframes:
            last = self.keyframes[-1]
            moved = np.linalg.norm(pose[:2, 3] - last.pose[:2, 3])
            dyaw = abs(pose2d(pose).heading_error_to(pose2d(last.pose)))
            if (
                moved < self.config.keyframe_distance
                and math.degrees(dyaw) < self.config.keyframe_angle
            ):
                if last.frames < self.config.keyframe_frames:
                    # Still near the keyframe: fold this scan into it.
                    rel = inverse(last.pose) @ pose
                    merged = np.concatenate(
                        [
                            last.points.astype(np.float64),
                            transform_points(rel, points_base.astype(np.float64)),
                        ]
                    )
                    last.points = voxel_downsample(merged, self.config.cloud_voxel / 2)
                    last.frames += 1
                    self._cloud.append(
                        transform_points(pose, points_base.astype(np.float64)).astype(
                            np.float32
                        )
                    )
                    self._cloud_points += len(points_base)
                    if self._cloud_points > 400_000:
                        self._compact()
                return False
        pts = voxel_downsample(
            points_base.astype(np.float64), self.config.cloud_voxel / 2
        )
        self.keyframes.append(Keyframe(pose.copy(), pts))
        self._cloud.append(
            transform_points(pose, pts.astype(np.float64)).astype(np.float32)
        )
        self._cloud_points += len(pts)
        if self._cloud_points > 400_000:
            self._compact()
        return True

    def _compact(self) -> None:
        parts = [self._merged, *self._cloud]
        self._merged = voxel_downsample(
            np.concatenate(parts).astype(np.float64), self.config.cloud_voxel
        )
        self._cloud = []
        self._cloud_points = len(self._merged)

    def cloud(self) -> np.ndarray:
        self._compact()
        return self._merged

    def grid(self) -> OccupancyGrid:
        cfg = self.config
        res = cfg.resolution
        cloud = self.cloud()
        traj = np.array(self.trajectory) if self.trajectory else np.zeros((1, 3))
        band = (cfg.band_low, cfg.band_high)
        relevant = (
            cloud[(cloud[:, 2] >= band[0]) & (cloud[:, 2] <= band[1])]
            if len(cloud)
            else cloud
        )
        xy = (
            np.concatenate([relevant[:, :2], traj[:, :2]])
            if len(relevant)
            else traj[:, :2]
        )
        lo = xy.min(axis=0) - 1.0
        hi = xy.max(axis=0) + 1.0
        w, h = (np.ceil((hi - lo) / res).astype(int) + 1).tolist()
        grid = OccupancyGrid(
            np.zeros((h, w), np.uint8), res, float(lo[0]), float(lo[1])
        )
        hits = np.zeros((h, w), np.int32)
        passes = np.zeros((h, w), np.int32)
        for kf in self.keyframes:
            pts = kf.points.astype(np.float64)
            # Occupied evidence: cells this keyframe saw something in, at a
            # height the robot could hit. Counted once per keyframe.
            in_map = transform_points(kf.pose, pts)
            zb = in_map[:, 2]
            band_pts = in_map[(zb >= band[0]) & (zb <= band[1])]
            if len(band_pts):
                r, c = grid.to_cell(band_pts[:, 0], band_pts[:, 1])
                ok = grid.inside(r, c)
                np.add.at(hits.ravel(), np.unique(r[ok] * w + c[ok]), 1)
            # Free evidence: cells a ray crossed before its first obstacle.
            near, clear = _virtual_scan(transform_points(_level(kf.pose), pts), band)
            bins = len(near)
            ang = -math.pi + (np.arange(bins) + 0.5) * 2 * math.pi / bins
            yaw = pose2d(kf.pose).yaw
            ox, oy = kf.pose[0, 3], kf.pose[1, 3]
            reach = np.where(np.isfinite(near), near - res, clear)
            steps = np.arange(0.0, max(float(reach.max(initial=0.0)), 0.0), res / 2)
            if len(steps):
                rr = steps[None, :]
                valid = rr < reach[:, None]
                ex = ox + rr * np.cos(ang + yaw)[:, None]
                ey = oy + rr * np.sin(ang + yaw)[:, None]
                r, c = grid.to_cell(ex[valid], ey[valid])
                ok = grid.inside(r, c)
                np.add.at(passes.ravel(), np.unique(r[ok] * w + c[ok]), 1)
        # A low object (a table) is only hit by the few rays that dip below
        # its top, while many rays pass over it, so a modest share of
        # keyframes seeing it is enough. Things that moved away during
        # mapping (people) were seen in few keyframes and drop out.
        occupied = (hits >= cfg.min_hits) & (hits * 5 >= passes)
        grid.data[passes > 0] = FREE
        # The robot drove here, so it's free (this also covers the blind
        # spot under the lidar).
        rad = max(1, int(cfg.footprint_radius / res))
        yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
        disk = (yy**2 + xx**2) <= rad**2
        tr, tc = grid.to_cell(traj[:, 0], traj[:, 1])
        for dr, dc in zip(yy[disk], xx[disk], strict=True):
            r, c = tr + dr, tc + dc
            ok = grid.inside(r, c)
            grid.data[r[ok], c[ok]] = FREE
            occupied[r[ok], c[ok]] = False
        grid.data[occupied] = OCCUPIED
        return grid

    def build(self, name: str = "", config: MapConfig | None = None) -> Map:
        traj = np.array(self.trajectory) if self.trajectory else np.zeros((0, 3))
        return Map(
            name=name,
            cloud=self.cloud(),
            grid=self.grid(),
            trajectory=traj.astype(np.float32),
            config=config or self.config,
        )


def _level(pose: np.ndarray) -> np.ndarray:
    """Rotation-only part of ``pose`` with yaw removed (levels a base-frame scan)."""
    p2 = pose2d(pose)
    out = np.eye(4)
    c, s = math.cos(-p2.yaw), math.sin(-p2.yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    out[:3, :3] = rz @ pose[:3, :3]
    out[2, 3] = pose[2, 3]
    return out


@dataclass
class Map:
    name: str
    cloud: np.ndarray
    grid: OccupancyGrid
    trajectory: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    config: MapConfig = field(default_factory=MapConfig)
    created: float = field(default_factory=time.time)

    @property
    def start(self) -> Pose2D:
        """Where mapping started: the map frame's origin."""
        return Pose2D(0.0, 0.0, 0.0)

    def save(self, directory: str | Path | None = None) -> Path:
        d = Path(directory) if directory is not None else map_path(self.name)
        d.mkdir(parents=True, exist_ok=True)
        import io

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            cloud=self.cloud.astype(np.float32),
            grid=self.grid.data,
            trajectory=self.trajectory.astype(np.float32),
        )
        secure_atomic_write_bytes(d / "map.npz", buf.getvalue())
        meta = {
            "version": FORMAT_VERSION,
            "name": self.name,
            "created": self.created,
            "resolution": self.grid.resolution,
            "origin": [self.grid.origin_x, self.grid.origin_y],
            "config": asdict(self.config),
        }
        secure_atomic_write_json(d / "map.json", meta, sort_keys=False)
        _write_png(d / "map.png", self.grid.image())
        return d

    @classmethod
    def load(cls, name_or_path: str | Path) -> Map:
        p = Path(name_or_path)
        d = p if p.is_dir() else map_path(str(name_or_path))
        if not (d / "map.npz").exists():
            raise FileNotFoundError(
                f"No map at {d}. Make one with `axol nav.map NAME`."
            )
        meta = json.loads(secure_read_text(d / "map.json"))
        if meta.get("version") != FORMAT_VERSION:
            raise ValueError(
                f"{d} is a version {meta.get('version')} map; this build reads {FORMAT_VERSION}"
            )
        with np.load(d / "map.npz") as z:
            cloud, data, traj = z["cloud"], z["grid"], z["trajectory"]
        known = MapConfig.__dataclass_fields__
        cfg = MapConfig(
            **{k: v for k, v in meta.get("config", {}).items() if k in known}
        )
        grid = OccupancyGrid(
            data, float(meta["resolution"]), *map(float, meta["origin"])
        )
        return cls(
            meta.get("name", d.name),
            cloud,
            grid,
            traj,
            cfg,
            float(meta.get("created", 0)),
        )


def _write_png(path: Path, rgb: np.ndarray) -> None:
    import cv2

    ok, enc = cv2.imencode(".png", rgb[:, :, ::-1])
    if ok:
        secure_atomic_write_bytes(path, enc.tobytes())
