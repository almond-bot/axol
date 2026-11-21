"""Localize the base on a saved map.

:class:`Localizer` runs :class:`~.odometry.LidarOdometry` for a smooth pose
every frame and registers each scan against the map's 3D cloud to remove the
odometry's drift::

    T_map_base = T_map_odom · T_odom_base

``T_map_odom`` is replaced whenever a scan matches the map well. When scans
stop matching (the robot was carried somewhere else, or the map is out of
date) the state goes to ``lost`` and navigation stops.

The starting pose comes from :meth:`Localizer.initialize`, which searches
around a guess (by default where the robot was last localized, else the
map's origin) over position and heading.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..lidar.stream import ImuSample, LidarFrame
from ..utils.state_files import secure_atomic_write_json, secure_read_text
from .geometry import Pose2D, inverse, planar_matrix, pose2d, transform_points
from .mapping import Map, map_path
from .odometry import LidarOdometry, OdometryConfig, OdometryResult, _gicp

_logger = logging.getLogger(__name__)

OK = "ok"
WEAK = "weak"
LOST = "lost"
INITIALIZING = "initializing"


@dataclass
class LocalizerConfig:
    """Tuning for :class:`Localizer`.

    Attributes:
        map_voxel:     Map cloud spacing used for matching (m).
        max_correspondence: Farthest a point may match (m).
        good_ratio:    Share of a scan's wall-height points that land on the map
                       (within ``score_distance``) for a trusted fix. Floor
                       and ceiling points are left out: every building has
                       them, so they match anywhere.
        lost_ratio:    Below this for ``lost_frames`` frames, the robot is lost.
        score_distance: How close (m) a point must land to the map to count.
        lost_frames:   Consecutive poor frames before ``lost``.
        every:         Register to the map every N frames (odometry fills in).
        search_radius: How far from the guess :meth:`Localizer.initialize`
                       looks (m).
        search_step:   Spacing of the position search (m).
        yaw_steps:     Headings tried when the heading is unknown.
    """

    map_voxel: float = 0.1
    max_correspondence: float = 0.6
    good_ratio: float = 0.60
    lost_ratio: float = 0.35
    score_distance: float = 0.15
    lost_frames: int = 10
    every: int = 1
    search_radius: float = 1.0
    search_step: float = 0.5
    yaw_steps: int = 12
    threads: int = 4


@dataclass
class Fix:
    """The localizer's answer for one frame."""

    stamp: float
    pose: np.ndarray  # T_map_base
    state: str
    match_ratio: float
    odometry: OdometryResult

    @property
    def pose2d(self) -> Pose2D:
        return pose2d(self.pose)


class Localizer:
    def __init__(
        self,
        nav_map: Map,
        t_base_lidar: np.ndarray,
        config: LocalizerConfig | None = None,
        odometry: OdometryConfig | None = None,
    ) -> None:
        self.sg = _gicp()
        self.map = nav_map
        self.config = config or LocalizerConfig()
        self.odometry = LidarOdometry(t_base_lidar, odometry)
        cloud = nav_map.cloud.astype(np.float64)
        self.target, self.tree = self.sg.preprocess_points(
            cloud,
            downsampling_resolution=self.config.map_voxel,
            num_neighbors=10,
            num_threads=self.config.threads,
        )
        # Wall-height map points, for scoring a match (see LocalizerConfig).
        band = cloud[(cloud[:, 2] > 0.3) & (cloud[:, 2] < 2.0)]
        self._structure = cKDTree(band if len(band) else cloud)
        self.t_map_odom = np.eye(4)
        self.state = INITIALIZING
        self._poor = 0
        self._frames = 0
        self.last: Fix | None = None

    # -- matching ------------------------------------------------------

    def _source(self, points_base: np.ndarray):
        src, _ = self.sg.preprocess_points(
            points_base.astype(np.float64),
            downsampling_resolution=max(self.config.map_voxel * 2, 0.2),
            num_neighbors=10,
            num_threads=self.config.threads,
        )
        return src

    def match(
        self, source, guess: np.ndarray, max_corr: float | None = None, iters: int = 30
    ):
        res = self.sg.align(
            self.target,
            source,
            self.tree,
            guess,
            registration_type="GICP",
            max_correspondence_distance=max_corr or self.config.max_correspondence,
            num_threads=self.config.threads,
            max_iterations=iters,
        )
        ratio = res.num_inliers / max(source.size(), 1)
        return res.T_target_source, ratio, bool(res.converged)

    def score(
        self,
        t_map_base: np.ndarray,
        points_base: np.ndarray,
        distance: float | None = None,
    ) -> float:
        """Share of the scan's wall-height points within ``score_distance`` of the map."""
        z = points_base[:, 2]
        pts = points_base[(z > 0.3) & (z < 2.0)]
        if len(pts) < 50:
            return 0.0
        if len(pts) > 2000:
            pts = pts[:: len(pts) // 2000]
        d, _ = self._structure.query(
            transform_points(t_map_base, pts.astype(np.float64)),
            distance_upper_bound=distance or self.config.score_distance,
        )
        return float(np.isfinite(d).mean())

    def _state_for(self, score: float) -> str:
        cfg = self.config
        return (
            OK if score >= cfg.good_ratio else WEAK if score >= cfg.lost_ratio else LOST
        )

    # -- start ---------------------------------------------------------

    def initialize(
        self,
        frame: LidarFrame,
        imu: list[ImuSample] | None = None,
        guess: Pose2D | None = None,
        *,
        know_heading: bool = False,
    ) -> Fix:
        """Find the robot on the map from one scan (it should be still).

        Searches positions within :attr:`LocalizerConfig.search_radius` of
        ``guess`` and, unless ``know_heading``, every
        360/:attr:`~LocalizerConfig.yaw_steps` degrees of heading. Returns a
        fix whose ``state`` is ``ok`` when the best match is trustworthy.
        """
        cfg = self.config
        guess = guess or load_last_pose(self.map.name) or self.map.start
        self.odometry.reset()
        odo = self.odometry.update(frame, imu)
        # The odometry frame is levelled from the IMU; keep that roll/pitch.
        level = odo.pose.copy()
        level[:3, 3] = 0.0
        source = self._source(odo.points if len(odo.points) else frame.xyz)
        yaws = (
            [guess.yaw]
            if know_heading
            else [
                guess.yaw + 2 * math.pi * k / cfg.yaw_steps
                for k in range(cfg.yaw_steps)
            ]
        )
        n = int(round(cfg.search_radius / cfg.search_step))
        offsets = [
            (dx * cfg.search_step, dy * cfg.search_step)
            for dx in range(-n, n + 1)
            for dy in range(-n, n + 1)
            if math.hypot(dx, dy) <= n + 1e-9
        ]
        pts = odo.points.astype(np.float64)
        # Coarse pass: a thinned scan, a few iterations from every candidate,
        # ranked with a loose score. Then refine the best few properly.
        coarse, _ = self.sg.preprocess_points(
            pts, downsampling_resolution=0.5, num_neighbors=10, num_threads=cfg.threads
        )
        ranked = []
        for yaw in yaws:
            for dx, dy in offsets:
                init = planar_matrix(guess.x + dx, guess.y + dy, yaw) @ level
                t, _, _ = self.match(coarse, init, max_corr=1.5, iters=10)
                ranked.append((self.score(t, pts, distance=0.3), t))
        ranked.sort(key=lambda item: -item[0])
        best = (None, -1.0)
        for _, t in ranked[:3]:
            t, _, _ = self.match(source, t, max_corr=1.0, iters=20)
            t, _, _ = self.match(source, t, iters=40)
            s = self.score(t, pts)
            if s > best[1]:
                best = (t, s)
        t_map_base, ratio = best
        self.t_map_odom = t_map_base @ inverse(odo.pose)
        self.state = self._state_for(ratio)
        self._poor = 0
        fix = Fix(odo.stamp, t_map_base, self.state, ratio, odo)
        self.last = fix
        _logger.info(
            "localized at %s (match %.0f%%, %s)", fix.pose2d, ratio * 100, self.state
        )
        return fix

    def set_pose(self, pose: Pose2D) -> None:
        """Declare where the robot is now (operator override)."""
        current = self.odometry.pose
        level = current.copy()
        level[:3, 3] = 0.0
        level = planar_matrix(0, 0, -pose2d(current).yaw) @ level
        self.t_map_odom = (
            planar_matrix(pose.x, pose.y, pose.yaw) @ level @ inverse(current)
        )
        self.state = WEAK
        self._poor = 0

    # -- tracking ------------------------------------------------------

    def update(
        self,
        frame: LidarFrame,
        imu: list[ImuSample] | None = None,
        prior: np.ndarray | None = None,
    ) -> Fix:
        cfg = self.config
        odo = self.odometry.update(frame, imu, prior)
        self._frames += 1
        predicted = self.t_map_odom @ odo.pose
        ratio = self.last.match_ratio if self.last else 0.0
        if self._frames % max(cfg.every, 1) == 0 and len(odo.points) >= 100:
            source = self._source(odo.points)
            t, _, converged = self.match(source, predicted)
            ratio = self.score(t, odo.points)
            jump = np.linalg.norm(t[:2, 3] - predicted[:2, 3])
            if converged and ratio >= cfg.lost_ratio and jump < 1.0:
                self.t_map_odom = t @ inverse(odo.pose)
            if ratio >= cfg.good_ratio:
                self._poor = 0
                self.state = OK
            elif ratio >= cfg.lost_ratio:
                self._poor = 0
                self.state = WEAK if self.state != LOST else LOST
            else:
                self._poor += 1
                if self._poor >= cfg.lost_frames:
                    self.state = LOST
        fix = Fix(odo.stamp, self.t_map_odom @ odo.pose, self.state, ratio, odo)
        self.last = fix
        return fix


# -- remembering where the robot was -----------------------------------------


def _last_pose_path(map_name: str):
    return map_path(map_name) / "last_pose.json"


def save_last_pose(map_name: str, pose: Pose2D) -> None:
    try:
        secure_atomic_write_json(_last_pose_path(map_name), pose.as_dict())
    except (OSError, ValueError):
        _logger.debug("could not save the last pose", exc_info=True)


def load_last_pose(map_name: str) -> Pose2D | None:
    try:
        p = _last_pose_path(map_name)
        if not p.exists():
            return None
        d = json.loads(secure_read_text(p))
        return Pose2D(float(d["x"]), float(d["y"]), float(d["yaw"]))
    except (OSError, ValueError, KeyError):
        return None
