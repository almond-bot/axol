"""Lidar(-inertial) odometry: track the base's 3D pose from scan to scan.

Each frame is:

1. moved into the base frame with the lidar mount and stripped of points on
   the robot itself (:attr:`OdometryConfig.self_box`);
2. de-skewed: the Mid-360 takes ~0.1 s per frame, so every point is moved to
   where it would have been seen at the frame's end, assuming the motion of
   the previous frame continues (rotation from the lidar's gyro when it has
   IMU samples);
3. registered against a rolling local map (small_gicp's VGICP on a Gaussian
   voxel map, which forgets voxels it hasn't seen recently) starting from
   that same constant-velocity prediction;
4. inserted into the local map.

The odometry frame starts at the base's first pose with +z against gravity
(from the lidar accelerometer when available), so the floor is the z=0
plane when the mount is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from ..lidar.stream import ImuSample, LidarFrame
from .geometry import interpolate_motion, inverse, rotation_between, transform_points


def _gicp():
    try:
        import small_gicp
    except ImportError:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "Mapping and navigation need small_gicp. Reinstall Axol, or run: "
            "pip install small-gicp==1.0.1"
        ) from None
    return small_gicp


@dataclass
class OdometryConfig:
    """Tuning for :class:`LidarOdometry`.

    Attributes:
        downsample:  Voxel size (m) the scan is thinned to before registration.
        map_voxel:   Voxel size (m) of the local map.
        max_correspondence: Farthest a point may be matched (m).
        lru_horizon: Frames a local-map voxel survives without being seen.
        threads:     Worker threads for registration.
        self_box:    ``(x0, x1, y0, y1, z0, z1)`` in the base frame. Points
                     inside are the robot (Jelly, Axol, the mast) and are dropped.
        min_points:  Fewest points a scan needs to be registered.
        deskew:      Correct each frame for the motion during its 0.1 s sweep.
    """

    downsample: float = 0.15
    map_voxel: float = 0.5
    max_correspondence: float = 1.0
    lru_horizon: int = 100
    threads: int = 4
    self_box: tuple[float, float, float, float, float, float] = (
        -0.40,
        0.40,
        -0.40,
        0.40,
        -0.10,
        2.20,
    )
    min_points: int = 300
    deskew: bool = True


@dataclass
class OdometryResult:
    """One registered frame.

    Attributes:
        stamp:     Host-monotonic time the pose refers to (the frame's end).
        pose:      4x4 ``T_odom_base``.
        points:    The de-skewed scan in the base frame, before thinning
                   (float32, ``(n, 3)``), for mapping and obstacle checks.
        inlier_ratio: Share of thinned points matched to the local map.
        converged: Whether the registration converged.
    """

    stamp: float
    pose: np.ndarray
    points: np.ndarray
    inlier_ratio: float = 1.0
    converged: bool = True
    extras: dict = field(default_factory=dict)


def crop_box(pts: np.ndarray, box: tuple[float, ...]) -> np.ndarray:
    """Points outside the axis-aligned ``box`` (x0, x1, y0, y1, z0, z1)."""
    x0, x1, y0, y1, z0, z1 = box
    inside = (
        (pts[:, 0] > x0)
        & (pts[:, 0] < x1)
        & (pts[:, 1] > y0)
        & (pts[:, 1] < y1)
        & (pts[:, 2] > z0)
        & (pts[:, 2] < z1)
    )
    return pts[~inside]


def gravity_alignment(accel_base: np.ndarray) -> np.ndarray:
    """Rotation (3x3) that turns the measured specific force onto +z.

    At rest the accelerometer reads +g straight up, so aligning it with +z
    levels the odometry frame.
    """
    return rotation_between(accel_base, np.array([0.0, 0.0, 1.0]))


class LidarOdometry:
    """Scan-to-local-map odometry. See the module docs."""

    def __init__(
        self,
        t_base_lidar: np.ndarray | None = None,
        config: OdometryConfig | None = None,
    ) -> None:
        self.sg = _gicp()
        self.t_base_lidar = (
            np.eye(4) if t_base_lidar is None else np.asarray(t_base_lidar, float)
        )
        self.config = config or OdometryConfig()
        self.reset()

    def reset(self, initial: np.ndarray | None = None) -> None:
        cfg = self.config
        self.voxelmap = self.sg.GaussianVoxelMap(cfg.map_voxel)
        self.voxelmap.set_lru(horizon=cfg.lru_horizon, clear_cycle=10)
        self.pose = np.eye(4) if initial is None else np.asarray(initial, float).copy()
        self.velocity = np.eye(4)  # motion over the last frame interval
        self.last_stamp: float | None = None
        self.frames = 0
        self._leveled = initial is not None

    # -- helpers -------------------------------------------------------

    def _gyro_rotation(
        self, imu: list[ImuSample], t0: float, t1: float
    ) -> np.ndarray | None:
        samples = [s for s in imu if t0 <= s.stamp <= t1]
        if len(samples) < 3:
            return None
        gyro = np.mean([s.gyro for s in samples], axis=0)
        w_base = self.t_base_lidar[:3, :3] @ gyro
        return Rotation.from_rotvec(w_base * (t1 - t0)).as_matrix()

    def _level(self, imu: list[ImuSample] | None) -> None:
        if self._leveled or not imu or len(imu) < 20:
            return
        accel = np.mean([s.accel for s in imu[-200:]], axis=0)
        if abs(np.linalg.norm(accel) - 9.81) > 1.5:
            return  # moving or bad data: try again next frame
        r = gravity_alignment(self.t_base_lidar[:3, :3] @ accel)
        self.pose[:3, :3] = r @ self.pose[:3, :3]
        self._leveled = True

    def predict(
        self, stamp: float, imu: list[ImuSample] | None, prior: np.ndarray | None
    ) -> np.ndarray:
        """Predicted relative motion from the last pose to ``stamp``."""
        if prior is not None:
            return np.asarray(prior, float)
        if self.last_stamp is None:
            return np.eye(4)
        dt = max(stamp - self.last_stamp, 1e-3)
        frac = min(dt / max(self._interval, 1e-3), 3.0)
        rots, trans = interpolate_motion(self.velocity, np.array([frac]))
        delta = np.eye(4)
        delta[:3, :3] = rots[0]
        delta[:3, 3] = trans[0]
        if imu:
            r = self._gyro_rotation(imu, self.last_stamp, stamp)
            if r is not None:
                delta[:3, :3] = r
        return delta

    _interval = 0.1

    # -- main ----------------------------------------------------------

    def preprocess(self, frame: LidarFrame, motion: np.ndarray) -> np.ndarray:
        """Base-frame, self-cropped, de-skewed points (float64)."""
        pts = transform_points(self.t_base_lidar, frame.xyz.astype(np.float64))
        if self.config.deskew and frame.duration > 1e-3 and len(pts):
            # Motion of the base during this frame, then each point moved
            # from the pose it was taken at to the pose at the frame's end.
            frac_frame = frame.duration / max(self._interval, 1e-3)
            rots, trans = interpolate_motion(motion, np.array([frac_frame]))
            m_end = np.eye(4)
            m_end[:3, :3], m_end[:3, 3] = rots[0], trans[0]
            f = frame.times.astype(np.float64) / max(self._interval, 1e-3)
            r_i, t_i = interpolate_motion(motion, f)
            at_start = np.einsum("nij,nj->ni", r_i, pts) + t_i
            pts = transform_points(inverse(m_end), at_start)
        pts = crop_box(pts, self.config.self_box)
        return pts

    def update(
        self,
        frame: LidarFrame,
        imu: list[ImuSample] | None = None,
        prior: np.ndarray | None = None,
    ) -> OdometryResult:
        cfg = self.config
        stamp = frame.stamp + frame.duration
        self._level(imu)
        delta = self.predict(stamp, imu, prior)
        pts = self.preprocess(frame, delta if self.frames else np.eye(4))
        if len(pts) < cfg.min_points:
            # Too little to register (lidar blocked?): coast on the prediction.
            self.pose = self.pose @ delta
            self.last_stamp = stamp
            return OdometryResult(
                stamp, self.pose.copy(), pts.astype(np.float32), 0.0, False
            )
        cloud, _tree = self.sg.preprocess_points(
            pts,
            downsampling_resolution=cfg.downsample,
            num_neighbors=10,
            num_threads=cfg.threads,
        )
        inlier_ratio, converged = 1.0, True
        if self.frames == 0:
            pose = self.pose
        else:
            guess = self.pose @ delta
            res = self.sg.align(
                self.voxelmap,
                cloud,
                guess,
                max_correspondence_distance=cfg.max_correspondence,
                num_threads=cfg.threads,
                max_iterations=30,
            )
            inlier_ratio = res.num_inliers / max(cloud.size(), 1)
            converged = bool(res.converged)
            pose = res.T_target_source if inlier_ratio > 0.2 else guess
        if self.last_stamp is not None and self.frames:
            dt = stamp - self.last_stamp
            if dt > 1e-3:
                self._interval = 0.8 * self._interval + 0.2 * min(dt, 0.5)
            self.velocity = inverse(self.pose) @ pose
        self.pose = pose
        self.voxelmap.insert(cloud, self.pose)
        self.last_stamp = stamp
        self.frames += 1
        return OdometryResult(
            stamp, self.pose.copy(), pts.astype(np.float32), inlier_ratio, converged
        )
