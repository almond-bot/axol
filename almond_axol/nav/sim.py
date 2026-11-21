"""A simulated room, Mid-360 and Jelly base for testing navigation without hardware.

The world is a floor, a ceiling and vertical wall segments (each with its own
height, so a 0.8 m box is a short "wall" loop). :class:`SimLidar` casts a
Mid-360-like ray pattern (360° around, -7° to +52° up) into it.
:class:`SimBase` integrates normalised Jelly commands with the same slew
limit and speed scale the real base uses, so the controller sees realistic lag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..lidar.stream import LidarFrame
from .geometry import Pose2D, inverse, planar_matrix, transform_points


@dataclass
class Wall:
    x1: float
    y1: float
    x2: float
    y2: float
    height: float = 2.5


@dataclass
class World:
    walls: list[Wall] = field(default_factory=list)
    ceiling: float = 3.0

    def add_box(self, cx: float, cy: float, sx: float, sy: float, h: float) -> None:
        x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
        for a, b in (
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ):
            self.walls.append(Wall(a[0], a[1], b[0], b[1], h))

    def segments(self) -> np.ndarray:
        return np.array([[w.x1, w.y1, w.x2, w.y2, w.height] for w in self.walls], float)


def office_world() -> World:
    """An 10 x 7 m room with an inner wall, a doorway and a few boxes."""
    w = World()
    for a, b in (
        ((0, 0), (10, 0)),
        ((10, 0), (10, 7)),
        ((10, 7), (0, 7)),
        ((0, 7), (0, 0)),
    ):
        w.walls.append(Wall(a[0], a[1], b[0], b[1], 3.0))
    # Inner wall at x=5 with a 1.4 m doorway between y=2.8 and y=4.2.
    w.walls.append(Wall(5, 0, 5, 2.8, 3.0))
    w.walls.append(Wall(5, 4.2, 5, 7, 3.0))
    w.add_box(2.0, 5.5, 1.2, 0.8, 0.75)  # table
    w.add_box(7.5, 1.5, 0.6, 0.6, 1.2)  # cabinet
    w.add_box(8.0, 5.0, 1.6, 0.5, 0.9)  # bench
    w.add_box(3.0, 1.2, 0.4, 0.4, 1.8)  # pillar-ish shelf
    return w


class SimLidar:
    """Cast a Mid-360-like scan from a pose in a :class:`World`."""

    def __init__(
        self,
        world: World,
        t_base_lidar: np.ndarray,
        *,
        points_per_frame: int = 8000,
        noise: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.world = world
        self.t_base_lidar = t_base_lidar
        self.n = points_per_frame
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._seq = 0

    def rays(self) -> np.ndarray:
        az = self.rng.uniform(0, 2 * math.pi, self.n)
        # Mid-360: -7° to 52°, denser near the horizon.
        el = np.radians(self.rng.uniform(-7.0, 52.0, self.n))
        return np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1
        )

    def cast(self, t_world_lidar: np.ndarray, dirs_lidar: np.ndarray) -> np.ndarray:
        """Ranges (inf for no hit) along lidar-frame directions."""
        o = t_world_lidar[:3, 3]
        d = dirs_lidar @ t_world_lidar[:3, :3].T
        best = np.full(len(d), np.inf)
        # Floor and ceiling.
        with np.errstate(divide="ignore", invalid="ignore"):
            r_floor = np.where(d[:, 2] < -1e-6, -o[2] / d[:, 2], np.inf)
            r_ceil = np.where(
                d[:, 2] > 1e-6, (self.world.ceiling - o[2]) / d[:, 2], np.inf
            )
        best = np.minimum(best, np.minimum(r_floor, r_ceil))
        seg = self.world.segments()
        if len(seg):
            p = seg[:, 0:2]
            e = seg[:, 2:4] - p  # (m,2)
            dx, dy = d[:, 0:1], d[:, 1:2]  # (n,1)
            denom = dx * e[None, :, 1] - dy * e[None, :, 0]  # (n,m)
            wx = p[None, :, 0] - o[0]
            wy = p[None, :, 1] - o[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (wx * e[None, :, 1] - wy * e[None, :, 0]) / denom
                u = (wx * dy - wy * dx) / denom
            z = o[2] + r * d[:, 2:3]
            ok = (
                (np.abs(denom) > 1e-9)
                & (r > 1e-3)
                & (u >= 0)
                & (u <= 1)
                & (z >= 0)
                & (z <= seg[None, :, 4])
            )
            r = np.where(ok, r, np.inf)
            best = np.minimum(best, r.min(axis=1))
        return best

    def scan(self, base: Pose2D, stamp: float, max_range: float = 40.0) -> LidarFrame:
        t_world_lidar = base.matrix() @ self.t_base_lidar
        dirs = self.rays()
        rng = self.cast(t_world_lidar, dirs)
        hit = np.isfinite(rng) & (rng < max_range) & (rng > 0.1)
        rng = rng[hit] + self.rng.normal(0.0, self.noise, hit.sum())
        xyz = (dirs[hit] * rng[:, None]).astype(np.float32)
        n = len(xyz)
        frame = LidarFrame(
            stamp=stamp,
            sensor_ns=int(stamp * 1e9),
            xyz=xyz,
            reflectivity=np.full(n, 100, np.uint8),
            tag=np.zeros(n, np.uint8),
            times=np.zeros(n, np.float32),
            duration=0.0,
            seq=self._seq,
        )
        self._seq += 1
        return frame


class SimBase:
    """Kinematic omni base driven by normalised Jelly commands.

    Mirrors the real base's behaviour that matters to a controller: the body
    command is slew limited (``slew`` per second, one vector limit, like the
    Rust core) and scaled to ``max_linear`` m/s and ``max_angular`` rad/s.
    """

    def __init__(
        self,
        pose: Pose2D,
        *,
        max_linear: float = 0.7,
        max_angular: float = 1.5,
        slew: float = 0.25,
        scale_error: float = 0.0,
    ) -> None:
        self.pose = pose
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.slew = slew
        self.scale = 1.0 + scale_error
        self.cmd = np.zeros(3)  # slew-limited normalised (vx, vy, wz)
        self.target = np.zeros(3)

    def set_command(self, vx: float, vy: float, wz: float, lift: int = 0) -> None:
        self.target = np.clip(np.array([vx, vy, wz], float), -1.0, 1.0)

    def step(self, dt: float) -> None:
        diff = self.target - self.cmd
        step = np.linalg.norm(diff)
        limit = self.slew * dt
        self.cmd = (
            self.target.copy() if step <= limit else self.cmd + diff * (limit / step)
        )
        vx, vy = (
            self.cmd[0] * self.max_linear * self.scale,
            self.cmd[1] * self.max_linear * self.scale,
        )
        wz = self.cmd[2] * self.max_angular * self.scale
        p = self.pose
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        self.pose = Pose2D(
            p.x + (c * vx - s * vy) * dt,
            p.y + (s * vx + c * vy) * dt,
            p.yaw + wz * dt,
        )


class SimRobot:
    """A real-time simulated robot: a lidar on a :class:`SimBase` in a world.

    Quacks like both a :class:`~almond_axol.lidar.LidarStream` (``frames()``,
    ``imu_between()``) and a Jelly (``set_command()``), so the mapping and
    navigation commands run unchanged with ``--sim``. With a ``route`` it
    follows those poses instead of taking drive commands (for mapping).
    Like Jelly, it stops when commands go stale.
    """

    def __init__(
        self,
        world: World,
        t_base_lidar: np.ndarray,
        start: Pose2D,
        *,
        route: list[Pose2D] | None = None,
        period: float = 0.1,
        realtime: bool = True,
        points_per_frame: int = 12000,
        command_timeout: float = 0.2,
    ) -> None:
        self.lidar = SimLidar(world, t_base_lidar, points_per_frame=points_per_frame)
        self.base = SimBase(start)
        self.route = list(route or [])
        self.period = period
        self.realtime = realtime
        self.command_timeout = command_timeout
        self._cmd_time = -1e9
        self._stopped = False

    def set_command(self, vx: float, vy: float, wz: float, lift: int = 0) -> None:
        import time

        self.base.set_command(vx, vy, wz)
        self._cmd_time = time.monotonic()

    def stop(self) -> None:
        self._stopped = True

    def frames(self, timeout: float = 2.0):  # noqa: ANN201
        import time

        t_next = time.monotonic()
        k = 0
        while not self._stopped:
            if self.realtime:
                t_next += self.period
                time.sleep(max(0.0, t_next - time.monotonic()))
            now = time.monotonic()
            if self.route:
                if k >= len(self.route):
                    return
                self.base.pose = self.route[k]
            else:
                if now - self._cmd_time > self.command_timeout:
                    self.base.set_command(0.0, 0.0, 0.0)
                for _ in range(4):
                    self.base.step(self.period / 4)
            k += 1
            yield self.lidar.scan(self.base.pose, now)

    def imu_between(self, t0: float, t1: float) -> list:
        return []


def office_route(
    start: Pose2D = Pose2D(1.5, 3.5, 0.0), step: float = 0.04
) -> list[Pose2D]:
    """A mapping drive through :func:`office_world`'s two rooms and back."""
    legs = [
        (Pose2D(1.5, 3.5, 0.0), Pose2D(8.5, 3.5, 0.0)),
        (Pose2D(8.5, 3.5, 0.0), Pose2D(8.5, 2.5, -math.pi / 2)),
        (Pose2D(8.5, 2.5, -math.pi / 2), Pose2D(6.5, 3.5, math.pi)),
        (Pose2D(6.5, 3.5, math.pi), Pose2D(1.5, 3.5, math.pi)),
        (Pose2D(1.5, 3.5, math.pi), Pose2D(1.0, 2.0, 1.5 * math.pi)),
        (Pose2D(1.0, 2.0, 1.5 * math.pi), Pose2D(1.5, 3.5, 2 * math.pi)),
    ]
    route = []
    for a, b in legs:
        n = max(
            2,
            int(
                max(math.hypot(b.x - a.x, b.y - a.y) / step, abs(b.yaw - a.yaw) / 0.05)
            ),
        )
        for i in range(n):
            f = i / n
            route.append(
                Pose2D(
                    a.x + f * (b.x - a.x),
                    a.y + f * (b.y - a.y),
                    a.yaw + f * (b.yaw - a.yaw),
                )
            )
    return route


def sim_frames_along(
    world: World,
    t_base_lidar: np.ndarray,
    poses: list[Pose2D],
    period: float = 0.1,
    **lidar_kwargs,
) -> list[LidarFrame]:
    """Scans from a list of base poses (for offline tests)."""
    lidar = SimLidar(world, t_base_lidar, **lidar_kwargs)
    return [lidar.scan(p, i * period) for i, p in enumerate(poses)]


__all__ = [
    "SimBase",
    "SimLidar",
    "SimRobot",
    "office_route",
    "Wall",
    "World",
    "inverse",
    "office_world",
    "planar_matrix",
    "sim_frames_along",
    "transform_points",
]
