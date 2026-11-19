"""One navigation session: localize on a map and drive to waypoints.

:class:`NavSession` wires the pieces together::

    lidar frames ──> Localizer ──> pose on the map
                                   │
           goal ──> Planner ──> path ──> PathFollower ──> command
                                   ObstacleMonitor ──┘ (stop / slow)
                                                     │
                                       drive thread ──> Jelly.set_command

:meth:`NavSession.step` processes one frame and is what tests drive
directly. :meth:`NavSession.start` runs it on a thread over a live
:class:`~almond_axol.lidar.LidarStream`.

The drive thread re-sends the latest command at 25 Hz, but only while it is
fresh (computed from a fix less than :attr:`NavConfig.command_fresh` s ago).
If localization or the lidar stalls, it sends one stop and then nothing, so
Jelly's own command timeout stops the wheels. That keeps the rule
:meth:`almond_axol.robot.jelly.Jelly.set_command` sets: never re-send a stale
command.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..lidar.stream import ImuSample, LidarFrame
from .controller import DriveCommand, DriveConfig, PathFollower
from .geometry import Pose2D, transform_points
from .localization import LOST, OK, Fix, Localizer, LocalizerConfig, save_last_pose
from .mapping import Map
from .odometry import OdometryConfig
from .planner import NoPath, Planner, PlannerConfig, path_length
from .safety import ObstacleMonitor, SafetyConfig
from .waypoints import NavWaypoints

_logger = logging.getLogger(__name__)

IDLE = "idle"
LOCALIZING = "localizing"
DRIVING = "driving"
BLOCKED = "blocked"
ARRIVED = "arrived"
FAILED = "failed"
STOPPED = "stopped"
NOT_LOCALIZED = "not localized"


class Drive(Protocol):
    def set_command(self, vx: float, vy: float, wz: float, lift: int = 0) -> None: ...


@dataclass
class NavConfig:
    drive: DriveConfig = field(default_factory=DriveConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    localizer: LocalizerConfig = field(default_factory=LocalizerConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    command_fresh: float = 0.25
    drive_rate: float = 25.0
    replan_after: float = 3.0
    give_up_after: float = 60.0
    obstacle_radius: float = 3.0


@dataclass
class NavStatus:
    state: str
    message: str
    pose: Pose2D | None
    localization: str
    match_ratio: float
    goal: str | None
    remaining: float
    path: list[tuple[float, float]]
    command: tuple[float, float, float]
    nearest_obstacle: float
    queue: list[str]


@dataclass
class _Goal:
    name: str
    pose: Pose2D


class NavSession:
    def __init__(
        self,
        nav_map: Map,
        t_base_lidar: np.ndarray,
        *,
        drive: Drive | None = None,
        config: NavConfig | None = None,
        initial_pose: Pose2D | None = None,
    ) -> None:
        self.config = config or NavConfig()
        self.map = nav_map
        self.t_base_lidar = t_base_lidar
        self.localizer = Localizer(
            nav_map, t_base_lidar, self.config.localizer, self.config.odometry
        )
        self.planner = Planner(nav_map.grid, self.config.planner)
        self.monitor = ObstacleMonitor(self.config.safety)
        self.waypoints = (
            NavWaypoints.load(nav_map.name) if nav_map.name else NavWaypoints("")
        )
        self.drive = drive
        self.initial_pose = initial_pose

        self._lock = threading.RLock()
        self.state = LOCALIZING
        self.message = "Finding the robot on the map..."
        self._queue: list[_Goal] = []
        self._loop = False
        self._goal: _Goal | None = None
        self._follower: PathFollower | None = None
        self.path: list[tuple[float, float]] = []
        self._blocked_since: float | None = None
        self._last_replan = 0.0
        self.fix: Fix | None = None
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_time = -math.inf
        self._nearest = math.inf
        self._remaining = 0.0
        self._saved_pose_at = 0.0
        self._pending_pose: Pose2D | None = None
        self.last_points: np.ndarray | None = None  # latest scan, map frame

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- goals (any thread) ---------------------------------------------

    def goto(self, targets: list[str | Pose2D], loop: bool = False) -> None:
        """Drive to each target in turn (waypoint names or poses)."""
        goals = []
        for t in targets:
            if isinstance(t, Pose2D):
                goals.append(_Goal(f"({t.x:.2f}, {t.y:.2f})", t))
            else:
                goals.append(_Goal(t, self.waypoints.get(t).pose))
        if not goals:
            return
        with self._lock:
            self._queue = goals
            self._loop = loop
            self._next_goal()

    def cancel(self, message: str = "Stopped.") -> None:
        with self._lock:
            self._queue = []
            self._goal = None
            self._follower = None
            self.path = []
            self._set_command(0.0, 0.0, 0.0)
            if self.state not in (LOCALIZING, NOT_LOCALIZED):
                self.state = STOPPED
            self.message = message

    def set_pose(self, pose: Pose2D) -> None:
        """Tell the localizer where the robot is (applied on the next frame)."""
        with self._lock:
            self._pending_pose = pose

    def add_waypoint(self, name: str) -> Pose2D:
        """Save the robot's current pose as a waypoint."""
        fix = self.fix
        if fix is None or fix.state == LOST:
            raise RuntimeError("The robot isn't localized, so there's no pose to save.")
        pose = fix.pose2d
        self.waypoints.set(name, pose)
        self.waypoints.save()
        return pose

    def _next_goal(self) -> None:
        if not self._queue:
            self._goal = None
            self._follower = None
            return
        goal = self._queue.pop(0)
        if self._loop:
            self._queue.append(goal)
        self._goal = goal
        self._follower = None
        self.state = DRIVING
        self.message = f"Going to {goal.name}."

    # -- per-frame logic -------------------------------------------------

    def _set_command(self, vx: float, vy: float, wz: float) -> None:
        self._cmd = self.config.drive.normalize(vx, vy, wz)
        self._cmd_time = time.monotonic()

    def _plan(self, pose: Pose2D) -> bool:
        assert self._goal is not None
        try:
            path = self.planner.plan(
                (pose.x, pose.y), (self._goal.pose.x, self._goal.pose.y)
            )
        except NoPath as exc:
            self.path = []
            self._follower = None
            self.message = f"Can't reach {self._goal.name}: {exc}"
            return False
        self.path = path
        self._follower = PathFollower(path, self._goal.pose.yaw, self.config.drive)
        self.message = f"Going to {self._goal.name} ({path_length(path):.1f} m)."
        return True

    def step(
        self,
        frame: LidarFrame,
        imu: list[ImuSample] | None = None,
        now: float | None = None,
    ) -> DriveCommand | None:
        """Process one frame. Returns the command it set, if driving."""
        now = time.monotonic() if now is None else now
        with self._lock:
            pending, self._pending_pose = self._pending_pose, None
        if self.state == LOCALIZING or self.fix is None:
            fix = self.localizer.initialize(frame, imu, self.initial_pose)
            with self._lock:
                self.fix = fix
                if fix.state != OK:
                    self.state = NOT_LOCALIZED
                    self.message = (
                        "Can't find the robot on the map. Set its pose, or drive it "
                        "to where mapping started."
                    )
                else:
                    self.state = IDLE
                    self.message = "Ready."
            return None
        if pending is not None:
            self.localizer.set_pose(pending)
        fix = self.localizer.update(frame, imu)
        pose = fix.pose2d
        points_map = transform_points(fix.pose, fix.odometry.points.astype(np.float64))
        with self._lock:
            self.fix = fix
            self.last_points = points_map.astype(np.float32)
            if now - self._saved_pose_at > 2.0 and fix.state == OK and self.map.name:
                save_last_pose(self.map.name, pose)
                self._saved_pose_at = now
            if fix.state == LOST:
                if self._goal is not None or self.state == DRIVING:
                    self._set_command(0.0, 0.0, 0.0)
                self.state = NOT_LOCALIZED
                self.message = (
                    "Lost on the map. Stopped. Set the robot's pose to continue."
                )
                return None
            if self.state == NOT_LOCALIZED:
                if fix.state != OK:
                    # Wait for a clear match before driving again.
                    if self._goal is not None:
                        self._set_command(0.0, 0.0, 0.0)
                    return None
                self.state = DRIVING if self._goal else IDLE
                self.message = "Localized." if not self._goal else self.message
            if self._goal is None:
                if self.state == DRIVING:
                    self.state = IDLE
                return None
            if self._follower is None and not self._plan(pose):
                self._set_command(0.0, 0.0, 0.0)
                self.state = FAILED
                self._queue = []
                self._goal = None
                return None
            assert self._follower is not None

            # Obstacle check against the motion we're about to command.
            preview = self._follower.command(pose)
            verdict = self.monitor.check(fix.odometry.points, preview.vx, preview.vy)
            self._nearest = verdict.nearest
            if verdict.stop and not preview.arrived:
                self._set_command(0.0, 0.0, 0.0)
                if self._blocked_since is None:
                    self._blocked_since = now
                self.state = BLOCKED
                waited = now - self._blocked_since
                self.message = f"Something is in the way. Waiting ({waited:.0f} s)."
                if waited > self.config.give_up_after:
                    self.state = FAILED
                    self.message = (
                        f"Gave up on {self._goal.name}: the way stayed blocked."
                    )
                    self._goal, self._follower, self._queue, self.path = (
                        None,
                        None,
                        [],
                        [],
                    )
                elif (
                    waited > self.config.replan_after
                    and now - self._last_replan > self.config.replan_after
                ):
                    self._last_replan = now
                    self._replan_around(points_map, pose)
                return None
            if self._blocked_since is not None:
                self._blocked_since = None
                self.state = DRIVING
                self.message = f"Going to {self._goal.name}."
            limit = self.config.drive.max_speed * (0.5 if verdict.slow else 1.0)
            cmd = self._follower.command(pose, speed_limit=limit)
            self._remaining = cmd.remaining
            if cmd.arrived:
                self._set_command(0.0, 0.0, 0.0)
                name = self._goal.name
                self.planner.set_obstacles(None)
                self._next_goal()
                if self._goal is None:
                    self.state = ARRIVED
                    self.message = f"Arrived at {name}."
                return cmd
            self._set_command(cmd.vx, cmd.vy, cmd.wz)
            return cmd

    def _replan_around(self, points_map: np.ndarray, pose: Pose2D) -> None:
        cfg = self.config.safety
        z = points_map[:, 2]
        near = (
            np.hypot(points_map[:, 0] - pose.x, points_map[:, 1] - pose.y)
            < self.config.obstacle_radius
        )
        obstacles = points_map[near & (z >= cfg.band_low) & (z <= cfg.band_high), :2]
        self.planner.set_obstacles(obstacles)
        if self._plan(pose):
            self.message = f"Found a way around. Going to {self._goal.name}."
        else:
            self.planner.set_obstacles(None)
            self._follower = None

    # -- status ------------------------------------------------------------

    def status(self) -> NavStatus:
        with self._lock:
            fix = self.fix
            return NavStatus(
                state=self.state,
                message=self.message,
                pose=fix.pose2d if fix else None,
                localization=fix.state if fix else LOCALIZING,
                match_ratio=fix.match_ratio if fix else 0.0,
                goal=self._goal.name if self._goal else None,
                remaining=self._remaining if self._goal else 0.0,
                path=list(self.path),
                command=self._cmd,
                nearest_obstacle=self._nearest,
                queue=[g.name for g in self._queue],
            )

    # -- threads -------------------------------------------------------------

    def start(self, stream) -> None:  # noqa: ANN001 - LidarStream-like
        self._stop.clear()
        t = threading.Thread(
            target=self._frame_loop, args=(stream,), name="nav-frames", daemon=True
        )
        self._threads = [t]
        if self.drive is not None:
            self._threads.append(
                threading.Thread(target=self._drive_loop, name="nav-drive", daemon=True)
            )
        for th in self._threads:
            th.start()

    def stop(self) -> None:
        self.cancel()
        self._stop.set()
        for th in self._threads:
            th.join(2.0)
        self._threads = []

    def _frame_loop(self, stream) -> None:  # noqa: ANN001
        last_end = None
        try:
            for frame in stream.frames():
                if self._stop.is_set():
                    return
                end = frame.stamp + frame.duration
                imu = stream.imu_between(
                    last_end if last_end is not None else end - 1.0, end + 0.005
                )
                last_end = end
                self.step(frame, imu)
        except Exception as exc:  # noqa: BLE001 - reported to the operator
            _logger.exception("navigation stopped")
            with self._lock:
                self._set_command(0.0, 0.0, 0.0)
                self.state = FAILED
                self.message = f"Navigation stopped: {exc}"
                self._goal, self._follower, self._queue = None, None, []

    def _drive_loop(self) -> None:
        assert self.drive is not None
        period = 1.0 / self.config.drive_rate
        sent_stop = False
        while not self._stop.is_set():
            with self._lock:
                cmd, age = self._cmd, time.monotonic() - self._cmd_time
                active = self._goal is not None or any(abs(v) > 1e-6 for v in cmd)
            if age <= self.config.command_fresh and active:
                self.drive.set_command(*cmd)
                sent_stop = all(abs(v) < 1e-6 for v in cmd)
            elif not sent_stop:
                self.drive.set_command(0.0, 0.0, 0.0)
                sent_stop = True
            time.sleep(period)
        self.drive.set_command(0.0, 0.0, 0.0)
