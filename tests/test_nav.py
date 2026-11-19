"""Lidar mapping, localization, planning and driving, closed loop in simulation."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from almond_axol.lidar.config import pose_matrix
from almond_axol.nav.controller import DriveConfig, PathFollower
from almond_axol.nav.geometry import Pose2D, planar_matrix, pose2d, wrap_angle
from almond_axol.nav.localization import (
    LOST,
    OK,
    Localizer,
    load_last_pose,
    save_last_pose,
)
from almond_axol.nav.mapping import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    Map,
    MapBuilder,
    list_maps,
    map_path,
    voxel_downsample,
)
from almond_axol.nav.odometry import LidarOdometry, crop_box
from almond_axol.nav.planner import NoPath, Planner, path_length
from almond_axol.nav.safety import ObstacleMonitor
from almond_axol.nav.session import ARRIVED, BLOCKED, IDLE, NavSession
from almond_axol.nav.sim import SimBase, SimLidar, Wall, office_route, office_world
from almond_axol.nav.waypoints import NavWaypoints

MOUNT = pose_matrix(0.2, 0.0, 0.9, 0.0, 0.0, 0.0)
START = Pose2D(1.5, 3.5, 0.0)  # where mapping starts in the office world
T0 = planar_matrix(START.x, START.y, START.yaw)


def world_to_map(p: Pose2D) -> Pose2D:
    return pose2d(np.linalg.inv(T0) @ p.matrix())


def map_to_world(p: Pose2D) -> Pose2D:
    return pose2d(T0 @ p.matrix())


_MAP: Map | None = None


def office_map() -> Map:
    """A map of the simulated office, built once for the module."""
    global _MAP
    if _MAP is None:
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=4000, seed=3)
        odo = LidarOdometry(MOUNT)
        builder = MapBuilder()
        for k, p in enumerate(office_route(step=0.08)):
            r = odo.update(lidar.scan(p, k * 0.1))
            builder.add(r.pose, r.points)
        _MAP = builder.build("")
    return _MAP


class _TempHome(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get("ALMOND_HOME")
        self.tmp = tempfile.mkdtemp()
        os.environ["ALMOND_HOME"] = self.tmp

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("ALMOND_HOME", None)
        else:
            os.environ["ALMOND_HOME"] = self._old


class GeometryTest(unittest.TestCase):
    def test_pose_round_trip(self) -> None:
        p = Pose2D(1.0, -2.0, 2.5)
        q = pose2d(p.matrix())
        self.assertAlmostEqual(q.x, 1.0)
        self.assertAlmostEqual(q.y, -2.0)
        self.assertAlmostEqual(q.yaw, 2.5)
        self.assertAlmostEqual(abs(wrap_angle(3 * math.pi)), math.pi, places=9)
        self.assertAlmostEqual(wrap_angle(0.5 + 4 * math.pi), 0.5, places=9)

    def test_crop_box_and_downsample(self) -> None:
        pts = np.array([[0, 0, 1], [1, 0, 1], [0.05, 0.05, 1.0]], float)
        self.assertEqual(len(crop_box(pts, (-0.4, 0.4, -0.4, 0.4, 0, 2))), 1)
        self.assertEqual(len(voxel_downsample(pts, 0.5)), 2)


class OdometryTest(unittest.TestCase):
    def test_tracks_a_drive_with_a_turn(self) -> None:
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=6000, seed=1)
        poses = [Pose2D(1.5 + 0.05 * i, 3.5, 0.0) for i in range(60)]
        poses += [Pose2D(4.5, 3.5, i * math.pi / 40) for i in range(1, 21)]
        odo = LidarOdometry(MOUNT)
        for k, p in enumerate(poses):
            r = odo.update(lidar.scan(p, k * 0.1))
        est = pose2d(T0 @ r.pose)
        truth = poses[-1]
        self.assertLess(math.hypot(est.x - truth.x, est.y - truth.y), 0.05)
        self.assertLess(abs(wrap_angle(est.yaw - truth.yaw)), math.radians(1.0))
        self.assertGreater(r.inlier_ratio, 0.8)


class MappingTest(_TempHome):
    def test_grid_shows_walls_doorway_and_table(self) -> None:
        m = office_map()
        g = m.grid

        def at(x: float, y: float) -> int:
            p = world_to_map(Pose2D(x, y, 0.0))
            return g.value_at(p.x, p.y)

        self.assertEqual(at(5.0, 1.5), OCCUPIED)  # inner wall
        self.assertEqual(at(5.0, 3.5), FREE)  # the doorway
        self.assertEqual(at(3.0, 3.5), FREE)  # the driven corridor
        self.assertEqual(at(1.4, 5.5), OCCUPIED)  # the 0.75 m table's edge
        self.assertEqual(at(-3.0, 3.5), UNKNOWN)  # outside the room
        self.assertLess(abs(np.percentile(m.cloud[:, 2], 1)), 0.05)  # floor at z = 0

    def test_save_load_and_list(self) -> None:
        m = office_map()
        m.name = "office"
        m.save()
        self.assertEqual(list_maps(), ["office"])
        back = Map.load("office")
        np.testing.assert_array_equal(back.grid.data, m.grid.data)
        self.assertEqual(back.cloud.shape, m.cloud.shape)
        self.assertTrue((map_path("office") / "map.png").exists())
        with self.assertRaises(ValueError):
            map_path("../etc")
        with self.assertRaises(FileNotFoundError):
            Map.load("missing")


class LocalizationTest(_TempHome):
    def test_finds_robot_from_a_wrong_guess_and_tracks_it(self) -> None:
        m = office_map()
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=6000, seed=7)
        loc = Localizer(m, MOUNT)
        truth = Pose2D(2.3, 2.6, 0.6)
        fix = loc.initialize(lidar.scan(truth, 0.0), guess=Pose2D(0.4, -0.4, 0.0))
        self.assertEqual(fix.state, OK)
        est = map_to_world(fix.pose2d)
        self.assertLess(math.hypot(est.x - truth.x, est.y - truth.y), 0.05)
        self.assertLess(abs(wrap_angle(est.yaw - truth.yaw)), math.radians(2))
        for k in range(1, 30):
            truth = Pose2D(truth.x + 0.05, truth.y, truth.yaw)
            fix = loc.update(lidar.scan(truth, k * 0.1))
        est = map_to_world(fix.pose2d)
        self.assertLess(math.hypot(est.x - truth.x, est.y - truth.y), 0.05)

    def test_reports_lost_in_an_unknown_place(self) -> None:
        m = office_map()
        other = SimLidar(_corridor_world(), MOUNT, points_per_frame=6000)
        loc = Localizer(m, MOUNT)
        fix = loc.initialize(other.scan(Pose2D(2, 1, 0), 0.0))
        self.assertNotEqual(fix.state, OK)

    def test_last_pose_round_trip(self) -> None:
        map_path("office").mkdir(parents=True)
        save_last_pose("office", Pose2D(1, 2, 3))
        self.assertEqual(load_last_pose("office"), Pose2D(1, 2, 3))
        self.assertIsNone(load_last_pose("nope"))


def _corridor_world():
    from almond_axol.nav.sim import World

    w = World()
    w.walls += [Wall(0, 0, 30, 0, 3), Wall(0, 2, 30, 2, 3), Wall(0, 0, 0, 2, 3)]
    return w


class PlannerTest(unittest.TestCase):
    def test_plans_through_the_doorway(self) -> None:
        m = office_map()
        planner = Planner(m.grid)
        goal = world_to_map(Pose2D(8.0, 3.0, 0.0))
        path = planner.plan((0.0, 0.0), (goal.x, goal.y))
        self.assertEqual(path[0], (0.0, 0.0))
        self.assertAlmostEqual(path[-1][0], goal.x)
        # It must cross x=5 (world) inside the doorway, y in [2.8, 4.2].
        world = [map_to_world(Pose2D(x, y, 0)) for x, y in path]
        for a, b in zip(world, world[1:], strict=False):
            if (a.x - 5.0) * (b.x - 5.0) < 0:
                y = a.y + (b.y - a.y) * (5.0 - a.x) / (b.x - a.x)
                self.assertTrue(2.8 + 0.4 < y < 4.2 - 0.4, y)
        self.assertLess(path_length(path), 9.0)
        # Every waypoint keeps the robot clear of obstacles.
        for x, y in path[1:]:
            self.assertGreater(planner.clearance_at(x, y), planner.config.robot_radius)

    def test_rejects_goals_in_walls_and_blocked_routes(self) -> None:
        m = office_map()
        planner = Planner(m.grid)
        wall = world_to_map(Pose2D(5.0, 1.5, 0))
        with self.assertRaises(NoPath):
            planner.plan((0.0, 0.0), (wall.x, wall.y))
        # Block the doorway with live obstacles: the other room is unreachable.
        door = [world_to_map(Pose2D(5.0, 2.8 + 0.05 * i, 0)) for i in range(29)]
        planner.set_obstacles(np.array([[p.x, p.y] for p in door]))
        far = world_to_map(Pose2D(8.0, 3.0, 0))
        with self.assertRaises(NoPath):
            planner.plan((0.0, 0.0), (far.x, far.y))
        planner.set_obstacles(None)
        planner.plan((0.0, 0.0), (far.x, far.y))


class FollowerTest(unittest.TestCase):
    def test_reaches_goal_with_slew_and_scale_error(self) -> None:
        cfg = DriveConfig()
        base = SimBase(Pose2D(0, 0, 0), scale_error=-0.1)
        follower = PathFollower([(0, 0), (2, 0), (2, 1.5)], math.pi / 2, cfg)
        for _ in range(600):
            cmd = follower.command(base.pose)
            if cmd.arrived:
                break
            base.set_command(*cfg.normalize(cmd.vx, cmd.vy, cmd.wz))
            for _ in range(4):
                base.step(0.025)
        self.assertTrue(cmd.arrived)
        self.assertLess(
            math.hypot(base.pose.x - 2, base.pose.y - 1.5), cfg.goal_tolerance + 0.02
        )
        self.assertLess(
            abs(wrap_angle(base.pose.yaw - math.pi / 2)),
            math.radians(cfg.yaw_tolerance + 1),
        )

    def test_normalize_clamps(self) -> None:
        cfg = DriveConfig(base_max_linear=0.5, base_max_angular=1.0)
        self.assertEqual(cfg.normalize(1.0, -0.25, 3.0), (1.0, -0.5, 1.0))


class SafetyTest(unittest.TestCase):
    def test_stop_slow_and_ignore(self) -> None:
        mon = ObstacleMonitor()
        clear = np.array([[3.0, 0.0, 0.5], [0.0, 0.0, 0.02], [0.5, 0.0, 2.5]] * 10)
        v = mon.check(clear)
        self.assertFalse(v.stop)
        self.assertFalse(v.slow)
        near = np.array([[0.45, 0.0, 0.5]] * 10)
        self.assertTrue(mon.check(near).stop)
        ahead = np.array([[0.9, 0.0, 0.5]] * 10)
        self.assertFalse(mon.check(ahead).stop)
        self.assertTrue(mon.check(ahead).slow)
        # Driving toward it at 0.3 m/s stretches the stop zone 0.45 m forward.
        self.assertTrue(mon.check(ahead, vx=0.3).stop)
        self.assertFalse(mon.check(ahead, vx=-0.3).stop)
        self.assertFalse(mon.check(np.array([[0.45, 0, 0.5]] * 3)).stop)  # noise


class WaypointsTest(_TempHome):
    def test_set_save_load_remove(self) -> None:
        map_path("office").mkdir(parents=True)
        wps = NavWaypoints.load("office")
        wps.set("bench", Pose2D(4.2, -1.5, 1.57))
        wps.set("dock", Pose2D(0, 0, 0))
        wps.set("bench", Pose2D(4.0, -1.0, 0.0))
        wps.save()
        back = NavWaypoints.load("office")
        self.assertEqual(back.names(), ["bench", "dock"])
        self.assertEqual(back.get("bench").pose, Pose2D(4.0, -1.0, 0.0))
        back.remove("dock")
        with self.assertRaises(KeyError):
            back.remove("dock")
        with self.assertRaises(ValueError):
            back.set(" bad", Pose2D(0, 0, 0))
        with self.assertRaises(ValueError):
            back.set("nan", Pose2D(float("nan"), 0, 0))


class _Drive:
    def __init__(self, base: SimBase) -> None:
        self.base = base

    def set_command(self, vx: float, vy: float, wz: float, lift: int = 0) -> None:
        self.base.set_command(vx, vy, wz)


def _run(
    session: NavSession, base: SimBase, lidar: SimLidar, steps: int, *, on_step=None
):  # noqa: ANN001
    t = 0.0
    for k in range(steps):
        session.step(lidar.scan(base.pose, t), now=t)
        if on_step is not None:
            on_step(k, session)
        base.set_command(*session._cmd)
        for _ in range(4):
            base.step(0.025)
        t += 0.1
        if session.status().state in (ARRIVED, "failed"):
            break
    return session.status()


class SessionTest(_TempHome):
    def test_drives_to_a_waypoint_through_the_doorway(self) -> None:
        m = office_map()
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=6000, seed=11)
        base = SimBase(Pose2D(2.3, 2.6, 0.6), scale_error=0.08)
        session = NavSession(m, MOUNT, initial_pose=Pose2D(0.5, -0.5, 0.0))
        goal = world_to_map(Pose2D(8.0, 3.0, math.pi / 2))

        def kick(k: int, s: NavSession) -> None:
            if k == 0:
                self.assertEqual(s.status().state, IDLE)
                s.goto([goal])

        st = _run(session, base, lidar, 500, on_step=kick)
        self.assertEqual(st.state, ARRIVED, st.message)
        self.assertLess(math.hypot(base.pose.x - 8.0, base.pose.y - 3.0), 0.12)

    def test_stops_for_an_unmapped_obstacle(self) -> None:
        m = office_map()
        world = office_world()
        # A box nobody mapped, right on the corridor.
        world.add_box(3.6, 3.5, 0.5, 0.5, 1.0)
        lidar = SimLidar(world, MOUNT, points_per_frame=6000, seed=5)
        base = SimBase(START)
        session = NavSession(m, MOUNT)
        goal = world_to_map(Pose2D(4.5, 3.5, 0.0))
        states = []

        def watch(k: int, s: NavSession) -> None:
            if k == 0:
                s.goto([goal])
            states.append(s.status().state)

        _run(session, base, lidar, 60, on_step=watch)
        self.assertIn(BLOCKED, states)
        # It stopped short of the box (box face at x=3.35; base half-length 0.3).
        self.assertLess(base.pose.x, 3.35 - 0.3)

    def test_waypoint_from_current_pose(self) -> None:
        m = office_map()
        m.name = "office"
        m.save()
        lidar = SimLidar(office_world(), MOUNT, points_per_frame=6000)
        session = NavSession(Map.load("office"), MOUNT)
        with self.assertRaises(RuntimeError):
            session.add_waypoint("too-early")
        session.step(lidar.scan(START, 0.0), now=0.0)
        session.step(lidar.scan(START, 0.1), now=0.1)
        pose = session.add_waypoint("dock")
        self.assertLess(math.hypot(pose.x, pose.y), 0.05)
        self.assertEqual(NavWaypoints.load("office").names(), ["dock"])
        self.assertNotEqual(session.status().localization, LOST)


if __name__ == "__main__":
    unittest.main()
