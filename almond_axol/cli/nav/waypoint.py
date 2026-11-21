"""
axol nav.waypoint

List, add and delete a map's waypoints. To save where the robot is now, use
**Save current position** in the ``axol nav.run`` map view, or ``add`` with
``--here`` (localizes from one scan; the robot should be still).

Usage:
    axol nav.waypoint office                       # list
    axol nav.waypoint office add bench 4.2 -1.5 90 # x y (m), heading (deg)
    axol nav.waypoint office add dock --here
    axol nav.waypoint office remove bench
"""

from __future__ import annotations

import argparse
import math

from . import add_sim_argument, lidar_setup


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "nav.waypoint", help="List, add or delete a map's waypoints."
    )
    p.add_argument("map", help="Map name.")
    p.add_argument(
        "action", nargs="?", choices=["list", "add", "remove"], default="list"
    )
    p.add_argument("name", nargs="?", help="Waypoint name.")
    p.add_argument("coords", nargs="*", type=float, help="X Y HEADING_DEG for add.")
    p.add_argument(
        "--here", action="store_true", help="Use the robot's current position."
    )
    add_sim_argument(p)
    p.set_defaults(func=run)


def _here(map_name: str, sim: bool):  # noqa: ANN202
    from ...nav.localization import LOST, Localizer
    from ...nav.mapping import Map

    nav_map = Map.load(map_name)
    cfg, t_base_lidar = lidar_setup(sim)
    if sim:
        from ...nav.geometry import Pose2D
        from ...nav.sim import SimLidar, office_world

        frame = SimLidar(office_world(), t_base_lidar).scan(Pose2D(1.5, 3.5, 0.0), 0.0)
        imu = []
    else:
        from ...lidar import LidarStream

        with LidarStream(cfg) as stream:
            stream.get(2.0)
            frame = stream.get(2.0)
            if frame is None:
                raise SystemExit(
                    "No lidar data. Run `axol lidar.info` to check the lidar."
                )
            imu = stream.imu_recent(200)
    fix = Localizer(nav_map, t_base_lidar).initialize(frame, imu)
    if fix.state == LOST:
        raise SystemExit("Can't find the robot on the map from here.")
    return fix.pose2d


def run(args: argparse.Namespace) -> None:
    from ...nav.geometry import Pose2D
    from ...nav.mapping import Map
    from ...nav.waypoints import NavWaypoints

    try:
        Map.load(args.map)  # the map must exist
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    wps = NavWaypoints.load(args.map)
    if args.action == "list":
        if not wps.items:
            print(f"No waypoints on {args.map!r} yet.")
        for w in wps.items:
            print(
                f"  {w.name:<20} x {w.x:7.2f} m  y {w.y:7.2f} m  heading {math.degrees(w.yaw):6.1f}°"
            )
        return
    if not args.name:
        raise SystemExit("Give a waypoint name.")
    if args.action == "remove":
        try:
            wps.remove(args.name)
        except KeyError as exc:
            raise SystemExit(exc.args[0]) from None
        wps.save()
        print(f"Deleted {args.name}.")
        return
    if args.here:
        pose = _here(args.map, args.sim)
    elif len(args.coords) == 3:
        x, y, yaw = args.coords
        pose = Pose2D(x, y, math.radians(yaw))
    else:
        raise SystemExit("Give X Y HEADING_DEG, or --here.")
    try:
        wps.set(args.name, pose)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    wps.save()
    print(
        f"Saved {args.name} at x {pose.x:.2f} m, y {pose.y:.2f} m, heading {math.degrees(pose.yaw):.0f}°."
    )
