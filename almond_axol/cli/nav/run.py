"""
axol nav.run / axol nav.goto

``nav.run`` localizes the robot on a map and opens the map view
(http://<robot>:8010), where you drive to waypoints, save new ones, click a
goal, or set the robot's pose. ``nav.goto`` drives to the waypoints you name
and exits when it gets there.

Jelly drives only when it's connected to this computer and nothing else is
using it (stop teleop first). Keep the E-Stop within reach. The arms should
be at rest: navigation doesn't move them, and it only keeps the base clear
of obstacles.

Usage:
    axol nav.run office
    axol nav.goto office bench
    axol nav.goto office bench dock --loop
    axol nav.run demo --sim
"""

from __future__ import annotations

import argparse
import asyncio
import math
import threading
import time
from dataclasses import replace

from . import add_port_argument, add_sim_argument, lidar_setup


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("map", help="Map name.")
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Top speed in m/s (default 0.3, at most 0.7).",
    )
    p.add_argument(
        "--no-drive",
        action="store_true",
        help="Localize and show the map, but never drive.",
    )
    p.add_argument(
        "--start",
        type=float,
        nargs=3,
        metavar=("X", "Y", "YAW_DEG"),
        default=None,
        help="Where the robot is on the map, if it can't find itself.",
    )
    add_port_argument(p)
    add_sim_argument(p)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "nav.run", help="Localize on a map and drive to waypoints from the map view."
    )
    _common(p)
    p.add_argument("--no-view", action="store_true", help="Don't start the map view.")
    p.set_defaults(func=run, goals=[], loop=False, exit_when_done=False)

    g = subparsers.add_parser(
        "nav.goto", help="Drive to one or more waypoints on a map."
    )
    _common(g)
    g.add_argument("goals", nargs="+", help="Waypoint names, visited in order.")
    g.add_argument(
        "--loop", action="store_true", help="Keep visiting them until Ctrl-C."
    )
    g.add_argument("--view", action="store_true", help="Also start the map view.")
    g.set_defaults(func=run, exit_when_done=True)


def _open_jelly():  # noqa: ANN202
    """Jelly configured for navigation, or None when it isn't connected."""
    from ...robot.jelly import Jelly, JellyConfig, detect_jelly
    from ...settings import shared_config

    cfg = detect_jelly(shared_config(JellyConfig, "teleop", "jelly"))
    if cfg is None or cfg.channel is None:
        return None
    # The planner steers the exact heading, so no snapping to the axes; the
    # lift stays where it is.
    return Jelly(replace(cfg, axis_snap_deg=0.0, lift=False))


def run(args: argparse.Namespace) -> None:
    from ...nav.controller import DriveConfig
    from ...nav.geometry import Pose2D
    from ...nav.mapping import Map
    from ...nav.session import ARRIVED, FAILED, NavConfig, NavSession

    try:
        nav_map = Map.load(args.map)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    cfg, t_base_lidar = lidar_setup(args.sim)
    drive_cfg = DriveConfig()
    if args.speed is not None:
        if not 0.05 <= args.speed <= 0.7:
            raise SystemExit("--speed must be between 0.05 and 0.7 m/s.")
        drive_cfg.max_speed = args.speed
    start = (
        Pose2D(args.start[0], args.start[1], math.radians(args.start[2]))
        if args.start
        else None
    )

    if args.sim:
        from ...nav.sim import SimRobot, office_world

        robot = SimRobot(office_world(), t_base_lidar, Pose2D(1.5, 3.5, 0.0))
        source, drive = robot, (None if args.no_drive else robot)
    else:
        from ...lidar import LidarStream

        source = LidarStream(cfg)
        drive = None if args.no_drive else _open_jelly()
        if drive is None and not args.no_drive:
            print("Jelly isn't connected to this computer, so the robot won't drive.")

    session = NavSession(
        nav_map,
        t_base_lidar,
        drive=None,
        config=NavConfig(drive=drive_cfg),
        initial_pose=start,
    )
    for goal in args.goals:
        try:
            session.waypoints.get(goal)
        except KeyError as exc:
            raise SystemExit(
                f"{exc.args[0]} Saved: {', '.join(session.waypoints.names()) or 'none'}."
            ) from None

    async def main() -> None:
        jelly = drive if drive is not None and hasattr(drive, "enable") else None
        if jelly is not None:
            await jelly.enable()
        session.drive = drive
        stop = threading.Event()
        view_thread = None
        try:
            if hasattr(source, "start"):
                source.start()
            session.start(source)
            show_view = (
                not getattr(args, "no_view", False)
                if not args.exit_when_done
                else getattr(args, "view", False)
            )
            if show_view:
                from ...nav.viewer import NavView

                view = NavView(args.port, session, nav_map, can_drive=drive is not None)
                view_thread = threading.Thread(
                    target=view.run, args=(stop,), daemon=True
                )
                view_thread.start()
                print(f"Map view on http://localhost:{args.port}")
            sent = False
            last_msg = ""
            while True:
                st = session.status()
                if st.message != last_msg:
                    last_msg = st.message
                    print(f"[{st.state}] {st.message}")
                if (
                    args.goals
                    and not sent
                    and st.state not in ("localizing", "not localized")
                ):
                    if drive is None:
                        print("Not driving (--no-drive or no Jelly).")
                    else:
                        session.goto(list(args.goals), loop=args.loop)
                    sent = True
                if (
                    args.exit_when_done
                    and sent
                    and (st.state in (ARRIVED, FAILED) or drive is None)
                ):
                    break
                await asyncio.sleep(0.2)
        finally:
            stop.set()
            session.stop()
            if hasattr(source, "stop"):
                source.stop()
            if jelly is not None:
                await jelly.disable()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
    time.sleep(0.1)
