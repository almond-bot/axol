"""
axol nav.map

Build a map of the space around the robot. Start where you want the map's
origin to be (a dock or a marked spot works well), then drive the robot
slowly through every area it should reach, with the headset or a gamepad.
Press Ctrl-C or **Save map** in the map view to save it.

The map view (http://<robot>:8010) shows the map growing as you drive.

Usage:
    axol nav.map office
    axol nav.map office --from drive.npz      # from a recording
    axol nav.map demo --sim                    # simulated office
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from . import add_port_argument, add_sim_argument, lidar_setup


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "nav.map", help="Build a map by driving the robot around."
    )
    p.add_argument("name", help="Map name (letters, digits, '-', '_', '.').")
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="Build from a lidar recording instead.",
    )
    p.add_argument("--no-view", action="store_true", help="Don't start the map view.")
    p.add_argument(
        "--overwrite", action="store_true", help="Replace a map with the same name."
    )
    add_port_argument(p)
    add_sim_argument(p)
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from ...nav.mapping import MapBuilder, list_maps, map_path
    from ...nav.odometry import LidarOdometry

    map_path(args.name)  # validates the name
    if args.name in list_maps() and not args.overwrite:
        raise SystemExit(
            f"A map named {args.name!r} exists. Pick another name or pass --overwrite."
        )

    cfg, t_base_lidar = lidar_setup(args.sim)
    if args.source:
        from ...lidar.recording import LidarRecording

        source = LidarRecording.load(args.source)
        print(
            f"Building from {args.source} ({len(source)} frames, {source.duration:.0f} s)."
        )
    elif args.sim:
        from ...nav.sim import SimRobot, office_route, office_world

        route = office_route()
        source = SimRobot(office_world(), t_base_lidar, route[0], route=route)
    else:
        from ...lidar import LidarStream

        source = LidarStream(cfg).start()

    view = None
    save = threading.Event()
    if not args.no_view:
        from ...nav.viewer import MapView

        view = MapView(args.port)
        button = view.gui.add_button("Save map")
        button.on_click(lambda _: save.set())
        print(f"Map view on http://localhost:{args.port}")

    odo = LidarOdometry(t_base_lidar)
    builder = MapBuilder()
    print("Mapping. Drive slowly through the space. Ctrl-C saves the map.")
    last_view = last_warn = 0.0
    last_end = None
    try:
        for frame in source.frames():
            end = frame.stamp + frame.duration
            imu = source.imu_between(
                last_end if last_end is not None else end - 1.0, end + 0.005
            )
            last_end = end
            result = odo.update(frame, imu)
            builder.add(result.pose, result.points)
            now = time.monotonic()
            if view is not None and now - last_view > 1.0:
                last_view = now
                from ...nav.geometry import pose2d, transform_points

                p = pose2d(result.pose)
                view.show_robot(p)
                view.show_scan(
                    transform_points(result.pose, result.points.astype(np.float64))
                )
                view.show_trail(np.array(builder.trajectory))
                if len(builder.keyframes) % 5 == 0:
                    view.show_cloud(builder.cloud())
                travelled = (
                    float(
                        np.sum(
                            np.linalg.norm(
                                np.diff(np.array(builder.trajectory)[:, :2], axis=0),
                                axis=1,
                            )
                        )
                    )
                    if len(builder.trajectory) > 1
                    else 0.0
                )
                view.status.content = (
                    f"**Mapping** {args.name}  \nDriven {travelled:.1f} m · "
                    f"{len(builder.keyframes)} keyframes · scan match {result.inlier_ratio * 100:.0f}%"
                )
            if result.inlier_ratio < 0.3 and odo.frames > 10 and now - last_warn > 5.0:
                last_warn = now
                print(
                    "  The scan barely matches. Drive slower, or move away from featureless walls."
                )
            if save.is_set():
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop = getattr(source, "stop", None)
        if stop is not None:
            stop()
    if odo.frames < 10:
        raise SystemExit(
            "Not enough lidar data for a map. Map for at least a few seconds."
        )
    print("Building the map...")
    nav_map = builder.build(args.name)
    path = nav_map.save()
    w, h = (
        nav_map.grid.shape[1] * nav_map.grid.resolution,
        nav_map.grid.shape[0] * nav_map.grid.resolution,
    )
    print(f"Saved map {args.name!r} ({w:.1f} x {h:.1f} m) to {path}.")
    if view is not None:
        view.show_grid(nav_map.grid)
        view.show_cloud(nav_map.cloud)
        view.status.content = f"**Saved** map {args.name}."
