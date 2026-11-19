"""Browser view (viser) for mapping and navigation.

Draws the map (occupancy grid on the floor and, optionally, the 3D cloud),
the robot, the live scan, the planned path and the waypoints. The navigation
view adds controls to drive to waypoints, save new ones, and set the robot's
pose or a goal by clicking the floor.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

import numpy as np

from .geometry import Pose2D
from .mapping import Map, OccupancyGrid

_ROBOT = (0.6, 0.6, 0.25)  # Jelly's footprint, drawn as a box


def _viser():
    try:
        import viser
    except ImportError:
        raise SystemExit(
            "The map view needs viser. Install it with: pip install 'almond-axol[sim]'"
        ) from None
    return viser


def _yaw_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def height_colors(z: np.ndarray, lo: float = -0.1, hi: float = 2.5) -> np.ndarray:
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


class MapView:
    """Shared drawing for the mapping and navigation views."""

    def __init__(self, port: int) -> None:
        viser = _viser()
        self.server = viser.ViserServer(port=port)
        self.scene = self.server.scene
        self.gui = self.server.gui
        self.port = port
        self._cloud = None
        self._scan = None
        self._grid = None
        self._path = None
        self._trail = None
        self._robot = self.scene.add_box(
            "/robot/body",
            color=(60, 140, 230),
            dimensions=_ROBOT,
            opacity=0.6,
            position=(0, 0, _ROBOT[2] / 2),
        )
        self._robot_frame = self.scene.add_frame(
            "/robot", axes_length=0.5, axes_radius=0.015
        )
        self._waypoints: dict[str, object] = {}
        self.status = self.gui.add_markdown("Starting...")
        self.prompt = ""

    # -- drawing ------------------------------------------------------------

    def frame_camera(self, cx: float, cy: float, size: float) -> None:
        """Start new viewers looking down at a ``size``-metre area around (cx, cy)."""
        cam = self.server.initial_camera
        cam.position = (cx, cy - 0.35 * size, 1.1 * size)
        cam.look_at = (cx, cy, 0.0)
        cam.up = (0.0, 0.0, 1.0)

    def show_grid(self, grid: OccupancyGrid) -> None:
        img = grid.image()
        h, w = grid.shape
        width, height = w * grid.resolution, h * grid.resolution
        center = (grid.origin_x + width / 2, grid.origin_y + height / 2, -0.01)
        self.frame_camera(center[0], center[1], max(width, height, 4.0))
        if self._grid is not None:
            self._grid.remove()
        self._grid = self.scene.add_image(
            "/map/grid",
            img,
            width,
            height,
            format="png",
            position=center,
            receive_shadow=False,
            cast_shadow=False,
        )

    def show_cloud(
        self, cloud: np.ndarray, max_points: int = 200_000, point_size: float = 0.03
    ) -> None:
        if len(cloud) > max_points:
            idx = np.random.default_rng(0).choice(len(cloud), max_points, replace=False)
            cloud = cloud[idx]
        if self._cloud is None:
            self._cloud = self.scene.add_point_cloud(
                "/map/cloud",
                cloud.astype(np.float32),
                height_colors(cloud[:, 2]),
                point_size=point_size,
            )
        else:
            self._cloud.points = cloud.astype(np.float32)
            self._cloud.colors = height_colors(cloud[:, 2])

    def show_scan(
        self, pts: np.ndarray | None, max_points: int = 20_000, top: float = 2.0
    ) -> None:
        """Draw the live scan, without the floor and the ceiling."""
        if pts is None or len(pts) == 0:
            return
        pts = pts[(pts[:, 2] > 0.05) & (pts[:, 2] < top)]
        if len(pts) == 0:
            return
        if len(pts) > max_points:
            pts = pts[:: int(math.ceil(len(pts) / max_points))]
        colors = np.tile(np.array([[255, 90, 30]], np.uint8), (len(pts), 1))
        if self._scan is None:
            self._scan = self.scene.add_point_cloud(
                "/scan", pts.astype(np.float32), colors, point_size=0.02
            )
        else:
            self._scan.points = pts.astype(np.float32)
            self._scan.colors = colors

    def show_robot(self, pose: Pose2D | None) -> None:
        if pose is None:
            return
        self._robot_frame.position = (pose.x, pose.y, 0.0)
        self._robot_frame.wxyz = _yaw_wxyz(pose.yaw)

    def show_path(self, path: list[tuple[float, float]]) -> None:
        if self._path is not None:
            self._path.remove()
            self._path = None
        if len(path) >= 2:
            p = np.array([[x, y, 0.05] for x, y in path], np.float32)
            segs = np.stack([p[:-1], p[1:]], axis=1)
            self._path = self.scene.add_line_segments(
                "/path", segs, (40, 200, 90), line_width=4
            )

    def show_trail(self, traj: np.ndarray) -> None:
        if len(traj) < 2:
            return
        p = traj[:, :3].astype(np.float32).copy()
        p[:, 2] = 0.04
        segs = np.stack([p[:-1], p[1:]], axis=1)
        if self._trail is not None:
            self._trail.remove()
        self._trail = self.scene.add_line_segments(
            "/trail", segs, (230, 160, 30), line_width=2
        )

    def show_waypoints(self, waypoints: list) -> None:  # noqa: ANN001 - NavWaypoint
        names = {w.name for w in waypoints}
        for name in list(self._waypoints):
            if name not in names:
                for h in self._waypoints.pop(name):
                    h.remove()
        for i, w in enumerate(waypoints):
            for h in self._waypoints.pop(w.name, ()):
                h.remove()
            frame = self.scene.add_frame(
                f"/waypoints/{i}",
                axes_length=0.3,
                axes_radius=0.012,
                position=(w.x, w.y, 0.0),
                wxyz=_yaw_wxyz(w.yaw),
            )
            label = self.scene.add_label(
                f"/waypoints/{i}/label", w.name, position=(0, 0, 0.35)
            )
            self._waypoints[w.name] = (frame, label)

    # -- clicking the floor ---------------------------------------------------

    def click_floor(
        self, prompt: str, on_point: Callable[[float, float], None]
    ) -> None:
        """Wait for one click on the floor and call ``on_point(x, y)``."""
        self.prompt = prompt

        @self.scene.on_pointer_event(event_type="click")
        def _(event) -> None:  # noqa: ANN001
            o, d = event.ray_origin, event.ray_direction
            self.scene.remove_pointer_callback()
            self.prompt = ""
            if o is None or d is None or abs(d[2]) < 1e-6:
                return
            t = -o[2] / d[2]
            if t <= 0:
                return
            on_point(o[0] + t * d[0], o[1] + t * d[1])


class NavView(MapView):
    """The navigation view: map, robot, path, waypoints and controls."""

    def __init__(self, port: int, session, nav_map: Map, *, can_drive: bool) -> None:  # noqa: ANN001
        super().__init__(port)
        self.session = session
        self.can_drive = can_drive
        self.show_grid(nav_map.grid)
        self.show_cloud(nav_map.cloud)
        self._cloud.visible = False
        gui = self.gui
        with gui.add_folder("Go to"):
            self.dropdown = gui.add_dropdown("Waypoint", self._names() or ["(none)"])
            self.go = gui.add_button(
                "Go to waypoint", disabled=not can_drive, icon=None
            )
            self.tour = gui.add_button("Visit all in order", disabled=not can_drive)
            self.loop = gui.add_checkbox("Repeat", False)
            self.click_goal = gui.add_button(
                "Click a goal on the map", disabled=not can_drive
            )
            self.stop = gui.add_button("Stop", color="red")
        with gui.add_folder("Waypoints"):
            self.new_name = gui.add_text("Name", "")
            self.save_here = gui.add_button("Save current position")
            self.remove = gui.add_button("Delete selected waypoint")
        with gui.add_folder("Robot"):
            self.set_pose = gui.add_button("Set robot pose by clicking")
            self.show_3d = gui.add_checkbox("Show 3D map", False)
        self._wire()
        self.show_waypoints(session.waypoints.items)

    def _names(self) -> list[str]:
        return self.session.waypoints.names()

    def _refresh_waypoints(self) -> None:
        names = self._names()
        self.dropdown.options = names or ["(none)"]
        self.show_waypoints(self.session.waypoints.items)

    def _wire(self) -> None:
        s = self.session

        @self.go.on_click
        def _(_) -> None:
            if self.dropdown.value in self._names():
                s.goto([self.dropdown.value])

        @self.tour.on_click
        def _(_) -> None:
            if self._names():
                s.goto(self._names(), loop=self.loop.value)

        @self.stop.on_click
        def _(_) -> None:
            s.cancel("Stopped from the map view.")

        @self.save_here.on_click
        def _(_) -> None:
            name = self.new_name.value.strip()
            if not name:
                self.prompt = "Type a name for the waypoint first."
                return
            try:
                s.add_waypoint(name)
            except (RuntimeError, ValueError) as exc:
                self.prompt = str(exc)
                return
            self.prompt = f"Saved waypoint {name}."
            self.new_name.value = ""
            self._refresh_waypoints()

        @self.remove.on_click
        def _(_) -> None:
            name = self.dropdown.value
            if name in self._names():
                s.waypoints.remove(name)
                s.waypoints.save()
                self._refresh_waypoints()

        @self.click_goal.on_click
        def _(_) -> None:
            def at(x: float, y: float) -> None:
                pose = s.status().pose
                yaw = math.atan2(y - pose.y, x - pose.x) if pose else 0.0
                s.goto([Pose2D(x, y, yaw)])

            self.click_floor("Click the floor where the robot should go.", at)

        @self.set_pose.on_click
        def _(_) -> None:
            def at(x: float, y: float) -> None:
                def facing(fx: float, fy: float) -> None:
                    s.set_pose(Pose2D(x, y, math.atan2(fy - y, fx - x)))

                self.click_floor("Now click a point the robot is facing.", facing)

            self.click_floor("Click the floor where the robot is.", at)

        @self.show_3d.on_update
        def _(_) -> None:
            self._cloud.visible = self.show_3d.value

    def run(self, stop: threading.Event) -> None:
        """Refresh the view until ``stop`` is set."""
        while not stop.is_set():
            st = self.session.status()
            self.show_robot(st.pose)
            self.show_path(st.path)
            self.show_scan(self.session.last_points)
            pose = (
                f"x {st.pose.x:.2f} m, y {st.pose.y:.2f} m, heading {math.degrees(st.pose.yaw):.0f}°"
                if st.pose
                else "unknown"
            )
            lines = [f"**{st.state.capitalize()}**: {st.message}", f"Position: {pose}"]
            lines.append(f"Map match: {st.match_ratio * 100:.0f}% ({st.localization})")
            if st.goal:
                lines.append(f"Distance left: {st.remaining:.1f} m")
            if math.isfinite(st.nearest_obstacle):
                lines.append(f"Nearest obstacle: {st.nearest_obstacle:.2f} m")
            if self.prompt:
                lines.append(f"*{self.prompt}*")
            if not self.can_drive:
                lines.append("Driving is off: Jelly isn't connected to this computer.")
            self.status.content = "  \n".join(lines)
            time.sleep(0.1)
