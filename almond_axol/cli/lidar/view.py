"""
axol lidar.view

Show the live point cloud in a browser (viser). Points are drawn in the
robot's base frame using the mount from ``axol lidar.setup``, coloured by
height or reflectivity. Sliders in the view set how many seconds of points
to keep and how big to draw them.

The Mid-360 places about 20k points per 0.1 s frame, roughly 1.3° apart, but
never in the same place twice, so the picture sharpens the longer points are
kept (while the robot is still).

Usage:
    axol lidar.view                 # then open http://<robot>:8010
    axol lidar.view --port 8011 --history 5
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

from ...lidar import LidarConfig, LidarStream


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "lidar.view", help="Show the live lidar point cloud in a browser."
    )
    p.add_argument("--port", type=int, default=8010, help="Viewer port (default 8010).")
    p.add_argument(
        "--history",
        type=float,
        default=2.0,
        help="Seconds of points to show at once (default 2, at most 10). The "
        "Mid-360 fills in its view over time, so longer looks denser.",
    )
    p.add_argument(
        "--point-size",
        type=float,
        default=0.01,
        help="Point size in metres (default 0.01).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=600_000,
        help="Most points sent to the browser at once (default 600000).",
    )
    p.set_defaults(func=run)


def height_colors(z: np.ndarray, lo: float = -0.2, hi: float = 2.5) -> np.ndarray:
    """Blue (low) to red (high) colours for heights, as uint8 RGB."""
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def to_base(xyz: np.ndarray, t_base_lidar: np.ndarray) -> np.ndarray:
    return (xyz @ t_base_lidar[:3, :3].T + t_base_lidar[:3, 3]).astype(np.float32)


def reflectivity_colors(refl: np.ndarray) -> np.ndarray:
    """Dark (matte) to bright (shiny) grey-blue colours for reflectivity, uint8 RGB."""
    t = np.clip(refl.astype(np.float32) / 150.0, 0.0, 1.0) ** 0.6
    base = np.array([60, 70, 90], np.float32)
    top = np.array([255, 245, 220], np.float32)
    return (base + t[:, None] * (top - base)).astype(np.uint8)


# Roughly what a browser on Wi-Fi keeps up with (bytes/s of point updates).
_SEND_BUDGET = 12e6
_MAX_HISTORY_S = 10.0


def run(args: argparse.Namespace) -> None:
    try:
        import viser
    except ImportError:
        raise SystemExit(
            "The viewer needs viser. Install it with: pip install 'almond-axol[sim]'"
        ) from None
    cfg = LidarConfig.load()
    t_base_lidar = cfg.mount.matrix()
    server = viser.ViserServer(port=args.port)
    server.initial_camera.position = (-6.0, -6.0, 5.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.5)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.scene.add_grid("/grid", width=10.0, height=10.0, cell_size=0.5)
    server.scene.add_frame(
        "/lidar", axes_length=0.15, axes_radius=0.005
    ).position = tuple(t_base_lidar[:3, 3])
    cloud = server.scene.add_point_cloud(
        "/points",
        np.zeros((1, 3), np.float32),
        np.zeros((1, 3), np.uint8),
        point_size=args.point_size,
        point_shape="circle",
        precision="float32",
    )
    status = server.gui.add_markdown("Starting...")
    history = server.gui.add_slider(
        "History (s)",
        0.1,
        _MAX_HISTORY_S,
        0.1,
        min(max(args.history, 0.1), _MAX_HISTORY_S),
        hint="Seconds of points shown at once. Longer is denser while the robot is still.",
    )
    size = server.gui.add_slider(
        "Point size (cm)", 0.2, 5.0, 0.1, round(args.point_size * 100, 1)
    )
    ceiling = server.gui.add_slider(
        "Hide above (m)",
        0.5,
        6.0,
        0.1,
        2.2,
        hint="Hide points higher than this above the floor, such as the ceiling.",
    )
    color_by = server.gui.add_dropdown("Colour by", ["Height", "Reflectivity"])

    @size.on_update
    def _(_) -> None:
        cloud.point_size = size.value / 100.0

    frames_kept = round(_MAX_HISTORY_S / cfg.frame_period)
    recent: collections.deque[tuple[np.ndarray, np.ndarray]] = collections.deque(
        maxlen=frames_kept
    )
    print(f"Lidar viewer on http://localhost:{args.port} (Ctrl-C to stop)")
    with LidarStream(cfg) as stream:
        last_status = last_send = 0.0
        interval = 0.0
        try:
            for frame in stream.frames():
                recent.append((to_base(frame.xyz, t_base_lidar), frame.reflectivity))
                now = time.monotonic()
                if now - last_send < interval:
                    continue
                keep = max(1, round(history.value / cfg.frame_period))
                shown = list(recent)[-keep:]
                pts = np.concatenate([p for p, _ in shown])
                refl = np.concatenate([r for _, r in shown])
                if ceiling.value < 6.0:
                    low = pts[:, 2] <= ceiling.value
                    pts, refl = pts[low], refl[low]
                if len(pts) == 0:
                    continue
                if len(pts) > args.max_points:
                    step = int(np.ceil(len(pts) / args.max_points))
                    pts, refl = pts[::step], refl[::step]
                cloud.points = pts
                cloud.colors = (
                    reflectivity_colors(refl)
                    if color_by.value == "Reflectivity"
                    else height_colors(pts[:, 2], hi=min(ceiling.value, 2.5))
                )
                last_send = now
                # Big clouds go out less often so a browser on Wi-Fi keeps up.
                interval = len(pts) * 15 / _SEND_BUDGET
                if now - last_status > 1.0:
                    last_status = now
                    s = stream.stats()
                    status.content = (
                        f"**{len(pts):,} points** · {s.points_per_s / 1000:.0f}k/s · "
                        f"{s.frames_per_s:.1f} frames/s · IMU {s.imu_per_s:.0f} Hz"
                    )
        except KeyboardInterrupt:
            pass
