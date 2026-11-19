"""
axol lidar.view

Show the live point cloud in a browser (viser). Points are drawn in the
robot's base frame using the mount from ``axol lidar.setup``, coloured by
height.

Usage:
    axol lidar.view                 # then open http://<robot>:8010
    axol lidar.view --port 8011 --history 1.0
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
        default=0.5,
        help="Seconds of points to show at once (default 0.5). The Mid-360 "
        "fills in its field of view over time, so longer looks denser.",
    )
    p.add_argument(
        "--point-size", type=float, default=0.02, help="Point size in metres."
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
    )
    status = server.gui.add_markdown("Starting...")
    keep = max(1, round(args.history / cfg.frame_period))
    recent: collections.deque[np.ndarray] = collections.deque(maxlen=keep)
    print(f"Lidar viewer on http://localhost:{args.port} (Ctrl-C to stop)")
    with LidarStream(cfg) as stream:
        last_status = 0.0
        try:
            for frame in stream.frames():
                recent.append(to_base(frame.xyz, t_base_lidar))
                pts = np.concatenate(recent)
                cloud.points = pts
                cloud.colors = height_colors(pts[:, 2])
                now = time.monotonic()
                if now - last_status > 1.0:
                    last_status = now
                    s = stream.stats()
                    status.content = (
                        f"**{len(pts):,} points** · {s.points_per_s / 1000:.0f}k/s · "
                        f"{s.frames_per_s:.1f} frames/s · IMU {s.imu_per_s:.0f} Hz"
                    )
        except KeyboardInterrupt:
            pass
