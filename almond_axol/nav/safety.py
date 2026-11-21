"""Stop the base for obstacles the map doesn't know about.

Every scan is checked, in the base frame, for points around the base: a
*stop zone* (the footprint grown by :attr:`SafetyConfig.stop_margin`,
stretched forward along the direction of travel by the distance the base
needs to stop) and a wider *slow zone*. Points must sit between
:attr:`SafetyConfig.band_low` and :attr:`SafetyConfig.band_high` above the
floor to count, so the floor and the ceiling don't trigger it. Points on the
robot itself were removed earlier, by the odometry's ``self_box``.

This is a second line of defence behind the planner, and both are behind
the E-Stop. It only sees what the lidar sees: low objects below the lidar's
field of view near the base, glass, and thin rails can be missed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyConfig:
    """Tuning for :class:`ObstacleMonitor`.

    Attributes:
        half_length:  Half the base's length along x (m).
        half_width:   Half the base's width along y (m).
        stop_margin:  Clear distance kept around the footprint (m).
        slow_margin:  Beyond the stop zone, slow down within this (m).
        stop_time:    Seconds of travel added ahead of the base in the
                      direction of motion (the base needs time to stop).
        band_low:     Ignore points lower than this above the floor (m).
        band_high:    Ignore points higher than this (m).
        min_points:   Points needed in a zone to trigger it (rejects noise).
    """

    half_length: float = 0.30
    half_width: float = 0.30
    stop_margin: float = 0.20
    slow_margin: float = 0.50
    stop_time: float = 1.5
    band_low: float = 0.08
    band_high: float = 1.90
    min_points: int = 8


@dataclass
class SafetyVerdict:
    stop: bool
    slow: bool
    nearest: float  # distance (m) from the footprint edge to the closest point
    stop_points: int


class ObstacleMonitor:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def check(
        self, points_base: np.ndarray, vx: float = 0.0, vy: float = 0.0
    ) -> SafetyVerdict:
        cfg = self.config
        if points_base is None or len(points_base) == 0:
            return SafetyVerdict(False, False, math.inf, 0)
        z = points_base[:, 2]
        pts = points_base[(z >= cfg.band_low) & (z <= cfg.band_high), :2]
        if len(pts) == 0:
            return SafetyVerdict(False, False, math.inf, 0)
        # Distance from the footprint rectangle (0 inside it).
        dx = np.maximum(np.abs(pts[:, 0]) - cfg.half_length, 0.0)
        dy = np.maximum(np.abs(pts[:, 1]) - cfg.half_width, 0.0)
        edge = np.hypot(dx, dy)
        nearest = float(edge.min())
        stop = edge <= cfg.stop_margin
        speed = math.hypot(vx, vy)
        if speed > 1e-3:
            # Sweep the stop zone ahead along the direction of travel.
            ux, uy = vx / speed, vy / speed
            ahead = speed * cfg.stop_time
            along = pts[:, 0] * ux + pts[:, 1] * uy
            across = -pts[:, 0] * uy + pts[:, 1] * ux
            reach = abs(cfg.half_length * ux) + abs(cfg.half_width * uy)
            width = abs(cfg.half_length * uy) + abs(cfg.half_width * ux)
            swept = (
                (along > 0)
                & (along <= reach + cfg.stop_margin + ahead)
                & (np.abs(across) <= width + cfg.stop_margin)
            )
            stop |= swept
        slow = edge <= cfg.stop_margin + cfg.slow_margin
        n_stop = int(stop.sum())
        return SafetyVerdict(
            n_stop >= cfg.min_points, int(slow.sum()) >= cfg.min_points, nearest, n_stop
        )
