"""Lidar mapping, localization and navigation for Axol on Jelly.

The pieces, each usable on its own:

- :class:`LidarOdometry`: the base's pose from scan to scan.
- :class:`MapBuilder` and :class:`Map`: build, save and load maps (a 3D
  cloud plus a 2D :class:`OccupancyGrid`).
- :class:`Localizer`: the base's pose on a saved map.
- :class:`Planner`: collision-free paths on a map's grid.
- :class:`PathFollower`: velocity commands that follow a path.
- :class:`ObstacleMonitor`: stop for obstacles the map doesn't know.
- :class:`NavSession`: all of the above wired to a lidar and Jelly.
- :class:`NavWaypoints`: named places on a map.

``axol nav.map``, ``nav.run``, ``nav.goto`` and ``nav.waypoint`` are built on
these. :mod:`.sim` has a simulated office, lidar and base for testing.
"""

from .controller import DriveCommand, DriveConfig, PathFollower
from .geometry import Pose2D
from .localization import Fix, Localizer, LocalizerConfig
from .mapping import Map, MapBuilder, MapConfig, OccupancyGrid, list_maps
from .odometry import LidarOdometry, OdometryConfig, OdometryResult
from .planner import NoPath, Planner, PlannerConfig
from .safety import ObstacleMonitor, SafetyConfig
from .session import NavConfig, NavSession, NavStatus
from .waypoints import NavWaypoint, NavWaypoints

__all__ = [
    "DriveCommand",
    "DriveConfig",
    "Fix",
    "LidarOdometry",
    "Localizer",
    "LocalizerConfig",
    "Map",
    "MapBuilder",
    "MapConfig",
    "NavConfig",
    "NavSession",
    "NavStatus",
    "NavWaypoint",
    "NavWaypoints",
    "NoPath",
    "ObstacleMonitor",
    "OccupancyGrid",
    "OdometryConfig",
    "OdometryResult",
    "PathFollower",
    "Planner",
    "PlannerConfig",
    "Pose2D",
    "SafetyConfig",
    "list_maps",
]
