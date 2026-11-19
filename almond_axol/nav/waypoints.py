"""Named places on a map that the base can drive to.

Stored next to the map as ``waypoints.json``::

    {
      "version": 1,
      "waypoints": [
        {"name": "dock", "x": 0.0, "y": 0.0, "yaw": 0.0},
        {"name": "bench", "x": 4.2, "y": -1.5, "yaw": 1.5708}
      ]
    }

``x`` and ``y`` are metres in the map frame and ``yaw`` is the heading in
radians (counter-clockwise from the map's +x axis). These are base poses; the
hand-taught arm poses of ``axol waypoints`` are a separate thing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.state_files import secure_atomic_write_json, secure_read_text
from .geometry import Pose2D
from .mapping import map_path

FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^[^\s/][^/]{0,63}$")


@dataclass
class NavWaypoint:
    name: str
    x: float
    y: float
    yaw: float

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 4),
        }


@dataclass
class NavWaypoints:
    map_name: str
    items: list[NavWaypoint] = field(default_factory=list)

    @staticmethod
    def path_for(map_name: str) -> Path:
        return map_path(map_name) / "waypoints.json"

    @classmethod
    def load(cls, map_name: str) -> NavWaypoints:
        p = cls.path_for(map_name)
        if not p.exists():
            return cls(map_name)
        data = json.loads(secure_read_text(p))
        if data.get("version", FORMAT_VERSION) != FORMAT_VERSION:
            raise ValueError(f"{p} has an unknown version")
        items = [
            NavWaypoint(
                str(w["name"]), float(w["x"]), float(w["y"]), float(w.get("yaw", 0.0))
            )
            for w in data.get("waypoints", [])
        ]
        return cls(map_name, items)

    def save(self) -> Path:
        p = self.path_for(self.map_name)
        secure_atomic_write_json(
            p,
            {"version": FORMAT_VERSION, "waypoints": [w.to_json() for w in self.items]},
            sort_keys=False,
        )
        return p

    def names(self) -> list[str]:
        return [w.name for w in self.items]

    def get(self, name: str) -> NavWaypoint:
        for w in self.items:
            if w.name == name:
                return w
        raise KeyError(f"No waypoint named {name!r} on map {self.map_name!r}.")

    def set(self, name: str, pose: Pose2D) -> NavWaypoint:
        if not _NAME_RE.match(name):
            raise ValueError(
                "A waypoint name can't be empty, start with a space or contain '/'."
            )
        if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
            raise ValueError("Waypoint coordinates must be finite numbers.")
        wp = NavWaypoint(name, pose.x, pose.y, pose.yaw)
        for i, w in enumerate(self.items):
            if w.name == name:
                self.items[i] = wp
                return wp
        self.items.append(wp)
        return wp

    def remove(self, name: str) -> None:
        before = len(self.items)
        self.items = [w for w in self.items if w.name != name]
        if len(self.items) == before:
            raise KeyError(f"No waypoint named {name!r} on map {self.map_name!r}.")
