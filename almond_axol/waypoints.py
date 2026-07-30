"""Hand-taught waypoints: the on-disk format behind ``axol waypoints``.

A waypoint is the pose both arms held at the moment the operator recorded it,
stored as joint angles (7 arm joints in :data:`~almond_axol.constants.Joint`
order plus the normalised gripper, i.e. the same ``(8,)`` vector
:meth:`~almond_axol.robot.base.RobotBase.motion_control` takes). Joint angles
are the source of truth because they are exactly what the robot reported;
the Cartesian pose the arm has to travel through is derived from them by
forward kinematics at planning time (:mod:`almond_axol.kinematics.path`).

The file is JSON so a taught path can be inspected, edited, or checked into a
repository by hand::

    {
      "version": 1,
      "waypoints": [
        {"label": "approach", "left": [0, 0, 0, 0.3, 0, 0, 0, 1.0],
                              "right": [0, 0, 0, -0.3, 0, 0, 0, 1.0]}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import ARM_JOINTS

# 7 arm joints + gripper, matching motion_control's argument shape.
JOINT_VECTOR_LEN = len(ARM_JOINTS) + 1

FORMAT_VERSION = 1


@dataclass
class Waypoint:
    """One recorded dual-arm pose.

    Attributes:
        left: Shape ``(8,)`` left-arm pose — 7 joint angles (rad, joint frame)
            then the gripper normalised to ``[0, 1]``. The joint angles are
            what the arm reported; the gripper is the opening to *command*
            here, which for a grasp is fully closed rather than wherever the
            fingers stalled against the object.
        right: Same for the right arm.
        label: Optional operator-facing name; defaults to the index at save
            time when unset.
    """

    left: np.ndarray
    right: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.left = _as_joint_vector(self.left, "left")
        self.right = _as_joint_vector(self.right, "right")

    def to_json(self) -> dict:
        """Return the JSON-serialisable form of this waypoint."""
        return {
            "label": self.label,
            "left": [round(float(v), 6) for v in self.left],
            "right": [round(float(v), 6) for v in self.right],
        }

    @classmethod
    def from_json(cls, data: dict) -> Waypoint:
        """Build a waypoint from one entry of a waypoint file."""
        return cls(
            left=np.asarray(data["left"], dtype=np.float32),
            right=np.asarray(data["right"], dtype=np.float32),
            label=str(data.get("label", "")),
        )


def _as_joint_vector(values: np.ndarray, side: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (JOINT_VECTOR_LEN,):
        raise ValueError(
            f"{side} waypoint must be a ({JOINT_VECTOR_LEN},) vector "
            f"(7 arm joints + gripper), got shape {arr.shape}"
        )
    return arr


@dataclass
class WaypointSet:
    """An ordered list of waypoints backed by a JSON file.

    The file is the live store: ``axol waypoints`` saves after every record,
    undo, and clear, so a taught path survives a crash or a restart and can be
    replayed later (including in ``--sim``) without re-teaching it.
    """

    waypoints: list[Waypoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.waypoints)

    def __iter__(self):
        return iter(self.waypoints)

    def __getitem__(self, index: int) -> Waypoint:
        return self.waypoints[index]

    def append(self, waypoint: Waypoint) -> None:
        """Add a waypoint to the end of the path."""
        self.waypoints.append(waypoint)

    def pop(self) -> Waypoint | None:
        """Remove and return the last waypoint, or ``None`` if empty."""
        return self.waypoints.pop() if self.waypoints else None

    def clear(self) -> None:
        """Drop every waypoint."""
        self.waypoints.clear()

    @classmethod
    def load(cls, path: str | Path) -> WaypointSet:
        """Read a waypoint file, returning an empty set if it does not exist."""
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        version = data.get("version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{p} is a version {version} waypoint file; this build reads "
                f"version {FORMAT_VERSION}"
            )
        return cls([Waypoint.from_json(entry) for entry in data.get("waypoints", [])])

    def save(self, path: str | Path) -> None:
        """Write the set to ``path``, creating parent directories as needed.

        Written to a temporary file and renamed so an interrupted save cannot
        truncate a path that took real time to teach.
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "waypoints": [
                {**wp.to_json(), "label": wp.label or f"waypoint {i + 1}"}
                for i, wp in enumerate(self.waypoints)
            ],
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(p)
