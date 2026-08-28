"""Cross-process lifecycle marker for an active teleop control loop.

The Rust core starts commanding CAN before PyRoKi has compiled its first IK
solution. That traffic safely holds the robot, but it is not part of the
teleop run operators want to evaluate in Diagnostics. A small marker in
``~/.almond`` lets the independently running diagnostics server distinguish
those phases for both the Python and Rust backends.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


ACTIVITY_PATH = Path.home() / ".almond" / "teleop-control-active.json"


@dataclass(frozen=True)
class TeleopActivity:
    """Identity and wall-clock boundary of one active teleop run."""

    token: str
    pid: int
    started_at: float


def _read(path: Path) -> TeleopActivity | None:
    try:
        raw = json.loads(path.read_text())
        activity = TeleopActivity(
            token=str(raw["token"]),
            pid=int(raw["pid"]),
            started_at=float(raw["startedAt"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if not activity.token or activity.pid <= 0 or activity.started_at <= 0.0:
        return None
    return activity


def active_teleop(path: Path = ACTIVITY_PATH) -> TeleopActivity | None:
    """Return the live teleop run, ignoring markers left behind by a crash."""
    activity = _read(path)
    if activity is None:
        return None
    try:
        os.kill(activity.pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # A live process owned by another uid is still a live marker.
    return activity


class TeleopActivityMarker:
    """Create an atomic marker and remove only the marker this object owns."""

    def __init__(self, path: Path = ACTIVITY_PATH) -> None:
        self._path = path
        self._activity: TeleopActivity | None = None

    def start(self) -> TeleopActivity:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        activity = TeleopActivity(
            token=uuid.uuid4().hex,
            pid=os.getpid(),
            started_at=time.time(),
        )
        tmp = self._path.with_name(f".{self._path.name}.{activity.token}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "token": activity.token,
                    "pid": activity.pid,
                    "startedAt": activity.started_at,
                },
                separators=(",", ":"),
            )
        )
        os.replace(tmp, self._path)
        self._activity = activity
        return activity

    def stop(self) -> None:
        activity = self._activity
        self._activity = None
        if activity is None:
            return
        current = _read(self._path)
        if current is None or current.token != activity.token:
            return
        try:
            self._path.unlink()
        except OSError:
            # This marker is diagnostics-only and must never interfere with
            # controller shutdown. A stale marker is rejected by its dead PID.
            pass
