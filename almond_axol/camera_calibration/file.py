"""The camera calibration file: one per robot, at ``~/.almond/camera_calibration.json``.

``axol calibration.cameras`` writes it; the MCAP recorder reads it at the start
of every episode. It is JSON::

    {
      "format": "almond-camera-calibration",
      "version": 1,
      "robot_serial": "E66138528F2B3F2A",
      "calibration_id": "cam-20261124T181500Z-3f9a21c4",
      "calibrated_at": "2026-11-24T18:15:00Z",
      "due_at": "2026-12-24T18:15:00Z",
      "status": "pass",
      "transforms": [
        {"parent": "root", "child": "head_camera_left_optical",
         "translation_m": [0.07, 0.03, 0.41],
         "rotation_xyzw": [-0.61, 0.61, -0.36, 0.36],
         "camera_serial": 41234567}
      ]
    }

Each transform is the child frame's pose in the parent frame (it maps child
coordinates into parent coordinates): translation in metres and a unit
quaternion ``x, y, z, w``. Frame names are the robot's URDF links. Only the
measured frames are listed; everything else comes from the URDF.

``camera_serial`` (optional) names the camera that was measured. A transform
whose camera has since been swapped is not used. ``due_at`` (optional) is when
the calibration expires: after it, the file is not used. Other keys (the
board, the per-camera report) are informational.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "almond-camera-calibration"
VERSION = 1
DEFAULT_PATH = Path.home() / ".almond" / "camera_calibration.json"
# Where each run keeps its report, images and a copy of the file it wrote.
REPORTS_DIR = Path.home() / ".almond" / "camera_calibration"
# A quaternion this far from unit length is a broken file, not rounding.
_UNIT_TOLERANCE = 1e-3
# How long a calibration stays valid unless the file says otherwise (days).
VALID_DAYS = 30


class CalibrationFileError(ValueError):
    """The file can't be used; the message says why, for the operator."""


@dataclass(frozen=True)
class FrameTransform:
    """One measured frame: ``child``'s pose in ``parent``."""

    parent: str
    child: str
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    camera_serial: int | None = None

    def matrix(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = quaternion_matrix(self.rotation_xyzw)
        out[:3, 3] = self.translation
        return out


@dataclass(frozen=True)
class CameraCalibration:
    """A parsed calibration file."""

    path: Path
    robot_serial: str
    calibration_id: str
    calibrated_at: str
    status: str
    transforms: tuple[FrameTransform, ...]
    raw: bytes
    sha256: str
    due_at: str = ""

    def problem_for(
        self, robot_serial: str | None, now: datetime | None = None
    ) -> str | None:
        """Why this file can't be used on ``robot_serial`` (at ``now``), or
        ``None``."""
        if self.status != "pass":
            return (
                f"{self.path.name} records a failed calibration ({self.calibration_id})"
            )
        due = parse_time(self.due_at) if self.due_at else None
        if due is not None and (now or datetime.now(UTC)) >= due:
            return (
                f"calibration {self.calibration_id} expired on {self.due_at}; "
                "calibrate again"
            )
        if robot_serial and self.robot_serial and self.robot_serial != robot_serial:
            return (
                f"{self.path.name} is for robot {self.robot_serial}, "
                f"not this robot ({robot_serial})"
            )
        if not robot_serial and self.robot_serial:
            return (
                f"this robot's serial is unknown, so {self.path.name} can't be matched"
            )
        return None


def parse_time(text: str) -> datetime | None:
    """An ISO 8601 time (``Z`` or an offset; UTC if neither), or ``None``."""
    try:
        out = datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    return out if out.tzinfo is not None else out.replace(tzinfo=UTC)


def utc_text(when: datetime) -> str:
    """``2026-11-24T18:15:00Z``: a UTC time to the second."""
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def due_after(calibrated_at: str, days: float = VALID_DAYS) -> str:
    """When a calibration made at ``calibrated_at`` expires (``""`` if never)."""
    start = parse_time(calibrated_at)
    if start is None or days <= 0:
        return ""
    return utc_text(start + timedelta(days=days))


def quaternion_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vector(value: Any, n: int, what: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != n:
        raise CalibrationFileError(f"{what} must be a list of {n} numbers")
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise CalibrationFileError(f"{what} must be a list of {n} numbers") from None
    if not all(math.isfinite(v) for v in out):
        raise CalibrationFileError(f"{what} has a value that is not a number")
    return out


def _transform(entry: Any, index: int) -> FrameTransform:
    where = f"transforms[{index}]"
    if not isinstance(entry, Mapping):
        raise CalibrationFileError(f"{where} is not an object")
    parent, child = entry.get("parent"), entry.get("child")
    if not isinstance(parent, str) or not parent:
        raise CalibrationFileError(f"{where} has no parent frame")
    if not isinstance(child, str) or not child:
        raise CalibrationFileError(f"{where} has no child frame")
    t = _vector(entry.get("translation_m"), 3, f"{where}.translation_m")
    q = _vector(entry.get("rotation_xyzw"), 4, f"{where}.rotation_xyzw")
    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise CalibrationFileError(
            f"{where}.rotation_xyzw is not a unit quaternion (length {norm:.4f})"
        )
    q = tuple(v / norm for v in q)
    serial = entry.get("camera_serial")
    try:
        camera_serial = int(serial) if serial not in (None, "") else None
    except (TypeError, ValueError):
        raise CalibrationFileError(f"{where}.camera_serial is not a number") from None
    return FrameTransform(
        parent=parent,
        child=child,
        translation=t,  # type: ignore[arg-type]
        rotation_xyzw=q,  # type: ignore[arg-type]
        camera_serial=camera_serial,
    )


def parse(raw: bytes, path: Path) -> CameraCalibration:
    """Parse a file's bytes. Raises :class:`CalibrationFileError`."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationFileError(f"{path.name} is not valid JSON ({exc})") from None
    if not isinstance(doc, Mapping):
        raise CalibrationFileError(f"{path.name} is not a JSON object")
    fmt = doc.get("format")
    if fmt is not None and fmt != FORMAT:
        raise CalibrationFileError(f"{path.name} is not a camera calibration file")
    version = doc.get("version", VERSION)
    if not isinstance(version, int) or version > VERSION:
        raise CalibrationFileError(
            f"{path.name} is version {version}; update the Axol software to read it"
        )
    calibration_id = doc.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise CalibrationFileError(f"{path.name} has no calibration_id")
    status = doc.get("status")
    if status not in ("pass", "fail"):
        raise CalibrationFileError(f"{path.name} has no status (pass or fail)")
    due_at = str(doc.get("due_at") or "")
    if due_at and parse_time(due_at) is None:
        raise CalibrationFileError(f"{path.name} has a due_at that is not a date")
    entries = doc.get("transforms")
    if not isinstance(entries, list) or not entries:
        raise CalibrationFileError(f"{path.name} lists no transforms")
    transforms = tuple(_transform(e, i) for i, e in enumerate(entries))
    children = [t.child for t in transforms]
    duplicate = sorted({c for c in children if children.count(c) > 1})
    if duplicate:
        raise CalibrationFileError(
            f"{path.name} places {', '.join(duplicate)} more than once"
        )
    return CameraCalibration(
        path=path,
        robot_serial=str(doc.get("robot_serial") or ""),
        calibration_id=calibration_id,
        calibrated_at=str(doc.get("calibrated_at") or ""),
        status=str(status),
        transforms=transforms,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        due_at=due_at,
    )


def load(path: Path = DEFAULT_PATH) -> CameraCalibration | None:
    """The file at ``path``, or ``None`` when there is none.

    Raises :class:`CalibrationFileError` when it exists but can't be used.
    """
    path = Path(path).expanduser()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CalibrationFileError(f"can't read {path}: {exc.strerror}") from None
    return parse(raw, path)


def transform_entry(
    parent: str,
    child: str,
    matrix: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    """A ``transforms`` entry for a 4x4 child-in-parent transform."""
    from ..recording.mcap.urdf_tree import quaternion_xyzw

    t = matrix[:3, 3]
    return {
        "parent": parent,
        "child": child,
        "translation_m": [float(v) for v in t],
        "rotation_xyzw": [float(v) for v in quaternion_xyzw(matrix[:3, :3])],
        **extra,
    }


def document(
    *,
    robot_serial: str,
    calibration_id: str,
    calibrated_at: str,
    status: str,
    transforms: list[dict[str, Any]],
    due_at: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A whole calibration file as a JSON-ready dict.

    ``due_at`` defaults to :data:`VALID_DAYS` after ``calibrated_at``; pass
    ``""`` for a calibration that never expires.
    """
    if due_at is None:
        due_at = due_after(calibrated_at)
    return {
        "format": FORMAT,
        "version": VERSION,
        "robot_serial": robot_serial,
        "calibration_id": calibration_id,
        "calibrated_at": calibrated_at,
        **({"due_at": due_at} if due_at else {}),
        "status": status,
        "transforms": transforms,
        **extra,
    }


def write(doc: Mapping[str, Any], path: Path = DEFAULT_PATH) -> Path:
    """Write ``doc`` to ``path`` atomically (a reader never sees half a file)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(doc, indent=2) + "\n"
    # Check what we write parses back (a bug here must not leave a broken file).
    parse(data.encode(), path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
