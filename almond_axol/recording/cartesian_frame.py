"""Pose-frame provenance for Axol Cartesian LeRobot datasets."""

from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..utils.state_files import secure_atomic_write_json

CARTESIAN_FRAME_ID = "flu-urdf-root-v0.1.32"
"""Forward-facing FLU world frame introduced by axol v0.1.32."""

URDF_ROOT_YAW_RADIANS = math.pi / 2

MANTIS_TCP_TRANSFORM_KEY = "mantis_tcp_transform"
"""Marker field naming the tracker→gripper transforms a Mantis dataset used.

Written by ``collect-data --mantis`` (see
:func:`almond_axol.mantis.calibration.tcp_transform_provenance`). Mantis
datasets without it were recorded by axol <= 0.2.4 with the retired
Rx(+90°) Vive rotation and need ``migrate-dataset --mantis-tcp-rotation``.
"""


def _axol_version() -> str:
    try:
        return version("almond-axol")
    except PackageNotFoundError:
        return "unknown"


def marker_path(dataset_root: Path | str) -> Path:
    return Path(dataset_root) / "meta" / "axol.json"


def read_cartesian_frame_marker(dataset_root: Path | str) -> dict[str, Any] | None:
    """Return the parsed ``meta/axol.json`` or ``None`` when absent."""
    path = marker_path(dataset_root)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return data


def write_cartesian_frame_marker(
    dataset_root: Path | str,
    *,
    migration: dict[str, Any] | None = None,
    mantis_tcp_transform: dict[str, Any] | None = None,
) -> None:
    """Atomically record the frame used by a new or migrated Cartesian dataset.

    ``mantis_tcp_transform`` is the provenance dict a Mantis recording used,
    stored so later constant changes can be migrated without guessing.
    """
    axol_version = _axol_version()
    data: dict[str, Any] = {
        "schema_version": 1,
        "cartesian_pose_frame": CARTESIAN_FRAME_ID,
        "urdf_root_yaw_radians": URDF_ROOT_YAW_RADIANS,
    }
    if migration is not None:
        data["migrated_by_axol_version"] = axol_version
        data["migration"] = migration
    else:
        data["recorded_by_axol_version"] = axol_version
    if mantis_tcp_transform is not None:
        data[MANTIS_TCP_TRANSFORM_KEY] = mantis_tcp_transform

    secure_atomic_write_json(marker_path(dataset_root), data, sort_keys=False, indent=4)


def update_cartesian_frame_marker(
    dataset_root: Path | str,
    *,
    migration: dict[str, Any],
    mantis_tcp_transform: dict[str, Any] | None = None,
) -> None:
    """Append a migration record to an existing marker, keeping its fields.

    Unlike :func:`write_cartesian_frame_marker` this preserves the recording
    provenance (version, frame) and accumulates ``migrations`` so a dataset
    that went through several repairs documents all of them.
    """
    data = read_cartesian_frame_marker(dataset_root) or {
        "schema_version": 1,
        "cartesian_pose_frame": CARTESIAN_FRAME_ID,
        "urdf_root_yaw_radians": URDF_ROOT_YAW_RADIANS,
    }
    migrations = data.get("migrations")
    if not isinstance(migrations, list):
        migrations = []
    migrations.append({**migration, "axol_version": _axol_version()})
    data["migrations"] = migrations
    if mantis_tcp_transform is not None:
        data[MANTIS_TCP_TRANSFORM_KEY] = mantis_tcp_transform
    secure_atomic_write_json(marker_path(dataset_root), data, sort_keys=False, indent=4)
