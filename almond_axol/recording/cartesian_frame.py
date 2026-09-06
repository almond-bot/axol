"""Pose-frame provenance for Axol Cartesian LeRobot datasets."""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..utils.state_files import secure_atomic_write_json

CARTESIAN_FRAME_ID = "flu-urdf-root-v0.1.32"
"""Forward-facing FLU world frame introduced by axol v0.1.32."""

URDF_ROOT_YAW_RADIANS = math.pi / 2


def write_cartesian_frame_marker(
    dataset_root: Path | str, *, migration: dict[str, Any] | None = None
) -> None:
    """Atomically record the frame used by a new or migrated Cartesian dataset."""
    try:
        axol_version = version("almond-axol")
    except PackageNotFoundError:
        axol_version = "unknown"
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

    path = Path(dataset_root) / "meta" / "axol.json"
    secure_atomic_write_json(path, data, sort_keys=False, indent=4)
