"""In-place repairs for Axol Cartesian LeRobot datasets.

Two migrations share one backup/transform/refresh-stats pipeline:

``--from-version`` — pre-v0.1.32 URDF frame. Axol v0.1.32 added a +90 degree
yaw to the URDF root.  Cartesian datasets recorded by earlier versions
therefore contain end-effector poses in the old world frame, and replaying
them with the new URDF rotates the motion by roughly 90 degrees.  The
migration applies the same rigid transform as the URDF change to every
Cartesian action and observation.

``--mantis-tcp-rotation`` — Mantis datasets recorded by axol <= 0.2.4.  Those
mapped every tracker pose to the gripper through an Rx(+90 deg)
tracker→gripper rotation that was ~90 degrees off the flat-back Vive mount
(corrected to Ry(180 deg) in :mod:`almond_axol.mantis.calibration`), so the
recorded gripper orientations are pitched away from where the operator held
the rig.  The migration replaces that body-frame rotation.  ``--swap-sides``
additionally repairs a session whose left/right trackers were bound to the
opposite rigs: the engage fit then faced the base backwards, so the columns
are swapped and every pose is yawed 180 degrees about the rest-pose gripper
midpoint.

Both refresh per-episode and dataset statistics afterwards.  Videos, gripper
values, timestamps, and all non-Cartesian values are left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.state_files import (
    secure_atomic_copy_file,
    secure_atomic_write_json,
    secure_unlink,
)

_MIGRATION_ID = "axol-urdf-root-yaw-v0.1.32"
_MANTIS_TCP_MIGRATION_ID = "axol-mantis-vive-tcp-rotation-v0.2.5"
_FIELDS = ("action", "observation.state")
_ARMS = ("left", "right")

# Per-arm Cartesian column layout: (position indices, rotation-vector indices).
_ArmLayout = tuple[list[int], list[int]]


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    secure_atomic_write_json(path, data, sort_keys=False, indent=4)


def _resolve_dataset(repo_id: str, root: str | None) -> Path:
    from ..recording.datasets import is_dataset_dir, lerobot_home, list_datasets

    direct = Path(repo_id).expanduser()
    if is_dataset_dir(direct):
        return direct.resolve()

    base = Path(root).expanduser() if root else lerobot_home()
    candidate = base / repo_id
    if is_dataset_dir(candidate):
        return candidate.resolve()

    available = list_datasets(base)
    listing = "\n".join(f"  {item.repo_id}" for item in available)
    suffix = f"\nDatasets found under {base}:\n{listing}" if listing else ""
    raise FileNotFoundError(
        f"No local LeRobot dataset found at {candidate} (missing meta/info.json)."
        f"{suffix}"
    )


def _validate_source_version(source_version: str) -> str:
    """Normalize and verify that the supplied recording version is old."""
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[.-].*)?", source_version)
    if match is None:
        raise ValueError(
            f"Invalid --from-version {source_version!r}; expected a version like v0.1.29."
        )
    parsed = tuple(int(part) for part in match.groups())
    if parsed >= (0, 1, 32):
        raise ValueError(
            f"axol {source_version} already uses the forward-facing URDF frame; "
            "this dataset must not be migrated."
        )
    return ".".join(str(part) for part in parsed)


def _pose_indices(info: dict[str, Any]) -> dict[str, list[tuple[list[int], list[int]]]]:
    """Return position/rotation indices for every Cartesian field and arm."""
    layouts: dict[str, list[tuple[list[int], list[int]]]] = {}
    features = info.get("features", {})
    for field in _FIELDS:
        spec = features.get(field)
        if spec is None:
            continue
        names = spec.get("names") or []
        ee_names = [name for name in names if isinstance(name, str) and "_ee." in name]
        if not ee_names:
            continue
        arms: list[tuple[list[int], list[int]]] = []
        for arm in _ARMS:
            expected = [
                f"{arm}_ee.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")
            ]
            missing = [name for name in expected if name not in names]
            if missing:
                raise ValueError(
                    f"{field} has an unrecognized Cartesian layout; missing {missing}. "
                    "No files were changed."
                )
            arms.append(
                (
                    [names.index(name) for name in expected[:3]],
                    [names.index(name) for name in expected[3:]],
                )
            )
        layouts[field] = arms

    if "action" not in layouts:
        raise ValueError(
            "This is not an Axol Cartesian dataset: action does not contain the "
            "left/right 6-axis EE pose names. Joint-space datasets do not need this migration."
        )
    return layouts


def _transform_matrix(matrix: Any, arms: list[tuple[list[int], list[int]]]) -> Any:
    """Premultiply each pose in ``matrix`` by the URDF's +90 degree root yaw."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from ..recording.cartesian_frame import URDF_ROOT_YAW_RADIANS

    out = np.asarray(matrix, dtype=np.float32).copy()
    root_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, URDF_ROOT_YAW_RADIANS]))
    for pos_idx, rot_idx in arms:
        pos = out[:, pos_idx].copy()
        # Rz(+90): (x, y, z) -> (-y, x, z).  Spell this out to avoid an
        # unnecessary float round trip for positions.
        out[:, pos_idx[0]] = -pos[:, 1]
        out[:, pos_idx[1]] = pos[:, 0]
        out[:, pos_idx[2]] = pos[:, 2]
        old_rotation = Rotation.from_rotvec(out[:, rot_idx])
        out[:, rot_idx] = (root_rotation * old_rotation).as_rotvec().astype(np.float32)
    return out


def _rest_gripper_midpoint() -> Any:
    """Base-frame midpoint of the two rest-pose gripper TCPs.

    The absolute engage fit places this point on the measured gripper
    midpoint, so a 180 degree base-yaw error (swapped rigs) pivots every
    recorded position about it. Computed from the default teleop rest pose
    with the URDF forward kinematics (JAX; a few seconds on first use).
    """
    import numpy as np

    from ..kinematics.fk import AxolForwardKinematics
    from ..teleop.config import VRTeleopConfig

    config = VRTeleopConfig()
    left, right = AxolForwardKinematics().ee_poses(
        np.asarray(config.rest_pose_left, dtype=np.float32),
        np.asarray(config.rest_pose_right, dtype=np.float32),
    )
    return 0.5 * (np.asarray(left[:3], dtype=np.float64) + right[:3])


def _mantis_tcp_correction_matrix() -> Any:
    """Body-frame rotation turning a legacy Rx(+90) gripper pose into Ry(180)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from ..mantis.calibration import (
        LEGACY_VIVE_TCP_ROTATION_QUAT,
        VIVE_TCP_ROTATION_QUAT,
    )

    legacy = Rotation.from_quat(np.asarray(LEGACY_VIVE_TCP_ROTATION_QUAT))
    current = Rotation.from_quat(np.asarray(VIVE_TCP_ROTATION_QUAT))
    # Recorded: R_rec = R_base^T R_tracker R_legacy. Wanted:
    # R_base^T R_tracker R_current = R_rec R_legacy^-1 R_current.
    return legacy.inv() * current


def _transform_mantis_tcp(
    matrix: Any,
    arms: list[_ArmLayout],
    *,
    swap_sides: bool,
    rest_midpoint: Any,
) -> Any:
    """Replace the legacy tracker→gripper rotation; optionally un-swap the rigs.

    Positions are untouched by the rotation fix (the mount translation was
    not changed). With ``swap_sides`` the left/right pose columns are
    exchanged first, then every pose is yawed 180 degrees about the
    rest-pose gripper midpoint — the engage fit's own pivot — so the frame
    matches the fit that a correctly bound session would have produced.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    out = np.asarray(matrix, dtype=np.float32).copy()
    correction = _mantis_tcp_correction_matrix()
    if swap_sides:
        if len(arms) != 2:
            raise ValueError("--swap-sides needs both a left and a right arm layout.")
        (l_pos, l_rot), (r_pos, r_rot) = arms
        left_pose = out[:, [*l_pos, *l_rot]].copy()
        out[:, [*l_pos, *l_rot]] = out[:, [*r_pos, *r_rot]]
        out[:, [*r_pos, *r_rot]] = left_pose
    yaw_flip = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi]))
    mid = np.asarray(rest_midpoint, dtype=np.float64)
    for pos_idx, rot_idx in arms:
        old_rotation = Rotation.from_rotvec(out[:, rot_idx].astype(np.float64))
        fixed = old_rotation * correction
        if swap_sides:
            fixed = yaw_flip * fixed
            pos = out[:, pos_idx].astype(np.float64) - mid
            # Rz(180): (x, y, z) -> (-x, -y, z) about the pivot.
            out[:, pos_idx[0]] = (-pos[:, 0] + mid[0]).astype(np.float32)
            out[:, pos_idx[1]] = (-pos[:, 1] + mid[1]).astype(np.float32)
        out[:, rot_idx] = fixed.as_rotvec().astype(np.float32)
    return out


def _table_matrix(table: Any, field: str) -> Any:
    import numpy as np

    return np.asarray(table[field].combine_chunks().to_pylist(), dtype=np.float32)


def _replace_matrix(table: Any, field: str, matrix: Any) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(field)
    original_type = table.schema.field(index).type
    values = pa.array(matrix.tolist(), type=original_type)
    return table.set_column(index, field, values)


def _write_parquet_atomic(path: Path, table: Any) -> None:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    compression = "snappy"
    if parquet.metadata.num_row_groups and parquet.metadata.num_columns:
        compression = parquet.metadata.row_group(0).column(0).compression.lower()
    mode = stat.S_IMODE(path.lstat().st_mode)
    with tempfile.TemporaryDirectory(prefix="axol-dataset-migrate-") as directory:
        staged = Path(directory) / "migrated.parquet"
        pq.write_table(table, staged, compression=compression)
        secure_atomic_copy_file(staged, path, mode=mode)


def _backup_files(
    dataset_root: Path, files: list[Path], backup_root: Path, migration_id: str
) -> dict[str, Any]:
    marker = dataset_root / "meta" / "axol.json"
    manifest: dict[str, Any] = {
        "migration": migration_id,
        "state": "backed_up",
        "files": [str(path.relative_to(dataset_root)) for path in files],
        "marker_existed": marker.exists(),
    }
    files_root = backup_root / "files"
    for source in files:
        destination = files_root / source.relative_to(dataset_root)
        mode = stat.S_IMODE(source.lstat().st_mode)
        secure_atomic_copy_file(source, destination, mode=mode)
    _atomic_json(backup_root / "manifest.json", manifest)
    return manifest


def _restore_backup(
    dataset_root: Path, backup_root: Path, manifest: dict[str, Any]
) -> None:
    files_root = backup_root / "files"
    for relative in manifest["files"]:
        source = files_root / relative
        destination = dataset_root / relative
        mode = stat.S_IMODE(source.lstat().st_mode)
        secure_atomic_copy_file(source, destination, mode=mode)
    if not manifest.get("marker_existed", False):
        secure_unlink(dataset_root / "meta" / "axol.json", missing_ok=True)


def _stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import numpy as np

    nested: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("stats/") or value is None:
            continue
        _, feature, statistic = key.split("/", 2)
        nested.setdefault(feature, {})[statistic] = np.asarray(value)
    return nested


def _refresh_episode_stats(
    dataset_root: Path,
    info: dict[str, Any],
    layouts: dict[str, list[tuple[list[int], list[int]]]],
    episode_files: list[Path],
) -> list[dict[str, dict[str, Any]]]:
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import compute_episode_stats

    data_template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    features = info["features"]
    all_stats: list[dict[str, dict[str, Any]]] = []
    cached_path: Path | None = None
    cached_table: Any = None

    for episode_file in episode_files:
        metadata_table = pq.read_table(episode_file)
        rows = metadata_table.to_pylist()
        for row in rows:
            data_path = dataset_root / data_template.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            if data_path != cached_path:
                cached_table = pq.read_table(
                    data_path, columns=[*layouts, "episode_index"]
                )
                cached_path = data_path
            episode_index = int(row["episode_index"])
            episode_ids = np.asarray(cached_table["episode_index"].combine_chunks())
            mask = episode_ids == episode_index
            if not mask.any():
                raise ValueError(f"Episode {episode_index} has no rows in {data_path}.")

            for field in layouts:
                matrix = _table_matrix(cached_table, field)[mask]
                computed = compute_episode_stats(
                    {field: matrix}, {field: features[field]}
                )[field]
                for statistic, value in computed.items():
                    row[f"stats/{field}/{statistic}"] = np.asarray(value).tolist()
            all_stats.append(_stats_from_row(row))

        import pyarrow as pa

        updated = pa.Table.from_pylist(rows, schema=metadata_table.schema)
        _write_parquet_atomic(episode_file, updated)

    return all_stats


@dataclass(frozen=True)
class _Plan:
    """One in-place migration: eligibility, the pose transform, the marker."""

    migration_id: str
    # Raises ValueError when the dataset (info.json + parsed meta/axol.json,
    # or None when absent) must not receive this migration.
    check_eligible: Callable[[dict[str, Any], dict[str, Any] | None], None]
    # Rewrites one (rows, dims) float matrix given its per-arm layout.
    transform: Callable[[Any, list[_ArmLayout]], Any]
    # Records the migration in meta/axol.json once the data is rewritten.
    write_marker: Callable[[Path], None]


def _run_migration(
    dataset_root: Path, plan: _Plan, *, dry_run: bool = False
) -> dict[str, int]:
    """Back up, transform, refresh stats, and mark — or restore on failure."""
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import serialize_dict

    from ..recording.cartesian_frame import read_cartesian_frame_marker

    dataset_root = dataset_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    layouts = _pose_indices(info)
    data_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    stats_path = dataset_root / "meta" / "stats.json"
    if not data_files or not episode_files or not stats_path.is_file():
        raise ValueError(
            "Dataset is incomplete: expected data parquet files, episode metadata, "
            "and meta/stats.json. No files were changed."
        )

    backup_root = dataset_root / "meta" / "migrations" / plan.migration_id
    manifest_path = backup_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("state") == "complete":
            raise ValueError(f"Dataset was already migrated ({manifest_path}).")
        if not dry_run:
            print(
                "Recovering an interrupted migration from its backup before retrying."
            )
            _restore_backup(dataset_root, backup_root, manifest)

    plan.check_eligible(info, read_cartesian_frame_marker(dataset_root))

    row_count = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
    summary = {
        "data_files": len(data_files),
        "episode_files": len(episode_files),
        "rows": row_count,
    }
    if dry_run:
        return summary

    marker_path = dataset_root / "meta" / "axol.json"
    files_to_backup = [*data_files, *episode_files, stats_path]
    if marker_path.is_file():
        files_to_backup.append(marker_path)
    if not manifest_path.exists():
        manifest = _backup_files(
            dataset_root, files_to_backup, backup_root, plan.migration_id
        )
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest["state"] = "backed_up"
        _atomic_json(manifest_path, manifest)

    try:
        manifest["state"] = "transforming_data"
        _atomic_json(manifest_path, manifest)
        for path in data_files:
            table = pq.read_table(path)
            for field, arms in layouts.items():
                if field not in table.column_names:
                    raise ValueError(f"{path} is missing the declared {field} column.")
                table = _replace_matrix(
                    table, field, plan.transform(_table_matrix(table, field), arms)
                )
            _write_parquet_atomic(path, table)

        manifest["state"] = "refreshing_stats"
        _atomic_json(manifest_path, manifest)
        episode_stats = _refresh_episode_stats(
            dataset_root, info, layouts, episode_files
        )
        if not episode_stats:
            raise ValueError("No episode statistics were found.")
        _atomic_json(stats_path, serialize_dict(aggregate_stats(episode_stats)))

        plan.write_marker(dataset_root)
        manifest["state"] = "complete"
        _atomic_json(manifest_path, manifest)
    except BaseException:
        _restore_backup(dataset_root, backup_root, manifest)
        manifest["state"] = "failed_restored"
        _atomic_json(manifest_path, manifest)
        raise

    return summary


def migrate_dataset(
    dataset_root: Path, *, source_version: str, dry_run: bool = False
) -> dict[str, int]:
    """Apply the pre-v0.1.32 -> current URDF frame migration in place."""
    from ..recording.cartesian_frame import (
        CARTESIAN_FRAME_ID,
        write_cartesian_frame_marker,
    )

    source_version = _validate_source_version(source_version)

    def check_eligible(_info: dict[str, Any], marker: dict[str, Any] | None) -> None:
        if marker is None:
            return
        if marker.get("cartesian_pose_frame") == CARTESIAN_FRAME_ID:
            raise ValueError("Dataset is already in the v0.1.32+ Cartesian pose frame.")
        raise ValueError(
            "Dataset has an unknown Cartesian pose-frame marker at "
            f"{dataset_root / 'meta' / 'axol.json'}; refusing to guess."
        )

    def write_marker(root: Path) -> None:
        write_cartesian_frame_marker(
            root,
            migration={
                "id": _MIGRATION_ID,
                "source_axol_version": source_version,
                "target": "axol >= v0.1.32",
            },
        )

    return _run_migration(
        dataset_root,
        _Plan(
            migration_id=_MIGRATION_ID,
            check_eligible=check_eligible,
            transform=_transform_matrix,
            write_marker=write_marker,
        ),
        dry_run=dry_run,
    )


def migrate_mantis_tcp_rotation(
    dataset_root: Path,
    *,
    swap_sides: bool = False,
    dry_run: bool = False,
    rest_midpoint: Any | None = None,
) -> dict[str, int]:
    """Repair a Mantis dataset recorded with the retired Rx(+90) Vive rotation.

    Eligible datasets are Mantis-recorded (``robot_type`` ``axol_mantis``),
    already in the v0.1.32+ pose frame, and carry no
    ``mantis_tcp_transform`` marker field — axol <= 0.2.4 wrote none, and
    every such session used the factory Vive constants unless the operator
    had a per-unit override (which this command cannot know about; do not
    run it on those). ``rest_midpoint`` overrides the FK-derived pivot for
    ``swap_sides`` (tests).
    """
    from ..mantis.calibration import DESIGN_TCP_TRANSFORM_ID
    from ..recording.cartesian_frame import (
        CARTESIAN_FRAME_ID,
        MANTIS_TCP_TRANSFORM_KEY,
        update_cartesian_frame_marker,
    )

    def check_eligible(info: dict[str, Any], marker: dict[str, Any] | None) -> None:
        robot_type = info.get("robot_type")
        if robot_type != "axol_mantis":
            raise ValueError(
                f"Dataset robot_type is {robot_type!r}, not 'axol_mantis'. Only "
                "Mantis-recorded poses went through the tracker→gripper "
                "transform; robot-recorded datasets do not need this migration."
            )
        if marker is None:
            raise ValueError(
                "Dataset has no meta/axol.json pose-frame marker, so it predates "
                "Mantis collection; refusing to guess."
            )
        if marker.get("cartesian_pose_frame") != CARTESIAN_FRAME_ID:
            raise ValueError(
                "Dataset is not in the v0.1.32+ Cartesian pose frame; run the "
                "--from-version migration first."
            )
        recorded = marker.get(MANTIS_TCP_TRANSFORM_KEY)
        if recorded is not None:
            recorded_id = recorded.get("id") if isinstance(recorded, dict) else None
            if recorded_id == DESIGN_TCP_TRANSFORM_ID:
                raise ValueError(
                    "Dataset was recorded (or already migrated) with the corrected "
                    "Ry(180°) Vive tracker→gripper rotation; nothing to fix."
                )
            raise ValueError(
                f"Dataset records tracker→gripper transforms {recorded_id!r}, not "
                "the retired factory constants; this migration only applies to "
                "datasets recorded with the factory Vive transform."
            )

    pivot = rest_midpoint
    if swap_sides and pivot is None:
        pivot = _rest_gripper_midpoint()

    def transform(matrix: Any, arms: list[_ArmLayout]) -> Any:
        return _transform_mantis_tcp(
            matrix, arms, swap_sides=swap_sides, rest_midpoint=pivot
        )

    def write_marker(root: Path) -> None:
        # The dataset now matches what a current session would have recorded
        # with the factory constants — stamp that provenance so collect-data
        # can append to it and this migration refuses to run twice. The
        # tracker family (and so the exact translation) is not recorded by
        # old datasets, hence no per-side values.
        update_cartesian_frame_marker(
            root,
            migration={
                "id": _MANTIS_TCP_MIGRATION_ID,
                "swap_sides": swap_sides,
                "from": "vive Rx(+90°) tracker→gripper rotation (axol <= 0.2.4)",
                "to": DESIGN_TCP_TRANSFORM_ID,
            },
            mantis_tcp_transform={
                "id": DESIGN_TCP_TRANSFORM_ID,
                "source": None,
                "left": None,
                "right": None,
                "migrated": True,
            },
        )

    return _run_migration(
        dataset_root,
        _Plan(
            migration_id=_MANTIS_TCP_MIGRATION_ID,
            check_eligible=check_eligible,
            transform=transform,
            write_marker=write_marker,
        ),
        dry_run=dry_run,
    )


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "migrate-dataset",
        help="Repair Cartesian datasets in place (URDF frame, Mantis TCP rotation).",
        description=(
            "In-place repairs for Cartesian LeRobot datasets. --from-version rotates "
            "actions and observations recorded by axol <= v0.1.31 into the "
            "forward-facing URDF frame introduced in v0.1.32. --mantis-tcp-rotation "
            "replaces the retired Rx(+90°) Vive tracker→gripper rotation in Mantis "
            "datasets recorded by axol <= 0.2.4 (add --swap-sides when the left/right "
            "trackers were bound to the opposite rigs). Each migration creates a "
            "recovery backup under meta/migrations first."
        ),
    )
    parser.add_argument(
        "--repo_id",
        "--repo-id",
        required=True,
        help="Local repo id under --root, or a path to a LeRobot dataset.",
    )
    parser.add_argument("--root", help="Dataset root (default: $HF_LEROBOT_HOME).")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--from-version",
        help=(
            "URDF-frame migration: axol version that recorded the data (must be "
            "older than v0.1.32)."
        ),
    )
    which.add_argument(
        "--mantis-tcp-rotation",
        action="store_true",
        help=(
            "Mantis migration: replace the retired Rx(+90°) Vive tracker→gripper "
            "rotation (axol <= 0.2.4) with the corrected Ry(180°)."
        ),
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help=(
            "With --mantis-tcp-rotation: the LEFT tracker was on the rig held in "
            "the RIGHT hand (and vice versa). Swaps the left/right pose columns and "
            "yaws every pose 180° about the rest-pose gripper midpoint."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would change without writing files.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.swap_sides and not args.mantis_tcp_rotation:
        raise SystemExit(
            "migrate-dataset: --swap-sides requires --mantis-tcp-rotation."
        )
    try:
        dataset_root = _resolve_dataset(args.repo_id, args.root)
        if args.mantis_tcp_rotation:
            migration_id = _MANTIS_TCP_MIGRATION_ID
            summary = migrate_mantis_tcp_rotation(
                dataset_root,
                swap_sides=args.swap_sides,
                dry_run=args.dry_run,
            )
        else:
            migration_id = _MIGRATION_ID
            summary = migrate_dataset(
                dataset_root,
                source_version=args.from_version,
                dry_run=args.dry_run,
            )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"migrate-dataset: {exc}") from None
    verb = "Would migrate" if args.dry_run else "Migrated"
    print(
        f"{verb} {summary['rows']} rows in {summary['data_files']} data file(s) "
        f"and refreshed {summary['episode_files']} episode metadata file(s)."
    )
    if not args.dry_run:
        backup = dataset_root / "meta" / "migrations" / migration_id
        print(f"Original parquet/stat files are recoverable from {backup}.")
