"""Migrate pre-v0.1.32 Cartesian datasets to the forward-facing URDF frame.

Axol v0.1.32 added a +90 degree yaw to the URDF root.  Cartesian datasets
recorded by earlier versions therefore contain end-effector poses in the old
world frame.  Replaying those poses with the new URDF rotates the motion by
roughly 90 degrees.

This command applies the same rigid transform as the URDF change to every
Cartesian action and observation, then refreshes per-episode and dataset
statistics.  Videos and all non-Cartesian values are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


_MIGRATION_ID = "axol-urdf-root-yaw-v0.1.32"
_FIELDS = ("action", "observation.state")
_ARMS = ("left", "right")


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


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
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression=compression)
    shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _backup_files(
    dataset_root: Path, files: list[Path], backup_root: Path
) -> dict[str, Any]:
    marker = dataset_root / "meta" / "axol.json"
    manifest: dict[str, Any] = {
        "migration": _MIGRATION_ID,
        "state": "backed_up",
        "files": [str(path.relative_to(dataset_root)) for path in files],
        "marker_existed": marker.exists(),
    }
    files_root = backup_root / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    for source in files:
        destination = files_root / source.relative_to(dataset_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _atomic_json(backup_root / "manifest.json", manifest)
    return manifest


def _restore_backup(
    dataset_root: Path, backup_root: Path, manifest: dict[str, Any]
) -> None:
    files_root = backup_root / "files"
    for relative in manifest["files"]:
        source = files_root / relative
        destination = dataset_root / relative
        tmp = destination.with_name(f".{destination.name}.restore.tmp")
        shutil.copy2(source, tmp)
        os.replace(tmp, destination)
    if not manifest.get("marker_existed", False):
        (dataset_root / "meta" / "axol.json").unlink(missing_ok=True)


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


def migrate_dataset(
    dataset_root: Path, *, source_version: str, dry_run: bool = False
) -> dict[str, int]:
    """Apply the pre-v0.1.32 -> current URDF frame migration in place."""
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import serialize_dict

    from ..recording.cartesian_frame import (
        CARTESIAN_FRAME_ID,
        write_cartesian_frame_marker,
    )

    source_version = _validate_source_version(source_version)
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

    backup_root = dataset_root / "meta" / "migrations" / _MIGRATION_ID
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

    marker_path = dataset_root / "meta" / "axol.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("cartesian_pose_frame") == CARTESIAN_FRAME_ID:
            raise ValueError("Dataset is already in the v0.1.32+ Cartesian pose frame.")
        raise ValueError(
            f"Dataset has an unknown Cartesian pose-frame marker at {marker_path}; "
            "refusing to guess."
        )

    row_count = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
    summary = {
        "data_files": len(data_files),
        "episode_files": len(episode_files),
        "rows": row_count,
    }
    if dry_run:
        return summary

    files_to_backup = [*data_files, *episode_files, stats_path]
    if not manifest_path.exists():
        manifest = _backup_files(dataset_root, files_to_backup, backup_root)
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
                    table, field, _transform_matrix(_table_matrix(table, field), arms)
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

        write_cartesian_frame_marker(
            dataset_root,
            migration={
                "id": _MIGRATION_ID,
                "source_axol_version": source_version,
                "target": "axol >= v0.1.32",
            },
        )
        manifest["state"] = "complete"
        _atomic_json(manifest_path, manifest)
    except BaseException:
        _restore_backup(dataset_root, backup_root, manifest)
        manifest["state"] = "failed_restored"
        _atomic_json(manifest_path, manifest)
        raise

    return summary


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "migrate-dataset",
        help="Migrate pre-v0.1.32 Cartesian data to the current URDF frame.",
        description=(
            "Rotate Cartesian actions and observations recorded by axol <= v0.1.31 "
            "into the forward-facing URDF frame introduced in v0.1.32. The migration "
            "is in-place and creates a recovery backup under meta/migrations first."
        ),
    )
    parser.add_argument(
        "--repo_id",
        "--repo-id",
        required=True,
        help="Local repo id under --root, or a path to a LeRobot dataset.",
    )
    parser.add_argument("--root", help="Dataset root (default: $HF_LEROBOT_HOME).")
    parser.add_argument(
        "--from-version",
        required=True,
        help="Axol version that recorded the data (must be older than v0.1.32).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would change without writing files.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    try:
        dataset_root = _resolve_dataset(args.repo_id, args.root)
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
        backup = dataset_root / "meta" / "migrations" / _MIGRATION_ID
        print(f"Original parquet/stat files are recoverable from {backup}.")
