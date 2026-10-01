"""migrate-dataset: the Mantis Vive tracker→gripper rotation repair."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from almond_axol.cli.migrate_dataset import (
    _MANTIS_TCP_MIGRATION_ID,
    _pose_indices,
    _transform_mantis_tcp,
    migrate_dataset,
    migrate_mantis_tcp_rotation,
)
from almond_axol.mantis.calibration import (
    DESIGN_TCP_TRANSFORM_ID,
    LEGACY_VIVE_TCP_ROTATION_QUAT,
    VIVE_TCP_ROTATION_QUAT,
)
from almond_axol.recording.cartesian_frame import (
    CARTESIAN_FRAME_ID,
    MANTIS_TCP_TRANSFORM_KEY,
    URDF_ROOT_YAW_RADIANS,
    read_cartesian_frame_marker,
)

_NAMES = [
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.rx",
    "left_ee.ry",
    "left_ee.rz",
    "left_gripper.pos",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.rx",
    "right_ee.ry",
    "right_ee.rz",
    "right_gripper.pos",
]
_REST_MID = np.array([-0.007, 0.0, 0.156])
_REST_ROTVEC = Rotation.from_matrix(
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
).as_rotvec()


def _legacy_rows(n: int, *, swapped: bool) -> np.ndarray:
    """Rows a legacy session would have recorded for a correctly held rig.

    The physical grippers sit at the rest poses with the rest orientation.
    A legacy session stored ``R_rest @ Ry180^-1 @ Rx90`` for the orientation;
    a swapped one additionally saw the base yawed 180° about the rest
    midpoint and wrote the physical right rig into the left columns.
    """
    legacy = Rotation.from_quat(np.asarray(LEGACY_VIVE_TCP_ROTATION_QUAT))
    current = Rotation.from_quat(np.asarray(VIVE_TCP_ROTATION_QUAT))
    rest = Rotation.from_rotvec(_REST_ROTVEC)
    tracker = rest * current.inv()  # physical tracker orientation in base
    physical = {
        "left": np.array([-0.007, 0.2, 0.156]),
        "right": np.array([-0.007, -0.2, 0.156]),
    }
    rows = np.zeros((n, 14), np.float32)
    for i in range(n):
        drift = np.array([0.01 * i, 0.0, 0.02 * i])
        for col, side in ((0, "left"), (7, "right")):
            pos = physical[side] + drift
            rot = tracker * legacy
            if swapped:
                yaw = Rotation.from_rotvec([0.0, 0.0, math.pi])
                pos = yaw.apply(pos - _REST_MID) + _REST_MID
                rot = yaw * rot
                col = 7 - col
            rows[i, col : col + 3] = pos
            rows[i, col + 3 : col + 6] = rot.as_rotvec()
        rows[i, 6] = 0.25
        rows[i, 13] = 0.75
    return rows


def _expected_rows(n: int) -> np.ndarray:
    rows = np.zeros((n, 14), np.float32)
    for i in range(n):
        drift = np.array([0.01 * i, 0.0, 0.02 * i])
        rows[i, 0:3] = np.array([-0.007, 0.2, 0.156]) + drift
        rows[i, 7:10] = np.array([-0.007, -0.2, 0.156]) + drift
        rows[i, 3:6] = _REST_ROTVEC
        rows[i, 10:13] = _REST_ROTVEC
        rows[i, 6] = 0.25
        rows[i, 13] = 0.75
    return rows


def _write_dataset(
    root: Path, rows: np.ndarray, *, robot_type: str, marker: dict | None
):
    feature = {"dtype": "float32", "shape": [14], "names": _NAMES}
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "fps": 60,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {"action": feature, "observation.state": feature},
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    if marker is not None:
        (root / "meta" / "axol.json").write_text(json.dumps(marker))

    matrix_type = pa.list_(pa.float32(), 14)
    n = len(rows)
    data = pa.table(
        {
            "action": pa.array(rows.tolist(), type=matrix_type),
            "observation.state": pa.array(rows.tolist(), type=matrix_type),
            "episode_index": pa.array([0] * n, type=pa.int64()),
        }
    )
    pq.write_table(data, root / "data" / "chunk-000" / "file-000.parquet")

    stats_cols: dict[str, object] = {
        "episode_index": pa.array([0], type=pa.int64()),
        "data/chunk_index": pa.array([0], type=pa.int64()),
        "data/file_index": pa.array([0], type=pa.int64()),
        "length": pa.array([n], type=pa.int64()),
    }
    stale = [0.0] * 14
    for field in ("action", "observation.state"):
        for statistic in (
            "min",
            "max",
            "mean",
            "std",
            "q01",
            "q10",
            "q50",
            "q90",
            "q99",
        ):
            stats_cols[f"stats/{field}/{statistic}"] = pa.array(
                [stale], type=pa.list_(pa.float64())
            )
        stats_cols[f"stats/{field}/count"] = pa.array([[n]], type=pa.list_(pa.int64()))
    pq.write_table(
        pa.table(stats_cols),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    (root / "meta" / "stats.json").write_text(json.dumps({"action": {"mean": stale}}))


def _read_rows(root: Path) -> np.ndarray:
    table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def _legacy_marker() -> dict:
    return {
        "schema_version": 1,
        "cartesian_pose_frame": CARTESIAN_FRAME_ID,
        "recorded_by_axol_version": "0.2.4",
    }


class TransformMathTest(unittest.TestCase):
    def setUp(self) -> None:
        info = {
            "features": {
                "action": {"names": _NAMES},
                "observation.state": {"names": _NAMES},
            }
        }
        self.arms = _pose_indices(info)["action"]

    def test_rotation_fix_restores_rest_orientation_and_keeps_positions(self) -> None:
        legacy = _legacy_rows(5, swapped=False)
        fixed = _transform_mantis_tcp(
            legacy, self.arms, swap_sides=False, rest_midpoint=_REST_MID
        )
        np.testing.assert_allclose(fixed, _expected_rows(5), atol=1e-5)
        np.testing.assert_array_equal(
            fixed[:, [0, 1, 2, 7, 8, 9]], legacy[:, [0, 1, 2, 7, 8, 9]]
        )

    def test_swap_sides_unswaps_columns_and_yaws_about_rest_midpoint(self) -> None:
        legacy = _legacy_rows(5, swapped=True)
        fixed = _transform_mantis_tcp(
            legacy, self.arms, swap_sides=True, rest_midpoint=_REST_MID
        )
        np.testing.assert_allclose(fixed, _expected_rows(5), atol=1e-5)
        # Gripper columns follow the CAN channel, not the tracker: untouched.
        np.testing.assert_array_equal(fixed[:, 6], legacy[:, 6])
        np.testing.assert_array_equal(fixed[:, 13], legacy[:, 13])


class MantisMigrationTest(unittest.TestCase):
    def _root(self) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_migrates_swapped_legacy_dataset_in_place(self) -> None:
        root = self._root()
        _write_dataset(
            root,
            _legacy_rows(6, swapped=True),
            robot_type="axol_mantis",
            marker=_legacy_marker(),
        )
        preview = migrate_mantis_tcp_rotation(
            root, swap_sides=True, dry_run=True, rest_midpoint=_REST_MID
        )
        self.assertEqual(preview["rows"], 6)
        self.assertFalse((root / "meta" / "migrations").exists())

        summary = migrate_mantis_tcp_rotation(
            root, swap_sides=True, rest_midpoint=_REST_MID
        )
        self.assertEqual(summary, {"data_files": 1, "episode_files": 1, "rows": 6})
        np.testing.assert_allclose(_read_rows(root), _expected_rows(6), atol=1e-5)

        marker = read_cartesian_frame_marker(root)
        assert marker is not None
        self.assertEqual(marker["recorded_by_axol_version"], "0.2.4")
        self.assertEqual(
            marker[MANTIS_TCP_TRANSFORM_KEY]["id"], DESIGN_TCP_TRANSFORM_ID
        )
        self.assertTrue(marker[MANTIS_TCP_TRANSFORM_KEY]["migrated"])
        self.assertEqual(marker["migrations"][0]["id"], _MANTIS_TCP_MIGRATION_ID)
        self.assertTrue(marker["migrations"][0]["swap_sides"])

        stats = json.loads((root / "meta" / "stats.json").read_text())
        np.testing.assert_allclose(
            stats["action"]["mean"], _expected_rows(6).mean(axis=0), atol=1e-4
        )
        episodes = pq.read_table(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).to_pylist()[0]
        np.testing.assert_allclose(
            episodes["stats/action/min"], _expected_rows(6).min(axis=0), atol=1e-5
        )

        backup = root / "meta" / "migrations" / _MANTIS_TCP_MIGRATION_ID
        manifest = json.loads((backup / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "complete")
        self.assertIn("meta/axol.json", manifest["files"])

        with self.assertRaisesRegex(ValueError, "already migrated"):
            migrate_mantis_tcp_rotation(root, rest_midpoint=_REST_MID)

    def test_refuses_ineligible_datasets(self) -> None:
        legacy = _legacy_rows(3, swapped=False)

        root = self._root()
        _write_dataset(root, legacy, robot_type="axol", marker=_legacy_marker())
        with self.assertRaisesRegex(ValueError, "axol_mantis"):
            migrate_mantis_tcp_rotation(root, rest_midpoint=_REST_MID)

        root = self._root()
        _write_dataset(root, legacy, robot_type="axol_mantis", marker=None)
        with self.assertRaisesRegex(ValueError, "no meta/axol.json"):
            migrate_mantis_tcp_rotation(root, rest_midpoint=_REST_MID)

        root = self._root()
        _write_dataset(
            root,
            legacy,
            robot_type="axol_mantis",
            marker={
                **_legacy_marker(),
                MANTIS_TCP_TRANSFORM_KEY: {"id": DESIGN_TCP_TRANSFORM_ID},
            },
        )
        with self.assertRaisesRegex(ValueError, "nothing to fix"):
            migrate_mantis_tcp_rotation(root, rest_midpoint=_REST_MID)

        root = self._root()
        _write_dataset(
            root,
            legacy,
            robot_type="axol_mantis",
            marker={**_legacy_marker(), MANTIS_TCP_TRANSFORM_KEY: {"id": "measured"}},
        )
        with self.assertRaisesRegex(ValueError, "factory Vive transform"):
            migrate_mantis_tcp_rotation(root, rest_midpoint=_REST_MID)
        # Nothing was touched by any refusal.
        np.testing.assert_array_equal(_read_rows(root), legacy)
        self.assertFalse((root / "meta" / "migrations").exists())

    def test_urdf_migration_still_yaws_unmarked_datasets(self) -> None:
        """The original --from-version path is unchanged by the shared runner."""
        legacy = _legacy_rows(4, swapped=False)
        root = self._root()
        _write_dataset(root, legacy, robot_type="axol", marker=None)

        summary = migrate_dataset(root, source_version="v0.1.29")
        self.assertEqual(summary, {"data_files": 1, "episode_files": 1, "rows": 4})

        yaw = Rotation.from_rotvec([0.0, 0.0, URDF_ROOT_YAW_RADIANS])
        expected = legacy.copy()
        for pos, rot in ((slice(0, 3), slice(3, 6)), (slice(7, 10), slice(10, 13))):
            expected[:, pos] = yaw.apply(legacy[:, pos])
            expected[:, rot] = (yaw * Rotation.from_rotvec(legacy[:, rot])).as_rotvec()
        np.testing.assert_allclose(_read_rows(root), expected, atol=1e-5)

        marker = read_cartesian_frame_marker(root)
        assert marker is not None
        self.assertEqual(marker["cartesian_pose_frame"], CARTESIAN_FRAME_ID)
        self.assertEqual(marker["migration"]["id"], "axol-urdf-root-yaw-v0.1.32")
        self.assertEqual(marker["migration"]["source_axol_version"], "0.1.29")
        stats = json.loads((root / "meta" / "stats.json").read_text())
        np.testing.assert_allclose(
            stats["action"]["mean"], expected.mean(axis=0), atol=1e-4
        )
        manifest = json.loads(
            (
                root
                / "meta"
                / "migrations"
                / "axol-urdf-root-yaw-v0.1.32"
                / "manifest.json"
            ).read_text()
        )
        self.assertEqual(manifest["state"], "complete")
        with self.assertRaisesRegex(ValueError, "already migrated"):
            migrate_dataset(root, source_version="v0.1.29")

    def test_urdf_migration_refuses_datasets_already_in_frame(self) -> None:
        root = self._root()
        _write_dataset(
            root,
            _legacy_rows(3, swapped=False),
            robot_type="axol_mantis",
            marker=_legacy_marker(),
        )
        with self.assertRaisesRegex(ValueError, "already in the v0.1.32"):
            migrate_dataset(root, source_version="v0.1.29")


if __name__ == "__main__":
    unittest.main()
