from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from almond_axol.recording.datasets import (
    RESUME_FILLABLE_FEATURES,
    dataset_features_for_robot,
    require_dataset_resume_schema,
)


EXPECTED_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["left_joint", "right_joint"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": ["left_joint", "right_joint"],
    },
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    },
}

LEROBOT_DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


class DatasetResumeSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "dataset"
        (self.root / "meta").mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_info(
        self,
        *,
        features: dict[str, object] | None = None,
        fps: object = 30,
    ) -> None:
        info = {
            "fps": fps,
            "features": features
            if features is not None
            else {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)},
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info))

    def _require(
        self,
        expected: dict[str, dict] | None = None,
        *,
        fps: int = 30,
        allowed: frozenset[str] = frozenset(),
    ) -> None:
        require_dataset_resume_schema(
            self.root,
            expected or EXPECTED_FEATURES,
            fps=fps,
            allowed_extra_features=allowed,
        )

    def test_exact_schema_with_lerobot_defaults_passes(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        actual_root = Path(self._tmp.name) / "lerobot-created"
        LeRobotDatasetMetadata.create(
            repo_id="test/resume-schema",
            fps=30,
            features=deepcopy(EXPECTED_FEATURES),
            root=actual_root,
        )

        require_dataset_resume_schema(actual_root, EXPECTED_FEATURES, fps=30)

    def test_action_same_width_with_different_names_fails(self) -> None:
        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features["action"]["names"] = ["left_ee.x", "right_ee.x"]  # type: ignore[index]
        self._write_info(features=features)

        with self.assertRaisesRegex(ValueError, "mismatched action"):
            self._require()

    def test_camera_key_and_shape_must_match(self) -> None:
        for change, expected_message in (
            ("key", "missing observation.images.front"),
            ("shape", "mismatched observation.images.front"),
        ):
            with self.subTest(change=change):
                features = {
                    **deepcopy(EXPECTED_FEATURES),
                    **deepcopy(LEROBOT_DEFAULT_FEATURES),
                }
                if change == "key":
                    features["observation.images.wrist"] = features.pop(
                        "observation.images.front"
                    )
                else:
                    features["observation.images.front"]["shape"] = (240, 320, 3)  # type: ignore[index]
                self._write_info(features=features)
                with self.assertRaisesRegex(ValueError, expected_message):
                    self._require()

    def test_fps_must_match_and_be_finite_integer_metadata(self) -> None:
        for stored_fps in (60, True, float("nan"), float("inf")):
            with self.subTest(stored_fps=stored_fps):
                self._write_info(fps=stored_fps)
                with self.assertRaisesRegex(ValueError, "configured for 30 fps"):
                    self._require()

    def test_malformed_metadata_fails_closed(self) -> None:
        info_path = self.root / "meta" / "info.json"
        for raw, expected_message in (
            ("not json", "could not be read"),
            (json.dumps([]), "no valid feature schema"),
            (json.dumps({"fps": 30, "features": []}), "no valid feature schema"),
        ):
            with self.subTest(raw=raw):
                info_path.write_text(raw)
                with self.assertRaisesRegex(ValueError, expected_message):
                    self._require()

        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features["action"] = {"dtype": "float32", "shape": [0], "names": []}
        self._write_info(features=features)
        with self.assertRaisesRegex(
            ValueError, "feature 'action' has an invalid shape"
        ):
            self._require()

    def test_missing_or_malformed_lerobot_defaults_fail(self) -> None:
        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        del features["task_index"]
        self._write_info(features=features)
        with self.assertRaisesRegex(ValueError, "missing LeRobot default task_index"):
            self._require()

        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features["timestamp"]["dtype"] = "float64"  # type: ignore[index]
        self._write_info(features=features)
        with self.assertRaisesRegex(ValueError, "mismatched LeRobot default timestamp"):
            self._require()

    def test_supported_pose_lag_and_intervention_extras_pass_exactly(self) -> None:
        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features.update(deepcopy(RESUME_FILLABLE_FEATURES))
        allowed = frozenset(RESUME_FILLABLE_FEATURES)
        self._write_info(features=features)
        self._require(allowed=allowed)

        expected_with_pose_lag = {
            **deepcopy(EXPECTED_FEATURES),
            "observation.pose_lag": deepcopy(
                RESUME_FILLABLE_FEATURES["observation.pose_lag"]
            ),
        }
        self._require(expected_with_pose_lag, allowed=allowed)

        # Axol Cartesian -> Mantis is the reverse direction: the established
        # dataset has no pose-lag column, so Mantis keeps that fixed schema.
        self._write_info()
        self._require(expected_with_pose_lag, allowed=allowed)
        with self.assertRaisesRegex(ValueError, "missing observation.pose_lag"):
            self._require(expected_with_pose_lag)

        malformed_expected_pose_lag = deepcopy(expected_with_pose_lag)
        malformed_expected_pose_lag["observation.pose_lag"]["shape"] = (2,)
        with self.assertRaisesRegex(
            ValueError, "mismatched optional observation.pose_lag"
        ):
            self._require(malformed_expected_pose_lag, allowed=allowed)

        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features.update(deepcopy(RESUME_FILLABLE_FEATURES))
        features["intervention"]["shape"] = (2,)  # type: ignore[index]
        self._write_info(features=features)
        with self.assertRaisesRegex(ValueError, "mismatched optional intervention"):
            self._require(allowed=allowed)

    def test_unsupported_extras_and_unknown_allowlist_fail(self) -> None:
        features = {**deepcopy(EXPECTED_FEATURES), **deepcopy(LEROBOT_DEFAULT_FEATURES)}
        features["observation.unsupported"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }
        self._write_info(features=features)
        with self.assertRaisesRegex(ValueError, "unexpected observation.unsupported"):
            self._require()
        with self.assertRaisesRegex(ValueError, "unknown fillable resume feature"):
            self._require(allowed=frozenset({"observation.unsupported"}))

    def test_dataset_features_for_robot_overrides_image_shape(self) -> None:
        robot = SimpleNamespace(
            action_features={"left_joint": float, "right_joint": float},
            observation_features={
                "left_joint": float,
                "right_joint": float,
                "front": (None, None, 3),
            },
        )

        features = dataset_features_for_robot(robot, image_shape=(240, 320, 3))

        self.assertEqual(features["action"]["shape"], (2,))
        self.assertEqual(features["action"]["names"], ["left_joint", "right_joint"])
        self.assertEqual(features["observation.images.front"]["shape"], (240, 320, 3))
        self.assertEqual(
            features["observation.images.front"]["info"], {"is_depth_map": False}
        )


if __name__ == "__main__":
    unittest.main()
