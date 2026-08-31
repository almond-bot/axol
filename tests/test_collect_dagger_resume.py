from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from almond_axol.cli import collect_dagger


class DaggerResumeSchemaTest(unittest.TestCase):
    @staticmethod
    def _config(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            task="pick",
            subtasks=None,
            episode_time_s=10,
            fps=60,
            teleop_hz=120,
            vcodec="h264",
            repo_id="local/dagger",
            root=str(root),
            rerun_ip=None,
            rerun_port=9876,
            robot_config=object(),
            teleop_config=object(),
            dataset_resolution="SVGA",
        )

    def test_malformed_intervention_shape_fails_with_schema_error(self) -> None:
        for malformed in (1, "1", {"length": 1}, [[1]]):
            with (
                self.subTest(shape=malformed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                meta = root / "meta"
                meta.mkdir()
                (meta / "info.json").write_text(
                    json.dumps(
                        {
                            "features": {
                                "intervention": {
                                    "dtype": "bool",
                                    "shape": malformed,
                                }
                            }
                        }
                    )
                )

                with self.assertRaisesRegex(ValueError, "required per-frame bool"):
                    collect_dagger._require_dagger_resume_schema(root)  # noqa: SLF001

    def test_missing_intervention_fails_before_hardware_or_workers_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(json.dumps({"features": {}}))
            (meta / "tasks.parquet").touch()
            (meta / "episodes").mkdir()
            cfg = self._config(root)

            with (
                mock.patch.object(collect_dagger, "IKResetController") as reset,
                mock.patch.object(collect_dagger, "_start_video_relay") as relay,
                mock.patch.object(collect_dagger, "DatasetRecorderProcess") as recorder,
                mock.patch("almond_axol.lerobot.robot.robot_axol.AxolRobot") as robot,
                self.assertRaisesRegex(
                    ValueError,
                    "silently leave new human-correction frames unlabeled",
                ),
            ):
                collect_dagger._run(  # noqa: SLF001
                    cfg,
                    stop_event=threading.Event(),
                    control=object(),
                )

            reset.assert_not_called()
            relay.assert_not_called()
            recorder.assert_not_called()
            robot.assert_not_called()

    def test_nonempty_unrecognized_root_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "customer-data"
            root.mkdir()
            sentinel = root / "do-not-delete.txt"
            sentinel.write_text("customer data")
            cfg = self._config(root)

            with (
                mock.patch.object(collect_dagger, "IKResetController") as reset,
                mock.patch.object(collect_dagger, "_start_video_relay") as relay,
                mock.patch.object(collect_dagger, "DatasetRecorderProcess") as recorder,
                mock.patch("almond_axol.lerobot.robot.robot_axol.AxolRobot") as robot,
                self.assertRaisesRegex(RuntimeError, "not an empty directory"),
            ):
                collect_dagger._run(  # noqa: SLF001
                    cfg,
                    stop_event=threading.Event(),
                    control=object(),
                )

            self.assertEqual(sentinel.read_text(), "customer data")
            reset.assert_not_called()
            relay.assert_not_called()
            recorder.assert_not_called()
            robot.assert_not_called()

    def test_gripper_capability_is_bound_before_teleop_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new-dataset"
            cfg = collect_dagger.DaggerConfig(
                policy_path="test-policy",
                policy_type="act",
                task="pick",
                repo_id="local/dagger",
                root=str(root),
            )
            cfg.robot_config.cameras["overhead"].serial = 1234
            cfg.robot_config.axol_config.has_gripper = False
            cfg.teleop_config.has_gripper = True
            seen: list[bool] = []

            def construct_teleop(config: object) -> object:
                seen.append(bool(getattr(config, "has_gripper")))
                raise RuntimeError("teleop construction sentinel")

            with (
                mock.patch("almond_axol.zed.stereo_serials", return_value=set()),
                mock.patch("almond_axol.lerobot.robot.robot_axol.AxolRobot"),
                mock.patch(
                    "almond_axol.lerobot.teleop.teleop_vr_dagger.DaggerVRTeleop",
                    side_effect=construct_teleop,
                ),
                self.assertRaisesRegex(RuntimeError, "construction sentinel"),
            ):
                collect_dagger._run(cfg)  # noqa: SLF001

            self.assertEqual(seen, [False])


if __name__ == "__main__":
    unittest.main()
