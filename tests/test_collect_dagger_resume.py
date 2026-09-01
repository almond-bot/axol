from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from almond_axol.cli import collect_dagger
from almond_axol.robot.base import (
    HardwareCleanupError,
    is_hardware_cleanup_uncertain,
)


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

    def test_session_error_is_preserved_and_marked_when_ik_cleanup_fails(
        self,
    ) -> None:
        primary = ValueError("episode failed")
        reset_failure = RuntimeError("IK child still alive")
        relay_failure = RuntimeError("relay close failed")

        collect_dagger._finish_dagger_cleanup(  # noqa: SLF001
            session_error=primary,
            disconnect_failure=None,
            teleop_failure=None,
            reset_failure=reset_failure,
            relay_failure=relay_failure,
        )

        self.assertTrue(is_hardware_cleanup_uncertain(primary))
        self.assertTrue(any("IK reset worker" in note for note in primary.__notes__))
        self.assertTrue(any("video relay" in note for note in primary.__notes__))

    def test_ik_cleanup_failure_is_hardware_cleanup_error_on_clean_exit(
        self,
    ) -> None:
        reset_failure = RuntimeError("IK child still alive")

        with self.assertRaisesRegex(
            HardwareCleanupError, "background ownership is uncertain"
        ) as raised:
            collect_dagger._finish_dagger_cleanup(  # noqa: SLF001
                session_error=None,
                disconnect_failure=None,
                teleop_failure=None,
                reset_failure=reset_failure,
                relay_failure=None,
            )

        self.assertIs(raised.exception.__cause__, reset_failure)

    def test_ik_uncertainty_is_not_hidden_by_teleop_cleanup_failure(self) -> None:
        teleop_failure = RuntimeError("VR loop close failed")
        reset_failure = RuntimeError("IK child still alive")

        with self.assertRaises(HardwareCleanupError) as raised:
            collect_dagger._finish_dagger_cleanup(  # noqa: SLF001
                session_error=None,
                disconnect_failure=None,
                teleop_failure=teleop_failure,
                reset_failure=reset_failure,
                relay_failure=None,
            )

        self.assertIs(raised.exception.__cause__, reset_failure)
        self.assertTrue(
            any("teleop disconnect" in note for note in raised.exception.__notes__)
        )

    def test_live_control_worker_blocks_all_owned_resource_cleanup(self) -> None:
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.shutdown_event = threading.Event()  # type: ignore[attr-defined]
        worker.start()
        owned_cleanups = [mock.Mock() for _ in range(4)]
        try:
            stopped, error = collect_dagger._stop_dagger_control_worker(  # noqa: SLF001
                worker,  # type: ignore[arg-type]
                timeout=0.01,
            )

            self.assertFalse(stopped)
            self.assertIsInstance(error, RuntimeError)
            for index, cleanup in enumerate(owned_cleanups):
                self.assertIsNone(
                    collect_dagger._cleanup_dagger_resource(  # noqa: SLF001
                        control_stopped=stopped,
                        label=f"owned-{index}",
                        cleanup=cleanup,
                    )
                )
                cleanup.assert_not_called()

            # The exact worker remains available for a final retry. Only after
            # that retry proves exit may its resources be touched.
            release.set()
            stopped, retry_error = collect_dagger._stop_dagger_control_worker(  # noqa: SLF001
                worker,  # type: ignore[arg-type]
                timeout=1.0,
            )
            self.assertTrue(stopped)
            self.assertIsNone(retry_error)
            collect_dagger._cleanup_dagger_resource(  # noqa: SLF001
                control_stopped=stopped,
                label="owned-0",
                cleanup=owned_cleanups[0],
            )
            owned_cleanups[0].assert_called_once_with()
        finally:
            release.set()
            worker.join(timeout=1.0)

    def test_live_control_worker_error_propagates_hardware_uncertainty(self) -> None:
        ownership_failure = HardwareCleanupError(
            "DAgger control loop still owns hardware"
        )

        with self.assertRaises(HardwareCleanupError) as raised:
            collect_dagger._finish_dagger_cleanup(  # noqa: SLF001
                session_error=None,
                disconnect_failure=None,
                teleop_failure=None,
                reset_failure=None,
                relay_failure=None,
                additional_failures=(("DAgger control loop", ownership_failure),),
            )

        self.assertIs(raised.exception, ownership_failure)

    def test_policy_result_returning_after_stop_never_reaches_hardware(self) -> None:
        control_loop = object.__new__(collect_dagger._DaggerControlLoop)  # noqa: SLF001
        control_loop.shutdown_event = threading.Event()
        control_loop.robot = SimpleNamespace(send_action=mock.Mock())
        control_loop.policy = SimpleNamespace(
            act=mock.Mock(
                side_effect=lambda _obs: (
                    control_loop.shutdown_event.set() or {"left.pos": 1.0}
                )
            )
        )
        control_loop.limiter = None
        control_loop.recorder = SimpleNamespace(publish=mock.Mock())

        with mock.patch.object(
            collect_dagger, "latest_observation", return_value={"joint": 0.0}
        ):
            result = control_loop._policy_tick(1.0)  # noqa: SLF001

        self.assertIsNone(result)
        control_loop.robot.send_action.assert_not_called()
        control_loop.recorder.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
