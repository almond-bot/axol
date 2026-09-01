from __future__ import annotations

import math
import tempfile
import threading
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from almond_axol.cli import run_policy
from almond_axol.cli.run_policy import (
    _align_action_chunk,
    _cleanup_after_episode_workers,
    _clear_episode_buffer_after_workers,
    _lingering_episode_thread_error,
    _shutdown_policy_server_process,
    _snap_to_newest_indices,
    _stop_episode_workers,
)
from almond_axol.lerobot import action_schema as action_schema_module
from almond_axol.lerobot.rollout import IKResetController
from almond_axol.robot.base import HardwareCleanupError


@dataclass
class _TimedAction:
    timestep: int
    action: torch.Tensor

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> torch.Tensor:
        return self.action


class RunPolicySafetyTest(unittest.TestCase):
    @staticmethod
    def _hub_fps_metadata(training_fps: int):
        def fetch(repo_id, filename, *, repo_type="model", revision=None):
            if repo_id == "org/policy" and filename == "train_config.json":
                return {
                    "dataset": {
                        "repo_id": "org/training-data",
                        "repo_type": "dataset",
                        "revision": "training-v2",
                        "root": "/training-host/private/dataset",
                    }
                }
            if repo_id == "org/training-data" and filename == "meta/info.json":
                if repo_type != "dataset" or revision != "training-v2":
                    raise AssertionError("training dataset revision was not preserved")
                return {"fps": training_fps}
            return None

        return fetch

    @staticmethod
    def _hub_run_config(**overrides):
        values = {
            "policy_path": "org/policy",
            "policy_type": "act",
            "task": "pick",
            "episode_time_s": 10,
            "fps": 60,
            "allow_fps_mismatch": False,
            "vcodec": "h264",
            "repo_id": None,
            "root": None,
            "push_to_hub": False,
            "device": "cpu",
            "server_host": "127.0.0.1",
            "server_port": 8765,
            "actions_per_chunk": 2,
            "chunk_size_threshold": 0.9,
            "aggregate_fn": "latest_only",
            "temporal_ensemble_coeff": 0.01,
            "rerun_ip": None,
            "rerun_port": 9876,
            "robot_config": object(),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_hub_training_fps_mismatch_fails_before_robot_construction(self) -> None:
        cfg = self._hub_run_config(fps=60)
        with (
            mock.patch.object(
                action_schema_module,
                "_hub_json",
                side_effect=self._hub_fps_metadata(30),
            ),
            mock.patch("almond_axol.lerobot.robot.robot_axol.AxolRobot") as robot_class,
            self.assertRaisesRegex(ValueError, r"--fps 60.*trained at \(30"),
        ):
            run_policy._run(  # noqa: SLF001
                cfg,
                stop_event=threading.Event(),
                control=object(),
            )

        robot_class.assert_not_called()

    def test_matching_hub_training_fps_is_accepted(self) -> None:
        cfg = self._hub_run_config(fps=60)
        with (
            mock.patch.object(
                action_schema_module,
                "_hub_json",
                side_effect=self._hub_fps_metadata(60),
            ),
            mock.patch.object(
                action_schema_module,
                "_read_json",
                side_effect=AssertionError(
                    "Hub metadata must not choose local filesystem paths"
                ),
            ),
        ):
            run_policy._check_training_fps(cfg)  # noqa: SLF001

    def test_hub_policy_without_training_fps_is_rejected(self) -> None:
        cfg = self._hub_run_config(fps=60)
        with (
            mock.patch.object(action_schema_module, "_hub_json", return_value=None),
            self.assertRaisesRegex(ValueError, "Could not determine.*Hub policy"),
        ):
            run_policy._check_training_fps(cfg)  # noqa: SLF001

    def test_hub_policy_with_unbounded_integer_fps_fails_safely(self) -> None:
        cfg = self._hub_run_config(fps=60)
        with (
            mock.patch.object(
                action_schema_module,
                "_hub_json",
                return_value={"fps": 10**10000},
            ),
            self.assertRaisesRegex(
                action_schema_module.ActionSchemaError,
                "Malformed training fps.*1 through 1000",
            ),
        ):
            run_policy._check_training_fps(cfg)  # noqa: SLF001

    def test_rollout_durability_failure_is_fatal_and_finalizes_without_saved_ack(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory) / "rollout"
            cfg = self._hub_run_config(
                repo_id="local/rollout",
                root=str(dataset_root),
                push_to_hub=True,
            )
            cfg.reset_torque_threshold = 5.0
            cfg.reset_gravity_comp_kd = 0.1
            cfg.ensemble_blend_s = 0.1
            cfg.align_fade_s = 0.1
            cfg.exec_max_vel = 1.0
            cfg.exec_max_accel = 1.0
            cfg.policy_torque_threshold = 5.0

            dataset = mock.Mock()
            robot = mock.Mock(
                name="axol",
                observation_features={},
                config=SimpleNamespace(observe_cartesian=False),
            )
            reset_controller = mock.Mock()
            reset_controller.return_to_rest.return_value = True
            client = mock.Mock(fatal_error=None, contact_tripped=None)
            client.start.return_value = True
            publisher = mock.Mock()
            control = mock.Mock()
            control.await_continue.return_value = True
            control.poll_choice.return_value = "s"
            durability_error = OSError("metadata footer fsync failed")

            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(run_policy, "_check_training_fps")
                )
                stack.enter_context(
                    mock.patch(
                        "almond_axol.utils.state_files."
                        "require_service_dataset_configuration"
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "almond_axol.utils.state_files.privileged_service_active",
                        return_value=False,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "almond_axol.lerobot.robot.robot_axol.AxolRobot",
                        return_value=robot,
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "lerobot.processor.make_default_processors",
                        return_value=(None, mock.Mock(), mock.Mock()),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "almond_axol.recording.datasets.dataset_features_for_robot",
                        return_value={"action": {"names": []}},
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "lerobot.datasets.lerobot_dataset.LeRobotDataset.create",
                        return_value=dataset,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "IKResetController",
                        return_value=reset_controller,
                    )
                )
                stack.enter_context(mock.patch.object(run_policy, "_wait_for_port"))
                stack.enter_context(
                    mock.patch(
                        "lerobot.async_inference.configs.RobotClientConfig",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "ActionPublisher",
                        return_value=publisher,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "_build_axol_robot_client",
                        return_value=client,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "RolloutCaptureThread",
                        return_value=mock.Mock(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy.threading,
                        "Thread",
                        return_value=mock.Mock(),
                    )
                )
                stack.enter_context(mock.patch.object(run_policy.time, "sleep"))
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "_stop_episode_workers",
                        return_value=(True, None),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_policy,
                        "make_episode_durable",
                        side_effect=durability_error,
                    )
                )
                restore = stack.enter_context(
                    mock.patch.object(run_policy, "restore_dataset_ownership")
                )
                log_say = stack.enter_context(mock.patch("lerobot.utils.utils.log_say"))
                stack.enter_context(mock.patch("signal.signal"))
                raised = stack.enter_context(
                    self.assertRaisesRegex(
                        run_policy.EpisodeDurabilityError,
                        "cannot continue safely",
                    )
                )
                run_policy._run(  # noqa: SLF001
                    cfg,
                    stop_event=threading.Event(),
                    control=control,
                )

        self.assertIs(raised.exception.__cause__, durability_error)
        self.assertNotIsInstance(raised.exception, RuntimeError)
        dataset.save_episode.assert_called_once_with()
        dataset.finalize.assert_called_once_with()
        # The write is counted for recovery, so a successful finalization may
        # push it; only the live operator acknowledgement remains suppressed.
        dataset.push_to_hub.assert_called_once_with()
        control.note_saved.assert_not_called()
        self.assertFalse(
            any(
                args and str(args[0]).startswith("Saved episode")
                for args, _ in log_say.call_args_list
            )
        )
        restore.assert_not_called()

    def test_nonempty_unrecognized_rollout_root_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "customer-data"
            root.mkdir()
            sentinel = root / "do-not-delete.txt"
            sentinel.write_text("customer data")
            cfg = SimpleNamespace(
                policy_path="unused",
                policy_type="act",
                task="pick",
                episode_time_s=10,
                fps=60,
                vcodec="h264",
                repo_id="local/rollout",
                root=str(root),
                push_to_hub=False,
                device="cpu",
                server_host="127.0.0.1",
                server_port=8765,
                actions_per_chunk=2,
                chunk_size_threshold=0.9,
                aggregate_fn="latest_only",
                temporal_ensemble_coeff=0.01,
                rerun_ip=None,
                rerun_port=9876,
                robot_config=object(),
            )

            with (
                mock.patch.object(run_policy, "_check_training_fps"),
                mock.patch(
                    "almond_axol.lerobot.robot.robot_axol.AxolRobot"
                ) as robot_class,
                self.assertRaisesRegex(RuntimeError, "not an empty directory"),
            ):
                run_policy._run(  # noqa: SLF001
                    cfg,
                    stop_event=threading.Event(),
                    control=object(),
                )

            self.assertEqual(sentinel.read_text(), "customer data")
            robot_class.return_value.connect.assert_not_called()

    def test_lingering_episode_worker_is_hardware_cleanup_failure(self) -> None:
        stopped = type("Worker", (), {"is_alive": lambda self: False})()
        lingering = type("Worker", (), {"is_alive": lambda self: True})()

        error = _lingering_episode_thread_error(
            [("stopped", stopped), ("control", lingering)]
        )

        self.assertIsInstance(error, HardwareCleanupError)
        self.assertIn("control", str(error))

    def test_live_capture_blocks_buffer_mutation_and_retains_worker(self) -> None:
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=True)
        worker.start()
        capture = SimpleNamespace(
            request_stop=mock.Mock(),
            unblock_inputs=mock.Mock(),
        )
        client = SimpleNamespace(
            shutdown_event=threading.Event(),
            start_barrier=mock.Mock(),
            stop=mock.Mock(),
        )
        workers = [("capture", worker)]
        dataset = mock.Mock()
        try:
            stopped, error = _stop_episode_workers(
                client=client,
                capture=capture,
                workers=workers,
                join_timeout=0.01,
            )

            self.assertFalse(stopped)
            self.assertIsInstance(error, HardwareCleanupError)
            self.assertIs(workers[0][1], worker)
            client.stop.assert_called_once_with()
            capture.unblock_inputs.assert_called_once_with()
            with self.assertRaisesRegex(HardwareCleanupError, "refusing to clear"):
                _clear_episode_buffer_after_workers(dataset, workers_stopped=stopped)
            dataset.clear_episode_buffer.assert_not_called()

            robot_cleanup = mock.Mock()
            dataset_finalize = mock.Mock()
            self.assertIsNone(
                _cleanup_after_episode_workers(
                    workers_stopped=stopped,
                    label="robot disconnect",
                    cleanup=robot_cleanup,
                )
            )
            self.assertIsNone(
                _cleanup_after_episode_workers(
                    workers_stopped=stopped,
                    label="dataset finalization",
                    cleanup=dataset_finalize,
                )
            )
            robot_cleanup.assert_not_called()
            dataset_finalize.assert_not_called()
        finally:
            release.set()
            worker.join(timeout=1.0)

    def test_capture_exit_is_proved_before_buffer_clear_after_unblock(self) -> None:
        events: list[str] = []
        release = threading.Event()

        class BlockingCapture(threading.Thread):
            def __init__(self) -> None:
                super().__init__(daemon=True)
                self.stop_event = threading.Event()

            def run(self) -> None:
                release.wait()
                events.append("worker-exit")

            def request_stop(self) -> None:
                events.append("stop-signal")
                self.stop_event.set()

            def unblock_inputs(self) -> None:
                events.append("input-unblock")
                release.set()

        capture = BlockingCapture()
        capture.start()
        client = SimpleNamespace(
            shutdown_event=threading.Event(),
            start_barrier=mock.Mock(),
            stop=mock.Mock(side_effect=lambda: events.append("client-close")),
        )
        dataset = SimpleNamespace(
            clear_episode_buffer=lambda: events.append("buffer-clear")
        )

        stopped, error = _stop_episode_workers(
            client=client,
            capture=capture,
            workers=[("capture", capture)],
            join_timeout=0.01,
        )
        self.assertTrue(stopped)
        self.assertIsInstance(error, RuntimeError)
        _clear_episode_buffer_after_workers(dataset, workers_stopped=stopped)

        self.assertFalse(capture.is_alive())
        self.assertLess(events.index("worker-exit"), events.index("buffer-clear"))

    def test_observation_camera_read_is_unblocked_without_recording(self) -> None:
        release = threading.Event()
        observation = threading.Thread(target=release.wait, daemon=True)
        observation.start()
        camera = SimpleNamespace(
            disconnect=mock.Mock(side_effect=release.set),
        )
        client = SimpleNamespace(
            shutdown_event=threading.Event(),
            start_barrier=mock.Mock(),
            stop=mock.Mock(),
            robot=SimpleNamespace(cameras={"overhead": camera}),
        )

        stopped, error = _stop_episode_workers(
            client=client,
            capture=None,
            workers=[("observation", observation)],
            join_timeout=0.01,
        )

        self.assertTrue(stopped)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(observation.is_alive())
        camera.disconnect.assert_called_once_with()

    def test_ik_reset_stop_joins_after_kill_before_clearing_reference(self) -> None:
        process = mock.Mock()
        process.is_alive.side_effect = [True, True, False]
        controller = object.__new__(IKResetController)
        controller._conn = None
        controller._proc = process
        controller._ready = True

        controller.stop()

        self.assertEqual(
            process.join.call_args_list,
            [
                mock.call(timeout=3.0),
                mock.call(timeout=2.0),
                mock.call(timeout=2.0),
            ],
        )
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertIsNone(controller._proc)
        self.assertFalse(controller._ready)

    def test_ik_reset_start_retains_and_reaps_child_after_pipe_failure(self) -> None:
        startup = OSError("child pipe close failed")
        parent_conn = mock.Mock()
        child_conn = mock.Mock()
        child_conn.close.side_effect = startup
        process = mock.Mock(pid=9123)
        process.is_alive.side_effect = [True, False]
        context = mock.Mock()
        context.Pipe.return_value = (parent_conn, child_conn)
        context.Process.return_value = process
        controller = object.__new__(IKResetController)
        controller._vr_cfg = object()
        controller._kin_cfg = object()
        controller._proc = None
        controller._conn = None
        controller._ready = False

        with (
            mock.patch("multiprocessing.get_context", return_value=context),
            self.assertRaisesRegex(OSError, "child pipe close failed") as raised,
        ):
            controller.start()

        self.assertIs(raised.exception, startup)
        process.start.assert_called_once_with()
        process.terminate.assert_called_once_with()
        self.assertEqual(
            process.join.call_args_list,
            [mock.call(timeout=3.0), mock.call(timeout=2.0)],
        )
        self.assertIsNone(controller._proc)
        self.assertIsNone(controller._conn)
        parent_conn.close.assert_called_once_with()

    def test_ik_reset_stop_retains_process_when_post_kill_exit_is_unproved(
        self,
    ) -> None:
        process = mock.Mock()
        process.is_alive.return_value = True
        controller = object.__new__(IKResetController)
        controller._conn = None
        controller._proc = process
        controller._ready = True

        with self.assertRaisesRegex(RuntimeError, "ownership is uncertain"):
            controller.stop()

        self.assertEqual(
            process.join.call_args_list,
            [
                mock.call(timeout=3.0),
                mock.call(timeout=2.0),
                mock.call(timeout=2.0),
            ],
        )
        self.assertIs(controller._proc, process)
        self.assertTrue(controller._ready)

    def test_policy_server_shutdown_joins_after_kill_and_proves_exit(self) -> None:
        process = mock.Mock()
        process.is_alive.side_effect = [True, True, False]

        _shutdown_policy_server_process(process)

        self.assertEqual(
            process.join.call_args_list,
            [mock.call(timeout=5.0), mock.call(timeout=2.0)],
        )
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def test_policy_server_shutdown_propagates_error_after_exit_proof(self) -> None:
        process = mock.Mock()
        terminate_error = OSError("terminate syscall failed")
        process.terminate.side_effect = terminate_error
        process.is_alive.side_effect = [True, True, False]

        with self.assertRaisesRegex(RuntimeError, "cleanup encountered") as raised:
            _shutdown_policy_server_process(process)

        self.assertIs(raised.exception.__cause__, terminate_error)
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.join.call_args_list,
            [mock.call(timeout=5.0), mock.call(timeout=2.0)],
        )

    def test_policy_server_shutdown_fails_if_post_kill_exit_is_unproved(
        self,
    ) -> None:
        process = mock.Mock()
        process.is_alive.return_value = True

        with self.assertRaisesRegex(RuntimeError, "ownership is uncertain"):
            _shutdown_policy_server_process(process)

        self.assertEqual(
            process.join.call_args_list,
            [mock.call(timeout=5.0), mock.call(timeout=2.0)],
        )

    def test_chunk_alignment_never_offsets_equivalent_wrapped_rotvecs(self) -> None:
        features = [
            "left_ee.x",
            "left_ee.y",
            "left_ee.z",
            "left_ee.rx",
            "left_ee.ry",
            "left_ee.rz",
            "left_gripper.pos",
        ]
        wrapped = -math.pi + 0.01
        incoming = [
            _TimedAction(
                timestep=10 + tick,
                action=torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, wrapped, 0.0],
                    dtype=torch.float32,
                ),
            )
            for tick in range(4)
        ]
        last_target = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, math.pi - 0.01, 1.0],
            dtype=np.float32,
        )

        _align_action_chunk(
            incoming,
            last_target=last_target,
            latest_action=9,
            align_ticks=4,
            exempt_indices=_snap_to_newest_indices(features),
        )

        # Linear translation still receives the intended continuity fade.
        self.assertAlmostEqual(incoming[0].action[0].item(), 1.0)
        self.assertAlmostEqual(incoming[1].action[0].item(), 0.75)
        # The same componentwise offset on rz would turn the new -pi
        # representative into +pi and fade it through identity. Rotation and
        # bang-bang gripper dimensions must remain exactly policy-authored.
        for timed_action in incoming:
            self.assertAlmostEqual(timed_action.action[5].item(), wrapped, places=6)
            self.assertEqual(timed_action.action[6].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
