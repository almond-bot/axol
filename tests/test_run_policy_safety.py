from __future__ import annotations

import math
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from almond_axol.cli.run_policy import (
    _align_action_chunk,
    _lingering_episode_thread_error,
    _snap_to_newest_indices,
)
from almond_axol.cli import run_policy
from almond_axol.lerobot import action_schema as action_schema_module
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
