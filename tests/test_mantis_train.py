from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from lerobot.scripts import lerobot_train

from almond_axol.cli import mantis_train


class MantisTrainRemoteSafetyTest(unittest.TestCase):
    def test_explicit_remote_targets_fail_before_training_dispatch(self) -> None:
        cases = (
            ["--dataset.repo_id=test/data", "--job.target=a10g-small"],
            ["--dataset.repo_id=test/data", "--job.target", "a10g-small"],
            ["--config_path=test/checkpoint", "--job.target=a100-large"],
            ["--config_path", "test/checkpoint", "--job.target", "a100-large"],
        )

        for argv in cases:
            with (
                self.subTest(argv=argv),
                mock.patch("almond_axol.mantis.train_patch.install") as install,
                mock.patch.object(lerobot_train, "main") as train_main,
                self.assertRaisesRegex(
                    SystemExit, r"does not support remote HF Jobs.*relative-EE"
                ),
            ):
                mantis_train.main(argv)

            install.assert_not_called()
            train_main.assert_not_called()

    def test_local_target_preserves_upstream_cli(self) -> None:
        argv = [
            "--dataset.repo_id=test/data",
            "--policy.type=act",
            "--job.target",
            "local",
        ]
        original_submit = lerobot_train.submit_to_hf
        try:
            with (
                mock.patch("almond_axol.mantis.train_patch.install") as install,
                mock.patch.object(lerobot_train, "main") as train_main,
                mock.patch.object(sys, "argv", ["test"]),
            ):
                mantis_train.main(argv)

                install.assert_called_once_with(lerobot_train)
                train_main.assert_called_once_with()
                self.assertEqual(sys.argv, ["lerobot-train", *argv])
        finally:
            lerobot_train.submit_to_hf = original_submit

    def test_remote_target_inherited_from_config_is_guarded(self) -> None:
        cfg = SimpleNamespace(job=SimpleNamespace(target="a10g-small"))

        with self.assertRaisesRegex(
            SystemExit, r"does not support remote HF Jobs.*relative-EE"
        ):
            mantis_train._reject_remote_submission(cfg)


if __name__ == "__main__":
    unittest.main()
