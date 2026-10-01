"""Reset pose configuration reaches the IK worker before any hardware opens."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.lerobot.rollout import IKResetController
from almond_axol.teleop.config import VRTeleopConfig


class ResetPoseTest(unittest.TestCase):
    def worker_config(self, controller: IKResetController):
        from almond_axol.teleop.worker import run_ik_worker

        process = Mock()
        process.is_alive.return_value = False
        context = Mock()
        parent, child = Mock(), Mock()
        context.Pipe.return_value = parent, child
        context.Process.return_value = process
        with patch("multiprocessing.get_context", return_value=context):
            try:
                controller.start()
                launch = context.Process.call_args.kwargs
                self.assertIs(launch["target"], run_ik_worker)
                self.assertIs(launch["args"][0], child)
                process.start.assert_called_once_with()
                child.close.assert_called_once_with()
                return launch["args"][1]
            finally:
                controller.stop()

    def test_omitted_poses_preserve_stock_worker_configuration(self) -> None:
        config = self.worker_config(IKResetController())
        stock = VRTeleopConfig()
        np.testing.assert_array_equal(config.rest_pose_left, stock.rest_pose_left)
        np.testing.assert_array_equal(config.rest_pose_right, stock.rest_pose_right)
        self.assertEqual(config.frequency, stock.frequency)
        self.assertEqual(config.reset_speed, stock.reset_speed)

    def test_pose_overrides_are_copied_into_the_worker_configuration(self) -> None:
        left = np.array([-0.8, 0.1, 0.2, 1.4, 0.3, 0.4, 0.35])
        right = [0.9, -0.1, -0.2, -1.4, -0.3, -0.4, -0.35]
        expected_left = left.astype(np.float32)
        expected_right = np.asarray(right, dtype=np.float32)
        controller = IKResetController(rest_pose_left=left, rest_pose_right=right)
        left[:] = 0
        right[:] = [0] * 7
        config = self.worker_config(controller)
        np.testing.assert_array_equal(config.rest_pose_left, expected_left)
        np.testing.assert_array_equal(config.rest_pose_right, expected_right)
        self.assertEqual(config.rest_pose_left.dtype, np.float32)
        self.assertEqual(config.rest_pose_right.dtype, np.float32)

    def test_each_arm_can_be_overridden_independently(self) -> None:
        stock = VRTeleopConfig()
        for side in ("left", "right"):
            with self.subTest(side=side):
                name = f"rest_pose_{side}"
                other = "rest_pose_right" if side == "left" else "rest_pose_left"
                config = self.worker_config(IKResetController(**{name: [0.2] * 7}))
                np.testing.assert_array_equal(
                    getattr(config, name), np.full(7, 0.2, dtype=np.float32)
                )
                np.testing.assert_array_equal(
                    getattr(config, other), getattr(stock, other)
                )

    def test_invalid_pose_is_rejected_before_worker_start(self) -> None:
        invalid = (
            [],
            [0.0] * 6,
            [0.0] * 8,
            [[0.0] * 7],
            0.0,
            [float("nan")] * 7,
            [float("inf")] * 7,
            [1e300] * 7,
            ["not an angle"] * 7,
        )
        with patch("multiprocessing.get_context") as spawn:
            for side in ("left", "right"):
                name = f"rest_pose_{side}"
                for value in invalid:
                    with self.subTest(side=side, value=value):
                        with self.assertRaisesRegex(
                            ValueError, f"{name}.*seven finite"
                        ):
                            IKResetController(**{name: value})
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
