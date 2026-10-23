"""Sessions refuse to start with an arm joint far from zero."""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from almond_axol.robot.start_pose import (
    START_POSE_TOLERANCE_DEG,
    StartPoseError,
    check_start_pose,
    start_pose_deviations,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.teleop import VRTeleop


def _arm(*joints_deg: float, gripper: float = 0.5) -> np.ndarray:
    """An (8,) arm reading: the given leading joints (deg), the rest at zero."""
    q = np.zeros(8, dtype=np.float32)
    q[: len(joints_deg)] = np.radians(joints_deg)
    q[7] = gripper
    return q


class StartPoseCheckTest(unittest.TestCase):
    def test_zero_passes(self) -> None:
        check_start_pose((_arm(), _arm()))

    def test_the_default_rest_pose_passes(self) -> None:
        cfg = VRTeleopConfig()
        check_start_pose(
            (np.append(cfg.rest_pose_left, 0.0), np.append(cfg.rest_pose_right, 0.0))
        )

    def test_tolerance_is_measured_from_zero_not_the_rest_pose(self) -> None:
        # 30° is within 25° of the default 18° rest elbow, but not of zero.
        with self.assertRaisesRegex(StartPoseError, "left elbow"):
            check_start_pose((_arm(0, 0, 0, 30), _arm()))

    def test_a_joint_just_inside_the_tolerance_passes(self) -> None:
        check_start_pose((_arm(), _arm(0, -(START_POSE_TOLERANCE_DEG - 0.5))))

    def test_a_joint_past_the_tolerance_is_refused_by_name(self) -> None:
        with self.assertRaisesRegex(StartPoseError, "right shoulder_2") as raised:
            check_start_pose((_arm(), _arm(0, -(START_POSE_TOLERANCE_DEG + 1.0))))
        self.assertIn("-26.0°", str(raised.exception))

    def test_every_offending_joint_is_listed(self) -> None:
        found = start_pose_deviations((_arm(60), _arm(0, 0, 0, 0, 0, 0, -60)))
        self.assertEqual(
            [(d.side, d.joint) for d in found],
            [("left", "shoulder_1"), ("right", "wrist_3")],
        )

    def test_the_gripper_is_not_checked(self) -> None:
        check_start_pose((_arm(gripper=1.0), _arm(gripper=0.0)), tolerance_deg=0.1)

    def test_an_absent_arm_is_skipped(self) -> None:
        check_start_pose((None, _arm()))

    def test_a_non_finite_reading_is_refused(self) -> None:
        left = _arm()
        left[2] = np.nan
        with self.assertRaisesRegex(StartPoseError, "left shoulder_3"):
            check_start_pose((left, _arm()))


class TeleopStartupRefusalTest(unittest.IsolatedAsyncioTestCase):
    async def test_far_arms_are_refused_before_the_ik_worker_starts(self) -> None:
        teleop = object.__new__(VRTeleop)
        teleop._vr_server = SimpleNamespace(enable=AsyncMock(), disable=AsyncMock())
        teleop._vr_thread = None
        teleop._vr_loop = None
        teleop._vr_stop = threading.Event()
        teleop._vr_ready = threading.Event()
        teleop._vr_start_error = None
        teleop._vr_cleanup_error = None
        teleop._ik_thread = None
        teleop._ik_stop = threading.Event()
        teleop._parent_conn = None
        teleop._ik_process = None
        teleop._jelly = None
        teleop._live = SimpleNamespace(announce=Mock(), apply=Mock())
        teleop._robot = SimpleNamespace(
            enable=AsyncMock(),
            disable=AsyncMock(),
            get_positions=AsyncMock(return_value=(_arm(0, 0, 0, 60), _arm())),
        )
        get_context = Mock()

        with (
            patch("almond_axol.teleop.teleop.multiprocessing.get_context", get_context),
            self.assertRaisesRegex(StartPoseError, "left elbow"),
        ):
            async with asyncio.timeout(1.0):
                await VRTeleop.__aenter__(teleop)

        get_context.assert_not_called()
        teleop._robot.disable.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
