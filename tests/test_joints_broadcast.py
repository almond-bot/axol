"""The ``joints`` push (HUD box readout) is shared by both teleop adapters.

``VRTeleopCore.maybe_broadcast_joints`` throttles to ``JOINT_BROADCAST_HZ``,
prefers the robot's measured positions over the command, and carries the IK
worker's pair status. ``axol teleop`` and collect-data (``AxolVRTeleop``)
both call it from their control loop.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np

from almond_axol.teleop import core as core_module
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore


class _Arm:
    def __init__(self, value: float) -> None:
        self.positions = np.full(8, value)


class _Robot:
    left = _Arm(0.5)
    right = _Arm(-0.5)


def _core(sent: list) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
        broadcast_json=sent.append,
    )


class JointsBroadcastTest(unittest.TestCase):
    def test_measured_positions_and_pair_status(self) -> None:
        sent: list = []
        core = _core(sent)
        core.pair_status = {"aligned": True}
        core.maybe_broadcast_joints(np.zeros(16), _Robot())
        (msg,) = sent
        self.assertEqual(msg["type"], "joints")
        self.assertEqual(msg["value"]["pair"], {"aligned": True})
        self.assertEqual(set(msg["value"]["q"].values()), {0.5, -0.5})
        self.assertEqual(msg["value"]["l_grip"], 0.5)

    def test_falls_back_to_the_command_without_measured_arms(self) -> None:
        sent: list = []
        _core(sent).maybe_broadcast_joints(np.full(16, 0.25), object())
        self.assertEqual(set(sent[0]["value"]["q"].values()), {0.25})

    def test_throttled(self) -> None:
        sent: list = []
        core = _core(sent)
        with patch.object(
            core_module.time, "perf_counter", side_effect=[10.0, 10.01, 10.2]
        ):
            for _ in range(3):
                core.maybe_broadcast_joints(np.zeros(16), object())
        self.assertEqual(len(sent), 2)


class CollectDataSendsJointsTest(unittest.TestCase):
    def test_get_action_pushes_joints(self) -> None:
        from almond_axol.lerobot.teleop.config_vr import AxolVRTeleopConfig
        from almond_axol.lerobot.teleop.teleop_vr import AxolVRTeleop

        teleop = AxolVRTeleop(AxolVRTeleopConfig())
        teleop.measured_robot = _Robot()
        with (
            patch.object(
                AxolVRTeleop,
                "is_connected",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(teleop._core, "compute_output", return_value=np.zeros(16)),
            patch.object(teleop._core, "maybe_broadcast_joints") as push,
        ):
            teleop.get_action()
        push.assert_called_once()
        self.assertIs(push.call_args.args[1], teleop.measured_robot)


if __name__ == "__main__":
    unittest.main()
