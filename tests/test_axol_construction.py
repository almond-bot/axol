"""``almond_axol.robot.Axol`` is the realtime-core robot and builds its own hardware.

``Axol(...)`` takes the same arguments as the low-level ``AxolHardware`` and
constructs it internally; ``hardware=`` wraps an existing one; the old
``RtAxol(AxolHardware(...))`` spelling keeps working behind a
``DeprecationWarning``.
"""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

from almond_axol.constants import Joint
from almond_axol.robot import Axol, AxolConfig, AxolHardware, RobotBase, Sim
from almond_axol.robot import axol as axol_module
from almond_axol.rt import Axol as RtModuleAxol
from almond_axol.rt import RtAxol


class AxolConstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        # No CAN, no core binary: construction must not need either.
        self.enterContext(patch.object(axol_module, "CanBus"))
        self.enterContext(patch("almond_axol.rt.robot.RtLink"))

    def test_axol_is_the_realtime_core_robot(self) -> None:
        self.assertIs(Axol, RtModuleAxol)
        self.assertTrue(issubclass(Axol, RobotBase))
        self.assertTrue(issubclass(Sim, RobotBase))
        self.assertFalse(issubclass(AxolHardware, Axol))

    def test_forwards_hardware_arguments(self) -> None:
        config = AxolConfig(has_gripper=False)
        robot = Axol(
            config,
            left_channel="can0",
            right_channel=None,
            left_joints={Joint.WRIST_2, Joint.WRIST_3},
            max_vel=1.0,
        )
        self.assertIsInstance(robot.hardware, AxolHardware)
        self.assertIs(robot.left, robot.hardware.left)
        self.assertIsNone(robot.right)
        assert robot.left is not None
        self.assertEqual(set(robot.left.motors), {Joint.WRIST_2, Joint.WRIST_3})
        self.assertEqual(robot._max_vel, 1.0)

    def test_default_config_when_omitted(self) -> None:
        robot = Axol(left_channel="can0", right_channel=None)
        assert robot.left is not None
        self.assertIn(Joint.GRIPPER, robot.left.motors)

    def test_hardware_keyword_wraps_an_existing_object(self) -> None:
        hardware = AxolHardware(left_channel="can0", right_channel=None)
        robot = Axol(hardware=hardware)
        self.assertIs(robot.hardware, hardware)
        with self.assertRaisesRegex(ValueError, "do not also pass"):
            Axol(hardware=hardware, left_channel="can1")
        with self.assertRaisesRegex(ValueError, "do not also pass"):
            Axol(AxolConfig(), hardware=hardware)

    def test_hardware_validation_still_applies(self) -> None:
        with self.assertRaisesRegex(ValueError, "different CAN interfaces"):
            Axol(left_channel="can0", right_channel="can0")

    def test_rtaxol_is_a_deprecated_alias(self) -> None:
        hardware = AxolHardware(left_channel="can0", right_channel=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            robot = RtAxol(hardware, max_vel=2.0)
            # The old idiom written against the new ``Axol`` still resolves.
            nested = RtAxol(Axol(hardware=hardware))
        self.assertEqual([w.category for w in caught], [DeprecationWarning] * 2)
        self.assertIsInstance(robot, Axol)
        self.assertIs(robot.hardware, hardware)
        self.assertEqual(robot._max_vel, 2.0)
        self.assertIs(nested.hardware, hardware)


if __name__ == "__main__":
    unittest.main()
