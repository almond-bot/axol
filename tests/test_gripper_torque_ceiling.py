"""The gripper can never be driven above GRIPPER_MAX_TORQUE (5 Nm)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from almond_axol.constants import GRIPPER_MAX_TORQUE, Joint
from almond_axol.motor.motor import Motor
from almond_axol.motor.types import ControlMode
from almond_axol.robot.config import ArmConfig, PositionForceConfig
from almond_axol.serve.settings import SettingsStore


class GripperTorqueCeilingTest(unittest.TestCase):
    def test_ceiling_is_five_nm(self) -> None:
        self.assertEqual(GRIPPER_MAX_TORQUE, 5.0)

    def test_config_accepts_up_to_the_ceiling(self) -> None:
        for torque in (0.5, 3.0, 5.0):
            with self.subTest(torque=torque):
                cfg = PositionForceConfig(torque_limit=torque, max_speed=10.0)
                self.assertEqual(cfg.torque_limit, torque)
        self.assertLessEqual(ArmConfig().gripper.torque_limit, GRIPPER_MAX_TORQUE)

    def test_config_rejects_above_the_ceiling(self) -> None:
        for torque in (5.01, 6.0, math.inf, math.nan):
            with self.subTest(torque=torque):
                with self.assertRaisesRegex(ValueError, "5 Nm maximum"):
                    PositionForceConfig(torque_limit=torque, max_speed=10.0)

    def test_settings_save_rejects_above_the_ceiling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SettingsStore(Path(directory) / "settings.json")
            store.update(values={"axol.right.gripper.torque_limit": 5})
            self.assertEqual(
                store.snapshot()["values"]["axol.right.gripper.torque_limit"], 5.0
            )
            for value in (5.5, 0, -1, math.nan, "lots"):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "Right gripper torque"):
                        store.update(values={"axol.right.gripper.torque_limit": value})
            # The rejected saves left the last good value in place.
            self.assertEqual(
                store.snapshot()["values"]["axol.right.gripper.torque_limit"], 5.0
            )


class MotorPositionForceClampTest(unittest.IsolatedAsyncioTestCase):
    async def test_command_torque_is_clamped(self) -> None:
        motor = object.__new__(Motor)
        motor.joint = Joint.GRIPPER
        motor.mode = ControlMode.POSITION_FORCE
        motor._driver = Mock(set_position_force=AsyncMock())

        await motor.set_position_force(1.0, 10.0, 2.0)
        await motor.set_position_force(1.0, 10.0, 12.0)

        torques = [c.args[2] for c in motor._driver.set_position_force.await_args_list]
        self.assertEqual(torques, [2.0, GRIPPER_MAX_TORQUE])


if __name__ == "__main__":
    unittest.main()
