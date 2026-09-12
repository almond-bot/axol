"""``almond_axol.robot.Axol`` / ``Mantis`` keep the classic API.

The realtime core is an implementation detail: ``Axol(...)`` takes the same
arguments as before the Rust port, exposes the same lifecycle / read / write
surface, and only differs behind the scenes (which calls go to the core's
telemetry versus the bus, and when the bus is available at all).
"""

from __future__ import annotations

import inspect
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from almond_axol.constants import Joint
from almond_axol.motor import MotorError
from almond_axol.robot import Axol, AxolConfig, Mantis, RobotBase, Sim
from almond_axol.robot import __all__ as robot_exports
from almond_axol.robot import axol as axol_module
from almond_axol.robot import mantis as mantis_module
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.mantis import MantisHardware
from almond_axol.rt import Axol as RtModuleAxol
from almond_axol.rt import Mantis as RtModuleMantis

# The public surface of ``almond_axol.robot.Axol`` before the realtime core
# existed. Every name must still be there; the core must not leak new
# required steps into user code.
_CLASSIC_AXOL_API = {
    "connect",
    "enable",
    "disable",
    "disconnect",
    "start_telemetry",
    "stop_telemetry",
    "wait_for_telemetry",
    "clear_errors",
    "set_control_mode",
    "get_positions",
    "get_velocities",
    "get_torques",
    "get_temperatures",
    "get_voltages",
    "get_error_codes",
    "get_holding",
    "get_gains",
    "set_gains",
    "set_zero_position",
    "set_acceleration",
    "set_positions_velocity",
    "set_velocity",
    "motion_control",
    "gravity_compensate",
    "reset_gravity_hold",
    "reset_command_state",
    "torque_residuals",
    "left",
    "right",
}


class AxolApiTest(unittest.TestCase):
    def setUp(self) -> None:
        # No CAN, no core binary: construction must not need either.
        self.enterContext(patch.object(axol_module, "CanBus"))
        self.enterContext(patch("almond_axol.rt.robot.RtLink"))

    def test_axol_is_the_one_robot_class(self) -> None:
        self.assertIs(Axol, RtModuleAxol)
        self.assertTrue(issubclass(Axol, RobotBase))
        self.assertTrue(issubclass(Sim, RobotBase))
        # The low-level object is internal: not part of the package surface.
        self.assertNotIn("AxolHardware", robot_exports)
        self.assertNotIn("MantisHardware", robot_exports)

    def test_classic_surface_is_intact(self) -> None:
        missing = sorted(name for name in _CLASSIC_AXOL_API if not hasattr(Axol, name))
        self.assertEqual(missing, [])
        # No leftovers from the wrapper era.
        for name in ("hardware", "detach"):
            self.assertFalse(hasattr(Axol, name), name)

    def test_classic_constructor_signature(self) -> None:
        params = inspect.signature(Axol).parameters
        positional = [
            name
            for name, p in params.items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        self.assertEqual(
            positional,
            ["config", "left_channel", "right_channel", "left_joints", "right_joints"],
        )
        # Core tuning is keyword-only and optional.
        for name in ("loop_hz", "watchdog_ms", "max_vel", "max_accel", "record"):
            self.assertIs(params[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNot(params[name].default, inspect.Parameter.empty)
        self.assertNotIn("hardware", params)

    def test_forwards_hardware_arguments(self) -> None:
        config = AxolConfig(has_gripper=False)
        robot = Axol(
            config,
            left_channel="can0",
            right_channel=None,
            left_joints={Joint.WRIST_2, Joint.WRIST_3},
            max_vel=1.0,
        )
        self.assertIsInstance(robot._robot, AxolHardware)
        self.assertIs(robot.left, robot._robot.left)
        self.assertIsNone(robot.right)
        assert robot.left is not None
        self.assertEqual(set(robot.left.motors), {Joint.WRIST_2, Joint.WRIST_3})
        self.assertEqual(robot._max_vel, 1.0)

    def test_default_config_when_omitted(self) -> None:
        robot = Axol(left_channel="can0", right_channel=None)
        assert robot.left is not None
        self.assertIn(Joint.GRIPPER, robot.left.motors)

    def test_hardware_validation_still_applies(self) -> None:
        with self.assertRaisesRegex(ValueError, "different CAN interfaces"):
            Axol(left_channel="can0", right_channel="can0")

    def test_wrap_builds_around_an_existing_object(self) -> None:
        hardware = AxolHardware(left_channel="can0", right_channel=None)
        robot = Axol._wrap(hardware, watchdog_ms=99.0)
        self.assertIs(robot._robot, hardware)
        self.assertEqual(robot._watchdog_ms, 99.0)
        self.assertFalse(robot._armed)


class AxolBusOwnershipTest(unittest.IsolatedAsyncioTestCase):
    """Which calls reach the bus depends on who owns it."""

    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module, "CanBus"))
        self.enterContext(patch("almond_axol.rt.robot.RtLink"))
        self.robot = Axol(left_channel="can0", right_channel=None)
        self.hardware = self.robot._robot

    async def test_quiet_bus_calls_go_to_the_hardware(self) -> None:
        pair = (np.zeros(8, dtype=np.float32), None)
        with (
            patch.object(self.hardware, "connect", AsyncMock()) as connect,
            patch.object(self.hardware, "get_positions", AsyncMock(return_value=pair)),
            patch.object(
                self.hardware, "get_holding", AsyncMock(return_value=([], None))
            ),
            patch.object(self.hardware, "start_telemetry", AsyncMock()) as telemetry,
            patch.object(self.hardware, "disconnect", AsyncMock()) as disconnect,
            patch.object(self.hardware, "disable", AsyncMock()) as disable,
        ):
            await self.robot.connect()
            self.assertIs(await self.robot.get_positions(), pair)
            self.assertEqual(await self.robot.get_holding(), ([], None))
            await self.robot.start_telemetry(100)
            await self.robot.disconnect()
            await self.robot.disable()
        connect.assert_awaited_once()
        telemetry.assert_awaited_once_with(100, torque=False)
        disconnect.assert_awaited_once()
        # No core was started for this session: a classic torque-off.
        disable.assert_awaited_once()

    async def test_register_calls_are_refused_while_the_core_owns_the_bus(self) -> None:
        self.robot._armed = True
        for call in (
            self.robot.connect(),
            self.robot.get_holding(),
            self.robot.get_temperatures(),
            self.robot.set_control_mode(None),  # type: ignore[arg-type]
            self.robot.set_gains(),
        ):
            with self.assertRaisesRegex(MotorError, "realtime core owns the CAN bus"):
                await call

    async def test_motion_requires_enable(self) -> None:
        with self.assertRaisesRegex(MotorError, "call enable\\(\\) first"):
            await self.robot.motion_control(left=np.zeros(8, dtype=np.float32))
        with self.assertRaisesRegex(MotorError, "call enable\\(\\) first"):
            await self.robot.gravity_compensate()

    async def test_cached_reads_while_armed_send_no_can(self) -> None:
        self.robot._armed = True
        positions = np.arange(8, dtype=np.float32)
        torques = np.full(8, 0.5, dtype=np.float32)
        arm = self.hardware.left
        assert arm is not None
        with (
            patch.object(type(arm), "positions", property(lambda _self: positions)),
            patch.object(type(arm), "torques", property(lambda _self: torques)),
            patch.object(self.hardware, "get_positions", AsyncMock()) as bus_read,
        ):
            pos_l, pos_r = await self.robot.get_positions()
            trq_l, _ = await self.robot.get_torques()
        bus_read.assert_not_awaited()
        assert pos_l is not None and trq_l is not None
        np.testing.assert_array_equal(pos_l, positions)
        self.assertIsNot(pos_l, positions)
        self.assertIsNone(pos_r)
        np.testing.assert_array_equal(trq_l, torques)


class MantisApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(mantis_module, "CanBus"))

    def test_mantis_is_the_one_rig_class(self) -> None:
        self.assertIs(Mantis, RtModuleMantis)
        self.assertTrue(issubclass(Mantis, RobotBase))
        for name in ("hardware", "robot", "detach"):
            self.assertFalse(hasattr(Mantis, name), name)

    def test_classic_constructor_signature(self) -> None:
        params = inspect.signature(Mantis).parameters
        positional = [
            name
            for name, p in params.items()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        self.assertEqual(positional, ["config", "left_channel", "right_channel"])
        self.assertIs(
            params["defer_gripper_enable"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        self.assertNotIn("hardware", params)

    def test_forwards_hardware_arguments(self) -> None:
        rig = Mantis(
            AxolConfig(),
            left_channel="can_l",
            right_channel=None,
            defer_gripper_enable=True,
            watchdog_ms=99.0,
        )
        self.assertIsInstance(rig._robot, MantisHardware)
        self.assertIs(rig.left, rig._robot.left)
        self.assertIsNone(rig.right)
        self.assertTrue(rig._robot._defer_gripper_enable)
        self.assertEqual(rig._watchdog_ms, 99.0)
        self.assertFalse(rig.armed)

    def test_wrap_builds_around_an_existing_object(self) -> None:
        hardware = MantisHardware(left_channel="can_l", right_channel=None)
        self.assertIs(Mantis._wrap(hardware)._robot, hardware)
        with self.assertRaisesRegex(ValueError, "different CAN interfaces"):
            Mantis(left_channel="can_l", right_channel="can_l")


if __name__ == "__main__":
    unittest.main()
