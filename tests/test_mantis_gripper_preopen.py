"""The Mantis pre-record gripper open: wide open, then torque off.

``collect-data --mantis`` opens both handheld grippers all the way right after
the CAN buses come up and releases them again, so the first episode starts
with the jaws already at their open stop (and the one-time hard-stop
calibration is out of the way before any take).
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from almond_axol.cli import collect_data
from almond_axol.motor import ControlMode, MotorError
from almond_axol.robot.axol import GRIPPER_TRAVEL
from almond_axol.robot.mantis import Mantis, MantisGripperArm


class _FakeGripperMotor:
    """A Damiao gripper stand-in: tracks commanded position, stops on a torque."""

    def __init__(self, *, start: float = 1.25, stop_after_steps: int = 2) -> None:
        self.position_value = start
        self.has_position = False
        self.calls: list[object] = []
        self._steps = 0
        self._stop_after = stop_after_steps

    async def enable(self) -> None:
        self.calls.append("enable")

    async def disable(self) -> None:
        self.calls.append("disable")

    async def set_control_mode(self, mode: ControlMode) -> None:
        self.calls.append(("mode", mode))

    async def get_position(self) -> float:
        return self.position_value

    async def set_impedance(self, target: float, *_: float) -> None:
        self.calls.append("sweep")
        self.position_value = target
        self._steps += 1

    async def get_torque(self) -> float:
        return 0.6 if self._steps >= self._stop_after else 0.1

    async def set_position_force(self, raw: float, speed: float, torque: float) -> None:
        self.calls.append(("pf", raw))
        self.position_value = raw
        self.has_position = True

    @property
    def position(self) -> float:
        return self.position_value


def _arm(motor: _FakeGripperMotor) -> MantisGripperArm:
    with patch("almond_axol.robot.mantis.Motor", return_value=motor):
        return MantisGripperArm(
            Mock(), SimpleNamespace(max_speed=10.0, torque_limit=0.5)
        )


def _mantis(left: MantisGripperArm, right: MantisGripperArm) -> Mantis:
    robot = object.__new__(Mantis)
    robot.left = left
    robot.right = right
    robot._left_bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    robot._right_bus = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    robot._defer_gripper_enable = True
    robot._connected = False
    robot._shutdown_pending = False
    robot._telemetry_settings = None
    robot._lifecycle_lock = asyncio.Lock()
    return robot


class GripperArmOpenFullyTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_fully_requires_an_enabled_calibrated_gripper(self) -> None:
        arm = _arm(_FakeGripperMotor())
        with self.assertRaisesRegex(MotorError, "must be enabled"):
            await arm.open_fully()

    async def test_open_fully_commands_the_open_stop_and_confirms_it(self) -> None:
        motor = _FakeGripperMotor()
        arm = _arm(motor)
        arm._enabled = True
        arm._calibrated = True
        arm._open_pos = 1.0
        arm._closed_pos = 1.0 + GRIPPER_TRAVEL
        arm._gripper_target = 0.2  # a stale trigger value must not win

        await arm.open_fully()

        self.assertEqual(motor.calls, [("pf", 1.0)])
        self.assertEqual(arm._gripper_target, 1.0)
        self.assertGreaterEqual(float(arm.positions[-1]), 0.95)

    async def test_open_fully_times_out_when_the_jaws_do_not_reach_open(self) -> None:
        motor = _FakeGripperMotor()
        arm = _arm(motor)
        arm._enabled = True
        arm._calibrated = True
        arm._open_pos = 1.0
        arm._closed_pos = 1.0 + GRIPPER_TRAVEL

        async def stuck_half_way(raw: float, speed: float, torque: float) -> None:
            motor.position_value = 1.0 + GRIPPER_TRAVEL / 2
            motor.has_position = True

        motor.set_position_force = stuck_half_way  # type: ignore[method-assign]
        with (
            patch("almond_axol.robot.mantis._OPEN_FULLY_POLL_S", 0.0),
            self.assertRaisesRegex(MotorError, "did not open all the way"),
        ):
            await arm.open_fully(timeout=0.02)

    async def test_preset_open_target_only_touches_the_gripper(self) -> None:
        arm = _arm(_FakeGripperMotor())
        arm._virtual_arm[:] = 0.5
        arm._gripper_target = 0.1
        arm.preset_open_target()
        self.assertEqual(arm._gripper_target, 1.0)
        self.assertTrue((arm._virtual_arm == 0.5).all())


class MantisOpenGrippersTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_grippers_sweeps_open_confirms_then_releases(self) -> None:
        left_motor, right_motor = _FakeGripperMotor(), _FakeGripperMotor(start=2.0)
        left, right = _arm(left_motor), _arm(right_motor)
        # A trigger value latched before torque-on must not be where the jaws go.
        left._gripper_target = 0.0
        robot = _mantis(left, right)

        with (
            patch("almond_axol.robot.axol._GRIPPER_CALIB_SETTLE", 0.0),
            patch("almond_axol.robot.mantis._OPEN_FULLY_POLL_S", 0.0),
        ):
            await robot.open_grippers()

        for arm, motor in ((left, left_motor), (right, right_motor)):
            with self.subTest(motor=motor):
                # deferred connect verifies torque-off, then: enable ->
                # impedance sweep to the stop -> position mode at the open stop
                # -> release.
                self.assertEqual(motor.calls[0], "disable")
                self.assertEqual(motor.calls[1], "enable")
                self.assertEqual(motor.calls[2], ("mode", ControlMode.IMPEDANCE))
                sweeps = [c for c in motor.calls if c == "sweep"]
                self.assertEqual(len(sweeps), 2)
                self.assertIn(("mode", ControlMode.POSITION_FORCE), motor.calls)
                pf = [c for c in motor.calls if isinstance(c, tuple) and c[0] == "pf"]
                self.assertTrue(pf)
                self.assertTrue(all(abs(c[1] - arm._open_pos) < 1e-9 for c in pf))
                self.assertEqual(motor.calls[-1], "disable")
                self.assertTrue(arm.is_calibrated)
                self.assertFalse(arm.is_enabled)
                self.assertGreaterEqual(float(arm.positions[-1]), 0.95)
        self.assertTrue(robot._connected)
        self.assertFalse(robot._shutdown_pending)

        # The first take's enable is now instant: no second calibration sweep.
        await robot.enable_grippers()
        for motor in (left_motor, right_motor):
            self.assertEqual(len([c for c in motor.calls if c == "sweep"]), 2)
        self.assertTrue(left.is_enabled and right.is_enabled)

    async def test_open_failure_releases_both_grippers_and_propagates(self) -> None:
        left = SimpleNamespace(
            preset_open_target=Mock(),
            enable=AsyncMock(),
            open_fully=AsyncMock(side_effect=MotorError("left jaws jammed")),
            force_disable=AsyncMock(),
        )
        right = SimpleNamespace(
            preset_open_target=Mock(),
            enable=AsyncMock(),
            open_fully=AsyncMock(),
            force_disable=AsyncMock(),
        )
        robot = _mantis(left, right)  # type: ignore[arg-type]
        robot._connected = True

        with self.assertRaisesRegex(MotorError, "left jaws jammed"):
            await robot.open_grippers()

        left.preset_open_target.assert_called_once_with()
        right.preset_open_target.assert_called_once_with()
        left.force_disable.assert_awaited_once_with()
        right.force_disable.assert_awaited_once_with()
        self.assertFalse(robot._shutdown_pending)

    async def test_cancel_mid_sweep_releases_both_grippers(self) -> None:
        started = asyncio.Event()

        async def slow_enable() -> None:
            started.set()
            await asyncio.sleep(60)

        left = SimpleNamespace(
            preset_open_target=Mock(),
            enable=AsyncMock(side_effect=slow_enable),
            open_fully=AsyncMock(),
            force_disable=AsyncMock(),
        )
        right = SimpleNamespace(
            preset_open_target=Mock(),
            enable=AsyncMock(),
            open_fully=AsyncMock(),
            force_disable=AsyncMock(),
        )
        robot = _mantis(left, right)  # type: ignore[arg-type]
        robot._connected = True

        task = asyncio.ensure_future(robot.open_grippers())
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        left.force_disable.assert_awaited_once_with()
        right.force_disable.assert_awaited_once_with()
        left.open_fully.assert_not_awaited()
        self.assertFalse(robot._lifecycle_lock.locked())


class _FakeRobotLoop:
    """A robot with a live event loop thread, like AxolRobot after connect()."""

    def __init__(self) -> None:
        self.event_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.event_loop.run_forever, daemon=True)
        self._thread.start()
        self.opened = 0
        self.cancelled = 0
        self.hang = False
        self.started = threading.Event()
        self.error: BaseException | None = None

    async def open_grippers_async(self) -> None:
        self.started.set()
        if self.error is not None:
            raise self.error
        try:
            if self.hang:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        self.opened += 1

    def close(self) -> None:
        self.event_loop.call_soon_threadsafe(self.event_loop.stop)
        self._thread.join(timeout=2)
        self.event_loop.close()


class CollectDataPreopenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = _FakeRobotLoop()
        self.addCleanup(self.robot.close)
        self.announced: list[str] = []

    def test_preopen_runs_on_the_robot_loop_and_announces(self) -> None:
        collect_data._preopen_mantis_grippers(
            self.robot, threading.Event(), self.announced.append
        )
        self.assertEqual(self.robot.opened, 1)
        self.assertEqual(self.announced, ["Opening Mantis grippers."])

    def test_preopen_without_a_stop_event(self) -> None:
        collect_data._preopen_mantis_grippers(self.robot, None, self.announced.append)
        self.assertEqual(self.robot.opened, 1)

    def test_stop_during_preopen_cancels_the_move_and_returns(self) -> None:
        self.robot.hang = True
        stop = threading.Event()

        def stop_once_the_sweep_is_running() -> None:
            self.robot.started.wait(timeout=2)
            stop.set()

        threading.Thread(target=stop_once_the_sweep_is_running, daemon=True).start()
        with patch.object(collect_data, "_PREOPEN_POLL_S", 0.001):
            collect_data._preopen_mantis_grippers(
                self.robot, stop, self.announced.append
            )
        # The move was interrupted (its CancelledError cleanup ran) and the
        # helper returned normally so the session unwinds via its stop path.
        self.assertEqual(self.robot.cancelled, 1)
        self.assertEqual(self.robot.opened, 0)

    def test_hardware_failure_propagates(self) -> None:
        self.robot.error = MotorError("no hard stop was detected")
        with self.assertRaisesRegex(MotorError, "no hard stop"):
            collect_data._preopen_mantis_grippers(
                self.robot, threading.Event(), self.announced.append
            )


if __name__ == "__main__":
    unittest.main()
