"""Software-fault teardown releases ownership without removing arm support."""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.robot.base import HardwareCleanupError
from almond_axol.rt.robot import Axol


class AdapterPreservingDisconnectTest(unittest.TestCase):
    def make_robot(self, *, cameras=None):
        robot = AxolRobot.__new__(AxolRobot)
        runtime = SimpleNamespace(disconnect=AsyncMock(), disable=AsyncMock())
        robot._axol = runtime
        robot.cameras = cameras or {}
        robot._connect_future = None
        robot._disconnect_future = None
        robot._fk = None
        robot._ik = None
        robot._loop = asyncio.new_event_loop()
        robot._loop_thread = threading.Thread(target=robot._loop.run_forever)
        robot._loop_thread.start()

        def clean_loop():
            if robot._loop is not None:
                robot._loop.call_soon_threadsafe(robot._loop.stop)
                robot._loop_thread.join(timeout=2.0)
                robot._loop.close()

        self.addCleanup(clean_loop)
        return robot, runtime

    def test_fault_teardown_preserves_motors_and_closes_resources(self):
        camera = SimpleNamespace(is_connected=True, disconnect=Mock())
        robot, runtime = self.make_robot(cameras={"overhead": camera})

        robot.disconnect_preserving_position()

        runtime.disconnect.assert_awaited_once_with()
        runtime.disable.assert_not_awaited()
        camera.disconnect.assert_called_once_with()
        self.assertIsNone(robot._axol)
        self.assertIsNone(robot._loop)
        self.assertIsNone(robot._loop_thread)

    def test_default_disconnect_still_disables(self):
        robot, runtime = self.make_robot()

        robot.disconnect()

        runtime.disable.assert_awaited_once_with()
        runtime.disconnect.assert_not_awaited()

    def test_camera_failure_does_not_skip_preserving_motor_teardown(self):
        error = RuntimeError("camera close failed")
        camera = SimpleNamespace(is_connected=True, disconnect=Mock(side_effect=error))
        robot, runtime = self.make_robot(cameras={"overhead": camera})

        with self.assertRaises(RuntimeError) as raised:
            robot.disconnect_preserving_position()

        self.assertIs(raised.exception, error)
        runtime.disconnect.assert_awaited_once_with()
        runtime.disable.assert_not_awaited()
        self.assertIsNone(robot._axol)
        self.assertIsNone(robot._loop)

    def test_failure_retains_runtime_and_event_loop_for_preserving_retry(self):
        robot, runtime = self.make_robot()
        runtime.disconnect.side_effect = HardwareCleanupError("core still running")

        with self.assertRaises(HardwareCleanupError):
            robot.disconnect_preserving_position()

        self.assertIs(robot._axol, runtime)
        self.assertTrue(robot._loop_thread.is_alive())
        self.assertIsNone(robot._disconnect_future)
        runtime.disable.assert_not_awaited()

        runtime.disconnect.side_effect = None
        # Generic cleanup retry must retain the earlier preserving intent.
        robot.disconnect()
        self.assertEqual(runtime.disconnect.await_count, 2)
        runtime.disable.assert_not_awaited()
        self.assertIsNone(robot._axol)


class RuntimePreservingDisconnectTest(unittest.IsolatedAsyncioTestCase):
    def make_robot(self, *, armed=True, started=True):
        robot = Axol.__new__(Axol)
        robot._armed = armed
        robot._core_started = started
        robot._rec = None
        robot._robot = SimpleNamespace(
            left=None, right=None, disconnect=AsyncMock(), disable=AsyncMock()
        )
        robot._link = SimpleNamespace(
            on_feedback=Mock(), close=AsyncMock(), disarm=AsyncMock(), _proc=None
        )
        return robot

    async def test_core_close_never_disarms_or_reopens_maintenance(self):
        robot = self.make_robot()

        await robot.disconnect()

        robot._link.close.assert_awaited_once_with()
        robot._link.disarm.assert_not_awaited()
        robot._robot.disconnect.assert_not_awaited()
        robot._robot.disable.assert_not_awaited()
        self.assertFalse(robot._armed)
        self.assertFalse(robot._core_started)

    async def test_link_failure_retains_ownership_and_retry_remains_preserving(self):
        robot = self.make_robot()
        error = RuntimeError("core termination failed")
        robot._link.close.side_effect = error
        robot._link._proc = Mock(poll=Mock(return_value=None))

        with self.assertRaises(HardwareCleanupError) as raised:
            await robot.disconnect()

        self.assertIs(raised.exception.__cause__, error)
        self.assertTrue(robot._armed)
        self.assertTrue(robot._core_started)
        robot._link.disarm.assert_not_awaited()
        robot._robot.disable.assert_not_awaited()

        robot._link.close.side_effect = None
        robot._link._proc.poll.return_value = 0
        # Even a generic disable retry cannot downgrade the preserving fault
        # teardown while ownership was still uncertain.
        await robot.disable()
        self.assertFalse(robot._armed)
        robot._link.disarm.assert_not_awaited()

    async def test_original_process_is_verified_if_close_clears_reference(self):
        robot = self.make_robot()
        process = Mock(poll=Mock(return_value=None))
        robot._link._proc = process

        async def lose_process_reference():
            robot._link._proc = None

        robot._link.close.side_effect = lose_process_reference
        with self.assertRaisesRegex(HardwareCleanupError, "still running"):
            await robot.disconnect()

        self.assertIs(robot._link._proc, process)
        self.assertTrue(robot._armed)
        self.assertTrue(robot._core_started)

    async def test_successful_close_with_live_core_is_still_ownership_failure(self):
        robot = self.make_robot()
        robot._link._proc = Mock(poll=Mock(return_value=None))

        with self.assertRaisesRegex(HardwareCleanupError, "still running"):
            await robot.disconnect()

        self.assertTrue(robot._armed)
        self.assertTrue(robot._core_started)
        robot._link.disarm.assert_not_awaited()
        robot._robot.disconnect.assert_not_awaited()

    async def test_unreadable_process_status_does_not_release_ownership(self):
        robot = self.make_robot()
        robot._link._proc = Mock(poll=Mock(side_effect=OSError("no status")))

        with self.assertRaisesRegex(HardwareCleanupError, "cannot verify"):
            await robot.disconnect()

        self.assertTrue(robot._armed)
        self.assertTrue(robot._core_started)

    async def test_core_started_before_arm_is_closed_without_bus_reacquisition(self):
        robot = self.make_robot(armed=False)

        await robot.disconnect()

        robot._link.close.assert_awaited_once_with()
        robot._robot.disconnect.assert_not_awaited()
        robot._link.disarm.assert_not_awaited()
        self.assertFalse(robot._core_started)

    async def test_maintenance_only_disconnect_closes_without_disabling(self):
        robot = self.make_robot(armed=False, started=False)

        await robot.disconnect()

        robot._robot.disconnect.assert_awaited_once_with()
        robot._robot.disable.assert_not_awaited()
        robot._link.close.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
