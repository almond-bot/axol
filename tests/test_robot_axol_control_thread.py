"""AxolRobot's event-loop thread is the control thread: realtime core + FIFO."""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

from almond_axol.lerobot.robot import robot_axol
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.utils import affinity


class AxolRobotControlThreadTest(unittest.TestCase):
    def test_event_loop_thread_enters_the_control_role_first(self) -> None:
        robot = AxolRobot.__new__(AxolRobot)
        robot._axol = None
        robot._connect_future = None
        robot._disconnect_future = None
        robot._loop_thread = None
        robot._loop = None
        robot._fk = object()
        robot.cameras = {}
        robot.config = type("Cfg", (), {"observe_cartesian": False})()

        seen: dict[str, str | None] = {}

        def enter() -> bool:
            seen["thread"] = threading.current_thread().name
            return True

        with (
            patch.object(
                robot_axol.affinity, "enter_control_thread", side_effect=enter
            ),
            patch.object(AxolRobot, "_connect_async", new=AsyncMock()),
        ):
            AxolRobot.connect(robot)
            try:
                # The claim happened on the loop thread itself, not the caller.
                self.assertEqual(seen.get("thread"), "axol-event-loop")
                self.assertIsInstance(robot.event_loop, asyncio.AbstractEventLoop)
            finally:
                robot._loop.call_soon_threadsafe(robot._loop.stop)
                robot._loop_thread.join(timeout=2.0)
                robot._loop.close()

    def test_refused_realtime_scheduling_surfaces_from_connect(self) -> None:
        # The claim happens on the loop thread, so a refusal there used to kill
        # that thread before run_forever() and surface 30 s later as a connect
        # timeout naming nothing. It has to come back as itself.
        robot = AxolRobot.__new__(AxolRobot)
        robot._axol = None
        robot._connect_future = None
        robot._disconnect_future = None
        robot._loop_thread = None
        robot._loop = None
        robot._fk = object()
        robot.cameras = {}
        robot.config = type("Cfg", (), {"observe_cartesian": False})()

        refusal = affinity.ControlSchedulingError("no rtprio grant")
        with (
            patch.object(
                robot_axol.affinity, "enter_control_thread", side_effect=refusal
            ),
            patch.object(AxolRobot, "_connect_async", new=AsyncMock()) as connect_async,
        ):
            with self.assertRaises(affinity.ControlSchedulingError):
                AxolRobot.connect(robot)

        # Nothing was scheduled onto a loop that will never run, and the robot
        # is left retryable once the operator has fixed the grant.
        connect_async.assert_not_awaited()
        self.assertIsNone(robot._loop)
        self.assertIsNone(robot._loop_thread)


if __name__ == "__main__":
    unittest.main()
