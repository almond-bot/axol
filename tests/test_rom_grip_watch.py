"""ROM soak grip safety: a gentle grasp, a let-go watch, the robot's gripper type.

A full-robot ``diag.rom-enable`` run clamps each gripper on an item for two
hours. The jaw never reaches its closed target, so the motor sits at the grasp
torque cap the whole time. These tests pin the three guards around that hold:
the cap itself, the watch that stops the soak when a gripper cuts out, and the
gripper type coming from the robot's settings (as it does for rom.disable).
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from almond_axol.constants import Joint
from almond_axol.diagnostics.rom import enable as rom


class GraspTorqueTest(unittest.TestCase):
    def test_grasp_cap_reads_back_under_the_continuous_rating(self) -> None:
        # The cap reads back ~1.9x higher than commanded on the DM-J4310,
        # which is rated 3.5 Nm continuous.
        self.assertLessEqual(rom.GRIPPER_TORQUE_LIMIT * 1.9, 2.5)


class GripWatchTest(unittest.TestCase):
    def test_a_light_hold_is_not_watched(self) -> None:
        watch = rom.GripWatch({"left": 0.2, "right": 2.0})
        self.assertEqual(set(watch.hold_nm), {"right"})
        for t in range(10):
            watch.update("left", 0.0, float(t))  # never raises

    def test_nothing_held_means_nothing_watched(self) -> None:
        self.assertFalse(rom.GripWatch({"left": 0.1}).watching)

    def test_torque_collapse_for_two_seconds_is_a_lost_grip(self) -> None:
        watch = rom.GripWatch({"left": 2.3})
        watch.update("left", 2.3, 0.0)
        watch.update("left", 0.05, 1.0)
        watch.update("left", 0.05, 2.9)
        with self.assertRaises(rom.GripLost) as ctx:
            watch.update("left", 0.05, 3.0)
        self.assertEqual(ctx.exception.side, "left")
        self.assertIn("Left gripper let go", str(ctx.exception))

    def test_a_short_dip_is_forgiven(self) -> None:
        watch = rom.GripWatch({"right": 2.0})
        watch.update("right", 0.1, 0.0)
        watch.update("right", 0.1, 1.5)
        watch.update("right", 1.2, 1.6)  # back above 25 % of the hold
        watch.update("right", 0.1, 2.0)
        watch.update("right", 0.1, 3.9)  # low again, but only 1.9 s

    def test_sign_does_not_matter(self) -> None:
        # The parcel right jaw closes toward negative angles: its hold
        # torque is negative.
        watch = rom.GripWatch({"right": 2.0})
        for t in range(30):
            watch.update("right", -1.9, t * 0.1)

    def test_non_finite_readings_are_ignored(self) -> None:
        watch = rom.GripWatch({"left": 2.0})
        for t in range(50):
            watch.update("left", float("nan"), t * 0.1)


class RunWatchedTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_watch_just_runs_the_soak(self) -> None:
        async def soak() -> int:
            return 7

        self.assertEqual(await rom.run_watched(soak(), None), 7)

    async def test_finished_soak_stops_the_watch(self) -> None:
        stopped = asyncio.Event()

        async def soak() -> str:
            await asyncio.sleep(0.01)
            return "done"

        async def watch() -> None:
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        self.assertEqual(await rom.run_watched(soak(), watch()), "done")
        self.assertTrue(stopped.is_set())

    async def test_lost_grip_cancels_the_soak_before_raising(self) -> None:
        soak_cancelled = asyncio.Event()

        async def soak() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                soak_cancelled.set()
                raise

        async def watch() -> None:
            await asyncio.sleep(0.01)
            raise rom.GripLost("right", 3.8, 0.0)

        with self.assertRaises(rom.GripLost):
            await rom.run_watched(soak(), watch())
        self.assertTrue(soak_cancelled.is_set())

    async def test_watch_grips_reads_each_present_gripper(self) -> None:
        gripper_i = rom.JOINT_INDEX[Joint.GRIPPER]
        torques = np.zeros(rom.NUM_JOINTS, dtype=np.float32)

        class _Robot:
            async def get_torques(self) -> tuple[Any, Any]:
                return torques.copy(), None

        watch = rom.GripWatch({"left": 2.0, "right": 2.0})
        torques[gripper_i] = 0.0
        with (
            patch.object(rom, "GRIP_WATCH_SECONDS", 0.05),
            patch.object(rom, "GRIP_WATCH_PERIOD", 0.01),
            self.assertRaises(rom.GripLost) as ctx,
        ):
            await asyncio.wait_for(rom.watch_grips(_Robot(), watch), 1.0)
        self.assertEqual(ctx.exception.side, "left")


class GripperTypeFromSettingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_rom_enable_uses_the_robots_gripper_type(self) -> None:
        seen: dict[str, Any] = {}

        class _Stop(Exception):
            pass

        def fake_axol(*, config: Any, **_: Any) -> Any:
            seen["config"] = config
            raise _Stop

        async def all_joints(_channel: str, joints: Any) -> set[Joint]:
            return set(joints)

        with (
            patch.object(rom, "resolve_gripper", return_value="parcel"),
            patch.object(rom, "probe_bus_joints", all_joints),
            patch.object(rom, "Axol", fake_axol),
            patch("builtins.print"),
            self.assertRaises(_Stop),
        ):
            await rom.run_axol(capture=False)
        self.assertEqual(seen["config"].gripper, "parcel")
        self.assertEqual(
            seen["config"].left.gripper.torque_limit, rom.GRIPPER_TORQUE_LIMIT
        )


class _SoakRobot:
    """Enough of ``rt.Axol`` for a grasp soak, with a gripper that cuts out."""

    def __init__(self) -> None:
        self.limp = None
        self.gripper_nm = {"left": 2.3, "right": 2.3}
        self.disconnected = False
        self.disabled = False

    async def enable(self) -> None:
        pass

    async def get_positions(self) -> tuple[Any, Any]:
        return np.zeros(rom.NUM_JOINTS), np.zeros(rom.NUM_JOINTS)

    async def get_torques(self) -> tuple[Any, Any]:
        gripper_i = rom.JOINT_INDEX[Joint.GRIPPER]
        left = np.zeros(rom.NUM_JOINTS)
        right = np.zeros(rom.NUM_JOINTS)
        left[gripper_i] = self.gripper_nm["left"]
        right[gripper_i] = -self.gripper_nm["right"]
        return left, right

    async def motion_control(self, left: Any, right: Any) -> None:
        pass

    def set_payload(self, *_: Any) -> None:
        pass

    async def disconnect(self) -> None:
        self.disconnected = True

    async def disable(self) -> None:
        self.disabled = True


class GripLostRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_lost_grip_stops_the_soak_returns_home_and_fails(self) -> None:
        robot = _SoakRobot()
        cycles = 0
        went_home = asyncio.Event()

        async def fake_cycle(_robot: Any, left_q: Any, right_q: Any, **_: Any):
            nonlocal cycles
            cycles += 1
            if cycles == 2:
                robot.gripper_nm["right"] = 0.02  # the right gripper cuts out
            await asyncio.sleep(0.05)
            return left_q, right_q

        async def fake_home(_robot: Any) -> None:
            went_home.set()

        async def no_wait(*_: Any, **__: Any) -> None:
            pass

        async def grip(_robot: Any, left_q: Any, right_q: Any, *_: Any):
            return left_q, right_q

        async def all_joints(_channel: str, joints: Any) -> set[Joint]:
            return set(joints)

        with (
            patch.object(rom, "resolve_gripper", return_value="parallel"),
            patch.object(rom, "probe_bus_joints", all_joints),
            patch.object(rom, "Axol", lambda **_: robot),
            patch.object(rom, "sweep_unchecked", no_wait),
            patch.object(rom, "_confirm", no_wait),
            patch.object(rom, "move_grippers", grip),
            patch.object(rom, "hold_pose", no_wait),
            patch.object(rom, "run_rom_cycle", fake_cycle),
            patch.object(rom, "return_home", fake_home),
            patch.object(rom, "GRIP_WATCH_SECONDS", 0.1),
            patch.object(rom, "GRIP_WATCH_PERIOD", 0.01),
            patch("builtins.print"),
            self.assertRaises(SystemExit) as ctx,
        ):
            await rom.run_axol(capture=False, payload_kg=0.0)

        self.assertIn("right gripper let go", str(ctx.exception))
        self.assertTrue(went_home.is_set())
        # The other gripper still holds its item: stay enabled for rom.disable.
        self.assertTrue(robot.disconnected)
        self.assertFalse(robot.disabled)
        self.assertLess(cycles, 6)


if __name__ == "__main__":
    unittest.main()
