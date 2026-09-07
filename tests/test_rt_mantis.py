"""``RtMantis``: the Mantis grippers driven through the Rust realtime core.

Per take the core is armed on the gripper buses (Python bring-up on the quiet
bus, hand-over, arm, first target through the core) and disarmed at the end
(core disables, Python verifies). While armed no gripper command touches the
Python bus path.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.motor import ControlMode
from almond_axol.robot.axol import GRIPPER_TRAVEL
from almond_axol.robot.mantis import Mantis, MantisGripperArm
from almond_axol.rt.link import RtLinkError
from almond_axol.rt.mantis import RtMantis


class _FakeGripperMotor:
    """A Damiao gripper stand-in on the Python (maintenance) bus path."""

    def __init__(self, *, start: float = 1.25) -> None:
        self.position_value = start
        self.has_position = False
        self.calls: list[object] = []
        self._position: float | None = None
        self._velocity: float | None = None
        self._torque: float | None = None
        self._feedback_ts: float | None = None

    async def enable(self) -> None:
        self.calls.append("enable")

    async def disable(self) -> None:
        self.calls.append("disable")

    async def set_control_mode(self, mode: ControlMode) -> None:
        self.calls.append(("mode", mode))

    async def get_position(self) -> float:
        self.calls.append("read")
        return self.position_value

    async def set_position_force(self, raw: float, speed: float, torque: float) -> None:
        self.calls.append(("pf", raw))
        self.position_value = raw
        self.has_position = True

    async def start_telemetry(self, hz: float, *, torque: bool = False) -> None:
        self.calls.append(("telemetry", hz))

    async def stop_telemetry(self) -> None:
        self.calls.append("stop_telemetry")

    @property
    def position(self) -> float:
        if self._position is not None:
            return self._position
        return self.position_value

    @property
    def torque(self) -> float:
        return self._torque if self._torque is not None else 0.0


class _FakeBus:
    def __init__(self, channel: str, log: list[str]) -> None:
        self.channel = channel
        self._log = log
        self.open = False

    async def start(self) -> None:
        self._log.append(f"bus.start {self.channel}")
        self.open = True

    async def close(self) -> None:
        self._log.append(f"bus.close {self.channel}")
        self.open = False


class _FakeLink:
    """Stands in for ``RtLink``: acks the protocol, echoes feedback on targets."""

    instances: list[_FakeLink] = []
    fail_on_arm = False

    def __init__(self, binary: str | None = None, trace_prefix: str | None = None):
        self.trace_prefix = trace_prefix
        self.log: list[str] = []
        self.config: str | None = None
        self.targets: list[tuple[int, int, list[tuple[float, ...]]]] = []
        self.on_feedback = None
        self.fault: str | None = None
        self.limp: str | None = None
        self.armed = False
        self.closed = False
        self.recording: list[bool] = []
        self._proc = None
        _FakeLink.instances.append(self)

    async def start(self) -> None:
        self.log.append("start")

    async def configure(self, text: str) -> None:
        self.log.append("configure")
        self.config = text

    async def prep(self) -> None:
        self.log.append("prep")

    async def arm(self) -> None:
        self.log.append("arm")
        if _FakeLink.fail_on_arm:
            raise RtLinkError("axol-rt: fault: bring-up failed")
        self.armed = True

    async def disarm(self) -> None:
        self.log.append("disarm")
        self.armed = False

    async def close(self) -> None:
        self.log.append("close")
        self.closed = True

    def send_target(self, side: int, seq: int, cmds: list[tuple[float, ...]]) -> None:
        if not self.armed:
            raise RtLinkError("axol-rt link is not connected")
        self.log.append(f"target {side}")
        self.targets.append((side, seq, cmds))
        if self.on_feedback is not None:
            raw = cmds[7][0]
            self.on_feedback(side, {7: (raw, 0.0, 0.1, time.time())})

    def set_recording_engaged(self, engaged: bool) -> None:
        self.recording.append(engaged)


def _arm(motor: _FakeGripperMotor, channel: str, log: list[str]) -> MantisGripperArm:
    bus = _FakeBus(channel, log)
    with patch("almond_axol.robot.mantis.Motor", return_value=motor):
        return MantisGripperArm(bus, SimpleNamespace(max_speed=10.0, torque_limit=0.5))


def _mantis(
    left: MantisGripperArm, right: MantisGripperArm, *, defer: bool = True
) -> Mantis:
    robot = object.__new__(Mantis)
    robot.left = left
    robot.right = right
    robot._left_bus = left._bus
    robot._right_bus = right._bus
    robot._defer_gripper_enable = defer
    robot._connected = False
    robot._shutdown_pending = False
    robot._telemetry_settings = None
    robot._lifecycle_lock = asyncio.Lock()
    return robot


def _calibrated(arm: MantisGripperArm) -> None:
    arm._calibrated = True
    arm._open_pos = 1.0
    arm._closed_pos = 1.0 + GRIPPER_TRAVEL


class RtMantisTakeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeLink.instances = []
        _FakeLink.fail_on_arm = False
        self.log: list[str] = []
        self.left_motor = _FakeGripperMotor()
        self.right_motor = _FakeGripperMotor()
        self.left = _arm(self.left_motor, "can_mantis_l", self.log)
        self.right = _arm(self.right_motor, "can_mantis_r", self.log)
        _calibrated(self.left)
        _calibrated(self.right)
        self.mantis = _mantis(self.left, self.right)
        patcher = patch("almond_axol.rt.mantis.RtLink", _FakeLink)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rt = RtMantis(self.mantis)

    async def test_connect_opens_buses_and_verifies_torque_off_without_a_core(
        self,
    ) -> None:
        await self.rt.enable()

        self.assertEqual(_FakeLink.instances, [])
        self.assertFalse(self.rt.armed)
        self.assertTrue(self.left._bus.open and self.right._bus.open)
        # Deferred mode establishes torque-off on connect.
        self.assertIn("disable", self.left_motor.calls)
        self.assertIn("disable", self.right_motor.calls)

    async def test_enable_grippers_hands_the_buses_to_the_core(self) -> None:
        await self.rt.connect()
        self.log.clear()
        self.left_motor.calls.clear()
        self.right_motor.calls.clear()

        await self.rt.enable_grippers()

        self.assertTrue(self.rt.armed)
        (link,) = _FakeLink.instances
        self.assertEqual(
            link.config.splitlines()[3:],
            ["gripper 0 can_mantis_l 8", "gripper 1 can_mantis_r 8"],
        )
        # Python bring-up (enable / POSITION_FORCE / first target / read)
        # happens after prep, on the quiet bus.
        self.assertEqual(link.log[:3], ["start", "configure", "prep"])
        bring_up = [c for c in self.left_motor.calls if c != "stop_telemetry"]
        self.assertEqual(
            bring_up[:4],
            ["enable", ("mode", ControlMode.POSITION_FORCE), ("pf", 1.0), "read"],
        )
        # Buses are closed before the core arms; the first core target
        # follows the arm and its feedback fills the caches.
        self.assertEqual(self.log, ["bus.close can_mantis_l", "bus.close can_mantis_r"])
        self.assertEqual(link.log[3:], ["arm", "target 0", "target 1"])
        self.assertTrue(self.left.core_driven and self.right.core_driven)
        self.assertEqual(self.rt._fb_packets, [1, 1])
        self.assertEqual(link.targets[0][2][7][:3], (1.0, 10.0, 0.5))
        self.assertEqual(link.targets[0][2][:7], [(0.0,) * 9] * 7)

    async def test_motion_control_streams_through_the_core_only(self) -> None:
        await self.rt.connect()
        await self.rt.enable_grippers()
        (link,) = _FakeLink.instances
        self.left_motor.calls.clear()
        self.right_motor.calls.clear()
        link.targets.clear()

        q_left = np.zeros(8, dtype=np.float32)
        q_left[-1] = 0.25
        q_right = np.zeros(8, dtype=np.float32)
        q_right[-1] = 0.75
        await self.rt.motion_control(left=q_left, right=q_right)

        self.assertEqual(self.left_motor.calls, [])
        self.assertEqual(self.right_motor.calls, [])
        self.assertEqual([t[0] for t in link.targets], [0, 1])
        left_raw = link.targets[0][2][7][0]
        right_raw = link.targets[1][2][7][0]
        self.assertAlmostEqual(left_raw, 1.0 + GRIPPER_TRAVEL * 0.75, places=5)
        self.assertAlmostEqual(right_raw, 1.0 + GRIPPER_TRAVEL * 0.25, places=5)
        # The core's feedback is what get_positions reports while armed —
        # no Python read touches the bus.
        pos_left, pos_right = await self.rt.get_positions()
        self.assertAlmostEqual(float(pos_left[-1]), 0.25, places=4)
        self.assertAlmostEqual(float(pos_right[-1]), 0.75, places=4)
        self.assertNotIn("read", self.left_motor.calls)
        # Virtual arm joints echo the latched targets.
        np.testing.assert_allclose(pos_left[:7], q_left[:7])
        newest_ts = self.rt._state_history[-1][0]
        snapshot = self.rt.state_nearest(newest_ts, timeout=0.0)
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertAlmostEqual(float(snapshot[0][-1]), 0.25, places=4)
        self.assertAlmostEqual(float(snapshot[1][-1]), 0.75, places=4)

    async def test_disable_grippers_disarms_then_verifies_torque_off(self) -> None:
        await self.rt.connect()
        await self.rt.enable_grippers()
        (link,) = _FakeLink.instances
        self.log.clear()
        self.left_motor.calls.clear()
        self.right_motor.calls.clear()

        await self.rt.disable_grippers()

        self.assertFalse(self.rt.armed)
        self.assertEqual(link.log[-2:], ["disarm", "close"])
        self.assertEqual(self.log, ["bus.start can_mantis_l", "bus.start can_mantis_r"])
        self.assertEqual(self.left_motor.calls, ["disable"])
        self.assertEqual(self.right_motor.calls, ["disable"])
        self.assertFalse(self.left.core_driven or self.right.core_driven)
        self.assertFalse(self.left.is_enabled or self.right.is_enabled)
        # Between takes motion_control only latches.
        q = np.zeros(8, dtype=np.float32)
        await self.rt.motion_control(left=q, right=q)
        self.assertEqual(self.left_motor.calls, ["disable"])
        self.assertEqual(len(link.targets), 2)

    async def test_second_take_uses_a_fresh_core_and_skips_calibration(self) -> None:
        await self.rt.connect()
        await self.rt.enable_grippers()
        await self.rt.disable_grippers()
        await self.rt.enable_grippers()

        self.assertEqual(len(_FakeLink.instances), 2)
        self.assertTrue(_FakeLink.instances[0].closed)
        self.assertTrue(self.rt.armed)
        self.assertNotIn(("mode", ControlMode.IMPEDANCE), self.left_motor.calls)
        await self.rt.disable()
        self.assertFalse(self.rt.armed)
        self.assertFalse(self.left._bus.open or self.right._bus.open)

    async def test_failed_arm_rolls_back_to_torque_off_and_python_buses(self) -> None:
        await self.rt.connect()
        _FakeLink.fail_on_arm = True
        self.left_motor.calls.clear()

        with self.assertRaises(RtLinkError):
            await self.rt.enable_grippers()

        (link,) = _FakeLink.instances
        self.assertFalse(self.rt.armed)
        self.assertEqual(link.log[-3:], ["arm", "disarm", "close"])
        self.assertTrue(self.left._bus.open and self.right._bus.open)
        self.assertEqual(self.left_motor.calls[-1], "disable")
        self.assertFalse(self.left.core_driven)
        self.assertFalse(self.left.is_enabled)

    async def test_fault_during_take_still_disables_from_python(self) -> None:
        await self.rt.connect()
        await self.rt.enable_grippers()
        (link,) = _FakeLink.instances
        link.fault = "fault: can_mantis_l: motor silent"
        self.left_motor.calls.clear()

        await self.rt.disable_grippers()

        # A Mantis gripper is always safe to release: unlike the arms, a
        # fault never leaves it holding.
        self.assertEqual(self.left_motor.calls, ["disable"])
        self.assertTrue(self.left._bus.open)

    async def test_open_grippers_runs_through_the_core_and_releases(self) -> None:
        await self.rt.connect()
        self.left._gripper_target = 0.2
        self.right._gripper_target = 0.3

        with patch("almond_axol.robot.mantis._OPEN_FULLY_POLL_S", 0.0):
            await self.rt.open_grippers()

        (link,) = _FakeLink.instances
        self.assertFalse(self.rt.armed)
        self.assertTrue(link.closed)
        # Every core target commanded the open stop (raw = open_pos).
        self.assertTrue(all(t[2][7][0] == 1.0 for t in link.targets))
        self.assertEqual(self.left_motor.calls[-1], "disable")
        self.assertEqual(self.right_motor.calls[-1], "disable")
        self.assertTrue(self.left._bus.open and self.right._bus.open)

    async def test_recording_gate_reaches_the_armed_core(self) -> None:
        rt = RtMantis(self.mantis, record="/tmp/axol-rt-mantis-test/trace")
        await rt.connect()
        await rt.enable_grippers()
        (link,) = _FakeLink.instances
        self.assertTrue(link.trace_prefix.endswith("_take1"))

        rt.set_recording_engaged(True)
        rt.set_recording_engaged(True)
        await rt.disable_grippers()

        self.assertEqual(link.recording, [True, False])

    async def test_non_deferred_enable_arms_immediately(self) -> None:
        mantis = _mantis(self.left, self.right, defer=False)
        rt = RtMantis(mantis)

        await rt.enable()

        self.assertTrue(rt.armed)
        self.assertEqual(len(_FakeLink.instances), 1)
        await rt.disable()
        self.assertFalse(rt.armed)


class RtMantisSurfaceTest(unittest.IsolatedAsyncioTestCase):
    async def test_axol_surface_stubs(self) -> None:
        left = _arm(_FakeGripperMotor(), "can_mantis_l", [])
        right = _arm(_FakeGripperMotor(), "can_mantis_r", [])
        rt = RtMantis(_mantis(left, right))

        self.assertIsNone(rt.fault)
        self.assertIsNone(rt.limp)
        self.assertEqual(rt.torque_residuals(), (None, None))
        self.assertFalse(rt.records_measurements_at_control_rate)
        rt.reset_command_state()
        rt.reset_gravity_hold()
        with self.assertRaises(NotImplementedError):
            await rt.gravity_compensate()
        self.assertIsNone(rt.state_nearest(time.perf_counter(), timeout=0.0))

    def test_robot_mantis_builds_an_rt_mantis(self) -> None:
        from almond_axol.lerobot.robot.config_mantis import MantisRobotConfig
        from almond_axol.lerobot.robot.robot_mantis import MantisRobot

        config = MantisRobotConfig(cameras={})
        with patch.object(MantisRobot, "_build_cameras", return_value=({}, [])):
            robot = MantisRobot(config, defer_gripper_enable=True)
        hardware = robot._build_hardware()
        self.assertIsInstance(hardware, RtMantis)
        self.assertIsInstance(hardware.robot, Mantis)
        self.assertTrue(hardware.robot._defer_gripper_enable)


if __name__ == "__main__":
    unittest.main()
