"""The idle links reopen a bus that died while the robot was switched off.

Switching Axol (or Jelly) off kills the control panel's idle connection: the
first frame nothing acknowledges makes the Rust proxy report a dead bus and
exit, and an adapter that loses power takes its interface with it. The link
stays ``connected`` and reports every motor unreachable, which is right while
the power is off. These pin that it also comes back by itself once the power
returns, instead of waiting for the operator to disconnect and reconnect.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import unittest
from typing import Any
from unittest.mock import patch

from almond_axol.constants import Joint
from almond_axol.motor import bus as bus_mod
from almond_axol.motor.bus import CanBus
from almond_axol.serve import jelly_link, robot_link
from almond_axol.serve.jelly_link import JellyLink, _WheelsDevice
from almond_axol.serve.robot_link import (
    STATE_BUSY,
    STATE_CONNECTED,
    RobotLink,
    _ArmLink,
)

_SETUP = "almond_axol.cli.can.setup"


class _FakeObserver:
    def __init__(self, commanded: set[Joint] | None = None) -> None:
        self.commanded = commanded or set()

    def commanded_joints(self, _fresh_s: float) -> set[Joint]:
        return self.commanded


class _Arm:
    """An arm whose command bus is dead until :meth:`open` replaces it."""

    def __init__(
        self,
        channel: str,
        *,
        dead: bool = True,
        answers: bool = True,
        observer: _FakeObserver | None = None,
    ) -> None:
        self.channel = channel
        self.side = "left"
        self.bus_dead = dead
        self.answers = answers
        self.observer = observer
        self.probes = 0
        self.closed = 0
        self.opened: list[list[Joint]] = []

    async def probe(self, _joints: list[Joint]) -> bool:
        self.probes += 1
        return self.answers

    async def close(self) -> None:
        self.closed += 1

    async def open(self, joints: list[Joint]) -> None:
        self.opened.append(joints)
        self.bus_dead = False

    async def sync_observer_ranges(self) -> None:
        pass


def _link(arms: list[_Arm], state: str = STATE_CONNECTED) -> RobotLink:
    link = object.__new__(RobotLink)
    link._lock = threading.Lock()
    link._lifecycle_lock = asyncio.Lock()
    link._arms = arms  # type: ignore[assignment]
    link._state = state
    link._profile = "axol"
    link._active_joints = [Joint.SHOULDER_1, Joint.SHOULDER_2]
    link._sync_task = None
    return link


class CommandWatchLoopTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.object(robot_link, "_COMMAND_REOPEN_INTERVAL_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _run_ticks(
        self,
        link: RobotLink,
        *,
        iface_up: bool = True,
        backlog: int | None = 0,
        ticks: int = 20,
    ) -> None:
        with (
            patch(f"{_SETUP}.iface_up", return_value=iface_up),
            patch(f"{_SETUP}.tx_backlog", return_value=backlog),
        ):
            task = asyncio.ensure_future(link._command_watch_loop())
            for _ in range(ticks):
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if link._sync_task is not None:
                await asyncio.gather(link._sync_task, return_exceptions=True)

    async def test_reopens_a_dead_bus_once_a_motor_answers(self) -> None:
        dead = _Arm("can_l")
        live = _Arm("can_r", dead=False)
        link = _link([dead, live])
        await self._run_ticks(link)
        self.assertEqual(dead.closed, 1, "the dead proxy is reaped first")
        self.assertEqual(dead.opened, [[Joint.SHOULDER_1, Joint.SHOULDER_2]])
        self.assertFalse(dead.bus_dead)
        self.assertEqual(live.probes, 0)
        self.assertEqual(live.opened, [])

    async def test_keeps_waiting_while_the_robot_is_off(self) -> None:
        arm = _Arm("can_l", answers=False)
        link = _link([arm])
        await self._run_ticks(link)
        self.assertGreater(arm.probes, 0)
        self.assertEqual(arm.opened, [])
        self.assertTrue(arm.bus_dead)

    async def test_waits_while_the_interface_is_down(self) -> None:
        arm = _Arm("can_l")
        link = _link([arm])
        await self._run_ticks(link, iface_up=False)
        self.assertEqual(arm.probes, 0)
        self.assertEqual(arm.opened, [])

    async def test_does_not_queue_a_probe_behind_unsent_frames(self) -> None:
        arm = _Arm("can_l")
        link = _link([arm])
        await self._run_ticks(link, backlog=3)
        self.assertEqual(arm.probes, 0)
        self.assertEqual(arm.opened, [])

    async def test_leaves_a_released_link_alone(self) -> None:
        arm = _Arm("can_l")
        link = _link([arm], state=STATE_BUSY)
        await self._run_ticks(link)
        self.assertEqual(arm.probes, 0)
        self.assertEqual(arm.opened, [])

    async def test_a_release_during_the_probe_wins(self) -> None:
        arm = _Arm("can_l")
        link = _link([arm])

        async def probe(_joints: list[Joint]) -> bool:
            link._state = STATE_BUSY  # a task took the buses meanwhile
            return True

        arm.probe = probe  # type: ignore[method-assign]
        await self._run_ticks(link)
        self.assertEqual(arm.opened, [])

    async def test_reopens_without_probing_an_arm_another_process_drives(
        self,
    ) -> None:
        arm = _Arm("can_l", observer=_FakeObserver({Joint.SHOULDER_1}))
        link = _link([arm])
        await self._run_ticks(link)
        self.assertEqual(arm.probes, 0)
        self.assertEqual(len(arm.opened), 1)


class _ProbeBus:
    """A short-lived probe bus whose proxy dies after ``frames`` sends."""

    instances: list[_ProbeBus] = []

    def __init__(self, channel: str, **kwargs: Any) -> None:
        self.channel = channel
        self.kwargs = kwargs
        self.is_open = False
        self.closed = False
        _ProbeBus.instances.append(self)

    async def __aenter__(self) -> _ProbeBus:
        self.is_open = True
        return self

    async def __aexit__(self, *_: object) -> None:
        self.closed = True
        self.is_open = False


class _ProbeMotor:
    answering: set[Joint] = set()
    asked: list[Joint] = []

    def __init__(self, bus: _ProbeBus, joint: Joint) -> None:
        self.bus = bus
        self.joint = joint

    async def get_error_code(self) -> Any:
        _ProbeMotor.asked.append(self.joint)
        if self.joint in _ProbeMotor.answering:
            return "OK"
        # Hub firmware 2.01+: the first unanswered frame kills the proxy.
        self.bus.is_open = False
        raise TimeoutError("no reply")


class ArmProbeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _ProbeBus.instances = []
        _ProbeMotor.answering = set()
        _ProbeMotor.asked = []
        for target, value in (("CanBus", _ProbeBus), ("Motor", _ProbeMotor)):
            patcher = patch.object(robot_link, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_powered_off_arm_costs_one_frame(self) -> None:
        arm = _ArmLink("can_l", "left")
        joints = [Joint.SHOULDER_1, Joint.SHOULDER_2, Joint.ELBOW]
        self.assertFalse(await arm.probe(joints))
        self.assertEqual(_ProbeMotor.asked, [Joint.SHOULDER_1])
        (bus,) = _ProbeBus.instances
        self.assertTrue(bus.closed)
        self.assertEqual(bus.kwargs, {"publishes_stall": False, "quiet": True})

    async def test_any_answering_motor_counts(self) -> None:
        _ProbeMotor.answering = {Joint.SHOULDER_1}
        arm = _ArmLink("can_l", "left")
        self.assertTrue(await arm.probe([Joint.SHOULDER_1, Joint.SHOULDER_2]))

    async def test_bus_dead_only_after_the_proxy_died(self) -> None:
        arm = _ArmLink("can_l", "left")
        self.assertFalse(arm.bus_dead, "never opened")

        class _Bus:
            is_open = True

            async def close(self) -> None:
                self.is_open = False

        arm._bus = _Bus()  # type: ignore[assignment]
        self.assertFalse(arm.bus_dead)
        arm._bus.is_open = False  # type: ignore[misc]
        self.assertTrue(arm.bus_dead)
        await arm.close()
        self.assertFalse(arm.bus_dead, "closed on purpose")


class QuietBusTest(unittest.TestCase):
    def test_quiet_bus_logs_its_lifecycle_at_debug(self) -> None:
        bus = CanBus("can_quiet_test", publishes_stall=False, quiet=True)
        with self.assertLogs(bus_mod._logger, level=logging.DEBUG) as logs:
            bus._mark_stalled("no node ACKing")
        self.assertTrue(all(r.levelno == logging.DEBUG for r in logs.records))

    def test_default_bus_still_warns(self) -> None:
        bus = CanBus("can_loud_test", publishes_stall=False)
        with self.assertLogs(bus_mod._logger, level=logging.WARNING):
            bus._mark_stalled("no node ACKing")


class _DeadWheels:
    """Stands in for the wheels device: bus dead until reopened."""

    name = "wheels"
    channel = "can_alm_axol_b"

    def __init__(self, *, answers: bool) -> None:
        self.bus_dead = True
        self.answers = answers
        self.probes = 0
        self.pings = 0
        self.opened = 0
        self.closed = 0

    async def probe(self) -> bool:
        self.probes += 1
        return self.answers

    async def ping(self) -> None:
        self.pings += 1

    async def close(self) -> None:
        self.closed += 1

    async def open(self) -> None:
        self.opened += 1
        self.bus_dead = False


class JellyReopenTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for name in ("_REOPEN_INTERVAL_S", "_PING_INTERVAL_S"):
            patcher = patch.object(jelly_link, name, 0.0)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _link(self, device: _DeadWheels, state: str = STATE_CONNECTED) -> Any:
        link = object.__new__(JellyLink)
        link._lock = threading.Lock()
        link._lifecycle_lock = asyncio.Lock()
        link._loop = asyncio.get_running_loop()
        dev_link = jelly_link._DeviceLink(device)  # type: ignore[arg-type]
        dev_link.state = state
        dev_link.bus_may_be_open = True
        return link, dev_link

    async def _run_ticks(
        self, link: JellyLink, dev_link: Any, *, iface_up: bool = True
    ) -> None:
        with (
            patch(f"{_SETUP}.iface_up", return_value=iface_up),
            patch(f"{_SETUP}.tx_backlog", return_value=0),
        ):
            task = asyncio.ensure_future(link._ping_loop(dev_link))
            for _ in range(20):
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_reopens_and_resumes_pinging_once_jelly_answers(self) -> None:
        device = _DeadWheels(answers=True)
        link, dev_link = self._link(device)
        await self._run_ticks(link, dev_link)
        self.assertEqual((device.closed, device.opened), (1, 1))
        self.assertGreater(device.pings, 0)
        self.assertIsNotNone(dev_link.last_ping)

    async def test_keeps_waiting_while_jelly_is_off(self) -> None:
        device = _DeadWheels(answers=False)
        link, dev_link = self._link(device)
        await self._run_ticks(link, dev_link)
        self.assertGreater(device.probes, 0)
        self.assertEqual((device.opened, device.pings), (0, 0))

    async def test_waits_while_the_interface_is_down(self) -> None:
        device = _DeadWheels(answers=True)
        link, dev_link = self._link(device)
        await self._run_ticks(link, dev_link, iface_up=False)
        self.assertEqual((device.probes, device.opened), (0, 0))

    async def test_leaves_a_released_device_alone(self) -> None:
        device = _DeadWheels(answers=True)
        link, dev_link = self._link(device, state=STATE_BUSY)
        await self._run_ticks(link, dev_link)
        self.assertEqual(device.opened, 0)


class WheelsProbeTest(unittest.IsolatedAsyncioTestCase):
    async def test_probe_uses_a_quiet_bus_and_stops_when_it_dies(self) -> None:
        _ProbeBus.instances = []
        asked: list[int] = []

        class _Wheel:
            def __init__(self, bus: _ProbeBus, motor_id: int, **_: Any) -> None:
                self.bus = bus
                self.motor_id = motor_id

            async def get_error_code(self) -> Any:
                asked.append(self.motor_id)
                self.bus.is_open = False
                raise TimeoutError("no reply")

        with (
            patch.object(jelly_link, "CanBus", _ProbeBus),
            patch.object(jelly_link, "DamiaoMotor", _Wheel),
        ):
            self.assertFalse(await _WheelsDevice().probe())
        self.assertEqual(len(asked), 1)
        (bus,) = _ProbeBus.instances
        self.assertEqual(bus.kwargs, {"publishes_stall": False, "quiet": True})


if __name__ == "__main__":
    unittest.main()
