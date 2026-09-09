"""Losing motor power mid-operation: classification, lockout, watchdog, telemetry.

Operators cut the motor PSU as an e-stop. Everything here covers what the stack
must do when that happens while an operation owns the CAN buses.
"""

from __future__ import annotations

import asyncio
import errno
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import can
import httpx

from almond_axol.motor import CanBus, MotorError, MotorStatus
from almond_axol.motor.bus import _flush_lock_for_running_loop, _tx_queue_full
from almond_axol.robot.axol import Axol
from almond_axol.robot.base import HardwareCleanupError
from almond_axol.serve import app as app_module
from almond_axol.serve.manager import Session
from almond_axol.serve.robot_link import (
    STATE_BUSY,
    STATE_CONNECTED,
    RobotLink,
    motor_faults,
)
from almond_axol.serve.runner import STALL_STOP_ERROR, OperationRunner

# The serve API doubles (settings store, session manager, updater) already exist
# for the reservation tests; reuse them rather than growing a second set.
from .test_serve_session_reservation import _Manager, _Settings, _Updater


class _FakeBus:
    """Stands in for the SocketCAN transport: no socket, just open/closed.

    ``stalled`` mirrors :attr:`CanBus.stalled`: the bus has seen its TX queue
    stop draining because no node ACKs (motor power gone). A bus whose
    interface merely dropped off USB times out the same way without it.
    """

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.closed = False
        self.stalled = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class _FakeDriver:
    """One motor at the CAN protocol boundary, powered or not.

    An unpowered motor answers nothing, exactly like a driver whose request
    frames time out; a powered one answers reads and reports whether it took
    the disable command.
    """

    def __init__(self, *, powered: bool = True, accepts_disable: bool = True) -> None:
        self.powered = powered
        self.accepts_disable = accepts_disable
        self.disable_calls = 0

    def set_feedback_callback(self, _callback: Any) -> None:
        pass

    def _answer(self) -> None:
        if not self.powered:
            raise MotorError("motor did not answer")

    async def disable(self) -> None:
        self.disable_calls += 1
        self._answer()
        if not self.accepts_disable:
            raise MotorError("motor did not confirm disabled")

    async def get_error_code(self) -> MotorStatus:
        self._answer()
        return MotorStatus.OK


def _axol_with_drivers(driver_for: Any) -> tuple[Axol, list[_FakeBus]]:
    """Build an Axol whose motors are ``driver_for(channel)`` fakes."""
    buses: list[_FakeBus] = []

    def make_bus(channel: str) -> _FakeBus:
        bus = _FakeBus(channel)
        buses.append(bus)
        return bus

    with (
        patch("almond_axol.robot.axol.CanBus", side_effect=make_bus),
        patch(
            "almond_axol.motor.motor.make_driver",
            side_effect=lambda bus, *_args, **_kwargs: driver_for(bus.channel),
        ),
    ):
        axol = Axol(left_channel="can-left", right_channel="can-right")
    return axol, buses


class DisableClassificationTest(unittest.IsolatedAsyncioTestCase):
    async def test_unpowered_arms_leave_no_cleanup_uncertainty(self) -> None:
        axol, buses = _axol_with_drivers(lambda _channel: _FakeDriver(powered=False))
        for bus in buses:
            bus.stalled = True

        await axol.disable()

        self.assertTrue(all(bus.closed for bus in buses))

    async def test_reachable_motor_that_will_not_disable_still_raises(self) -> None:
        drivers = {
            "can-left": lambda: _FakeDriver(powered=False),
            "can-right": lambda: _FakeDriver(accepts_disable=False),
        }
        axol, buses = _axol_with_drivers(lambda channel: drivers[channel]())
        for bus in buses:
            bus.stalled = True

        with self.assertRaises(MotorError):
            await axol.disable()

        # Both buses stay open: the failure has to remain retryable.
        self.assertFalse(any(bus.closed for bus in buses))

    async def test_silent_motors_on_an_unstalled_bus_still_raise(self) -> None:
        # Every read times out but the bus never declared a stall: that is the
        # CAN interface dropping off USB, not motor power going away. The
        # motors may still be holding torque, so this must stay uncertain.
        axol, buses = _axol_with_drivers(lambda _channel: _FakeDriver(powered=False))

        with self.assertRaises(MotorError):
            await axol.disable()

        self.assertFalse(any(bus.closed for bus in buses))


class TxQueueFullClassificationTest(unittest.TestCase):
    """Both shapes python-can gives a full TX queue must be recognised.

    Missing either one re-raises out of ``CanBus._send`` instead of dropping
    the frame, so the command never times out upstream as a ``MotorError`` and
    the bus never accumulates the overflow that declares the stall — which
    takes the watchdog and the unpowered classification down with it.
    """

    def test_errno_form_is_recognised(self) -> None:
        exc = can.CanOperationError("send failed", error_code=errno.ENOBUFS)

        self.assertTrue(_tx_queue_full(exc))

    def test_message_only_form_is_recognised(self) -> None:
        # python-can's SocketCAN backend raises exactly this, with no errno,
        # after retrying a partial write for its send timeout.
        exc = can.CanOperationError("Transmit buffer full")

        self.assertTrue(_tx_queue_full(exc))

    def test_unrelated_can_errors_are_left_alone(self) -> None:
        self.assertFalse(_tx_queue_full(can.CanOperationError("Failed to transmit")))
        self.assertFalse(_tx_queue_full(OSError(errno.ENODEV, "No such device")))


class FlushLockLoopAffinityTest(unittest.TestCase):
    """The flush lock must not outlive the loop that took it.

    A bus is owned by an operation's loop while it holds the arms and by the
    idle link's loop afterwards. A single module-level ``asyncio.Lock`` binds
    to the first of those and then rejects the second, which strands every
    later reconnect behind "bound to a different event loop".
    """

    def test_each_loop_gets_its_own_lock(self) -> None:
        async def take() -> asyncio.Lock:
            lock = _flush_lock_for_running_loop()
            async with lock:
                return lock

        first = asyncio.run(take())
        second = asyncio.run(take())

        self.assertIsNot(first, second)

    def test_the_same_loop_reuses_one_lock(self) -> None:
        async def take_twice() -> tuple[asyncio.Lock, asyncio.Lock]:
            return _flush_lock_for_running_loop(), _flush_lock_for_running_loop()

        first, second = asyncio.run(take_twice())

        self.assertIs(first, second)

    def test_a_dead_loop_does_not_strand_the_next_one(self) -> None:
        # Acquire without releasing, exactly as a loop torn down mid-flush
        # leaves it, then prove a fresh loop can still flush.
        async def acquire_and_abandon() -> None:
            await _flush_lock_for_running_loop().acquire()

        asyncio.run(acquire_and_abandon())

        async def flush_again() -> bool:
            async with _flush_lock_for_running_loop():
                return True

        self.assertTrue(asyncio.run(flush_again()))


class _LockedOutLink:
    """Robot link whose motors read as *motors* say after a task released it."""

    def __init__(self, motors: list[dict[str, Any]]) -> None:
        self._motors = motors
        self.state = STATE_BUSY
        self.reacquires = 0
        self.releases = 0
        self.probes = 0

    def profile(self) -> str:
        return "axol"

    def channels(self) -> tuple[str | None, str | None]:
        return "can-left", "can-right"

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "connected": self.state in (STATE_CONNECTED, STATE_BUSY),
            "lastPing": time.time(),
            "channels": {"left": "can-left", "right": "can-right"},
            "profile": "axol",
            "hasGripper": True,
            "motors": self._motors,
        }

    def reacquire(self) -> bool:
        if self.state != STATE_BUSY:
            return False
        self.reacquires += 1
        self.state = STATE_CONNECTED
        return True

    def probe(self) -> dict[str, Any]:
        self.probes += 1
        if self.state != STATE_CONNECTED:
            raise RuntimeError(f"robot link is {self.state}")
        return self.status()

    def release(self) -> None:
        self.releases += 1
        self.state = STATE_BUSY

    def motor_faults(self) -> list[dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        pass


def _motor(joint: str, *, reachable: bool | None, status: str | None) -> dict[str, Any]:
    return {
        "arm": "left",
        "joint": joint,
        "reachable": reachable,
        "status": status,
        "temperature": None,
        "voltage": None,
    }


class LockoutExemptionTest(unittest.IsolatedAsyncioTestCase):
    def _locked_out_runner(self, robot: _LockedOutLink) -> OperationRunner:
        """A runner whose last operation failed to confirm its torque-off."""
        from almond_axol.serve.commands import COMMANDS

        def fail(_cfg: Any, *, stop_event: threading.Event) -> None:
            del stop_event
            raise HardwareCleanupError("robot disable failed")

        runner = OperationRunner(robot_link=robot)
        session = Session("cleanup-test", {})
        session.status = "running"
        runner._session = session
        command = SimpleNamespace(
            load_entrypoint=lambda: fail,
            load_episode_control=lambda: None,
        )
        with (
            patch.dict(COMMANDS, {"cleanup-test": command}),
            patch("almond_axol.serve.runner._Capture") as capture,
        ):
            capture.return_value.__enter__.return_value = None
            runner._run_thread(
                session,
                "cleanup-test",
                SimpleNamespace(),
                20,
                needs_robot=True,
                manage_bridge=False,
            )
        self.assertTrue(runner.hardware_cleanup_lockout())
        return runner

    async def _client(
        self, runner: OperationRunner, robot: _LockedOutLink
    ) -> httpx.AsyncClient:
        updater = _Updater(lambda: True)
        with (
            patch.object(app_module, "SessionManager", return_value=_Manager()),
            patch.object(app_module, "OperationRunner", return_value=runner),
            patch.object(app_module, "SettingsStore", return_value=_Settings()),
            patch.object(app_module, "RobotLink", return_value=robot),
            patch.object(app_module, "SelfUpdater", return_value=updater),
        ):
            app = app_module.create_app()
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_host_restart_is_offered_while_the_lockout_holds(self) -> None:
        robot = _LockedOutLink([_motor("SHOULDER_1", reachable=False, status=None)])
        runner = self._locked_out_runner(robot)
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")

        async with await self._client(runner, robot) as client:
            with (
                patch.object(app_module.os, "geteuid", return_value=0),
                patch.object(app_module.subprocess, "run", return_value=completed),
            ):
                response = await client.post("/api/host/restart")

        self.assertEqual(response.status_code, 200)

    async def test_clear_lockout_refuses_while_a_motor_still_answers(self) -> None:
        robot = _LockedOutLink(
            [
                _motor("SHOULDER_1", reachable=False, status=None),
                _motor("WRIST_2", reachable=True, status="OK"),
            ]
        )
        runner = self._locked_out_runner(robot)

        async with await self._client(runner, robot) as client:
            response = await client.post("/api/op/clear-lockout")

        self.assertEqual(response.status_code, 409)
        self.assertIn("wrist_2", response.json()["error"])
        self.assertTrue(runner.hardware_cleanup_lockout())
        self.assertTrue(runner.is_running())
        # The probe only borrowed the buses: a refused lockout hands them back
        # so the idle link is not left sitting on channels the failed
        # operation may still hold.
        self.assertEqual(robot.reacquires, 1)
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.state, STATE_BUSY)

    async def test_clear_lockout_keeps_a_link_it_did_not_reconnect(self) -> None:
        # The operator connected the panel themselves before asking; a refusal
        # must not yank that connection away.
        robot = _LockedOutLink([_motor("WRIST_2", reachable=True, status="OK")])
        robot.state = STATE_CONNECTED
        runner = self._locked_out_runner(robot)

        async with await self._client(runner, robot) as client:
            response = await client.post("/api/op/clear-lockout")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(robot.reacquires, 0)
        self.assertEqual(robot.releases, 0)
        self.assertEqual(robot.state, STATE_CONNECTED)

    async def test_clear_lockout_releases_once_every_motor_is_silent(self) -> None:
        robot = _LockedOutLink(
            [
                _motor("SHOULDER_1", reachable=False, status=None),
                _motor("WRIST_2", reachable=True, status="DISABLED"),
            ]
        )
        runner = self._locked_out_runner(robot)

        async with await self._client(runner, robot) as client:
            response = await client.post("/api/op/clear-lockout")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cleared"])
        self.assertEqual(robot.reacquires, 1)
        self.assertFalse(runner.hardware_cleanup_lockout())
        self.assertFalse(runner.is_running())

    async def test_clear_lockout_keeps_the_lockout_when_nothing_was_read(self) -> None:
        robot = _LockedOutLink([_motor("SHOULDER_1", reachable=None, status=None)])
        runner = self._locked_out_runner(robot)

        async with await self._client(runner, robot) as client:
            response = await client.post("/api/op/clear-lockout")

        self.assertEqual(response.status_code, 409)
        self.assertTrue(runner.hardware_cleanup_lockout())
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.state, STATE_BUSY)


class BusStallWatchdogTest(unittest.TestCase):
    def _stalled_bus(self, channel: str) -> CanBus:
        """A CanBus flagged stalled the way a dead (unpowered) bus flags itself."""
        bus = object.__new__(CanBus)
        bus._channel = channel
        bus._stalled = False
        bus._lost = False
        bus._wake = asyncio.Event()
        bus._mark_stalled(OSError("ENOBUFS"))
        return bus

    def test_sustained_stall_stops_the_operation(self) -> None:
        runner = OperationRunner()
        session = Session("teleop", {})
        session.status = "running"
        runner._session = session
        stops: list[bool] = []
        watchdog = threading.Thread(
            target=runner._watch_bus_stall,
            args=(session, threading.Event()),
            daemon=True,
        )

        with (
            patch.object(runner, "stop", side_effect=lambda: stops.append(True)),
            patch("almond_axol.serve.runner._STALL_POLL_S", 0.01),
            patch("almond_axol.serve.runner.STALL_DETECT_S", 0.02),
        ):
            # The bus goes dead partway through the run, which is the only
            # stall this operation owns.
            watchdog.start()
            time.sleep(0.05)
            bus = self._stalled_bus("can-stall-test")
            self.addCleanup(lambda: asyncio.run(_discard_stall(bus)))
            watchdog.join(timeout=5.0)

        self.assertFalse(watchdog.is_alive())
        self.assertEqual(stops, [True])
        self.assertEqual(session.status, "error")
        self.assertEqual(session.error, STALL_STOP_ERROR)

    def test_stall_inherited_from_an_earlier_run_is_not_this_run_s(self) -> None:
        runner = OperationRunner()
        session = Session("teleop", {})
        session.status = "running"
        runner._session = session
        bus = self._stalled_bus("can-inherited-test")
        self.addCleanup(lambda: asyncio.run(_discard_stall(bus)))
        stop_event = threading.Event()
        threading.Timer(0.05, stop_event.set).start()

        with (
            patch.object(runner, "stop", side_effect=AssertionError("stopped")),
            patch("almond_axol.serve.runner._STALL_POLL_S", 0.01),
            patch("almond_axol.serve.runner.STALL_DETECT_S", 0.01),
        ):
            runner._watch_bus_stall(session, stop_event)

        self.assertEqual(session.status, "running")

    def test_stall_that_recovers_leaves_the_operation_running(self) -> None:
        runner = OperationRunner()
        session = Session("teleop", {})
        session.status = "running"
        runner._session = session
        stop_event = threading.Event()
        # Stalled once, then ACKing again — a transient overflow, not power loss.
        readings = iter([frozenset({"can-left"}), frozenset()])

        def sample() -> frozenset[str]:
            reading = next(readings, frozenset())
            if not reading:
                stop_event.set()
            return reading

        with (
            patch.object(runner, "stop", side_effect=AssertionError("stopped")),
            patch("almond_axol.serve.runner.stalled_channels", sample),
            patch("almond_axol.serve.runner._STALL_POLL_S", 0.01),
        ):
            runner._watch_bus_stall(session, stop_event)

        self.assertEqual(session.status, "running")
        self.assertIsNone(session.error)


async def _discard_stall(bus: CanBus) -> None:
    """Take the test bus back out of the process-wide stalled set."""
    bus._bus = None
    bus._reader_task = None
    await bus.close()


class _FakeMotor:
    """A link-owned motor at the driver boundary: answers, or does not."""

    def __init__(self, *, powered: bool = True) -> None:
        self.powered = powered

    async def _read(self, value: Any) -> Any:
        if not self.powered:
            raise MotorError("motor did not answer")
        return value

    async def get_error_code(self) -> MotorStatus:
        return await self._read(MotorStatus.DISABLED)

    async def get_temperature(self) -> float:
        return await self._read(31.0)

    async def get_voltage(self) -> float:
        return await self._read(48.0)

    async def get_position(self) -> float:
        return await self._read(0.0)

    async def get_velocity(self) -> float:
        return await self._read(0.0)

    async def get_torque(self) -> float:
        return await self._read(0.0)


class StaleTelemetryTest(unittest.TestCase):
    def _connected_link(self) -> RobotLink:
        with (
            patch("almond_axol.serve.robot_link.CanBus", side_effect=_FakeBus),
            patch(
                "almond_axol.serve.robot_link.Motor",
                side_effect=lambda *_args, **_kwargs: _FakeMotor(),
            ),
        ):
            link = RobotLink(left_channel="can-left", right_channel="can-right")
            self.addCleanup(link._loop.call_soon_threadsafe, link._loop.stop)
            link._submit(link._open_and_start())
            link._set_state(STATE_CONNECTED)
            link._submit(link._ping_once())
        return link

    def test_release_drops_the_health_a_task_took_over(self) -> None:
        link = self._connected_link()
        self.assertTrue(all(m["reachable"] for m in link.status()["motors"]))

        with patch.object(link, "_submit", return_value=None):
            link.release()

        status = link.status()
        self.assertEqual(status["state"], STATE_BUSY)
        self.assertTrue(all(m["reachable"] is None for m in status["motors"]))
        self.assertEqual(status["reachableCount"], 0)
        self.assertEqual(link.hub.snapshot()["slow"], {})

    def test_unknown_motors_are_not_reported_as_faults(self) -> None:
        motors = [_motor("SHOULDER_1", reachable=None, status=None)]

        self.assertEqual(motor_faults(motors, connected=True), [])

    def test_unreachable_motors_are_still_faults(self) -> None:
        motors = [_motor("SHOULDER_1", reachable=False, status=None)]

        self.assertEqual(
            [f["problem"] for f in motor_faults(motors, connected=True)],
            ["unreachable"],
        )


if __name__ == "__main__":
    unittest.main()
