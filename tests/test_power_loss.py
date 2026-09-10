"""Losing motor power mid-operation: classification, lockout, watchdog, telemetry.

Operators cut the motor PSU as an e-stop. Everything here covers what the stack
must do when that happens while an operation owns the CAN buses.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import can
import httpx

from almond_axol.motor import CanBus, MotorError, MotorStatus
from almond_axol.motor.bus import (
    is_stall_report,
    set_channel_stalled,
    stalled_channels,
)
from almond_axol.robot.axol import Axol
from almond_axol.robot.base import HardwareCleanupError
from almond_axol.rt.link import RtLink
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
from tests.test_serve_session_reservation import _Manager, _Settings, _Updater


# What the Rust transports put on the wire when nothing ACKs for STALL_DETECT
# (``proxy.rs`` for the maintenance proxy, ``serve.rs`` for the armed core).
_PROXY_STALL = (
    "CAN can-left TX queue stalled >1s (e-stop or unpowered motors); commands "
    "stopped, stale queue purged"
)
_CORE_STALL = (
    "fault: can-left: TX queue stalled >1s — no node ACKing frames (e-stop / "
    "motors unpowered?); commands stopped, stale queue purged"
)


class _FakeBus:
    """Stands in for the Rust-proxy transport: no process, just open/closed.

    ``stalled`` mirrors :attr:`CanBus.stalled`: the proxy reported its TX
    queue stopped draining because no node ACKs (motor power gone). A proxy
    that died for any other reason fails every send the same way without it.
    """

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.closed = False
        self.stalled = False

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    def _add_listener(self, _listener: Any) -> None:
        pass

    def enable_observer_mode(self) -> None:
        pass


class _FakeDriver:
    """One motor at the CAN protocol boundary, powered or not.

    An unpowered motor answers nothing, exactly like a driver whose request
    frames time out — or, once the stalled proxy has exited, whose sends the
    bus refuses outright (``bus_gone``); a powered one answers reads and
    reports whether it took the disable command.
    """

    def __init__(
        self,
        *,
        powered: bool = True,
        accepts_disable: bool = True,
        bus_gone: bool = False,
    ) -> None:
        self.powered = powered
        self.accepts_disable = accepts_disable
        self.bus_gone = bus_gone
        self.disable_calls = 0
        # Firmware gain ceilings Axol.__init__ checks the config against.
        self.kp_max = 500.0
        self.kd_max = 5.0

    def set_feedback_callback(self, _callback: Any) -> None:
        pass

    def _answer(self) -> None:
        if self.bus_gone:
            raise can.CanOperationError("axol-rt proxy for can-left exited")
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

    async def test_unpowered_arms_behind_an_exited_proxy_are_unpowered_too(
        self,
    ) -> None:
        # The proxy purges the queue and exits on a stall, so by the time
        # disable() probes the motors the bus itself refuses every send. That
        # is still "nothing on the arm answered" — the stall flag is the proof.
        axol, buses = _axol_with_drivers(lambda _channel: _FakeDriver(bus_gone=True))
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


def _stalled_bus(channel: str) -> CanBus:
    """A CanBus flagged stalled the way its proxy's stall report flags it."""
    bus = object.__new__(CanBus)
    bus._channel = channel
    bus._stalled = False
    bus._mark_stalled(_PROXY_STALL.replace("can-left", channel))
    return bus


class FreshBusStallFlagTest(unittest.TestCase):
    def test_opening_a_bus_clears_a_stale_stall_on_its_channel(self) -> None:
        # A bus abandoned open on a dead loop (a failed teardown that kept its
        # buses) never clears its stall. The next bus on that channel is the
        # one whose state matters, and it has seen no stall yet.
        stale = _stalled_bus("can-stale-test")
        self.addCleanup(_discard_stall, stale)
        self.assertTrue(stale.stalled)
        self.assertIn("can-stale-test", stalled_channels())

        fresh = CanBus("can-stale-test")

        self.assertNotIn("can-stale-test", stalled_channels())
        self.assertFalse(fresh.stalled)
        self.assertEqual(fresh.channel, "can-stale-test")

    def test_closing_a_stalled_bus_withdraws_its_channel(self) -> None:
        bus = CanBus("can-close-test")
        self.addCleanup(_discard_stall, bus)
        bus._mark_stalled(_PROXY_STALL.replace("can-left", "can-close-test"))
        self.assertIn("can-close-test", stalled_channels())

        asyncio.run(bus.close())

        self.assertNotIn("can-close-test", stalled_channels())


class StallReportClassificationTest(unittest.IsolatedAsyncioTestCase):
    """The proxy's stall report is the one signal that the bus is *dead*.

    Every other way the proxy can go away — it crashed, the interface
    vanished, a protocol error — fails sends the same way while the motors may
    still be powered and holding torque, so only the stall text may set the
    flag; missing it takes the watchdog and the unpowered classification down.
    """

    def test_both_transports_stall_texts_are_recognised(self) -> None:
        self.assertTrue(is_stall_report(_PROXY_STALL))
        self.assertTrue(is_stall_report(_CORE_STALL))
        self.assertFalse(is_stall_report("CAN can-left send failed: ENODEV"))
        self.assertFalse(is_stall_report("fault: arm: bus thread died"))

    async def _feed(self, bus: CanBus, *payloads: bytes) -> None:
        reader = asyncio.StreamReader()
        for payload in payloads:
            reader.feed_data(len(payload).to_bytes(4, "little") + payload)
        reader.feed_eof()
        bus._reader = reader
        bus._state = "open"
        await bus._read_loop()

    async def test_proxy_stall_report_marks_the_bus_stalled(self) -> None:
        bus = CanBus("can-report-test")
        self.addCleanup(_discard_stall, bus)

        await self._feed(bus, b"E" + _PROXY_STALL.encode())

        self.assertTrue(bus.stalled)
        self.assertIn("can-report-test", stalled_channels())
        self.assertIn("stalled", bus._closed_reason or "")

    async def test_other_proxy_errors_leave_the_bus_unstalled(self) -> None:
        bus = CanBus("can-report-test")
        self.addCleanup(_discard_stall, bus)

        await self._feed(bus, b"ECAN can-report-test send failed: ENODEV")

        self.assertFalse(bus.stalled)
        self.assertNotIn("can-report-test", stalled_channels())
        self.assertIsNotNone(bus._closed_reason)


class CoreStallReportTest(unittest.TestCase):
    """While armed the realtime core owns the sockets, so its ``fault:`` is
    where the e-stop shows up; the link must publish it for the watchdog."""

    def _link(self) -> RtLink:
        link = RtLink(binary="/nonexistent/axol-rt")
        self.addCleanup(link._stalled_ifaces.clear)
        self.addCleanup(set_channel_stalled, "can-left", False)
        return link

    def test_stall_fault_names_its_interface(self) -> None:
        link = self._link()

        link._note_stall(_CORE_STALL)

        self.assertIn("can-left", stalled_channels())
        self.assertNotIn("can-right", stalled_channels())

    def test_other_faults_are_not_stalls(self) -> None:
        link = self._link()

        link._note_stall("fault: arm: bus thread died")
        link._note_stall("fault: can-left: motor 0x03 silent for 1.0s")

        self.assertNotIn("can-left", stalled_channels())

    def test_closing_the_link_withdraws_the_stall(self) -> None:
        link = self._link()
        link._note_stall(_CORE_STALL)
        self.assertIn("can-left", stalled_channels())

        asyncio.run(link.close())

        self.assertNotIn("can-left", stalled_channels())


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
        client, _updater = await self._client_and_updater(runner, robot)
        return client

    async def _client_and_updater(
        self, runner: OperationRunner, robot: _LockedOutLink
    ) -> tuple[httpx.AsyncClient, _Updater]:
        """The API client plus the updater double built with the app's real
        ``_is_idle``, which is what the panel's host tile gates on."""
        updaters: list[_Updater] = []

        def make_updater(is_idle: Any) -> _Updater:
            updater = _Updater(is_idle)
            updaters.append(updater)
            return updater

        with (
            patch.object(app_module, "SessionManager", return_value=_Manager()),
            patch.object(app_module, "OperationRunner", return_value=runner),
            patch.object(app_module, "SettingsStore", return_value=_Settings()),
            patch.object(app_module, "RobotLink", return_value=robot),
            patch.object(app_module, "SelfUpdater", side_effect=make_updater),
        ):
            app = app_module.create_app()
        transport = httpx.ASGITransport(app=app)
        (updater,) = updaters
        return httpx.AsyncClient(transport=transport, base_url="http://test"), updater

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

    async def test_host_reads_idle_while_the_lockout_holds(self) -> None:
        # The panel disables the Restart / Shutdown confirmation on
        # ``update.idle``. The lockout reserves the robot, not the host, and
        # ending the process is a documented way out of it, so the host must
        # not read busy or the API exemption above is unreachable from the UI.
        robot = _LockedOutLink([_motor("SHOULDER_1", reachable=False, status=None)])
        runner = self._locked_out_runner(robot)

        client, updater = await self._client_and_updater(runner, robot)
        async with client:
            self.assertTrue(updater._is_idle())
        # The lockout still reserves the robot; only the host is free.
        self.assertTrue(runner.is_running())

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
            bus = _stalled_bus("can-stall-test")
            self.addCleanup(_discard_stall, bus)
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
        bus = _stalled_bus("can-inherited-test")
        self.addCleanup(_discard_stall, bus)
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


def _discard_stall(bus: CanBus) -> None:
    """Take the test bus back out of the process-wide stalled set."""
    set_channel_stalled(bus.channel, False)


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
            patch("almond_axol.motor.observer.CanBus", side_effect=_FakeBus),
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
