"""The idle-time Jelly wheel / lift links and their control-panel API.

``serve.jelly_link.JellyLink`` mirrors the arm link's state machine for the
two Jelly devices: connect / disconnect per device, ``release`` hands every
connected bus to a task (``busy``) and ``reacquire`` brings them back. These
tests drive that machine with the CAN devices mocked out, check the status
snapshots the panel renders, and exercise the ``/api/jelly`` endpoints plus
the release / reacquire handover around subprocess launches.
"""

from __future__ import annotations

import asyncio
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx

from almond_axol.serve import app as app_module
from almond_axol.serve import jelly_link
from almond_axol.serve.jelly_link import (
    JELLY_DEVICES,
    JellyLink,
    _LiftDevice,
    _WheelsDevice,
    device_presence,
)
from almond_axol.serve.robot_link import (
    STATE_BUSY,
    STATE_CONNECTED,
    STATE_DISCONNECTED,
    STATE_ERROR,
)
from almond_axol.serve.runner import OperationRunner
from tests.test_serve_session_reservation import (
    _Jelly,
    _Manager,
    _Robot,
    _Runner,
    _hub_state,
    _test_app,
)


def _lift_frame(
    *,
    permille: int = 750,
    flags: int = 0x01,
    fault_mask: int | None = 0,
    driver_state: int | None = 0x03,
) -> bytes:
    data = struct.pack("<HhBb", permille, 0, flags, 0)
    if fault_mask is not None and driver_state is not None:
        data += bytes([fault_mask, driver_state])
    return data


class _FakeMotor:
    """A Damiao wheel that answers (or does not) the idle ping."""

    def __init__(
        self,
        *,
        reachable: bool = True,
        status: str = "OK",
        temperature: float = 31.0,
        voltage: float = 24.2,
    ) -> None:
        self.reachable = reachable
        self._status = status
        self._temperature = temperature
        self._voltage = voltage

    async def get_error_code(self) -> Any:
        if not self.reachable:
            raise TimeoutError("no reply")
        return SimpleNamespace(name=self._status)

    async def get_temperature(self) -> float:
        return self._temperature

    async def get_voltage(self) -> float:
        return self._voltage


class _MockedDevices:
    """Patch both devices' CAN I/O and the interface bring-up out of a link."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.open_error: dict[str, Exception] = {}
        self.close_error: dict[str, Exception] = {}
        self.enable_error: dict[str, Exception] = {}

    def __enter__(self) -> _MockedDevices:
        mocked = self

        async def open_wheels(device: _WheelsDevice) -> None:
            mocked.opened.append("wheels")
            if "wheels" in mocked.open_error:
                raise mocked.open_error["wheels"]

        async def open_lift(device: _LiftDevice) -> None:
            mocked.opened.append("lift")
            if "lift" in mocked.open_error:
                raise mocked.open_error["lift"]

        async def close_wheels(device: _WheelsDevice) -> None:
            mocked.closed.append("wheels")
            if "wheels" in mocked.close_error:
                raise mocked.close_error["wheels"]

        async def close_lift(device: _LiftDevice) -> None:
            mocked.closed.append("lift")
            if "lift" in mocked.close_error:
                raise mocked.close_error["lift"]

        def enable_can(link: JellyLink, dev_link: Any) -> None:
            if dev_link.name in mocked.enable_error:
                raise mocked.enable_error[dev_link.name]

        self._patches = [
            patch.object(_WheelsDevice, "open", open_wheels),
            patch.object(_WheelsDevice, "close", close_wheels),
            patch.object(_WheelsDevice, "ping", AsyncMock()),
            patch.object(_LiftDevice, "open", open_lift),
            patch.object(_LiftDevice, "close", close_lift),
            patch.object(_LiftDevice, "ping", AsyncMock()),
            patch.object(JellyLink, "_enable_can", enable_can),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for p in reversed(self._patches):
            p.stop()


class JellyLinkStateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mocked = _MockedDevices().__enter__()
        self.addCleanup(self.mocked.__exit__, None, None, None)
        self.link = JellyLink()
        self.addCleanup(self.link.shutdown)

    def test_starts_disconnected_with_unknown_health(self) -> None:
        status = self.link.status()
        self.assertEqual(set(status), set(JELLY_DEVICES))
        for device in JELLY_DEVICES:
            self.assertEqual(status[device]["state"], STATE_DISCONNECTED)
            self.assertFalse(status[device]["connected"])
            self.assertIsNone(status[device]["error"])
            self.assertTrue(status[device]["channel"])
        self.assertEqual(status["wheels"]["motorCount"], 4)
        self.assertEqual(status["wheels"]["reachableCount"], 0)
        self.assertTrue(all(m["reachable"] is None for m in status["wheels"]["motors"]))
        self.assertIsNone(status["lift"]["reachable"])
        self.assertIsNone(status["lift"]["status"])

    def test_connect_and_disconnect_are_per_device(self) -> None:
        status = self.link.connect("wheels")
        self.assertEqual(status["wheels"]["state"], STATE_CONNECTED)
        self.assertEqual(status["lift"]["state"], STATE_DISCONNECTED)
        self.assertEqual(self.mocked.opened, ["wheels"])

        # Connecting an already-connected device is a no-op, not a reopen.
        self.link.connect("wheels")
        self.assertEqual(self.mocked.opened, ["wheels"])

        status = self.link.connect("lift")
        self.assertEqual(status["lift"]["state"], STATE_CONNECTED)
        self.assertEqual(status["lift"]["reachable"], False)

        status = self.link.disconnect("wheels")
        self.assertEqual(status["wheels"]["state"], STATE_DISCONNECTED)
        self.assertEqual(status["lift"]["state"], STATE_CONNECTED)
        self.assertEqual(self.mocked.closed, ["wheels"])

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.link.connect("legs")

    def test_missing_interface_reports_error_state(self) -> None:
        self.mocked.enable_error["wheels"] = RuntimeError(
            "Jelly wheels bus can_alm_axol_b not found"
        )
        status = self.link.connect("wheels")
        self.assertEqual(status["wheels"]["state"], STATE_ERROR)
        self.assertIn("not found", status["wheels"]["error"])
        self.assertEqual(self.mocked.opened, [])
        # A later successful connect clears the error.
        del self.mocked.enable_error["wheels"]
        status = self.link.connect("wheels")
        self.assertEqual(status["wheels"]["state"], STATE_CONNECTED)
        self.assertIsNone(status["wheels"]["error"])

    def test_open_failure_reports_error_state(self) -> None:
        self.mocked.open_error["lift"] = OSError("No such device")
        status = self.link.connect("lift")
        self.assertEqual(status["lift"]["state"], STATE_ERROR)
        self.assertIn("No such device", status["lift"]["error"])

    def test_release_marks_connected_devices_busy_and_reacquire_reopens(self) -> None:
        self.link.connect("wheels")
        self.link.connect("lift")
        self.mocked.opened.clear()

        self.link.release()
        status = self.link.status()
        self.assertEqual(status["wheels"]["state"], STATE_BUSY)
        self.assertEqual(status["lift"]["state"], STATE_BUSY)
        # Busy still counts as connected for the panel, but nobody polls.
        self.assertTrue(status["wheels"]["connected"])
        self.assertIsNone(status["lift"]["reachable"])
        self.assertTrue(all(m["reachable"] is None for m in status["wheels"]["motors"]))
        self.assertEqual(sorted(self.mocked.closed), ["lift", "wheels"])

        with self.assertRaises(RuntimeError):
            self.link.disconnect("wheels")

        self.assertTrue(self.link.reacquire())
        status = self.link.status()
        self.assertEqual(status["wheels"]["state"], STATE_CONNECTED)
        self.assertEqual(status["lift"]["state"], STATE_CONNECTED)
        self.assertEqual(sorted(self.mocked.opened), ["lift", "wheels"])

    def test_release_skips_disconnected_devices(self) -> None:
        self.link.connect("lift")
        self.link.release()
        status = self.link.status()
        self.assertEqual(status["wheels"]["state"], STATE_DISCONNECTED)
        self.assertEqual(status["lift"]["state"], STATE_BUSY)
        self.assertEqual(self.mocked.closed, ["lift"])
        self.assertTrue(self.link.reacquire())
        self.assertEqual(self.link.status()["wheels"]["state"], STATE_DISCONNECTED)

    def test_reacquire_without_release_is_a_no_op(self) -> None:
        self.link.connect("wheels")
        self.mocked.opened.clear()
        self.assertFalse(self.link.reacquire())
        self.assertEqual(self.mocked.opened, [])

    def test_release_failure_leaves_device_in_error_and_raises(self) -> None:
        self.link.connect("wheels")
        self.link.connect("lift")
        self.mocked.close_error["wheels"] = RuntimeError("CAN close timed out")
        with self.assertRaises(RuntimeError) as ctx:
            self.link.release()
        self.assertIn("wheels", str(ctx.exception))
        status = self.link.status()
        self.assertEqual(status["wheels"]["state"], STATE_ERROR)
        self.assertIn("CAN close timed out", status["wheels"]["error"])
        # The device that did close is still handed over cleanly.
        self.assertEqual(status["lift"]["state"], STATE_BUSY)

    def test_reacquire_failure_leaves_device_in_error(self) -> None:
        self.link.connect("wheels")
        self.link.release()
        self.mocked.open_error["wheels"] = OSError("adapter unplugged")
        self.assertFalse(self.link.reacquire())
        status = self.link.status()
        self.assertEqual(status["wheels"]["state"], STATE_ERROR)
        self.assertIn("adapter unplugged", status["wheels"]["error"])

    def test_disconnect_all_leaves_busy_devices_alone(self) -> None:
        self.link.connect("wheels")
        self.link.connect("lift")
        # Only the wheels get borrowed by a task.
        self.link.disconnect("lift")
        self.link.release()
        self.link.connect("lift")
        status = self.link.disconnect_all()
        self.assertEqual(status["wheels"]["state"], STATE_BUSY)
        self.assertEqual(status["lift"]["state"], STATE_DISCONNECTED)

    def test_error_after_failed_close_is_cleaned_up_on_next_connect(self) -> None:
        self.link.connect("wheels")
        self.mocked.close_error["wheels"] = RuntimeError("CAN close timed out")
        self.link.disconnect("wheels")
        self.assertEqual(self.link.status()["wheels"]["state"], STATE_ERROR)
        del self.mocked.close_error["wheels"]
        self.mocked.closed.clear()
        self.mocked.opened.clear()
        status = self.link.connect("wheels")
        # The bus may still have been open, so the retry closes before opening.
        self.assertEqual(self.mocked.closed, ["wheels"])
        self.assertEqual(self.mocked.opened, ["wheels"])
        self.assertEqual(status["wheels"]["state"], STATE_CONNECTED)


class WheelsDeviceSnapshotTest(unittest.TestCase):
    def test_ping_records_health_per_wheel(self) -> None:
        device = _WheelsDevice()
        device._motors = {
            1: _FakeMotor(),
            2: _FakeMotor(reachable=False),
            3: _FakeMotor(status="OVER_TEMPERATURE", temperature=91.0),
            4: _FakeMotor(status="DISABLED"),
        }
        device._locks = {i: asyncio.Lock() for i in device._motors}
        asyncio.run(device.ping())
        snapshot = device.snapshot()
        self.assertEqual(snapshot["motorCount"], 4)
        self.assertEqual(snapshot["reachableCount"], 3)
        by_id = {m["id"]: m for m in snapshot["motors"]}
        self.assertEqual(
            [m["name"] for m in snapshot["motors"]],
            ["front_left", "front_right", "back_left", "back_right"],
        )
        self.assertTrue(by_id[1]["reachable"])
        self.assertEqual(by_id[1]["status"], "OK")
        self.assertEqual(by_id[1]["temperature"], 31.0)
        self.assertEqual(by_id[1]["voltage"], 24.2)
        self.assertFalse(by_id[2]["reachable"])
        self.assertIsNone(by_id[2]["status"])
        self.assertEqual(by_id[3]["status"], "OVER_TEMPERATURE")
        self.assertEqual(by_id[3]["temperature"], 91.0)
        self.assertEqual(by_id[4]["status"], "DISABLED")


class LiftDeviceSnapshotTest(unittest.TestCase):
    def test_status_frame_is_decoded_and_freshness_tracked(self) -> None:
        device = _LiftDevice()
        self.assertFalse(device.reachable())
        device._on_message(
            SimpleNamespace(
                arbitration_id=0x421,
                data=_lift_frame(
                    permille=750, flags=0x01 | 0x20, fault_mask=0, driver_state=0x03
                ),
            )
        )
        self.assertTrue(device.reachable())
        snapshot = device.snapshot(polling=True)
        self.assertTrue(snapshot["reachable"])
        board = snapshot["status"]
        self.assertEqual(board["heightPercent"], 75.0)
        self.assertTrue(board["homed"])
        self.assertFalse(board["moving"])
        self.assertFalse(board["homing"])
        self.assertFalse(board["stallFault"])
        self.assertFalse(board["atLower"])
        self.assertTrue(board["atUpper"])
        self.assertTrue(board["driversEnabled"])
        self.assertTrue(board["vmPresent"])
        self.assertEqual(board["driverFaultMask"], 0)

    def test_legacy_frame_leaves_driver_health_unknown(self) -> None:
        device = _LiftDevice()
        device._on_message(
            SimpleNamespace(
                arbitration_id=0x421,
                data=_lift_frame(
                    permille=0xFFFF, flags=0x08, fault_mask=None, driver_state=None
                ),
            )
        )
        board = device.snapshot(polling=True)["status"]
        self.assertIsNone(board["heightPercent"])
        self.assertFalse(board["homed"])
        self.assertTrue(board["stallFault"])
        self.assertIsNone(board["driversEnabled"])
        self.assertIsNone(board["vmPresent"])
        self.assertIsNone(board["driverFaultMask"])

    def test_other_frames_are_ignored(self) -> None:
        device = _LiftDevice()
        device._on_message(SimpleNamespace(arbitration_id=0x420, data=_lift_frame()))
        device._on_message(SimpleNamespace(arbitration_id=0x421, data=b"\x00\x00"))
        self.assertIsNone(device.status)
        self.assertFalse(device.reachable())

    def test_stale_status_means_board_stopped_answering(self) -> None:
        device = _LiftDevice()
        device._on_message(SimpleNamespace(arbitration_id=0x421, data=_lift_frame()))
        device.last_status_monotonic = time.monotonic() - jelly_link._LIFT_FRESH_S - 1.0
        self.assertFalse(device.reachable())
        snapshot = device.snapshot(polling=True)
        self.assertFalse(snapshot["reachable"])
        # The last decoded frame is still reported so the panel can show it.
        self.assertIsNotNone(snapshot["status"])
        self.assertIsNone(device.snapshot(polling=False)["reachable"])


class DevicePresenceTest(unittest.TestCase):
    def test_presence_follows_the_pinned_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sys_net = Path(tmp)
            with (
                patch.object(jelly_link, "_SYS_NET", sys_net),
                patch.object(
                    jelly_link, "wheels_channel", return_value="can_alm_axol_b"
                ),
                patch.object(jelly_link, "lift_channel", return_value="can_alm_axol_c"),
            ):
                absent = device_presence("wheels")
                self.assertEqual(
                    absent, {"channel": "can_alm_axol_b", "present": False, "up": False}
                )

                wheels = sys_net / "can_alm_axol_b"
                wheels.mkdir()
                (wheels / "flags").write_text("0x1003\n")
                self.assertEqual(
                    device_presence("wheels"),
                    {"channel": "can_alm_axol_b", "present": True, "up": True},
                )

                lift = sys_net / "can_alm_axol_c"
                lift.mkdir()
                (lift / "flags").write_text("0x1002\n")
                self.assertEqual(
                    device_presence("lift"),
                    {"channel": "can_alm_axol_c", "present": True, "up": False},
                )


class JellyApiTest(unittest.IsolatedAsyncioTestCase):
    def _client(self, app: Any) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_status_reports_both_devices(self) -> None:
        jelly = _Jelly(states={"wheels": "connected", "lift": "disconnected"})
        app = _test_app(_Manager(), _Runner(), jelly=jelly)
        async with self._client(app) as client:
            response = await client.get("/api/jelly/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["wheels"]["state"], "connected")
        self.assertTrue(body["wheels"]["connected"])
        self.assertEqual(body["lift"]["state"], "disconnected")

    async def test_connect_and_disconnect_round_trip(self) -> None:
        jelly = _Jelly()
        app = _test_app(_Manager(), _Runner(), jelly=jelly)
        async with self._client(app) as client:
            connected = await client.post("/api/jelly/lift/connect")
            self.assertEqual(connected.status_code, 200)
            self.assertEqual(connected.json()["lift"]["state"], "connected")
            self.assertEqual(connected.json()["wheels"]["state"], "disconnected")

            disconnected = await client.post("/api/jelly/lift/disconnect")
            self.assertEqual(disconnected.status_code, 200)
            self.assertEqual(disconnected.json()["lift"]["state"], "disconnected")
        self.assertEqual(jelly.connects, ["lift"])
        self.assertEqual(jelly.disconnects, ["lift"])

    async def test_unknown_device_is_404(self) -> None:
        app = _test_app(_Manager(), _Runner())
        async with self._client(app) as client:
            connect = await client.post("/api/jelly/legs/connect")
            disconnect = await client.post("/api/jelly/legs/disconnect")
        self.assertEqual(connect.status_code, 404)
        self.assertEqual(disconnect.status_code, 404)

    async def test_connect_refuses_while_hardware_is_owned(self) -> None:
        jelly = _Jelly()
        app = _test_app(_Manager(), _Runner(running=True), jelly=jelly)
        async with self._client(app) as client:
            connect = await client.post("/api/jelly/wheels/connect")
            disconnect = await client.post("/api/jelly/wheels/disconnect")
        self.assertEqual(connect.status_code, 409)
        self.assertEqual(disconnect.status_code, 409)
        self.assertEqual(jelly.connects, [])
        self.assertEqual(jelly.disconnects, [])

    async def test_disconnect_of_busy_device_is_409(self) -> None:
        jelly = _Jelly(states={"wheels": "busy", "lift": "disconnected"})
        app = _test_app(_Manager(), _Runner(), jelly=jelly)
        async with self._client(app) as client:
            response = await client.post("/api/jelly/wheels/disconnect")
        self.assertEqual(response.status_code, 409)
        self.assertIn("owns its bus", response.json()["error"])

    async def test_manual_disconnect_pauses_automatic_connect_for_that_device(
        self,
    ) -> None:
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        app = _test_app(_Manager(), _Runner(), jelly=jelly)
        with (
            patch.object(app_module, "_list_can_interfaces", return_value=[]),
            patch.object(app_module, "_attached_hub_state", return_value=_hub_state()),
            patch.object(
                app_module,
                "device_presence",
                side_effect=lambda device: {
                    "channel": jelly.channels[device],
                    "present": True,
                    "up": True,
                },
            ),
        ):
            async with self._client(app) as client:
                disconnected = await client.post("/api/jelly/wheels/disconnect")
                self.assertEqual(disconnected.status_code, 200)

                inventory = await client.get("/api/can/interfaces")
                self.assertEqual(inventory.status_code, 200, inventory.text)
                devices = inventory.json()["devices"]
                self.assertTrue(devices["wheels"]["automaticConnectSuppressed"])
                self.assertFalse(devices["lift"]["automaticConnectSuppressed"])
                self.assertEqual(devices["wheels"]["channel"], "can_alm_axol_b")

                automatic = await client.post(
                    "/api/jelly/wheels/connect", json={"automatic": True}
                )
                self.assertEqual(automatic.status_code, 409)
                self.assertTrue(automatic.json()["automaticConnectSuppressed"])
                self.assertEqual(jelly.connects, [])

                # The lift's automatic connect is unaffected.
                lift = await client.post(
                    "/api/jelly/lift/connect", json={"automatic": True}
                )
                self.assertEqual(lift.status_code, 200)

                # A manual connect clears the pause.
                manual = await client.post("/api/jelly/wheels/connect")
                self.assertEqual(manual.status_code, 200)
                self.assertEqual(manual.json()["wheels"]["state"], "connected")
                inventory = await client.get("/api/can/interfaces")
                self.assertFalse(
                    inventory.json()["devices"]["wheels"]["automaticConnectSuppressed"]
                )
        self.assertEqual(jelly.connects, ["lift", "wheels"])

    async def test_subprocess_launch_hands_over_and_restores_jelly(self) -> None:
        manager = _Manager()
        robot = _Robot()
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        app = _test_app(manager, _Runner(), robot, jelly=jelly)
        async with self._client(app) as client:
            launched = await client.post(
                "/api/diagnostics/run", json={"command": "lift.home", "args": {}}
            )
            self.assertEqual(launched.status_code, 200, launched.text)
            self.assertEqual(robot.releases, 1)
            self.assertEqual(jelly.releases, 1)
            self.assertEqual(jelly.states, {"wheels": "busy", "lift": "busy"})

            blocked = await client.post("/api/jelly/wheels/connect")
            self.assertEqual(blocked.status_code, 409)

            while not manager.queues:
                await asyncio.sleep(0)
            manager.sessions[0].status = "exited"
            manager.queues[0].put_nowait(None)
            # The watcher restores both links from worker threads; give it a
            # bounded moment rather than a fixed number of loop turns.
            deadline = time.monotonic() + 5.0
            while jelly.reacquires == 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)

        self.assertEqual(robot.reacquires, 1)
        self.assertEqual(jelly.reacquires, 1)
        self.assertEqual(jelly.states, {"wheels": "connected", "lift": "connected"})

    async def test_jelly_release_failure_restores_robot_and_refuses_launch(
        self,
    ) -> None:
        manager = _Manager()
        robot = _Robot()
        jelly = _Jelly(states={"wheels": "connected", "lift": "disconnected"})
        jelly.release_error = RuntimeError("wheel bus close timed out")
        app = _test_app(manager, _Runner(), robot, jelly=jelly)
        async with self._client(app) as client:
            response = await client.post(
                "/api/diagnostics/run", json={"command": "lift.home", "args": {}}
            )
        self.assertEqual(response.status_code, 409)
        self.assertIn("Jelly", response.json()["error"])
        self.assertIn("wheel bus close timed out", response.json()["error"])
        self.assertEqual(manager.sessions, [])
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.reacquires, 1)

    async def test_camera_only_diagnostic_leaves_jelly_alone(self) -> None:
        manager = _Manager()
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        app = _test_app(manager, _Runner(), jelly=jelly)
        async with self._client(app) as client:
            launched = await client.post(
                "/api/run", json={"command": "diag.zed-cable", "args": {}}
            )
        self.assertEqual(launched.status_code, 200, launched.text)
        self.assertEqual(jelly.releases, 0)
        self.assertEqual(jelly.states, {"wheels": "connected", "lift": "connected"})

    async def test_shutdown_tears_down_jelly_after_robot(self) -> None:
        events: list[str] = []
        manager = _Manager()
        runner = _Runner()
        robot = _Robot()
        jelly = _Jelly()
        robot.shutdown = lambda: events.append("robot")  # type: ignore[method-assign]
        jelly.shutdown = lambda: events.append("jelly")  # type: ignore[method-assign]
        app = _test_app(manager, runner, robot, jelly=jelly)
        await app.router.on_shutdown[-1]()
        self.assertEqual(events, ["robot", "jelly"])


class OperationRunnerJellyTest(unittest.TestCase):
    """The in-process operation runner hands Jelly over like the arm link."""

    def _start(
        self,
        runner: OperationRunner,
        args: dict[str, Any],
        config: Any = None,
    ) -> Any:
        from almond_axol.serve.commands import COMMANDS

        async def _noop(_cfg: Any) -> None:
            return None

        with (
            patch.object(
                runner, "_build_config", return_value=config or SimpleNamespace()
            ),
            patch.object(runner, "_attach_cameras_to_teleop"),
            patch.object(runner, "_start_stall_watchdog"),
            patch.object(COMMANDS["teleop"], "load_entrypoint", return_value=_noop),
            patch("almond_axol.cli.teleop._prepare_mantis_teleop"),
        ):
            session = runner.start("teleop", args)
            thread = runner._thread
            if thread is not None:
                thread.join(timeout=5.0)
                self.assertFalse(thread.is_alive())
        return session

    def test_hardware_run_releases_then_reacquires_jelly(self) -> None:
        robot = _Robot()
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        runner = OperationRunner(robot_link=robot, jelly_link=jelly)
        session = self._start(runner, {})
        self.assertEqual(session.status, "exited", session.error)
        self.assertEqual(jelly.releases, 1)
        self.assertEqual(jelly.reacquires, 1)
        self.assertEqual(jelly.states, {"wheels": "connected", "lift": "connected"})
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.reacquires, 1)

    def test_sim_run_leaves_jelly_alone(self) -> None:
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        runner = OperationRunner(robot_link=_Robot(), jelly_link=jelly)
        session = self._start(runner, {"sim": True}, SimpleNamespace(sim=True))
        self.assertEqual(session.status, "exited", session.error)
        self.assertEqual(jelly.releases, 0)
        self.assertEqual(jelly.reacquires, 0)

    def test_mantis_run_leaves_jelly_alone(self) -> None:
        jelly = _Jelly(states={"wheels": "connected", "lift": "connected"})
        robot = _Robot(profile="mantis", channels=("can_mantis_l", "can_mantis_r"))
        runner = OperationRunner(robot_link=robot, jelly_link=jelly)
        session = self._start(
            runner,
            {"mantis": True, "mantis_source": "quest"},
            SimpleNamespace(
                mantis=True,
                mantis_source="quest",
                left_channel="can_mantis_l",
                right_channel="can_mantis_r",
            ),
        )
        self.assertEqual(session.status, "exited", session.error)
        self.assertEqual(jelly.releases, 0)
        self.assertEqual(robot.releases, 1)

    def test_jelly_release_failure_aborts_and_restores_robot(self) -> None:
        robot = _Robot()
        jelly = _Jelly(states={"wheels": "connected", "lift": "disconnected"})
        jelly.release_error = RuntimeError("wheel bus close timed out")
        runner = OperationRunner(robot_link=robot, jelly_link=jelly)
        session = self._start(runner, {})
        self.assertEqual(session.status, "error")
        self.assertIn("wheel bus close timed out", session.error or "")
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.reacquires, 1)
        self.assertIsNone(runner._thread)


if __name__ == "__main__":
    unittest.main()
