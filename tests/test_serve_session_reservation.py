from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import httpx

from almond_axol.serve import app as app_module
from almond_axol.robot.base import HardwareCleanupError
from almond_axol.serve.manager import Session
from almond_axol.serve.robot_link import (
    RobotLink,
    STATE_BUSY,
    STATE_CONNECTED,
    STATE_DISCONNECTED,
    STATE_ERROR,
)
from almond_axol.serve.runner import OperationRunner
from almond_axol.serve.settings import SettingsStore
from almond_axol.utils import state_files


_UNSET = object()


class _Settings:
    def __init__(
        self,
        recording_root: str | None = None,
        merged_values: dict[str, Any] | None = None,
    ) -> None:
        self.recording_root = recording_root
        self.merged_values = merged_values or {}

    def can_channels(self) -> tuple[str, str]:
        return "can-left", "can-right"

    def mantis_can_channels(self) -> tuple[str, str]:
        return "can_mantis_l", "can_mantis_r"

    def has_gripper(self) -> bool:
        return True

    def snapshot(self) -> dict[str, Any]:
        values = (
            {"recording.root": self.recording_root}
            if self.recording_root is not None
            else {}
        )
        return {"values": values, "cameras": None, "advanced": {}}

    def merged_args(self, _command: str, args: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.merged_values)
        merged.update(args)
        return merged

    def effective_axol_can_channels(
        self, command: str, args: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        merged = self.merged_args(command, args)
        if command in {
            "collect-data",
            "collect-dagger",
            "replay-dataset",
            "run-policy",
        }:
            keys = ("robot_config.left_channel", "robot_config.right_channel")
        else:
            keys = ("left_channel", "right_channel")

        def resolve(key: str, default: str) -> str | None:
            value = merged.get(key, default)
            if value is None or not str(value).strip():
                return default
            text = str(value).strip()
            return None if text.lower() in ("null", "none") else text

        return resolve(keys[0], "can-left"), resolve(keys[1], "can-right")

    def effective_mantis_can_channels(
        self, command: str, args: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        merged = self.merged_args(command, args)
        if command == "collect-data":
            keys = ("robot_config.left_channel", "robot_config.right_channel")
        else:
            keys = ("left_channel", "right_channel")
        return (
            merged.get(keys[0], "can_mantis_l"),
            merged.get(keys[1], "can_mantis_r"),
        )


class _Robot:
    def __init__(
        self,
        *_args: Any,
        profile: str = "axol",
        channels: tuple[str | None, str | None] = ("can-left", "can-right"),
        has_gripper: bool = True,
        state: str = "connected",
        last_ping: float | None | object = _UNSET,
        **_kwargs: Any,
    ) -> None:
        self._profile = profile
        self._channels = channels
        self._has_gripper = has_gripper
        self._state = state
        self._last_ping = time.time() if last_ping is _UNSET else last_ping
        self.releases = 0
        self.reacquires = 0
        self.connects = 0
        self.disconnects = 0
        self.set_channel_calls = 0
        self.release_error: Exception | None = None
        self.fault_check_entered = threading.Event()
        self.fault_check_gate: threading.Event | None = None

    def profile(self) -> str:
        return self._profile

    def channels(self) -> tuple[str | None, str | None]:
        return self._channels

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "connected": self._state in ("connected", "busy"),
            "lastPing": self._last_ping,
            "channels": {"left": self._channels[0], "right": self._channels[1]},
            "profile": self._profile,
            "hasGripper": self._has_gripper,
        }

    def connect(self) -> dict[str, Any]:
        self.connects += 1
        self._state = "connected"
        self._last_ping = time.time()
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.disconnects += 1
        self._state = "disconnected"
        self._last_ping = None
        return self.status()

    def set_channels(
        self,
        left: str | None,
        right: str | None,
        *,
        profile: str = "axol",
    ) -> None:
        self.set_channel_calls += 1
        self._channels = (left, right)
        self._profile = profile

    def release(self) -> None:
        self.releases += 1
        if self.release_error is not None:
            self._state = "error"
            raise self.release_error
        self._state = "busy"

    def reacquire(self) -> None:
        self.reacquires += 1
        self._state = "connected"

    def motor_faults(self) -> list[dict[str, Any]]:
        self.fault_check_entered.set()
        if self.fault_check_gate is not None:
            self.fault_check_gate.wait(timeout=1.0)
        return []

    def shutdown(self) -> None:
        pass


class _Manager:
    def __init__(self, sessions: list[Session] | None = None) -> None:
        self.sessions = sessions or []
        self.start_entered = asyncio.Event()
        self.start_gate: asyncio.Event | None = None
        self.queues: list[asyncio.Queue[str | None]] = []

    def list(self) -> list[dict[str, Any]]:
        return [session.to_dict() for session in self.sessions]

    async def start(
        self, command: str, args: dict[str, Any], *, stdin_pipe: bool = False
    ) -> Session:
        del stdin_pipe
        self.start_entered.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        session = Session(command, args)
        session.status = "running"
        self.sessions.append(session)
        return session

    def get(self, session_id: str) -> Session | None:
        return next((s for s in self.sessions if s.id == session_id), None)

    def subscribe(self, _session: Session) -> asyncio.Queue[str | None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.queues.append(queue)
        return queue

    def unsubscribe(self, _session: Session, _queue: asyncio.Queue[str | None]) -> None:
        pass

    async def stop(self, _session_id: str) -> bool:
        return True

    async def shutdown(self) -> None:
        pass


class _Runner:
    def __init__(self, running: bool = False) -> None:
        self.running = running
        self.session: Session | None = None
        self.starts = 0

    def is_running(self) -> bool:
        return self.running

    def start(
        self,
        command: str,
        args: dict[str, Any],
        **_kwargs: Any,
    ) -> Session:
        self.starts += 1
        self.session = Session(command, args)
        self.session.status = "running"
        self.running = True
        return self.session

    def current(self) -> Session | None:
        return self.session

    def get(self, session_id: str) -> Session | None:
        return self.session if self.session and self.session.id == session_id else None

    async def shutdown(self) -> None:
        pass


class _Updater:
    def __init__(self, is_idle: Any) -> None:
        self._is_idle = is_idle
        self.blocked = False
        self.active = False
        self.start_calls = 0
        self.provision_calls = 0
        self.version = "test"
        self.commit = "test"
        self.release_install = True

    @property
    def launches_blocked(self) -> bool:
        return self.blocked

    @property
    def maintenance_active(self) -> bool:
        return self.active

    def launch_block_reason(self) -> str | None:
        if not self.blocked:
            return None
        return "server maintenance is in progress"

    def ensure_provisioned(self) -> None:
        self.provision_calls += 1

    async def status(self, *, force: bool = False) -> dict[str, Any]:
        del force
        return {"state": "updating" if self.active else "idle"}

    def start(self) -> tuple[bool, str | None]:
        self.start_calls += 1
        if self.active:
            return False, "server maintenance is already in progress"
        if not self._is_idle():
            return False, "server is busy; stop the running operation first"
        self.blocked = True
        self.active = True
        return True, None


class _FakeBridgeProcess:
    def __init__(self, *, kill_stops: bool) -> None:
        self._alive = True
        self._kill_stops = kill_stops
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        if self._kill_stops:
            self._alive = False


class _FakeBridgeQueue:
    def __init__(self) -> None:
        self.closed = False
        self.cancelled_join = False

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        self.cancelled_join = True


class _FakeClosingArm:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.failure is not None:
            raise self.failure


class _CancellationResistantClosingArm:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_calls = 0
        self.active = 0
        self.max_active = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.close_calls == 1:
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    # Model a driver coroutine that must finish native cleanup
                    # before cancellation can unwind.
                    await self.release.wait()
        finally:
            self.active -= 1


def _test_app(
    manager: _Manager,
    runner: _Runner,
    robot: _Robot | None = None,
    settings: _Settings | None = None,
    updater: _Updater | None = None,
) -> Any:
    robot = robot or _Robot()
    settings = settings or _Settings()
    updater = updater or _Updater(lambda: True)

    def make_updater(is_idle: Any) -> _Updater:
        updater._is_idle = is_idle
        return updater

    with (
        patch.object(app_module, "SessionManager", return_value=manager),
        patch.object(app_module, "OperationRunner", return_value=runner),
        patch.object(app_module, "SettingsStore", return_value=settings),
        patch.object(app_module, "RobotLink", return_value=robot),
        patch.object(
            app_module,
            "SelfUpdater",
            side_effect=make_updater,
        ),
    ):
        return app_module.create_app()


class SessionReservationApiTest(unittest.IsolatedAsyncioTestCase):
    async def _client(
        self, manager: _Manager, runner: _Runner, robot: _Robot | None = None
    ) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=_test_app(manager, runner, robot))
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_operation_refuses_active_diagnostic_session(self) -> None:
        diagnostic = Session("tracker.identify", {})
        diagnostic.status = "running"
        manager = _Manager([diagnostic])
        runner = _Runner()
        async with await self._client(manager, runner) as client:
            response = await client.post(
                "/api/op/start", json={"op": "teleop", "args": {"sim": True}}
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("setup or diagnostics session", response.json()["error"])
        self.assertEqual(runner.starts, 0)

    async def test_can_inventory_reports_configured_profile_presence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            settings.update(
                values={
                    "robot.left_channel": "bench-axol",
                    "robot.right_channel": "null",
                    "mantis.left_channel": "rig-left",
                    "mantis.right_channel": "rig-right",
                }
            )
            interfaces = [
                {"name": "bench-axol", "up": False},
                {"name": "rig-left", "up": True},
                {"name": "rig-right", "up": False},
                {"name": "can0", "up": True},
            ]
            app = _test_app(_Manager(), _Runner(), settings=settings)
            transport = httpx.ASGITransport(app=app)
            with patch.object(
                app_module, "_list_can_interfaces", return_value=interfaces
            ):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.get("/api/can/interfaces")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["interfaces"], interfaces)
        self.assertEqual(
            body["profiles"],
            {
                "axol": {
                    "channels": {"left": "bench-axol", "right": None},
                    "present": True,
                    "up": False,
                    "automaticConnectSuppressed": False,
                },
                "mantis": {
                    "channels": {"left": "rig-left", "right": "rig-right"},
                    "present": True,
                    "up": False,
                    "automaticConnectSuppressed": False,
                },
            },
        )

    async def test_can_inventory_bootstraps_only_exact_persisted_usb_profiles(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            app = _test_app(_Manager(), _Runner(), settings=settings)
            transport = httpx.ASGITransport(app=app)
            with (
                patch.object(app_module, "_list_can_interfaces", return_value=[]),
                patch.object(
                    app_module,
                    "_attached_configured_hub_profiles",
                    return_value={"axol", "mantis"},
                ),
            ):
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.get("/api/can/interfaces")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["profiles"],
            {
                "axol": {
                    "channels": {
                        "left": "can_alm_axol_l",
                        "right": "can_alm_axol_r",
                    },
                    "present": True,
                    "up": False,
                    "automaticConnectSuppressed": False,
                },
                "mantis": {
                    "channels": {
                        "left": "can_mantis_l",
                        "right": "can_mantis_r",
                    },
                    "present": True,
                    "up": False,
                    "automaticConnectSuppressed": False,
                },
            },
        )

    def test_can_profile_presence_requires_the_complete_configured_map(self) -> None:
        interfaces = [{"name": "left", "up": True}]

        self.assertEqual(
            app_module._can_profile_presence(
                interfaces, ("left", "right"), require_both=False
            )["present"],
            False,
        )
        self.assertEqual(
            app_module._can_profile_presence(
                interfaces, ("left", None), require_both=False
            )["present"],
            True,
        )
        self.assertEqual(
            app_module._can_profile_presence(
                interfaces, ("left", None), require_both=True
            )["present"],
            False,
        )
        self.assertEqual(
            app_module._can_profile_presence(
                [{"name": "can0", "up": True}],
                ("configured-left", None),
                require_both=False,
            )["present"],
            False,
        )
        self.assertEqual(
            app_module._can_profile_presence(
                interfaces, ("left", "left"), require_both=True
            )["present"],
            False,
        )
        self.assertFalse(
            app_module._can_profile_presence(
                [],
                ("default-left", None),
                require_both=False,
                configured_usb_present=True,
                profile_channels=("default-left", "default-right"),
            )["present"]
        )
        self.assertTrue(
            app_module._can_profile_presence(
                [],
                ("default-left", "default-right"),
                require_both=False,
                configured_usb_present=True,
                profile_channels=("default-left", "default-right"),
            )["present"]
        )
        self.assertFalse(
            app_module._can_profile_presence(
                [],
                ("custom-left", None),
                require_both=False,
                configured_usb_present=True,
                profile_channels=("default-left", "default-right"),
            )["present"]
        )

    async def test_gripperless_survey_rejects_gripper_enabled_request(self) -> None:
        manager = _Manager()
        runner = _Runner()
        robot = _Robot(has_gripper=False)
        async with await self._client(manager, runner, robot) as client:
            response = await client.post(
                "/api/op/start",
                json={
                    "op": "teleop",
                    "args": {"axol.has_gripper": True},
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("connected Axol survey is gripperless", response.json()["error"])
        self.assertEqual(runner.starts, 0)
        self.assertEqual(robot.releases, 0)

    async def test_update_barrier_blocks_every_hardware_and_session_launch(
        self,
    ) -> None:
        manager = _Manager()
        runner = _Runner()
        robot = _Robot()
        updater = _Updater(lambda: True)
        app = _test_app(manager, runner, robot, updater=updater)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            started = await client.post("/api/update/start")
            requests = (
                client.post(
                    "/api/op/start",
                    json={"op": "teleop", "args": {"sim": True}},
                ),
                client.post(
                    "/api/diagnostics/run",
                    json={"command": "diag.zed-cable", "args": {}},
                ),
                client.post("/api/run", json={"command": "diag.zed-cable", "args": {}}),
                client.post("/api/robot/connect"),
                client.post("/api/robot/disconnect"),
                client.get("/api/can/interfaces"),
                client.get("/api/robot/motors/left/shoulder_1"),
                client.get("/api/cameras/detect"),
                client.get("/api/cameras/preview/123"),
                client.post("/api/cameras/restart-daemon"),
                client.get("/api/tracker/bindings"),
                client.get("/api/usb/status"),
                client.post("/api/usb/connect"),
                client.post("/api/usb/proximity", json={"disabled": True}),
                client.post("/api/host/restart"),
                client.post("/api/host/shutdown"),
            )
            responses = [await request for request in requests]

        self.assertEqual(started.status_code, 200)
        self.assertEqual(updater.start_calls, 1)
        for response in responses:
            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("maintenance", response.json()["error"])
        self.assertEqual(runner.starts, 0)
        self.assertEqual(manager.sessions, [])
        self.assertEqual(robot.connects, 0)
        self.assertEqual(robot.disconnects, 0)

    async def test_update_start_is_atomic_with_session_launch(self) -> None:
        manager = _Manager()
        manager.start_gate = asyncio.Event()
        runner = _Runner()
        updater = _Updater(lambda: True)
        app = _test_app(manager, runner, updater=updater)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            launch = asyncio.create_task(
                client.post(
                    "/api/diagnostics/run",
                    json={"command": "diag.zed-cable", "args": {}},
                )
            )
            await manager.start_entered.wait()
            update_request = asyncio.create_task(client.post("/api/update/start"))
            await asyncio.sleep(0)
            self.assertEqual(updater.start_calls, 0)

            manager.start_gate.set()
            launched = await launch
            update_response = await update_request

            manager.sessions[0].status = "exited"
            while not manager.queues:
                await asyncio.sleep(0)
            await manager.queues[0].put(None)
            await asyncio.sleep(0)

        self.assertEqual(launched.status_code, 200)
        self.assertEqual(update_response.status_code, 409)
        self.assertIn("busy", update_response.json()["error"])
        self.assertFalse(updater.blocked)

    async def test_hosted_dataset_listing_ignores_custom_root_and_symlink(self) -> None:
        boundary = Path("/var/lib/almond-axol/datasets")
        with tempfile.TemporaryDirectory() as directory:
            linked_root = Path(directory) / "custom-root"
            linked_root.symlink_to("/etc", target_is_directory=True)
            settings = _Settings(str(linked_root))
            with (
                patch.object(
                    app_module, "privileged_service_active", return_value=True
                ),
                patch.object(
                    app_module,
                    "validated_service_dataset_root",
                    return_value=boundary,
                ),
                patch(
                    "almond_axol.recording.datasets.list_datasets",
                    return_value=[],
                ) as list_datasets,
            ):
                app = _test_app(_Manager(), _Runner(), settings=settings)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    response = await client.get("/api/datasets")

        self.assertEqual(response.status_code, 200)
        list_datasets.assert_called_once_with(boundary)

    async def test_root_create_app_marks_privileged_embedding(self) -> None:
        with (
            patch.object(app_module.os, "geteuid", return_value=0),
            patch.object(app_module.os, "umask") as umask,
            patch.object(app_module, "mark_privileged_service") as mark_service,
        ):
            _test_app(_Manager(), _Runner())

        mark_service.assert_called_once_with()
        umask.assert_called_once_with(0o027)

    async def test_robot_link_changes_refuse_active_operation(self) -> None:
        manager = _Manager()
        runner = _Runner(running=True)
        robot = _Robot()
        async with await self._client(manager, runner, robot) as client:
            connect = await client.post("/api/robot/connect")
            disconnect = await client.post("/api/robot/disconnect")

        self.assertEqual(connect.status_code, 409)
        self.assertEqual(disconnect.status_code, 409)
        self.assertEqual(robot.connects, 0)
        self.assertEqual(robot.disconnects, 0)

    async def test_robot_link_changes_refuse_diagnostic_cleanup_window(self) -> None:
        diagnostic = Session("motor.info", {})
        diagnostic.status = "running"
        manager = _Manager([diagnostic])
        runner = _Runner()
        robot = _Robot()
        async with await self._client(manager, runner, robot) as client:
            connect = await client.post("/api/robot/connect")
            disconnect = await client.post("/api/robot/disconnect")

        self.assertEqual(connect.status_code, 409)
        self.assertEqual(disconnect.status_code, 409)
        self.assertEqual(robot.connects, 0)
        self.assertEqual(robot.disconnects, 0)

    async def test_manual_disconnect_suppresses_same_automatic_target_server_wide(
        self,
    ) -> None:
        robot = _Robot(channels=("can-left", "can-right"), profile="axol")
        app = _test_app(_Manager(), _Runner(), robot)
        transport = httpx.ASGITransport(app=app)
        with (
            patch.object(
                app_module,
                "_list_can_interfaces",
                return_value=[
                    {"name": "can-left", "up": True},
                    {"name": "can-right", "up": True},
                ],
            ),
            patch.object(
                app_module, "_attached_configured_hub_profiles", return_value=set()
            ),
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                disconnected = await client.post("/api/robot/disconnect")
                automatic = await client.post(
                    "/api/robot/connect",
                    json={"profile": "axol", "automatic": True},
                )
                inventory = await client.get("/api/can/interfaces")

        self.assertEqual(disconnected.status_code, 200)
        self.assertEqual(automatic.status_code, 409)
        self.assertTrue(automatic.json()["automaticConnectSuppressed"])
        self.assertTrue(
            inventory.json()["profiles"]["axol"]["automaticConnectSuppressed"]
        )
        self.assertEqual(robot.connects, 0)

    async def test_changed_automatic_target_releases_manual_disconnect_pause(
        self,
    ) -> None:
        robot = _Robot(channels=("can-left", "can-right"), profile="axol")
        async with await self._client(_Manager(), _Runner(), robot) as client:
            disconnected = await client.post("/api/robot/disconnect")
            changed_target = await client.post(
                "/api/robot/connect",
                json={"profile": "mantis", "automatic": True},
            )

        self.assertEqual(disconnected.status_code, 200)
        self.assertEqual(changed_target.status_code, 200)
        self.assertTrue(changed_target.json()["connected"])
        self.assertEqual(changed_target.json()["profile"], "mantis")
        self.assertEqual(robot.connects, 1)

    async def test_changed_saved_map_releases_manual_disconnect_pause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            robot = _Robot(channels=settings.can_channels(), profile="axol")
            app = _test_app(_Manager(), _Runner(), robot, settings)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post("/api/robot/disconnect")
                settings.update(
                    values={
                        "robot.left_channel": "replacement-left",
                        "robot.right_channel": "replacement-right",
                    }
                )
                changed_map = await client.post(
                    "/api/robot/connect",
                    json={"profile": "axol", "automatic": True},
                )

        self.assertEqual(changed_map.status_code, 200)
        self.assertEqual(
            changed_map.json()["channels"],
            {"left": "replacement-left", "right": "replacement-right"},
        )
        self.assertEqual(robot.connects, 1)

    async def test_explicit_connect_releases_manual_disconnect_pause(self) -> None:
        robot = _Robot(channels=("can-left", "can-right"), profile="axol")
        async with await self._client(_Manager(), _Runner(), robot) as client:
            await client.post("/api/robot/disconnect")
            manual = await client.post("/api/robot/connect", json={"profile": "axol"})
            automatic = await client.post(
                "/api/robot/connect",
                json={"profile": "axol", "automatic": True},
            )

        self.assertEqual(manual.status_code, 200)
        self.assertEqual(automatic.status_code, 200)
        self.assertEqual(robot.connects, 2)

    async def test_rejected_manual_connect_does_not_clear_disconnect_pause(
        self,
    ) -> None:
        robot = _Robot(channels=("can-left", "can-right"), profile="axol")
        async with await self._client(_Manager(), _Runner(), robot) as client:
            await client.post("/api/robot/disconnect")
            rejected = await client.post(
                "/api/robot/connect",
                json={
                    "profile": "axol",
                    "channelsSet": True,
                    "leftChannel": "can-same",
                    "rightChannel": "can-same",
                },
            )
            automatic = await client.post(
                "/api/robot/connect",
                json={"profile": "axol", "automatic": True},
            )

        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(automatic.status_code, 409)
        self.assertEqual(robot.connects, 0)

    async def test_rejected_manual_disconnect_does_not_pause_automatic_connect(
        self,
    ) -> None:
        runner = _Runner(running=True)
        robot = _Robot(channels=("can-left", "can-right"), profile="axol")
        async with await self._client(_Manager(), runner, robot) as client:
            rejected = await client.post("/api/robot/disconnect")
            runner.running = False
            automatic = await client.post(
                "/api/robot/connect",
                json={"profile": "axol", "automatic": True},
            )

        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(automatic.status_code, 200)
        self.assertEqual(robot.connects, 1)

    async def test_duplicate_axol_channels_are_rejected_without_bus_actions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            robot = _Robot()
            app = _test_app(_Manager(), _Runner(), robot, settings)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                settings_response = await client.put(
                    "/api/settings",
                    json={
                        "values": {
                            "robot.left_channel": "can-shared",
                            "robot.right_channel": "can-shared",
                        }
                    },
                )
                connect_response = await client.post(
                    "/api/robot/connect",
                    json={
                        "profile": "axol",
                        "channelsSet": True,
                        "leftChannel": "can-shared",
                        "rightChannel": "can-shared",
                    },
                )

        self.assertEqual(settings_response.status_code, 400)
        self.assertEqual(connect_response.status_code, 400)
        self.assertIn("distinct interfaces", settings_response.json()["error"])
        self.assertIn("distinct interfaces", connect_response.json()["error"])
        self.assertEqual(settings.snapshot()["values"], {})
        self.assertEqual(robot.disconnects, 0)
        self.assertEqual(robot.set_channel_calls, 0)
        self.assertEqual(robot.connects, 0)

    async def test_diagnostic_refuses_active_operation(self) -> None:
        manager = _Manager()
        runner = _Runner(running=True)
        async with await self._client(manager, runner) as client:
            response = await client.post(
                "/api/diagnostics/run",
                json={"command": "tracker.identify", "args": {"backend": "survive"}},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("operation is running", response.json()["error"])
        self.assertEqual(manager.sessions, [])

        # The legacy plain-command endpoint must not bypass the same owner
        # reservation.
        async with await self._client(_Manager(), _Runner(running=True)) as client:
            response = await client.post(
                "/api/run",
                json={"command": "tracker.identify", "args": {"backend": "survive"}},
            )
        self.assertEqual(response.status_code, 409)

    async def test_subprocess_uses_effective_args_for_checks_session_and_history(
        self,
    ) -> None:
        manager = _Manager()
        runner = _Runner()
        robot = _Robot()
        settings = _Settings(
            merged_values={
                "cycles": 99,
                "left_channel": "can-left",
                "right_channel": "can-right",
            }
        )
        effective = {
            "cycles": 3,
            "left_channel": "can-left",
            "right_channel": "can-right",
        }

        def begin(
            session_id: str, command: str, args: dict[str, Any]
        ) -> dict[str, Any]:
            return {"sessionId": session_id, "command": command, "args": args}

        with (
            patch.object(
                app_module.DiagnosticsRunStore, "begin", side_effect=begin
            ) as begin_mock,
            patch.object(app_module.DiagnosticsRunStore, "finalize") as finalize,
            patch.object(app_module, "scoped_motor_faults", return_value=[]) as scoped,
        ):
            app = _test_app(manager, runner, robot, settings)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/diagnostics/run",
                    json={"command": "diag.lift-cycle", "args": {"cycles": 3}},
                )

            self.assertEqual(response.status_code, 200)
            session = manager.sessions[0]
            self.assertEqual(session.args, effective)
            self.assertEqual(response.json()["session"]["args"], effective)
            self.assertEqual(response.json()["run"]["args"], effective)
            fault_scope = {
                "joints": (
                    "SHOULDER_1,SHOULDER_2,SHOULDER_3,ELBOW,WRIST_1,WRIST_2,WRIST_3"
                ),
            }
            scoped.assert_called_once_with([], fault_scope)
            begin_mock.assert_called_once_with(session.id, "diag.lift-cycle", effective)

            # Let the background watcher restore the shared CAN reservation.
            session.status = "exited"
            await asyncio.sleep(0)
            for queue in manager.queues:
                queue.put_nowait(None)
            for _ in range(20):
                await asyncio.sleep(0)
                if finalize.called:
                    break

        self.assertTrue(finalize.called)

    async def test_lift_cycle_unknown_arm_arg_cannot_hide_other_side_fault(
        self,
    ) -> None:
        manager = _Manager()
        robot = _Robot()
        robot.motor_faults = Mock(
            return_value=[
                {
                    "arm": "right",
                    "joint": "SHOULDER_1",
                    "problem": "unreachable",
                    "temperature": None,
                }
            ]
        )
        settings = _Settings(
            merged_values={
                "left_channel": "can-left",
                "right_channel": "can-right",
            }
        )
        app = _test_app(manager, _Runner(), robot, settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/diagnostics/run",
                json={
                    "command": "diag.lift-cycle",
                    "args": {"cycles": 1, "arm": "left"},
                },
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("right shoulder_1 (unreachable)", response.json()["error"])
        self.assertEqual(manager.sessions, [])
        self.assertEqual(robot.releases, 0)

    async def test_unknown_args_cannot_hide_faults_for_other_motor_commands(
        self,
    ) -> None:
        cases = (
            (
                "diag.mantis-trigger",
                "mantis",
                {"joints": "SHOULDER_1"},
                "GRIPPER",
            ),
            ("diag.rom-enable", "axol", {"arm": "left"}, "SHOULDER_1"),
        )
        for command, profile, injected, joint in cases:
            with self.subTest(command=command):
                manager = _Manager()
                robot = _Robot(profile=profile)
                robot.motor_faults = Mock(
                    return_value=[
                        {
                            "arm": "right",
                            "joint": joint,
                            "problem": "unreachable",
                            "temperature": None,
                        }
                    ]
                )
                app = _test_app(manager, _Runner(), robot)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/api/diagnostics/run",
                        json={"command": command, "args": injected},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn(
                    f"right {joint.lower()} (unreachable)", response.json()["error"]
                )
                self.assertEqual(manager.sessions, [])
                self.assertEqual(robot.releases, 0)

    async def test_motor_subprocess_rejects_profile_and_channel_overrides(
        self,
    ) -> None:
        cases = (
            (
                "diag.rom-enable",
                {"target": "axol"},
                "does not match the connected Mantis profile",
            ),
            (
                "diag.rom-enable",
                {"target": "mantis", "left_channel": "can-wrong"},
                "ROM's left CAN channel override",
            ),
            (
                "diag.mantis-trigger",
                {"right_channel": "can-wrong"},
                "Mantis trigger's right CAN channel override",
            ),
        )
        for command, args, message in cases:
            with self.subTest(command=command, args=args):
                manager = _Manager()
                robot = _Robot(
                    profile="mantis",
                    channels=("can-mantis-left", "can-mantis-right"),
                )
                async with await self._client(manager, _Runner(), robot) as client:
                    response = await client.post(
                        "/api/diagnostics/run",
                        json={"command": command, "args": args},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn(message, response.json()["error"])
                self.assertEqual(manager.sessions, [])
                self.assertEqual(robot.releases, 0)

    def test_motor_subprocess_args_are_derived_from_connected_survey(self) -> None:
        now = time.time()
        mantis = _Robot(
            profile="mantis",
            channels=("can-mantis-left", "can-mantis-right"),
            last_ping=now,
        ).status()
        prepared, error = app_module._prepare_motor_launch_args(
            "diag.rom-enable", mantis, {}, now=now
        )
        self.assertIsNone(error)
        self.assertEqual(
            prepared,
            {
                "target": "mantis",
                "joints": "gripper",
                "left_channel": "can-mantis-left",
                "right_channel": "can-mantis-right",
            },
        )

        prepared, error = app_module._prepare_motor_launch_args(
            "diag.mantis-trigger", mantis, {}, now=now
        )
        self.assertIsNone(error)
        self.assertEqual(prepared["left_channel"], "can-mantis-left")
        self.assertEqual(prepared["right_channel"], "can-mantis-right")

        axol = _Robot(
            channels=(None, "can-bench-right"),
            last_ping=now,
        ).status()
        prepared, error = app_module._prepare_motor_launch_args(
            "diag.rom-disable", axol, {}, now=now
        )
        self.assertIsNone(error)
        self.assertTrue(prepared["no_left"])
        self.assertNotIn("left_channel", prepared)
        self.assertEqual(prepared["right_channel"], "can-bench-right")

        prepared, error = app_module._prepare_motor_launch_args(
            "motor.set-zero-pos",
            axol,
            {"arm": "right", "id": 1},
            now=now,
        )
        self.assertIsNone(error)
        self.assertEqual(prepared["channel"], "can-bench-right")

    async def test_axol_only_commands_are_rejected_for_mantis_profile(self) -> None:
        for command in (
            "diag.lift-cycle",
            "lift.home",
            "lift.goto",
            "motor.set-zero-pos",
        ):
            with self.subTest(command=command):
                manager = _Manager()
                robot = _Robot(profile="mantis")
                async with await self._client(manager, _Runner(), robot) as client:
                    response = await client.post(
                        "/api/diagnostics/run",
                        json={"command": command, "args": {}},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn(
                    "requires the axol hardware profile", response.json()["error"]
                )
                self.assertEqual(manager.sessions, [])

    async def test_lift_cycle_requires_current_matching_axol_survey(self) -> None:
        settings = _Settings(
            merged_values={
                "left_channel": "can-left",
                "right_channel": "can-right",
            }
        )
        cases = (
            (_Robot(state="disconnected"), "Connect the Axol robot link"),
            (_Robot(last_ping=1.0), "survey is stale"),
            (
                _Robot(channels=("can-other", "can-right")),
                "do not match the connected Axol survey",
            ),
        )
        for robot, message in cases:
            with self.subTest(message=message):
                manager = _Manager()
                app = _test_app(manager, _Runner(), robot, settings)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/api/diagnostics/run",
                        json={"command": "diag.lift-cycle", "args": {"cycles": 1}},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn(message, response.json()["error"])
                self.assertEqual(manager.sessions, [])
                self.assertEqual(robot.releases, 0)

    def test_lift_cycle_survey_match_respects_skipped_side(self) -> None:
        robot = _Robot(channels=("can-left", "can-right"))
        status = robot.status()
        error = app_module._lift_cycle_link_error(
            status,
            {"no_left": True, "right_channel": "can-right"},
            now=status["lastPing"],
        )
        self.assertIsNone(error)

        error = app_module._lift_cycle_link_error(
            status,
            {"no_left": True, "right_channel": "can-other"},
            now=status["lastPing"],
        )
        self.assertIn("do not match", error or "")

        error = app_module._lift_cycle_link_error(
            status,
            {"no_left": True, "no_right": True},
            now=status["lastPing"],
        )
        self.assertIn("at least one", error or "")

    async def test_motor_operation_requires_current_connected_survey(self) -> None:
        cases = (
            (
                _Robot(state="disconnected"),
                {},
                "Connect the Axol robot link",
            ),
            (_Robot(last_ping=1.0), {}, "survey is stale"),
            (_Robot(last_ping=None), {}, "survey is stale"),
            (
                _Robot(),
                {"left_channel": "can-other"},
                "effective Axol CAN mapping does not match",
            ),
            (
                _Robot(
                    profile="mantis",
                    channels=("can_mantis_l", "can_mantis_r"),
                    last_ping=None,
                ),
                {"mantis": True},
                "Mantis robot survey is stale",
            ),
        )
        for robot, args, message in cases:
            with self.subTest(message=message):
                manager = _Manager()
                runner = _Runner()
                async with await self._client(manager, runner, robot) as client:
                    response = await client.post(
                        "/api/op/start",
                        json={"op": "teleop", "args": args},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn(message, response.json()["error"])
                self.assertEqual(runner.starts, 0)

    async def test_mantis_string_booleans_cannot_bypass_mantis_survey(self) -> None:
        for value in ("yes", "on", "TRUE"):
            with self.subTest(value=value):
                manager = _Manager()
                runner = _Runner()
                robot = _Robot(profile="axol")
                async with await self._client(manager, runner, robot) as client:
                    response = await client.post(
                        "/api/op/start",
                        json={"op": "teleop", "args": {"mantis": value}},
                    )

                self.assertEqual(response.status_code, 409)
                self.assertIn("Connect the Mantis robot link", response.json()["error"])
                self.assertEqual(runner.starts, 0)
                self.assertEqual(robot.releases, 0)

    async def test_malformed_boolean_is_rejected_before_hardware_checks(self) -> None:
        for value in (1, "1", "maybe"):
            with self.subTest(value=value):
                manager = _Manager()
                runner = _Runner()
                robot = _Robot()
                async with await self._client(manager, runner, robot) as client:
                    response = await client.post(
                        "/api/op/start",
                        json={"op": "teleop", "args": {"mantis": value}},
                    )

                self.assertEqual(response.status_code, 400)
                self.assertIn("mantis must be a boolean", response.json()["error"])
                self.assertEqual(runner.starts, 0)
                self.assertEqual(robot.releases, 0)

    async def test_robot_free_string_booleans_skip_axol_survey_consistently(
        self,
    ) -> None:
        for key, value in (("sim", "yes"), ("cart_only", "on")):
            with self.subTest(key=key, value=value):
                manager = _Manager()
                runner = _Runner()
                robot = _Robot(state="disconnected", last_ping=None)
                async with await self._client(manager, runner, robot) as client:
                    response = await client.post(
                        "/api/op/start",
                        json={"op": "teleop", "args": {key: value}},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(runner.starts, 1)
                assert runner.session is not None
                self.assertIs(runner.session.args[key], True)

    async def test_motor_operation_binds_launch_to_surveyed_channels(self) -> None:
        manager = _Manager()
        runner = _Runner()
        robot = _Robot(channels=("can-bench-left", "can-bench-right"))
        settings = _Settings(
            merged_values={
                "left_channel": "can-bench-left",
                "right_channel": "can-bench-right",
            }
        )
        app = _test_app(manager, runner, robot, settings)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/op/start",
                json={"op": "teleop", "args": {}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(runner.starts, 1)
        assert runner.session is not None
        self.assertEqual(runner.session.args["left_channel"], "can-bench-left")
        self.assertEqual(runner.session.args["right_channel"], "can-bench-right")

    async def test_subprocess_release_failure_is_actionable_and_fail_closed(
        self,
    ) -> None:
        manager = _Manager()
        robot = _Robot()
        robot.release_error = RuntimeError("CAN close timed out")
        async with await self._client(manager, _Runner(), robot) as client:
            response = await client.post(
                "/api/diagnostics/run",
                json={"command": "can.setup", "args": {}},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("command was not started", response.json()["error"])
        self.assertIn("hardware remains locked", response.json()["error"])
        self.assertIn("CAN close timed out", response.json()["error"])
        self.assertEqual(manager.sessions, [])
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.reacquires, 0)

    async def test_operation_release_failure_is_actionable_and_fail_closed(
        self,
    ) -> None:
        manager = _Manager()
        robot = _Robot()
        robot.release_error = RuntimeError("CAN close timed out")
        runner = OperationRunner(robot_link=robot)
        with (
            patch.object(runner, "_build_config", return_value=SimpleNamespace()),
            patch.object(runner, "_attach_cameras_to_teleop"),
        ):
            app = _test_app(manager, runner, robot)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/op/start",
                    json={"op": "teleop", "args": {}},
                )

        self.assertEqual(response.status_code, 409)
        self.assertIn("operation was not started", response.json()["error"])
        self.assertIn("hardware remains locked", response.json()["error"])
        self.assertIn("CAN close timed out", response.json()["error"])
        self.assertEqual(response.json()["session"]["status"], "error")
        self.assertEqual(robot.releases, 1)
        self.assertEqual(robot.reacquires, 0)
        self.assertIsNone(runner._thread)

    async def test_diagnostic_start_race_cannot_overlap_operation(self) -> None:
        manager = _Manager()
        manager.start_gate = asyncio.Event()
        runner = _Runner()
        async with await self._client(manager, runner) as client:
            diagnostic_request = asyncio.create_task(
                client.post(
                    "/api/diagnostics/run",
                    json={
                        "command": "tracker.identify",
                        "args": {"backend": "survive"},
                    },
                )
            )
            await asyncio.wait_for(manager.start_entered.wait(), timeout=1.0)

            operation_request = asyncio.create_task(
                client.post(
                    "/api/op/start",
                    json={"op": "teleop", "args": {"sim": True}},
                )
            )
            await asyncio.sleep(0)
            self.assertFalse(operation_request.done())

            manager.start_gate.set()
            diagnostic_response, operation_response = await asyncio.gather(
                diagnostic_request, operation_request
            )

            self.assertEqual(diagnostic_response.status_code, 200)
            self.assertEqual(operation_response.status_code, 409)
            self.assertEqual(runner.starts, 0)

            # Let the background watcher finish instead of leaking a task into
            # the isolated event-loop teardown.
            manager.sessions[0].status = "exited"
            await asyncio.sleep(0)
            for queue in manager.queues:
                queue.put_nowait(None)
            await asyncio.sleep(0)

    async def test_operation_start_race_cannot_overlap_diagnostic(self) -> None:
        manager = _Manager()
        runner = _Runner()
        robot = _Robot()
        robot.fault_check_gate = threading.Event()
        async with await self._client(manager, runner, robot) as client:
            operation_request = asyncio.create_task(
                client.post("/api/op/start", json={"op": "teleop", "args": {}})
            )
            entered = await asyncio.to_thread(robot.fault_check_entered.wait, 1.0)
            self.assertTrue(entered)

            diagnostic_request = asyncio.create_task(
                client.post(
                    "/api/diagnostics/run",
                    json={
                        "command": "tracker.identify",
                        "args": {"backend": "survive"},
                    },
                )
            )
            await asyncio.sleep(0)
            self.assertFalse(diagnostic_request.done())

            robot.fault_check_gate.set()
            operation_response, diagnostic_response = await asyncio.gather(
                operation_request, diagnostic_request
            )

        self.assertEqual(operation_response.status_code, 200)
        self.assertEqual(diagnostic_response.status_code, 409)
        self.assertEqual(manager.sessions, [])


class OperationRunnerOwnershipTest(unittest.TestCase):
    def test_mantis_link_rebinds_when_saved_script_exists_but_netdevs_do_not(
        self,
    ) -> None:
        link = object.__new__(RobotLink)
        link._profile = "mantis"
        link._arms = [
            SimpleNamespace(channel="can_mantis_l"),
            SimpleNamespace(channel="can_mantis_r"),
        ]

        with (
            patch.object(link, "_configured_interfaces_present", return_value=False),
            patch.object(Path, "exists", return_value=True),
            patch("almond_axol.cli.can.setup.ensure_mantis_setup") as ensure_setup,
            patch("almond_axol.cli.can.setup.bring_up_can") as bring_up,
        ):
            link._enable_can()

        ensure_setup.assert_called_once_with()
        bring_up.assert_not_called()

    def test_robot_link_rejects_duplicate_axol_channels_before_arm_creation(
        self,
    ) -> None:
        with (
            patch("almond_axol.serve.robot_link._ArmLink") as arm_link,
            self.assertRaisesRegex(ValueError, "distinct interfaces"),
        ):
            RobotLink("can-shared", "can-shared")

        arm_link.assert_not_called()

    def test_robot_link_set_channels_rejects_duplicate_axol_map(self) -> None:
        link = object.__new__(RobotLink)
        link._profile = "axol"
        link._arms = []
        link._lock = threading.Lock()
        link._state = STATE_DISCONNECTED
        link._buses_may_be_open = False
        link.hub = SimpleNamespace(clear_slow=Mock())

        with self.assertRaisesRegex(ValueError, "distinct interfaces"):
            link.set_channels("can-shared", "can-shared")

        self.assertEqual(link.channels(), (None, None))
        link.hub.clear_slow.assert_not_called()

    def test_robot_link_rejects_duplicate_mantis_channels_before_arm_creation(
        self,
    ) -> None:
        with (
            patch("almond_axol.serve.robot_link._ArmLink") as arm_link,
            self.assertRaisesRegex(ValueError, "distinct interfaces"),
        ):
            RobotLink("can-shared", "can-shared", profile="mantis")

        arm_link.assert_not_called()

    def test_robot_link_set_channels_rejects_duplicate_mantis_map(self) -> None:
        link = object.__new__(RobotLink)
        link._profile = "axol"
        link._arms = []
        link._lock = threading.Lock()
        link._state = STATE_DISCONNECTED
        link._buses_may_be_open = False
        link.hub = SimpleNamespace(clear_slow=Mock())

        with self.assertRaisesRegex(ValueError, "distinct interfaces"):
            link.set_channels("can-shared", "can-shared", profile="mantis")

        self.assertEqual(link.profile(), "axol")
        self.assertEqual(link.channels(), (None, None))
        link.hub.clear_slow.assert_not_called()

    def test_hosted_root_override_is_reflected_in_session_args(self) -> None:
        boundary = Path("/var/lib/almond-axol/datasets")
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsStore(Path(directory) / "settings.json")
            settings.update(values={"recording.root": "/home/operator/legacy"})
            runner = OperationRunner(settings=settings)
            with (
                patch.object(
                    state_files, "privileged_service_active", return_value=True
                ),
                patch.object(
                    state_files,
                    "service_dataset_path_for_repo_id",
                    return_value=boundary / "owner" / "dataset",
                ),
                patch.object(
                    runner,
                    "_build_config",
                    return_value=SimpleNamespace(),
                ),
                patch.object(threading.Thread, "start"),
            ):
                session = runner.start(
                    "replay-dataset",
                    {"repo_id": "owner/dataset", "root": "/etc"},
                )

        self.assertEqual(session.args["root"], str(boundary / "owner" / "dataset"))

    def test_robot_link_disconnect_refuses_busy_owner(self) -> None:
        link = object.__new__(RobotLink)
        link._lock = threading.Lock()
        link._state = STATE_BUSY
        link._submit = Mock()

        with self.assertRaisesRegex(RuntimeError, "task owns"):
            link.disconnect()
        link._submit.assert_not_called()

    def test_robot_link_release_failure_remains_fail_closed_on_retry(self) -> None:
        def timeout(coro: Any) -> None:
            if coro is not None:
                coro.close()
            raise TimeoutError("close timed out")

        link = object.__new__(RobotLink)
        link._lock = threading.Lock()
        link._state = STATE_CONNECTED
        link._error = None
        link._buses_may_be_open = True
        link.hub = SimpleNamespace(push_state=lambda _state: None)
        link._stop_and_close = Mock(return_value=None)
        link._submit = Mock(side_effect=timeout)

        for _attempt in range(2):
            with self.assertRaisesRegex(RuntimeError, "could not release"):
                link.release()
            self.assertEqual(link._state, STATE_ERROR)
            self.assertTrue(link._buses_may_be_open)
        self.assertEqual(link._submit.call_count, 2)

    def test_robot_link_reacquire_propagates_open_failure(self) -> None:
        def timeout(coro: Any) -> None:
            if coro is not None:
                coro.close()
            raise TimeoutError("open timed out")

        link = object.__new__(RobotLink)
        link._lock = threading.Lock()
        link._state = STATE_BUSY
        link._error = None
        link.hub = SimpleNamespace(push_state=lambda _state: None)
        link._submit = Mock(side_effect=timeout)

        with self.assertRaisesRegex(RuntimeError, "could not reacquire"):
            link.reacquire()

        self.assertEqual(link._state, STATE_ERROR)

    def test_robot_link_reopen_invalidates_survey_until_ping_completes(self) -> None:
        async def scenario() -> None:
            ping_attempted = asyncio.Event()

            async def open_arm(_joints: object) -> None:
                return None

            async def fail_ping() -> dict[str, object]:
                ping_attempted.set()
                raise RuntimeError("survey failed")

            async def fail_sample() -> dict[str, object]:
                raise RuntimeError("sample failed")

            arm = SimpleNamespace(
                health={"SHOULDER_1": {"reachable": True}},
                open=open_arm,
                ping=fail_ping,
                sample=fail_sample,
            )
            link = object.__new__(RobotLink)
            link._profile = "axol"
            link._has_gripper_provider = None
            link._active_joints = None
            link._lock = threading.Lock()
            link._lifecycle_lock = asyncio.Lock()
            link._buses_may_be_open = False
            link._arms = [arm]
            link._ping_task = None
            link._sample_task = None
            link._last_ping = time.time()
            link._loop = asyncio.get_running_loop()
            link.hub = SimpleNamespace(clear_slow=Mock(), push_slow=Mock())

            await link._open_and_start()
            await ping_attempted.wait()
            await asyncio.sleep(0)

            self.assertIsNone(link._last_ping)
            self.assertEqual(arm.health, {})
            link.hub.clear_slow.assert_called_once_with()

            for task in (link._ping_task, link._sample_task):
                assert task is not None
                task.cancel()
            await asyncio.gather(
                link._ping_task,
                link._sample_task,
                return_exceptions=True,
            )

        asyncio.run(scenario())

    def test_robot_link_lifecycle_retry_cannot_overlap_cancelled_close(self) -> None:
        async def scenario() -> None:
            arm = _CancellationResistantClosingArm()
            link = object.__new__(RobotLink)
            link._lock = threading.Lock()
            link._buses_may_be_open = True
            link._ping_task = None
            link._sample_task = None
            link._lifecycle_lock = asyncio.Lock()
            link._arms = [arm]

            first = asyncio.create_task(link._stop_and_close())
            await arm.started.wait()
            first.cancel()
            await asyncio.sleep(0)
            retry = asyncio.create_task(link._stop_and_close())
            await asyncio.sleep(0)

            self.assertEqual(arm.close_calls, 1)
            arm.release.set()
            await asyncio.gather(first, retry, return_exceptions=True)
            self.assertEqual(arm.close_calls, 2)
            self.assertEqual(arm.max_active, 1)

        asyncio.run(scenario())

    def test_bus_close_failure_keeps_ownership_and_attempts_every_arm(self) -> None:
        failed = _FakeClosingArm(RuntimeError("close failed"))
        healthy = _FakeClosingArm()
        link = object.__new__(RobotLink)
        link._lock = threading.Lock()
        link._buses_may_be_open = True
        link._ping_task = None
        link._sample_task = None
        link._lifecycle_lock = asyncio.Lock()
        link._arms = [failed, healthy]

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            asyncio.run(link._stop_and_close())

        self.assertEqual(failed.close_calls, 1)
        self.assertEqual(healthy.close_calls, 1)
        self.assertTrue(link._buses_may_be_open)

        failed.failure = None
        asyncio.run(link._stop_and_close())
        self.assertFalse(link._buses_may_be_open)

    def test_robot_release_failure_aborts_axol_and_mantis_before_worker(self) -> None:
        cases = (
            (
                "axol",
                {},
                SimpleNamespace(),
            ),
            (
                "mantis",
                {"mantis": True, "mantis_source": "quest"},
                SimpleNamespace(
                    mantis=True,
                    mantis_source="quest",
                    left_channel="can_mantis_l",
                    right_channel="can_mantis_r",
                ),
            ),
        )
        for profile, args, config in cases:
            with self.subTest(profile=profile):
                robot = _Robot(profile=profile)
                robot.release_error = RuntimeError("CAN close timed out")
                runner = OperationRunner(robot_link=robot)
                with (
                    patch.object(runner, "_build_config", return_value=config),
                    patch.object(runner, "_attach_cameras_to_teleop"),
                    patch("almond_axol.cli.teleop._prepare_mantis_teleop"),
                    patch(
                        "almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"
                    ),
                ):
                    session = runner.start("teleop", args)

                self.assertEqual(session.status, "error")
                self.assertIn("CAN close timed out", session.error or "")
                self.assertEqual(robot.releases, 1)
                self.assertIsNone(runner._thread)

    def test_tracker_bridge_escalates_to_kill_and_reaps(self) -> None:
        process = _FakeBridgeProcess(kill_stops=True)
        command_queue = _FakeBridgeQueue()
        runner = OperationRunner()
        runner._bridge_process = process  # type: ignore[assignment]
        runner._bridge_stop_event = threading.Event()
        runner._bridge_commands = command_queue
        session = Session("teleop", {})
        session.status = "stopping"

        runner._stop_tracker_bridge(session)

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(process.is_alive())
        self.assertTrue(command_queue.closed)
        self.assertTrue(command_queue.cancelled_join)
        self.assertIsNone(runner._bridge_process)

    def test_unkillable_tracker_bridge_retains_busy_ownership(self) -> None:
        process = _FakeBridgeProcess(kill_stops=False)
        runner = OperationRunner()
        runner._bridge_process = process  # type: ignore[assignment]
        runner._bridge_stop_event = threading.Event()
        runner._bridge_commands = _FakeBridgeQueue()
        session = Session("teleop", {})
        session.status = "stopping"

        runner._stop_tracker_bridge(session)

        self.assertEqual(session.status, "error")
        self.assertIn("could not be killed", session.error or "")
        self.assertIs(runner._bridge_process, process)
        self.assertTrue(runner.is_running())

    def test_terminal_status_stays_busy_until_worker_cleanup_finishes(self) -> None:
        runner = OperationRunner()
        session = Session("teleop", {})
        session.status = "error"
        runner._session = session
        runner._thread = type("Worker", (), {"is_alive": lambda self: True})()

        self.assertTrue(runner.is_running())

        runner._thread = type("Worker", (), {"is_alive": lambda self: False})()
        self.assertFalse(runner.is_running())

    def test_uncertain_command_cleanup_skips_reacquire_and_stays_busy(self) -> None:
        robot = _Robot()
        runner = OperationRunner(robot_link=robot)
        session = Session("run-policy", {})
        session.status = "error"
        session.error = "HardwareCleanupError: robot disconnect failed"
        runner._session = session

        runner._finish(session, needs_robot=True, cleanup_uncertain=True)

        self.assertEqual(robot.reacquires, 0)
        self.assertEqual(session.status, "error")
        self.assertTrue(runner.is_running())
        self.assertTrue(
            any("safety lockout" in line for line in session.log), list(session.log)
        )

    def test_thread_command_cleanup_error_enters_safety_lockout(self) -> None:
        from almond_axol.serve.commands import COMMANDS

        def fail(_cfg: Any, *, stop_event: threading.Event) -> None:
            del stop_event
            raise HardwareCleanupError("robot disconnect failed")

        command = SimpleNamespace(
            load_entrypoint=lambda: fail,
            load_episode_control=lambda: None,
        )
        robot = _Robot()
        runner = OperationRunner(robot_link=robot)
        session = Session("cleanup-test", {})
        session.status = "running"
        runner._session = session

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

        self.assertEqual(session.status, "error")
        self.assertEqual(robot.reacquires, 0)
        self.assertTrue(runner.is_running())

    def test_reacquire_failure_is_reported_instead_of_success(self) -> None:
        robot = _Robot()
        robot.reacquire = Mock(side_effect=RuntimeError("open failed"))
        runner = OperationRunner(robot_link=robot)
        session = Session("run-policy", {})
        session.status = "running"

        runner._finish(session, needs_robot=True)

        self.assertEqual(session.status, "error")
        self.assertIn("open failed", session.error or "")
        self.assertFalse(any("link reacquired" in line for line in session.log))

    def test_recording_camera_gate_runs_before_operation_thread(self) -> None:
        runner = OperationRunner()
        config = type(
            "Config",
            (),
            {
                "robot_config": type(
                    "RobotConfig",
                    (),
                    {
                        "cameras": {
                            "left_arm": type(
                                "Camera", (), {"serial": 12345, "record": False}
                            )()
                        }
                    },
                )()
            },
        )()
        with patch.object(runner, "_build_config", return_value=config):
            session = runner.start("collect-data", {"mantis": False})

        self.assertEqual(session.status, "error")
        self.assertIn("recording enabled", session.error or "")
        self.assertIsNone(runner._thread)

    def test_external_tracker_runtime_gate_runs_before_operation_thread(self) -> None:
        runner = OperationRunner()
        config = type(
            "Config",
            (),
            {
                "mantis": True,
                "mantis_source": "lighthouse",
                "left_channel": "can_mantis_l",
                "right_channel": "can_mantis_r",
            },
        )()
        with (
            patch.object(runner, "_build_config", return_value=config),
            patch("almond_axol.cli.teleop._prepare_mantis_teleop"),
            patch(
                "almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness",
                side_effect=RuntimeError("unsupported tracker runtime"),
            ),
        ):
            session = runner.start("teleop", {"mantis": True})

        self.assertEqual(session.status, "error")
        self.assertIn("unsupported tracker runtime", session.error or "")
        self.assertIsNone(runner._thread)

    def test_nested_mantis_robot_profile_cannot_bypass_top_level_mode(self) -> None:
        robot = _Robot(profile="axol")
        runner = OperationRunner(robot_link=robot)
        session = runner.start(
            "collect-data",
            {
                "repo_id": "test/repo",
                "task": "test",
                "mantis": False,
                "robot_config.type": "axol_mantis",
            },
            cameras={
                "record_resolution": "SVGA",
                "serials": {"left_arm": 123},
                "record": {"left_arm": True},
            },
        )

        self.assertEqual(session.status, "error")
        self.assertIn(
            "hardware profile does not match mantis=False", session.error or ""
        )
        self.assertEqual(robot.releases, 0)
        self.assertIsNone(runner._thread)

    def test_config_path_cannot_hide_a_safety_mode_from_launch_classification(
        self,
    ) -> None:
        cases = (
            ("teleop", {}, SimpleNamespace(mantis=True, sim=False, cart_only=False)),
            ("teleop", {}, SimpleNamespace(mantis=False, sim=True, cart_only=False)),
            ("teleop", {}, SimpleNamespace(mantis=False, sim=False, cart_only=True)),
        )
        for op_id, args, config in cases:
            with self.subTest(config=config):
                robot = _Robot(profile="axol")
                runner = OperationRunner(robot_link=robot)
                with patch.object(runner, "_build_config", return_value=config):
                    session = runner.start(op_id, args)

                self.assertEqual(session.status, "error")
                self.assertIn("does not match the submitted", session.error or "")
                self.assertIn("config_path", session.error or "")
                self.assertEqual(robot.releases, 0)
                self.assertIsNone(runner._thread)

    def test_gripperless_survey_rejects_gripper_enabled_operation_config(self) -> None:
        robot = _Robot(profile="axol", has_gripper=False)
        runner = OperationRunner(robot_link=robot)
        config = SimpleNamespace(
            mantis=False,
            sim=False,
            cart_only=False,
            axol=SimpleNamespace(has_gripper=True),
        )
        with (
            patch.object(runner, "_build_config", return_value=config),
            patch.object(runner, "_attach_cameras_to_teleop"),
        ):
            session = runner.start("teleop", {})

        self.assertEqual(session.status, "error")
        self.assertIn("connected Axol survey is gripperless", session.error or "")
        self.assertEqual(robot.releases, 0)
        self.assertIsNone(runner._thread)

    def test_parsed_mantis_source_controls_managed_bridge(self) -> None:
        robot = _Robot(profile="mantis", channels=("can_mantis_l", "can_mantis_r"))
        runner = OperationRunner(robot_link=robot)
        config = SimpleNamespace(
            mantis=True,
            sim=False,
            cart_only=False,
            mantis_source="quest",
            left_channel="can_mantis_l",
            right_channel="can_mantis_r",
        )
        worker = Mock()
        with (
            patch.object(runner, "_build_config", return_value=config),
            patch.object(runner, "_attach_cameras_to_teleop"),
            patch("almond_axol.cli.teleop._prepare_mantis_teleop"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch(
                "almond_axol.serve.runner.threading.Thread", return_value=worker
            ) as thread,
        ):
            session = runner.start("teleop", {"mantis": True})

        self.assertEqual(session.status, "running")
        run_args = thread.call_args.kwargs["args"]
        self.assertFalse(run_args[-1])
        worker.start.assert_called_once_with()

    def test_mantis_string_boolean_still_enables_managed_bridge(self) -> None:
        robot = _Robot(profile="mantis", channels=("can_mantis_l", "can_mantis_r"))
        runner = OperationRunner(robot_link=robot)
        config = SimpleNamespace(
            mantis=True,
            mantis_source="lighthouse",
            left_channel="can_mantis_l",
            right_channel="can_mantis_r",
        )
        worker = Mock()
        with (
            patch.object(runner, "_build_config", return_value=config),
            patch.object(runner, "_attach_cameras_to_teleop"),
            patch("almond_axol.cli.teleop._prepare_mantis_teleop"),
            patch("almond_axol.cli.mantis_bridge.require_mantis_tracker_readiness"),
            patch("almond_axol.cli.mantis_bridge.set_managed_pose_source_id"),
            patch(
                "almond_axol.serve.runner.threading.Thread", return_value=worker
            ) as thread,
        ):
            session = runner.start(
                "teleop", {"mantis": "yes", "mantis_source": "lighthouse"}
            )

        self.assertEqual(session.status, "running")
        self.assertEqual(robot.releases, 1)
        run_args = thread.call_args.kwargs["args"]
        self.assertTrue(run_args[-1])
        worker.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
