from __future__ import annotations

import asyncio
import tempfile
import threading
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
    STATE_ERROR,
)
from almond_axol.serve.runner import OperationRunner
from almond_axol.serve.settings import SettingsStore
from almond_axol.utils import state_files


class _Settings:
    def __init__(self, recording_root: str | None = None) -> None:
        self.recording_root = recording_root

    def can_channels(self) -> tuple[str, str]:
        return "can-left", "can-right"

    def has_gripper(self) -> bool:
        return True

    def snapshot(self) -> dict[str, Any]:
        values = (
            {"recording.root": self.recording_root}
            if self.recording_root is not None
            else {}
        )
        return {"values": values, "cameras": None, "advanced": {}}


class _Robot:
    def __init__(self, *_args: Any, profile: str = "axol", **_kwargs: Any) -> None:
        self._profile = profile
        self.releases = 0
        self.reacquires = 0
        self.connects = 0
        self.disconnects = 0
        self.release_error: Exception | None = None
        self.fault_check_entered = threading.Event()
        self.fault_check_gate: threading.Event | None = None

    def profile(self) -> str:
        return self._profile

    def channels(self) -> tuple[str, str]:
        return "can-left", "can-right"

    def status(self) -> dict[str, Any]:
        return {"state": "connected"}

    def connect(self) -> dict[str, Any]:
        self.connects += 1
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.disconnects += 1
        return {"state": "disconnected"}

    def release(self) -> None:
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error

    def reacquire(self) -> None:
        self.reacquires += 1

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
) -> Any:
    robot = robot or _Robot()
    settings = settings or _Settings()
    with (
        patch.object(app_module, "SessionManager", return_value=manager),
        patch.object(app_module, "OperationRunner", return_value=runner),
        patch.object(app_module, "SettingsStore", return_value=settings),
        patch.object(app_module, "RobotLink", return_value=robot),
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


if __name__ == "__main__":
    unittest.main()
