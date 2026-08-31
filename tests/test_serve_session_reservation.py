from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any
from unittest.mock import patch

import httpx

from almond_axol.serve import app as app_module
from almond_axol.serve.manager import Session
from almond_axol.serve.runner import OperationRunner


class _Settings:
    def __init__(self) -> None:
        pass

    def can_channels(self) -> tuple[str, str]:
        return "can-left", "can-right"

    def has_gripper(self) -> bool:
        return True


class _Robot:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.releases = 0
        self.reacquires = 0
        self.fault_check_entered = threading.Event()
        self.fault_check_gate: threading.Event | None = None

    def profile(self) -> str:
        return "axol"

    def release(self) -> None:
        self.releases += 1

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


def _test_app(manager: _Manager, runner: _Runner, robot: _Robot | None = None) -> Any:
    robot = robot or _Robot()
    with (
        patch.object(app_module, "SessionManager", return_value=manager),
        patch.object(app_module, "OperationRunner", return_value=runner),
        patch.object(app_module, "SettingsStore", return_value=_Settings()),
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
    def test_terminal_status_stays_busy_until_worker_cleanup_finishes(self) -> None:
        runner = OperationRunner()
        session = Session("teleop", {})
        session.status = "error"
        runner._session = session
        runner._thread = type("Worker", (), {"is_alive": lambda self: True})()

        self.assertTrue(runner.is_running())

        runner._thread = type("Worker", (), {"is_alive": lambda self: False})()
        self.assertFalse(runner.is_running())

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
