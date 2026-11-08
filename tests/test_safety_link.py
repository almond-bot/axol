"""The control panel's always-on safety board link (``serve.safety_link``)."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import httpx

from almond_axol.robot.safety import SafetyError, decode_status
from almond_axol.serve import app as app_module
from almond_axol.serve import safety_link as link_module
from almond_axol.serve.safety_link import SafetyLink
from tests.test_serve_session_reservation import _Manager, _Runner, _test_app

_ON = decode_status(bytes([0x8F, 0, 0, 0x02, 0x20, 0x67, 0x90, 0x65]))
_STOPPED = decode_status(bytes([0x81, 0, 1, 0x02, 0x20, 0x67, 0, 0]))


class FakeBoard:
    """Stands in for ``SafetyBoard``; records what the link asks of it."""

    instances: list[FakeBoard] = []
    answers = True

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.status = None
        self.status_age = None
        self.running = False
        self.failure = None
        self.calls: list[str] = []
        FakeBoard.instances.append(self)

    def status_is_fresh(self, max_age_s: float = 1.0) -> bool:
        return self.status is not None

    async def start(self, *, bring_up: bool = True) -> None:
        self.calls.append(f"start(bring_up={bring_up})")
        self.running = True
        if FakeBoard.answers:
            self.status = _ON
            self.status_age = 0.1

    async def close(self) -> None:
        self.calls.append("close")
        self.running = False

    async def stop(self):  # noqa: ANN201
        self.calls.append("stop")
        self.status = _STOPPED
        return _STOPPED

    async def remote_start(self):  # noqa: ANN201
        self.calls.append("remote_start")
        self.status = _ON
        return _ON

    async def power_cycle(self):  # noqa: ANN201
        self.calls.append("power_cycle")
        return _ON


class SafetyLinkTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeBoard.instances = []
        FakeBoard.answers = True
        self.iface_up = True
        patches = (
            patch.object(link_module, "SafetyBoard", FakeBoard),
            patch.object(
                link_module, "resolve_safety_channel", return_value="can-test"
            ),
            patch(
                "almond_axol.cli.can.setup.iface_up",
                side_effect=lambda _c: self.iface_up,
            ),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.link = SafetyLink(start=False)
        self.addCleanup(self.link.shutdown)

    def tick(self) -> None:
        self.link._submit(self.link._tick())

    def test_opens_the_board_when_the_bus_is_up(self) -> None:
        self.tick()
        self.assertEqual(len(FakeBoard.instances), 1)
        self.assertEqual(FakeBoard.instances[0].calls, ["start(bring_up=False)"])
        status = self.link.status()
        self.assertTrue(status["connected"])
        self.assertTrue(status["detected"])
        self.assertTrue(status["reachable"])
        self.assertTrue(status["status"]["outputOn"])
        self.assertEqual(status["status"]["firmware"], "0.2")
        self.assertEqual(status["channel"], "can-test")

    def test_waits_for_the_bus(self) -> None:
        self.iface_up = False
        self.tick()
        self.assertEqual(FakeBoard.instances, [])
        self.assertFalse(self.link.status()["connected"])
        self.assertFalse(self.link.status()["detected"])

    def test_a_jelly_without_a_board_stays_undetected(self) -> None:
        FakeBoard.answers = False
        self.tick()
        self.tick()
        status = self.link.status()
        self.assertTrue(status["connected"])
        self.assertFalse(status["detected"])
        self.assertIsNone(status["status"])
        with self.assertRaisesRegex(SafetyError, "no safety board"):
            self.link.start()

    def test_stop_and_start_go_through_the_board(self) -> None:
        self.tick()
        status = self.link.stop()
        self.assertFalse(status["status"]["outputOn"])
        self.assertEqual(status["status"]["stopCause"], "can")
        status = self.link.start()
        self.assertTrue(status["status"]["outputOn"])
        self.assertEqual(
            FakeBoard.instances[0].calls,
            ["start(bring_up=False)", "stop", "remote_start"],
        )

    def test_power_cycle_goes_through_the_board(self) -> None:
        self.tick()
        status = self.link.power_cycle()
        self.assertTrue(status["status"]["outputOn"])
        self.assertFalse(status["powerCycling"])
        self.assertEqual(FakeBoard.instances[0].calls[-1], "power_cycle")

    def test_power_cycle_needs_a_detected_board(self) -> None:
        FakeBoard.answers = False
        self.tick()
        with self.assertRaisesRegex(SafetyError, "no safety board"):
            self.link.power_cycle()

    def test_stop_without_a_board_tells_the_operator_to_use_the_estop(self) -> None:
        with self.assertRaisesRegex(SafetyError, "Press the E-Stop"):
            self.link.stop()

    def test_reopens_after_the_bus_goes_away(self) -> None:
        self.tick()
        first = FakeBoard.instances[0]
        first.running = False  # the poll task died with the interface
        first.failure = OSError("interface lost")
        self.tick()
        self.assertIn("close", first.calls)
        self.assertEqual(len(FakeBoard.instances), 2)
        self.assertTrue(FakeBoard.instances[1].running)
        # Still detected while it was gone, so the panel never hid the control.
        self.assertTrue(self.link.status()["detected"])

    def test_pause_closes_and_holds_until_resume(self) -> None:
        self.tick()
        self.link.pause()
        self.assertIn("close", FakeBoard.instances[0].calls)
        self.assertTrue(self.link.status()["paused"])
        self.tick()
        self.assertEqual(len(FakeBoard.instances), 1)
        self.link.resume()
        self.tick()
        self.assertEqual(len(FakeBoard.instances), 2)
        self.assertFalse(self.link.status()["paused"])

    def test_follows_a_channel_change(self) -> None:
        self.tick()
        with patch.object(
            link_module, "resolve_safety_channel", return_value="can-other"
        ):
            self.tick()
        self.assertIn("close", FakeBoard.instances[0].calls)
        self.assertEqual(FakeBoard.instances[1].channel, "can-other")


class SafetyApiTest(unittest.IsolatedAsyncioTestCase):
    """The endpoints answer whatever else the server is doing."""

    async def _client(self, safety: Mock, *, running: bool = True) -> httpx.AsyncClient:
        runner = _Runner()
        runner.is_running = lambda: running  # type: ignore[method-assign]
        with patch.object(app_module, "SafetyLink", return_value=safety):
            app = _test_app(_Manager(), runner)
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_stop_works_while_an_operation_runs(self) -> None:
        safety = Mock()
        safety.stop.return_value = {"detected": True}
        async with await self._client(safety) as client:
            response = await client.post("/api/safety/stop")
        self.assertEqual(response.status_code, 200)
        safety.stop.assert_called_once_with()

    async def test_unconfirmed_stop_is_an_error(self) -> None:
        safety = Mock()
        safety.stop.side_effect = SafetyError(
            "The safety board did not confirm the stop."
        )
        async with await self._client(safety) as client:
            response = await client.post("/api/safety/stop")
        self.assertEqual(response.status_code, 503)
        self.assertIn("did not confirm", response.json()["error"])

    async def test_refused_start_says_why(self) -> None:
        safety = Mock()
        safety.start.side_effect = SafetyError("The E-Stop is pressed.")
        async with await self._client(safety) as client:
            response = await client.post("/api/safety/start")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "The E-Stop is pressed.")

    async def test_power_cycle_refused_while_an_operation_runs(self) -> None:
        safety = Mock()
        async with await self._client(safety) as client:
            response = await client.post("/api/safety/power-cycle")
        self.assertEqual(response.status_code, 409)
        self.assertIn("Stop the running operation", response.json()["error"])
        safety.power_cycle.assert_not_called()

    async def test_power_cycle_when_idle(self) -> None:
        safety = Mock()
        safety.power_cycle.return_value = {"detected": True}
        async with await self._client(safety, running=False) as client:
            response = await client.post("/api/safety/power-cycle")
        self.assertEqual(response.status_code, 200)
        safety.power_cycle.assert_called_once_with()

    async def test_refused_power_cycle_says_why(self) -> None:
        safety = Mock()
        safety.power_cycle.side_effect = SafetyError("The E-Stop is pressed.")
        async with await self._client(safety, running=False) as client:
            response = await client.post("/api/safety/power-cycle")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"], "The E-Stop is pressed.")

    async def test_status(self) -> None:
        safety = Mock()
        safety.status.return_value = {"detected": False}
        async with await self._client(safety) as client:
            response = await client.get("/api/safety/status")
        self.assertEqual(response.json(), {"detected": False})


if __name__ == "__main__":
    unittest.main()
