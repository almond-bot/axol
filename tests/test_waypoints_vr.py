"""``axol waypoints --teach vr``: teaching through a VR teleop session.

The session is driven with a fake teleop and robot: what matters here is the
hand-off (teleop runs while teaching, is stood down for playback and handed
back after), the headset buttons, and what a recording stores. The pieces it
adds to the teleop stack — the ``r_a`` frame field, the server's prompt
banner, and the core's suspend/resume — are checked against the real classes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.cli import waypoints
from almond_axol.cli.waypoints import (
    GRIP_CLOSED,
    GRIP_OPEN,
    WaypointsCmdConfig,
    _HeadsetButtons,
    _Session,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.vr.config import VRServerConfig
from almond_axol.vr.models import VRFrame
from almond_axol.vr.server import VRServer
from almond_axol.waypoints import WaypointSet


def _frame(**buttons: bool) -> SimpleNamespace:
    return SimpleNamespace(
        r_a=buttons.get("r_a", False),
        l_stick_click=buttons.get("l_stick_click", False),
        r_stick_click=buttons.get("r_stick_click", False),
    )


class _Control:
    """Session control that records what was published and feeds commands."""

    def __init__(self) -> None:
        self.states: list[tuple[str, str, list[dict[str, str]], int]] = []
        self.commands: list[str] = []

    def poll(self) -> str | None:
        return self.commands.pop(0) if self.commands else None

    def set_state(
        self, phase: str, message: str, controls: list[dict[str, str]], count: int
    ) -> None:
        self.states.append((phase, message, controls, count))

    def close(self) -> None:
        pass


class _Robot:
    """Enough of RobotBase for a sim session that never plays."""

    def __init__(self) -> None:
        self.left_pose = np.arange(8, dtype=np.float32) / 10.0
        self.right_pose = -np.arange(8, dtype=np.float32) / 10.0

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.left_pose.copy(), self.right_pose.copy()


class _Teleop:
    """Stands in for VRTeleop: records the lifecycle calls the session makes."""

    def __init__(self) -> None:
        self.listeners: list = []
        self.banners: list[str | None] = []
        self.grip_targets = (1.0, 1.0)
        self.is_resetting = False
        self.events: list[str] = []
        self.running = threading.Event()

    def add_frame_listener(self, callback) -> None:
        self.listeners.append(callback)

    def press(self, **buttons: bool) -> None:
        """One press-and-release of the given buttons, as the headset sends it."""
        for listener in self.listeners:
            listener(_frame(**buttons))
            listener(_frame())

    def set_banner(self, text: str | None) -> None:
        self.banners.append(text)

    async def run(self) -> None:
        self.events.append("run")
        self.running.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.running.clear()
            self.events.append("run-cancelled")

    def suspend(self) -> None:
        self.events.append("suspend")

    async def resume(self) -> None:
        self.events.append("resume")


class _NoSolver:
    """_SolverHandle stand-in: the JAX solver is never needed here."""

    def __init__(self, *_: object) -> None:
        pass

    is_ready = False

    def get(self) -> None:
        return None


class _SessionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.file = Path(tmp.name) / "waypoints.json"
        patcher = patch.object(waypoints, "_SolverHandle", _NoSolver)
        patcher.start()
        self.addCleanup(patcher.stop)

    def session(
        self, *, labels: list[str] | None = None
    ) -> tuple[_Session, _Teleop, _Control]:
        cfg = WaypointsCmdConfig(
            file=str(self.file), sim=True, teach="vr", labels=labels
        )
        teleop = _Teleop()
        control = _Control()
        session = _Session(
            cfg,
            _Robot(),  # type: ignore[arg-type]
            control,  # type: ignore[arg-type]
            threading.Event(),
            teleop=teleop,  # type: ignore[arg-type]
        )
        return session, teleop, control


class HeadsetButtonsTest(unittest.TestCase):
    def test_each_press_is_one_command(self) -> None:
        buttons = _HeadsetButtons()
        for frame in (
            _frame(r_a=True),
            _frame(r_a=True),  # still held: no repeat
            _frame(),
            _frame(r_a=True),
            _frame(l_stick_click=True),
            _frame(r_stick_click=True, l_stick_click=True),
        ):
            buttons.feed(frame)
        commands = []
        while (command := buttons.poll()) is not None:
            commands.append(command)
        self.assertEqual(commands, ["record", "record", "undo", "play"])

    def test_clear_drops_stale_presses(self) -> None:
        buttons = _HeadsetButtons()
        buttons.feed(_frame(r_a=True))
        buttons.clear()
        self.assertIsNone(buttons.poll())


class PlayOnlyTest(unittest.TestCase):
    def test_sim_teaches_only_in_vr(self) -> None:
        self.assertTrue(waypoints._play_only(WaypointsCmdConfig(sim=True)))
        self.assertFalse(waypoints._play_only(WaypointsCmdConfig(sim=True, teach="vr")))
        self.assertTrue(
            waypoints._play_only(
                WaypointsCmdConfig(sim=True, teach="vr", play_only=True)
            )
        )

    def test_sim_vr_starts_from_an_empty_file(self) -> None:
        # Hand-mode sim refuses a file with nothing to play; VR sim teaches it.
        with TemporaryDirectory() as tmp:
            cfg = WaypointsCmdConfig(file=str(Path(tmp) / "none.json"), sim=True)
            with self.assertRaisesRegex(ValueError, "fewer than two"):
                waypoints._run(cfg, control=_Control())  # type: ignore[arg-type]
            cfg.teach = "vr"
            with patch.object(waypoints, "_session") as session:
                session.return_value = asyncio.sleep(0)
                waypoints._run(cfg, control=_Control())  # type: ignore[arg-type]
            session.assert_called_once()


class VrTeachingTest(_SessionTestCase):
    def test_record_stores_the_trigger_snapped_open_or_closed(self) -> None:
        session, teleop, _ = self.session()
        teleop.grip_targets = (0.2, 0.8)
        asyncio.run(session._record())
        waypoint = WaypointSet.load(self.file)[0]
        self.assertEqual(waypoint.left[7], GRIP_CLOSED)
        self.assertEqual(waypoint.right[7], GRIP_OPEN)
        np.testing.assert_allclose(waypoint.left[:7], np.arange(7) / 10.0)

    def test_record_waits_for_the_return_to_rest(self) -> None:
        session, teleop, control = self.session()
        teleop.is_resetting = True
        asyncio.run(session._record())
        self.assertEqual(len(session.store), 0)
        self.assertIn("returning to rest", control.states[-1][1])

    def test_labels_name_waypoints_and_prompt_the_next(self) -> None:
        session, _, control = self.session(labels=["home", "pick"])
        session._publish_teaching()
        phase, message, controls, _ = control.states[-1]
        self.assertEqual(phase, "teaching")
        self.assertIn("home (1 of 2)", message)
        self.assertEqual(controls[0]["label"], "Record home")
        asyncio.run(session._record())
        self.assertIn("pick (2 of 2)", control.states[-1][1])
        asyncio.run(session._record())
        asyncio.run(session._record())
        labels = [wp.label for wp in WaypointSet.load(self.file)]
        # Past the end of the list the default name applies.
        self.assertEqual(labels, ["home", "pick", "waypoint 3"])

    def test_headset_banner_carries_the_phase_hint(self) -> None:
        session, teleop, _ = self.session()
        session._publish_teaching()
        self.assertIn("A record", teleop.banners[-1])
        session._publish("playing", "Moving to waypoint 1", stoppable=True)
        self.assertEqual(teleop.banners[-1], "Moving to waypoint 1\nA stop")
        session._publish("planning", "Planning…")
        self.assertEqual(teleop.banners[-1], "Planning…")

    def test_no_gripper_controls_in_vr(self) -> None:
        session, _, control = self.session()
        session._publish_teaching()
        commands = {c["command"] for c in control.states[-1][2]}
        self.assertFalse(commands & {"grip-left", "grip-right"})

    def test_a_stops_playback_and_stick_clicks_do_nothing(self) -> None:
        session, teleop, _ = self.session()
        teleop.press(r_stick_click=True)
        self.assertFalse(session._interrupted())
        teleop.press(r_a=True)
        self.assertTrue(session._interrupted())

    def test_teaching_runs_teleop_until_play_then_stands_it_down(self) -> None:
        session, teleop, _ = self.session()

        async def scenario() -> None:
            task = asyncio.create_task(session.teach())
            await asyncio.to_thread(teleop.running.wait, 5)
            teleop.press(r_a=True)
            teleop.press(r_a=True)
            teleop.press(l_stick_click=True)  # undo the second
            teleop.press(r_a=True)
            teleop.press(r_stick_click=True)  # play
            await asyncio.wait_for(task, 5)

        asyncio.run(scenario())
        self.assertEqual(len(WaypointSet.load(self.file)), 2)
        self.assertEqual(teleop.events, ["run", "run-cancelled", "suspend"])
        self.assertTrue(session._under_position_control)

        # The next teaching pass hands the arms back before teleop runs again.
        async def again() -> None:
            task = asyncio.create_task(session.teach())
            await asyncio.to_thread(teleop.running.wait, 5)
            session._control.commands.append("quit")  # type: ignore[union-attr]
            await asyncio.wait_for(task, 5)

        asyncio.run(again())
        self.assertEqual(
            teleop.events[3:], ["resume", "run", "run-cancelled", "suspend"]
        )
        self.assertTrue(session._quit)

    def test_play_waits_while_teleop_returns_to_rest(self) -> None:
        session, teleop, control = self.session()
        asyncio.run(session._record())
        asyncio.run(session._record())
        teleop.is_resetting = True
        self.assertFalse(asyncio.run(session._handle_teaching_command("play")))
        self.assertIn("reach rest", control.states[-1][1])
        teleop.is_resetting = False
        self.assertTrue(asyncio.run(session._handle_teaching_command("play")))

    def test_a_teleop_fault_ends_teaching(self) -> None:
        session, teleop, _ = self.session()

        async def broken() -> None:
            raise RuntimeError("motor fault")

        teleop.run = broken  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "motor fault"):
            asyncio.run(session.teach())
        self.assertTrue(session._teleop_suspended)


class TeleopPiecesTest(unittest.TestCase):
    def test_frame_carries_a_defaulting_to_released(self) -> None:
        pose = {
            "position": {"x": 0, "y": 0, "z": 0},
            "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
        }
        base = {
            "l_ee": pose,
            "r_ee": pose,
            "l_elbow": {"x": 0, "y": 0, "z": 0},
            "r_elbow": {"x": 0, "y": 0, "z": 0},
        }
        self.assertFalse(VRFrame.model_validate(base).r_a)
        self.assertTrue(VRFrame.model_validate({**base, "r_a": True}).r_a)

    def test_banner_is_replayed_to_late_joiners(self) -> None:
        server = VRServer(VRServerConfig())
        sent: list[dict] = []

        class _Ws:
            async def send_text(self, text: str) -> None:
                sent.append(json.loads(text))

        asyncio.run(server._send_session_config(_Ws()))  # type: ignore[arg-type]
        self.assertNotIn("banner", [m["type"] for m in sent])
        server.set_banner("Teaching")
        sent.clear()
        asyncio.run(server._send_session_config(_Ws()))  # type: ignore[arg-type]
        self.assertIn({"type": "banner", "value": "Teaching"}, sent)

    def test_banner_broadcasts_on_the_server_loop(self) -> None:
        server = VRServer(VRServerConfig())
        sent: list[str] = []

        async def scenario() -> None:
            async def broadcast(text: str) -> None:
                sent.append(text)

            server.broadcast_text = broadcast  # type: ignore[method-assign]
            server._loop = asyncio.get_running_loop()
            await asyncio.to_thread(server.set_banner, "Recorded home")
            await asyncio.to_thread(server.set_banner, "Recorded home")  # no-op
            await asyncio.to_thread(server.set_banner, None)
            await asyncio.sleep(0.05)

        asyncio.run(scenario())
        self.assertEqual(
            [json.loads(text)["value"] for text in sent], ["Recorded home", None]
        )

    def test_core_suspend_then_resume_realigns_through_a_reset(self) -> None:
        core = VRTeleopCore(
            VRTeleopConfig(), logging.getLogger(__name__), lambda _enabled: None
        )
        core.left_enabled = core.right_enabled = True
        core.suspend_tracking()
        self.assertFalse(core.teleop_enabled)
        self.assertTrue(core._ik_paused)
        self.assertFalse(core.is_resetting)
        pose = np.zeros(8, dtype=np.float32)
        pose[7] = GRIP_CLOSED
        core.resume_tracking(pose, pose)
        self.assertFalse(core._ik_paused)
        # The queued reset re-seats the IK worker at the arms' real pose.
        self.assertTrue(core.reset_pending)
        self.assertEqual((core.l_grip, core.r_grip), (GRIP_CLOSED, GRIP_CLOSED))


if __name__ == "__main__":
    unittest.main()
