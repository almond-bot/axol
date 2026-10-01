"""The shared IK-ready wait: no 60 s cliff, fast failure on worker death."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from almond_axol.lerobot import rollout
from almond_axol.teleop import core


class _Conn:
    """A pipe end that becomes readable after ``polls_until_ready`` polls."""

    def __init__(self, polls_until_ready: int | None) -> None:
        self.polls = 0
        self.polls_until_ready = polls_until_ready

    def poll(self, _timeout: float) -> bool:
        self.polls += 1
        return (
            self.polls_until_ready is not None and self.polls >= self.polls_until_ready
        )

    def recv(self):
        return ("ready", [0.0] * 14, list(range(7)), list(range(7, 14)), [])


class _Proc:
    def __init__(self, alive: bool = True, exitcode: int | None = None) -> None:
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


def _controller(conn: _Conn, proc: _Proc) -> rollout.IKResetController:
    controller = rollout.IKResetController.__new__(rollout.IKResetController)
    controller._conn = conn
    controller._proc = proc
    controller._ready = False
    return controller


class WaitReadyTest(unittest.TestCase):
    def setUp(self) -> None:
        p = patch.object(core, "_IK_READY_POLL_S", 0.0)
        p.start()
        self.addCleanup(p.stop)

    def test_a_startup_past_the_old_60s_still_succeeds(self) -> None:
        # An Orin NX needs over a minute; teleop waits on the same worker with
        # no deadline. Simulated clock: each poll advances 1 s.
        clock = iter(range(0, 10_000))
        conn = _Conn(polls_until_ready=90)
        with patch.object(core.time, "monotonic", lambda: next(clock)):
            self.assertTrue(_controller(conn, _Proc()).wait_ready())

    def test_a_dead_worker_fails_immediately_with_its_exit_code(self) -> None:
        conn = _Conn(polls_until_ready=None)
        with self.assertRaisesRegex(RuntimeError, "exit code -9"):
            _controller(conn, _Proc(alive=False, exitcode=-9)).wait_ready()
        self.assertEqual(conn.polls, 1)

    def test_a_hung_worker_still_times_out(self) -> None:
        clock = iter(range(0, 10_000, 100))
        with (
            patch.object(core.time, "monotonic", lambda: next(clock)),
            self.assertRaisesRegex(TimeoutError, "within 300s"),
        ):
            _controller(_Conn(polls_until_ready=None), _Proc()).wait_ready()

    def test_stop_aborts_the_wait(self) -> None:
        stops = iter([False, False, True])
        controller = _controller(_Conn(polls_until_ready=None), _Proc())
        self.assertFalse(controller.wait_ready(stopped=lambda: next(stops)))
        self.assertFalse(controller._ready)

    def test_return_to_rest_aborts_when_stopped_during_startup(self) -> None:
        controller = _controller(_Conn(polls_until_ready=None), _Proc())
        with patch.object(controller, "_play_to_rest") as play:
            self.assertFalse(controller.return_to_rest(object(), stopped=lambda: True))
        play.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TeleopPathsShareTheWaitTest(unittest.TestCase):
    def test_teleop_and_collect_data_use_the_shared_wait(self) -> None:
        # They used to block on a bare recv() with no limit at all.
        from pathlib import Path

        root = Path(rollout.__file__).resolve().parents[1]
        for rel in ("teleop/teleop.py", "lerobot/teleop/teleop_vr.py"):
            source = (root / rel).read_text()
            with self.subTest(rel):
                self.assertIn("wait_for_ik_ready, parent_conn, process", source)
                self.assertNotIn("run_in_executor(None, parent_conn.recv)", source)

    def test_unexpected_handshake_is_an_error(self) -> None:
        conn = _Conn(polls_until_ready=1)
        conn.recv = lambda: ("error", "boom")
        with self.assertRaisesRegex(RuntimeError, "Unexpected IK worker handshake"):
            core.wait_for_ik_ready(conn, _Proc())
