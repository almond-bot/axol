from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from almond_axol.cli import collect_dagger
from almond_axol.cli.collect_dagger import (
    _DaggerControlLoop,
    _stop_dagger_control_worker,
)
from almond_axol.utils import affinity


class _BlockingControlThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.shutdown_event = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:
        self.shutdown_event.wait(1.0)
        self.stopped.set()


class _RejectedRecorder:
    def poll_capture_error(self) -> str:
        return "camera alignment failed"


class DaggerControlCleanupTest(unittest.TestCase):
    def test_capture_rejection_is_episode_local_not_fatal(self) -> None:
        control_loop = _DaggerControlLoop(
            robot=object(),
            policy=object(),
            teleop=object(),
            recorder=_RejectedRecorder(),
            fps=30,
            teleop_hz=120,
        )

        with patch.object(collect_dagger.affinity, "enter_control_thread") as enter:
            control_loop.run()

        # The pacing thread is half of the control path: realtime core + FIFO,
        # claimed by the thread itself the moment it starts running.
        enter.assert_called_once_with()
        self.assertEqual(control_loop.capture_error, "camera alignment failed")
        self.assertIsNone(control_loop.fatal_error)

    def test_refused_realtime_scheduling_is_fatal_for_the_episode(self) -> None:
        # Claiming the control role sits inside the fault boundary, so a denied
        # real-time class reaches the supervisor instead of quietly collecting
        # an episode of hitched actions.
        control_loop = _DaggerControlLoop(
            robot=object(),
            policy=object(),
            teleop=object(),
            recorder=_RejectedRecorder(),
            fps=30,
            teleop_hz=120,
        )
        refusal = affinity.ControlSchedulingError("no rtprio grant")

        with patch.object(
            collect_dagger.affinity, "enter_control_thread", side_effect=refusal
        ):
            control_loop.run()

        self.assertIs(control_loop.fatal_error, refusal)
        # It failed before the loop body, so no capture verdict was reached.
        self.assertIsNone(control_loop.capture_error)

    def test_outer_cleanup_signals_and_joins_started_control_thread(self) -> None:
        control_thread = _BlockingControlThread()
        control_thread.start()

        stopped, error = _stop_dagger_control_worker(control_thread, timeout=1.0)

        self.assertTrue(stopped)
        self.assertIsNone(error)
        self.assertTrue(control_thread.shutdown_event.is_set())
        self.assertTrue(control_thread.stopped.is_set())
        self.assertFalse(control_thread.is_alive())

    def test_cleanup_accepts_thread_that_was_constructed_but_not_started(
        self,
    ) -> None:
        control_thread = _BlockingControlThread()

        stopped, error = _stop_dagger_control_worker(control_thread, timeout=0.0)

        self.assertTrue(stopped)
        self.assertIsNone(error)
        self.assertTrue(control_thread.shutdown_event.is_set())


if __name__ == "__main__":
    unittest.main()
