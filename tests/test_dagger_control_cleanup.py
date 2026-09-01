from __future__ import annotations

import threading
import unittest

from almond_axol.cli.collect_dagger import _stop_dagger_control_thread


class _BlockingControlThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.shutdown_event = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:
        self.shutdown_event.wait(1.0)
        self.stopped.set()


class DaggerControlCleanupTest(unittest.TestCase):
    def test_outer_cleanup_signals_and_joins_started_control_thread(self) -> None:
        control_thread = _BlockingControlThread()
        control_thread.start()

        stopped = _stop_dagger_control_thread(control_thread, timeout_s=1.0)

        self.assertTrue(stopped)
        self.assertTrue(control_thread.shutdown_event.is_set())
        self.assertTrue(control_thread.stopped.is_set())
        self.assertFalse(control_thread.is_alive())

    def test_cleanup_accepts_thread_that_was_constructed_but_not_started(
        self,
    ) -> None:
        control_thread = _BlockingControlThread()

        stopped = _stop_dagger_control_thread(control_thread, timeout_s=0.0)

        self.assertTrue(stopped)
        self.assertTrue(control_thread.shutdown_event.is_set())


if __name__ == "__main__":
    unittest.main()
