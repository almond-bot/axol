import gc
import logging
import threading
import time
import unittest

from almond_axol.utils.stall_diag import (
    GcHold,
    StallWatchdog,
    format_thread_stack,
    install_gc_pause_logger,
    thread_sched_state,
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


def _logger(name: str) -> tuple[logging.Logger, _Capture]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _Capture()
    logger.handlers = [handler]
    return logger, handler


def _wait_for(pred, timeout_s: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


class StallWatchdogTest(unittest.TestCase):
    def test_reports_stall_with_stack_and_state_then_resumption(self) -> None:
        logger, cap = _logger("test.stall.report")
        wd = StallWatchdog("row", threshold_s=0.05, logger=logger)
        release = threading.Event()
        started = threading.Event()

        def worker() -> None:
            wd.beat()
            started.set()
            release.wait(5.0)  # the "stall": blocked in a C-level wait
            wd.beat()

        wd.start()
        t = threading.Thread(target=worker)
        t.start()
        self.assertTrue(started.wait(1.0))
        try:
            self.assertTrue(
                _wait_for(lambda: any("no heartbeat" in m for m in cap.messages()))
            )
            report = next(m for m in cap.messages() if "no heartbeat" in m)
            # The stack names the blocking call in this test's worker.
            self.assertIn("release.wait", report)
            self.assertIn("state=", report)
            self.assertNotIn("monitor overslept", report)
            release.set()
            t.join(1.0)
            self.assertTrue(
                _wait_for(lambda: any("heartbeat resumed" in m for m in cap.messages()))
            )
            resumed = next(m for m in cap.messages() if "heartbeat resumed" in m)
            self.assertRegex(resumed, r"resumed after \d+ ms")
            self.assertEqual(wd.stalls, 1)
            self.assertGreaterEqual(wd.longest_stall_s, 0.05)
        finally:
            release.set()
            wd.stop()

    def test_healthy_heartbeat_never_reports(self) -> None:
        logger, cap = _logger("test.stall.quiet")
        wd = StallWatchdog("tick", threshold_s=0.05, logger=logger)
        stop = threading.Event()

        def worker() -> None:
            while not stop.is_set():
                wd.beat()
                time.sleep(0.005)

        wd.start()
        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.3)
        stop.set()
        t.join(1.0)
        wd.stop()
        self.assertEqual(cap.messages(), [])
        self.assertEqual(wd.stalls, 0)

    def test_suspend_masks_intentional_idle(self) -> None:
        logger, cap = _logger("test.stall.suspend")
        wd = StallWatchdog("tick", threshold_s=0.03, logger=logger)
        wd.beat()
        wd.start()
        wd.suspend()
        time.sleep(0.15)
        self.assertEqual(cap.messages(), [])
        wd.resume()
        # Resume counts as a fresh heartbeat: no immediate report either.
        time.sleep(0.01)
        self.assertEqual(cap.messages(), [])
        time.sleep(0.15)  # now the thread really is silent
        wd.stop()
        self.assertTrue(any("no heartbeat" in m for m in cap.messages()))

    def test_stall_that_ends_while_monitor_frozen_is_reported_as_process_pause(
        self,
    ) -> None:
        # Simulate a stop-the-world pause with a monitor poll long enough that
        # the whole stall fits between two monitor wake-ups.
        logger, cap = _logger("test.stall.frozen")
        wd = StallWatchdog("tick", threshold_s=0.02, logger=logger, poll_s=0.15)
        wd.beat()
        wd.start()
        time.sleep(0.08)  # heartbeat gap of ~80 ms, > threshold, < poll
        stop = threading.Event()

        def healthy_again() -> None:
            while not stop.is_set():
                wd.beat()
                time.sleep(0.002)

        t = threading.Thread(target=healthy_again)
        t.start()
        try:
            self.assertTrue(
                _wait_for(
                    lambda: any("process-wide pause" in m for m in cap.messages()),
                    timeout_s=1.0,
                )
            )
        finally:
            stop.set()
            t.join(1.0)
            wd.stop()
        self.assertEqual(wd.stalls, 1)
        self.assertFalse(any("no heartbeat" in m for m in cap.messages()))

    def test_rejects_non_positive_threshold(self) -> None:
        with self.assertRaises(ValueError):
            StallWatchdog("x", threshold_s=0.0)


class HelpersTest(unittest.TestCase):
    def test_thread_sched_state_reads_current_thread(self) -> None:
        state = thread_sched_state(threading.get_native_id())
        # Linux exposes all three files; elsewhere the dict may be empty.
        if state:
            self.assertIn(state.get("state"), {"R", "S", "D", "T"})
            self.assertIn("run_ns", state)

    def test_thread_sched_state_handles_missing_thread(self) -> None:
        self.assertEqual(thread_sched_state(None), {})
        self.assertEqual(thread_sched_state(2**30), {})

    def test_format_thread_stack_names_current_frame(self) -> None:
        text = format_thread_stack(threading.get_ident())
        self.assertIn("test_format_thread_stack_names_current_frame", text)
        self.assertIn("<thread has no live frame>", format_thread_stack(1))
        self.assertIn("<thread not bound>", format_thread_stack(None))


class GcToolsTest(unittest.TestCase):
    def test_gc_pause_logger_reports_long_collections_only(self) -> None:
        logger, cap = _logger("test.gc.pause")
        uninstall = install_gc_pause_logger(logger, min_ms=0.0)
        try:
            gc.collect()
        finally:
            uninstall()
        self.assertTrue(any("gc: generation-2" in m for m in cap.messages()))
        cap.records.clear()
        uninstall = install_gc_pause_logger(logger, min_ms=1e9)
        try:
            gc.collect()
        finally:
            uninstall()
        self.assertEqual(cap.messages(), [])
        # Uninstalled: a further collection logs nothing.
        gc.collect()
        self.assertEqual(cap.messages(), [])

    def test_gc_hold_disables_and_restores_collection(self) -> None:
        logger, cap = _logger("test.gc.hold")
        self.assertTrue(gc.isenabled())
        hold = GcHold("take", logger=logger)
        hold.begin()
        try:
            self.assertFalse(gc.isenabled())
            self.assertTrue(hold.held)
            hold.begin()  # re-entrant no-op
            self.assertFalse(gc.isenabled())
        finally:
            hold.end()
        self.assertTrue(gc.isenabled())
        self.assertFalse(hold.held)
        hold.end()  # no-op without a matching begin
        self.assertTrue(gc.isenabled())
        self.assertTrue(any("deferred gc.collect swept" in m for m in cap.messages()))

    def test_gc_hold_respects_previously_disabled_collector(self) -> None:
        gc.disable()
        try:
            hold = GcHold("take")
            hold.begin()
            hold.end()
            self.assertFalse(gc.isenabled())
        finally:
            gc.enable()


if __name__ == "__main__":
    unittest.main()
