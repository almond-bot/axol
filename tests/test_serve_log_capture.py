"""serve runner _Capture: every logging record reaches the session exactly once.

Seen in a field log export: each control-process line appeared twice, as
``INFO name: msg`` (the session handler) and ``INFO:name:msg`` (a root
StreamHandler that predated the op, still bound to the fd-2 stream the capture
had redirected into the session pipe). Child-process lines appeared once.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import unittest

from almond_axol.serve import runner


class _Session:
    def __init__(self) -> None:
        self.id = "test-session"
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def emit(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)

    def matching(self, needle: str) -> list[str]:
        with self._lock:
            return [line for line in self.lines if needle in line]


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)


@unittest.skipUnless(sys.platform != "win32", "fd-level redirect")
class CaptureSingleEmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        self._saved_streams = (sys.stdout, sys.stderr)
        root.handlers = []

    def tearDown(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if handler not in self._saved_handlers:
                root.removeHandler(handler)
        root.handlers = self._saved_handlers
        root.setLevel(self._saved_level)
        sys.stdout, sys.stderr = self._saved_streams

    def test_preexisting_stderr_handler_does_not_double_emit(self) -> None:
        # The axol_pi shape: a root StreamHandler on the real stderr, installed
        # before the op (basicConfig at import), default "LEVEL:name:msg" format.
        pre = logging.StreamHandler(sys.stderr)
        pre.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        root = logging.getLogger()
        root.addHandler(pre)
        session = _Session()

        with runner._Capture(session, logging.INFO):  # noqa: SLF001
            logging.getLogger("axol_pi.cli.collect_data").info("loop: 120.0 Hz")
            # The pipe path is asynchronous; give a stray copy time to arrive.
            _wait_for(lambda: len(session.matching("loop: 120.0 Hz")) >= 2, timeout=0.5)
        # After exit the reader thread has drained (EOF once fds are restored).
        lines = session.matching("loop: 120.0 Hz")
        self.assertEqual(lines, ["INFO axol_pi.cli.collect_data: loop: 120.0 Hz"])
        # The pre-existing handler is back on the real stream afterwards.
        self.assertIs(pre.stream, sys.stderr)
        self.assertNotIsInstance(pre.stream, runner._StreamTee)  # noqa: SLF001

    def test_op_basicconfig_force_emits_once_and_is_detached_on_exit(self) -> None:
        # This package's CLIs call basicConfig(force=True) inside the op: the
        # session handler is replaced by a StreamHandler on the tee.
        session = _Session()
        with runner._Capture(session, logging.INFO):  # noqa: SLF001
            logging.basicConfig(level=logging.INFO, force=True)
            logging.getLogger("almond_axol.cli.collect_data").info("Recording started.")
            _wait_for(
                lambda: len(session.matching("Recording started.")) >= 2, timeout=0.5
            )
        self.assertEqual(len(session.matching("Recording started.")), 1)
        (handler,) = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        ]
        # Not left pointing at the finished session's tee.
        self.assertNotIsInstance(handler.stream, runner._StreamTee)  # noqa: SLF001
        self.assertIs(handler.stream, sys.stderr)
        before = len(session.lines)
        logging.getLogger("almond_axol.serve").info("after the op")
        self.assertEqual(len(session.lines), before)

    def test_child_style_fd_output_still_reaches_the_session_once(self) -> None:
        # Native / child-process output bypasses Python streams and is caught by
        # the fd redirect; nothing else may also carry it.
        import os

        session = _Session()
        with runner._Capture(session, logging.INFO):  # noqa: SLF001
            os.write(2, b"WARNING:almond_axol.recording.record_proc:child line\n")
            _wait_for(lambda: len(session.matching("child line")) >= 1)
        self.assertEqual(
            session.matching("child line"),
            ["WARNING:almond_axol.recording.record_proc:child line"],
        )


if __name__ == "__main__":
    unittest.main()
