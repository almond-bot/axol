"""Hands-on steps answered from the control panel.

A command run with ``--web-prompts`` prints a ``[prompt] ...`` line and blocks
on stdin; the panel shows the text with a Continue button that writes a line
back. The panel's Stop sends SIGINT.
"""

from __future__ import annotations

import signal
import sys
import threading


def ask(line: str) -> str:
    """Print ``line`` (the ``[prompt]`` marker) and return the panel's answer.

    Returns ``""`` at end of input. Raises :class:`KeyboardInterrupt` on the
    first SIGINT, even when called from a coroutine under ``asyncio.run``.
    There, the runner's SIGINT handler only cancels the main task, which can't
    happen while this read blocks the event loop, and Python retries the
    interrupted read: a Stop would do nothing until a second one. The default
    handler is installed before the marker is printed, so a Stop that lands
    right after the panel sees the prompt is not lost either.
    """
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGINT)
        if previous is not None:
            signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        print(line, flush=True)
        return sys.stdin.readline()
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
