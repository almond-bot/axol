"""Durable binding for the fixed TCP ports Axol servers use.

Several servers bind a *fixed* port and can be started/stopped repeatedly within
one long-running ``axol serve`` process — the VR WebSocket server (8000), the
viser sim viewer (8002), and the control panel itself (8001). A previous run (or
a crashed/force-killed one) can leave a listener squatting on the port, so a
plain bind fails with ``[Errno 98] address already in use`` and the server comes
up dead. This module centralizes "bind, reclaiming the port if necessary" so
every fixed port behaves identically.

The reclaim never targets our own PID, so it can't take down the control-panel
process that hosts several of these servers at once.
"""

from __future__ import annotations

import errno
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import time

_logger = logging.getLogger(__name__)

# Well-known fixed port. The VR WebSocket server binds it and the Quest-over-USB
# ``adb reverse`` tunnel forwards the headset's loopback copy to it, so the two
# must agree — this is the single source of truth for both (``vr.config`` and
# ``utils.adb`` import it) so they can't drift and silently break USB teleop.
VR_PORT = 8000

# How hard ``open_listen_socket`` tries before giving up. A couple of plain
# retries absorb a previous server still releasing the socket; after that we
# evict whatever is squatting on the port.
_BIND_RETRIES = 12
_BIND_RETRY_DELAY = 0.25  # seconds between attempts


def listening_pids(port: int) -> set[int]:
    """PIDs (other than our own) with a LISTEN socket on ``port``.

    Uses ``ss`` (iproute2, always present on the Jetson — CAN bring-up already
    relies on it). Returns an empty set if ``ss`` is missing or unparseable, in
    which case the caller simply can't reclaim and surfaces the bind error.
    """
    ss = shutil.which("ss")
    if ss is None:
        return set()
    try:
        proc = subprocess.run(
            [ss, "-ltnpH", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    me = os.getpid()
    return {
        pid
        for pid in (int(m.group(1)) for m in re.finditer(r"pid=(\d+)", proc.stdout))
        if pid != me
    }


def _signal_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError) as exc:
        _logger.debug("could not signal pid %d: %s", pid, exc)


def reclaim_port(port: int) -> None:
    """Best-effort: terminate any *other* process listening on ``port``.

    Used for ports whose value is fixed (a headset tunnel, a bookmarked viewer
    URL, a client's ``--server_port``), where a leftover listener from a crashed
    or not-fully-stopped previous run has to be evicted rather than worked
    around. Never targets our own PID, so it can't suicide a process that hosts
    several of these servers at once.
    """
    pids = listening_pids(port)
    if not pids:
        return
    _logger.warning(
        "port %d held by pid(s) %s; terminating to reclaim it",
        port,
        ", ".join(map(str, sorted(pids))),
    )
    for pid in pids:
        _signal_pid(pid, signal.SIGTERM)
    time.sleep(_BIND_RETRY_DELAY)
    for pid in listening_pids(port):
        _signal_pid(pid, signal.SIGKILL)


def open_listen_socket(host: str, port: int) -> socket.socket:
    """Bind a listening socket on ``(host, port)``, reclaiming it if necessary.

    Binding eagerly here — rather than letting an async server bind inside a
    background task — means a real conflict raises to the caller instead of
    being logged and swallowed. The returned socket is bound but *not* listening
    yet; uvicorn/asyncio call ``listen()`` when they adopt it.

    The retry/reclaim sequence: bind immediately (the normal case), then retry
    briefly to let a previous server finish releasing the socket, then evict any
    process still holding it, then keep retrying until it's ours.
    """
    last_exc: OSError | None = None
    for attempt in range(_BIND_RETRIES):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR lets us bind over a socket left in TIME_WAIT by a server
        # that just exited (e.g. one we reclaimed a moment ago).
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            last_exc = exc
            # Attempt 0: just wait for a mid-shutdown server to let go.
            # From attempt 1 on, actively evict the squatter.
            if attempt >= 1:
                reclaim_port(port)
            time.sleep(_BIND_RETRY_DELAY)
            continue
        return sock
    raise OSError(
        errno.EADDRINUSE,
        f"port {port} is still in use after {_BIND_RETRIES} attempts; "
        "another process is holding it and could not be reclaimed",
    ) from last_exc
