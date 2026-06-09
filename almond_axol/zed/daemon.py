"""Control of the ZED X daemon (``zed_x_daemon``) on the ZED box.

The daemon only enumerates GMSL cameras when it starts, so a camera plugged in
after boot stays invisible to the SDK until the daemon restarts. Restart it
before opening cameras.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

_logger = logging.getLogger(__name__)

# Give the daemon time to enumerate the sensors on the GMSL links before a
# client tries to open them.
_DAEMON_RESTART_WAIT_S = 5.0


def restart_zed_daemon() -> None:
    """Restart ``zed_x_daemon`` and block until it has had time to settle.

    We can't know when the cameras were last (re)plugged, so call this before
    any attempt to open ZED cameras. Blocks for ~5 seconds after the restart.

    Raises:
        RuntimeError: If the ``systemctl restart`` command fails (e.g. no
            passwordless sudo when running headless).
    """
    cmd = ["systemctl", "restart", "zed_x_daemon"]
    if os.geteuid() != 0:
        # `-n` so a headless session (web control panel) fails fast instead of
        # blocking on a password prompt that can never be answered.
        cmd = ["sudo", "-n", *cmd]
    _logger.info("Restarting zed_x_daemon...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Failed to restart zed_x_daemon: {detail}")
    _logger.info("Waiting %.0fs for zed_x_daemon to settle...", _DAEMON_RESTART_WAIT_S)
    time.sleep(_DAEMON_RESTART_WAIT_S)
