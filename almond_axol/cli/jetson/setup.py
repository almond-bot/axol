"""
axol jetson.setup

Pin the Jetson clocks the real-time loops need: the NVENC/VIC engine devfreq
nodes (so the camera relay's hardware H.264 encode stays low-latency) and the
CPU governor (so the bursty IK loop isn't underclocked). See
:mod:`almond_axol.utils.jetson` for why the Tegra defaults hurt.

These reset on every reboot, so this runs at boot from the systemd unit the
host installer registers (``ExecStartPre`` on ``axol.service``), and once
during install. It is intentionally *not* called from ``teleop`` /
``collect-data`` / ``serve`` — those just run. Best-effort and a no-op on
non-Jetson machines.
"""

from __future__ import annotations

import logging

from ...utils.jetson import pin_realtime_clocks


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``jetson.setup`` subcommand."""
    subparsers.add_parser(
        "jetson.setup",
        help="Pin the Jetson NVENC/VIC and CPU clocks for the real-time loops.",
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Pin the Jetson engine + CPU clocks (interactive sudo when on a tty)."""
    # Surface the pin functions' INFO/WARNING logs (which command did what);
    # force=True in case an imported dependency already installed a handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    pin_realtime_clocks(interactive=True)
