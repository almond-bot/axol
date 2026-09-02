"""
axol jetson.setup

Apply the per-boot Jetson tuning the real-time loops need: select the MAXN
power mode, pin the NVENC/VIC/GPU engine clocks (so the camera relay's
hardware H.264 encode and ZED SDK processing stay low-latency), set the CPU
``performance`` governor (so the bursty IK loop isn't underclocked), steer the
CAN adapters' USB-controller interrupt onto a CAN core (so real-time camera
work can never delay motor feedback), and schedule the Argus camera daemon
``SCHED_FIFO`` on the camera cores (so all-camera frame drops under load stop).
See :mod:`almond_axol.utils.jetson` for why each Tegra default hurts.

All of it resets on every reboot, so this runs at boot from the systemd unit
the host installer registers (``ExecStartPre`` on ``axol.service``), and once
during install. It is intentionally *not* called from ``teleop`` /
``collect-data`` / ``serve`` — those just run. Best-effort and a no-op on
non-Jetson machines.

The install-time counterpart is ``axol provision``, which grants the
operator's login the rtprio allowance a manual ``axol serve`` needs to run the
camera relay's capture chain real-time in the first place.
"""

from __future__ import annotations

import logging

from ...utils.jetson import pin_realtime_clocks


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``jetson.setup`` subcommand."""
    subparsers.add_parser(
        "jetson.setup",
        help=(
            "Per-boot Jetson tuning: MAXN, engine + CPU clocks, CAN interrupt "
            "placement, and real-time camera daemon scheduling."
        ),
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Apply the Jetson real-time tuning (interactive sudo when on a tty)."""
    # Surface the pin functions' INFO/WARNING logs (which command did what);
    # force=True in case an imported dependency already installed a handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    pin_realtime_clocks(interactive=True)
