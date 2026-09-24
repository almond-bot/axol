"""
axol jetson.setup

Alias of ``axol provision --boot``: the per-boot host tuning — the MAXN power
mode, NVENC/VIC/GPU engine clock pins, the CPU ``performance`` governor, the
CAN adapters' USB-controller interrupt steered onto a CAN core, and the Argus
camera daemon ``SCHED_FIFO`` on the camera cores. See
:mod:`almond_axol.utils.jetson` for why each Tegra default hurts.

``axol provision`` applies this as its last step, so operators never run it
separately; the systemd unit re-applies it at each boot via ``ExecStartPre``.
Units written by installers before ``provision --boot`` existed still call
``axol jetson.setup``, which is why it stays. Best-effort and a no-op on
non-Jetson machines.
"""

from __future__ import annotations

import logging


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``jetson.setup`` subcommand."""
    subparsers.add_parser(
        "jetson.setup",
        help=(
            "Per-boot host tuning (alias of `axol provision --boot`; "
            "`axol provision` already applies it)."
        ),
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    """Apply the per-boot host tuning (interactive sudo when on a tty)."""
    from ..provision import tune_host

    # Surface the pin functions' INFO/WARNING logs (which command did what);
    # force=True in case an imported dependency already installed a handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    tune_host(interactive=True)
