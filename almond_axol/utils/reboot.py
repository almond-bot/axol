"""Reboot-required bookkeeping for provisioning.

A few provisioning steps change something that only takes effect at boot (the
ZED Box's GMSL kernel driver, a Jetson power mode nvpmodel will only switch
across a reboot). Each records *why* with :func:`request`; whoever owns the
host's lifecycle reboots at its own safe point:

* ``axol provision`` run by an operator reboots at the end of a clean run.
* The hosted installer runs ``axol provision --no-reboot`` and reboots with
  ``axol provision --apply-reboot`` as its last step, once the service is
  written.
* ``axol serve``'s self-updater reboots instead of restarting the service,
  and only while idle -- a provision it spawns never reboots on its own.

The marker lives on ``/run`` (tmpfs), so the reboot itself clears it, and it
survives a failed provision: re-running after a fix still reboots even though
the step that asked (e.g. the driver upgrade) is now a no-op.

A reason still pending after the reboot it caused is not rebooted for again
(:func:`reboot_host` returns False): a step that keeps asking would otherwise
reboot-loop the robot through the service's startup provision.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .sudo import run_root

_logger = logging.getLogger(__name__)

MARKER = Path("/run/almond-axol/reboot-required")
# Persistent (not tmpfs): the reasons the last automatic reboot was for.
_LAST_ATTEMPT = Path("/var/lib/almond-axol/last-auto-reboot")


def request(reason: str) -> None:
    """Record that ``reason`` needs a reboot to take effect (needs root)."""
    run_root(["mkdir", "-p", str(MARKER.parent)], check=True)
    run_root(["tee", "-a", str(MARKER)], input_text=reason + "\n", check=True)


def pending() -> list[str]:
    """Distinct reasons recorded since the last boot, oldest first."""
    try:
        lines = MARKER.read_text().splitlines()
    except OSError:
        return []
    return list(dict.fromkeys(line.strip() for line in lines if line.strip()))


def _summary(reasons: list[str]) -> str:
    return "; ".join(reasons)


def clear_attempt() -> None:
    """Forget the last automatic reboot once nothing is pending any more."""
    if _LAST_ATTEMPT.exists():
        run_root(["rm", "-f", str(_LAST_ATTEMPT)])


def reboot_host(reasons: list[str]) -> bool:
    """Reboot now for ``reasons``; False when that would repeat the last reboot.

    Raises when the reboot command itself fails.
    """
    summary = _summary(reasons)
    try:
        last = _LAST_ATTEMPT.read_text().strip()
    except OSError:
        last = ""
    if last == summary:
        _logger.warning(
            "a reboot is still required (%s), but the last automatic reboot was "
            "for exactly this and did not clear it; not rebooting again. "
            "Investigate, then reboot by hand (sudo reboot).",
            summary,
        )
        return False
    run_root(["mkdir", "-p", str(_LAST_ATTEMPT.parent)], check=True)
    run_root(["tee", str(_LAST_ATTEMPT)], input_text=summary + "\n", check=True)
    _logger.warning("REBOOTING NOW: %s", summary)
    run_root(["systemctl", "reboot"], check=True)
    return True
