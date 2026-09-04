"""Persistent ``SCHED_FIFO`` allowance for the operator's login.

The camera relay elevates its capture chain to ``SCHED_FIFO``
(:func:`almond_axol.utils.affinity.prioritize_capture_threads`) so a 60 Hz
exposure is never lost to a descheduled source thread. A non-root process may
only do that when its ``RLIMIT_RTPRIO`` hard limit is at least the priority
requested. The systemd unit the hosted installer writes grants that
(``LimitRTPRIO``), but a manual ``axol serve`` from a login shell inherits the
PAM default of **zero**, the relay silently falls back to CFS, and the first
recording under load discards its episode on skipped exposures — the failure
seen on a customer robot on 2026-09-02, where the same checkout worked from a
shell that happened to have the allowance.

:func:`install` writes a ``/etc/security/limits.d`` drop-in granting the
operator's account the top rung of the stack's FIFO ladder
(:data:`almond_axol.utils.affinity.MAX_FIFO_PRIORITY`). PAM applies it at the
next login, so it survives reboots and needs no per-shell ``prlimit``. It is
an ``axol provision`` step: an install-time, host-level grant like the udev
rules and group memberships provisioned alongside it.
"""

from __future__ import annotations

import logging
import os
import resource
from pathlib import Path

from .affinity import MAX_FIFO_PRIORITY
from .sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

LIMITS_PATH = Path("/etc/security/limits.d/50-axol-rtprio.conf")


def operator_user() -> str | None:
    """Best-effort login of the operator that runs ``axol`` interactively.

    ``SUDO_USER`` when provisioning runs under ``sudo``; otherwise (a root
    ``axol serve`` under systemd, or the installer run directly as root) the
    owner of the first ``/home/*`` entry — the same heuristic the hosted
    installer uses to locate the dataset owner. ``None`` when neither
    resolves.
    """
    user = os.environ.get("SUDO_USER")
    if user and user != "root":
        return user
    try:
        homes = sorted(Path("/home").iterdir())
    except OSError:
        return None
    for home in homes:
        try:
            owner = home.owner()
        except (KeyError, OSError):
            continue
        if owner != "root":
            return owner
    return None


def limits_text(user: str) -> str:
    """The ``limits.d`` drop-in granting ``user`` the stack's FIFO ceiling."""
    return (
        "# Written by `axol provision` (almond_axol.utils.rtprio).\n"
        "# Lets a manual `axol serve` from this account run the camera relay's\n"
        f"# capture chain SCHED_FIFO (relay uses {MAX_FIFO_PRIORITY} at most);\n"
        "# without it the relay falls back to CFS and drops exposures under\n"
        "# recording load. The systemd unit grants the same via LimitRTPRIO.\n"
        f"{user}\t-\trtprio\t{MAX_FIFO_PRIORITY}\n"
    )


def current_limit() -> int:
    """This process's ``RLIMIT_RTPRIO`` hard limit (what a child may request)."""
    _soft, hard = resource.getrlimit(resource.RLIMIT_RTPRIO)
    # 99 is the highest SCHED_FIFO priority Linux offers, so an unlimited hard
    # limit is equivalent to it.
    return 99 if hard == resource.RLIM_INFINITY else int(hard)


def install() -> None:
    """Grant the operator a persistent rtprio allowance via ``limits.d``.

    Idempotent and best-effort: a no-op when the drop-in already matches, and
    only a warning (with the manual command) when root cannot be obtained.
    The grant applies to the operator's *next* login; the current shell keeps
    the limit it started with, which is why the message says so.
    """
    user = operator_user()
    if user is None:
        _logger.info("no operator account found; skipping the rtprio grant")
        return
    wanted = limits_text(user)
    try:
        if LIMITS_PATH.read_text() == wanted:
            _logger.info("rtprio grant already in place for %s (%s)", user, LIMITS_PATH)
            return
    except OSError:
        pass
    if not prime_sudo():
        _logger.warning(
            "rtprio grant needs root; a manual `axol serve` from %s's shell will "
            "run the camera relay without SCHED_FIFO and drop exposures under "
            "recording load. Run manually: printf '%%s\\t-\\trtprio\\t%d\\n' %s | "
            "sudo tee %s",
            user,
            MAX_FIFO_PRIORITY,
            user,
            LIMITS_PATH,
        )
        return
    run_root(["mkdir", "-p", str(LIMITS_PATH.parent)], check=True)
    run_root(["tee", str(LIMITS_PATH)], input_text=wanted, check=True)
    _logger.info(
        "rtprio %d granted to %s via %s — takes effect at %s's next login "
        "(current shells keep ulimit -r %d)",
        MAX_FIFO_PRIORITY,
        user,
        LIMITS_PATH,
        user,
        current_limit(),
    )
