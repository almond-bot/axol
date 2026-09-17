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
(:data:`almond_axol.utils.affinity.MAX_FIFO_PRIORITY`). It is an ``axol
provision`` step: an install-time, host-level grant like the udev rules and
group memberships provisioned alongside it.

Only ``pam_limits`` applies that file, and it is wired into the ``sshd`` and
``login`` PAM stacks alone — so the grant reaches a normal ``ssh`` login and
nothing else. A session logind registers under any other service inherits its
parent's limit instead and stays at zero however many times ``axol provision``
runs: Tailscale SSH (``Service=tailscaled``, whose unit has
``LimitRTPRIO=0``), a systemd unit without ``LimitRTPRIO``, cron, a container.
Since #312 the control loop refuses to start in that state rather than hitch,
so writing the file is no longer the whole job — :func:`verify_session` checks
whether the session running ``provision`` can actually use what was just
granted, because reporting success while ``ulimit -r`` stays 0 is what cost a
day on 2026-09-17.
"""

from __future__ import annotations

import logging
import os
import pwd
import resource
from pathlib import Path

from .affinity import MAX_FIFO_PRIORITY
from .paths import ALMOND_HOME_ENV, almond_home
from .sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

LIMITS_PATH = Path("/etc/security/limits.d/50-axol-rtprio.conf")


def operator_user() -> str | None:
    """Best-effort login of the operator that runs ``axol`` interactively.

    ``SUDO_USER`` when provisioning runs under ``sudo``. A root ``axol serve``
    under systemd has no ``SUDO_USER``, so next the owner of its explicit
    ``ALMOND_HOME``. Otherwise (older units without that environment, or the
    installer run directly as root) the owner of the first ``/home/*`` entry —
    the same heuristic the hosted installer uses to locate the dataset owner.
    ``None`` when none resolves.
    """
    user = os.environ.get("SUDO_USER")
    if user and user != "root":
        return user
    if os.environ.get(ALMOND_HOME_ENV):
        try:
            user = almond_home().owner()
            if user != "root":
                return user
        except (KeyError, OSError):
            pass
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


# Ancestors that only pass rlimits along; naming one would say nothing about
# where a zero came from.
_TRANSPARENT_COMMS = frozenset(
    {
        "axol",
        "bash",
        "dash",
        "fish",
        "ksh",
        "login",
        "python",
        "python3",
        "sh",
        "su",
        "sudo",
        "uv",
        "zsh",
    }
)


def _limit_inherited_from() -> str | None:
    """Name the nearest ancestor this process's rlimits were inherited from.

    ``RLIMIT_RTPRIO`` crosses fork and exec untouched, so when ``pam_limits``
    never ran the effective limit is simply whatever the session's parent
    held. Naming that ancestor is the quickest way to explain a zero —
    ``tailscaled`` for Tailscale SSH, ``cron``, a container shim, a systemd
    unit — and it needs nothing but ``/proc``. Shells and interpreters in
    between are skipped: they only pass the limit down. ``None`` when
    ``/proc`` is unreadable or only transparent ancestors are found.
    """
    pid = os.getppid()
    for _ in range(32):  # bounded; a login session is never this deep
        if pid <= 1:
            return None
        try:
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return None
        if comm and comm not in _TRANSPARENT_COMMS:
            return comm
        parent = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    parent = int(line.split()[1])
                except (IndexError, ValueError):
                    return None
                break
        if parent is None:
            return None
        pid = parent
    return None


def _current_user() -> str | None:
    """This process's effective login name, or None when it cannot be read."""
    try:
        return pwd.getpwuid(os.geteuid()).pw_name
    except (KeyError, OSError):
        return None


def verify_session(user: str) -> bool:
    """Report whether *this* session can actually use the grant for ``user``.

    :func:`install` writes the drop-in, but only ``pam_limits`` applies it,
    and that is wired into the ``sshd`` and ``login`` PAM stacks alone. A
    session logind registers under any other service — Tailscale SSH reports
    ``Service=tailscaled`` — never runs it and silently keeps whatever limit
    its parent held. Provisioning then reports a grant that is real on disk
    and absent from every shell the operator actually uses: on 2026-09-17 a
    manual ``axol provision`` logged success while ``ulimit -r`` stayed 0, the
    control loop ran CFS, and the hitching that caused was read as a
    regression in the two PRs that had just fixed it.

    Silent in the two cases where this process's own limit says nothing:

    * **running as root** — the hosted installer and the systemd unit. Root
      holds ``CAP_SYS_NICE``, whose check short-circuits ``RLIMIT_RTPRIO``
      entirely (``sched(7)``), so a zero there is both normal and harmless.
      Root's default limit *is* zero on a stock Ubuntu, so warning here would
      fire on every install and mean nothing.
    * **provisioning for somebody else** — under ``sudo``, ``user`` is the
      operator :func:`operator_user` resolved rather than whoever is running
      this, so the current limit is unrelated to their future logins.

    Returns True when the grant is usable here, or when the check
    does not apply.
    """
    if os.geteuid() == 0:
        _logger.debug(
            "root holds CAP_SYS_NICE, which bypasses RLIMIT_RTPRIO; not "
            "checking this session's limit"
        )
        return True
    current = _current_user()
    if current is not None and current != user:
        _logger.debug(
            "granted rtprio to %s while running as %s; this session's limit "
            "says nothing about theirs",
            user,
            current,
        )
        return True
    limit = current_limit()
    if limit >= MAX_FIFO_PRIORITY:
        _logger.info(
            "this session's rtprio limit is %d — real-time scheduling available",
            limit,
        )
        return True
    source = _limit_inherited_from()
    _logger.warning(
        "this session's rtprio limit is %d, not %d, so the grant in %s is not "
        "in effect here: pam_limits applies it and only the `sshd` and `login` "
        "PAM stacks run pam_limits%s. Reconnect over `ssh` and confirm with "
        "`ulimit -r`, or raise it in place with `sudo prlimit --pid $$ "
        "--rtprio=%d:%d`. Until then the control loop refuses to start and the "
        "camera relay's capture chain runs CFS, dropping exposures under "
        "recording load",
        limit,
        MAX_FIFO_PRIORITY,
        LIMITS_PATH,
        f" — this shell descends from {source}" if source else "",
        MAX_FIFO_PRIORITY,
        MAX_FIFO_PRIORITY,
    )
    return False


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
            verify_session(user)
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
        "rtprio %d granted to %s via %s — takes effect at %s's next login",
        MAX_FIFO_PRIORITY,
        user,
        LIMITS_PATH,
        user,
    )
    verify_session(user)
