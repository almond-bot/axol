"""Passwordless escalation for the e-stop CAN purge.

Cutting motor power (the operators' e-stop) leaves the kernel holding up to
``txqueuelen`` position commands per arm interface: nothing on the wire ACKs,
so the TX queue fills instead of draining. Those frames are not discarded when
the process dies — they belong to the interface — and they replay the instant
the motors are powered back up and enabled, snapping the arm to wherever it
was *commanded* around the e-stop rather than where it came to rest. The
realtime core therefore flaps the interface as soon as it declares the bus
stalled (``purge_tx_queue`` in ``rust/axol-rt/src/safety.rs``).

That flap needs root. The hosted install runs ``axol serve`` as root under
systemd and escalates for free, but a manual ``axol serve`` / ``axol teleop``
from the operator's shell runs as the operator, and the core's ``sudo -n``
then fails unless a credential happens to be cached — which it is not, an
e-stop being minutes to hours into a session while sudo's timestamp lasts 15.
The purge silently did nothing on every such run (seen on
``almond-jelly-zed-box`` on 2026-09-16: a manual serve, the systemd unit
stopped, and an arm that jerked on the next start).

:func:`install` writes the ``sudoers.d`` drop-in that closes that hole: the
operator may run the bring-up script, and only it, as root without a password.
It is an ``axol provision`` step, alongside the other host-level grants
(:mod:`almond_axol.utils.rtprio`, the udev rules, the group memberships).

The grant is narrow on purpose. Its targets are root-owned and only
root-writable — ``/etc/almond-axol/can/startup.sh`` is exactly why
``can.setup`` stopped generating that script below the operator-writable
``~/.almond`` tree (see :func:`almond_axol.cli.provision.
_neutralize_legacy_can_root_execution`) — so the rule cannot be turned into a
root-code-execution primitive by editing what it points at.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..constants import (
    CAN_BASE,
    CAN_BRINGUP_SCRIPT,
    CAN_CHEST,
    CAN_LEFT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    CAN_RESET_SCRIPT,
    CAN_RIGHT,
)
from .rtprio import operator_user
from .sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

# No dot in the name: sudo ignores every file in sudoers.d containing one.
SUDOERS_PATH = Path("/etc/sudoers.d/50-axol-can-purge")

# The interfaces the core may flap one at a time when no bring-up script is
# installed yet (``safety.rs`` falls back to plain ``ip link`` then). The
# script covers the pair properly; this is only for the un-provisioned host.
_PURGEABLE_INTERFACES = (
    CAN_LEFT,
    CAN_RIGHT,
    CAN_BASE,
    CAN_CHEST,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
)


def _program_paths(program: str, candidates: tuple[str, ...]) -> list[str]:
    """Every absolute path sudo might match *program* at on this host.

    sudoers matches the path sudo resolved through ``secure_path``, compared
    literally — it does not follow symlinks — so a rule naming only
    ``/usr/bin/bash`` never matches a host that resolves ``bash`` to
    ``/bin/bash``, even where one is a symlink to the other. Listing each
    existing spelling keeps the grant working on both layouts without
    widening it: they name the same executable.
    """
    found: list[str] = []
    resolved = shutil.which(program)
    if resolved:
        found.append(resolved)
    found.extend(c for c in candidates if c not in found and Path(c).exists())
    return found or [candidates[0]]


def purge_commands() -> list[str]:
    """Every command line the realtime core runs to purge a poisoned queue.

    Two callers, one grant: the realtime core's ``purge_tx_queue``
    (``rust/axol-rt/src/safety.rs``) runs the arm hub's USB reset script,
    else the bring-up script, falling back to a per-interface down/up pair,
    and the bring-up backstop (``almond_axol.cli.can.setup.purge_stale_tx``)
    runs the same scripts, falling back to the full configure sequence.

    Imported lazily: ``cli.can.setup`` is a heavy import and this is the only
    thing needed from it.
    """
    from ..cli.can.setup import _BITRATE, _TXQUEUELEN

    commands = [
        f"{bash} {script}"
        # The reset script first: the core prefers it for the arm hub, whose
        # firmware keeps frames a flap of the bring-up script cannot reach.
        for script in (CAN_RESET_SCRIPT, CAN_BRINGUP_SCRIPT)
        for bash in _program_paths("bash", ("/usr/bin/bash", "/bin/bash"))
    ]
    for ip in _program_paths("ip", ("/usr/sbin/ip", "/sbin/ip", "/usr/bin/ip")):
        for iface in _PURGEABLE_INTERFACES:
            for form in (
                "down",
                "up",
                # The configure steps a flap through `bring_up_interfaces`
                # issues between the two (`_bring_up_interfaces_locked`).
                # Without them a non-root fallback flap stops with the
                # interface down, which is worse than never flapping.
                f"type can bitrate {_BITRATE}",
                f"txqueuelen {_TXQUEUELEN}",
            ):
                commands.append(f"{ip} link set {iface} {form}")
    return commands


def sudoers_text(user: str) -> str:
    """The drop-in granting *user* passwordless use of :func:`purge_commands`."""
    commands = purge_commands()
    alias = ", \\\n".join(
        (" " * 4 if index else "") + command for index, command in enumerate(commands)
    )
    return (
        "# Written by `axol provision` (almond_axol.utils.can_purge).\n"
        "# Lets the realtime core flap a CAN interface without a password when\n"
        "# it declares the bus stalled — the e-stop. Without it the kernel's\n"
        "# queued position commands survive the session and replay when the\n"
        "# motors come back, snapping the arm to where it was commanded when\n"
        "# power died. The hosted `axol serve` is root and needs none of this;\n"
        "# this covers a manual serve/teleop from the operator's shell.\n"
        "# Every target is root-owned and root-writable only.\n"
        f"Cmnd_Alias AXOL_CAN_PURGE = {alias}\n"
        f"{user} ALL=(root) NOPASSWD: AXOL_CAN_PURGE\n"
    )


def _is_valid_sudoers(path: Path) -> bool:
    """Whether ``visudo`` accepts *path*; never install what it rejects.

    A malformed file in ``sudoers.d`` breaks **every** sudo invocation on the
    host, so the generated bytes are checked before they are published.
    ``visudo -c`` needs no privileges to check a file it can read.
    """
    visudo = shutil.which("visudo") or "/usr/sbin/visudo"
    try:
        checked = subprocess.run(
            [visudo, "-cf", str(path)], capture_output=True, text=True
        )
    except OSError as exc:
        _logger.warning("could not run visudo to check the CAN purge rule: %s", exc)
        return False
    if checked.returncode != 0:
        _logger.warning(
            "visudo rejected the generated CAN purge rule: %s",
            (checked.stderr or checked.stdout).strip(),
        )
    return checked.returncode == 0


def install() -> None:
    """Grant the operator passwordless use of the e-stop CAN purge.

    Idempotent and best-effort, like the other provisioning grants: a no-op
    when the installed rule already matches, and a warning naming the manual
    command when root cannot be obtained. Takes effect immediately — unlike
    the rtprio grant, no re-login is involved.
    """
    user = operator_user()
    if user is None or user == "root":
        _logger.info("no operator account found; skipping the CAN purge grant")
        return
    wanted = sudoers_text(user)

    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / SUDOERS_PATH.name
        staged.write_text(wanted)
        staged.chmod(0o440)
        if not _is_valid_sudoers(staged):
            raise RuntimeError(
                f"refusing to install {SUDOERS_PATH}: visudo rejected it "
                "(a malformed sudoers.d file breaks sudo host-wide)"
            )
        # 0440 root-owned, so an operator cannot read the installed copy back
        # to compare; ask root to do the comparison instead of re-installing
        # (and re-prompting) on every provision run.
        if os.geteuid() == 0 and SUDOERS_PATH.exists():
            if SUDOERS_PATH.read_text() == wanted:
                _logger.info(
                    "CAN purge grant already in place for %s (%s)", user, SUDOERS_PATH
                )
                return
        elif os.geteuid() != 0:
            if not prime_sudo():
                _logger.warning(
                    "the CAN purge grant needs root; without it a manual "
                    "`axol serve` from %s's shell cannot clear the CAN TX "
                    "queue after an e-stop, and the arm jerks to its "
                    "pre-e-stop command on the next start. Run manually: "
                    "sudo install -o root -g root -m 0440 <file> %s",
                    user,
                    SUDOERS_PATH,
                )
                return
            if run_root(["cmp", "-s", str(staged), str(SUDOERS_PATH)]).returncode == 0:
                _logger.info(
                    "CAN purge grant already in place for %s (%s)", user, SUDOERS_PATH
                )
                return
        run_root(
            [
                "install",
                "-d",
                "-o",
                "root",
                "-g",
                "root",
                "-m",
                "0755",
                str(SUDOERS_PATH.parent),
            ],
            check=True,
        )
        run_root(
            [
                "install",
                "-o",
                "root",
                "-g",
                "root",
                "-m",
                "0440",
                str(staged),
                str(SUDOERS_PATH),
            ],
            check=True,
        )
    _logger.info(
        "CAN e-stop purge granted to %s via %s — a manual `axol serve` can now "
        "flap the bus when motor power dies",
        user,
        SUDOERS_PATH,
    )
