"""Shared privilege-escalation helper.

A handful of Axol operations need root for system commands (CAN bring-up, the
persistent ``can.setup`` configuration, ZED daemon restarts). The hosted
install runs ``axol serve`` as root under systemd, so those commands run
directly; interactive CLI use from a terminal escalates via ``sudo``, which
prompts on the tty as usual.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _finish(
    proc: subprocess.CompletedProcess[str], *, check: bool, cmd: list[str]
) -> subprocess.CompletedProcess[str]:
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit code {proc.returncode}"
        raise RuntimeError(f"`{cmd[0]}` failed: {detail}")
    return proc


def prime_sudo() -> bool:
    """Ensure subsequent ``sudo -n`` invocations will succeed.

    Root processes (e.g. ZED daemon restarts) are spawned with
    ``sudo -n`` so headless contexts (web control panel) fail fast instead of
    blocking on a password prompt that can never be answered. Call this first
    so interactive CLI use still works: when there is a tty and no cached
    credentials, ``sudo -v`` prompts once and caches them for the ``sudo -n``
    invocations that follow.

    Returns True when ``sudo -n`` will work (root, passwordless/cached sudo,
    or credentials just cached via the tty prompt); False when escalation is
    impossible.
    """
    if os.geteuid() == 0:
        return True
    if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
        return True
    if sys.stdin is not None and sys.stdin.isatty():
        return subprocess.run(["sudo", "-v"]).returncode == 0
    return False


def run_root(
    cmd: list[str],
    *,
    input_text: str | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` as root, escalating via ``sudo`` only as needed.

    Already root (``geteuid() == 0``): run ``cmd`` directly. Otherwise prepend
    ``sudo``, which prompts on the controlling tty (/dev/tty) when a password
    is needed — independent of stdout/stderr, so output capture doesn't hide
    the prompt.

    ``input_text`` is forwarded to the command's stdin, so commands like
    ``tee`` and ``crontab -`` work.
    """
    if os.geteuid() != 0:
        cmd = ["sudo", *cmd]
    proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    return _finish(proc, check=check, cmd=cmd)
