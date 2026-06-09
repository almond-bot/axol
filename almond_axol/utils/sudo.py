"""Shared privilege-escalation helper.

A handful of Axol operations need root for system commands (CAN bring-up, the
persistent ``can.setup`` configuration, PTP clock-sync daemons). The hosted
install runs ``axol serve`` as root under systemd, so those commands run
directly; interactive CLI use from a terminal escalates via ``sudo``, which
prompts on the tty as usual.
"""

from __future__ import annotations

import os
import subprocess


def _finish(
    proc: subprocess.CompletedProcess[str], *, check: bool, cmd: list[str]
) -> subprocess.CompletedProcess[str]:
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit code {proc.returncode}"
        raise RuntimeError(f"`{cmd[0]}` failed: {detail}")
    return proc


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
