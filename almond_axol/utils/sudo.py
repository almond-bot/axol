"""Shared privilege-escalation helpers.

Several Axol operations need root for a handful of system commands (CAN
bring-up, the persistent ``can.setup`` configuration, PTP clock-sync daemons).
None of the entry points run as root themselves, so each escalates individual
commands via ``sudo``. This module is the single place that knows how to do
that consistently across two execution contexts:

* In-process, run-to-completion commands (``run_root``) — used by CAN bring-up
  and ``can.setup``. Tries cached/passwordless sudo first, then falls back to a
  supplied password fed on stdin (``sudo -S``), and raises a typed exception
  when a password is needed or rejected so callers can prompt the operator.
* Spawned long-lived daemons in a child process (PTP). Those can't run to
  completion here, but they share :data:`SUDO_PASSWORD_ENV` and the sentinel
  markers/codes below so the same password the operator typed flows through and
  the same "needs sudo" / "bad password" states surface in the UI.
"""

from __future__ import annotations

import os
import subprocess

# Out-of-band channel for a sudo password: the web control panel forwards the
# password the operator typed via the environment so it never lands in argv /
# process listings. Read by both in-process callers and spawned CLI commands.
SUDO_PASSWORD_ENV = "AXOL_SUDO_PASSWORD"

# Printed to stdout (followed by a distinct exit code) by spawned CLI commands
# so a parent orchestrator can tell a *missing* password apart from a *wrong*
# one, instead of just watching a daemon die.
SUDO_REQUIRED_MARKER = "AXOL_SUDO_REQUIRED"
SUDO_BAD_PASSWORD_MARKER = "AXOL_SUDO_BAD_PASSWORD"
SUDO_REQUIRED_CODE = 87
SUDO_BAD_PASSWORD_CODE = 88


class SudoPasswordRequired(Exception):
    """A privileged command needs root, but passwordless sudo is unavailable
    and no password was supplied.

    Callers surface this to the UI so it can prompt for a password and retry.
    """


class SudoPasswordIncorrect(Exception):
    """The supplied sudo password was rejected by ``sudo``."""


def password_from_env() -> str | None:
    """Return the out-of-band sudo password, if one was provided."""
    return os.environ.get(SUDO_PASSWORD_ENV) or None


def _is_auth_failure(proc: subprocess.CompletedProcess[str]) -> bool:
    """True when ``sudo -n`` failed *because it needed a password*.

    Distinguishes "sudo couldn't authenticate (the command never ran)" from
    "sudo authenticated and the command itself failed" — only the former should
    be retried with a password, otherwise a side-effecting command could run
    twice.
    """
    if proc.returncode == 0:
        return False
    err = (proc.stderr or "").lower()
    return (
        "password is required" in err
        or "a terminal is required" in err
        or "no askpass" in err
        or "a password is required" in err
    )


def _is_bad_password(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    err = (proc.stderr or "").lower()
    return "incorrect password" in err or "try again" in err


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
    password: str | None = None,
    input_text: str | None = None,
    allow_prompt: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` as root, escalating via ``sudo`` only as needed.

    Resolution order:

    1. Already root (``geteuid() == 0``): run ``cmd`` directly.
    2. ``sudo -n`` (passwordless config or cached credentials). ``sudo`` exits
       *before* exec'ing ``cmd`` when it can't authenticate, so this never
       double-runs a side-effecting command.
    3. If step 2 needed a password:
       * with ``password`` → retry ``sudo -S`` feeding it on stdin
         (:class:`SudoPasswordIncorrect` if rejected);
       * else with ``allow_prompt`` (interactive terminal) → plain ``sudo`` so
         it can prompt on the tty;
       * else → :class:`SudoPasswordRequired`.

    ``input_text`` is forwarded to the command's stdin (after the password line
    when ``sudo -S`` is used), so commands like ``tee`` and ``crontab -`` work.
    """
    if os.geteuid() == 0:
        proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
        return _finish(proc, check=check, cmd=cmd)

    probe = subprocess.run(
        ["sudo", "-n", *cmd], input=input_text, capture_output=True, text=True
    )
    if not _is_auth_failure(probe):
        # Either it ran (success) or failed for a non-auth reason; don't retry.
        return _finish(probe, check=check, cmd=cmd)

    if password:
        stdin = f"{password}\n" + (input_text or "")
        result = subprocess.run(
            ["sudo", "-S", "-p", "", *cmd],
            input=stdin,
            capture_output=True,
            text=True,
        )
        if _is_bad_password(result):
            raise SudoPasswordIncorrect()
        return _finish(result, check=check, cmd=cmd)

    if allow_prompt:
        # Interactive terminal: plain ``sudo`` prompts for the password on the
        # controlling tty (/dev/tty), independent of stdout/stderr, so we can
        # still capture output for the caller while the prompt stays visible.
        result = subprocess.run(
            ["sudo", *cmd], input=input_text, capture_output=True, text=True
        )
        return _finish(result, check=check, cmd=cmd)

    raise SudoPasswordRequired()
