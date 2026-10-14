"""Root ``apt-get`` / ``dpkg`` runs that nothing of ours can interrupt.

A package manager killed part-way leaves dpkg with a half-configured package,
and every later install on the host then fails until an operator runs ``dpkg
--configure -a``. Provisioning installs packages from several places that can
be torn down mid-run -- a subprocess timeout, the self-updater's bound on a
whole ``axol provision``, and ``axol.service`` stopping or restarting (an
update, the installer, a reboot from the panel) while its startup heal is
installing -- so every package-manager call goes through
:func:`run_package_manager`, which:

* never times out (a hung run is the caller's to report, not to kill);
* runs as root in its own session, so a kill aimed at the caller's process
  group does not reach it;
* runs in its own transient systemd scope when systemd is the init, so
  stopping ``axol.service`` (which kills the service's whole cgroup) leaves it
  to finish;
* waits for another package manager's lock instead of failing at once, and
  never prompts (debconf), since its output is captured.

:func:`failure_detail` turns a failed run into one actionable line: the
package dpkg could not process and the repair command, rather than apt's
generic last line ("Sub-process /usr/bin/dpkg returned an error code (1)").
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# How long apt waits for another package manager (unattended-upgrades, an
# operator's apt, a run of ours that outlived its caller) to release the lock.
DPKG_LOCK_WAIT_S = 600
_APT_COMMANDS = frozenset({"apt-get", "apt"})
_REPAIR = "repair the host with `sudo dpkg --configure -a`, then retry"


def _systemd_booted() -> bool:
    """Whether systemd is the running init (``sd_booted``'s check)."""
    return Path("/run/systemd/system").is_dir()


def package_manager_command(cmd: list[str]) -> list[str]:
    """The full argv :func:`run_package_manager` runs for ``cmd``."""
    if Path(cmd[0]).name in _APT_COMMANDS:
        cmd = [cmd[0], "-o", f"DPkg::Lock::Timeout={DPKG_LOCK_WAIT_S}", *cmd[1:]]
    full = ["env", "DEBIAN_FRONTEND=noninteractive", *cmd]
    systemd_run = shutil.which("systemd-run")
    if systemd_run is not None and _systemd_booted():
        full = [
            systemd_run,
            "--scope",
            "--quiet",
            "--collect",
            "--description=Axol package install",
            *full,
        ]
    if os.geteuid() != 0:
        full = ["sudo", *full]
    return full


def run_package_manager(
    cmd: list[str], *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run an ``apt-get`` / ``dpkg`` command as root, to completion.

    Escalates through ``sudo`` like :func:`almond_axol.utils.sudo.run_root`
    (callers prime it first). Only a root caller gets its own session: sudo's
    cached credentials are tied to the terminal, which a new session drops,
    and an operator's Ctrl-C at that terminal is deliberate. With ``check``, a
    failure raises ``RuntimeError`` carrying :func:`failure_detail`.
    """
    full = package_manager_command(cmd)
    proc = subprocess.run(
        full,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        start_new_session=os.geteuid() == 0,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {failure_detail(proc)}")
    return proc


def failure_detail(proc: subprocess.CompletedProcess[str]) -> str:
    """One line explaining a failed package-manager run, with the fix."""
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    lines = text.splitlines()
    broken: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip().startswith("Errors were encountered while processing"):
            continue
        for following in lines[index + 1 :]:
            if not following.strip() or not following[:1].isspace():
                break
            broken.append(following.strip())
    if broken:
        names = ", ".join(dict.fromkeys(broken))
        return f"dpkg could not process {names}; {_REPAIR}"
    if "dpkg was interrupted" in text:
        return f"an earlier package install was interrupted; {_REPAIR}"
    if "returned an error code" in text and "dpkg" in text:
        return f"dpkg failed while configuring packages; {_REPAIR}"
    tail = [line.strip() for line in lines if line.strip()]
    return tail[-1] if tail else f"exit code {proc.returncode}"
