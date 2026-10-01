"""Keep the ZED SDK's per-camera calibration files readable by every account.

The SDK caches each camera's factory calibration in ``/usr/local/zed/settings/
SN<serial>.conf`` the first time that camera is opened on a host, creating the
file as whatever user the opening process runs as, with the process umask. The
hosted install runs ``axol serve`` as root under systemd (with a ``027`` umask),
so on a fresh box every calibration file lands ``root:root 0640``. That is fine
until the operator opens the same camera from their own login — a dev-checkout
``axol serve``, ``axol teleop``, ``axol diag.zed-cable``, ``ZED_Explorer`` —
at which point the SDK can neither read the cached file nor overwrite it with a
fresh download, and ``open()`` fails with ``CALIBRATION FILE NOT AVAILABLE``
(logged as "Unable to download the calibration file"). The half-opened Argus
session that leaves behind then takes every later attempt down with it
(``CAMERA STREAM FAILED TO START``), which looks like a camera or cable fault.

The Stereolabs installer already puts the operator in the ``zed`` group and
makes the SDK tree ``user:zed 0770``. :func:`share_calibration_files` extends
that to the calibration cache: group ``zed``, group read/write on every file,
and the setgid bit on the directory so files the root service creates *later*
(a camera plugged in next month) inherit the group and are readable through
the root umask's ``0640``. Root applies it directly; an operator's terminal run
escalates through the shared ``sudo`` helper like the other provisioning steps.
"""

from __future__ import annotations

import grp
import logging
import os
import stat
from pathlib import Path

from ..utils.sudo import prime_sudo, run_root

_logger = logging.getLogger(__name__)

SETTINGS_DIR = Path("/usr/local/zed/settings")
SHARE_GROUP = "zed"

# Directory: group rwx (traverse + let the operator's SDK write new/updated
# files) plus setgid so new files inherit the group regardless of who opens
# the camera first. Files: group rw. Both are applied additively so nothing the
# Stereolabs installer set is taken away.
_DIR_GROUP_BITS = stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID
_FILE_GROUP_BITS = stat.S_IRGRP | stat.S_IWGRP


def calibration_files(directory: Path = SETTINGS_DIR) -> list[Path]:
    """Every cached calibration file (``SN<serial>.conf``) under ``directory``."""
    try:
        return sorted(p for p in directory.glob("SN*.conf") if p.is_file())
    except OSError:
        return []


def calibration_path(serial: int, directory: Path = SETTINGS_DIR) -> Path:
    return directory / f"SN{int(serial)}.conf"


def unreadable_calibration_files(directory: Path = SETTINGS_DIR) -> list[Path]:
    """Cached calibration files this process cannot read (always empty as root)."""
    return [p for p in calibration_files(directory) if not os.access(p, os.R_OK)]


def calibration_hint(serial: int | None = None, directory: Path = SETTINGS_DIR) -> str:
    """One-sentence diagnosis to append to a camera-open error, or ``""``.

    Non-empty only when the calibration file for ``serial`` (or, with no
    serial, any cached calibration file) exists but is unreadable by this
    process — the signature of a file written by another account.
    """
    if serial is not None:
        path = calibration_path(serial, directory)
        unreadable = [path] if path.exists() and not os.access(path, os.R_OK) else []
    else:
        unreadable = unreadable_calibration_files(directory)
    if not unreadable:
        return ""
    shown = ", ".join(str(p) for p in unreadable[:3])
    if len(unreadable) > 3:
        shown += f" (+{len(unreadable) - 3} more)"
    return (
        f" The ZED calibration file {shown} exists but is not readable by this "
        f"user (uid {os.geteuid()}) — it was written by another account, "
        "typically the root axol service. Fix with `sudo axol provision` "
        f"(or: sudo chgrp {SHARE_GROUP} {shown} && sudo chmod g+rw {shown})."
    )


def _share_plan(directory: Path, gid: int) -> list[list[str]]:
    """Commands (without ``sudo``) that bring ``directory`` to the shared layout.

    Empty when everything is already in place, so callers can stay quiet and
    skip escalation on an already-provisioned host.
    """
    plan: list[list[str]] = []
    st = directory.stat()
    if st.st_gid != gid:
        plan.append(["chgrp", str(gid), str(directory)])
    if (st.st_mode & _DIR_GROUP_BITS) != _DIR_GROUP_BITS:
        plan.append(["chmod", "g+rwx,g+s", str(directory)])

    regroup: list[str] = []
    remode: list[str] = []
    for path in calibration_files(directory):
        fst = path.stat()
        if fst.st_gid != gid:
            regroup.append(str(path))
        if (fst.st_mode & _FILE_GROUP_BITS) != _FILE_GROUP_BITS:
            remode.append(str(path))
    if regroup:
        plan.append(["chgrp", str(gid), *regroup])
    if remode:
        plan.append(["chmod", "g+rw", *remode])
    return plan


def share_calibration_files(
    *, directory: Path = SETTINGS_DIR, group: str = SHARE_GROUP
) -> bool:
    """Make the calibration cache group-shared (see the module docstring).

    Idempotent and a quiet no-op without the ZED SDK. Returns ``True`` when the
    cache is shared (already, or now), ``False`` when it could not be repaired
    from this process (no ``zed`` group, no way to escalate, command failure)
    after logging why — never raises, so provisioning and camera start-up
    can treat it as best-effort.
    """
    if not directory.is_dir():
        _logger.info("no ZED calibration cache at %s; skipping", directory)
        return True
    try:
        gid = grp.getgrnam(group).gr_gid
    except KeyError:
        _logger.warning(
            "no `%s` group on this host; cannot share the ZED calibration cache "
            "%s between root and operator logins",
            group,
            directory,
        )
        return False
    try:
        plan = _share_plan(directory, gid)
    except OSError as exc:
        _logger.warning(
            "cannot inspect the ZED calibration cache %s: %s", directory, exc
        )
        return False
    if not plan:
        _logger.info(
            "ZED calibration cache %s already shared (group %s)", directory, group
        )
        return True
    if not prime_sudo():
        _logger.warning(
            "ZED calibration cache %s needs root to share between accounts; "
            "run `sudo axol provision` (or: sudo chgrp -R %s %s && "
            "sudo chmod g+rwx,g+s %s && sudo chmod g+rw %s/SN*.conf)",
            directory,
            group,
            directory,
            directory,
            directory,
        )
        return False
    try:
        for cmd in plan:
            run_root(cmd, check=True)
    except (OSError, RuntimeError) as exc:
        _logger.warning(
            "could not share the ZED calibration cache %s: %s", directory, exc
        )
        return False
    _logger.info("ZED calibration cache %s shared with group %s", directory, group)
    return True


def ensure_calibration_readable() -> None:
    """Best-effort start-up hook for anything about to open ZED cameras.

    As root (the hosted service, ``sudo axol ...``) it always reconciles the
    cache, so the directory carries the setgid bit before the first camera is
    ever opened on a box. As an operator it only acts when a cached file is
    actually unreadable — that is the one case worth a ``sudo`` prompt on an
    interactive terminal; headless runs log the manual fix instead.
    """
    try:
        if os.geteuid() == 0 or unreadable_calibration_files():
            share_calibration_files()
    except Exception as exc:  # noqa: BLE001 - never block camera start-up on this
        _logger.warning("ZED calibration cache check failed: %s", exc)
