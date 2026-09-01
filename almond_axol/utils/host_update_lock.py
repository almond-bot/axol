"""Host-wide serialization for Axol install/update/provision mutations."""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
from collections.abc import Iterator
from pathlib import Path

LOCK_ENV = "AXOL_UPDATE_LOCK_FD"
STATE_DIR = Path("/var/lib/almond-axol")
LOCK_PATH = STATE_DIR / "update.lock"


class HostUpdateLockError(RuntimeError):
    """The host update lock is unsafe or already owned."""


def _validate_directory() -> None:
    try:
        # Keep the trust boundary root-only while allowing the persisted
        # non-root operator to traverse to the group-readable datasets child.
        created = False
        try:
            STATE_DIR.mkdir(mode=0o751, parents=False)
            created = True
        except FileExistsError:
            pass
        if created:
            # The managed service is root with the operator's effective group;
            # force privileged state back to root:root after creation.
            os.chown(STATE_DIR, 0, 0)
            os.chmod(STATE_DIR, 0o751)
        metadata = STATE_DIR.stat(follow_symlinks=False)
    except OSError as exc:
        raise HostUpdateLockError(
            "cannot prepare the Axol update state directory"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or metadata.st_mode & 0o022
    ):
        raise HostUpdateLockError("the Axol update state directory is unsafe")


def _validate_fd(fd: int) -> None:
    try:
        descriptor = os.fstat(fd)
        path = LOCK_PATH.stat(follow_symlinks=False)
    except OSError as exc:
        raise HostUpdateLockError("cannot validate the Axol update lock") from exc
    if (
        not stat.S_ISREG(descriptor.st_mode)
        or descriptor.st_uid != 0
        or descriptor.st_gid != 0
        or stat.S_IMODE(descriptor.st_mode) != 0o600
        or (descriptor.st_dev, descriptor.st_ino) != (path.st_dev, path.st_ino)
    ):
        raise HostUpdateLockError("the Axol update lock is unsafe")


def _inherited_fd() -> int | None:
    raw = os.environ.get(LOCK_ENV)
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdecimal():
        raise HostUpdateLockError(f"{LOCK_ENV} is invalid")
    fd = int(raw)
    _validate_fd(fd)
    return fd


@contextlib.contextmanager
def host_update_lock() -> Iterator[None]:
    """Take the non-blocking root lock, accepting a validated inherited FD.

    The hosted Bash installer already owns the lock when it invokes
    ``axol provision``. It passes that exact descriptor through ``LOCK_ENV``;
    re-locking the inherited open-file description succeeds without deadlock.
    Standalone/root startup provisioning opens and owns the same lock itself.
    """
    if os.geteuid() != 0:
        raise HostUpdateLockError("Axol provisioning requires root")
    _validate_directory()
    inherited = _inherited_fd()
    owned_fd: int | None = None
    if inherited is None:
        flags = os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            try:
                owned_fd = os.open(LOCK_PATH, flags | os.O_CREAT | os.O_EXCL, 0o600)
                os.fchown(owned_fd, 0, 0)
                os.fchmod(owned_fd, 0o600)
            except FileExistsError:
                owned_fd = os.open(LOCK_PATH, flags)
        except OSError as exc:
            if owned_fd is not None:
                os.close(owned_fd)
            raise HostUpdateLockError("cannot open the Axol update lock") from exc
        fd = owned_fd
        _validate_fd(fd)
    else:
        fd = inherited
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise HostUpdateLockError(
                "another Axol install, update, or provisioning transaction is active"
            ) from exc
        yield
    finally:
        if owned_fd is not None:
            os.close(owned_fd)
