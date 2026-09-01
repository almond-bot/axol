"""Race-safe file operations for root services writing operator-owned trees.

The hosted service intentionally shares ``ALMOND_HOME`` and dataset paths with
an unprivileged operator.  Ordinary ``Path.write_text`` or a predictable
``file.tmp`` is unsafe there: an operator can replace a parent or temporary
path with a symlink while the root process is writing.  These helpers walk
every directory component with ``openat(..., O_NOFOLLOW)``, keep the resulting
directory descriptor pinned, and perform final operations relative to it.
"""

from __future__ import annotations

import json
import os
import pwd
import re
import secrets
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
_FILE_READ_FLAGS = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
_PRIVILEGED_SERVICE_ENV = "AXOL_PRIVILEGED_SERVICE"
_SERVICE_DATASET_ROOT_ENV = "AXOL_SERVICE_DATASET_ROOT"
_SERVICE_OPERATOR_UID_ENV = "AXOL_OPERATOR_UID"
_SERVICE_OPERATOR_GID_ENV = "AXOL_OPERATOR_GID"
_SERVICE_REPO_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
HOSTED_DATASET_ROOT = Path("/var/lib/almond-axol/datasets")


class UnsafeStatePathError(OSError):
    """A state path contains a symlink, special file, or hard-linked file."""


def privileged_service_active() -> bool:
    """Whether this is the hosted root service sharing operator state."""
    return os.geteuid() == 0 and os.environ.get(_PRIVILEGED_SERVICE_ENV) == "1"


def mark_privileged_service() -> None:
    """Mark this process and its operation children as the hosted service."""
    os.environ[_PRIVILEGED_SERVICE_ENV] = "1"


def _service_numeric_id(name: str) -> int:
    raw = os.environ.get(name, "0")
    if not raw.isascii() or not raw.isdecimal():
        raise UnsafeStatePathError(f"{name} must be a numeric id")
    value = int(raw)
    if value < 0 or value > 2**31 - 1:
        raise UnsafeStatePathError(f"{name} is outside the supported range")
    return value


def service_operator_ids() -> tuple[int, int]:
    """Return the installer's numeric operator identity, fail closed."""
    return (
        _service_numeric_id(_SERVICE_OPERATOR_UID_ENV),
        _service_numeric_id(_SERVICE_OPERATOR_GID_ENV),
    )


def service_operator_gid() -> int:
    """Return the installed service's read-only dataset group, fail closed."""
    return service_operator_ids()[1]


def _existing_root_controlled_directory(path: Path, *, label: str) -> bool:
    """Whether ``path`` exists as a root-owned, non-writable directory.

    Missing directories are initialized by :func:`configure_root_service_dataset`.
    An existing unsafe directory is never silently repaired: it may already
    contain attacker-selected entries from when it was writable.
    """
    try:
        metadata = secure_directory_stat(path)
    except FileNotFoundError:
        return False
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise UnsafeStatePathError(
            f"{label} is not root-controlled at {path}: expected root ownership "
            "with no group/world write bits"
        )
    return True


def _resolve_service_operator(operator: str | None) -> tuple[int, int, str]:
    """Resolve the non-root account a manual root serve should expose data to."""
    candidate = operator if operator is not None else os.environ.get("SUDO_USER")
    if not candidate or candidate == "root":
        raise UnsafeStatePathError(
            "manual root `axol serve` needs a non-root operator account; run it "
            "through `sudo` from that account or pass `--operator USER`"
        )
    try:
        entry = pwd.getpwnam(candidate)
    except KeyError as exc:
        raise UnsafeStatePathError(
            f"manual root `axol serve` operator does not exist: {candidate!r}"
        ) from exc
    if entry.pw_uid == 0:
        raise UnsafeStatePathError(
            "manual root `axol serve` operator must be a non-root account"
        )
    return entry.pw_uid, entry.pw_gid, entry.pw_name


def configure_root_service_dataset(operator: str | None = None) -> Path:
    """Initialize the fixed hosted dataset boundary for a root ``axol serve``.

    Installed systemd services already provide the three service environment
    variables. A manual ``sudo axol serve`` does not, so resolve its operator
    from ``SUDO_USER`` (or ``--operator``), create the same immutable
    ``/var/lib`` boundary as the installer, and export the exact environment
    consumed by all dataset confinement and ownership helpers.

    A caller-supplied dataset path is deliberately unsupported. Existing
    directories must already be root-controlled before their modes/groups are
    normalized, preventing a formerly writable tree with planted links from
    being promoted into the privileged service boundary.
    """
    if os.geteuid() != 0:
        raise UnsafeStatePathError(
            "hosted dataset storage can only be initialized by root"
        )

    root = _absolute(HOSTED_DATASET_ROOT / ".directory-sentinel").parent
    configured = os.environ.get(_SERVICE_DATASET_ROOT_ENV)
    if configured is not None:
        configured_root = _absolute(Path(configured) / ".directory-sentinel").parent
        if configured_root != root:
            raise UnsafeStatePathError(
                "root `axol serve` dataset storage must use the fixed boundary "
                f"{root}, not {configured_root}"
            )

    have_installed_identity = all(
        os.environ.get(name)
        for name in (_SERVICE_OPERATOR_UID_ENV, _SERVICE_OPERATOR_GID_ENV)
    )
    if operator is not None or not have_installed_identity:
        uid, gid, _operator_name = _resolve_service_operator(operator)
    else:
        uid, gid = service_operator_ids()

    parent = root.parent
    # /var/lib (or the patched equivalent in a focused test) must already be an
    # immutable root boundary. Only then may we safely create fixed child names.
    _require_root_controlled_directory(parent.parent, label="service state root")
    if not _existing_root_controlled_directory(parent, label="service state root"):
        secure_ensure_directory(parent, mode=0o751)
    if not _existing_root_controlled_directory(root, label="service dataset root"):
        secure_ensure_directory(root, mode=0o2750)

    # Operators receive read/traverse access through their primary group, never
    # write access. Dataset saves apply the same group and modes to descendants.
    # Keep the shared service-state parent root:root: the update lock and boot
    # verifier require that exact ownership. Execute-only access lets the
    # operator traverse to the group-restricted dataset child without exposing
    # update markers or allowing directory enumeration.
    secure_chown_directory(parent, 0, 0, mode=0o751)
    secure_chown_directory(root, 0, gid, mode=0o2750)
    _require_root_controlled_directory(root, label="service dataset root")

    os.environ[_SERVICE_DATASET_ROOT_ENV] = str(root)
    os.environ[_SERVICE_OPERATOR_UID_ENV] = str(uid)
    os.environ[_SERVICE_OPERATOR_GID_ENV] = str(gid)
    os.environ["HF_LEROBOT_HOME"] = str(root)
    return root


def _require_root_controlled_directory(path: str | Path, *, label: str) -> Path:
    """Prove a directory and all ancestors are root-owned and non-writable."""
    candidate = _absolute(Path(path) / ".directory-sentinel").parent
    chain = [candidate, *candidate.parents]
    for component in reversed(chain):
        if component == Path(component.anchor):
            metadata = os.stat(component, follow_symlinks=False)
        else:
            metadata = secure_directory_stat(component)
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise UnsafeStatePathError(
                f"{label} is not root-controlled at {component}: expected "
                "root ownership with no group/world write bits"
            )
    return candidate


def validated_service_dataset_root() -> Path:
    """Return the immutable hosted dataset root, or fail closed.

    Service-side discovery must use the same sealed boundary as mutation.
    Otherwise a stale ``recording.root`` value could turn a root API into an
    arbitrary local-file scanner even though recording itself is confined.
    """
    if not privileged_service_active():
        raise UnsafeStatePathError(
            "hosted dataset storage is only available to the privileged service"
        )
    configured = os.environ.get(_SERVICE_DATASET_ROOT_ENV)
    if not configured:
        raise UnsafeStatePathError(
            "root control-panel dataset storage is not configured; reinstall "
            "Axol or run the dataset command directly as the non-root operator"
        )
    service_operator_ids()
    return _require_root_controlled_directory(
        configured,
        label="service dataset root",
    )


def service_dataset_path_for_repo_id(repo_id: object) -> Path:
    """Map one hosted repo id to a specific directory below the sealed root.

    LeRobot accepts a local path independently of ``repo_id``. The panel must
    not let either field reintroduce arbitrary root filesystem access, so its
    effective path is derived solely from a conservative one- or two-component
    repo id (``name`` or ``owner/name``).
    """
    boundary = validated_service_dataset_root()
    if not isinstance(repo_id, str) or repo_id != repo_id.strip():
        raise UnsafeStatePathError(
            "hosted dataset repo_id must be an ASCII name or owner/name"
        )
    components = repo_id.split("/")
    if (
        len(components) not in (1, 2)
        or components[0] == "hub"
        or any(
            _SERVICE_REPO_COMPONENT.fullmatch(component) is None or ".." in component
            for component in components
        )
    ):
        raise UnsafeStatePathError(
            "hosted dataset repo_id must be an ASCII name or owner/name "
            "using only letters, digits, '.', '_', and '-'"
        )
    return boundary.joinpath(*components)


def confine_service_dataset_path(path: str | Path, *, label: str) -> Path:
    """Confine hosted third-party dataset I/O to an immutable root-owned tree.

    LeRobot/PyArrow/PyAV reopen names internally, so checking for symlinks in
    an operator-owned tree cannot close the race. The installed service instead
    uses a root-owned, non-group-writable boundary below ``/var/lib``. Once its
    full ancestry and every existing descendant are proven immutable to the
    operator, ordinary library path resolution cannot be redirected.
    """
    if not privileged_service_active():
        return _absolute(Path(path) / ".path-sentinel").parent
    boundary = validated_service_dataset_root()
    candidate = require_path_beneath(path, boundary, label=label)
    relative = candidate.relative_to(boundary)
    current = boundary
    for component in relative.parts:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if (
            metadata.st_uid != 0
            or stat.S_ISLNK(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise UnsafeStatePathError(f"{label} is not root-controlled at {current}")
    return candidate


def require_service_dataset_configuration() -> None:
    """Validate the hosted immutable dataset boundary before hardware starts."""
    if not privileged_service_active():
        return
    validated_service_dataset_root()


def require_path_beneath(
    path: str | Path,
    root: str | Path,
    *,
    label: str,
) -> Path:
    """Fail closed unless ``path`` is a non-root descendant of ``root``.

    This prevents a control-panel argument from directly naming ``/etc`` or
    another privileged tree. Existing components are also rejected if they are
    links. The check is defense in depth for third-party writers and is not a
    substitute for descriptor-relative I/O: an operator can still race a
    later library that re-resolves the path.
    """
    candidate = _absolute(Path(path) / ".path-sentinel").parent
    boundary = _absolute(Path(root) / ".path-sentinel").parent
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise UnsafeStatePathError(
            f"{label} must stay below the configured operator root {boundary}"
        ) from exc
    if not relative.parts:
        raise UnsafeStatePathError(f"{label} must name a directory below {boundary}")

    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafeStatePathError(f"{label} contains a symlink: {current}")
    return candidate


def _absolute(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))
    if candidate.name in {"", ".", ".."}:
        raise UnsafeStatePathError("state path must name a file")
    return candidate


def _open_parent(path: str | Path, *, create: bool) -> tuple[Path, int, str]:
    """Open the parent directory without following any path-component link."""
    target = _absolute(path)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in target.parent.parts[1:]:
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                parent_owner = os.fstat(descriptor)
                created = False
                try:
                    os.mkdir(component, mode=0o770, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    # Another writer won the create. The no-follow open below
                    # still decides whether the resulting entry is safe.
                    pass
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                try:
                    if created:
                        if os.geteuid() == 0:
                            os.fchown(child, parent_owner.st_uid, parent_owner.st_gid)
                        directory_mode = (
                            0o2770 if parent_owner.st_mode & stat.S_ISGID else 0o770
                        )
                        os.fchmod(child, directory_mode)
                        # Persist both the initialized directory and the entry
                        # in its parent before either descriptor is released.
                        os.fsync(child)
                        os.fsync(descriptor)
                except Exception:
                    # ``descriptor`` still names the parent until the normal
                    # handoff below. Do not leak the newly opened child when
                    # initialization fails before that handoff.
                    os.close(child)
                    raise
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return target, descriptor, target.name


def _open_directory(path: str | Path) -> tuple[Path, int]:
    target = _absolute(Path(path) / ".directory-sentinel").parent
    _unused, parent_fd, name = _open_parent(target, create=False)
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    return target, descriptor


def _regular_entry(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        result = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode):
        raise UnsafeStatePathError(f"refusing non-regular state file {name!r}")
    if result.st_nlink != 1:
        raise UnsafeStatePathError(f"refusing hard-linked state file {name!r}")
    return result


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while saving state")
        view = view[written:]


def _secure_atomic_publish(
    path: str | Path,
    *,
    mode: int,
    write: Callable[[int], None],
) -> None:
    target, parent_fd, name = _open_parent(path, create=True)
    # The containing state directory is operator-writable. A random filename
    # alone is not sufficient: after discovering it, the operator could
    # unlink/recreate it between our close and rename. Keep the payload in a
    # root-private staging directory reached through a pinned descriptor;
    # renaming that directory entry in the shared parent cannot change what
    # ``src_dir_fd`` names.
    staging_name = f".{name}.{secrets.token_hex(16)}.stage"
    temporary_name = "payload"
    staging_fd: int | None = None
    staging_identity: tuple[int, int] | None = None
    temporary_fd: int | None = None
    published = False
    try:
        previous = _regular_entry(parent_fd, name)
        owner = previous if previous is not None else os.fstat(parent_fd)
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        os.fchmod(staging_fd, 0o700)
        staging_stat = os.fstat(staging_fd)
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=staging_fd,
        )
        os.fchmod(temporary_fd, mode)
        if os.geteuid() == 0:
            os.fchown(temporary_fd, owner.st_uid, owner.st_gid)
        write(temporary_fd)
        os.fsync(temporary_fd)

        payload_stat = os.fstat(temporary_fd)
        payload_entry = os.stat(
            temporary_name,
            dir_fd=staging_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(payload_entry.st_mode)
            or payload_entry.st_nlink != 1
            or (payload_entry.st_dev, payload_entry.st_ino)
            != (payload_stat.st_dev, payload_stat.st_ino)
        ):
            raise UnsafeStatePathError(
                f"secure staging file was substituted while writing {target}"
            )

        # Fail if an attacker inserted a link/special file after the first
        # check. A regular-file swap is safe: replace never follows it.
        _regular_entry(parent_fd, name)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=staging_fd,
            dst_dir_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
    except OSError as exc:
        if isinstance(exc, UnsafeStatePathError):
            raise
        raise OSError(f"could not securely write {target}: {exc}") from exc
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if staging_fd is not None and not published:
            try:
                os.unlink(temporary_name, dir_fd=staging_fd)
            except FileNotFoundError:
                pass
        if staging_fd is not None:
            os.close(staging_fd)
        # The operator can rename this private directory entry in the shared
        # parent, but cannot enter it or substitute its payload. Remove it only
        # when the original name still identifies the directory we created.
        if staging_identity is not None:
            try:
                entry = os.stat(
                    staging_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (entry.st_dev, entry.st_ino) == staging_identity:
                    os.rmdir(staging_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)


def secure_atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    mode: int = 0o600,
) -> None:
    """Atomically replace a regular file without following operator links."""
    _secure_atomic_publish(
        path,
        mode=mode,
        write=lambda descriptor: _write_all(descriptor, data),
    )


def secure_atomic_copy_file(
    source: str | Path,
    destination: str | Path,
    *,
    mode: int = 0o600,
) -> None:
    """Copy a regular source into a securely published destination."""
    target, source_parent_fd, source_name = _open_parent(source, create=False)
    try:
        source_fd = os.open(source_name, _FILE_READ_FLAGS, dir_fd=source_parent_fd)
    finally:
        os.close(source_parent_fd)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise UnsafeStatePathError(
                f"refusing an unsafe atomic-copy source {target}"
            )

        def copy_to(destination_fd: int) -> None:
            while chunk := os.read(source_fd, 1024 * 1024):
                _write_all(destination_fd, chunk)

        _secure_atomic_publish(destination, mode=mode, write=copy_to)
    finally:
        os.close(source_fd)


def secure_atomic_write_text(
    path: str | Path,
    text: str,
    *,
    mode: int = 0o600,
) -> None:
    secure_atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def secure_atomic_write_json(
    path: str | Path,
    value: Any,
    *,
    mode: int = 0o600,
    sort_keys: bool = True,
    indent: int | None = 2,
) -> None:
    secure_atomic_write_text(
        path,
        json.dumps(value, indent=indent, sort_keys=sort_keys) + "\n",
        mode=mode,
    )


def secure_open_text_read(
    path: str | Path,
    *,
    newline: str | None = None,
) -> TextIO:
    """Open one regular, single-link file without following any symlink."""
    target, parent_fd, name = _open_parent(path, create=False)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, _FILE_READ_FLAGS, dir_fd=parent_fd)
        result = os.fstat(descriptor)
        if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
            raise UnsafeStatePathError(f"refusing unsafe state file {target}")
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(parent_fd)
    return os.fdopen(descriptor, "r", encoding="utf-8", newline=newline)


def secure_read_text(path: str | Path) -> str:
    with secure_open_text_read(path) as stream:
        return stream.read()


def secure_open_new_text(
    path: str | Path,
    *,
    mode: int = 0o600,
    newline: str | None = None,
) -> TextIO:
    """Create one new file exclusively without following directory links."""
    target, parent_fd, name = _open_parent(path, create=True)
    descriptor: int | None = None
    created = False
    try:
        owner = os.fstat(parent_fd)
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
            dir_fd=parent_fd,
        )
        created = True
        os.fchmod(descriptor, mode)
        if os.geteuid() == 0:
            os.fchown(descriptor, owner.st_uid, owner.st_gid)
        os.fsync(parent_fd)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                # Preserve the operation that made the exclusive create
                # unusable; cleanup failure is secondary and cannot expose a
                # caller-owned descriptor.
                pass
        raise OSError(f"could not securely create {target}: {exc}") from exc
    finally:
        os.close(parent_fd)
    assert descriptor is not None
    return os.fdopen(descriptor, "w", encoding="utf-8", newline=newline)


def secure_ensure_directory(path: str | Path, *, mode: int = 0o700) -> None:
    """Create/open one directory no-follow and enforce its exact mode/owner."""
    target = _absolute(Path(path) / ".directory-sentinel").parent
    _unused, descriptor, _name = _open_parent(
        target / ".directory-sentinel", create=True
    )
    try:
        os.fchmod(descriptor, mode)
        if os.geteuid() == 0:
            os.fchown(descriptor, 0, 0)
        os.fsync(descriptor)
    except OSError as exc:
        raise OSError(
            f"could not securely initialize directory {target}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def secure_directory_stat(path: str | Path) -> os.stat_result:
    """Stat a directory reached without following any path component."""
    _target, descriptor = _open_directory(path)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def secure_chown_directory(
    path: str | Path,
    uid: int,
    gid: int,
    *,
    mode: int | None = None,
) -> None:
    """Change one directory's owner/mode through a pinned no-follow descriptor."""
    target, descriptor = _open_directory(path)
    try:
        os.fchown(descriptor, uid, gid)
        if mode is not None:
            os.fchmod(descriptor, mode)
    except OSError as exc:
        raise OSError(
            f"could not securely change ownership of {target}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)


def secure_chown_tree(
    path: str | Path,
    uid: int,
    gid: int,
    *,
    directory_mode: int | None = None,
    file_mode: int | None = None,
) -> None:
    """Recursively normalize one tree without following/re-resolving names."""
    target, root_fd = _open_directory(path)

    def visit(directory_fd: int) -> None:
        for name in os.listdir(directory_fd):
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(entry.st_mode):
                child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if (opened.st_dev, opened.st_ino) != (
                        entry.st_dev,
                        entry.st_ino,
                    ):
                        raise UnsafeStatePathError(
                            f"directory entry changed during ownership restore: {name}"
                        )
                    visit(child_fd)
                    os.fchown(child_fd, uid, gid)
                    if directory_mode is not None:
                        os.fchmod(child_fd, directory_mode)
                finally:
                    os.close(child_fd)
                continue
            if stat.S_ISREG(entry.st_mode):
                if entry.st_nlink != 1:
                    raise UnsafeStatePathError(
                        f"refusing hard-linked file during ownership restore: {name}"
                    )
                child_fd = os.open(
                    name,
                    _FILE_READ_FLAGS,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or (opened.st_dev, opened.st_ino)
                        != (entry.st_dev, entry.st_ino)
                    ):
                        raise UnsafeStatePathError(
                            f"file entry changed during ownership restore: {name}"
                        )
                    os.fchown(child_fd, uid, gid)
                    if file_mode is not None:
                        os.fchmod(child_fd, file_mode)
                finally:
                    os.close(child_fd)
                continue
            # Symlinks and special files are never followed or handed to
            # chown(2); their presence makes the restore fail closed.
            raise UnsafeStatePathError(
                f"refusing non-regular entry during ownership restore: {name}"
            )

    try:
        visit(root_fd)
        os.fchown(root_fd, uid, gid)
        if directory_mode is not None:
            os.fchmod(root_fd, directory_mode)
    except OSError as exc:
        if isinstance(exc, UnsafeStatePathError):
            raise
        raise OSError(
            f"could not securely restore ownership of {target}: {exc}"
        ) from exc
    finally:
        os.close(root_fd)


def secure_rmdir(path: str | Path, *, missing_ok: bool = False) -> None:
    """Remove one empty directory through a no-follow parent descriptor."""
    target, parent_fd, name = _open_parent(path, create=False)
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafeStatePathError(f"refusing non-directory state path {target}")
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileNotFoundError:
        if not missing_ok:
            raise
    except OSError as exc:
        if isinstance(exc, UnsafeStatePathError):
            raise
        raise OSError(f"could not securely remove directory {target}: {exc}") from exc
    finally:
        os.close(parent_fd)


def secure_unlink(path: str | Path, *, missing_ok: bool = False) -> None:
    """Unlink an exact directory entry through a no-follow parent descriptor."""
    target, parent_fd, name = _open_parent(path, create=False)
    try:
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileNotFoundError:
        if not missing_ok:
            raise
    except OSError as exc:
        raise OSError(f"could not securely remove {target}: {exc}") from exc
    finally:
        os.close(parent_fd)


def secure_list_names(path: str | Path) -> list[str]:
    """List a directory reached without following any path-component link."""
    target = _absolute(Path(path) / ".listing-sentinel")
    _unused, parent_fd, _name = _open_parent(target, create=False)
    try:
        return os.listdir(parent_fd)
    finally:
        os.close(parent_fd)
