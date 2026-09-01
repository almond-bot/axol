"""User-initiated update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) performs
the first install with ``uv tool install`` from PyPI, pinned to the version of
the latest GitHub release, and runs ``axol serve`` under a systemd service with
``Restart=always``. Subsequent control-panel updates do not fetch that mutable
web endpoint: every SDK wheel embeds the reviewed installer, and the updater
extracts it only after verifying the exact target wheel against strict PyPI
metadata and its SHA-256. This module surfaces whether a newer release exists
and lets the operator apply it on demand:

- :meth:`SelfUpdater.status` answers the polled control-panel indicator. It
  reports the installed version and the highest release tag (resolved by a
  read-only ``git ls-remote --tags`` against the repository, debounced and
  cached), so the UI can show "update available" and a button. Nothing upgrades
  as a side effect of this check. An update is offered only when a release tag
  with a *higher* version than the installed one exists -- commits landing on
  ``main`` between releases are invisible to installs.
- :meth:`SelfUpdater.start` is the Update button. It delegates the selected
  release to the installer embedded in that release's exact PyPI wheel. The
  wheel is selected from strict PyPI metadata and its size and SHA-256 are
  verified before the installer is extracted into a root-only directory and
  launched in a named transient systemd unit. That worker owns the host
  transaction: it stops this service, stages and provisions the exact PyPI
  release, then admits a health-verified candidate.
  Hosts originally installed from GitHub (pre-PyPI releases) migrate to the
  PyPI artifact through that same installer path.

Because the reinstall rebuilds the tool environment, the hosted installer also
owns restoration of out-of-band dependencies and runtime verification. A VIVE
Ultimate runtime is an explicit operator opt-in, so an already-installed exact
pyvut pin is preserved in the transaction and verified before the candidate is
started. An explicitly installed published ``lerobot_robot_axol`` plugin is
likewise carried into the same transaction; direct/custom plugin sources block
because their provenance cannot be reconstructed safely. On aarch64,
PyPI's pinned PyTorch 2.10 wheel is CPU-only; an existing CUDA-enabled or custom
build therefore blocks before the force reinstall rather than being silently
replaced. That comparatively expensive torch probe runs only for an explicit
update, never during the read-only status poll.

The read-only ``git ls-remote --tags`` indicator is deliberately separate from
the *destructive* reinstall: the reinstall rebuilds (and so prunes
pyzed/PyGObject from) the env on every run, so it only runs when the operator
explicitly asks. The cheap ``ls-remote`` can poll freely without touching the
steady-state install.

``axol provision`` also runs once at startup. This legacy self-heal matters for
a host upgraded into the GStreamer-pipeline build by an older release that did
not yet provision it. Current updates run the same idempotent command inside
the hosted transaction.

Dev checkouts (``uv run axol serve`` from a clone) are untouched: the package
metadata then points at a local directory, not an index or git install, and
the updater no-ops.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Callable

from ..cli.update_preflight import release_update_requirements

_logger = logging.getLogger(__name__)

_PACKAGE = "almond-axol"
# The version-pinned reinstall must reproduce the hosted installer's
# requirement (web/app/public/install): same extras, same Python. Keep the two
# in sync.
_EXTRAS = "lerobot,sim,tracker"
_PYTHON_VERSION = "3.13"
# Where release tags live. Index (PyPI) installs carry no repository metadata,
# so the release check falls back to this; git installs keep using their own
# origin URL so forks still see their own releases.
_REPO_URL = "https://github.com/almond-bot/axol"
# New releases use ``release-v0.1.37``. Releases through v0.1.36 used the
# legacy ``v0.1.2`` namespace, which the old destructive updater still polls.
# Never publish another legacy-v tag: keeping all future releases under the
# new namespace prevents an unmigrated old server (or cached old UI tab) from
# invoking that updater. This updater accepts both namespaces so the first
# hardened install can compare itself with the historical releases.
_TAG_RE = re.compile(r"^(?:(?:release-)?v)?(\d+(?:\.\d+)*)$")
_FIRST_HARDENED_RELEASE = (0, 1, 37)
# Minimum seconds between read-only `git ls-remote` checks. The status endpoint
# is polled, so without this every poll would spawn a git process; the check is
# cheap and the indicator does not need to be more current than this.
_REMOTE_DEBOUNCE_S = 60.0
_UPDATE_MONITOR_INTERVAL_S = 1.0
_MANTIS_STABILITY_ATTEMPTS = 20
_MANTIS_STABLE_SAMPLES = 5
_MANTIS_STABILITY_INTERVAL_S = 0.5
# Give well-behaved subprocesses a brief graceful-stop window before shutdown
# escalates to SIGKILL.  The final wait is deliberately unbounded: after kill,
# reaping a root child is part of shutdown ownership, not optional cleanup.
_PROCESS_TERMINATE_TIMEOUT_S = 5.0
_SERVICE_NAME = "axol.service"
_MANTIS_SERVICE_NAME = "axol-mantis.service"
_UPDATE_GUARD_MARKER = Path("/var/lib/almond-axol/update-incomplete")
_UPDATE_START_TOKEN = Path("/var/lib/almond-axol/update-start-once")
_UPDATE_VERIFYING_MARKER = Path("/var/lib/almond-axol/update-verifying")
_UPDATE_GUARD_DROPIN = Path("/etc/systemd/system/axol.service.d/20-update-guard.conf")
_UPDATE_GUARD_CONTENT = (
    "[Unit]\n"
    f"ConditionPathExists=|!{_UPDATE_GUARD_MARKER}\n"
    f"ConditionPathExists=|{_UPDATE_START_TOKEN}\n"
)
_MANTIS_START_DIR = Path("/run/almond-axol")
_MANTIS_START_TOKEN = _MANTIS_START_DIR / "mantis-update-start-once"
_MANTIS_UPDATE_GUARD_CONTENT = (
    "[Unit]\n"
    f"ConditionPathExists=|!{_UPDATE_GUARD_MARKER}\n"
    f"ConditionPathExists=|{_MANTIS_START_TOKEN}\n"
    "\n[Service]\n"
    f"ExecStartPre=/usr/bin/rm -f -- {_MANTIS_START_TOKEN}\n"
)
_MANTIS_UPDATE_GUARD_DROPIN = Path(
    "/etc/systemd/system/axol-mantis.service.d/20-update-guard.conf"
)
_PYPI_RELEASE_URL = "https://pypi.org/pypi/almond-axol/{version}/json"
_PYPI_FILES_HOST = "files.pythonhosted.org"
_WHEEL_INSTALLER_PATH = "almond_axol/_installer.sh"
_UPDATE_WORKER_ROOT = Path("/var/lib/almond-axol/update-workers")
_MAX_PYPI_METADATA_BYTES = 2 * 1024 * 1024
_MAX_RELEASE_WHEEL_BYTES = 16 * 1024 * 1024
_MAX_INSTALLER_BYTES = 512 * 1024
_MAX_WHEEL_METADATA_BYTES = 256 * 1024
_MANAGED_UV_EXECUTABLE = "/usr/local/bin/uv"
_MANAGED_AXOL_EXECUTABLE = "/usr/local/bin/axol"
_MANAGED_SYSTEMCTL_EXECUTABLE = "/usr/bin/systemctl"
_SYSTEMD_RUN_EXECUTABLE = "/usr/bin/systemd-run"
_BASH_EXECUTABLE = "/bin/bash"
_RM_EXECUTABLE = "/usr/bin/rm"
_OPERATOR_USER_ENV = "AXOL_OPERATOR_USER"
_OPERATOR_USER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*[$]?")
_MANAGED_UPDATE_ENV = {
    "AXOL_PRIVILEGED_SERVICE": "1",
    "UV_TOOL_DIR": "/opt/axol/uv/tools",
    "UV_PYTHON_INSTALL_DIR": "/opt/axol/uv/python",
    "UV_TOOL_BIN_DIR": "/usr/local/bin",
}


@dataclass(frozen=True)
class _ReleaseWheel:
    """One immutable, pure-Python release wheel selected from PyPI metadata."""

    filename: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class _StagedRelease:
    """Root-only target wheel and installer handed together to systemd."""

    directory: Path
    wheel: Path
    installer: Path
    sha256: str


def _release_wheel_on_pypi(version: str) -> _ReleaseWheel | None:
    """Select one strictly described target wheel from exact-version metadata."""
    if parse_version(version) is None:
        return None
    request = urllib.request.Request(
        _PYPI_RELEASE_URL.format(version=version),
        headers={"User-Agent": "almond-axol-self-update"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:
                return None
            raw_payload = response.read(_MAX_PYPI_METADATA_BYTES + 1)
            if len(raw_payload) > _MAX_PYPI_METADATA_BYTES:
                return None
            payload = json.loads(raw_payload)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if not isinstance(payload, dict):
        return None
    info = payload.get("info")
    artifacts = payload.get("urls")
    if (
        not isinstance(info, dict)
        or info.get("name") != _PACKAGE
        or info.get("version") != version
        or not isinstance(artifacts, list)
    ):
        return None

    expected_filename = f"almond_axol-{version}-py3-none-any.whl"
    candidates = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and item.get("packagetype") == "bdist_wheel"
        and item.get("yanked") is False
        and item.get("filename") == expected_filename
    ]
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    url = candidate.get("url")
    digest = candidate.get("digests")
    size = candidate.get("size")
    if (
        not isinstance(url, str)
        or not isinstance(digest, dict)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 < size <= _MAX_RELEASE_WHEEL_BYTES
    ):
        return None
    sha256 = digest.get("sha256")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        return None
    try:
        parsed_url = urllib.parse.urlsplit(url)
        hostname = parsed_url.hostname
        port = parsed_url.port
    except ValueError:
        return None
    if (
        parsed_url.scheme != "https"
        or hostname != _PYPI_FILES_HOST
        or port is not None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
        or urllib.parse.unquote(parsed_url.path).rsplit("/", 1)[-1] != expected_filename
    ):
        return None
    return _ReleaseWheel(expected_filename, url, sha256, size)


def _release_available_on_pypi(version: str) -> bool:
    """Whether PyPI exposes the one wheel shape accepted by the updater."""
    return _release_wheel_on_pypi(version) is not None


def _validate_update_worker_root() -> None:
    """Create the staging leaf only below the installer's protected state root."""
    state_root = _UPDATE_WORKER_ROOT.parent
    state_stat = state_root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(state_stat.st_mode)
        or state_stat.st_uid != 0
        or state_stat.st_gid != 0
        or state_stat.st_mode & 0o022
    ):
        raise OSError("unsafe managed update state directory")
    created = False
    try:
        _UPDATE_WORKER_ROOT.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        pass
    if created:
        # axol.service deliberately has the operator as its effective group.
        # Normalize privileged leaves created by that root process before
        # validating them or placing release bytes inside.
        os.chown(_UPDATE_WORKER_ROOT, 0, 0)
        os.chmod(_UPDATE_WORKER_ROOT, 0o700)
    worker_stat = _UPDATE_WORKER_ROOT.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(worker_stat.st_mode)
        or worker_stat.st_uid != 0
        or worker_stat.st_gid != 0
        or worker_stat.st_mode & 0o077
    ):
        raise OSError("unsafe update-worker staging directory")


def _validate_release_stage(stage: Path) -> None:
    stage_stat = stage.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(stage_stat.st_mode)
        or stage_stat.st_uid != 0
        or stage_stat.st_gid != 0
        or stat.S_IMODE(stage_stat.st_mode) != 0o700
    ):
        raise OSError("unsafe release staging directory")


def _validate_release_wheel(wheel: Path, version: str) -> bytes:
    """Read the installer only after validating its wheel identity and shape."""
    expected_metadata = f"almond_axol-{version}.dist-info/METADATA"
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata_names = [
                name for name in names if name.endswith(".dist-info/METADATA")
            ]
            if (
                metadata_names != [expected_metadata]
                or names.count(_WHEEL_INSTALLER_PATH) != 1
            ):
                raise ValueError("release wheel is missing required package metadata")
            metadata_info = archive.getinfo(expected_metadata)
            installer_info = archive.getinfo(_WHEEL_INSTALLER_PATH)
            if not 0 < metadata_info.file_size <= _MAX_WHEEL_METADATA_BYTES:
                raise ValueError("release wheel metadata has an invalid size")
            if not 0 < installer_info.file_size <= _MAX_INSTALLER_BYTES:
                raise ValueError("release installer has an invalid size")
            installer_mode = installer_info.external_attr >> 16
            if stat.S_ISLNK(installer_mode) or (
                stat.S_IFMT(installer_mode) not in {0, stat.S_IFREG}
            ):
                raise ValueError("release installer is not a regular file")
            metadata = BytesParser().parsebytes(archive.read(metadata_info))
            installer = archive.read(installer_info)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ValueError("release wheel could not be validated") from exc
    if metadata.get_all("Name") != [_PACKAGE] or metadata.get_all("Version") != [
        version
    ]:
        raise ValueError("release wheel identity does not match the selected release")
    if not installer.startswith(b"#!/usr/bin/env bash\n") or b"\x00" in installer:
        raise ValueError("release installer is not a valid shell script")
    return installer


def _stage_release(artifact: _ReleaseWheel, version: str) -> _StagedRelease:
    """Download and retain the verified wheel beside its embedded installer."""
    _validate_update_worker_root()
    stage = Path(
        tempfile.mkdtemp(prefix=f"release-{version}-", dir=_UPDATE_WORKER_ROOT)
    )
    wheel_path = stage / artifact.filename
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    wheel_fd = -1
    installer_path: Path | None = None
    try:
        os.chown(stage, 0, 0)
        os.chmod(stage, 0o700)
        _validate_release_stage(stage)
        wheel_fd = os.open(wheel_path, open_flags, 0o600)
        os.fchown(wheel_fd, 0, 0)
        os.fchmod(wheel_fd, 0o600)
        digest = hashlib.sha256()
        downloaded = 0
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": "almond-axol-self-update"},
        )
        wheel_output = os.fdopen(wheel_fd, "wb")
        wheel_fd = -1
        with (
            wheel_output as output,
            urllib.request.urlopen(request, timeout=30) as response,  # noqa: S310
        ):
            if getattr(response, "status", 200) != 200:
                raise OSError("release wheel download failed")
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > artifact.size or downloaded > _MAX_RELEASE_WHEEL_BYTES:
                    raise ValueError("release wheel exceeded its declared size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if downloaded != artifact.size or digest.hexdigest() != artifact.sha256:
            raise ValueError("release wheel failed size or SHA-256 verification")

        installer = _validate_release_wheel(wheel_path, version)
        installer_path = stage / "install"
        installer_fd = os.open(installer_path, open_flags, 0o700)
        try:
            os.fchown(installer_fd, 0, 0)
            os.fchmod(installer_fd, 0o700)
            installer_output = os.fdopen(installer_fd, "wb")
            installer_fd = -1
            with installer_output as output:
                output.write(installer)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if installer_fd >= 0:
                os.close(installer_fd)
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        worker_root_fd = os.open(_UPDATE_WORKER_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(worker_root_fd)
        finally:
            os.close(worker_root_fd)
        return _StagedRelease(stage, wheel_path, installer_path, artifact.sha256)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if wheel_fd >= 0:
            os.close(wheel_fd)


def _write_durable_root_file(path: Path, content: str, *, mode: int) -> None:
    """Atomically write a root-owned guard file below a protected directory."""
    parent = path.parent
    created = False
    try:
        # Every current caller's fixed system parent already exists. Avoid
        # parents=True so a malformed host cannot make us create and trust a
        # whole unchecked ancestry in one operation.
        parent.mkdir(mode=0o755)
        created = True
    except FileExistsError:
        pass
    if created:
        os.chown(parent, 0, 0)
        os.chmod(parent, 0o755)
    parent_stat = parent.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != 0
        or parent_stat.st_gid != 0
        or parent_stat.st_mode & 0o022
    ):
        raise OSError(f"unsafe update-guard directory: {parent}")
    if path.is_symlink():
        raise OSError(f"update-guard path must not be a symlink: {path}")

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary)
    try:
        os.fchown(fd, 0, 0)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if created:
            ancestor_fd = os.open(parent.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(ancestor_fd)
            finally:
                os.close(ancestor_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary_path.unlink(missing_ok=True)


def _remove_durable_file(path: Path) -> None:
    """Remove a guard marker and durably commit the directory entry change."""
    path.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _remove_staged_release(path: Path) -> None:
    """Delete only a root-owned release directory created by this updater."""
    if path.parent != _UPDATE_WORKER_ROOT or not path.name.startswith("release-"):
        raise OSError("refusing to remove a path outside release staging")
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
    ):
        raise OSError("refusing to remove an unsafe release staging path")
    shutil.rmtree(path)
    directory_fd = os.open(_UPDATE_WORKER_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_mantis_start_token(version: str) -> None:
    """Authorize exactly one Mantis start while the durable guard is armed."""
    try:
        _MANTIS_START_DIR.mkdir(mode=0o700)
        os.chown(_MANTIS_START_DIR, 0, 0)
        os.chmod(_MANTIS_START_DIR, 0o700)
    except FileExistsError:
        pass
    metadata = _MANTIS_START_DIR.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise OSError("unsafe Mantis update-start directory")
    _write_durable_root_file(
        _MANTIS_START_TOKEN,
        f"target-version={version}\n",
        mode=0o600,
    )


def parse_version(text: str) -> tuple[int, ...] | None:
    """Parse plain, historical ``v``, or hardened ``release-v`` versions."""
    match = _TAG_RE.fullmatch(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def installed_origin() -> tuple[str, str] | None:
    """``(git url, commit id)`` for a git tool install.

    Read from PEP 610 ``direct_url.json``. Returns ``None`` for dev checkouts
    (directory installs) or when the metadata is missing. The url is where the
    updater looks for release tags; the commit id is what is currently
    installed.
    """
    try:
        dist = distribution(_PACKAGE)
    except PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    vcs = data.get("vcs_info") or {}
    commit = vcs.get("commit_id")
    url = data.get("url")
    if not commit or not url:
        return None
    # PEP 610 stores the plain repository URL, but strip a `git+` pip-scheme
    # prefix defensively so `git ls-remote` gets a clean URL.
    if url.startswith("git+"):
        url = url[len("git+") :]
    return url, commit


def installed_from_index() -> bool:
    """Whether this is a regular index (PyPI) install of the package.

    PEP 610: index installs carry **no** ``direct_url.json`` at all, while git
    installs record ``vcs_info`` and dev checkouts (editable / directory
    installs) record ``dir_info`` -- so "metadata exists but no dist at all"
    and "metadata present" both mean not-an-index-install.
    """
    try:
        dist = distribution(_PACKAGE)
    except PackageNotFoundError:
        return False
    return dist.read_text("direct_url.json") is None


def _git(repo_root: Path, *args: str) -> bytes | None:
    """Raw stdout of a git command in ``repo_root``; ``None`` on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def installed_commit() -> str | None:
    """The git commit this backend is running, or ``None`` when unknown.

    For a git tool install it is the PEP 610 pinned commit; for a dev checkout
    it is the checkout's HEAD, with a ``-dirty.<hash>`` suffix over any
    uncommitted changes so two different working-tree states never share an
    identity. The web bundle bakes its own build commit in at build time
    (``buildCommit()`` in web/app/vite.config.ts — the dirty-hash scheme must
    stay identical), so the control panel can compare the two and warn when
    the UI and the backend are on different code. Works on forks too — it
    never references the upstream repository.
    """
    origin = installed_origin()
    if origin is not None:
        return origin[1]
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None
    head = _git(repo_root, "rev-parse", "HEAD")
    commit = head.decode("utf-8", "replace").strip() if head else ""
    if not commit:
        return None
    status = _git(repo_root, "status", "--porcelain")
    if status is None or not status.strip():
        return commit
    diff = _git(repo_root, "diff", "HEAD") or b""
    digest = hashlib.sha256(status + diff).hexdigest()[:8]
    return f"{commit}-dirty.{digest}"


def installed_version() -> str | None:
    """Installed release version (the pyproject ``version``), e.g. ``"0.1.2"``.

    ``None`` only when the package metadata is missing entirely.
    """
    try:
        return distribution(_PACKAGE).version
    except PackageNotFoundError:
        return None


class SelfUpdater:
    """Read-only release indicator + explicit, user-initiated upgrade.

    The control panel polls :meth:`status` (which reports the installed version
    and whether a newer release tag exists) and triggers :meth:`start` from an
    Update button. Nothing upgrades automatically.

    ``is_idle`` reports whether it is safe to hand host ownership to the update
    worker (no operation running; a connected robot is fine). Once systemd
    accepts that transient unit, the worker survives this service being stopped.
    """

    def __init__(self, is_idle: Callable[[], bool]) -> None:
        self._is_idle = is_idle
        self._origin = installed_origin()
        self._version = installed_version()
        self._commit = installed_commit()
        # Cached newest release (tag + parsed-out version) and when it was last
        # resolved, so the polled status endpoint answers immediately and only
        # re-runs `git ls-remote --tags` at most once per debounce window
        # rather than on every poll.
        self._remote_tag: str | None = None
        self._remote_version: str | None = None
        self._remote_checked_at = 0.0
        self._remote_task: asyncio.Task[None] | None = None
        self._owned_processes: set[asyncio.subprocess.Process] = set()
        self._process_owner_tasks: set[asyncio.Task[object]] = set()
        self._shutting_down = False
        self._shutdown_lock = asyncio.Lock()
        self._shutdown_task: asyncio.Task[None] | None = None
        # Update lifecycle surfaced to the UI: "idle" | "updating" | "error".
        self._state = "idle"
        self._error: str | None = None
        # Current step while ``state == "updating"`` so the UI can show progress
        # instead of an opaque spinner: "upgrading" | "provisioning" |
        # "restarting" (``None`` when not updating).
        self._phase: str | None = None
        self._update_task: asyncio.Task[None] | None = None
        # A systemd-run wrapper remains our child until it confirms that the
        # named transient unit was accepted.  Only then does ownership transfer
        # to systemd (the worker must survive axol.service being stopped).
        self._update_worker_unit: str | None = None
        self._update_worker_stage: Path | None = None
        self._update_worker_handed_off = False
        self._update_monitor_task: asyncio.Task[None] | None = None
        self._update_monitor_stop = asyncio.Event()
        # A force reinstall mutates the environment that every subsequently
        # spawned Axol command imports.  Once maintenance starts, hardware and
        # session launches stay fail-closed until every install/provision/
        # verification step succeeds and this process restarts.  On failure
        # the old process remains alive for status/retry/reboot, but may not
        # launch code from a potentially partial environment.
        marker_present = (
            _UPDATE_GUARD_MARKER.exists() or _UPDATE_GUARD_MARKER.is_symlink()
        )
        verifying_present = (
            _UPDATE_VERIFYING_MARKER.exists() or _UPDATE_VERIFYING_MARKER.is_symlink()
        )
        token_present = _UPDATE_START_TOKEN.exists() or _UPDATE_START_TOKEN.is_symlink()
        self._launches_blocked = marker_present or verifying_present or token_present
        # A retry is allowed while a prior transaction's durable barrier is
        # present so the UI can repair the host. Keep ownership of that older
        # barrier distinct from the temporary one raised by the new attempt:
        # an early, non-mutating retry failure must never revive this process.
        self._inherited_launch_barrier = self._launches_blocked
        # ExecStartPre moves the one-shot token to ``update-verifying`` before
        # importing this candidate. ExecStartPost removes that attempt marker
        # after Axol's health dwell; the installer removes the durable marker
        # only after Mantis is stable too. Only that exact candidate state may
        # clear its in-memory barrier without a process restart; deleting a
        # generic failed-update marker never revives an old unsafe process.
        self._candidate_verification = marker_present and verifying_present
        if self._launches_blocked:
            self._state = "error"
            self._error = (
                "a previous update did not complete verification; retry the "
                "update or repair the service"
            )
        # The optional Quest bootstrap helper imports from the same uv tool
        # environment.  An update must stop it before replacing that runtime,
        # then restore its prior enabled/active intent only after verification.
        # ``None`` means the next guard arm must snapshot systemd; retaining a
        # bool across a failed attempt preserves a disabled-but-manually-active
        # helper while this server process remains alive.
        self._mantis_restore_requested: bool | None = None
        self._mantis_enable_requested: bool | None = None
        # The GStreamer camera stack is provisioned once per process (covers a
        # host that upgraded into this build from an older release). ``_env_lock``
        # serializes everything that mutates the uv tool environment -- the
        # startup heal and any runtime verification sharing that environment.
        self._provision_started = False
        self._provision_task: asyncio.Task[None] | None = None
        self._provision_launch_lock = asyncio.Lock()
        self._env_lock = asyncio.Lock()

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def commit(self) -> str | None:
        """Git commit of the running backend (see :func:`installed_commit`)."""
        return self._commit

    @property
    def release_install(self) -> bool:
        """Whether this backend is a release install (PyPI or tag-pinned git).

        Release installs only ever sit on released versions, so the control
        panel compares *versions* against them (a hosted UI built from main
        legitimately differs in commit between releases). Dev checkouts can be
        on any commit, so the panel compares commits directly.
        """
        return self._origin is not None or installed_from_index()

    @property
    def enabled(self) -> bool:
        """Updatable only for release installs with uv available."""
        return self.release_install and shutil.which("uv") is not None

    @property
    def maintenance_active(self) -> bool:
        """Whether an install/provision task is actively mutating the host."""
        update_active = self._state == "updating" or (
            self._update_task is not None and not self._update_task.done()
        )
        provision_active = (
            self._provision_task is not None and not self._provision_task.done()
        )
        return update_active or provision_active

    @staticmethod
    def _signal_process(
        proc: asyncio.subprocess.Process, *, force: bool = False
    ) -> None:
        """Stop an owned child, tolerating a concurrent natural exit."""
        if proc.returncode is not None:
            return
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

    async def _communicate_owned(
        self, *args: str, **kwargs: object
    ) -> tuple[asyncio.subprocess.Process, bytes, bytes | None]:
        """Spawn, communicate with, and account for one subprocess child."""
        if self._shutting_down:
            raise OSError("self-updater is shutting down")
        owner = asyncio.current_task()
        if owner is not None:
            self._process_owner_tasks.add(owner)
        try:
            proc = await asyncio.create_subprocess_exec(*args, **kwargs)
            self._owned_processes.add(proc)
            if self._shutting_down:
                self._signal_process(proc)
            try:
                out, err = await proc.communicate()
            except BaseException:
                # Task cancellation must not discard the only handle to a root
                # child. Reap it here even outside the application drain.
                self._signal_process(proc)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(proc.wait()),
                        timeout=_PROCESS_TERMINATE_TIMEOUT_S,
                    )
                except TimeoutError:
                    self._signal_process(proc, force=True)
                    await asyncio.shield(proc.wait())
                raise
            finally:
                self._owned_processes.discard(proc)
            return proc, out, err
        finally:
            if owner is not None:
                self._process_owner_tasks.discard(owner)

    @staticmethod
    def _unit_is_stopped(result: subprocess.CompletedProcess[str] | None) -> bool:
        """Require positive systemd state, never an inspection error."""
        if result is None or not isinstance(result.stdout, str):
            return False
        fields: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key] = value
        if fields.get("LoadState") == "not-found":
            return True
        return result.returncode == 0 and fields.get("ActiveState") in {
            "failed",
            "inactive",
        }

    async def _stop_uncommitted_update_worker(self) -> bool:
        """Stop an uncertain worker and report only positively proven cleanup."""
        if self._update_worker_handed_off:
            return True
        unit = self._update_worker_unit
        stage = self._update_worker_stage
        self._update_worker_stage = None
        if unit is None:
            if stage is not None:
                try:
                    _remove_staged_release(stage)
                except OSError:
                    _logger.warning("self-update: could not clean unused release stage")
            return True
        # Claim cleanup before yielding so concurrent failure/shutdown paths do
        # not issue competing stop requests for the same transient unit.
        self._update_worker_unit = None
        try:
            await asyncio.to_thread(self._systemctl, "stop", unit)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            status = await asyncio.to_thread(
                self._systemctl,
                "show",
                "--property=LoadState",
                "--property=ActiveState",
                unit,
            )
        except (OSError, subprocess.SubprocessError):
            status = None
        stopped = self._unit_is_stopped(status)
        if not stopped:
            # A wrapper can fail or be interrupted after asking PID 1 to create
            # the unit. Best-effort SIGKILL closes that uncertain-launch window,
            # but only a subsequent state query proves ownership is gone.
            try:
                await asyncio.to_thread(
                    self._systemctl,
                    "kill",
                    "--kill-who=all",
                    "--signal=SIGKILL",
                    unit,
                )
                status = await asyncio.to_thread(
                    self._systemctl,
                    "show",
                    "--property=LoadState",
                    "--property=ActiveState",
                    unit,
                )
            except (OSError, subprocess.SubprocessError):
                status = None
            stopped = self._unit_is_stopped(status)
        if not stopped:
            # The unit may still need both staged files. Retain them and keep
            # every hardware launch blocked until an operator repairs/retries.
            self._update_worker_unit = unit
            self._update_worker_stage = stage
            _logger.warning(
                "self-update: could not prove cleanup of uncommitted %s", unit
            )
            return False
        if stage is not None:
            try:
                _remove_staged_release(stage)
            except OSError:
                _logger.warning("self-update: could not clean stopped worker stage")
        return True

    async def _monitor_update_worker(self, unit: str) -> None:
        """Surface an early hosted-worker exit while this server remains alive."""
        while True:
            try:
                await asyncio.wait_for(
                    self._update_monitor_stop.wait(),
                    timeout=_UPDATE_MONITOR_INTERVAL_S,
                )
                return
            except TimeoutError:
                pass
            try:
                status = await asyncio.to_thread(
                    self._systemctl,
                    "show",
                    "--property=LoadState",
                    "--property=ActiveState",
                    "--property=SubState",
                    unit,
                )
            except (OSError, subprocess.SubprocessError):
                # Unknown D-Bus state is not evidence that the root worker has
                # stopped. Keep monitoring with the launch barrier intact.
                continue
            if self._shutting_down:
                return
            fields: dict[str, str] = {}
            if status is not None:
                for line in status.stdout.splitlines():
                    key, separator, value = line.partition("=")
                    if separator:
                        fields[key] = value
            terminal = fields.get("LoadState") == "not-found" or (
                status.returncode == 0
                and fields.get("ActiveState") in {"failed", "inactive"}
            )
            if not terminal:
                continue
            self._update_worker_handed_off = False
            self._update_worker_unit = None
            # Recovery of an interrupted, already-healthy candidate finishes in
            # place: the installer intentionally exits without force-reinstalling
            # or restarting this process. Its durable marker unlink is the exact
            # candidate commit this process has tracked since ExecStartPre.
            self._refresh_candidate_commit()
            if not self._launches_blocked:
                self._state = "idle"
                self._error = None
                self._phase = None
                return
            self._fail(
                "the hosted update worker stopped before replacing the server; "
                "retry the update or inspect the system journal",
                launches_unsafe=True,
            )
            return

    async def _drain_shutdown(self) -> None:
        """Implementation of the single shared shutdown drain."""
        tasks = tuple(
            {
                task
                for task in (
                    self._remote_task,
                    self._update_task,
                    self._provision_task,
                    self._update_monitor_task,
                    *self._process_owner_tasks,
                )
                if task is not None
                and not task.done()
                and task is not asyncio.current_task()
            }
        )
        processes = tuple(self._owned_processes)
        for proc in processes:
            self._signal_process(proc)
        process_waiters = tuple(asyncio.create_task(proc.wait()) for proc in processes)
        drain_waiters = (*tasks, *process_waiters)
        if drain_waiters:
            _, pending = await asyncio.wait(
                drain_waiters, timeout=_PROCESS_TERMINATE_TIMEOUT_S
            )
            if pending:
                for proc in tuple(self._owned_processes):
                    self._signal_process(proc, force=True)
                await asyncio.gather(*pending, return_exceptions=True)
        await self._stop_uncommitted_update_worker()

    async def shutdown(self) -> None:
        """Stop and reap locally owned work before the API process exits.

        A successfully accepted transient update unit is intentionally not
        stopped: systemd owns it and it is responsible for replacing this
        service. Any unconfirmed wrapper/unit and every provision/refresh child
        remain ours and are drained exactly once.
        """
        async with self._shutdown_lock:
            self._shutting_down = True
            self._update_monitor_stop.set()
            if self._shutdown_task is None:
                self._shutdown_task = asyncio.create_task(self._drain_shutdown())
            drain = self._shutdown_task
        try:
            await asyncio.shield(drain)
        except asyncio.CancelledError:
            # Uvicorn/systemd cancellation must not abandon a root child or an
            # unconfirmed transient unit midway through ownership transfer.
            await drain
            raise

    @property
    def launches_blocked(self) -> bool:
        """Whether starting hardware or a new session is currently unsafe."""
        self._refresh_candidate_commit()
        return self._launches_blocked or self.maintenance_active

    def _refresh_candidate_commit(self) -> None:
        """Observe ExecStartPost's durable commit for this exact candidate."""
        if not self._candidate_verification:
            return
        transaction_state_remains = any(
            path.exists() or path.is_symlink()
            for path in (_UPDATE_GUARD_MARKER, _UPDATE_VERIFYING_MARKER)
        )
        if transaction_state_remains:
            return
        self._candidate_verification = False
        self._inherited_launch_barrier = False
        self._launches_blocked = False
        if self._state == "error" and self._error is not None:
            self._state = "idle"
            self._error = None

    def _interrupted_candidate_repair_available(self) -> bool:
        """Whether this exact healthy candidate may retry its interrupted commit.

        The normal updater only permits a strictly newer release. A candidate
        whose Axol health check already consumed ``update-verifying`` can instead
        need the same release installer to finish Mantis restore, marker removal,
        and rollback promotion. Keep that exception tied to the exact process
        that started with both transaction markers and to its owned marker; a
        generic failed-update marker must never authorize a same-version run.
        """
        if (
            not self._candidate_verification
            or not self._inherited_launch_barrier
            or self._version is None
            or self._remote_version != self._version
            or _UPDATE_START_TOKEN.exists()
            or _UPDATE_START_TOKEN.is_symlink()
            or _UPDATE_VERIFYING_MARKER.exists()
            or _UPDATE_VERIFYING_MARKER.is_symlink()
        ):
            return False

        expected = f"target-version={self._version}\n".encode("ascii")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(_UPDATE_GUARD_MARKER, flags)
        except OSError:
            return False
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                return False
            return os.read(fd, len(expected) + 1) == expected
        except OSError:
            return False
        finally:
            os.close(fd)

    def launch_block_reason(self) -> str | None:
        """A fixed, credential-safe launch rejection for the serve API."""
        if not self.launches_blocked:
            return None
        if self.maintenance_active:
            return (
                "server maintenance is in progress — wait for it to finish "
                "before starting hardware or a session"
            )
        return (
            "server maintenance did not complete safely — retry the update or "
            "restart/repair the service before starting hardware or a session"
        )

    @staticmethod
    def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [_MANAGED_SYSTEMCTL_EXECUTABLE, *args],
            capture_output=True,
            text=True,
            timeout=15,
        )

    @staticmethod
    def _release_wheel(version: str) -> _ReleaseWheel | None:
        return _release_wheel_on_pypi(version)

    @staticmethod
    def _stage_release(artifact: _ReleaseWheel, version: str) -> _StagedRelease:
        return _stage_release(artifact, version)

    def _managed_service_error(self) -> str | None:
        """Validate that this process is the hosted root service."""
        required_executables = (
            _MANAGED_SYSTEMCTL_EXECUTABLE,
            _MANAGED_UV_EXECUTABLE,
            _MANAGED_AXOL_EXECUTABLE,
        )
        if os.geteuid() != 0 or any(
            not Path(executable).is_file() for executable in required_executables
        ):
            return "self-update requires the managed root axol.service"
        mismatched_layout = [
            name
            for name, expected in _MANAGED_UPDATE_ENV.items()
            if os.environ.get(name) != expected
        ]
        if mismatched_layout:
            return (
                "self-update requires the hosted installer layout; rerun the "
                "hosted installer to repair " + ", ".join(mismatched_layout)
            )
        try:
            active = self._systemctl("is-active", "--quiet", _SERVICE_NAME)
            main_pid = self._systemctl(
                "show", "--property=MainPID", "--value", _SERVICE_NAME
            )
        except (OSError, subprocess.SubprocessError):
            return "could not verify the managed axol.service before updating"
        if active.returncode != 0 or main_pid.returncode != 0:
            return "self-update requires the active managed axol.service"
        try:
            service_pid = int(main_pid.stdout.strip())
        except ValueError:
            return "could not verify the managed axol.service process"
        if service_pid != os.getpid():
            return "self-update must run from the managed axol.service process"
        return None

    @staticmethod
    def _operator_user_for_update() -> tuple[str | None, str | None]:
        """Return the installer identity persisted by the managed service.

        The transient worker does not inherit this process's environment unless
        it is passed explicitly with ``systemd-run --setenv``.  Refuse a missing
        or stale account rather than letting the root installer guess the first
        home directory and silently move customer state to another login.
        """
        value = os.environ.get(_OPERATOR_USER_ENV)
        if value == "root":
            return value, None
        if value is None or _OPERATOR_USER_RE.fullmatch(value) is None:
            return None, (
                "self-update requires the operator identity persisted by the "
                "hosted installer; rerun the hosted installer to repair it"
            )
        try:
            account = pwd.getpwnam(value)
        except KeyError:
            return None, (
                "the hosted installer operator account no longer exists; rerun "
                "the installer from the intended operator login"
            )
        if account.pw_uid == 0 or account.pw_name != value:
            return None, "the hosted installer operator identity is invalid"
        return value, None

    def _arm_durable_update_guard(self) -> str | None:
        """Prevent a partial new environment from starting after crash/reboot.

        The currently running service must stay alive to finish the update and
        report failures, so this deliberately disables without stopping it. A
        permanent systemd condition observes the marker before every future
        start, including ``Restart=always`` restarts of this same process.
        """
        managed_error = self._managed_service_error()
        if managed_error is not None:
            return managed_error
        try:
            mantis_load_state = self._systemctl(
                "show",
                "--property=LoadState",
                "--value",
                _MANTIS_SERVICE_NAME,
            )
        except (OSError, subprocess.SubprocessError):
            return "could not verify the managed axol.service before updating"
        if mantis_load_state.returncode != 0 or not mantis_load_state.stdout.strip():
            return "could not inspect the optional axol-mantis.service before updating"
        mantis_installed = mantis_load_state.stdout.strip() != "not-found"

        if self._mantis_restore_requested is None:
            if mantis_installed:
                try:
                    mantis_enabled = self._systemctl(
                        "is-enabled", "--quiet", _MANTIS_SERVICE_NAME
                    )
                    mantis_active = self._systemctl(
                        "is-active", "--quiet", _MANTIS_SERVICE_NAME
                    )
                except (OSError, subprocess.SubprocessError):
                    return (
                        "could not inspect the optional axol-mantis.service state "
                        "before updating"
                    )
                self._mantis_enable_requested = mantis_enabled.returncode == 0
                self._mantis_restore_requested = (
                    self._mantis_enable_requested or mantis_active.returncode == 0
                )
            else:
                self._mantis_enable_requested = False
                self._mantis_restore_requested = False

        try:
            _write_durable_root_file(
                _UPDATE_GUARD_DROPIN,
                _UPDATE_GUARD_CONTENT,
                mode=0o644,
            )
            # Write this even before the optional unit exists. If it is later
            # installed while a failed update marker remains, its first start
            # is protected too.
            _write_durable_root_file(
                _MANTIS_UPDATE_GUARD_DROPIN,
                _MANTIS_UPDATE_GUARD_CONTENT,
                mode=0o644,
            )
            reload_result = self._systemctl("daemon-reload")
            if reload_result.returncode != 0:
                return "could not activate the durable update restart guard"
            guarded_services = [(_SERVICE_NAME, _UPDATE_GUARD_DROPIN)]
            if mantis_installed:
                guarded_services.append(
                    (_MANTIS_SERVICE_NAME, _MANTIS_UPDATE_GUARD_DROPIN)
                )
            for service_name, dropin in guarded_services:
                loaded = self._systemctl(
                    "show", "--property=DropInPaths", "--value", service_name
                )
                if loaded.returncode != 0 or str(dropin) not in loaded.stdout:
                    return "could not verify the durable update restart guard"

            # The conditions are live before the marker appears. A crash before
            # this write can safely restart the untouched environment; every
            # crash after it is condition-blocked.
            for stale_path in (
                _UPDATE_START_TOKEN,
                _UPDATE_VERIFYING_MARKER,
                _MANTIS_START_TOKEN,
            ):
                try:
                    _remove_durable_file(stale_path)
                except FileNotFoundError:
                    pass
            _write_durable_root_file(
                _UPDATE_GUARD_MARKER,
                f"target-version={self._version or '0'}\n",
                mode=0o600,
            )
            if mantis_installed:
                stopped = self._systemctl("stop", _MANTIS_SERVICE_NAME)
                if stopped.returncode != 0:
                    return (
                        "could not stop axol-mantis.service before updating its "
                        "shared runtime"
                    )
                still_active = self._systemctl(
                    "is-active", "--quiet", _MANTIS_SERVICE_NAME
                )
                if still_active.returncode == 0:
                    return (
                        "axol-mantis.service remained active; the shared runtime "
                        "was not changed"
                    )
            disabled = self._systemctl("disable", _SERVICE_NAME)
            if disabled.returncode != 0:
                return "could not disable axol.service for the guarded update"
        except (OSError, subprocess.SubprocessError):
            return "could not establish the durable update restart guard"
        return None

    def _mantis_service_is_stable(self) -> bool:
        """Require one active Mantis MainPID across a short startup dwell."""
        stable_pid: str | None = None
        stable_samples = 0
        for _attempt in range(_MANTIS_STABILITY_ATTEMPTS):
            try:
                pid_result = self._systemctl(
                    "show",
                    "--property=MainPID",
                    "--value",
                    _MANTIS_SERVICE_NAME,
                )
                active = self._systemctl("is-active", "--quiet", _MANTIS_SERVICE_NAME)
            except (OSError, subprocess.SubprocessError):
                return False
            pid = pid_result.stdout.strip()
            if (
                pid_result.returncode == 0
                and active.returncode == 0
                and pid.isascii()
                and pid.isdecimal()
                and int(pid) > 0
            ):
                if pid == stable_pid:
                    stable_samples += 1
                else:
                    stable_pid = pid
                    stable_samples = 1
                if stable_samples >= _MANTIS_STABLE_SAMPLES:
                    return True
            else:
                stable_pid = None
                stable_samples = 0
            time.sleep(_MANTIS_STABILITY_INTERVAL_S)
        return False

    def _contain_failed_shared_commit(self, *, stop_mantis: bool) -> list[str]:
        """Re-arm the guard and positively contain services after commit failure."""
        recovery_errors: list[str] = []
        try:
            _write_durable_root_file(
                _UPDATE_GUARD_MARKER,
                f"target-version={self._version or '0'}\n",
                mode=0o600,
            )
        except OSError:
            recovery_errors.append("the durable marker could not be restored")
        try:
            _remove_durable_file(_MANTIS_START_TOKEN)
        except FileNotFoundError:
            pass
        except OSError:
            recovery_errors.append("the Mantis one-shot token could not be cleared")

        if stop_mantis:
            try:
                self._systemctl("stop", _MANTIS_SERVICE_NAME)
                status = self._systemctl(
                    "show",
                    "--property=LoadState",
                    "--property=ActiveState",
                    _MANTIS_SERVICE_NAME,
                )
                if not self._unit_is_stopped(status):
                    recovery_errors.append(
                        "axol-mantis.service could not be proven stopped"
                    )
            except (OSError, subprocess.SubprocessError):
                recovery_errors.append("axol-mantis.service stop could not be verified")

        try:
            self._systemctl("disable", _SERVICE_NAME)
            unit_state = self._systemctl(
                "show", "--property=UnitFileState", "--value", _SERVICE_NAME
            )
            if unit_state.returncode != 0 or unit_state.stdout.strip() != "disabled":
                recovery_errors.append("axol.service could not be proven disabled")
        except (OSError, subprocess.SubprocessError):
            recovery_errors.append("axol.service disable could not be verified")
        return recovery_errors

    def _disarm_durable_update_guard(self) -> str | None:
        """Re-enable verified code and restore the optional bootstrap helper."""
        if not _UPDATE_GUARD_MARKER.is_file():
            return "the durable update restart guard disappeared during verification"
        try:
            # Keep the marker in place while enabling. If this process crashes
            # at any earlier boundary, the permanent condition still refuses
            # the service start. Only verified code reaches the final unlink.
            enabled = self._systemctl("enable", _SERVICE_NAME)
            if enabled.returncode != 0:
                return (
                    "updated Axol was verified, but axol.service could not be enabled"
                )
            is_enabled = self._systemctl("is-enabled", "--quiet", _SERVICE_NAME)
            if is_enabled.returncode != 0:
                return "updated Axol was verified, but axol.service is not enabled"
            if self._mantis_enable_requested:
                mantis_enabled = self._systemctl("enable", _MANTIS_SERVICE_NAME)
                if mantis_enabled.returncode != 0:
                    return (
                        "updated Axol was verified, but axol-mantis.service could "
                        "not be enabled"
                    )
                mantis_is_enabled = self._systemctl(
                    "is-enabled", "--quiet", _MANTIS_SERVICE_NAME
                )
                if mantis_is_enabled.returncode != 0:
                    return (
                        "updated Axol was verified, but axol-mantis.service is not "
                        "enabled"
                    )
        except (OSError, subprocess.SubprocessError):
            return "could not safely re-enable the verified axol.service"

        if self._mantis_restore_requested:
            restore_error: str | None = None
            try:
                _write_mantis_start_token(self._version or "0")
                started = self._systemctl("start", _MANTIS_SERVICE_NAME)
                if started.returncode != 0:
                    restore_error = "axol-mantis.service could not be started"
                elif _MANTIS_START_TOKEN.exists() or _MANTIS_START_TOKEN.is_symlink():
                    restore_error = (
                        "axol-mantis.service did not consume its one-shot start token"
                    )
                elif not self._mantis_service_is_stable():
                    restore_error = (
                        "axol-mantis.service did not remain active on one stable PID"
                    )
            except (OSError, subprocess.SubprocessError):
                restore_error = "axol-mantis.service could not be verified"

            if restore_error is not None:
                recovery_errors = self._contain_failed_shared_commit(stop_mantis=True)
                detail = (
                    f"; recovery incomplete: {', '.join(recovery_errors)}"
                    if recovery_errors
                    else "; services remain update-guarded"
                )
                return f"updated Axol was verified, but {restore_error}{detail}"

        # The main service remains launch-blocked throughout Mantis startup.
        # Commit the shared transaction only after every required service is
        # stable; a crash before this unlink blocks both services on reboot.
        try:
            _remove_durable_file(_UPDATE_GUARD_MARKER)
        except OSError:
            recovery_errors = self._contain_failed_shared_commit(
                stop_mantis=bool(self._mantis_restore_requested)
            )
            detail = (
                f"; recovery incomplete: {', '.join(recovery_errors)}"
                if recovery_errors
                else "; services remain update-guarded"
            )
            return (
                "could not durably commit the verified Axol and Mantis services"
                f"{detail}"
            )

        self._mantis_restore_requested = None
        self._mantis_enable_requested = None
        return None

    async def ensure_provisioned(self) -> None:
        """Run the once-per-process ``axol provision`` startup heal (see below)."""
        await self._ensure_provision_once()

    def _update_available(self) -> bool:
        """A release tag with a strictly higher version than the install exists."""
        if not self.enabled or self._version is None or self._remote_version is None:
            return False
        current = parse_version(self._version)
        latest = parse_version(self._remote_version)
        return current is not None and latest is not None and latest > current

    async def status(self, *, force: bool = False) -> dict[str, object]:
        """Snapshot for the control panel.

        With ``force`` (a fresh page load / explicit check), resolve the newest
        release tag synchronously -- bypassing the debounce -- so the response
        reflects reality immediately rather than a cached value up to a debounce
        window stale. Otherwise schedule a debounced background refresh and
        return the cached release (``None`` until the first ``git ls-remote``
        resolves), which keeps the steady-state poll cheap.

        Reads ``is_idle`` live so the UI can gate the Update button on a server
        that is safe to hand to the hosted update transaction.
        """
        if force:
            # Await an in-flight background check rather than racing a second
            # ls-remote against it; otherwise resolve now.
            if self._remote_task is not None and not self._remote_task.done():
                await self._remote_task
            else:
                await self.refresh_remote()
        else:
            self._schedule_remote_refresh()
        normal_update_available = self._update_available()
        candidate_repair_available = self._interrupted_candidate_repair_available()
        launches_blocked = self.launches_blocked
        repair_can_take_ownership = (
            candidate_repair_available and not self.maintenance_active
        )
        return {
            "enabled": self.enabled,
            "version": self._version,
            "remoteVersion": self._remote_version,
            "updateAvailable": normal_update_available or candidate_repair_available,
            "idle": self._is_idle()
            and (not launches_blocked or repair_can_take_ownership),
            "state": self._state,
            "phase": self._phase,
            "error": self._error,
            "maintenanceActive": self.maintenance_active,
        }

    def start(self) -> tuple[bool, str | None]:
        """Begin a user-initiated upgrade; returns ``(started, reason)``.

        Refuses (``started=False`` with a human-readable reason) for a dev
        checkout, when no newer release is known, when an update is already
        running, or when an operation is running. The UI disables the button in
        those cases, but guard here too. On success a named transient systemd
        unit runs the hosted update transaction independently of this process.
        """
        if self._shutting_down:
            return False, "server is shutting down"
        if not self.enabled:
            return False, "not a release install"
        if self.maintenance_active:
            return False, "server maintenance is already in progress"
        if not (
            self._update_available() or self._interrupted_candidate_repair_available()
        ):
            return False, "no update available"
        if not self._is_idle():
            return False, "server is busy; stop the running operation first"
        # ``launches_blocked`` refreshes a successfully committed candidate
        # before we snapshot whether this retry inherited an unsafe state.
        self._inherited_launch_barrier = self.launches_blocked
        self._state = "updating"
        self._error = None
        self._launches_blocked = True
        self._update_task = asyncio.create_task(self._run_update())
        return True, None

    def _schedule_remote_refresh(self) -> None:
        """Kick off a debounced ``git ls-remote --tags`` if the cache is stale."""
        if self._shutting_down or not self.enabled:
            return
        if self._remote_task is not None and not self._remote_task.done():
            return
        now = time.monotonic()
        # Honor the debounce even after a failed/empty resolve (``_remote_checked_at``
        # is stamped regardless) so a poll loop can't spawn ls-remote continuously
        # when offline. The initial 0.0 lets the first poll through.
        if (
            self._remote_checked_at
            and now - self._remote_checked_at < _REMOTE_DEBOUNCE_S
        ):
            return
        self._remote_task = asyncio.create_task(self.refresh_remote())

    async def refresh_remote(self) -> None:
        """Resolve the newest release tag via read-only ``git ls-remote --tags``.

        Updates the cache only; never upgrades. Cheap and safe on a steady-state
        install (unlike the reinstall, which would prune the camera stack),
        which is why the indicator can poll it freely.
        """
        if not self.release_install:
            self._remote_checked_at = time.monotonic()
            return
        # Git installs check their own origin (forks see their own releases);
        # index installs carry no repository metadata, so use the canonical repo.
        url = self._origin[0] if self._origin is not None else _REPO_URL
        latest = await self._resolve_latest_release(url)
        self._remote_checked_at = time.monotonic()
        if latest is not None:
            self._remote_tag, self._remote_version = latest

    async def _resolve_latest_release(self, url: str) -> tuple[str, str] | None:
        """``(tag, version)`` of the highest release tag, via ``git ls-remote --tags``.

        Read-only and cheap, so it drives the polled "update available"
        indicator without touching the install. ``None`` on any failure
        (offline, no release tags yet); the caller keeps the last known value.
        """
        try:
            proc, out, _ = await self._communicate_owned(
                "git",
                "ls-remote",
                "--tags",
                url,
                "refs/tags/v*",
                "refs/tags/release-v*",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            _logger.warning("self-update: could not run git ls-remote: %s", exc)
            return None
        if proc.returncode != 0:
            return None
        best: tuple[tuple[int, ...], str] | None = None
        for line in out.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].startswith("refs/tags/"):
                continue
            # Annotated tags are also listed peeled as "<tag>^{}"; the tag name
            # is the same either way, so just strip the marker.
            tag = parts[1][len("refs/tags/") :].removesuffix("^{}")
            version = parse_version(tag)
            if version is None:
                continue
            # Legacy tags remain visible as history, but a future numeric
            # vX.Y.Z must never outrank the release-v namespace. Such a tag is
            # exactly what unmigrated destructive updaters still watch.
            if tag.startswith("v") and version >= _FIRST_HARDENED_RELEASE:
                continue
            if best is None or version > best[0]:
                best = (version, tag)
        if best is None:
            return None
        return best[1], ".".join(str(part) for part in best[0])

    def _fail(self, message: str, *, launches_unsafe: bool = True) -> None:
        """Record an update failure for the UI and log it."""
        _logger.warning("self-update: %s", message)
        # ``launches_unsafe=False`` means this attempt failed before changing
        # the host. It does not authorize clearing a barrier inherited from an
        # interrupted update or failed startup provision.
        self._launches_blocked = launches_unsafe or self._inherited_launch_barrier
        self._inherited_launch_barrier = self._launches_blocked
        self._error = message
        self._state = "error"
        self._phase = None

    async def _run_update(self) -> None:
        # Never rebuild the environment underneath this live Python process.
        # Launch the installer extracted from the SHA-256-verified target wheel
        # in its own transient systemd service; that transaction takes the host
        # lock, stops this unit, stages and provisions the exact release, then
        # admits one health-verified candidate boot. The worker survives systemd
        # stopping axol.service because it owns a separate cgroup.
        try:
            if (
                self._update_worker_unit is not None
                and not self._update_worker_handed_off
                and not await self._stop_uncommitted_update_worker()
            ):
                self._fail(
                    "a prior update worker could not be proven stopped; repair "
                    "the transient systemd unit before retrying",
                    launches_unsafe=True,
                )
                return
            if not self.release_install or self._remote_tag is None:
                self._fail("no release to install", launches_unsafe=False)
                return
            # Snapshot the release being installed: a background status poll
            # could refresh the cached remote mid-update.
            tag, target_version = self._remote_tag, self._remote_version
            update_requirements, update_preflight_error = release_update_requirements()
            if update_preflight_error is not None:
                self._fail(update_preflight_error, launches_unsafe=False)
                return
            try:
                artifact = (
                    None
                    if target_version is None
                    else await asyncio.to_thread(self._release_wheel, target_version)
                )
            except Exception:  # noqa: BLE001 - remote metadata is untrusted input
                artifact = None
            if artifact is None:
                self._fail(
                    "the selected release does not have one verifiable pure-Python "
                    "wheel on PyPI; the current installation was not changed — "
                    "retry after release publishing completes",
                    launches_unsafe=False,
                )
                return
            if (
                not tag.startswith("release-v")
                or parse_version(tag) is None
                or parse_version(tag) < _FIRST_HARDENED_RELEASE
            ):
                self._fail(
                    "the selected release is not in the hardened release namespace",
                    launches_unsafe=False,
                )
                return
            managed_error = await asyncio.to_thread(self._managed_service_error)
            if managed_error is not None:
                self._fail(managed_error, launches_unsafe=False)
                return
            operator_user, operator_error = self._operator_user_for_update()
            if operator_error is not None or operator_user is None:
                self._fail(
                    operator_error
                    or "the hosted installer operator identity is invalid",
                    launches_unsafe=False,
                )
                return
            for executable in (
                _SYSTEMD_RUN_EXECUTABLE,
                _BASH_EXECUTABLE,
                _RM_EXECUTABLE,
            ):
                if not Path(executable).is_file():
                    self._fail(
                        "the hosted update worker is unavailable; run the hosted "
                        "installer directly",
                        launches_unsafe=False,
                    )
                    return
            # The preflight above deliberately runs in the current environment
            # for immediate feedback. The installer repeats it while holding
            # the cross-process transaction lock before it changes anything.
            del update_requirements
            try:
                staged = await asyncio.to_thread(
                    self._stage_release, artifact, target_version
                )
            except Exception:  # noqa: BLE001 - never execute an unverified artifact
                self._fail(
                    "the target release wheel or its embedded installer could not "
                    "be verified; the current installation was not changed",
                    launches_unsafe=False,
                )
                return
            worker_name = f"axol-update-{os.getpid()}-{time.monotonic_ns()}"
            self._update_worker_unit = f"{worker_name}.service"
            self._update_worker_stage = staged.directory
            self._update_worker_handed_off = False
            launch_command = [
                _SYSTEMD_RUN_EXECUTABLE,
                "--quiet",
                "--collect",
                "--no-block",
                f"--unit={worker_name}.service",
                "--property=Type=exec",
                "--property=TimeoutStartSec=30min",
                # Type=exec is considered started immediately after execve, so
                # TimeoutStartSec does not bound the installer itself. Cap its
                # total lifetime while leaving enough room for Jetson package
                # installs and the single-threaded zed-gstreamer build.
                "--property=RuntimeMaxSec=45min",
                # Keep the wheel beside the installer for the transaction, then
                # let systemd remove the root-only staging directory whether it
                # succeeds or fails.
                f"--property=ExecStopPost={_RM_EXECUTABLE} -rf -- {staged.directory}",
                f"--setenv=AXOL_RELEASE_TAG={tag}",
                f"--setenv={_OPERATOR_USER_ENV}={operator_user}",
                f"--setenv=AXOL_RELEASE_WHEEL={staged.wheel}",
                f"--setenv=AXOL_RELEASE_WHEEL_SHA256={staged.sha256}",
                _BASH_EXECUTABLE,
                str(staged.installer),
            ]
            self._phase = "upgrading"
            try:
                proc, _, _ = await self._communicate_owned(
                    *launch_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError:
                stopped = await self._stop_uncommitted_update_worker()
                self._fail(
                    "could not launch the hosted update worker; run the hosted "
                    "installer directly",
                    launches_unsafe=not stopped,
                )
                return
            if proc.returncode != 0:
                stopped = await self._stop_uncommitted_update_worker()
                self._fail(
                    "the hosted update worker could not be started; run the "
                    "hosted installer directly and inspect the system journal",
                    launches_unsafe=not stopped,
                )
                return
            self._update_worker_handed_off = True
            # ExecStopPost now owns deletion; shutdown must not remove a stage
            # whose transient unit has not opened its installer and wheel yet.
            self._update_worker_stage = None
            _logger.info(
                "self-update: handed %s (v%s -> v%s) to %s.service",
                tag,
                self._version,
                target_version,
                worker_name,
            )
            self._phase = "restarting"
            if not self._shutting_down:
                self._update_monitor_task = asyncio.create_task(
                    self._monitor_update_worker(f"{worker_name}.service")
                )
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            await self._stop_uncommitted_update_worker()
            self._fail(f"{type(exc).__name__}: {exc}")

    async def _ensure_provision_once(self) -> None:
        """Provision system deps once per process, in the background.

        Older update paths could install this build without running its new
        provisioning steps. Run the idempotent heal on the first control-panel
        contact; current hosted transactions provision before candidate boot.
        Gated to the exact managed service process before raising the launch
        barrier. In particular, a release-installed CLI may also be run
        manually as root for development and must remain able to use hardware.
        """
        async with self._provision_launch_lock:
            if (
                self._shutting_down
                or self._provision_started
                or self.launches_blocked
                or not self.enabled
            ):
                return

            # This runs while create_app holds both global launch reservations,
            # so a successful check can atomically become the maintenance
            # barrier below. Do not set that barrier first: a manual root
            # ``axol serve`` is a supported development workflow but is not a
            # process systemd can safely hand to the hosted updater.
            managed_error = await asyncio.to_thread(self._managed_service_error)
            self._provision_started = True
            if managed_error is not None:
                _logger.info(
                    "self-update: startup provisioning skipped outside the "
                    "managed service (%s)",
                    managed_error,
                )
                return
            if self._shutting_down:
                return

            self._launches_blocked = True
            self._provision_task = asyncio.create_task(self._run_startup_provision())

    async def _run_startup_provision(self) -> None:
        """Provision once while keeping every new hardware launch reserved."""
        try:
            guard_error = await asyncio.to_thread(self._arm_durable_update_guard)
            if guard_error is not None:
                self._fail(guard_error)
                return
            if self._shutting_down:
                # The guard was armed before shutdown won the race. Retain it:
                # no provisioning/verification completed in this process.
                return
            error = await self._provision()
        except Exception as exc:  # noqa: BLE001 - fixed, credential-safe detail
            self._fail(
                "startup provisioning failed unexpectedly "
                f"({type(exc).__name__}); restart or repair the service"
            )
            return
        if error is not None:
            self._fail(error)
            return
        guard_error = await asyncio.to_thread(self._disarm_durable_update_guard)
        if guard_error is not None:
            self._fail(guard_error)
            return
        # A user update cannot begin while this task is active.  Once startup
        # healing has completed successfully, normal launches are safe again.
        self._inherited_launch_barrier = False
        self._launches_blocked = False

    async def _provision(self) -> str | None:
        """Run ``axol provision`` in the background (the single provisioning path).

        This is the exact idempotent, self-gating command the hosted installer
        runs to restore out-of-band dependencies and patched camera plugins.
        ``_env_lock`` prevents overlapping startup heals/runtime verification.
        """
        axol = Path(_MANAGED_AXOL_EXECUTABLE)
        if not axol.is_file():
            error = (
                f"axol is not installed at {_MANAGED_AXOL_EXECUTABLE}; "
                "cannot provision the managed environment"
            )
            _logger.warning("self-update: %s", error)
            return error
        async with self._env_lock:
            try:
                proc, _, _ = await self._communicate_owned(
                    str(axol),
                    "provision",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError as exc:
                error = f"could not run axol provision: {exc}"
                _logger.warning("self-update: %s", error)
                return error
        if proc.returncode != 0:
            _logger.warning(
                "self-update: `axol provision` failed (%s); run it directly for details",
                proc.returncode,
            )
            # Command output can include package-index credentials. Keep logs
            # and the UI deterministic and direct the operator to an explicit
            # foreground run instead of copying captured output anywhere.
            return (
                f"updated Axol, but provisioning failed ({proc.returncode}); "
                "run `axol provision` directly, repair the host, and retry"
            )
        _logger.info("self-update: provisioning complete")
        return None

    async def _verify_ultimate_runtime(self) -> str | None:
        """Verify an opted-in Ultimate runtime with the newly installed CLI."""
        axol = Path(_MANAGED_AXOL_EXECUTABLE)
        if not axol.is_file():
            return (
                "updated Axol, but cannot restore VIVE Ultimate: "
                f"{_MANAGED_AXOL_EXECUTABLE} is unavailable"
            )
        async with self._env_lock:
            try:
                proc, _, _ = await self._communicate_owned(
                    str(axol),
                    "tracker.ultimate.install",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except OSError as exc:
                return f"updated Axol, but could not restore VIVE Ultimate: {exc}"
        if proc.returncode == 0:
            _logger.info("self-update: VIVE Ultimate runtime restored and verified")
            return None
        _logger.warning(
            "self-update: VIVE Ultimate runtime restore failed (%s)", proc.returncode
        )
        # The child may invoke package tooling configured with private indexes.
        # Do not expose captured output through the control-panel API.
        return (
            "updated Axol, but VIVE Ultimate runtime restore failed"
            f" ({proc.returncode}); run `axol tracker.ultimate.install` directly"
        )
