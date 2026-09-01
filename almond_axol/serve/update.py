"""User-initiated update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) installs
the package with ``uv tool install`` from PyPI, pinned to the version of the
latest GitHub release (the release workflow publishes every release to PyPI),
and runs ``axol serve`` under a systemd service with ``Restart=always``. This
module surfaces, to the control panel, whether a newer release exists and lets
the operator apply the update on demand:

- :meth:`SelfUpdater.status` answers the polled control-panel indicator. It
  reports the installed version and the highest release tag (resolved by a
  read-only ``git ls-remote --tags`` against the repository, debounced and
  cached), so the UI can show "update available" and a button. Nothing upgrades
  as a side effect of this check. An update is offered only when a release tag
  with a *higher* version than the installed one exists -- commits landing on
  ``main`` between releases are invisible to installs.
- :meth:`SelfUpdater.start` is the Update button. It reinstalls the tool pinned
  to the newest release's version from PyPI; once the reinstall succeeds, the
  process exits so systemd restarts it on the new code. The UI then
  hard-reloads. Hosts originally installed from GitHub (pre-PyPI releases)
  migrate to the PyPI artifact on their next update through the same path.

Because the reinstall rebuilds the tool environment, anything that isn't a
declared PyPI dependency is dropped and must be reinstalled before we restart
onto the new code (pyzed, PyGObject), along with the patched zedxonesrc/zedsrc
plugins. Rather than enumerate those steps here, this just shells out to ``axol
provision`` -- the single provisioning path the hosted installer also runs, so
the two can't drift. Every step there is idempotent and self-gating.  A VIVE
Ultimate runtime is an explicit operator opt-in, so an already-installed exact
pyvut pin is preserved in the uv transaction and verified with the newly
installed ``tracker.ultimate.install`` command before restart. An explicitly
installed published ``lerobot_robot_axol`` plugin is likewise carried into the
same transaction; direct/custom plugin sources block because their provenance
cannot be reconstructed safely. On aarch64,
PyPI's pinned PyTorch 2.10 wheel is CPU-only; an existing CUDA-enabled or custom
build therefore blocks before the force reinstall rather than being silently
replaced. That comparatively expensive torch probe runs only for an explicit
update, never during the read-only status poll.

The read-only ``git ls-remote --tags`` indicator is deliberately separate from
the *destructive* reinstall: the reinstall rebuilds (and so prunes
pyzed/PyGObject from) the env on every run, so it only runs when the operator
explicitly asks. The cheap ``ls-remote`` can poll freely without touching the
steady-state install.

``axol provision`` runs both after an upgrade *and* once at startup. The startup
run matters for a host that upgraded *into* the GStreamer-pipeline build from an
older release: that upgrade was performed by the *old* code, which knew nothing
about ``axol provision`` -- so the new code self-heals on its first control-panel
contact after the restart.

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
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
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
# systemd's Restart=always uses this code like any other; chosen to make the
# intentional self-restart recognizable in `journalctl`.
_RESTART_EXIT_CODE = 0
_SERVICE_NAME = "axol.service"
_MANTIS_SERVICE_NAME = "axol-mantis.service"
_UPDATE_GUARD_MARKER = Path("/var/lib/almond-axol/update-incomplete")
_UPDATE_GUARD_DROPIN = Path("/etc/systemd/system/axol.service.d/20-update-guard.conf")
_UPDATE_GUARD_CONTENT = f"[Unit]\nConditionPathExists=!{_UPDATE_GUARD_MARKER}\n"
_MANTIS_UPDATE_GUARD_DROPIN = Path(
    "/etc/systemd/system/axol-mantis.service.d/20-update-guard.conf"
)
_PYPI_RELEASE_URL = "https://pypi.org/pypi/almond-axol/{version}/json"
_MANAGED_UV_EXECUTABLE = "/usr/local/bin/uv"
_MANAGED_UPDATE_ENV = {
    "AXOL_PRIVILEGED_SERVICE": "1",
    "UV_TOOL_DIR": "/opt/axol/uv/tools",
    "UV_PYTHON_INSTALL_DIR": "/opt/axol/uv/python",
    "UV_TOOL_BIN_DIR": "/usr/local/bin",
}


def _release_available_on_pypi(version: str) -> bool:
    """Whether PyPI has at least one non-yanked artifact for an exact release."""
    if parse_version(version) is None:
        return False
    request = urllib.request.Request(
        _PYPI_RELEASE_URL.format(version=version),
        headers={"User-Agent": "almond-axol-self-update"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:
                return False
            payload = json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError):
        return False
    if not isinstance(payload, dict):
        return False
    info = payload.get("info")
    artifacts = payload.get("urls")
    return (
        isinstance(info, dict)
        and info.get("version") == version
        and isinstance(artifacts, list)
        and any(
            isinstance(item, dict)
            and item.get("packagetype") in {"bdist_wheel", "sdist"}
            and item.get("yanked") is not True
            for item in artifacts
        )
    )


def _write_durable_root_file(path: Path, content: str, *, mode: int) -> None:
    """Atomically write a root-owned guard file below a protected directory."""
    parent = path.parent
    parent.mkdir(parents=True, mode=0o755, exist_ok=True)
    parent_stat = parent.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != 0
        or parent_stat.st_mode & 0o022
    ):
        raise OSError(f"unsafe update-guard directory: {parent}")
    if path.is_symlink():
        raise OSError(f"update-guard path must not be a symlink: {path}")

    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary)
    try:
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

    ``is_idle`` reports whether it is safe to restart (no operation running; a
    connected robot is fine). The restart is a plain ``os._exit``; systemd's
    ``Restart=always`` brings the server back on the upgraded code.
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
        # Update lifecycle surfaced to the UI: "idle" | "updating" | "error".
        self._state = "idle"
        self._error: str | None = None
        # Current step while ``state == "updating"`` so the UI can show progress
        # instead of an opaque spinner: "upgrading" | "provisioning" |
        # "restarting" (``None`` when not updating).
        self._phase: str | None = None
        self._update_task: asyncio.Task[None] | None = None
        # A force reinstall mutates the environment that every subsequently
        # spawned Axol command imports.  Once maintenance starts, hardware and
        # session launches stay fail-closed until every install/provision/
        # verification step succeeds and this process restarts.  On failure
        # the old process remains alive for status/retry/reboot, but may not
        # launch code from a potentially partial environment.
        self._launches_blocked = (
            _UPDATE_GUARD_MARKER.exists() or _UPDATE_GUARD_MARKER.is_symlink()
        )
        if self._launches_blocked:
            self._state = "error"
            self._error = (
                "a previous update did not complete verification; retry the "
                "update or repair the service"
            )
        # Set when an upgrade landed but the server was busy; restart at the
        # next idle opportunity (a subsequent status poll re-checks).
        self._restart_pending = False
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
        # tag-pinned reinstall and every `axol provision` (startup heal +
        # post-upgrade reinstall) -- so they can never rebuild/install into it
        # at the same time.
        self._provision_started = False
        self._provision_task: asyncio.Task[None] | None = None
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

    @property
    def launches_blocked(self) -> bool:
        """Whether starting hardware or a new session is currently unsafe."""
        return self._launches_blocked or self.maintenance_active

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
            ["systemctl", *args],
            capture_output=True,
            text=True,
            timeout=15,
        )

    @staticmethod
    def _release_available(version: str) -> bool:
        return _release_available_on_pypi(version)

    def _arm_durable_update_guard(self) -> str | None:
        """Prevent a partial new environment from starting after crash/reboot.

        The currently running service must stay alive to finish the update and
        report failures, so this deliberately disables without stopping it. A
        permanent systemd condition observes the marker before every future
        start, including ``Restart=always`` restarts of this same process.
        """
        systemctl = shutil.which("systemctl")
        uv = shutil.which("uv")
        if os.geteuid() != 0 or systemctl is None:
            return "self-update requires the managed root axol.service"
        mismatched_layout = [
            name
            for name, expected in _MANAGED_UPDATE_ENV.items()
            if os.environ.get(name) != expected
        ]
        if uv != _MANAGED_UV_EXECUTABLE:
            mismatched_layout.append("uv executable")
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
            mantis_load_state = self._systemctl(
                "show",
                "--property=LoadState",
                "--value",
                _MANTIS_SERVICE_NAME,
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
                _UPDATE_GUARD_CONTENT,
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

            # The condition is live before the marker appears. A crash before
            # this write can safely restart the untouched environment; every
            # crash after it is condition-blocked.
            _write_durable_root_file(
                _UPDATE_GUARD_MARKER,
                "Axol update incomplete; repair or retry before service start.\n",
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
            _remove_durable_file(_UPDATE_GUARD_MARKER)
        except (OSError, subprocess.SubprocessError):
            return "could not safely re-enable the verified axol.service"

        if self._mantis_restore_requested:
            restore_error: str | None = None
            try:
                started = self._systemctl("start", _MANTIS_SERVICE_NAME)
                if started.returncode != 0:
                    restore_error = "axol-mantis.service could not be started"
                else:
                    active = self._systemctl(
                        "is-active", "--quiet", _MANTIS_SERVICE_NAME
                    )
                    if active.returncode != 0:
                        restore_error = "axol-mantis.service did not become active"
            except (OSError, subprocess.SubprocessError):
                restore_error = "axol-mantis.service could not be verified"

            if restore_error is not None:
                recovery_errors: list[str] = []
                try:
                    _write_durable_root_file(
                        _UPDATE_GUARD_MARKER,
                        "Axol update incomplete; repair or retry before service start.\n",
                        mode=0o600,
                    )
                except OSError:
                    recovery_errors.append("the durable marker could not be restored")
                try:
                    stopped = self._systemctl("stop", _MANTIS_SERVICE_NAME)
                    if stopped.returncode != 0:
                        recovery_errors.append(
                            "axol-mantis.service could not be stopped"
                        )
                except (OSError, subprocess.SubprocessError):
                    recovery_errors.append(
                        "axol-mantis.service stop could not be verified"
                    )
                try:
                    disabled = self._systemctl("disable", _SERVICE_NAME)
                    if disabled.returncode != 0:
                        recovery_errors.append("axol.service could not be disabled")
                except (OSError, subprocess.SubprocessError):
                    recovery_errors.append("axol.service disable could not be verified")
                detail = (
                    f"; recovery incomplete: {', '.join(recovery_errors)}"
                    if recovery_errors
                    else "; services remain update-guarded"
                )
                return f"updated Axol was verified, but {restore_error}{detail}"

        self._mantis_restore_requested = None
        self._mantis_enable_requested = None
        return None

    def ensure_provisioned(self) -> None:
        """Run the once-per-process ``axol provision`` startup heal (see below)."""
        self._ensure_provision_once()

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

        Reads ``is_idle`` live so the UI can gate the Update button on a
        safe-to-restart server. If an upgrade landed while the server was busy,
        this also re-attempts the deferred restart.
        """
        if self._restart_pending:
            self._maybe_restart()
        if force:
            # Await an in-flight background check rather than racing a second
            # ls-remote against it; otherwise resolve now.
            if self._remote_task is not None and not self._remote_task.done():
                await self._remote_task
            else:
                await self.refresh_remote()
        else:
            self._schedule_remote_refresh()
        return {
            "enabled": self.enabled,
            "version": self._version,
            "remoteVersion": self._remote_version,
            "updateAvailable": self._update_available(),
            "idle": self._is_idle() and not self.launches_blocked,
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
        those cases, but guard here too. On success the reinstall + provision
        run in the background and the process exits when idle so systemd
        relaunches the new code.
        """
        if not self.enabled:
            return False, "not a release install"
        if self.maintenance_active:
            return False, "server maintenance is already in progress"
        if not self._update_available():
            return False, "no update available"
        if not self._is_idle():
            return False, "server is busy; stop the running operation first"
        self._state = "updating"
        self._error = None
        self._launches_blocked = True
        self._update_task = asyncio.create_task(self._run_update())
        return True, None

    def _schedule_remote_refresh(self) -> None:
        """Kick off a debounced ``git ls-remote --tags`` if the cache is stale."""
        if not self.enabled:
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
            proc = await asyncio.create_subprocess_exec(
                "git",
                "ls-remote",
                "--tags",
                url,
                "refs/tags/v*",
                "refs/tags/release-v*",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
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
        self._launches_blocked = launches_unsafe
        self._error = message
        self._state = "error"
        self._phase = None

    async def _run_update(self) -> None:
        # The destructive part of the flow, run only on an explicit request.
        # The tag-pinned `uv tool install --force` rewrites the whole tool env
        # (pruning pyzed/PyGObject), so it must not overlap an `axol provision`
        # installing them into that same env (a concurrent startup heal). Both
        # take ``_env_lock``; we release it before the post-upgrade
        # `_provision()` below, which re-acquires it (the lock is not reentrant).
        try:
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
            if target_version is None or not await asyncio.to_thread(
                self._release_available, target_version
            ):
                self._fail(
                    "the selected release is not yet available from PyPI; "
                    "the current installation was not changed — retry later",
                    launches_unsafe=False,
                )
                return
            guard_error = await asyncio.to_thread(self._arm_durable_update_guard)
            if guard_error is not None:
                self._fail(guard_error)
                return
            # Reinstall pinned to the newest release's version, from PyPI (the
            # release workflow publishes every release there; GitHub tags stay
            # the source of truth for what the newest release *is*). `uv tool
            # upgrade` cannot be used here: it re-resolves the originally
            # requested version, so it would never move to a new release. The
            # requirement mirrors the hosted installer's.
            requirement = f"{_PACKAGE}[{_EXTRAS}]=={target_version or tag.lstrip('v')}"
            install_command = [
                "uv",
                "tool",
                "install",
                "--python",
                _PYTHON_VERSION,
                "--force",
            ]
            for update_requirement in update_requirements:
                # Keep every explicit opt-in inside the same resolver
                # transaction. If a preserved dependency cannot be restored,
                # uv fails without reporting a successful Axol update.
                install_command.extend(["--with", update_requirement])
            install_command.append(requirement)
            self._phase = "upgrading"
            async with self._env_lock:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *install_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    await proc.communicate()
                except OSError as exc:
                    self._fail(f"could not run uv: {exc}")
                    return
                if proc.returncode != 0:
                    _logger.warning(
                        "self-update: uv tool install failed (%s)", proc.returncode
                    )
                    # uv output can contain a configured private-index URL.
                    # Keep API/UI errors fixed and credential-free.
                    self._fail(
                        f"uv tool install failed ({proc.returncode}); run the "
                        "hosted installer directly for diagnostic output"
                    )
                    return

            # The reinstall rebuilt the env, so reprovision before restarting
            # onto the new code (pyzed/PyGObject were pruned).
            self._phase = "provisioning"
            provision_error = await self._provision()
            if provision_error is not None:
                self._fail(provision_error)
                return
            if any(
                requirement.startswith("git+https://github.com/nijkah/pyvut.git@")
                for requirement in update_requirements
            ):
                # Run the command from the newly installed Axol release.  It
                # verifies API shape/system dependencies and moves to a newer
                # expected pyvut pin if that release changed it.  A failure is
                # surfaced without restarting into a degraded tracker setup.
                ultimate_error = await self._verify_ultimate_runtime()
                if ultimate_error is not None:
                    self._fail(ultimate_error)
                    return

            guard_error = await asyncio.to_thread(self._disarm_durable_update_guard)
            if guard_error is not None:
                self._fail(guard_error)
                return

            # The install succeeded, so the target tag is what's on disk now.
            # Deliberately don't re-read the installed version through
            # importlib.metadata here: its path caches can still serve this
            # process the pre-install metadata, which would make a real upgrade
            # look like a no-op and skip the restart. `start()` only runs when
            # the release is strictly newer, so a successful install always
            # warrants the restart.
            _logger.info(
                "self-update: installed %s (v%s -> v%s); restarting when idle",
                tag,
                self._version,
                target_version,
            )
            self._phase = "restarting"
            self._restart_pending = True
            self._maybe_restart()
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self._fail(f"{type(exc).__name__}: {exc}")

    def _ensure_provision_once(self) -> None:
        """Provision system deps once per process, in the background.

        The upgrade reinstall is performed by the *old* code, so a host that
        upgraded *into* this build never ran ``axol provision`` for it. Run it
        on the first control-panel contact after we (re)start onto code that
        needs it; ``axol provision`` is idempotent, so it's a cheap no-op once
        satisfied. Gated to real (git) tool installs, like the updater itself.
        """
        if self._provision_started or self.launches_blocked or not self.enabled:
            return
        self._provision_started = True
        self._launches_blocked = True
        self._provision_task = asyncio.create_task(self._run_startup_provision())

    async def _run_startup_provision(self) -> None:
        """Provision once while keeping every new hardware launch reserved."""
        try:
            guard_error = await asyncio.to_thread(self._arm_durable_update_guard)
            if guard_error is not None:
                self._fail(guard_error)
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
        self._launches_blocked = False

    async def _provision(self) -> str | None:
        """Run ``axol provision`` in the background (the single provisioning path).

        The upgrade reinstall rebuilds the tool env and drops everything that
        isn't a PyPI dependency (pyzed, PyGObject); ``axol provision`` reinstalls
        them and (re)builds the patched zedxonesrc/zedsrc plugins. It is the
        exact command the hosted installer runs, so the two can't drift, and it
        is idempotent + self-gating (a no-op without the ZED SDK / apt / NVENC).
        Takes ``_env_lock`` so it can't overlap another provision or the
        upgrade reinstall (both also rewrite the tool env).
        """
        axol = shutil.which("axol")
        if axol is None:
            error = "axol is not on PATH; cannot provision the updated environment"
            _logger.warning("self-update: %s", error)
            return error
        async with self._env_lock:
            try:
                proc = await asyncio.create_subprocess_exec(
                    axol,
                    "provision",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await proc.communicate()
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
        axol = shutil.which("axol")
        if axol is None:
            return "updated Axol, but cannot restore VIVE Ultimate: axol is not on PATH"
        async with self._env_lock:
            try:
                proc = await asyncio.create_subprocess_exec(
                    axol,
                    "tracker.ultimate.install",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await proc.communicate()
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

    def _maybe_restart(self) -> None:
        if not self._is_idle():
            _logger.info("self-update: server busy; restart deferred")
            return
        _logger.info("self-update: exiting for restart (systemd relaunches)")
        # Skip uvicorn's graceful shutdown: there is nothing running (is_idle)
        # and a clean, immediate exit lets systemd relaunch right away.
        os._exit(_RESTART_EXIT_CODE)
