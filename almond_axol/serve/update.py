"""User-initiated update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) installs
the package with ``uv tool install`` from GitHub, pinned to the latest release
tag, and runs ``axol serve`` under a systemd service with ``Restart=always``.
This module surfaces, to the control panel, whether a newer release tag exists
on the repository and lets the operator apply the update on demand:

- :meth:`SelfUpdater.status` answers the polled control-panel indicator. It
  reports the installed version and the highest release tag (resolved by a
  read-only ``git ls-remote --tags``, debounced and cached), so the UI can show
  "update available" and a button. Nothing upgrades as a side effect of this
  check. An update is offered only when a release tag with a *higher* version
  than the installed one exists -- commits landing on ``main`` between releases
  are invisible to installs.
- :meth:`SelfUpdater.start` is the Update button. It reinstalls the tool pinned
  to the newest release tag; if the installed version then advanced, the
  process exits so systemd restarts it on the new code. The UI then
  hard-reloads.

Because the reinstall rebuilds the tool environment, anything that isn't a
declared PyPI dependency is dropped and must be reinstalled before we restart
onto the new code (pyzed, PyGObject), along with the patched zedxonesrc/zedsrc
plugins. Rather than enumerate those steps here, this just shells out to ``axol
provision`` -- the single provisioning path the hosted installer also runs, so
the two can't drift. Every step there is idempotent and self-gating.

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
metadata then points at a local directory, not a git URL, and the updater
no-ops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from importlib.metadata import PackageNotFoundError, distribution
from typing import Callable

_logger = logging.getLogger(__name__)

_PACKAGE = "almond-axol"
# The tag-pinned reinstall must reproduce the hosted installer's requirement
# (web/app/public/install): same extras, same Python. Keep the two in sync.
_EXTRAS = "lerobot,sim"
_PYTHON_VERSION = "3.13"
# Release tags look like ``v0.1.2``: a leading "v" plus dotted integers.
# Anything else (pre-release suffixes, arbitrary tags) is ignored by the
# updater, so cutting a release is what makes installs see an update.
_TAG_RE = re.compile(r"^v(\d+(?:\.\d+)*)$")
# Minimum seconds between read-only `git ls-remote` checks. The status endpoint
# is polled, so without this every poll would spawn a git process; the check is
# cheap and the indicator does not need to be more current than this.
_REMOTE_DEBOUNCE_S = 60.0
# systemd's Restart=always uses this code like any other; chosen to make the
# intentional self-restart recognizable in `journalctl`.
_RESTART_EXIT_CODE = 0


def parse_version(text: str) -> tuple[int, ...] | None:
    """``(0, 1, 2)`` for ``"0.1.2"`` or ``"v0.1.2"``; ``None`` when not a release version."""
    match = _TAG_RE.match(text if text.startswith("v") else f"v{text}")
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
        # Set when an upgrade landed but the server was busy; restart at the
        # next idle opportunity (a subsequent status poll re-checks).
        self._restart_pending = False
        # The GStreamer camera stack is provisioned once per process (covers a
        # host that upgraded into this build from an older release). ``_env_lock``
        # serializes everything that mutates the uv tool environment -- the
        # tag-pinned reinstall and every `axol provision` (startup heal +
        # post-upgrade reinstall) -- so they can never rebuild/install into it
        # at the same time.
        self._provision_started = False
        self._env_lock = asyncio.Lock()

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def enabled(self) -> bool:
        """Updatable only when installed from git and uv is available."""
        return self._origin is not None and shutil.which("uv") is not None

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
            "idle": self._is_idle(),
            "state": self._state,
            "phase": self._phase,
            "error": self._error,
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
            return False, "not a git tool install"
        if self._state == "updating" or (
            self._update_task is not None and not self._update_task.done()
        ):
            return False, "an update is already in progress"
        if not self._update_available():
            return False, "no update available"
        if not self._is_idle():
            return False, "server is busy; stop the running operation first"
        self._state = "updating"
        self._error = None
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
        if self._origin is None:
            self._remote_checked_at = time.monotonic()
            return
        url, _commit = self._origin
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
            if best is None or version > best[0]:
                best = (version, tag)
        if best is None:
            return None
        return best[1], ".".join(str(part) for part in best[0])

    def _fail(self, message: str) -> None:
        """Record an update failure for the UI and log it."""
        _logger.warning("self-update: %s", message)
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
            if self._origin is None or self._remote_tag is None:
                self._fail("no release to install")
                return
            url, _commit = self._origin
            # Reinstall pinned to the newest release tag. `uv tool upgrade`
            # cannot be used here: it re-resolves the originally requested
            # revision (the previous tag), so it would never move to a new
            # release. The requirement mirrors the hosted installer's.
            requirement = f"{_PACKAGE}[{_EXTRAS}] @ git+{url}@{self._remote_tag}"
            self._phase = "upgrading"
            async with self._env_lock:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "uv",
                        "tool",
                        "install",
                        "--python",
                        _PYTHON_VERSION,
                        "--force",
                        requirement,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    out, _ = await proc.communicate()
                except OSError as exc:
                    self._fail(f"could not run uv: {exc}")
                    return
                if proc.returncode != 0:
                    tail = out.decode("utf-8", "replace").strip().splitlines()
                    self._fail(
                        f"uv tool install failed: {tail[-1] if tail else 'no output'}"
                    )
                    return

                # The reinstall rewrites the tool environment; re-read the
                # metadata from disk to see whether the installed version moved
                # past the running one (still under the lock, so it reflects
                # this install).
                new_version = installed_version()

            # Always reprovision after an upgrade: it ran (so the env was rebuilt
            # and pyzed/PyGObject pruned) whether or not the version advanced.
            self._phase = "provisioning"
            await self._provision()

            if new_version is None or new_version == self._version:
                # The release didn't actually advance the install; nothing to
                # restart onto. Clear the spinner and report up to date.
                self._state = "idle"
                self._phase = None
                _logger.info("self-update: already up to date (v%s)", self._version)
                return

            _logger.info(
                "self-update: upgraded v%s -> v%s; restarting when idle",
                self._version,
                new_version,
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
        if self._provision_started or not self.enabled:
            return
        self._provision_started = True
        asyncio.create_task(self._provision())

    async def _provision(self) -> None:
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
            _logger.warning("self-update: axol not on PATH; cannot provision")
            return
        async with self._env_lock:
            try:
                proc = await asyncio.create_subprocess_exec(
                    axol,
                    "provision",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                out, _ = await proc.communicate()
            except OSError as exc:
                _logger.warning("self-update: could not run axol provision: %s", exc)
                return
        if proc.returncode != 0:
            tail = out.decode("utf-8", "replace").strip().splitlines()
            _logger.warning(
                "self-update: `axol provision` failed (%s): %s",
                proc.returncode,
                tail[-1] if tail else "no output",
            )
        else:
            _logger.info("self-update: provisioning complete")

    def _maybe_restart(self) -> None:
        if not self._is_idle():
            _logger.info("self-update: server busy; restart deferred")
            return
        _logger.info("self-update: exiting for restart (systemd relaunches)")
        # Skip uvicorn's graceful shutdown: there is nothing running (is_idle)
        # and a clean, immediate exit lets systemd relaunch right away.
        os._exit(_RESTART_EXIT_CODE)
