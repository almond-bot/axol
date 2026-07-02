"""User-initiated update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) installs
the package with ``uv tool install`` from GitHub and runs ``axol serve`` under
a systemd service with ``Restart=always``. This module surfaces, to the control
panel, whether the tracked ``main`` has moved past the installed commit and lets
the operator apply the update on demand:

- :meth:`SelfUpdater.status` answers the polled control-panel indicator. It
  reports the installed commit and the tracked ref's head (resolved by a
  read-only ``git ls-remote``, debounced and cached), so the UI can show "update
  available" and a button. Nothing upgrades as a side effect of this check.
- :meth:`SelfUpdater.start` is the Update button. It runs ``uv tool upgrade
  almond-axol``; if the installed commit then advanced, the process exits so
  systemd restarts it on the new code. The UI then hard-reloads.

Because ``uv tool upgrade`` rebuilds the tool environment, anything that isn't a
declared PyPI dependency is dropped and must be reinstalled before we restart
onto the new code (pyzed, PyGObject), along with the patched zedxonesrc/zedsrc
plugins. Rather than enumerate those steps here, this just shells out to ``axol
provision`` — the single provisioning path the hosted installer also runs, so
the two can't drift. Every step there is idempotent and self-gating.

The read-only ``git ls-remote`` indicator is deliberately separate from the
*destructive* ``uv tool upgrade``: the upgrade rebuilds (and so prunes
pyzed/PyGObject from) the env on every run, even when the commit hasn't moved,
so it only runs when the operator explicitly asks. The cheap ``ls-remote`` can
poll freely without touching the steady-state install.

``axol provision`` runs both after an upgrade *and* once at startup. The startup
run matters for a host that upgraded *into* the GStreamer-pipeline build from an
older ``main``: that upgrade was performed by the *old* code, which knew nothing
about ``axol provision`` — so the new code self-heals on its first control-panel
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
import shutil
import time
from importlib.metadata import PackageNotFoundError, distribution
from typing import Callable

_logger = logging.getLogger(__name__)

_PACKAGE = "almond-axol"
# Minimum seconds between read-only `git ls-remote` checks. The status endpoint
# is polled, so without this every poll would spawn a git process; the check is
# cheap and the indicator does not need to be more current than this.
_REMOTE_DEBOUNCE_S = 60.0
# systemd's Restart=always uses this code like any other; chosen to make the
# intentional self-restart recognizable in `journalctl`.
_RESTART_EXIT_CODE = 0


def installed_origin() -> tuple[str, str | None, str] | None:
    """``(git url, tracked revision | None, commit id)`` for a git tool install.

    Read from PEP 610 ``direct_url.json``. Returns ``None`` for dev checkouts
    (directory installs) or when the metadata is missing. The tracked revision
    is the branch/tag the install follows (``None`` for the default branch), and
    the commit id is what is currently installed -- comparing the two against the
    remote head is how the updater decides whether there is anything new.
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
    return url, vcs.get("requested_revision"), commit


def installed_commit() -> str | None:
    """Git commit of the installed package, from PEP 610 ``direct_url.json``.

    Returns ``None`` for dev checkouts (directory installs) or when the
    metadata is missing.
    """
    origin = installed_origin()
    return origin[2] if origin is not None else None


class SelfUpdater:
    """Read-only update indicator + explicit, user-initiated upgrade.

    The control panel polls :meth:`status` (which reports the installed commit
    and whether the tracked ref has advanced) and triggers :meth:`start` from an
    Update button. Nothing upgrades automatically.

    ``is_idle`` reports whether it is safe to restart (no operation running; a
    connected robot is fine). The restart is a plain ``os._exit``; systemd's
    ``Restart=always`` brings the server back on the upgraded code.
    """

    def __init__(self, is_idle: Callable[[], bool]) -> None:
        self._is_idle = is_idle
        self._commit = installed_commit()
        # Cached remote head + when it was last resolved, so the polled status
        # endpoint answers immediately and only re-runs `git ls-remote` at most
        # once per debounce window rather than on every poll.
        self._remote_commit: str | None = None
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
        # host that upgraded into this build from an older main). ``_env_lock``
        # serializes everything that mutates the uv tool environment -- the
        # `uv tool upgrade` and every `axol provision` (startup heal +
        # post-upgrade reinstall) -- so they can never rebuild/install into it
        # at the same time.
        self._provision_started = False
        self._env_lock = asyncio.Lock()

    @property
    def commit(self) -> str | None:
        return self._commit

    @property
    def enabled(self) -> bool:
        """Updatable only when installed from git and uv is available."""
        return self._commit is not None and shutil.which("uv") is not None

    def ensure_provisioned(self) -> None:
        """Run the once-per-process ``axol provision`` startup heal (see below)."""
        self._ensure_provision_once()

    async def status(self, *, force: bool = False) -> dict[str, object]:
        """Snapshot for the control panel.

        With ``force`` (a fresh page load / explicit check), resolve the remote
        head synchronously -- bypassing the debounce -- so the response reflects
        reality immediately rather than a cached value up to a debounce window
        stale. Otherwise schedule a debounced background refresh and return the
        cached head (``None`` until the first ``git ls-remote`` resolves), which
        keeps the steady-state poll cheap.

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
        remote = self._remote_commit
        update_available = bool(
            self.enabled and remote is not None and remote != self._commit
        )
        return {
            "enabled": self.enabled,
            "commit": self._commit,
            "remoteCommit": remote,
            "updateAvailable": update_available,
            "idle": self._is_idle(),
            "state": self._state,
            "phase": self._phase,
            "error": self._error,
        }

    def start(self) -> tuple[bool, str | None]:
        """Begin a user-initiated upgrade; returns ``(started, reason)``.

        Refuses (``started=False`` with a human-readable reason) for a dev
        checkout, when nothing new is known, when an update is already running,
        or when an operation is running. The UI disables the button in those
        cases, but guard here too. On success the upgrade + provision run in the
        background and the process exits when idle so systemd relaunches the new
        code.
        """
        if not self.enabled:
            return False, "not a git tool install"
        if self._state == "updating" or (
            self._update_task is not None and not self._update_task.done()
        ):
            return False, "an update is already in progress"
        remote = self._remote_commit
        if remote is None or remote == self._commit:
            return False, "no update available"
        if not self._is_idle():
            return False, "server is busy; stop the running operation first"
        self._state = "updating"
        self._error = None
        self._update_task = asyncio.create_task(self._run_update())
        return True, None

    def _schedule_remote_refresh(self) -> None:
        """Kick off a debounced ``git ls-remote`` if the cache is stale."""
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
        """Resolve the tracked ref's head via read-only ``git ls-remote``.

        Updates the cache only; never upgrades. Cheap and safe on a steady-state
        install (unlike ``uv tool upgrade``, which would prune the camera stack),
        which is why the indicator can poll it freely.
        """
        origin = installed_origin()
        if origin is None:
            self._remote_checked_at = time.monotonic()
            return
        url, revision, _current = origin
        remote = await self._resolve_remote(url, revision)
        self._remote_checked_at = time.monotonic()
        if remote is not None:
            self._remote_commit = remote

    async def _resolve_remote(self, url: str, revision: str | None) -> str | None:
        """Commit the tracked git ref currently points at, via ``git ls-remote``.

        Read-only and cheap, so it drives the polled "update available"
        indicator without touching the install. ``None`` on any failure (offline,
        bad ref); the caller keeps the last known value.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "ls-remote",
                url,
                revision or "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
        except OSError as exc:
            _logger.warning("self-update: could not run git ls-remote: %s", exc)
            return None
        if proc.returncode != 0:
            return None
        first = out.decode("utf-8", "replace").split("\n", 1)[0].strip()
        return first.split()[0] if first else None

    def _fail(self, message: str) -> None:
        """Record an update failure for the UI and log it."""
        _logger.warning("self-update: %s", message)
        self._error = message
        self._state = "error"
        self._phase = None

    async def _run_update(self) -> None:
        # The destructive part of the flow, run only on an explicit request.
        # `uv tool upgrade` rewrites the whole tool env (pruning pyzed/PyGObject),
        # so it must not overlap an `axol provision` installing them into that
        # same env (a concurrent startup heal). Both take ``_env_lock``; we
        # release it before the post-upgrade `_provision()` below, which
        # re-acquires it (the lock is not reentrant).
        try:
            self._phase = "upgrading"
            async with self._env_lock:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "uv",
                        "tool",
                        "upgrade",
                        _PACKAGE,
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
                        f"uv tool upgrade failed: {tail[-1] if tail else 'no output'}"
                    )
                    return

                # The upgrade rewrites the tool environment; re-read the metadata
                # from disk to see whether the installed commit moved past the
                # running one (still under the lock, so it reflects this upgrade).
                new_commit = installed_commit()

            # Always reprovision after an upgrade: it ran (so the env was rebuilt
            # and pyzed/PyGObject pruned) whether or not the commit advanced.
            self._phase = "provisioning"
            await self._provision()

            if new_commit is None or new_commit == self._commit:
                # The ref didn't actually advance the install; nothing to restart
                # onto. Clear the spinner and report up to date.
                self._state = "idle"
                self._phase = None
                _logger.info("self-update: already up to date (%s)", self._commit)
                return

            _logger.info(
                "self-update: upgraded %s -> %s; restarting when idle",
                self._commit,
                new_commit,
            )
            self._phase = "restarting"
            self._restart_pending = True
            self._maybe_restart()
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self._fail(f"{type(exc).__name__}: {exc}")

    def _ensure_provision_once(self) -> None:
        """Provision system deps once per process, in the background.

        ``uv tool upgrade`` is performed by the *old* code, so a host that
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

        ``uv tool upgrade`` rebuilds the tool env and drops everything that
        isn't a PyPI dependency (pyzed, PyGObject); ``axol provision`` reinstalls
        them and (re)builds the patched zedxonesrc/zedsrc plugins. It is the
        exact command the hosted installer runs, so the two can't drift, and it
        is idempotent + self-gating (a no-op without the ZED SDK / apt / NVENC).
        Takes ``_env_lock`` so it can't overlap another provision or the
        `uv tool upgrade` (both also rewrite the tool env).
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
