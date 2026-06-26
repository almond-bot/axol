"""Self-update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) installs
the package with ``uv tool install`` from GitHub and runs ``axol serve`` under
a systemd service with ``Restart=always``. This module keeps that install in
sync with ``main``: whenever the control panel talks to the server, a debounced
background task checks the tracked git ref with a read-only ``git ls-remote``
and, only when it points at a new commit, runs ``uv tool upgrade almond-axol``;
if the installed commit then changed *and* nothing is running, the process exits
so systemd restarts it on the new code.

Because ``uv tool upgrade`` rebuilds the tool environment, anything that isn't a
declared PyPI dependency is dropped and must be reinstalled before we restart
onto the new code (pyzed, PyGObject), along with the patched zedxonesrc/zedsrc
plugins. Rather than enumerate those steps here, this just shells out to ``axol
provision`` — the single provisioning path the hosted installer also runs, so
the two can't drift. Every step there is idempotent and self-gating.

That same pruning is why the ``git ls-remote`` pre-check matters: ``uv tool
upgrade`` rebuilds (and so prunes pyzed/PyGObject from) the env on *every* run,
even when the commit hasn't moved, so running it on an already-current install
would strip the camera stack out from under the running server. Gating it on a
genuinely new commit keeps a steady-state install untouched.

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
# Minimum seconds between upgrade attempts; the check is triggered by polled
# API endpoints, so without this every status poll would spawn a uv process.
_DEBOUNCE_S = 5 * 60.0
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
    """Debounced ``uv tool upgrade`` + restart-when-idle.

    ``is_idle`` reports whether it is safe to restart (no operation running,
    no live sessions). The restart is a plain ``os._exit``; systemd's
    ``Restart=always`` brings the server back on the upgraded code.
    """

    def __init__(self, is_idle: Callable[[], bool]) -> None:
        self._is_idle = is_idle
        self._commit = installed_commit()
        self._last_check = 0.0
        self._task: asyncio.Task[None] | None = None
        # Set when an upgrade landed but the server was busy; restart at the
        # next idle opportunity without waiting out the debounce again.
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

    def poke(self) -> None:
        """Request an update check; debounced and never blocks the caller."""
        self._ensure_provision_once()
        if self._restart_pending:
            self._maybe_restart()
            return
        if not self.enabled:
            return
        if self._task is not None and not self._task.done():
            return
        now = time.monotonic()
        if now - self._last_check < _DEBOUNCE_S:
            return
        self._last_check = now
        self._task = asyncio.create_task(self._check())

    async def _remote_commit(self, url: str, revision: str | None) -> str | None:
        """Commit the tracked git ref currently points at, via ``git ls-remote``.

        Read-only and cheap. Resolving the remote head up front lets us skip the
        *destructive* ``uv tool upgrade`` entirely when there is nothing new:
        that upgrade rebuilds the tool env and prunes every non-PyPI dep (pyzed,
        PyGObject) on each run, and `axol provision` only puts them back when we
        actually move to new code -- so running it on an already-current install
        would strip the camera stack out from under the running server (relay ->
        ``ModuleNotFoundError: pyzed``) and never restore it.
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

    async def _check(self) -> None:
        # `uv tool upgrade` rebuilds the tool env and prunes the non-PyPI deps
        # (pyzed, PyGObject) on *every* run, even when the tracked commit hasn't
        # moved -- and we only reprovision them when we actually advance to new
        # code. So resolve the remote head with a read-only `git ls-remote` first
        # and only run the destructive upgrade when there is genuinely something
        # new; otherwise an up-to-date install would lose its camera stack
        # (relay -> ModuleNotFoundError: pyzed) and never get it back.
        origin = installed_origin()
        if origin is None:
            return
        url, revision, current = origin
        remote = await self._remote_commit(url, revision)
        if remote is None:
            _logger.info("self-update: could not resolve remote commit; skipping")
            return
        if remote == current:
            _logger.info("self-update: already up to date (%s)", current)
            return

        # New commit on the tracked ref. `uv tool upgrade` rewrites the whole
        # tool env, so it must not overlap an `axol provision` installing
        # pyzed/PyGObject into that same env (a concurrent startup heal). Both
        # take ``_env_lock``; we release it before the post-upgrade
        # `_provision()` below, which re-acquires it (the lock is not reentrant).
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
                _logger.warning("self-update: could not run uv: %s", exc)
                return
            if proc.returncode != 0:
                tail = out.decode("utf-8", "replace").strip().splitlines()
                _logger.warning(
                    "self-update: `uv tool upgrade` failed (%s): %s",
                    proc.returncode,
                    tail[-1] if tail else "no output",
                )
                return

            # The upgrade rewrites the tool environment; re-read the metadata
            # from disk to see whether the installed commit moved past the
            # running one (still under the lock, so it reflects this upgrade).
            new_commit = installed_commit()

        # Always reprovision after an upgrade: it ran (so the env was rebuilt and
        # pyzed/PyGObject pruned) whether or not the commit ended up advancing.
        await self._provision()

        if new_commit is None or new_commit == self._commit:
            _logger.info("self-update: already up to date (%s)", self._commit)
            return

        _logger.info(
            "self-update: upgraded %s -> %s; restarting when idle",
            self._commit,
            new_commit,
        )
        self._restart_pending = True
        self._maybe_restart()

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
