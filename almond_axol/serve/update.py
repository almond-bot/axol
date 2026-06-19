"""Self-update for ``axol serve`` installed as a uv tool.

The hosted installer (``curl https://axol.almond.bot/install | bash``) installs
the package with ``uv tool install`` from GitHub and runs ``axol serve`` under
a systemd service with ``Restart=always``. This module keeps that install in
sync with ``main``: whenever the control panel talks to the server, a debounced
background task runs ``uv tool upgrade almond-axol``, and if the installed git
commit changed *and* nothing is running, the process exits so systemd restarts
it on the new code. Because the upgrade rebuilds the tool environment, pyzed
(not on PyPI, installed by ``axol zed.install``) is reinstalled first when the
ZED SDK is present.

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
from pathlib import Path
from typing import Callable

_logger = logging.getLogger(__name__)

_PACKAGE = "almond-axol"
_ZED_SDK_DIR = Path("/usr/local/zed")
# Minimum seconds between upgrade attempts; the check is triggered by polled
# API endpoints, so without this every status poll would spawn a uv process.
_DEBOUNCE_S = 5 * 60.0
# systemd's Restart=always uses this code like any other; chosen to make the
# intentional self-restart recognizable in `journalctl`.
_RESTART_EXIT_CODE = 0


def installed_commit() -> str | None:
    """Git commit of the installed package, from PEP 610 ``direct_url.json``.

    Returns ``None`` for dev checkouts (directory installs) or when the
    metadata is missing.
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
    return vcs.get("commit_id")


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

    @property
    def commit(self) -> str | None:
        return self._commit

    @property
    def enabled(self) -> bool:
        """Updatable only when installed from git and uv is available."""
        return self._commit is not None and shutil.which("uv") is not None

    def poke(self) -> None:
        """Request an update check; debounced and never blocks the caller."""
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

    async def _check(self) -> None:
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
            if proc.returncode != 0:
                tail = out.decode("utf-8", "replace").strip().splitlines()
                _logger.warning(
                    "self-update: `uv tool upgrade` failed (%s): %s",
                    proc.returncode,
                    tail[-1] if tail else "no output",
                )
                return
        except OSError as exc:
            _logger.warning("self-update: could not run uv: %s", exc)
            return

        # The upgrade rewrites the tool environment; re-read the metadata from
        # disk to see whether the installed commit moved past the running one.
        new_commit = installed_commit()
        if new_commit is None or new_commit == self._commit:
            _logger.info("self-update: already up to date (%s)", self._commit)
            return

        _logger.info(
            "self-update: upgraded %s -> %s; restarting when idle",
            self._commit,
            new_commit,
        )
        await self._reinstall_pyzed()
        self._restart_pending = True
        self._maybe_restart()

    async def _reinstall_pyzed(self) -> None:
        """Reinstall pyzed after an upgrade rebuilt the tool environment.

        pyzed is not on PyPI and thus not a declared dependency, so
        ``uv tool upgrade`` drops it whenever it re-syncs the environment.
        Running ``axol zed.install`` (the *upgraded* entry point) puts the
        wheel back into the new environment before we restart into it.
        """
        if not _ZED_SDK_DIR.exists():
            return
        axol = shutil.which("axol")
        if axol is None:
            _logger.warning("self-update: axol not on PATH; cannot reinstall pyzed")
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                axol,
                "zed.install",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
        except OSError as exc:
            _logger.warning("self-update: could not run axol zed.install: %s", exc)
            return
        if proc.returncode != 0:
            tail = out.decode("utf-8", "replace").strip().splitlines()
            _logger.warning(
                "self-update: pyzed reinstall failed (%s): %s",
                proc.returncode,
                tail[-1] if tail else "no output",
            )
        else:
            _logger.info("self-update: reinstalled pyzed into the upgraded environment")

    def _maybe_restart(self) -> None:
        if not self._is_idle():
            _logger.info("self-update: server busy; restart deferred")
            return
        _logger.info("self-update: exiting for restart (systemd relaunches)")
        # Skip uvicorn's graceful shutdown: there is nothing running (is_idle)
        # and a clean, immediate exit lets systemd relaunch right away.
        os._exit(_RESTART_EXIT_CODE)
