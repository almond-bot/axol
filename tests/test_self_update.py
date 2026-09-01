from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from almond_axol.serve import update


class _Dist:
    def __init__(self, direct_url: str | None, version: str = "1.2.3") -> None:
        self._direct_url = direct_url
        self.version = version

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self._direct_url


class _AsyncProc:
    def __init__(self, output: bytes = b"", returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode

    async def communicate(self):
        return self.output, b""


def test_version_and_install_metadata(monkeypatch) -> None:
    assert update.parse_version("v1.2.30") == (1, 2, 30)
    assert update.parse_version("1.0") == (1, 0)
    assert update.parse_version("v1.0-rc1") is None

    vcs = json.dumps(
        {
            "url": "git+https://example.test/project.git",
            "vcs_info": {"commit_id": "abc123"},
        }
    )
    monkeypatch.setattr(update, "distribution", lambda name: _Dist(vcs, "2.0"))
    assert update.installed_origin() == (
        "https://example.test/project.git",
        "abc123",
    )
    assert not update.installed_from_index()
    assert update.installed_version() == "2.0"

    monkeypatch.setattr(update, "distribution", lambda name: _Dist(None))
    assert update.installed_origin() is None
    assert update.installed_from_index()

    monkeypatch.setattr(update, "distribution", lambda name: _Dist("not json"))
    assert update.installed_origin() is None

    def missing(name):
        raise update.PackageNotFoundError(name)

    monkeypatch.setattr(update, "distribution", missing)
    assert update.installed_origin() is None
    assert not update.installed_from_index()
    assert update.installed_version() is None


def test_git_helper_and_checkout_commit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        update.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"output\n"),
    )
    assert update._git(tmp_path, "status") == b"output\n"
    monkeypatch.setattr(
        update.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"bad"),
    )
    assert update._git(tmp_path, "status") is None

    def explode(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(update.subprocess, "run", explode)
    assert update._git(tmp_path, "status") is None

    monkeypatch.setattr(update, "installed_origin", lambda: ("url", "pinned"))
    assert update.installed_commit() == "pinned"

    monkeypatch.setattr(update, "installed_origin", lambda: None)
    fake_module = tmp_path / "pkg" / "serve" / "update.py"
    fake_module.parent.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(update, "__file__", str(fake_module))
    outputs = iter([b"deadbeef\n", b" M file\n", b"diff bytes"])
    monkeypatch.setattr(update, "_git", lambda *args: next(outputs))
    digest = hashlib.sha256(b" M file\n" + b"diff bytes").hexdigest()[:8]
    assert update.installed_commit() == f"deadbeef-dirty.{digest}"


def _updater(monkeypatch, *, idle: bool = True, index: bool = True):
    monkeypatch.setattr(update, "installed_origin", lambda: None)
    monkeypatch.setattr(update, "installed_from_index", lambda: index)
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "installed_commit", lambda: "commit")
    monkeypatch.setattr(update.shutil, "which", lambda name: f"/bin/{name}")
    return update.SelfUpdater(lambda: idle)


def test_remote_release_resolution_and_status(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        tag_output = (
            b"a refs/tags/v1.2.0\n"
            b"b refs/tags/v2.0.0^{}\n"
            b"c refs/tags/not-a-release\n"
            b"malformed\n"
        )

        async def spawn(*args, **kwargs):
            return _AsyncProc(tag_output)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        assert await updater._resolve_latest_release("origin") == ("v2.0.0", "2.0.0")
        await updater.refresh_remote()
        status = await updater.status(force=True)
        assert status["enabled"] is True
        assert status["version"] == "1.0.0"
        assert status["remoteVersion"] == "2.0.0"
        assert status["updateAvailable"] is True
        assert status["idle"] is True

        async def bad_spawn(*args, **kwargs):
            return _AsyncProc(b"", returncode=2)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", bad_spawn)
        assert await updater._resolve_latest_release("origin") is None

        async def missing_spawn(*args, **kwargs):
            raise OSError("missing")

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", missing_spawn)
        assert await updater._resolve_latest_release("origin") is None

    asyncio.run(exercise())


def test_start_guards_and_debounced_refresh(monkeypatch) -> None:
    async def exercise() -> None:
        disabled = _updater(monkeypatch, index=False)
        assert disabled.start() == (False, "not a release install")

        updater = _updater(monkeypatch)
        assert updater.start() == (False, "no update available")
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        updater._state = "updating"
        assert updater.start() == (False, "an update is already in progress")

        busy = update.SelfUpdater(lambda: False)
        busy._remote_tag = "v2.0.0"
        busy._remote_version = "2.0.0"
        assert busy.start() == (
            False,
            "server is busy; stop the running operation first",
        )

        calls: list[str] = []

        async def refreshed() -> None:
            calls.append("refresh")

        updater._state = "idle"
        updater.refresh_remote = refreshed  # type: ignore[method-assign]
        updater._remote_checked_at = 0
        updater._schedule_remote_refresh()
        assert updater._remote_task is not None
        await updater._remote_task
        assert calls == ["refresh"]
        updater._remote_checked_at = update.time.monotonic()
        updater._schedule_remote_refresh()
        assert calls == ["refresh"]

    asyncio.run(exercise())


def test_successful_update_and_provision(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        commands: list[tuple[str, ...]] = []

        async def spawn(*args, **kwargs):
            commands.append(args)
            return _AsyncProc(b"done")

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        await updater._run_update()
        assert commands[0][-1] == "almond-axol[lerobot,sim]==2.0.0"
        assert commands[1] == ("/bin/axol", "provision")
        assert updater._phase == "restarting"
        assert updater._restart_pending
        assert exits == [0]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "could not run uv"),
        ("returncode", "uv tool install failed: last line"),
    ],
)
def test_update_failures(monkeypatch, failure: str, message: str) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"

        async def spawn(*args, **kwargs):
            if failure == "missing":
                raise OSError("not found")
            return _AsyncProc(b"first\nlast line", returncode=1)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        await updater._run_update()
        assert updater._state == "error"
        assert message in (updater._error or "")
        assert updater._phase is None

    asyncio.run(exercise())


def test_provision_once_and_failure_paths(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        tasks: list[object] = []

        def capture(coro):
            tasks.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(update.asyncio, "create_task", capture)
        updater.ensure_provisioned()
        updater.ensure_provisioned()
        assert len(tasks) == 1

        monkeypatch.setattr(update.shutil, "which", lambda name: None)
        await updater._provision()

        monkeypatch.setattr(update.shutil, "which", lambda name: "/bin/axol")

        async def failed(*args, **kwargs):
            return _AsyncProc(b"bad provision", returncode=1)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", failed)
        await updater._provision()

        async def missing(*args, **kwargs):
            raise OSError("gone")

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", missing)
        await updater._provision()

    asyncio.run(exercise())
