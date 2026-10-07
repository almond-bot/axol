from __future__ import annotations

import asyncio
import hashlib
import json
import time
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
    """A finished subprocess; update steps get its output in their log file."""

    pid = 4242

    def __init__(self, output: bytes = b"", returncode: int = 0, **spawn) -> None:
        self.output = output
        self.returncode = returncode
        sink = spawn.get("stdout")
        if hasattr(sink, "write"):
            sink.write(output)

    async def communicate(self):
        return self.output, b""

    async def wait(self):
        return self.returncode


@pytest.fixture(autouse=True)
def _step_logs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(update, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(update, "_LOG_FALLBACK_DIR", tmp_path / "fallback")


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
            return _AsyncProc(tag_output, **kwargs)

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
            return _AsyncProc(b"", returncode=2, **kwargs)

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
        monkeypatch.setattr(update, "_installed_extras", lambda: "lerobot,sim,jax")
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        commands: list[tuple[str, ...]] = []

        async def spawn(*args, **kwargs):
            commands.append(args)
            return _AsyncProc(b"done", **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        await updater._run_update()
        assert commands[0][-1] == "almond-axol[lerobot,sim,jax]==2.0.0"
        # The post-reinstall provision is strict: the realtime core must be
        # current before the service restarts onto the new code.
        assert commands[1] == ("/bin/axol", "provision", "--no-reboot", "--require-rt")
        assert updater._phase == "restarting"
        assert updater._restart_pending
        assert exits == [0]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ({"jax"}, "lerobot,sim,jax"),
        ({"mink"}, "lerobot,sim,mink"),
        ({"jax", "mink"}, "lerobot,sim,jax,mink"),
        (set(), "lerobot,sim,jax"),
    ],
)
def test_update_preserves_kinematics_backends(monkeypatch, installed, expected) -> None:
    def dist(name):
        if name not in installed:
            raise update.PackageNotFoundError(name)
        return _Dist(None)

    monkeypatch.setattr(update, "distribution", dist)
    assert update._installed_extras() == expected


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
            return _AsyncProc(b"first\nlast line", returncode=1, **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        await updater._run_update()
        assert updater._state == "error"
        assert message in (updater._error or "")
        assert updater._phase is None

    asyncio.run(exercise())


def test_provision_failure_names_the_failed_step(monkeypatch) -> None:
    # A failed post-upgrade provision must surface the step `axol provision`
    # reported, not blame axol-rt when (say) the ZED driver pin was the culprit.
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        tail = (
            b"Provisioning failed for: ZED Box camera driver (zed.driver). "
            b"See the log above, repair the host, and retry."
        )

        async def spawn(*args, **kwargs):
            if args[1] == "provision":
                return _AsyncProc(b"log line\n" + tail, returncode=1, **kwargs)
            return _AsyncProc(b"done", **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        await updater._run_update()
        assert updater._state == "error"
        assert "zed.driver" in (updater._error or "")
        assert "axol-rt" not in (updater._error or "")

    asyncio.run(exercise())


def test_provision_once_and_failure_paths(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        tasks: list[object] = []

        def capture(coro, **kwargs):
            tasks.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(update.asyncio, "create_task", capture)
        updater.ensure_provisioned()
        updater.ensure_provisioned()
        assert len(tasks) == 1

        monkeypatch.setattr(update.shutil, "which", lambda name: None)
        assert not await updater._provision()

        monkeypatch.setattr(update.shutil, "which", lambda name: "/bin/axol")

        async def failed(*args, **kwargs):
            return _AsyncProc(b"bad provision", returncode=1, **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", failed)
        assert not await updater._provision()

        async def missing(*args, **kwargs):
            raise OSError("gone")

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", missing)
        assert not await updater._provision()

        commands: list[tuple[str, ...]] = []

        async def ok(*args, **kwargs):
            commands.append(args)
            return _AsyncProc(b"done", **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", ok)
        assert await updater._provision()
        assert await updater._provision(require_rt=True)
        assert commands == [
            ("/bin/axol", "provision", "--no-reboot"),
            ("/bin/axol", "provision", "--no-reboot", "--require-rt"),
        ]

    asyncio.run(exercise())


def _ok_spawn(*args, **kwargs):
    async def spawn(*a, **kwargs):
        return _AsyncProc(b"done", **kwargs)

    return spawn


@pytest.mark.parametrize(("rebooted", "exits_expected"), [(True, []), (False, [0])])
def test_update_reboots_instead_of_restarting_when_required(
    monkeypatch, rebooted: bool, exits_expected: list[int]
) -> None:
    # A new camera driver only loads at boot: the update ends in a host reboot
    # (under the same idle gate). If the loop guard refuses it, the service
    # still restarts onto the new code.
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", _ok_spawn())
        monkeypatch.setattr(update.reboot, "pending", lambda: ["driver"])
        calls: list[list[str]] = []

        def reboot_host(reasons):
            calls.append(reasons)
            return rebooted

        monkeypatch.setattr(update.reboot, "reboot_host", reboot_host)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        await updater._run_update()
        assert calls == [["driver"]]
        assert exits == exits_expected
        assert not updater._restart_pending

    asyncio.run(exercise())


def test_startup_heal_reboots_but_never_restart_loops(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", _ok_spawn())
        monkeypatch.setattr(update.reboot, "pending", lambda: ["driver"])
        calls: list[list[str]] = []

        def refused(reasons):
            calls.append(reasons)
            return False

        monkeypatch.setattr(update.reboot, "reboot_host", refused)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        await updater._provision_on_startup()
        # The reboot was attempted; refused, it must not fall back to a
        # service restart (that would rerun this heal on every start).
        assert calls == [["driver"]]
        assert exits == []
        assert updater._phase is None

    asyncio.run(exercise())


def test_busy_server_defers_the_reboot(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch, idle=False)
        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", _ok_spawn())
        monkeypatch.setattr(update.reboot, "pending", lambda: ["driver"])
        calls: list[list[str]] = []
        monkeypatch.setattr(
            update.reboot, "reboot_host", lambda reasons: calls.append(reasons)
        )
        await updater._provision_on_startup()
        assert calls == []
        assert updater._restart_pending

    asyncio.run(exercise())


def test_step_returns_when_a_background_child_keeps_its_output_open() -> None:
    # A daemon a provision step starts inherits the step's output. Waiting on
    # that output (rather than the step's exit) hung updates indefinitely.
    async def exercise() -> tuple[int | None, list[str]]:
        return await asyncio.wait_for(
            update._run_step(
                ["sh", "-c", "sleep 3 & echo first; echo done"],
                name="step",
                timeout=10.0,
            ),
            2.0,
        )

    assert asyncio.run(exercise()) == (0, ["first", "done"])
    log = (update._LOG_DIR / "step.log").read_text()
    assert "$ sh -c" in log
    assert "first\ndone\n" in log


def test_hung_step_times_out_and_its_process_group_is_killed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(update, "_KILL_GRACE_S", 2.0)
    marker = tmp_path / "survived"

    async def exercise() -> tuple[int | None, list[str]]:
        return await update._run_step(
            [
                "sh",
                "-c",
                f"(sleep 1; touch {marker}) & echo working; sleep 30",
            ],
            name="hung",
            timeout=0.3,
        )

    assert asyncio.run(exercise()) == (None, ["working"])
    time.sleep(1.5)
    # The step's background child was in its process group and died with it.
    assert not marker.exists()


def test_timed_out_upgrade_fails_retryably(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        monkeypatch.setattr(update, "_UV_INSTALL_TIMEOUT_S", 0.05)
        monkeypatch.setattr(update, "_KILL_GRACE_S", 0.05)
        signals: list[int] = []
        released = asyncio.Event()

        class _Hung(_AsyncProc):
            async def wait(self):
                await released.wait()
                return -9

        async def spawn(*args, **kwargs):
            return _Hung(b"Resolved 300 packages\n", **kwargs)

        def killpg(pid, sig):
            signals.append(sig)
            released.set()

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(update.os, "killpg", killpg)
        await updater._run_update()
        assert updater._state == "error"
        assert "timed out" in (updater._error or "")
        assert "Resolved 300 packages" in (updater._error or "")
        assert signals == [update.signal.SIGTERM]
        # Not wedged in "updating": the panel's retry is accepted again.
        assert updater.start() == (True, None)
        updater._update_task.cancel()

    asyncio.run(exercise())


def test_provision_timeout_names_the_last_output(monkeypatch) -> None:
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        monkeypatch.setattr(update, "_PROVISION_TIMEOUT_S", 0.05)
        monkeypatch.setattr(update, "_KILL_GRACE_S", 0.05)

        class _Hung(_AsyncProc):
            async def wait(self):
                await asyncio.sleep(3600)

        async def spawn(*args, **kwargs):
            return _Hung(b"==> GStreamer + PyGObject (gst.install)\n", **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(update.os, "killpg", lambda pid, sig: None)
        assert not await updater._provision(require_rt=True)
        assert "timed out" in updater._provision_failure_message()
        assert "gst.install" in updater._provision_failure_message()

    asyncio.run(exercise())


def test_optional_provision_failures_still_restart_onto_the_new_code(
    monkeypatch,
) -> None:
    # A host whose Lighthouse or zed-gstreamer build fails must not be stuck
    # on the old release: every retry reinstalled and then refused to restart.
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        updater._remote_tag = "v2.0.0"
        updater._remote_version = "2.0.0"
        warning = (
            update._OPTIONAL_FAILURE_PREFIX
            + "Lighthouse tracking (tracker.install), patched zed-gstreamer "
            "plugins (gst.build-zed). Axol runs without them."
        )

        async def spawn(*args, **kwargs):
            if args[1] == "provision":
                output = f"step log\n{warning}\nREBOOT REQUIRED: x. Left pending."
                return _AsyncProc(output.encode(), **kwargs)
            return _AsyncProc(b"done", **kwargs)

        monkeypatch.setattr(update.asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(update.reboot, "pending", lambda: [])
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        await updater._run_update()
        assert exits == [0]
        status = await updater.status()
        assert status["warning"] == warning
        assert status["error"] is None

    asyncio.run(exercise())


def test_optional_failure_prefix_matches_provision() -> None:
    from almond_axol.cli import provision

    assert update._OPTIONAL_FAILURE_PREFIX == provision.OPTIONAL_FAILURE_PREFIX


def test_restart_waits_for_an_install_in_progress(monkeypatch) -> None:
    # Exiting the service mid-provision takes its children down with it.
    async def exercise() -> None:
        updater = _updater(monkeypatch)
        exits: list[int] = []
        monkeypatch.setattr(update.os, "_exit", exits.append)
        updater._restart_pending = True
        async with updater._env_lock:
            assert updater.installing
            status = await updater.status()
            assert status["installing"] is True
            assert exits == []
        assert not updater.installing
        await updater.status()
        assert exits == [0]

    asyncio.run(exercise())
