from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from almond_axol.serve import runner
from almond_axol.serve.manager import Session
from almond_axol.serve.settings import SettingsStore


class _Thread:
    def __init__(self, target=None, args=(), **kwargs) -> None:
        self.target = target
        self.args = args
        self.started = False
        self.joins: list[float | None] = []
        self.alive = True

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return self.alive


class _RobotLink:
    def __init__(self, *, release_error: bool = False) -> None:
        self.released = 0
        self.reacquired = 0
        self.release_error = release_error

    def release(self) -> None:
        self.released += 1
        if self.release_error:
            raise RuntimeError("link failed")

    def reacquire(self) -> None:
        self.reacquired += 1


def test_output_forwarding_stream_and_logging_filters() -> None:
    lines: list[str] = []
    original = io.StringIO()
    tee = runner._StreamTee(original, lines.append)
    assert tee.write("one\ntwo") == 7
    tee.write(" continued\n")
    tee.flush()
    assert original.getvalue() == "one\ntwo continued\n"
    assert lines == ["one", "two continued"]
    assert not tee.isatty()

    runner._forward_line(lines.append, "\x1b[31mred\x1b[0m")
    runner._forward_line(lines.append, "INFO:     uvicorn noise")
    assert lines[-1] == "red"

    handler = runner._SessionLogHandler(lines.append)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(logging.LogRecord("robot", logging.INFO, "", 1, "moving", (), None))
    handler.emit(
        logging.LogRecord("uvicorn.access", logging.INFO, "", 1, "GET", (), None)
    )
    assert lines[-1] == "moving"


def test_operation_start_config_error_and_single_operation_guard(monkeypatch) -> None:
    fake_threads: list[_Thread] = []

    def make_thread(*args, **kwargs):
        thread = _Thread(*args, **kwargs)
        fake_threads.append(thread)
        return thread

    monkeypatch.setattr(runner.threading, "Thread", make_thread)
    monkeypatch.setattr(runner.multiprocessing, "active_children", lambda: [])
    operation = runner.OperationRunner()
    monkeypatch.setattr(operation, "_build_config", lambda op, args: SimpleNamespace())
    session = operation.start("teleop", {"sim": True})
    assert session.status == "running"
    assert operation.current() is session
    assert operation.get(session.id) is session
    assert operation.get("missing") is None
    assert operation.is_running()
    assert fake_threads[0].started
    with pytest.raises(RuntimeError, match="already running"):
        operation.start("teleop", {"sim": True})
    with pytest.raises(KeyError):
        runner.OperationRunner().start("does-not-exist", {})

    broken = runner.OperationRunner()

    def invalid(op, args):
        raise ValueError("bad option")

    monkeypatch.setattr(broken, "_build_config", invalid)
    failed = broken.start("teleop", {"sim": True})
    assert failed.status == "error"
    assert "bad option" in (failed.error or "")
    assert not broken.is_running()


def test_operation_releases_and_reacquires_robot(monkeypatch) -> None:
    monkeypatch.setattr(runner.threading, "Thread", _Thread)
    monkeypatch.setattr(runner.multiprocessing, "active_children", lambda: [])
    link = _RobotLink()
    operation = runner.OperationRunner(link)
    monkeypatch.setattr(operation, "_build_config", lambda op, args: SimpleNamespace())
    session = operation.start("teleop", {})
    assert link.released == 1
    operation._finish(session, needs_robot=True)
    assert link.reacquired == 1
    assert session.status == "exited"
    assert session.exit_code == 0
    assert session.log[-1] == "[serve] robot link reacquired"

    warning_link = _RobotLink(release_error=True)
    warning_op = runner.OperationRunner(warning_link)
    monkeypatch.setattr(warning_op, "_build_config", lambda op, args: SimpleNamespace())
    warning = warning_op.start("teleop", {})
    assert any("release warning" in line for line in warning.log)


def test_operation_stop_and_episode_control(monkeypatch) -> None:
    operation = runner.OperationRunner()
    assert operation.stop() is None
    session = Session("demo", {})
    session.status = "running"
    operation._session = session
    worker = _Thread()
    worker.alive = False
    operation._thread = worker  # type: ignore[assignment]
    watchdogs: list[_Thread] = []

    def make_thread(*args, **kwargs):
        item = _Thread(*args, **kwargs)
        item.alive = False
        watchdogs.append(item)
        return item

    monkeypatch.setattr(runner.threading, "Thread", make_thread)
    stopped = operation.stop()
    assert stopped is session
    assert session.status == "stopping"
    assert operation._stop_event.is_set()
    assert watchdogs[0].started
    assert operation.stop() is session

    pushed: list[str] = []
    operation._policy_control = SimpleNamespace(
        push=pushed.append, snapshot=lambda: {"phase": "waiting"}
    )
    assert operation.episode_command("save")
    assert pushed == ["save"]
    assert operation.policy_state() == {"phase": "waiting"}
    operation._policy_control = None
    assert not operation.episode_command("save")
    assert operation.policy_state() is None


def test_operation_kills_only_new_children(monkeypatch) -> None:
    killed: list[int] = []

    def child(pid: int, name: str, *, fails: bool = False):
        def kill() -> None:
            if fails:
                raise OSError("gone")
            killed.append(pid)

        return SimpleNamespace(pid=pid, name=name, kill=kill)

    old = child(1, "old")
    recorder = child(2, "dataset-recorder")
    relay = child(3, "relay")
    failed = child(4, "failed", fails=True)
    monkeypatch.setattr(
        runner.multiprocessing,
        "active_children",
        lambda: [old, recorder, relay, failed],
    )
    operation = runner.OperationRunner()
    session = Session("demo", {})
    operation._session = session
    operation._baseline_children = {1}
    assert operation._kill_op_children(session, spare={"dataset-recorder"})
    assert killed == [3]
    assert any("failed to kill pid 4" in line for line in session.log)
    assert not operation._kill_op_children(Session("other", {}))


def test_runner_camera_helpers_and_attach(monkeypatch) -> None:
    operation = runner.OperationRunner()
    assert operation._camera_serials(None) == {}
    assert operation._camera_serials(
        {
            "serials": {
                "overhead": " 12 ",
                "left_arm": "bad",
                "right_arm": -1,
                "unknown": 20,
            }
        }
    ) == {"overhead": 12}
    assert operation._branch(None, "stream", "overhead", True) == (True, None)
    assert operation._branch(
        {"stream": {"overhead": "both"}}, "stream", "overhead", True
    ) == (True, "both")
    assert operation._branch(
        {"stream": {"overhead": False}}, "stream", "overhead", True
    ) == (False, None)

    monkeypatch.setattr(runner, "stereo_serials", lambda: {12})
    session = Session("teleop", {})
    cfg = SimpleNamespace(cameras={}, camera_eyes={}, resolution=None)
    cameras = {
        "serials": {"overhead": 12, "left_arm": 13},
        "stream_resolution": "SVGA",
        "stream": {"overhead": "both", "left_arm": False},
    }
    operation._attach_cameras_to_teleop(cfg, cameras, session)
    assert cfg.cameras == {"overhead": 12}
    assert cfg.camera_eyes == {"overhead": "both"}
    assert cfg.resolution == "SVGA"
    assert "stereo: overhead" in session.log[-1]


def test_settings_reset_normalization_and_merged_args(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"
    path.parent.mkdir()
    path.write_text("corrupt")
    store = SettingsStore(path)
    assert store.snapshot() == {"values": {}, "cameras": None, "advanced": {}}
    assert store.can_channels() == ("can_alm_axol_l", "can_alm_axol_r")
    assert store.has_gripper()

    store.update(
        values={
            "robot.left_channel": "none",
            "robot.right_channel": " can7 ",
            "robot.has_gripper": "false",
            "robot.left_stiffness": 0.5,
        },
        cameras={"serials": {"overhead": "123"}},
        advanced={"axol.left.elbow.kp": 40},
    )
    assert store.can_channels() == (None, "can7")
    assert not store.has_gripper()
    assert store.cameras() == {"serials": {"overhead": "123"}}
    merged = store.merged_args("teleop", {"axol.left_stiffness": 0.9})
    assert merged["axol.left_stiffness"] == 0.9
    assert merged["axol.left.elbow.kp"] == 40

    store.update(
        values={"robot.left_stiffness": None},
        cameras=None,
        advanced={"axol.left.elbow.kp": None},
    )
    assert store.cameras() is None
    assert "robot.left_stiffness" not in store.snapshot()["values"]
    with pytest.raises(KeyError, match="unknown advanced"):
        store.update(advanced={"unknown.value": 1})


def test_runner_shutdown_calls_blocking_stop(monkeypatch) -> None:
    async def exercise() -> None:
        operation = runner.OperationRunner()
        operation._session = Session("demo", {})
        operation._session.status = "running"
        called: list[bool] = []
        monkeypatch.setattr(operation, "_stop_blocking", lambda: called.append(True))
        await operation.shutdown()
        assert called == [True]

    asyncio.run(exercise())
