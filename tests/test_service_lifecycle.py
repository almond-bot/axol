from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from almond_axol.serve import manager, telemetry


class _AsyncLines:
    def __init__(self, *lines: bytes) -> None:
        self._lines = iter(lines)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _Stdin:
    def __init__(self, *, broken: bool = False) -> None:
        self.writes: list[bytes] = []
        self.broken = broken

    def write(self, value: bytes) -> None:
        if self.broken:
            raise BrokenPipeError
        self.writes.append(value)

    async def drain(self) -> None:
        return None


class _Proc:
    def __init__(self, *lines: bytes, returncode: int | None = None) -> None:
        self.stdout = _AsyncLines(*lines)
        self.stdin = _Stdin()
        self.pid = 1234
        self.returncode = returncode

    async def wait(self) -> int:
        return 0 if self.returncode is None else self.returncode


def test_session_stream_input_and_resumable_log() -> None:
    async def exercise() -> None:
        session = manager.Session("demo", {"speed": 2})
        proc = _Proc(b"first\n", b"bad-utf8-\xff\n")
        session.proc = proc  # type: ignore[assignment]
        queue = asyncio.Queue(maxsize=1)
        session.subscribers.add(queue)

        seen: list[str] = []
        assert await manager.pump_into(proc, session, "child", seen.append) == 0
        assert seen == ["first", "bad-utf8-�"]
        assert await session.send_input("continue")
        assert proc.stdin.writes == [b"continue\n"]
        assert await queue.get() == "[child] first"

        lines, offset = session.read_log(1)
        assert lines == ["[child] bad-utf8-�"]
        assert offset == 2
        assert session.to_dict()["pid"] == 1234

        proc.returncode = 0
        assert not await session.send_input("ignored")
        session.proc = None
        assert not await session.send_input("ignored")

    asyncio.run(exercise())


def test_session_threadsafe_fanout_and_broken_input() -> None:
    class ImmediateLoop:
        def call_soon_threadsafe(self, callback, *args) -> None:
            callback(*args)

    async def exercise() -> None:
        session = manager.Session("demo", {})
        session.loop = ImmediateLoop()  # type: ignore[assignment]
        queue = asyncio.Queue(maxsize=2)
        session.subscribers.add(queue)
        session.emit("hello")
        session.close_stream()
        assert await queue.get() == "hello"
        assert await queue.get() is None

        proc = _Proc()
        proc.stdin = _Stdin(broken=True)
        session.proc = proc  # type: ignore[assignment]
        assert not await session.send_input("answer")

    asyncio.run(exercise())


def test_session_manager_launch_error_and_completed_process(monkeypatch) -> None:
    async def exercise() -> None:
        sessions = manager.SessionManager()

        async def fail_spawn(*args, **kwargs):
            raise OSError("no executable")

        monkeypatch.setattr(manager, "spawn_proc", fail_spawn)
        failed = await sessions.start_raw("broken", ["broken"])
        assert failed.status == "error"
        assert "no executable" in (failed.error or "")

        proc = _Proc(b"ready\n", returncode=7)

        async def fake_spawn(*args, **kwargs):
            return proc

        spawned: list[object] = []

        def capture_task(coro):
            spawned.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(manager, "spawn_proc", fake_spawn)
        monkeypatch.setattr(manager.asyncio, "create_task", capture_task)
        running = await sessions.start_raw("demo", ["demo", "--flag"])
        assert running.status == "running"
        assert running in [sessions.get(running.id)]
        assert sessions.list()[-1]["command"] == "demo"
        assert spawned

        await sessions._pump(running)
        assert running.status == "exited"
        assert running.exit_code == 7
        assert running.log[-1].endswith("code 7")
        assert await sessions.stop(running.id)
        assert not await sessions.stop("missing")

    asyncio.run(exercise())


def test_session_manager_custom_teardown_and_shutdown() -> None:
    async def exercise() -> None:
        sessions = manager.SessionManager()
        called: list[str] = []
        session = manager.Session("inline", {})

        async def teardown() -> None:
            called.append("done")

        session.teardown = teardown
        sessions._sessions[session.id] = session
        queue = sessions.subscribe(session)
        sessions.unsubscribe(session, queue)
        await sessions.shutdown()
        assert called == ["done"]

    asyncio.run(exercise())


def test_telemetry_hub_snapshot_history_and_subscription(monkeypatch) -> None:
    times = iter([100.0, 101.0, 102.0, 103.0])
    monkeypatch.setattr(telemetry.time, "time", lambda: next(times))
    hub = telemetry.TelemetryHub()

    async def exercise() -> None:
        queue = hub.subscribe()
        hub.push_frame({"left:J1": [1.0, 2.0, 3.0]})
        hub.push_slow({"left:J1": {"temperature": 30}})
        hub.push_state("ready")
        hub.push_state("ready")

        assert (await queue.get())["type"] == "frame"
        assert (await queue.get())["type"] == "slow"
        assert await queue.get() == {"type": "state", "state": "ready"}
        assert queue.empty()
        snapshot = hub.snapshot()
        assert snapshot["state"] == "ready"
        assert snapshot["latest"]["t"] == 100.0
        assert snapshot["slow"]["left:J1"]["temperature"] == 30
        hub.unsubscribe(queue)

    asyncio.run(exercise())
    hub._frames = deque(({"t": float(i), "m": {}} for i in range(10)), maxlen=20)
    monkeypatch.setattr(telemetry.time, "time", lambda: 10.0)
    assert [f["t"] for f in hub.history(5, max_frames=3)] == [5.0, 6.0, 8.0]
    assert [f["t"] for f in hub.frames_between(2.0, 4.0)] == [2.0, 3.0, 4.0]
    hub.clear_slow()
    assert hub.snapshot()["slow"] == {}


def test_diagnostics_store_round_trip_csv_and_clear(
    tmp_path: Path, monkeypatch
) -> None:
    now = iter([10.0, 20.0])
    monkeypatch.setattr(telemetry.time, "time", lambda: next(now))
    hub = telemetry.TelemetryHub()
    hub._frames.extend(
        [{"t": 12.0, "m": {}}, {"t": 18.0, "m": {}}, {"t": 30.0, "m": {}}]
    )
    store = telemetry.DiagnosticsRunStore(hub, tmp_path / "runs")
    meta = store.begin("session", "rom-test", {"arm": "left"})

    capture = tmp_path / "capture.csv"
    capture.write_text(
        "t,left:J1:pos,left:J1:vel,left:J1:tq,ignored\n1.0,1,2,3,x\n2.0,4,,6,x\n"
    )
    store.finalize(meta, "exited", 0, [f"[telemetry] csv={capture}", "ok"])
    assert meta["frameCount"] == 2
    assert store.list() == [meta]

    loaded = store.load(meta["id"], max_frames=1)
    assert loaded is not None
    assert loaded["frames"] == [{"t": 1.0, "m": {"left:J1": [1.0, 2.0, 3.0]}}]
    assert loaded["log"][-1] == "ok"
    assert store.load("missing") is None

    assert store.clear() == 1
    assert not capture.exists()
    assert store.list() == []
    assert store.clear() == 0


def test_diagnostics_store_tolerates_corrupt_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "bad.meta.json").write_text("not json")
    (runs / "ok.meta.json").write_text(json.dumps({"id": "ok", "startedAt": 1}))
    (runs / "ok.data.json").write_text("bad")
    store = telemetry.DiagnosticsRunStore(telemetry.TelemetryHub(), runs)
    assert store.list() == [{"id": "ok", "startedAt": 1}]
    assert store.load("ok") == {
        "meta": {"id": "ok", "startedAt": 1},
        "frames": [],
        "log": [],
    }

    malformed = tmp_path / "bad.csv"
    malformed.write_text("wrong,header\n1,2\n")
    assert telemetry._read_csv_frames(malformed) == []
    malformed.write_text("t,left:J1:pos\nnot-a-time,1\n")
    assert telemetry._read_csv_frames(malformed) == []
