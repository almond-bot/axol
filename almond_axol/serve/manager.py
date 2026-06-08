"""Spawns and supervises ``axol <command>`` subprocesses for the web UI.

Each launch becomes a :class:`Session`: an asyncio subprocess with its stdout
(and merged stderr) pumped line-by-line into a bounded ring buffer and fanned
out to any connected log subscribers. Sessions are stopped by signalling the
whole process group so the command's own children (uvicorn, viser, the IK
worker) are torn down too.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from .commands import COMMANDS, build_argv

_LOG_BUFFER = 4000
_QUEUE_MAX = 1000
_STOP_GRACE_S = 6.0


async def spawn_proc(argv_tail: list[str]) -> asyncio.subprocess.Process:
    """Spawn ``python -m almond_axol <argv_tail>`` with merged, streamed stdout.

    Shared by plain sessions and the ZED orchestrator. Runs in its own session
    (process group) so the whole tree can be signalled on teardown, and uses the
    serving interpreter so ``axol`` need not be on PATH.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "almond_axol",
        *argv_tail,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )


async def pump_into(
    proc: asyncio.subprocess.Process,
    session: Session,
    prefix: str | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Stream ``proc`` stdout into ``session`` (optionally tagged) until EOF.

    Returns the process exit code. ``on_line`` receives each raw (untagged)
    line so callers can watch for readiness markers (e.g. PTP lock).
    """
    assert proc.stdout is not None
    tag = f"[{prefix}] " if prefix else ""
    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            session.emit(f"{tag}{line}")
            if on_line is not None:
                on_line(line)
    except Exception as exc:  # pragma: no cover - defensive
        session.emit(f"[serve] log stream error: {exc!r}")
    return await proc.wait()


class Session:
    """A single running (or finished) CLI subprocess."""

    def __init__(self, command_id: str, args: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.command_id = command_id
        self.args = args
        self.status = "starting"  # starting | running | exited | error
        self.exit_code: int | None = None
        self.error: str | None = None
        self.started_at = time.time()
        self.proc: asyncio.subprocess.Process | None = None
        self.log: deque[str] = deque(maxlen=_LOG_BUFFER)
        # Monotonic count of every line ever emitted (the deque drops old ones);
        # lets the HTTP log-polling endpoint serve a stable resumable offset.
        self.total_emitted = 0
        self.subscribers: set[asyncio.Queue[str | None]] = set()
        # When set, the manager calls this instead of the default process-group
        # signal on stop() — the ZED orchestrator uses it to unwind every step.
        self.teardown: Callable[[], Awaitable[None]] | None = None
        # Server event loop. When emit/close_stream run off-loop (the in-process
        # operation runner emits from worker threads), subscriber wakeups are
        # marshalled back here via call_soon_threadsafe. None for subprocess
        # sessions, which only ever emit from the server loop.
        self.loop: asyncio.AbstractEventLoop | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command_id,
            "args": self.args,
            "status": self.status,
            "exitCode": self.exit_code,
            "error": self.error,
            "startedAt": self.started_at,
            "pid": self.proc.pid if self.proc else None,
        }

    @staticmethod
    def _safe_put(q: asyncio.Queue[str | None], item: str | None) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def _fanout(self, item: str | None) -> None:
        loop = self.loop
        for q in list(self.subscribers):
            if loop is not None:
                loop.call_soon_threadsafe(self._safe_put, q, item)
            else:
                self._safe_put(q, item)

    def emit(self, line: str) -> None:
        """Append a log line and fan it out to live subscribers."""
        self.log.append(line)
        self.total_emitted += 1
        self._fanout(line)

    def close_stream(self) -> None:
        """Signal end-of-stream to subscribers so their WebSockets can close."""
        self._fanout(None)

    def read_log(self, offset: int) -> tuple[list[str], int]:
        """Return log lines at/after ``offset`` plus the next offset to poll.

        ``offset`` indexes :attr:`total_emitted`; lines older than the retained
        buffer are silently skipped (the caller just resumes from the buffer).
        """
        buf = list(self.log)
        start = self.total_emitted - len(buf)
        begin = max(0, offset - start)
        return buf[begin:], self.total_emitted


class SessionManager:
    """Owns the set of live sessions and their lifecycles."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def new_session(
        self, command_id: str, args: dict[str, Any] | None = None
    ) -> Session:
        """Create and register a tracked session with no subprocess of its own.

        For aggregators that drive their own processes/tails and emit into the
        session manually (e.g. the long-lived PTP clock-sync link). It shows up
        in ``list()`` and is tailable through the normal log endpoints.
        """
        session = Session(command_id, args or {})
        self._sessions[session.id] = session
        return session

    async def start(self, command_id: str, args: dict[str, Any]) -> Session:
        if command_id not in COMMANDS:
            raise KeyError(command_id)

        cli = COMMANDS[command_id].cli
        argv = build_argv(command_id, args)
        return await self.start_raw(command_id, [cli, *argv], args=args)

    async def start_raw(
        self,
        command_id: str,
        argv_tail: list[str],
        *,
        args: dict[str, Any] | None = None,
    ) -> Session:
        """Spawn an arbitrary ``axol`` invocation as a tracked session.

        ``argv_tail`` is the full CLI tail (subcommand + flags). Used both for
        schema-validated commands and the ZED helper endpoints the orchestrator
        drives remotely (``zed.sync-clocks`` / ``zed.stream``).
        """
        session = Session(command_id, args or {})
        self._sessions[session.id] = session
        try:
            proc = await spawn_proc(argv_tail)
        except OSError as exc:
            session.status = "error"
            session.error = f"Failed to launch command: {exc}"
            session.emit(f"[serve] error: {session.error}")
            return session

        session.proc = proc
        session.status = "running"
        session.emit(f"[serve] $ axol {' '.join(argv_tail)}".rstrip())
        asyncio.create_task(self._pump(session))
        return session

    async def start_orchestrated(
        self, command_id: str, args: dict[str, Any], zed: dict[str, Any]
    ) -> Session:
        """Start a ZED-orchestrated ``collect-data`` / ``run-policy`` session.

        The session's logs aggregate every orchestration step (clock sync,
        camera streaming, the main command); ``stop`` unwinds them all.
        """
        from .orchestrator import ZedOrchestrator

        if command_id not in COMMANDS:
            raise KeyError(command_id)
        session = Session(command_id, args)
        self._sessions[session.id] = session
        orch = ZedOrchestrator(session, command_id, args, zed)
        session.teardown = orch.stop
        asyncio.create_task(orch.run())
        return session

    async def _pump(self, session: Session) -> None:
        proc = session.proc
        assert proc is not None
        rc = await pump_into(proc, session)
        session.exit_code = rc
        session.status = "exited"
        session.emit(f"[serve] process exited with code {rc}")
        session.close_stream()

    async def stop(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if session.teardown is not None:
            await session.teardown()
            return True
        proc = session.proc
        if proc is None or proc.returncode is not None:
            return True

        session.emit("[serve] stopping...")
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return True

        # SIGINT mimics Ctrl-C so commands run their KeyboardInterrupt cleanup.
        _signal_group(pgid, signal.SIGINT)
        try:
            await asyncio.wait_for(proc.wait(), timeout=_STOP_GRACE_S)
            return True
        except asyncio.TimeoutError:
            pass

        _signal_group(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
            return True
        except asyncio.TimeoutError:
            session.emit("[serve] forcing kill")
            _signal_group(pgid, signal.SIGKILL)
            await proc.wait()
            return True

    def subscribe(self, session: Session) -> asyncio.Queue[str | None]:
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_QUEUE_MAX)
        session.subscribers.add(q)
        return q

    def unsubscribe(self, session: Session, q: asyncio.Queue[str | None]) -> None:
        session.subscribers.discard(q)

    async def shutdown(self) -> None:
        """Stop every running session (used on server shutdown)."""
        for session_id in list(self._sessions):
            await self.stop(session_id)


def _signal_group(pgid: int, sig: int) -> None:
    if sys.platform == "win32":  # pragma: no cover - posix only
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
