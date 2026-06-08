"""In-process runner for the four core operations.

Unlike :class:`~almond_axol.serve.manager.SessionManager` (which spawns the
generic calibration/setup commands as ``axol <cmd>`` subprocesses), the four
core operations — teleop, gravity-comp, collect-data, run-policy — run *inside*
the serve process here, so they share the persistent robot connection instead
of opening their own from a child process.

Only one operation runs at a time. Its ``logging`` output and ``print``s are
captured into a :class:`~almond_axol.serve.manager.Session` ring buffer (the
same object the log WebSocket streams), so the UI sees live output exactly as
it did for subprocesses.

- teleop / gravity-comp are asyncio: they run on a dedicated event loop in a
  worker thread and are stopped by cancelling the task (both already tear down
  cleanly on ``CancelledError`` via their ``async with`` robot context).
- collect-data / run-policy are blocking/threaded: they run on a worker thread
  and are stopped via a ``threading.Event`` (run-policy additionally takes a
  queue-backed episode control for save/rerecord/quit from the UI).

Before a hardware operation starts the runner releases the robot link's CAN
bus; when the operation ends it hands the bus back.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any

from .manager import Session

_logger = logging.getLogger(__name__)

# Operations that need exclusive ownership of the CAN bus (everything except
# sim teleop, which is decided per-run from the ``sim`` arg).
_HARDWARE_OPS = {"teleop", "gravity-comp", "collect-data", "run-policy"}
_ASYNC_OPS = {"teleop", "gravity-comp"}
_OP_IDS = {"teleop", "gravity-comp", "collect-data", "run-policy"}


class _StreamTee:
    """Mirror a stream to the original fd and emit each completed line."""

    def __init__(self, original: Any, sink: Any) -> None:
        self._original = original
        self._sink = sink
        self._buf = ""

    def write(self, s: str) -> int:
        try:
            self._original.write(s)
        except Exception:  # noqa: BLE001 - original may be closed
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._sink(line)
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False


class _SessionLogHandler(logging.Handler):
    """Logging handler that forwards formatted records into a session."""

    def __init__(self, sink: Any) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink(self.format(record))
        except Exception:  # noqa: BLE001
            pass


class _Capture:
    """Route ``logging`` + stdout/stderr into a session for one op's lifetime."""

    def __init__(self, session: Session, level: int) -> None:
        self._session = session
        self._level = level
        self._handler: _SessionLogHandler | None = None
        self._old_stdout: Any = None
        self._old_stderr: Any = None
        self._old_root_level: int | None = None

    def __enter__(self) -> _Capture:
        sink = self._session.emit
        self._handler = _SessionLogHandler(sink)
        self._handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        root = logging.getLogger()
        self._old_root_level = root.level
        root.setLevel(self._level)
        root.addHandler(self._handler)
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamTee(self._old_stdout, sink)
        sys.stderr = _StreamTee(self._old_stderr, sink)
        return self

    def __exit__(self, *_: object) -> None:
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        root = logging.getLogger()
        if self._handler is not None:
            root.removeHandler(self._handler)
        if self._old_root_level is not None:
            root.setLevel(self._old_root_level)


class OperationRunner:
    """Runs one core operation in-process at a time, with log capture."""

    def __init__(self, robot_link: Any = None) -> None:
        self._robot_link = robot_link
        self._lock = threading.Lock()
        self._session: Session | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # asyncio op plumbing (set while an async op runs).
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_task: asyncio.Task[Any] | None = None
        # run-policy episode control (set while run-policy runs).
        self._policy_control: Any = None
        # ZED orchestrator (set while an orchestrated collect/policy run runs).
        self._orch: Any = None

    # -- lookup / subscribe (mirrors SessionManager so app.py can reuse it) --

    def get(self, session_id: str) -> Session | None:
        s = self._session
        return s if s is not None and s.id == session_id else None

    def current(self) -> Session | None:
        return self._session

    def subscribe(self, session: Session) -> "asyncio.Queue[str | None]":
        q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=1000)
        session.subscribers.add(q)
        return q

    def unsubscribe(self, session: Session, q: "asyncio.Queue[str | None]") -> None:
        session.subscribers.discard(q)

    def is_running(self) -> bool:
        s = self._session
        return s is not None and s.status in ("starting", "running")

    # -- lifecycle ----------------------------------------------------------

    def start(
        self,
        op_id: str,
        args: dict[str, Any],
        zed: dict[str, Any] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> Session:
        if op_id not in _OP_IDS:
            raise KeyError(op_id)
        with self._lock:
            if self.is_running():
                raise RuntimeError("an operation is already running")
            session = Session(op_id, args)
            # The op runs on a worker thread; route subscriber wakeups back to
            # the server loop so the log WebSocket stays responsive.
            session.loop = loop
            self._session = session
            self._stop_event = threading.Event()

        orchestrate = (
            zed is not None
            and bool(zed.get("enabled"))
            and op_id in ("collect-data", "run-policy")
        )

        # Build the config up front for the non-orchestrated path so config
        # errors surface synchronously; the orchestrated path builds it inside
        # the worker (args are augmented with ZED network flags first).
        cfg: Any = None
        if not orchestrate:
            try:
                cfg = self._build_config(op_id, args)
            except Exception as exc:  # noqa: BLE001 - surface config errors to UI
                session.status = "error"
                session.error = f"{type(exc).__name__}: {exc}"
                session.emit(f"[serve] config error: {session.error}")
                session.close_stream()
                return session

        is_sim = op_id == "teleop" and bool(args.get("sim"))
        needs_robot = op_id in _HARDWARE_OPS and not is_sim
        log_level = self._log_level(args)

        session.status = "running"
        session.emit(f"[serve] starting {op_id} (in-process)")

        if needs_robot and self._robot_link is not None:
            session.emit("[serve] releasing robot link for task")
            try:
                self._robot_link.release()
            except Exception as exc:  # noqa: BLE001
                session.emit(f"[serve] robot release warning: {exc}")

        if orchestrate:
            target = self._run_orchestrated
            run_args = (session, op_id, args, zed, log_level, needs_robot)
        elif op_id in _ASYNC_OPS:
            target = self._run_async
            run_args = (session, op_id, cfg, log_level, needs_robot)
        else:
            target = self._run_thread
            run_args = (session, op_id, cfg, log_level, needs_robot)
        self._thread = threading.Thread(
            target=target, args=run_args, name=f"axol-op-{op_id}", daemon=True
        )
        self._thread.start()
        return session

    def stop(self) -> Session | None:
        session = self._session
        if session is None:
            return None
        session.emit("[serve] stopping…")
        self._stop_event.set()
        loop, task = self._async_loop, self._async_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        # Orchestrated run: tear down the ZED pipeline (covers a stop pressed
        # while still bringing PTP/streaming up, before the task starts).
        orch, orch_loop = self._orch, self._async_loop
        if orch is not None and orch_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(orch.stop(), orch_loop)
            except RuntimeError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=20.0)
        return self._session

    def episode_command(self, command: str) -> bool:
        """Forward a run-policy episode command (start/s/r/q) to its control."""
        control = self._policy_control
        if control is None:
            return False
        control.push(command)
        return True

    async def shutdown(self) -> None:
        if self.is_running():
            await asyncio.to_thread(self.stop)

    # -- config building ----------------------------------------------------

    def _build_config(self, op_id: str, args: dict[str, Any]) -> Any:
        from ..cli.config import parse
        from .commands import COMMANDS, build_argv

        config_class = COMMANDS[op_id].load()
        argv = build_argv(op_id, args)
        return parse(config_class, argv)

    def _log_level(self, args: dict[str, Any]) -> int:
        raw = str(args.get("log_level", "INFO")).upper()
        return getattr(logging, raw, logging.INFO)

    # -- async ops (teleop / gravity-comp) ----------------------------------

    def _run_async(
        self,
        session: Session,
        op_id: str,
        cfg: Any,
        log_level: int,
        needs_robot: bool,
    ) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._async_loop = loop

        async def _wrap() -> None:
            if op_id == "teleop":
                from ..cli.teleop import _run as core
            else:
                from ..cli.gravity_comp import _run as core
            await core(cfg)

        with _Capture(session, log_level):
            try:
                task = loop.create_task(_wrap())
                self._async_task = task
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                session.error = f"{type(exc).__name__}: {exc}"
                session.status = "error"
                session.emit(f"[serve] error: {session.error}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:  # noqa: BLE001
                    pass
                loop.close()
                self._async_loop = None
                self._async_task = None
        self._finish(session, needs_robot)

    # -- thread ops (collect-data / run-policy) -----------------------------

    def _run_thread(
        self,
        session: Session,
        op_id: str,
        cfg: Any,
        log_level: int,
        needs_robot: bool,
    ) -> None:
        with _Capture(session, log_level):
            try:
                if op_id == "collect-data":
                    from ..cli.collect_data import _run as core

                    core(cfg, stop_event=self._stop_event)
                else:
                    from ..cli.run_policy import _QueuePolicyControl
                    from ..cli.run_policy import _run as core

                    control = _QueuePolicyControl(self._stop_event)
                    self._policy_control = control
                    core(cfg, stop_event=self._stop_event, control=control)
            except Exception as exc:  # noqa: BLE001
                session.error = f"{type(exc).__name__}: {exc}"
                session.status = "error"
                session.emit(f"[serve] error: {session.error}")
            finally:
                self._policy_control = None
        self._finish(session, needs_robot)

    # -- orchestrated ops (collect-data / run-policy + ZED box) -------------

    def _run_orchestrated(
        self,
        session: Session,
        op_id: str,
        args: dict[str, Any],
        zed: dict[str, Any],
        log_level: int,
        needs_robot: bool,
    ) -> None:
        from .orchestrator import ZedOrchestrator

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._async_loop = loop

        async def run_main(main_args: dict[str, Any]) -> int:
            cfg = self._build_config(op_id, main_args)
            if op_id == "collect-data":
                from ..cli.collect_data import _run as core

                await asyncio.to_thread(core, cfg, self._stop_event)
            else:
                from ..cli.run_policy import _QueuePolicyControl
                from ..cli.run_policy import _run as core

                control = _QueuePolicyControl(self._stop_event)
                self._policy_control = control
                await asyncio.to_thread(core, cfg, self._stop_event, control)
            return 0

        orch = ZedOrchestrator(session, op_id, args, zed, run_main=run_main)
        self._orch = orch

        with _Capture(session, log_level):
            try:
                loop.run_until_complete(orch.run())
            except Exception as exc:  # noqa: BLE001
                session.error = f"{type(exc).__name__}: {exc}"
                session.status = "error"
                session.emit(f"[serve] error: {session.error}")
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:  # noqa: BLE001
                    pass
                loop.close()
                self._async_loop = None
                self._policy_control = None
                self._orch = None
        self._finish(session, needs_robot)

    # -- shared teardown ----------------------------------------------------

    def _finish(self, session: Session, needs_robot: bool) -> None:
        if session.status not in ("error",):
            session.status = "exited"
            session.exit_code = 0
        session.emit(f"[serve] {session.command_id} finished")
        session.close_stream()
        if needs_robot and self._robot_link is not None:
            try:
                self._robot_link.reacquire()
                session.emit("[serve] robot link reacquired")
            except Exception as exc:  # noqa: BLE001
                _logger.debug("robot reacquire failed: %s", exc)
