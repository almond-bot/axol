"""In-process runner for the core operations.

Unlike :class:`~almond_axol.serve.manager.SessionManager` (which spawns the
generic calibration/setup commands as ``axol <cmd>`` subprocesses), the core
operations — teleop, gravity-comp, collect-data, run-policy, replay-dataset —
run *inside* the serve process here, so they share the persistent robot
connection instead of opening their own from a child process.

Only one operation runs at a time. Its ``logging`` output and ``print``s are
captured into a :class:`~almond_axol.serve.manager.Session` ring buffer (the
same object the log WebSocket streams), so the UI sees live output exactly as
it did for subprocesses.

- teleop / gravity-comp are asyncio: they run on a dedicated event loop in a
  worker thread and are stopped by cancelling the task (both already tear down
  cleanly on ``CancelledError`` via their ``async with`` robot context).
- collect-data / run-policy / replay-dataset are blocking/threaded: they run on
  a worker thread and are stopped via a ``threading.Event`` (run-policy
  additionally takes a queue-backed episode control for save/rerecord/quit from
  the UI).

Before a hardware operation starts the runner releases the robot link's CAN
bus; when the operation ends it hands the bus back.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import re
import sys
import threading
from typing import Any

from ..zed import stereo_serials
from .manager import Session

_logger = logging.getLogger(__name__)

# How long a stop waits for the op to unwind cleanly before force-killing the
# op's child subprocesses, and how long it then waits for the freed-up worker
# thread to finish. The stop grace must exceed the ~30s soft-shutdown park cap
# plus planning margin: a stopping op may still be easing the arms down to
# zero, and force-killing its children mid-park would take out the IK worker
# planning the move (dropping the arms) and the dataset recorder mid-save
# (losing the episode) on every panel Stop. Past the grace the worker thread
# can still be stuck for *minutes* on a child subprocess with no stop check in
# between — either tearing down (``recorder.close()`` waits up to 180s for an
# in-flight save; the video relay / IK joins add more) or still starting up
# (it blocks in ``teleop.connect()`` on the IK worker's "ready" message, which
# only arrives after JAX finishes compiling the IK solver — a cold compile is
# minutes) — and the only reliable way to unstick those remains killing the
# children out from under it, which makes the blocked join/recv return.
_STOP_GRACE_S = 40.0
_FORCE_GRACE_S = 5.0

# Operations that need exclusive ownership of the CAN bus (everything except
# sim teleop, which is decided per-run from the ``sim`` arg).
_HARDWARE_OPS = {
    "teleop",
    "gravity-comp",
    "collect-data",
    "run-policy",
    "replay-dataset",
}
_ASYNC_OPS = {"teleop", "gravity-comp"}
_OP_IDS = {"teleop", "gravity-comp", "collect-data", "run-policy", "replay-dataset"}

# Loggers whose records we never forward to the UI: webserver lifecycle,
# access logs, low-level asyncio chatter. We still want the underlying ops'
# own logs (``almond_axol.*``, ``can.*``, lerobot, jaxls, pyroki, etc.).
_IGNORED_LOGGER_PREFIXES = (
    "uvicorn",
    "fastapi",
    "starlette",
    "watchfiles",
    "websockets",
    "httptools",
    "asyncio",
)

# ANSI escape sequences (colours, cursor moves). uvicorn colourises its logs,
# and various native tools emit them too; strip them before matching/emitting so
# the filter below works and the UI console isn't littered with raw ``\x1b[..``.
# The real terminal still gets the coloured originals via the tee's echo.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# uvicorn's DefaultFormatter / AccessFormatter writes lines like
# "INFO:     Started server process [...]"  or
# "INFO:     127.0.0.1:36514 - \"GET /api/robot/status HTTP/1.1\" 200 OK".
# Detect that distinctive ``LEVEL:<4+ spaces>`` prefix so the same lines that
# go to the actual terminal don't also pollute the op's session log. Matched
# against the ANSI-stripped line, since uvicorn colourises the ``LEVEL`` prefix.
_UVICORN_LINE = re.compile(r"^(INFO|WARNING|ERROR|DEBUG|CRITICAL|TRACE):\s{2,}")

# The camera slots the control panel can configure serials for.
_CAMERA_SLOTS = ("overhead", "left_arm", "right_arm")


def _forward_line(sink: Any, line: str) -> None:
    """Emit a completed output line to a session sink.

    Strips ANSI escapes (the UI console renders them as literal ``\\x1b[..``),
    then drops uvicorn's own access / lifecycle lines — those still reach the
    real terminal via the tee, but shouldn't pollute the op's session log.
    """
    line = _ANSI_ESCAPE.sub("", line)
    if _UVICORN_LINE.match(line):
        return
    try:
        sink(line)
    except Exception:  # noqa: BLE001 - a broken sink must never break output
        pass


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
            _forward_line(self._sink, line)
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False


class _SessionLogHandler(logging.Handler):
    """Logging handler that forwards formatted records into a session.

    Drops records from web-server / framework loggers (``uvicorn.*`` etc.)
    so an operation's log feed only contains output from the operation
    itself and the libraries it uses.
    """

    def __init__(self, sink: Any) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        name = record.name or ""
        for prefix in _IGNORED_LOGGER_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                return
        try:
            self._sink(self.format(record))
        except Exception:  # noqa: BLE001
            pass


class _Capture:
    """Route ``logging`` + stdout/stderr into a session for one op's lifetime.

    Two capture layers run together:

    * Python-level ``logging``/``print`` go through a :class:`_StreamTee` that
      writes to a *saved dup* of the original fd (so the real terminal/journald
      still see them) and forwards each line to the session.
    * OS file descriptors 1 and 2 are redirected to a pipe drained by a reader
      thread. This is the only way to capture output that bypasses Python's
      ``sys.stdout``/``sys.stderr`` objects: native libraries (the ZED SDK,
      GStreamer, CUDA) writing straight to fd 2, and the spawned video-relay /
      recorder / encoder subprocesses that inherit these fds. Without it a
      camera/ZED failure only reaches the service log and the UI shows a bare
      "exited" with no reason.

    Because the tee writes Python output to the saved fd (never fd 1/2) and the
    reader drains only fd 1/2, a line is captured by exactly one path — no
    double emission. The redirect is installed before the op spawns its children
    so they inherit the pipe; if it can't be set up we fall back to Python-only
    capture (the prior behaviour).
    """

    def __init__(self, session: Session, level: int) -> None:
        self._session = session
        self._level = level
        self._handler: _SessionLogHandler | None = None
        self._old_stdout: Any = None
        self._old_stderr: Any = None
        self._old_root_level: int | None = None
        # fd-level tee state (None when unavailable / not installed).
        self._saved_out_fd: int | None = None
        self._saved_err_fd: int | None = None
        self._saved_out: Any = None
        self._saved_err: Any = None
        self._reader: threading.Thread | None = None

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
        try:
            self._install_fd_tee(sink)
        except Exception:  # noqa: BLE001 - degrade to Python-only capture
            _logger.exception(
                "fd-level log capture unavailable; native/child-process output "
                "won't reach the UI log"
            )
            sys.stdout = _StreamTee(self._old_stdout, sink)
            sys.stderr = _StreamTee(self._old_stderr, sink)
        return self

    def _install_fd_tee(self, sink: Any) -> None:
        # Save the real fds so we can keep echoing to the terminal/journald and
        # restore them on exit.
        self._saved_out_fd = os.dup(1)
        self._saved_err_fd = os.dup(2)
        pipe_r, pipe_w = os.pipe()
        # Start the drainer *before* redirecting fd 1/2, so the pipe always has a
        # reader — otherwise a failure between the redirect and the thread start
        # could let the pipe fill and wedge the whole serve process.
        self._reader = threading.Thread(
            target=self._drain_pipe,
            args=(pipe_r, self._saved_err_fd, sink),
            name=f"log-tee-{self._session.id}",
            daemon=True,
        )
        self._reader.start()
        # Point the process's fd 1/2 at the pipe: everything writing to them from
        # now on — native code and inherited child processes — lands there.
        os.dup2(pipe_w, 1)
        os.dup2(pipe_w, 2)
        os.close(pipe_w)
        # Line-buffered text wrappers over the *saved* fds. Python-level writes
        # go here (real terminal + session) and never touch the pipe, so a
        # print/log line isn't captured twice. closefd=False: we own the fds.
        self._saved_out = os.fdopen(self._saved_out_fd, "w", buffering=1, closefd=False)
        self._saved_err = os.fdopen(self._saved_err_fd, "w", buffering=1, closefd=False)
        sys.stdout = _StreamTee(self._saved_out, sink)
        sys.stderr = _StreamTee(self._saved_err, sink)

    @staticmethod
    def _drain_pipe(pipe_r: int, echo_fd: int, sink: Any) -> None:
        """Read fd-level output, echo it to the terminal, forward lines."""
        buf = ""
        try:
            while True:
                chunk = os.read(pipe_r, 65536)
                if not chunk:
                    break
                # Echo the raw bytes to the saved fd so the service log is
                # unchanged; os.write (not the buffered stream) avoids racing
                # the tee's writes to the same fd from the op thread.
                try:
                    os.write(echo_fd, chunk)
                except OSError:
                    pass
                buf += chunk.decode("utf-8", "replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    _forward_line(sink, line)
        except OSError:
            pass
        finally:
            if buf:
                _forward_line(sink, buf)
            try:
                os.close(pipe_r)
            except OSError:
                pass

    def __exit__(self, *_: object) -> None:
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr
        root = logging.getLogger()
        if self._handler is not None:
            root.removeHandler(self._handler)
        if self._old_root_level is not None:
            root.setLevel(self._old_root_level)
        self._teardown_fd_tee()

    def _teardown_fd_tee(self) -> None:
        for stream in (self._saved_out, self._saved_err):
            if stream is not None:
                try:
                    stream.flush()
                except Exception:  # noqa: BLE001
                    pass
        # Restore the real fds. This drops the serve process's own write ends of
        # the pipe; once the op's children (which inherited them) are gone the
        # reader hits EOF and exits. Their teardown already ran in the op body.
        if self._saved_out_fd is not None:
            try:
                os.dup2(self._saved_out_fd, 1)
            except OSError:
                pass
        if self._saved_err_fd is not None:
            try:
                os.dup2(self._saved_err_fd, 2)
            except OSError:
                pass
        # Wait briefly for the reader to drain; it's a daemon, so a lingering
        # child can't wedge shutdown.
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        for fd in (self._saved_out_fd, self._saved_err_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._saved_out = self._saved_err = None
        self._saved_out_fd = self._saved_err_fd = None
        self._reader = None


class OperationRunner:
    """Runs one core operation in-process at a time, with log capture."""

    def __init__(self, robot_link: Any = None, settings: Any = None) -> None:
        self._robot_link = robot_link
        # Shared operator settings (serve.settings.SettingsStore). Folded into
        # every op start beneath the request's own args, so per-run values win.
        self._settings = settings
        self._lock = threading.Lock()
        self._session: Session | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Set once a stop has been kicked off for the current op, so a second
        # Stop click (or a stop racing the op's own exit) is a no-op.
        self._stopping = False
        # multiprocessing children that already existed when the op started, so
        # the force-kill only targets subprocesses this op spawned (relay,
        # recorder, IK worker) and never anything the serve process owns.
        self._baseline_children: set[int] = set()
        # asyncio op plumbing (set while an async op runs).
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_task: asyncio.Task[Any] | None = None
        # run-policy episode control (set while run-policy runs).
        self._policy_control: Any = None

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
        # "stopping" still counts as running: the op owns the CAN bus until its
        # worker thread unwinds, so a new op must not start until it's gone.
        s = self._session
        return s is not None and s.status in ("starting", "running", "stopping")

    # -- lifecycle ----------------------------------------------------------

    def start(
        self,
        op_id: str,
        args: dict[str, Any],
        cameras: dict[str, Any] | None = None,
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
            self._stopping = False
            # Snapshot pre-existing children so a later force-kill only targets
            # the subprocesses this op is about to spawn.
            self._baseline_children = {c.pid for c in multiprocessing.active_children()}

        # Fold the shared settings and camera spec in and build the config up
        # front, so every config error — a bad stored value as much as a bad
        # request arg — surfaces synchronously as a failed session instead of
        # an unhandled 500 that would leave this session wedged in "starting".
        try:
            # Shared operator settings go beneath the request's args (per-run
            # values win); the stored camera spec is the fallback when the
            # request didn't carry one (older UIs still send it explicitly).
            if self._settings is not None:
                args = self._settings.merged_args(op_id, args)
                if cameras is None:
                    cameras = self._settings.cameras()

            # Fold the camera spec into the argv-style args for collect-data /
            # run-policy (their camera serials are required draccus inputs).
            if cameras and op_id in ("collect-data", "run-policy"):
                args = self._merge_camera_args(args, cameras)

            cfg = self._build_config(op_id, args)
        except Exception as exc:  # noqa: BLE001 - surface config errors to UI
            session.status = "error"
            session.error = f"{type(exc).__name__}: {exc}"
            session.emit(f"[serve] config error: {session.error}")
            session.close_stream()
            return session

        # Teleop relays the local ZED cameras to the headset when serials are
        # configured (its ``cameras`` dict isn't reachable via flat argv).
        # Best-effort: camera streaming is an optional add-on for teleop, and
        # the spec is now always present via the settings store — a host that
        # can't apply it (e.g. no ZED stack on a dev machine running sim) still
        # gets a camera-less teleop instead of a failed start.
        if op_id == "teleop":
            try:
                self._attach_cameras_to_teleop(cfg, cameras, session)
            except Exception as exc:  # noqa: BLE001
                session.emit(
                    f"[serve] teleop: camera streaming unavailable ({exc}); "
                    "continuing without cameras"
                )

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

        if op_id in _ASYNC_OPS:
            target = self._run_async
        else:
            target = self._run_thread
        run_args = (session, op_id, cfg, log_level, needs_robot)
        self._thread = threading.Thread(
            target=target, args=run_args, name=f"axol-op-{op_id}", daemon=True
        )
        self._thread.start()
        return session

    def stop(self) -> Session | None:
        """Begin stopping the current op and return immediately.

        Signals the op to unwind and hands off to a background watchdog that
        force-kills the op's child subprocesses if it doesn't exit within the
        grace period — so the HTTP request never blocks on a slow teardown
        (e.g. a 180s dataset save). The session goes to ``stopping`` now and to
        ``exited`` once the worker thread finishes (or is abandoned).
        """
        session = self._session
        if session is None:
            return None
        if not self.is_running():
            # Already finished (or finishing): nothing to stop, and we must not
            # flip a terminal session back to "stopping".
            return session
        if not self._begin_stop(session):
            return session
        thread = self._thread
        watchdog = threading.Thread(
            target=self._await_stop,
            args=(session, thread),
            name="axol-op-stop",
            daemon=True,
        )
        watchdog.start()
        return session

    def _begin_stop(self, session: Session) -> bool:
        """Signal the op to unwind. Returns ``False`` if there's nothing to stop.

        The status check + flip happen under the lock, and the terminal
        transitions (:meth:`_mark_terminal`) take the same lock, so a Stop that
        races the op's own exit can't resurrect a finished session as
        ``stopping``: either we flip a still-running op to ``stopping`` (and a
        later ``_finish`` moves it to ``exited``), or the op already reached a
        terminal state and we bail.
        """
        with self._lock:
            if self._stopping:
                return False
            if session.status not in ("starting", "running"):
                return False
            self._stopping = True
            session.status = "stopping"
        session.emit("[serve] stopping…")
        self._stop_event.set()
        loop, task = self._async_loop, self._async_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        return True

    def _await_stop(self, session: Session, thread: threading.Thread | None) -> None:
        """Wait for the op to exit, force-killing its children if it stalls."""
        if thread is None:
            return
        thread.join(timeout=_STOP_GRACE_S)
        if thread.is_alive():
            session.emit(
                f"[serve] still stopping after {_STOP_GRACE_S:.0f}s — "
                "force-killing the operation's child processes"
            )
            self._kill_op_children(session)
            thread.join(timeout=_FORCE_GRACE_S)
        if thread.is_alive():
            # The thread is stuck on something we can't kill (an in-process
            # native call); leave it abandoned rather than block forever. It
            # still owns the CAN bus, so the session stays "running"/"stopping"
            # and is_running() keeps a new op from clobbering it.
            session.emit(
                "[serve] operation did not exit after force-kill and is now "
                "abandoned — restart axol serve if it persists"
            )

    def _kill_op_children(self, session: Session) -> None:
        """SIGKILL every subprocess this op spawned (relay, recorder, IK worker).

        The worker thread blocks on these children either while tearing down
        (``recorder.close()`` / ``relay.shutdown()`` / ``teleop.disconnect()``
        joins) or while starting up (waiting on the IK worker's "ready" message
        across the pipe while it compiles JAX). Killing the children makes the
        blocked join/recv return so the thread can finish.
        """
        # Snapshot the targets under the lock and only if this is still the
        # op being stopped: a slow watchdog could otherwise wake after the
        # stopped worker exited and a *new* op started, and SIGKILL the new
        # op's subprocesses. ``start()`` swaps ``_session`` / ``_baseline_children``
        # under the same lock, so either we see the old op (and its children) or
        # we see the swap and bail — never a mix.
        with self._lock:
            if self._session is not session:
                return
            targets = [
                c
                for c in multiprocessing.active_children()
                if c.pid not in self._baseline_children
            ]
        for child in targets:
            session.emit(
                f"[serve] killing child process {child.name} (pid {child.pid})"
            )
            try:
                child.kill()
            except Exception as exc:  # noqa: BLE001 - best-effort
                session.emit(f"[serve] failed to kill pid {child.pid}: {exc}")

    def episode_command(self, command: str) -> bool:
        """Forward a run-policy episode command (start/s/r/q) to its control."""
        control = self._policy_control
        if control is None:
            return False
        control.push(command)
        return True

    def policy_state(self) -> dict[str, Any] | None:
        """run-policy episode phase/message/count, or None if no policy is running.

        Read by /api/op/status so the control panel reflects whether an episode
        is recording or sitting at the between-episode gate on any computer.
        """
        control = self._policy_control
        return control.snapshot() if control is not None else None

    async def shutdown(self) -> None:
        # Server shutdown: stop and block until the op is actually gone (unlike
        # the API stop, which returns early and lets the watchdog finish).
        if self.is_running():
            await asyncio.to_thread(self._stop_blocking)

    def _stop_blocking(self) -> None:
        session = self._session
        thread = self._thread
        if session is None:
            return
        self._begin_stop(session)
        self._await_stop(session, thread)

    # -- config building ----------------------------------------------------

    @staticmethod
    def _camera_serials(cameras: dict[str, Any] | None) -> dict[str, int]:
        """Valid ``slot -> serial`` pairs from a camera spec (empty if none)."""
        serials: dict[str, int] = {}
        for slot, raw in ((cameras or {}).get("serials") or {}).items():
            if slot not in _CAMERA_SLOTS:
                continue
            try:
                serial = int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if serial > 0:
                serials[slot] = serial
        return serials

    @staticmethod
    def _resolution(
        cameras: dict[str, Any] | None, key: str, legacy: str | None = None
    ) -> str | None:
        """Resolution name for a branch, or ``None`` when off/unset.

        A value of ``"off"`` (or empty) disables the whole branch (streaming or
        recording). ``legacy`` reads the old single ``resolution`` key as the
        streaming resolution for back-compat.
        """
        from ..lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS

        val = (cameras or {}).get(key)
        if val is None and legacy:
            val = (cameras or {}).get(legacy)
        val = str(val or "").strip()
        if not val or val.lower() == "off":
            return None
        if val not in ZED_RESOLUTION_DIMS:
            raise ValueError(f"unknown ZED resolution {val!r}")
        return val

    @staticmethod
    def _branch(
        cameras: dict[str, Any] | None, key: str, slot: str, global_on: bool
    ) -> tuple[bool, str | None]:
        """``(enabled, eyes)`` for one camera in one branch (stream / record).

        ``cameras[key][slot]`` is per-camera participation: ``False`` / ``"off"``
        opts the camera out; an eye name (``"both"`` / ``"left"`` / ``"right"``)
        opts a stereo camera in with that eye selection; ``True`` (or absent,
        the default) opts a mono camera in. The branch is only enabled when its
        global resolution is set (``global_on``).
        """
        raw = ((cameras or {}).get(key) or {}).get(slot, True)
        enabled = raw not in (False, None, "off", "")
        eyes = raw if raw in ("both", "left", "right") else None
        return (global_on and enabled), eyes

    def _merge_camera_args(
        self, args: dict[str, Any], cameras: dict[str, Any]
    ) -> dict[str, Any]:
        """Fold a camera spec into collect-data / run-policy form args.

        Serials, capture resolution, stereo flag, per-branch enable, and per-eye
        selection map to dotted ``robot_config.cameras.*`` keys the command
        schema emits, so the result is parsed like any CLI override (those fields
        are hidden from the UI's generated form — the Cameras dialog owns them).
        Whether a slot is stereo is auto-detected from its serial (see
        :func:`almond_axol.zed.stereo_serials`).

        Streaming (the headset) and recording (the dataset) are configured
        independently, globally and per camera:

        - ``stream_resolution`` / ``record_resolution`` set the capture and
          dataset-downscale resolutions; ``"off"`` (or unset) disables that whole
          branch. ``record_resolution`` maps to collect-data's top-level
          ``dataset_resolution`` (run-policy has no such field; ``build_argv``
          drops the unknown key there).
        - ``stream`` / ``record`` are per-slot maps deciding which cameras (and,
          for stereo, which eyes) take part — mapping to ``cameras.<slot>.stream``
          / ``.record`` plus ``.stream_eyes`` (headset) and ``.eyes`` (dataset).

        A camera that takes part in neither branch is omitted entirely. Capture
        resolution per camera is the streaming resolution when it streams, else
        the recording resolution (no point grabbing larger than it records).
        """
        from ..lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS

        merged = dict(args)
        serials = self._camera_serials(cameras)
        stream_res = self._resolution(cameras, "stream_resolution", legacy="resolution")
        record_res = self._resolution(cameras, "record_resolution")
        detected = stereo_serials()

        for slot, serial in serials.items():
            streams, s_eyes = self._branch(
                cameras, "stream", slot, stream_res is not None
            )
            records, r_eyes = self._branch(
                cameras, "record", slot, record_res is not None
            )
            if not (streams or records):
                continue  # disabled in both branches → don't open this camera
            prefix = f"robot_config.cameras.{slot}"
            merged[f"{prefix}.serial"] = serial
            merged[f"{prefix}.stream"] = streams
            merged[f"{prefix}.record"] = records
            if serial in detected:
                # Flag stereo explicitly so apply_detected_stereo leaves our eye
                # selection untouched. Recorded eyes default to the head/wrist
                # policy when omitted; stream eyes default to follow them.
                merged[f"{prefix}.stereo"] = True
                merged[f"{prefix}.eyes"] = r_eyes or (
                    "both" if slot == "overhead" else "left"
                )
                if s_eyes is not None:
                    merged[f"{prefix}.stream_eyes"] = s_eyes
            # Capture at the streaming resolution when the camera streams,
            # otherwise at the (smaller) recording resolution.
            cap = stream_res if streams else record_res
            if cap:
                dims = ZED_RESOLUTION_DIMS[cap]
                merged[f"{prefix}.width"] = dims[0]
                merged[f"{prefix}.height"] = dims[1]

        if record_res is not None:
            merged["dataset_resolution"] = record_res
        return merged

    def _attach_cameras_to_teleop(
        self, cfg: Any, cameras: dict[str, Any] | None, session: Session
    ) -> None:
        """Point teleop's headset video at the configured local ZED cameras.

        Teleop only streams (no recording), so only the streaming branch applies:
        cameras opted out of streaming (or all, when streaming is globally off)
        are left off the headset feed. Teleop's ``cameras`` / ``camera_eyes``
        fields are dicts (not reachable via flat argv), so they're written onto
        the built config directly.
        """
        serials = self._camera_serials(cameras)
        if not serials:
            return
        stream_res = self._resolution(cameras, "stream_resolution", legacy="resolution")
        detected = stereo_serials()
        cam_map: dict[str, int] = {}
        camera_eyes: dict[str, str] = {}
        for slot, serial in serials.items():
            streams, s_eyes = self._branch(
                cameras, "stream", slot, stream_res is not None
            )
            if not streams:
                continue
            cam_map[slot] = serial
            if serial in detected and s_eyes is not None:
                camera_eyes[slot] = s_eyes
        if not cam_map:
            session.emit("[serve] teleop: streaming disabled (no cameras)")
            return
        cfg.cameras = cam_map
        if camera_eyes:
            cfg.camera_eyes = camera_eyes
        if stream_res:
            cfg.resolution = stream_res
        stereo_slots = sorted(s for s in cam_map if cam_map[s] in detected)
        stereo_note = f" (stereo: {', '.join(stereo_slots)})" if stereo_slots else ""
        resolution_note = f" @ {stream_res}" if stream_res else ""
        session.emit(
            "[serve] teleop: streaming cameras to the headset "
            f"({', '.join(sorted(cam_map))}){stereo_note}{resolution_note}"
        )

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
                self._mark_terminal(
                    session, "error", error=f"{type(exc).__name__}: {exc}"
                )
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

    # -- thread ops (collect-data / run-policy / replay-dataset) ------------

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
                elif op_id == "replay-dataset":
                    from ..cli.replay_dataset import _run as core

                    core(cfg, stop_event=self._stop_event)
                else:
                    from ..cli.run_policy import _QueuePolicyControl
                    from ..cli.run_policy import _run as core

                    control = _QueuePolicyControl(self._stop_event)
                    self._policy_control = control
                    core(cfg, stop_event=self._stop_event, control=control)
            except Exception as exc:  # noqa: BLE001
                self._mark_terminal(
                    session, "error", error=f"{type(exc).__name__}: {exc}"
                )
                session.emit(f"[serve] error: {session.error}")
            finally:
                self._policy_control = None
        self._finish(session, needs_robot)

    # -- shared teardown ----------------------------------------------------

    def _mark_terminal(
        self, session: Session, status: str, *, error: str | None = None
    ) -> None:
        """Move a session to a terminal state (``exited`` / ``error``).

        Taken under the lock so it can't interleave with :meth:`_begin_stop`'s
        running-check-then-flip. Never downgrades an existing ``error`` (the
        first failure wins, and a stop arriving after an error stays ``error``).
        """
        with self._lock:
            if session.status == "error":
                return
            if error is not None:
                session.error = error
            session.status = status
            if status == "exited":
                session.exit_code = 0

    def _finish(self, session: Session, needs_robot: bool) -> None:
        self._mark_terminal(session, "exited")
        session.emit(f"[serve] {session.command_id} finished")
        session.close_stream()
        if needs_robot and self._robot_link is not None:
            try:
                self._robot_link.reacquire()
                session.emit("[serve] robot link reacquired")
            except Exception as exc:  # noqa: BLE001
                _logger.debug("robot reacquire failed: %s", exc)
