"""Name the thread that stalled, while it is still stalled.

The recording stack fails closed on any long pause: a 120 Hz control tick that
does not publish robot state for ~100 ms, or a recorder capture row that does
not drain its encoded-frame queue for ~2 s, ends the episode. The validators
that do so are correct but blind — they report *that* a stall happened, never
*what the stalled thread was doing* or *why it lost the CPU*. This module fills
that gap with two cheap, dependency-free tools:

* :class:`StallWatchdog` — a heartbeat monitor for one hot thread. The watched
  thread calls :meth:`StallWatchdog.beat` once per tick/row. A monitor thread
  polls the heartbeat age and, the moment it exceeds the threshold, logs the
  stalled thread's Python stack together with its kernel scheduling state from
  ``/proc/self/task/<tid>`` (``R`` runnable-but-not-running = CPU starvation,
  ``D`` = blocked on disk I/O, ``S`` = sleeping inside a C call or lock) and the
  runnable-wait time it accrued during the stall. When the beat resumes, the
  stall's total duration and how much of it was spent runnable are logged.

  A stop-the-world pause (a gen-2 garbage collection, or a C extension holding
  the GIL) freezes the monitor too, so it cannot report *during* such a stall.
  It notices instead that its own sleep overran and reports the stall, when it
  ends, as a process-wide freeze — which is the distinguishing fact.

* :func:`install_gc_pause_logger` — a ``gc.callbacks`` hook that logs every
  cyclic-GC pass longer than a threshold, with its generation and duration.
  Full collections have been measured at ~100 ms in the video relay and ~500 ms
  in run-policy on the Jetson; both disable the collector for exactly that
  reason, and this hook shows whether the control or recorder process is
  paying the same price.

* :class:`GcHold` — the discipline those two flows already apply, packaged:
  collect once while nothing time-critical runs, then hold automatic
  collection for the duration of a take and sweep the deferred garbage
  afterwards (logging how long the sweep took: a multi-hundred-ms sweep is the
  pause that would otherwise have landed mid-take).

Everything here is Linux-oriented but degrades to "no kernel state" elsewhere.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
import time
import traceback
from typing import Callable

_logger = logging.getLogger(__name__)

# Innermost frames kept when logging a stalled thread's stack.
_STACK_FRAMES = 14


def thread_sched_state(native_id: int | None) -> dict[str, object]:
    """Kernel scheduling facts for one thread of this process.

    Returns ``{"state", "wchan", "run_ns", "wait_ns"}`` read from
    ``/proc/self/task/<tid>/{stat,wchan,schedstat}``. Missing files (non-Linux,
    thread exited) leave the corresponding keys out so callers can print what
    they got.
    """
    out: dict[str, object] = {}
    if native_id is None:
        return out
    base = f"/proc/self/task/{native_id}"
    try:
        with open(f"{base}/stat") as fh:
            data = fh.read()
        rp = data.rfind(")")
        if rp >= 0:
            fields = data[rp + 2 :].split()
            if fields:
                out["state"] = fields[0]
    except OSError:
        pass
    try:
        with open(f"{base}/wchan") as fh:
            wchan = fh.read().strip()
        if wchan and wchan != "0":
            out["wchan"] = wchan
    except OSError:
        pass
    try:
        with open(f"{base}/schedstat") as fh:
            parts = fh.read().split()
        out["run_ns"] = int(parts[0])
        out["wait_ns"] = int(parts[1])
    except (OSError, IndexError, ValueError):
        pass
    return out


def format_thread_stack(ident: int | None, limit: int = _STACK_FRAMES) -> str:
    """The innermost ``limit`` frames of thread ``ident``, or a note if gone."""
    if ident is None:
        return "  <thread not bound>"
    frame = sys._current_frames().get(ident)
    if frame is None:
        return "  <thread has no live frame>"
    lines = traceback.format_stack(frame, limit=limit)
    return "".join(lines).rstrip()


def _describe_state(sched: dict[str, object]) -> str:
    state = sched.get("state")
    if state is None:
        return "state=?"
    wchan = sched.get("wchan")
    tag = {
        "R": "runnable (waiting for a CPU, or on-CPU)",
        "D": "uninterruptible sleep (disk/DMA/page fault)",
        "S": "sleeping (blocked in a C call, lock, or syscall)",
        "T": "stopped",
        "Z": "zombie",
    }.get(str(state), "")
    text = f"state={state}"
    if tag:
        text += f" [{tag}]"
    if wchan:
        text += f" wchan={wchan}"
    return text


class StallWatchdog:
    """Heartbeat monitor that attributes a stall of one hot thread.

    Args:
        name: Label for the watched work (``"control tick"``, ``"recorder
            row"``); every log line starts with it.
        threshold_s: Heartbeat age at which the thread counts as stalled.
        logger: Where to report; defaults to this module's logger.
        poll_s: Monitor wake-up period. Defaults to a quarter of the threshold
            (never below 2 ms), so a stall is caught within ~25 % of the
            threshold and the monitor costs a few wake-ups per tick at most.

    The watched thread must call :meth:`beat` regularly; the first call binds
    the watchdog to that thread. :meth:`suspend` parks the monitor while the
    thread is legitimately idle (between episodes, during a save the loop
    deliberately waits out); :meth:`resume` re-arms it with a fresh heartbeat.
    """

    def __init__(
        self,
        name: str,
        threshold_s: float,
        *,
        logger: logging.Logger | None = None,
        poll_s: float | None = None,
    ) -> None:
        if threshold_s <= 0:
            raise ValueError("threshold_s must be positive")
        self.name = name
        self.threshold_s = float(threshold_s)
        self._poll_s = (
            max(0.002, self.threshold_s / 4.0) if poll_s is None else float(poll_s)
        )
        self._logger = logger or _logger
        self._last_beat = time.perf_counter()
        self._ident: int | None = None
        self._native_id: int | None = None
        self._suspended = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.stalls = 0
        self.longest_stall_s = 0.0

    # -- watched-thread side ------------------------------------------------

    def bind(self) -> None:
        """Bind to the calling thread without counting a heartbeat."""
        self._ident = threading.get_ident()
        self._native_id = threading.get_native_id()

    def beat(self) -> None:
        """Record liveness; call once per tick/row from the watched thread."""
        if self._ident is None:
            self.bind()
        self._last_beat = time.perf_counter()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "StallWatchdog":
        with self._lock:
            if self._thread is not None:
                return self
            self._stop.clear()
            self._last_beat = time.perf_counter()
            self._thread = threading.Thread(
                target=self._run, name=f"stall-watchdog:{self.name}", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=max(1.0, 4 * self._poll_s))

    def suspend(self) -> None:
        """Stop judging heartbeats until :meth:`resume` (thread idles on purpose)."""
        self._suspended.set()

    def resume(self) -> None:
        """Re-arm after :meth:`suspend`, treating now as a fresh heartbeat."""
        self._last_beat = time.perf_counter()
        self._suspended.clear()

    # -- monitor ------------------------------------------------------------

    def _run(self) -> None:
        in_stall = False
        stall_started = 0.0
        stall_sched: dict[str, object] = {}
        frozen_with_it = False
        prev_wake = time.perf_counter()
        prev_beat = self._last_beat
        while not self._stop.wait(self._poll_s):
            now = time.perf_counter()
            monitor_gap = now - prev_wake
            prev_wake = now
            if self._suspended.is_set():
                in_stall = False
                prev_beat = self._last_beat
                continue
            last = self._last_beat
            age = now - last
            if not in_stall:
                if (
                    monitor_gap > self.threshold_s
                    and last > prev_beat
                    and last - prev_beat > self.threshold_s
                ):
                    # A heartbeat gap began and ended while this monitor was
                    # itself asleep past the threshold: both threads were
                    # frozen together, so the gap is only visible after the
                    # fact — and it is process-wide by construction.
                    self.stalls += 1
                    self.longest_stall_s = max(self.longest_stall_s, last - prev_beat)
                    self._logger.warning(
                        "%s: heartbeat gap of %.0f ms passed while the monitor "
                        "was frozen for %.0f ms: process-wide pause (gen-2 GC "
                        "or a C call holding the GIL), not a blocked thread",
                        self.name,
                        (last - prev_beat) * 1e3,
                        monitor_gap * 1e3,
                    )
                if age > self.threshold_s:
                    in_stall = True
                    stall_started = last
                    self.stalls += 1
                    stall_sched = thread_sched_state(self._native_id)
                    # A healthy monitor notices within one poll of the
                    # threshold. Noticing much later means its own sleep
                    # overran: the whole process was frozen (gen-2 GC, a C
                    # call holding the GIL, or the process descheduled), and
                    # the stack below shows where the thread *resumed*, not
                    # where it waited.
                    frozen_with_it = age > self.threshold_s + 2 * self._poll_s
                    self._logger.warning(
                        "%s: no heartbeat for %.0f ms%s; %s\n%s",
                        self.name,
                        age * 1e3,
                        " (monitor overslept too: process-wide pause, e.g. "
                        "gen-2 GC or a C call holding the GIL)"
                        if frozen_with_it
                        else "",
                        _describe_state(stall_sched),
                        format_thread_stack(self._ident),
                    )
                prev_beat = last
                continue
            if last > stall_started:
                in_stall = False
                duration = last - stall_started
                self.longest_stall_s = max(self.longest_stall_s, duration)
                after = thread_sched_state(self._native_id)
                detail = ""
                if "wait_ns" in after and "wait_ns" in stall_sched:
                    waited = (int(after["wait_ns"]) - int(stall_sched["wait_ns"])) / 1e6
                    ran = (int(after["run_ns"]) - int(stall_sched["run_ns"])) / 1e6
                    detail = (
                        f" (since the report: runnable-but-waiting {waited:.0f} ms,"
                        f" on-CPU {ran:.0f} ms)"
                    )
                self._logger.warning(
                    "%s: heartbeat resumed after %.0f ms%s%s",
                    self.name,
                    duration * 1e3,
                    detail,
                    "; the monitor was frozen with it" if frozen_with_it else "",
                )
            prev_beat = last


def install_gc_pause_logger(
    logger: logging.Logger | None = None, min_ms: float = 20.0
) -> Callable[[], None]:
    """Log every cyclic-GC pass that pauses the process for ``min_ms`` or more.

    Returns a callable that uninstalls the hook. The callback runs on whichever
    thread triggered the collection, after the world has already resumed, so it
    costs the hot path nothing beyond the log call itself.
    """
    log = logger or _logger
    started: dict[str, float] = {}

    def _on_gc(phase: str, info: dict) -> None:
        if phase == "start":
            started["t0"] = time.perf_counter()
            return
        t0 = started.pop("t0", None)
        if t0 is None:
            return
        pause_ms = (time.perf_counter() - t0) * 1e3
        if pause_ms >= min_ms:
            log.warning(
                "gc: generation-%d collection paused every thread for %.0f ms "
                "(%d objects collected)",
                info.get("generation", -1),
                pause_ms,
                info.get("collected", 0),
            )

    gc.callbacks.append(_on_gc)

    def _uninstall() -> None:
        try:
            gc.callbacks.remove(_on_gc)
        except ValueError:
            pass

    return _uninstall


class GcHold:
    """Hold automatic cyclic GC across a time-critical window.

    :meth:`begin` sweeps now (while nothing is moving) and disables automatic
    collection; :meth:`end` re-enables it and sweeps the garbage deferred during
    the window, logging the sweep's duration — the pause that would otherwise
    have landed mid-take. Re-entrant-safe: nested ``begin`` calls are no-ops,
    ``end`` without a matching ``begin`` is a no-op, and ``end`` never leaves
    the collector disabled even if the sweep raises. Refcounting still frees
    the acyclic per-tick objects (dicts, numpy views, byte strings) immediately;
    only reference cycles wait for the sweep.
    """

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        self.name = name
        self._logger = logger or _logger
        self._held = False
        self._was_enabled = True

    @property
    def held(self) -> bool:
        return self._held

    def begin(self, *, collect: bool = True) -> None:
        """Hold automatic collection; ``collect=False`` skips the up-front sweep.

        Skip it when the caller already swept at the end of the previous window
        and a full collection here would itself be a visible hitch (a control
        loop that is still moving the robot when the window opens).
        """
        if self._held:
            return
        self._was_enabled = gc.isenabled()
        t0 = time.perf_counter()
        if collect:
            gc.collect()
        gc.disable()
        self._held = True
        self._logger.debug(
            "%s: automatic collection held (pre-window sweep %s, %.0f ms)",
            self.name,
            "done" if collect else "skipped",
            (time.perf_counter() - t0) * 1e3,
        )

    def end(self) -> None:
        if not self._held:
            return
        self._held = False
        if self._was_enabled:
            gc.enable()
        t0 = time.perf_counter()
        try:
            collected = gc.collect()
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1e3
        self._logger.info(
            "%s: deferred gc.collect swept %d objects in %.0f ms after the window",
            self.name,
            collected,
            elapsed_ms,
        )


def freeze_startup_heap() -> int:
    """Move everything allocated so far out of the collector's reach.

    Call once after imports and connection setup: the modules, robot model and
    kinematics graphs are permanent for the session, and a full collection
    that has to traverse them is what makes gen-2 pauses long. Returns the
    number of objects frozen. Pair with :func:`unfreeze_heap` when the caller
    is an operation inside a longer-lived process (``axol serve``) so the host
    process's collector sees its heap again afterwards.
    """
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()


def unfreeze_heap() -> None:
    """Return frozen objects to the collector (inverse of :func:`freeze_startup_heap`)."""
    gc.unfreeze()


__all__ = [
    "GcHold",
    "StallWatchdog",
    "format_thread_stack",
    "freeze_startup_heap",
    "install_gc_pause_logger",
    "thread_sched_state",
    "unfreeze_heap",
]
