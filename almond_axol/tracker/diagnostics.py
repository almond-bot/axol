"""Bounded, best-effort traces around Lighthouse tracking failures.

Native instrumentation writes atomic NDJSON packets to a separate nonblocking
pipe. The collector drains that pipe independently of pose delivery. Public
record/trigger methods only enqueue work; compression and disk I/O run on a
separate writer thread. Diagnostic failure must never change tracking policy.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import queue
import select
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NATIVE_FD_ENV = "AXOL_SURVIVE_DIAGNOSTIC_FD"


class FaultDiagnosticRecorder:
    """Keep recent observations, then save pre/post windows after a fault.

    ``start()`` returns the nonblocking pipe's write fd. Pass it to the child
    with ``pass_fds`` and ``AXOL_SURVIVE_DIAGNOSTIC_FD``, then call
    ``child_started()`` to close the parent's copy. Call ``arm()`` only after
    tracking first becomes ready. Fault triggers before arming are ignored.

    Each ring and capture has both event and encoded-byte limits. At most one
    capture is collecting and one is queued behind the writer. A session saves
    at most three captures, with a cooldown, so an idle fault cannot create an
    unbounded stream of files. Dropped/truncated diagnostics are explicit in
    each summary. Time windows use Python receipt time; native monotonic time
    and device ticks are preserved separately, never relabelled capture time.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        pre_seconds: float = 10.0,
        post_seconds: float = 3.0,
        max_events: int = 160_000,
        max_bytes: int = 64 * 1024 * 1024,
        max_packet_bytes: int = 4096,
        queue_size: int = 4096,
        max_captures: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        if (
            pre_seconds < 0
            or post_seconds < 0
            or cooldown_seconds < 0
            or not all(
                math.isfinite(value)
                for value in (pre_seconds, post_seconds, cooldown_seconds)
            )
            or min(max_events, max_bytes, max_packet_bytes, queue_size, max_captures)
            < 1
        ):
            raise ValueError("diagnostic limits must be finite and positive")
        self.directory = (
            Path(directory)
            if directory is not None
            else Path.home() / ".almond" / "tracker" / "diagnostics"
        )
        self._metadata = dict(metadata or {})
        self._pre_seconds = pre_seconds
        self._post_seconds = post_seconds
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._max_packet_bytes = max_packet_bytes
        self._max_captures = max_captures
        self._cooldown_seconds = cooldown_seconds
        self._events: queue.Queue[tuple[float, str, dict[str, Any]]] = queue.Queue(
            maxsize=queue_size
        )
        self._triggers: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._writes: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._ring: deque[tuple[float, bytes]] = deque()
        self._ring_bytes = 0
        self._capacity_evicted_at = -math.inf
        self._capture: dict[str, Any] | None = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._collector_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: threading.Thread | None = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._packet = bytearray()
        self._discard_packet = False
        self._armed = False
        self._accepting = False
        self._closed = False
        self._pending = False
        self._captures_requested = 0
        self._last_trigger = -math.inf
        self._latest_path: str | None = None
        self._error: str | None = None
        self._state = "disabled"
        self._native_last_seq: int | None = None
        self._counters = dict.fromkeys(
            (
                "native_packets",
                "native_invalid_packets",
                "native_oversized_packets",
                "native_partial_packets",
                "native_reported_drops",
                "native_sequence_gaps",
                "native_sequence_inversions",
                "python_events",
                "python_queue_drops",
                "python_invalid_events",
                "python_oversized_events",
                "ring_capacity_evictions",
                "ring_age_evictions",
                "capture_capacity_evictions",
                "captures_saved",
                "capture_write_failures",
            ),
            0,
        )

    def start(self) -> int:
        """Start background collection; no hardware is opened by this class."""
        import fcntl

        with self._state_lock:
            if self._closed or self._thread is not None:
                raise RuntimeError("diagnostic recorder can only be started once")
            self._read_fd, self._write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
            # Absorb short scheduler stalls without losing native evidence.
            # This stays bounded and nonblocking; kernel/user quotas may refuse
            # growth, in which case native drop counters remain authoritative.
            try:
                fcntl.fcntl(self._write_fd, fcntl.F_SETPIPE_SZ, 1024 * 1024)
            except OSError:
                pass
            self._metadata.update(
                started_unix=time.time(),
                started_monotonic=time.monotonic(),
                native_clock="CLOCK_MONOTONIC seconds; device clocks remain per-device",
                window_clock="Python monotonic receipt time",
                native_pipe_bytes=fcntl.fcntl(self._write_fd, fcntl.F_GETPIPE_SZ),
            )
            self._accepting = True
            self._state = "waiting_for_ready"
            self._writer = threading.Thread(
                target=self._write_loop, name="tracker-diagnostic-writer", daemon=True
            )
            self._thread = threading.Thread(
                target=self._collect_loop, name="tracker-diagnostic-reader", daemon=True
            )
            try:
                self._writer.start()
                self._thread.start()
            except Exception as exc:
                self._accepting = False
                self._state = "error"
                self._error = f"could not start diagnostic threads: {exc}"
                self._stop.set()
                self._collector_done.set()
                os.close(self._read_fd)
                os.close(self._write_fd)
                self._read_fd = None
                self._write_fd = None
                raise
            return self._write_fd

    def child_started(self) -> None:
        """Drop the parent's write end after successful spawn (or spawn failure)."""
        with self._state_lock:
            fd, self._write_fd = self._write_fd, None
        if fd is not None:
            os.close(fd)

    def arm(self) -> None:
        with self._state_lock:
            if self._accepting:
                self._armed = True
                if self._state == "waiting_for_ready":
                    self._state = "armed"

    def record(self, kind: str, **fields: Any) -> None:
        """Enqueue a sparse Python observation; values must be JSON-compatible.

        Small primitive containers are copied. There is no serialization,
        compression, or disk I/O on the caller.
        """
        if not self._accepting:
            return
        try:
            if not isinstance(kind, str) or len(kind) > 128:
                raise ValueError("invalid diagnostic event kind")
            fields = self._bounded_fields(fields)
        except (TypeError, ValueError):
            self._counters["python_invalid_events"] += 1
            return
        try:
            self._events.put_nowait((time.monotonic(), kind, fields))
        except queue.Full:
            self._counters["python_queue_drops"] += 1

    def trigger(self, reason: str, context: dict[str, Any] | None = None) -> bool:
        """Request a fault window without blocking; return whether accepted."""
        now = time.monotonic()
        try:
            snapshot = self._bounded_fields(context or {})
        except (TypeError, ValueError):
            snapshot = {"diagnostic_context_error": "context exceeds diagnostic limits"}
        with self._state_lock:
            if (
                not self._accepting
                or not self._armed
                or self._pending
                or self._captures_requested >= self._max_captures
                or now - self._last_trigger < self._cooldown_seconds
            ):
                return False
            self._triggers.put_nowait(
                {
                    "reason": str(reason)[:4096],
                    "context": snapshot,
                    "trigger_monotonic": now,
                    "trigger_unix": time.time(),
                }
            )
            self._pending = True
            self._captures_requested += 1
            self._last_trigger = now
            self._state = "capturing"
        return True

    def status(self) -> dict[str, Any]:
        """Cheap best-effort snapshot; counters may advance while copied."""
        with self._state_lock:
            return {
                "enabled": self._thread is not None,
                "state": self._state,
                "armed": self._armed,
                "capture_count": self._captures_requested,
                "capture_limit": self._max_captures,
                "pending_capture": self._pending,
                "latest_path": self._latest_path,
                "error": self._error,
                "counters": dict(self._counters),
            }

    def close(self) -> None:
        """Flush a pending partial window; limit shutdown waits on slow disks."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
        self.child_started()
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=5.0)
        if self._writer is not None and self._writer.ident is not None:
            self._writer.join(timeout=5.0)
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                self._error = "diagnostic reader did not stop within 5 seconds"
            elif self._writer is not None and self._writer.is_alive():
                self._error = "diagnostic writer still flushing after 5 seconds"
            self._state = "closed"

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._error = message
        logger.warning("Tracking diagnostic recorder: %s", message)

    def _collect_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._poll_native()
                self._drain_python()
                self._begin_capture()
                self._expire_ring(time.monotonic())
                if (
                    self._capture is not None
                    and time.monotonic() >= self._capture["end_monotonic"]
                ):
                    self._finish_capture("complete")
            # Drain buffered input once; do not wait for the child's EOF.
            self._poll_native(wait=0)
            self._drain_python()
            self._begin_capture()
        except Exception as exc:
            self._set_error(f"collector failed: {exc}")
            with self._state_lock:
                self._accepting = False
                self._pending = False
                self._state = "error"
        finally:
            if self._packet or self._discard_packet:
                self._counters["native_partial_packets"] += 1
            self._packet.clear()
            if self._capture is not None:
                self._finish_capture("shutdown")
            if self._read_fd is not None:
                os.close(self._read_fd)
                self._read_fd = None
            self._collector_done.set()

    def _poll_native(self, wait: float = 0.01) -> None:
        fd = self._read_fd
        if fd is None:
            self._stop.wait(wait)
            return
        readable, _, _ = select.select([fd], [], [], wait)
        if not readable:
            return
        # Bound a pass so busy native traffic cannot starve triggers/timers.
        for _ in range(4):
            try:
                data = os.read(fd, 65536)
            except BlockingIOError:
                break
            if not data:
                if self._packet or self._discard_packet:
                    self._counters["native_partial_packets"] += 1
                    self._packet.clear()
                    self._discard_packet = False
                os.close(fd)
                self._read_fd = None
                break
            self._consume_native(data, time.monotonic())

    def _consume_native(self, data: bytes, received: float) -> None:
        pieces = data.split(b"\n")
        for index, piece in enumerate(pieces):
            complete = index < len(pieces) - 1
            if not self._discard_packet:
                if len(self._packet) + len(piece) > self._max_packet_bytes:
                    self._counters["native_oversized_packets"] += 1
                    self._packet.clear()
                    self._discard_packet = True
                else:
                    self._packet.extend(piece)
            if complete:
                if not self._discard_packet:
                    self._native_event(bytes(self._packet), received)
                self._packet.clear()
                self._discard_packet = False

    def _native_event(self, packet: bytes, received: float) -> None:
        try:
            event = json.loads(packet)
            if (
                not isinstance(event, dict)
                or event.get("v") != 1
                or not isinstance(event.get("kind"), str)
                or not isinstance(event.get("t"), (int, float))
                or not math.isfinite(event["t"])
                or not isinstance(event.get("seq"), int)
                or not isinstance(event.get("drops"), int)
                or event["drops"] < 0
            ):
                raise ValueError("invalid event envelope")
            event.update(origin="native", received_monotonic=received)
            encoded = self._encode(event)
        except (ValueError, TypeError, RecursionError, OverflowError):
            self._counters["native_invalid_packets"] += 1
            return
        self._counters["native_packets"] += 1
        self._counters["native_reported_drops"] = max(
            self._counters["native_reported_drops"], event["drops"]
        )
        seq = event["seq"]
        if self._native_last_seq is not None:
            if seq <= self._native_last_seq:
                self._counters["native_sequence_inversions"] += 1
            elif seq > self._native_last_seq + 1:
                self._counters["native_sequence_gaps"] += (
                    seq - self._native_last_seq - 1
                )
        self._native_last_seq = max(seq, self._native_last_seq or seq)
        self._append(received, encoded)

    @staticmethod
    def _encode(event: dict[str, Any]) -> bytes:
        return (
            json.dumps(event, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        )

    @staticmethod
    def _bounded_fields(fields: dict[str, Any]) -> dict[str, Any]:
        """Copy a small primitive snapshot without serialization or disk work.

        Queue length alone does not bound memory if a caller supplies a giant
        string or container. Limit the traversal and primitive payload too.
        """
        remaining_nodes = 512
        remaining_chars = 8192

        def copy(value: Any, depth: int = 0) -> Any:
            nonlocal remaining_nodes, remaining_chars
            remaining_nodes -= 1
            if remaining_nodes < 0 or depth > 8:
                raise ValueError("diagnostic event is too complex")
            if isinstance(value, int) and value.bit_length() > 128:
                raise ValueError("diagnostic integer is too large")
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, str):
                remaining_chars -= len(value)
                if remaining_chars < 0:
                    raise ValueError("diagnostic event is too large")
                return value
            if isinstance(value, (list, tuple)):
                if len(value) > remaining_nodes:
                    raise ValueError("diagnostic list is too large")
                return [copy(item, depth + 1) for item in value]
            if isinstance(value, dict):
                if len(value) * 2 > remaining_nodes:
                    raise ValueError("diagnostic object is too large")
                if not all(isinstance(key, str) for key in value):
                    raise TypeError("diagnostic keys must be strings")
                return {
                    copy(key, depth + 1): copy(item, depth + 1)
                    for key, item in value.items()
                }
            raise TypeError("diagnostic values must be JSON primitives")

        return copy(fields)

    def _drain_python(self) -> None:
        for _ in range(self._events.maxsize):
            try:
                observed, kind, fields = self._events.get_nowait()
            except queue.Empty:
                break
            try:
                event = dict(fields)
                event.update(v=1, origin="python", kind=kind, t=observed)
                encoded = self._encode(event)
            except (ValueError, TypeError, RecursionError, OverflowError):
                self._counters["python_invalid_events"] += 1
                continue
            if len(encoded) > self._max_packet_bytes * 2:
                self._counters["python_oversized_events"] += 1
                continue
            self._counters["python_events"] += 1
            self._append(observed, encoded)

    def _append(self, received: float, encoded: bytes) -> None:
        self._ring.append((received, encoded))
        self._ring_bytes += len(encoded)
        while len(self._ring) > self._max_events or self._ring_bytes > self._max_bytes:
            stamp, removed = self._ring.popleft()
            self._ring_bytes -= len(removed)
            self._capacity_evicted_at = max(self._capacity_evicted_at, stamp)
            self._counters["ring_capacity_evictions"] += 1
        capture = self._capture
        if (
            capture is not None
            and capture["start_monotonic"] <= received <= capture["end_monotonic"]
        ):
            capture["events"].append((received, encoded))
            capture["bytes"] += len(encoded)
            self._trim_capture(capture)

    def _expire_ring(self, now: float) -> None:
        # Native and Python events can be interleaved out of receipt-time order
        # within a polling pass. Keep at most one pass of extra age; filtering
        # the chosen window below is exact even in that case.
        while self._ring and self._ring[0][0] < now - self._pre_seconds:
            _, removed = self._ring.popleft()
            self._ring_bytes -= len(removed)
            self._counters["ring_age_evictions"] += 1

    def _begin_capture(self) -> None:
        if self._capture is not None:
            return
        try:
            capture = self._triggers.get_nowait()
        except queue.Empty:
            return
        start = capture["trigger_monotonic"] - self._pre_seconds
        end = capture["trigger_monotonic"] + self._post_seconds
        capture.update(
            start_monotonic=start,
            end_monotonic=end,
            pre_window_capacity_truncated=self._capacity_evicted_at >= start,
            counters_at_trigger=dict(self._counters),
            events=deque(item for item in self._ring if start <= item[0] <= end),
            truncated_events=0,
        )
        capture["bytes"] = sum(len(item[1]) for item in capture["events"])
        self._capture = capture

    def _trim_capture(self, capture: dict[str, Any]) -> None:
        while (
            len(capture["events"]) > self._max_events
            or capture["bytes"] > self._max_bytes
        ):
            _, removed = capture["events"].popleft()
            capture["bytes"] -= len(removed)
            capture["truncated_events"] += 1
            self._counters["capture_capacity_evictions"] += 1

    def _finish_capture(self, stop_reason: str) -> None:
        capture, self._capture = self._capture, None
        if capture is None:
            return
        capture.update(
            finished_monotonic=time.monotonic(),
            stop_reason=stop_reason,
            counters_at_finish=dict(self._counters),
        )
        with self._state_lock:
            self._pending = False
            self._state = "saving"
        try:
            self._writes.put_nowait(capture)
        except queue.Full:
            self._counters["capture_write_failures"] += 1
            self._set_error("writer backlog: diagnostic capture could not be queued")

    def _write_loop(self) -> None:
        while True:
            try:
                capture = self._writes.get(timeout=0.05)
            except queue.Empty:
                if self._collector_done.is_set():
                    return
                continue
            try:
                path = self._save_capture(capture)
            except Exception as exc:
                self._counters["capture_write_failures"] += 1
                self._set_error(f"could not save capture: {exc}")
                with self._state_lock:
                    if not self._pending and not self._closed:
                        self._state = "error"
                continue
            self._counters["captures_saved"] += 1
            with self._state_lock:
                self._latest_path = str(path)
                if not self._pending and not self._closed:
                    self._state = "saved"
            logger.warning("Tracking fault diagnostic saved to %s", path)

    def _save_capture(self, capture: dict[str, Any]) -> Path:
        name = f"fault-{int(capture['trigger_unix'] * 1000)}-{uuid.uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / f".{name}.partial"
        destination = self.directory / name
        temporary.mkdir(mode=0o700)
        events = capture.pop("events")
        # Receipt times can be interleaved across the two input queues. Sort
        # only on the writer thread; native t/device ticks remain unchanged.
        ordered = sorted(events, key=lambda item: item[0])
        summary = {
            "format_version": 1,
            "session": self._metadata,
            "limits": {
                "pre_seconds": self._pre_seconds,
                "post_seconds": self._post_seconds,
                "max_events": self._max_events,
                "max_encoded_bytes": self._max_bytes,
                "max_native_packet_bytes": self._max_packet_bytes,
                "max_captures": self._max_captures,
                "cooldown_seconds": self._cooldown_seconds,
            },
            **capture,
            "event_count": len(ordered),
            "first_received_monotonic": ordered[0][0] if ordered else None,
            "last_received_monotonic": ordered[-1][0] if ordered else None,
            "events_file": "events.jsonl.gz",
        }
        # Failed artifacts remain explicitly .partial; only a complete pair
        # of files is exposed as a finished capture directory.
        with gzip.open(temporary / "events.jsonl.gz", "wb", compresslevel=1) as out:
            for _, encoded in ordered:
                out.write(encoded)
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        temporary.rename(destination)
        return destination
