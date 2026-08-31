"""In-memory motor telemetry hub + on-disk diagnostics run store.

The hub is the meeting point between the robot link (which samples the motors
on its own thread/loop, see :mod:`.robot_link`) and the web dashboard (which
subscribes over a WebSocket on the server loop). Producers push from the link
thread; fan-out to subscriber queues is marshalled onto each subscriber's own
event loop, mirroring the Session log fan-out in :mod:`.manager`.

Wire/message shapes (also served over REST for backfill):

- fast frame, ~SAMPLE_HZ per second while the link owns the CAN bus::

    {"type": "frame", "t": <epoch s>,
     "m": {"left:SHOULDER_1": [pos_rad, vel_rad_s, torque_nm], ...}}

- slow sweep, ~1 Hz (piggybacks the health ping)::

    {"type": "slow", "t": <epoch s>,
     "m": {"left:SHOULDER_1": {"temperature": C, "voltage": V,
            "status": "OK", "reachable": true}, ...}}

- link state transitions (why the stream went quiet)::

    {"type": "state", "state": "busy"}

A *diagnostics run* wraps a launched session (script) with the telemetry
frames observed during its lifetime, persisted under
``~/.almond/diagnostics/runs`` so past runs (ROM test, motor health, ...) can
be charted later. Scripts that own the CAN bus themselves can contribute a
richer capture by printing a ``[telemetry] csv=<path>`` marker (see
:mod:`almond_axol.diagnostics.telemetry_log`); the store picks the file up
when the session ends.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from ..utils.paths import almond_path
from ..utils.state_files import (
    secure_atomic_write_json,
    secure_list_names,
    secure_open_text_read,
    secure_read_text,
    secure_unlink,
)

_logger = logging.getLogger(__name__)

# Fast-sample cadence (position / velocity / torque, all motors).
SAMPLE_HZ = 10.0
# Ring buffer length: 10 minutes at SAMPLE_HZ.
_BUFFER_FRAMES = int(SAMPLE_HZ * 600)

RUNS_DIR = almond_path("diagnostics", "runs")
CAPTURES_DIR = almond_path("diagnostics", "captures")

# Marker a bus-owning diagnostic script prints to hand over its own capture.
_CSV_MARKER = re.compile(r"\[telemetry\] csv=(\S+)")
_RUN_ID = re.compile(r"^[0-9a-f]{12}$")
_RUN_FILE = re.compile(r"^([0-9a-f]{12})\.(meta|data)\.json$")


def _capture_path(value: object) -> Path | None:
    """Accept only lexical descendants of the diagnostics capture directory."""
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    root = Path(os.path.abspath(CAPTURES_DIR))
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def motor_key(arm: str, joint: str) -> str:
    return f"{arm}:{joint}"


class TelemetryHub:
    """Ring buffer + subscriber fan-out for live motor telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: deque[dict[str, Any]] = deque(maxlen=_BUFFER_FRAMES)
        # motor key -> latest slow reading (temperature/voltage/status/...).
        self._slow: dict[str, dict[str, Any]] = {}
        self._slow_t: float | None = None
        self._state = "disconnected"
        # subscriber queue -> the event loop it must be woken on.
        self._subs: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}

    # -- producer side (called from the robot-link thread) -------------------

    def push_frame(self, motors: dict[str, list[float]]) -> None:
        frame = {"t": time.time(), "m": motors}
        with self._lock:
            self._frames.append(frame)
        self._fanout({"type": "frame", **frame})

    def push_slow(self, motors: dict[str, dict[str, Any]]) -> None:
        now = time.time()
        with self._lock:
            self._slow.update(motors)
            self._slow_t = now
        self._fanout({"type": "slow", "t": now, "m": motors})

    def push_state(self, state: str) -> None:
        changed = False
        with self._lock:
            if state != self._state:
                self._state = state
                changed = True
        if changed:
            self._fanout({"type": "state", "state": state})

    def clear_slow(self) -> None:
        with self._lock:
            self._slow = {}
            self._slow_t = None

    def _fanout(self, message: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs.items())
        for queue, loop in subs:
            loop.call_soon_threadsafe(self._safe_put, queue, message)

    @staticmethod
    def _safe_put(queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # a stalled client drops frames rather than backing up the bus

    # -- consumer side (server loop) -----------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subs[queue] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subs.pop(queue, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._frames[-1] if self._frames else None
            return {
                "state": self._state,
                "sampleHz": SAMPLE_HZ,
                "slow": dict(self._slow),
                "slowT": self._slow_t,
                "latest": latest,
            }

    def history(self, seconds: float, max_frames: int = 3000) -> list[dict[str, Any]]:
        """Buffered frames from the last ``seconds``, strided to ``max_frames``."""
        cutoff = time.time() - seconds
        with self._lock:
            frames = [f for f in self._frames if f["t"] >= cutoff]
        if len(frames) > max_frames:
            stride = len(frames) / max_frames
            frames = [frames[int(i * stride)] for i in range(max_frames)]
        return frames

    def frames_between(self, start: float, end: float) -> list[dict[str, Any]]:
        with self._lock:
            return [f for f in self._frames if start <= f["t"] <= end]


class DiagnosticsRunStore:
    """Persists diagnostics runs (metadata + telemetry capture) to disk."""

    def __init__(self, hub: TelemetryHub, runs_dir: Path = RUNS_DIR) -> None:
        self._hub = hub
        self._dir = runs_dir

    def begin(
        self, session_id: str, command: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": uuid.uuid4().hex[:12],
            "sessionId": session_id,
            "command": command,
            "args": args,
            "startedAt": time.time(),
            "endedAt": None,
            "status": "running",
            "exitCode": None,
        }

    def finalize(
        self, meta: dict[str, Any], status: str, exit_code: int | None, log: list[str]
    ) -> None:
        meta["endedAt"] = time.time()
        meta["status"] = status
        meta["exitCode"] = exit_code

        frames = self._hub.frames_between(meta["startedAt"], meta["endedAt"])
        csv_path = None
        for line in log:
            match = _CSV_MARKER.search(line)
            if match:
                csv_path = match.group(1)
        meta["telemetryCsv"] = csv_path
        meta["frameCount"] = len(frames)

        try:
            secure_atomic_write_json(
                self._dir / f"{meta['id']}.meta.json", meta, indent=None
            )
            secure_atomic_write_json(
                self._dir / f"{meta['id']}.data.json",
                {"frames": frames, "log": log},
                indent=None,
            )
        except OSError as exc:
            _logger.warning("failed to persist diagnostics run: %s", exc)

    def clear(self) -> int:
        """Delete every persisted run record and its telemetry capture.

        Returns the number of run records (``*.meta.json``) removed. A run's
        ROM-style CSV capture lives outside the runs dir (see
        :class:`~almond_axol.diagnostics.telemetry_log.TelemetryCsvLogger`), so
        it's deleted here too — otherwise Clear would leave orphaned capture
        files with no dashboard path to remove them.
        """
        try:
            names = secure_list_names(self._dir)
        except (FileNotFoundError, OSError):
            return 0
        # Delete referenced CSV captures first, before the metas that name them.
        for name in names:
            if not name.endswith(".meta.json") or _RUN_FILE.fullmatch(name) is None:
                continue
            meta_path = self._dir / name
            try:
                csv_path = _capture_path(
                    json.loads(secure_read_text(meta_path)).get("telemetryCsv")
                )
            except (OSError, ValueError):
                continue
            if csv_path is not None:
                try:
                    secure_unlink(csv_path, missing_ok=True)
                except OSError as exc:
                    _logger.warning("failed to delete capture %s: %s", csv_path, exc)

        removed = 0
        for name in names:
            if _RUN_FILE.fullmatch(name) is None:
                continue
            path = self._dir / name
            is_meta = path.name.endswith(".meta.json")
            try:
                secure_unlink(path)
                if is_meta:
                    removed += 1
            except OSError as exc:
                _logger.warning("failed to delete run record %s: %s", path, exc)
        return removed

    def list(self) -> list[dict[str, Any]]:
        try:
            names = secure_list_names(self._dir)
        except (FileNotFoundError, OSError):
            return []
        runs: list[dict[str, Any]] = []
        for name in names:
            if not name.endswith(".meta.json") or _RUN_FILE.fullmatch(name) is None:
                continue
            path = self._dir / name
            try:
                runs.append(json.loads(secure_read_text(path)))
            except (OSError, ValueError) as exc:
                _logger.debug("skipping unreadable run meta %s: %s", path, exc)
        runs.sort(key=lambda r: r.get("startedAt") or 0, reverse=True)
        return runs

    def load(self, run_id: str, max_frames: int = 3000) -> dict[str, Any] | None:
        if _RUN_ID.fullmatch(run_id) is None:
            return None
        meta_path = self._dir / f"{run_id}.meta.json"
        try:
            meta = json.loads(secure_read_text(meta_path))
        except FileNotFoundError:
            return None
        data: dict[str, Any] = {"frames": [], "log": []}
        data_path = self._dir / f"{run_id}.data.json"
        try:
            data = json.loads(secure_read_text(data_path))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            _logger.warning("unreadable run data %s: %s", data_path, exc)

        frames = data.get("frames", [])
        csv_path = _capture_path(meta.get("telemetryCsv"))
        if csv_path is not None:
            csv_frames = _read_csv_frames(csv_path)
            # A script's own capture is denser and spans bus-owned time the
            # server couldn't observe — prefer it when present.
            if csv_frames:
                frames = csv_frames
        if len(frames) > max_frames:
            stride = len(frames) / max_frames
            frames = [frames[int(i * stride)] for i in range(max_frames)]
        return {"meta": meta, "frames": frames, "log": data.get("log", [])}


def _read_csv_frames(path: Path) -> list[dict[str, Any]]:
    """Parse a wide-format telemetry CSV (see diagnostics.telemetry_log).

    Columns: ``t`` then ``<arm>:<JOINT>:pos|vel|tq``. Missing cells are empty.
    """
    frames: list[dict[str, Any]] = []
    try:
        with secure_open_text_read(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header or header[0] != "t":
                return []
            # column index -> (motor key, field index into [pos, vel, tq])
            fields = {"pos": 0, "vel": 1, "tq": 2}
            columns: dict[int, tuple[str, int]] = {}
            for i, name in enumerate(header[1:], start=1):
                key, _, field = name.rpartition(":")
                if key and field in fields:
                    columns[i] = (key, fields[field])
            for row in reader:
                motors: dict[str, list[float | None]] = {}
                for i, (key, slot) in columns.items():
                    if i >= len(row) or row[i] == "":
                        continue
                    values = motors.setdefault(key, [None, None, None])
                    values[slot] = float(row[i])
                if motors:
                    frames.append({"t": float(row[0]), "m": motors})
    except (OSError, ValueError) as exc:
        _logger.warning("failed to parse telemetry csv %s: %s", path, exc)
        return []
    return frames
