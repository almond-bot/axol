"""Teleop flight recorder.

Captures every stage boundary of the teleop pipeline — raw VR pose, filtered
pose, EE target, IK output, smoothed command, measured joints — as
timestamped rows in fixed-size ring buffers, so one jittery session can be
attributed offline stage by stage (``axol diag.offline``).

Activation is by the ``record`` field on
:class:`~almond_axol.teleop.config.VRTeleopConfig`::

    axol teleop --teleop.record demo1

Every tap site holds that config — including the IK worker, which receives
it pickled through the subprocess spawn — so the prefix needs no other
plumbing. Each process/stage writes ``<prefix>_<stage>.npz`` (the IK worker
writes ``_ik``, the smoothing core ``_cmd``, and the Rust-backed robot writes
``_meas`` plus a compact ``_rt`` file containing its motor-facing 240 Hz
command, derivative, feedforward, damping, timing, and feedback internals).
The measured loop records at 240 Hz from Rust core feedback. All rows are stamped with
``time.monotonic()``, which shares one epoch across processes on Linux, so the
files can be merged on time.

Recording covers **engaged segments only**: each tap gates its recorder on
its process's engage state (:meth:`TeleopRecorder.set_engaged`), so the
capture starts at engagement and there is no leading stretch of rest-pose
rows. Disengaging writes the files immediately (in a background thread, so
the control loop never blocks on compression) — which also survives a
SIGTERM'd session that would skip the :mod:`atexit` fallback dump — and
re-engaging discards the previous segment and starts over: one recording
holds the **latest** engage→disengage segment. The atexit dump remains as
a fallback for a session that ends while still engaged.

Recording costs one array copy into a preallocated buffer per tick —
negligible against a CAN round-trip — and is entirely inert when the
field is unset (every hook holds ``None``).
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

# Default ring capacity: 5 minutes at the production 240 Hz control rate
# (Rust-core meas runs at 240 Hz; cmd and IK run around 120 Hz and
# therefore retain up to twice as long). Older rows are overwritten, so
# a long engaged segment keeps its *last* 5 minutes — disengage shortly
# after the moment you want captured.
_DEFAULT_CAPACITY = 72_000

# Where bare recording names land. ``--teleop.record demo1`` writes
# ``~/.almond/recordings/demo1_*.npz``; ``axol motion.build`` resolves bare
# prefixes against the same directory (and defaults to its newest recording),
# so record → build needs no paths at all. A prefix containing a path
# separator is used verbatim, as before.
RECORDINGS_DIR = Path.home() / ".almond" / "recordings"

_RT_TRACE_COLUMNS = (
    "tick",
    "time_s",
    "seq",
    "slot",
    "motor_id",
    "mode",
    "target_p",
    "cmd_p",
    "cmd_v",
    "cmd_a",
    "cmd_v_fast",
    "meas_p",
    "motor_v",
    "meas_v",
    "meas_tau",
    "gravity_ff",
    "friction_ff",
    "inertia_ff",
    "damping_ff",
    "total_ff",
    "kd_host",
    "damp_w0",
    "damp_q",
    "tick_dt",
    "fb_dt",
)


def resolve_prefix(prefix: str) -> str:
    """A bare recording name becomes a path in :data:`RECORDINGS_DIR`."""
    if os.sep in prefix or (os.altsep and os.altsep in prefix):
        return prefix
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RECORDINGS_DIR / prefix)


def compact_rt_trace(prefix: str) -> Path | None:
    """Compact the gated Rust CSVs into ``<prefix>_rt.npz``.

    Rust writes CSV through a non-blocking background thread so compression
    and Python IPC cannot perturb its 240 Hz loop. After disarm, combine both
    arm files into columnar arrays and remove the verbose intermediates. Rows
    stay flat with ``side``/``slot`` columns so partial-arm captures need no
    padding and diagnostics can select a joint directly.
    """
    prefix = resolve_prefix(prefix)
    raw_paths = [Path(f"{prefix}_rt-{side}.csv") for side in ("left", "right")]
    chunks: dict[str, list[np.ndarray]] = {
        "side": [],
        **{("t" if name == "time_s" else name): [] for name in _RT_TRACE_COLUMNS},
    }
    found_rows = False
    found_files = False
    for side, path in enumerate(raw_paths):
        if not path.exists():
            continue
        found_files = True
        with path.open("rb") as raw:
            header = raw.readline().decode("ascii", "replace").strip().split(",")
            has_rows = bool(raw.read(1))
        if header != list(_RT_TRACE_COLUMNS):
            raise ValueError(f"unexpected Rust trace schema in {path}")
        if not has_rows:
            continue
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if values.shape[1] != len(_RT_TRACE_COLUMNS):
            raise ValueError(
                f"unexpected Rust trace width in {path}: {values.shape[1]}"
            )
        found_rows = True
        chunks["side"].append(np.full(len(values), side, dtype=np.uint8))
        for index, source_name in enumerate(_RT_TRACE_COLUMNS):
            name = "t" if source_name == "time_s" else source_name
            if source_name == "tick":
                array = values[:, index].astype(np.uint64)
            elif source_name == "seq":
                array = values[:, index].astype(np.uint32)
            elif source_name in {"slot", "motor_id"}:
                array = values[:, index].astype(np.uint8)
            elif source_name == "time_s":
                # Python monotonic timestamps are large enough that float32
                # would quantize away the 4.17 ms control-tick spacing.
                array = values[:, index].astype(np.float64)
            else:
                array = values[:, index].astype(np.float32)
            chunks[name].append(array)
        del values
    if not found_rows:
        if found_files:
            for path in raw_paths:
                path.unlink(missing_ok=True)
        return None

    packed = {name: np.concatenate(parts) for name, parts in chunks.items() if parts}
    packed["schema_version"] = np.asarray(1, dtype=np.uint8)
    destination = Path(f"{prefix}_rt.npz")
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **packed)
    temporary.replace(destination)
    for path in raw_paths:
        path.unlink(missing_ok=True)
    _logger.info(
        "Flight recorder: wrote %d Rust rows to %s", len(packed["t"]), destination
    )
    return destination


def resolve_or_latest(prefix: str | None, stage: str | tuple[str, ...] = "cmd") -> str:
    """Resolve a recording prefix, or default to the newest recording.

    ``None`` picks the recording whose ``_<stage>.npz`` in
    :data:`RECORDINGS_DIR` is newest (several stages may be given — e.g.
    ``("cmd", "gc")`` accepts teleop and gravity-comp captures alike); a
    bare name resolves there; a path prefix is used verbatim. Raises
    ``SystemExit`` with an actionable message when the default is requested
    but nothing has been recorded.
    """
    if prefix is not None:
        return resolve_prefix(prefix)
    stages = (stage,) if isinstance(stage, str) else stage
    candidates = [
        (path, f"_{s}.npz")
        for s in stages
        for path in RECORDINGS_DIR.glob(f"*_{s}.npz")
    ]
    newest = max(candidates, key=lambda c: c[0].stat().st_mtime, default=None)
    if newest is None:
        raise SystemExit(
            f"No recordings in {RECORDINGS_DIR} — record one first with "
            "`axol teleop --teleop.record NAME` (or `axol gravity-comp "
            "--record NAME` for a hand-guided one), or pass a prefix."
        )
    path, suffix = newest
    return str(path)[: -len(suffix)]


def list_recordings() -> list[dict[str, object]]:
    """The buildable recordings in :data:`RECORDINGS_DIR`, newest first.

    One entry per prefix that has a motion-source stage — teleop's ``_cmd``
    or gravity comp's ``_gc`` (``_cmd`` wins when both exist, matching
    ``build_motion``). ``duration_s`` spans the capture's timestamps
    (teleop: the engaged segment; gravity comp: the whole session) and is
    ``None`` for a file that can't be read.
    """
    if not RECORDINGS_DIR.is_dir():
        return []
    by_name: dict[str, dict[str, object]] = {}
    for stage, kind in (("cmd", "teleop"), ("gc", "gravity-comp")):
        suffix = f"_{stage}.npz"
        for path in RECORDINGS_DIR.glob(f"*{suffix}"):
            name = path.name[: -len(suffix)]
            if name in by_name:
                continue
            duration: float | None
            try:
                with np.load(path) as data:
                    t = data["t"]
                    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            except Exception:  # noqa: BLE001 - a corrupt file still gets listed
                duration = None
            by_name[name] = {
                "name": name,
                "kind": kind,
                "modified_at": path.stat().st_mtime,
                "duration_s": duration,
            }
    return sorted(
        by_name.values(),
        key=lambda r: r["modified_at"],  # type: ignore[arg-type,return-value]
        reverse=True,
    )


class TeleopRecorder:
    """Fixed-capacity ring buffer of timestamped float32 rows.

    Starts *disengaged*: :meth:`record` is a no-op until the owning stage
    reports engagement via :meth:`set_engaged`, so a recording holds exactly
    one engage→disengage segment (the latest one) with no rest-pose lead-in.

    Args:
        path: Output ``.npz`` path written by :meth:`dump`.
        fields: Mapping of field name to vector width for each row.
        capacity: Ring size in rows; the oldest rows are overwritten.
    """

    def __init__(
        self, path: str, fields: dict[str, int], capacity: int = _DEFAULT_CAPACITY
    ) -> None:
        self._path = path
        self._fields = dict(fields)
        self._capacity = capacity
        self._buffers = {
            name: np.zeros((capacity, width), dtype=np.float32)
            for name, width in self._fields.items()
        }
        self._t = np.zeros(capacity, dtype=np.float64)
        self._head = 0
        self._count = 0
        self._engaged = False
        self._writer: threading.Thread | None = None

    def record(
        self,
        *,
        timestamp: float | None = None,
        **values: np.ndarray | float,
    ) -> None:
        """Append one row, normally stamped with ``time.monotonic()``.

        ``timestamp`` lets a source with a more accurate sample clock retain
        it—RT feedback uses reconstructed kernel receive times rather than
        Python socket-delivery time. No-op while disengaged (see
        :meth:`set_engaged`).
        """
        if not self._engaged:
            return
        i = self._head
        self._t[i] = time.monotonic() if timestamp is None else timestamp
        for name, buf in self._buffers.items():
            v = values.get(name)
            if v is None:
                buf[i] = np.nan
            else:
                buf[i] = np.asarray(v, dtype=np.float32).reshape(-1)
        self._head = (i + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def set_engaged(self, engaged: bool) -> None:
        """Gate recording to the engaged segment. Call once per tick.

        Rising edge: the previous segment is discarded — a re-engage
        overwrites, so the recording always holds the latest segment.
        Falling edge: the segment is written to disk immediately, in a
        background thread (compressing a long segment can take a second —
        blocking the control loop that long would starve the CAN stream).
        Writing at disengage rather than only at exit also survives a
        session that later dies without running :mod:`atexit` hooks (the
        serve manager escalates to SIGTERM when Ctrl-C cleanup runs long).
        """
        if engaged == self._engaged:
            return
        self._engaged = engaged
        if engaged:
            self._join_writer()
            self._head = 0
            self._count = 0
        else:
            self._start_writer(self._snapshot())

    def dump(self) -> None:
        """Write the buffered rows to the ``.npz`` path (atexit fallback).

        Waits for any in-flight disengage write first, then rewrites the
        file from the current buffer — idempotent when the buffer hasn't
        changed since that write.
        """
        self._join_writer()
        data = self._snapshot()
        if data is not None:
            self._write(data)

    def _snapshot(self) -> dict[str, np.ndarray] | None:
        """Copy the buffered rows (oldest first), or ``None`` when empty."""
        if self._count == 0:
            return None
        if self._count < self._capacity:
            order = np.arange(self._count)
        else:
            order = np.roll(np.arange(self._capacity), -self._head)
        out = {name: buf[order] for name, buf in self._buffers.items()}
        out["t"] = self._t[order]
        return out

    def _start_writer(self, data: dict[str, np.ndarray] | None) -> None:
        if data is None:
            return
        self._join_writer()
        self._writer = threading.Thread(
            target=self._write, args=(data,), daemon=True, name="teleop-rec-dump"
        )
        self._writer.start()

    def _join_writer(self) -> None:
        if self._writer is not None:
            self._writer.join(timeout=30.0)
            self._writer = None

    def _write(self, data: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self._path, **data)
        _logger.info("Flight recorder: wrote %d rows to %s", len(data["t"]), self._path)


def make(
    prefix: str | None,
    stage: str,
    fields: dict[str, int],
    capacity: int = _DEFAULT_CAPACITY,
) -> TeleopRecorder | None:
    """Build a recorder for ``stage`` if ``prefix`` is set, else ``None``.

    ``<prefix>_<stage>.npz`` is written at every disengage (see
    :meth:`TeleopRecorder.set_engaged`); the :mod:`atexit` registration is a
    fallback for a session that ends while still engaged.
    """
    if not prefix:
        return None
    prefix = resolve_prefix(prefix)
    rec = TeleopRecorder(f"{prefix}_{stage}.npz", fields, capacity)
    atexit.register(rec.dump)
    _logger.info("Flight recorder active: stage %r -> %s_%s.npz", stage, prefix, stage)
    return rec
