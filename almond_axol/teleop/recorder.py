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
writes ``_ik``, the smoothing core ``_cmd``, the control loop ``_meas``).
All rows are stamped with ``time.monotonic()``, which shares one epoch
across processes on Linux, so the files can be merged on time.

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
# (the cmd/meas taps run once per control tick, so this is the sizing rate;
# the IK tap at 120 Hz keeps twice as long). Older rows are overwritten, so
# a long engaged segment keeps its *last* 5 minutes — disengage shortly
# after the moment you want captured.
_DEFAULT_CAPACITY = 72_000

# Where bare recording names land. ``--teleop.record demo1`` writes
# ``~/.almond/recordings/demo1_*.npz``; ``axol motion.build`` resolves bare
# prefixes against the same directory (and defaults to its newest recording),
# so record → build needs no paths at all. A prefix containing a path
# separator is used verbatim, as before.
RECORDINGS_DIR = Path.home() / ".almond" / "recordings"


def resolve_prefix(prefix: str) -> str:
    """A bare recording name becomes a path in :data:`RECORDINGS_DIR`."""
    if os.sep in prefix or (os.altsep and os.altsep in prefix):
        return prefix
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RECORDINGS_DIR / prefix)


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

    def record(self, **values: np.ndarray | float) -> None:
        """Append one row; ``t`` is stamped automatically (``time.monotonic``).

        No-op while disengaged (see :meth:`set_engaged`).
        """
        if not self._engaged:
            return
        i = self._head
        self._t[i] = time.monotonic()
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
