"""Teleop jitter flight recorder.

Captures every stage boundary of the teleop pipeline — raw VR pose, filtered
pose, EE target, IK output, smoothed command, measured joints — as
timestamped rows in fixed-size ring buffers, so one jittery session can be
attributed offline stage by stage (``axol diag.teleop-jitter``).

Activation is by the ``record`` field on
:class:`~almond_axol.teleop.config.VRTeleopConfig`::

    axol teleop --teleop.record /tmp/jit

Every tap site holds that config — including the IK worker, which receives
it pickled through the subprocess spawn — so the prefix needs no other
plumbing. Each process/stage writes ``<prefix>_<stage>.npz`` on exit (the
IK worker writes ``_ik``, the smoothing core ``_cmd``, the control loop
``_meas``). All rows are stamped with ``time.monotonic()``, which shares
one epoch across processes on Linux, so the files can be merged on time.

Recording costs one array copy into a preallocated buffer per tick —
negligible against a CAN round-trip — and is entirely inert when the
field is unset (every hook holds ``None``).
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

# Default ring capacity: 5 minutes at the production 240 Hz control rate
# (the cmd/meas taps run once per control tick, so this is the sizing rate;
# the IK tap at 120 Hz keeps twice as long). Older rows are overwritten, so
# a long session keeps its *last* 5 minutes — end the session shortly after
# the moment you want captured.
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


def resolve_or_latest(prefix: str | None, stage: str = "cmd") -> str:
    """Resolve a recording prefix, or default to the newest recording.

    ``None`` picks the recording whose ``_<stage>.npz`` in
    :data:`RECORDINGS_DIR` is newest; a bare name resolves there; a path
    prefix is used verbatim. Raises ``SystemExit`` with an actionable
    message when the default is requested but nothing has been recorded.
    """
    if prefix is not None:
        return resolve_prefix(prefix)
    suffix = f"_{stage}.npz"
    newest = max(
        RECORDINGS_DIR.glob(f"*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if newest is None:
        raise SystemExit(
            f"No recordings in {RECORDINGS_DIR} — record one first with "
            "`axol teleop --teleop.record NAME`, or pass a prefix."
        )
    return str(newest)[: -len(suffix)]


class JitterRecorder:
    """Fixed-capacity ring buffer of timestamped float32 rows.

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

    def record(self, **values: np.ndarray | float) -> None:
        """Append one row; ``t`` is stamped automatically (``time.monotonic``)."""
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

    def dump(self) -> None:
        """Write the buffered rows (oldest first) to the ``.npz`` path."""
        if self._count == 0:
            return
        if self._count < self._capacity:
            order = np.arange(self._count)
        else:
            order = np.roll(np.arange(self._capacity), -self._head)
        out = {name: buf[order] for name, buf in self._buffers.items()}
        out["t"] = self._t[order]
        np.savez_compressed(self._path, **out)
        _logger.info("Jitter recorder: wrote %d rows to %s", self._count, self._path)


def make(
    prefix: str | None,
    stage: str,
    fields: dict[str, int],
    capacity: int = _DEFAULT_CAPACITY,
) -> JitterRecorder | None:
    """Build a recorder for ``stage`` if ``prefix`` is set, else ``None``.

    The recorder is registered with :mod:`atexit` so a normal shutdown (or
    Ctrl-C) writes ``<prefix>_<stage>.npz`` without any explicit dump call.
    """
    if not prefix:
        return None
    prefix = resolve_prefix(prefix)
    rec = JitterRecorder(f"{prefix}_{stage}.npz", fields, capacity)
    atexit.register(rec.dump)
    _logger.info("Jitter recorder active: stage %r -> %s_%s.npz", stage, prefix, stage)
    return rec
