"""Teleop jitter flight recorder.

Captures every stage boundary of the teleop pipeline — raw VR pose, filtered
pose, EE target, IK output, smoothed command, measured joints — as
timestamped rows in fixed-size ring buffers, so one jittery session can be
attributed offline stage by stage (``axol diag.teleop-jitter``).

Activation is by environment variable so it reaches the IK subprocess
without any plumbing::

    AXOL_JITTER_RECORD=/tmp/jit axol teleop

Each process/stage writes ``<prefix>_<stage>.npz`` on exit (the IK worker
writes ``_ik``, the smoothing core ``_cmd``, the control loop ``_meas``).
All rows are stamped with ``time.monotonic()``, which shares one epoch
across processes on Linux, so the files can be merged on time.

Recording costs one array copy into a preallocated buffer per tick —
negligible against a CAN round-trip — and is entirely inert when the
environment variable is unset (every hook holds ``None``).
"""

from __future__ import annotations

import atexit
import logging
import os
import time

import numpy as np

_logger = logging.getLogger(__name__)

ENV_VAR = "AXOL_JITTER_RECORD"

# Default ring capacity: 5 minutes at 120 Hz. Older rows are overwritten, so
# a long session keeps its *last* 5 minutes — end the session shortly after
# reproducing the jitter.
_DEFAULT_CAPACITY = 36_000


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


def from_env(
    stage: str, fields: dict[str, int], capacity: int = _DEFAULT_CAPACITY
) -> JitterRecorder | None:
    """Build a recorder for ``stage`` if :data:`ENV_VAR` is set, else ``None``.

    The recorder is registered with :mod:`atexit` so a normal shutdown (or
    Ctrl-C) writes ``<prefix>_<stage>.npz`` without any explicit dump call.
    """
    prefix = os.environ.get(ENV_VAR)
    if not prefix:
        return None
    rec = JitterRecorder(f"{prefix}_{stage}.npz", fields, capacity)
    atexit.register(rec.dump)
    _logger.info("Jitter recorder active: stage %r -> %s_%s.npz", stage, prefix, stage)
    return rec
