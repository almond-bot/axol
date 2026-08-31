"""Dataset recording, off the control loop.

Moves LeRobot dataset writing (camera capture, encode, ``save_episode``) out of
``collect-data``'s 120 Hz control process — see :mod:`record_proc`.

Public API
──────────
    DatasetRecorderProcess     Recorder running in a dedicated subprocess
    InProcessRecorder          Degraded fallback (recorder in the control process)
    default_vcodec             Pick a video codec that can open on this machine
    make_episode_durable       Flush a just-saved episode so a kill can't lose it
    restore_dataset_ownership  Safely expose a root-recorded dataset for reading
"""

from .ownership import restore_dataset_ownership
from .record_proc import (
    DatasetRecorderProcess,
    InProcessRecorder,
    default_vcodec,
    make_episode_durable,
)

__all__ = [
    "DatasetRecorderProcess",
    "InProcessRecorder",
    "default_vcodec",
    "make_episode_durable",
    "restore_dataset_ownership",
]
