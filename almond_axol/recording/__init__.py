"""Dataset recording, off the control loop.

Moves LeRobot dataset writing (camera capture, encode, ``save_episode``) out of
``collect-data``'s 120 Hz control process — see :mod:`record_proc`.

Public API
──────────
    DatasetRecorderProcess     Recorder running in a dedicated subprocess
    InProcessRecorder          Degraded fallback (recorder in the control process)
    default_vcodec             Pick a video codec that can open on this machine
    restore_dataset_ownership  Hand a root-recorded dataset back to the operator
"""

from .ownership import restore_dataset_ownership
from .record_proc import DatasetRecorderProcess, InProcessRecorder, default_vcodec

__all__ = [
    "DatasetRecorderProcess",
    "InProcessRecorder",
    "default_vcodec",
    "restore_dataset_ownership",
]
