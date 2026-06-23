"""CPU-core isolation for the real-time control loop vs. background video work.

During ``collect-data`` the box runs the 120 Hz control loop + IK alongside the
video relay (WebRTC encode/send + raw-frame shm copy) and the dataset recorder
(+ its NVENC encoders). Even with spare cores, the Linux scheduler lets that
background work land on the same core as the control-loop thread, stalling it for
tens of ms (~25-45ms loop gaps → arm jerk). Nicing the background down fixes the
jerk but starves the relay's WebRTC send, corrupting the headset feed.

Partitioning cores fixes both: the control loop + IK get dedicated cores nothing
else runs on (no preemption → no jerk), and the relay + recorder run at normal
priority on the remaining cores (no deprioritized send → clean feed).

``pin_realtime`` / ``pin_background`` apply the partition to the calling process
(new threads inherit it, and ``subprocess`` children inherit the relay/recorder
affinity). Best-effort and self-gating: a no-op on machines with too few cores or
without ``sched_setaffinity`` (e.g. macOS), so off-Jetson dev is unaffected.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning (and it would
# starve one side), so isolation is skipped.
_MIN_CORES = 4


def _realtime_core_count(n: int) -> int:
    """How many cores to reserve for the control loop + IK on an ``n``-core host.

    Control loop ~0.5 core + IK ~1 (bursty) ≈ 2 cores of demand, so 3 cores on a
    Jetson-class box (≥6) give the loop real headroom while leaving the majority
    (≥5 → the relay/recorder/encoders need ~4); on smaller hosts reserve 2.
    """
    return 3 if n >= 6 else 2


def core_partition() -> tuple[set[int], set[int]] | None:
    """``(realtime_cores, background_cores)`` for this host, or ``None``.

    Uses the cores actually available to the process (respects any existing
    affinity / cgroup limit). Returns ``None`` when isolation isn't applicable.
    """
    try:
        avail = sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return None  # not Linux / not supported (e.g. macOS dev)
    if len(avail) < _MIN_CORES:
        return None
    n_rt = _realtime_core_count(len(avail))
    return set(avail[:n_rt]), set(avail[n_rt:])


def pin_realtime() -> bool:
    """Pin the calling process to the realtime cores (control loop + IK)."""
    return _pin(0)


def pin_background() -> bool:
    """Pin the calling process to the background cores (relay / recorder / gst)."""
    return _pin(1)


def _pin(which: int) -> bool:
    part = core_partition()
    if part is None:
        return False
    cores = part[which]
    try:
        os.sched_setaffinity(0, cores)  # type: ignore[attr-defined]
    except OSError as exc:
        _logger.debug("could not set CPU affinity to %s: %s", sorted(cores), exc)
        return False
    _logger.info(
        "pinned to %s cores %s",
        "realtime" if which == 0 else "background",
        sorted(cores),
    )
    return True
