"""CPU-core isolation for the real-time / latency-critical work during recording.

During ``collect-data`` the box runs three kinds of work that contend for cores:

* **realtime** — the 120 Hz control loop + IK. Background work landing on its
  cores stalls it for tens of ms (arm jerk).
* **relay** — the headset video: gst NVENC encode + the WebRTC (aiortc) *send*
  loop, which is latency-sensitive. In teleop it has its cores to itself and the
  feed is clean; once recording starts, the dataset recorder + its NVENC
  encoders pile onto the same cores and starve the send — packets go out late
  and bursty (0% loss but rising jitter), so the live feed gets laggy + grainy.
* **background** — the dataset recorder + its per-camera NVENC encoders. Pure
  throughput; tolerant of an occasional dropped frame.

Partitioning the cores by role keeps each group off the others': the control
loop never gets preempted (no jerk), and the relay's send gets prompt CPU like
it does in teleop (clean feed), while the dataset encode runs on its own cores.

``pin_realtime`` / ``pin_relay`` / ``pin_background`` apply the partition to the
calling process (new threads inherit it; ``subprocess`` children inherit the
relay/recorder affinity). Best-effort and self-gating: a no-op on machines with
too few cores or without ``sched_setaffinity`` (e.g. macOS), so off-Jetson dev is
unaffected.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning, so isolation
# is skipped (the groups collapse onto whatever's available).
_MIN_CORES = 4


def core_groups() -> dict[str, set[int]] | None:
    """``{"realtime", "relay", "background"}`` → core sets for this host, or ``None``.

    Based on the machine's *physical* core count, NOT the process's current
    affinity: the control process pins itself before spawning the relay/recorder,
    and those children inherit its restricted mask — they must still compute the
    full partition and ``sched_setaffinity`` to their own group (allowed even from
    a restricted mask). Reading the inherited mask would wrongly see only the
    realtime cores.

    8+ cores: control 3 / relay 1 / dataset rest. The dataset recorder re-encodes
    three camera streams from raw RGBA and is the heaviest throughput consumer, so
    it gets the widest group; the relay keeps a single core because its headset
    H.264 runs on the hardware NVENC block (``nvv4l2h264enc``) and that core mainly
    carries the latency-sensitive WebRTC send. On 6-7 cores the control group
    shrinks to 2; below 6 there's no room to isolate the relay, so relay shares the
    background group (still kept off the control cores). ``None`` when partitioning
    isn't applicable at all.
    """
    n = os.cpu_count()
    if not n or n < _MIN_CORES:
        return None
    if n >= 8:
        rt = set(range(3))
        relay = {3}
        bg = set(range(4, n))
    elif n >= 6:
        rt = {0, 1}
        relay = {2, 3}
        bg = set(range(4, n))
    else:  # 4-5 cores: isolate control only; relay + dataset share the rest
        rt = {0, 1}
        relay = bg = set(range(2, n))
    return {"realtime": rt, "relay": relay, "background": bg}


def pin_realtime() -> bool:
    """Pin the calling process to the realtime cores (control loop + IK)."""
    return _pin("realtime")


def pin_relay() -> bool:
    """Pin the calling process to the relay cores (headset encode + WebRTC send)."""
    return _pin("relay")


def pin_background() -> bool:
    """Pin the calling process to the background cores (dataset recorder + gst)."""
    return _pin("background")


def pin_relay_send_thread() -> bool:
    """Pin the *calling thread* to a single dedicated core within the relay group.

    aiortc runs the whole WebRTC send (SRTP + sendto for every stream) on one
    asyncio thread, which is CPU-bound and pegs a core. Pinning that thread to one
    relay core — while the relay's other threads (gst encode + the dataset
    raw-branch shm copy) stay on the full relay group — keeps the recording
    workload from stealing the send's CPU, so it doesn't fall behind. Call from
    the relay's event-loop thread after its other threads exist. No-op where
    affinity isn't available or the relay group has only one core.
    """
    groups = core_groups()
    if groups is None:
        return False
    relay = sorted(groups["relay"])
    if len(relay) < 2:
        return False  # nothing to dedicate
    send_core = {relay[0]}
    try:
        os.sched_setaffinity(0, send_core)  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:
        _logger.debug("could not pin relay send thread to %s: %s", send_core, exc)
        return False
    _logger.info("pinned relay send thread to core %s", relay[0])
    return True


def _pin(group: str) -> bool:
    groups = core_groups()
    if groups is None:
        return False
    cores = groups[group]
    try:
        os.sched_setaffinity(0, cores)  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:  # AttributeError: no sched_* (macOS)
        _logger.debug("could not set CPU affinity to %s: %s", sorted(cores), exc)
        return False
    _logger.info("pinned to %s cores %s", group, sorted(cores))
    return True
