"""CPU-core isolation for the real-time / latency-critical work during recording.

During ``collect-data`` the box runs five kinds of work that contend for cores:

* **can** — the two Rust 240 Hz CAN loops. On an 8+ core robot each arm owns a
  dedicated core; the Rust process pins the bus threads individually.
* **realtime** — the Python 120 Hz target loop plus its web/VR/teleop and
  IK-dispatch threads. It has a separate core from CAN, so Python or camera
  activity cannot delay a motor tick.
* **ik** — the out-of-process JAX IK solver (a ~1-core solve). On 8+ cores it
  gets a dedicated core so recording load can't preempt it mid-solve (which drops
  its rate ~115 -> ~80 Hz); on smaller hosts it shares the realtime cores.
* **relay** — the headset video: gst NVENC encode + the WebRTC (aiortc) *send*
  loop, which is latency-sensitive. In teleop it has its cores to itself and the
  feed is clean; once recording starts, the dataset raw-branch piles onto the
  same cores and starves the send — packets go out late and bursty (0% loss but
  rising jitter), so the live feed gets laggy + grainy.
* **background** — the dataset recorder plus throughput-oriented relay
  GStreamer work. It may share CPU with camera/VIC/NVENC dispatch, but never
  with control, IK, CAN, or the latency-sensitive WebRTC Python loop.

Partitioning the cores by role keeps each group off the others': the control
loop never gets preempted (no jerk), IK solves at full rate, and the relay's send
gets prompt CPU like it does in teleop (clean feed), while the dataset encode
runs on its own cores.

``pin_realtime`` / ``pin_ik`` / ``pin_relay`` / ``pin_background`` apply the
partition to the calling process (new threads inherit it; ``subprocess`` children
inherit the relay/recorder affinity). The CAN set is exported to ``axol-rt``,
which assigns one bus thread to each core itself. Best-effort and self-gating: a
no-op on machines with too few cores or without ``sched_setaffinity`` (e.g.
macOS), so off-Jetson dev is unaffected.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning, so isolation
# is skipped (the groups collapse onto whatever's available).
_MIN_CORES = 4


def core_groups() -> dict[str, set[int]] | None:
    """``{"can", "realtime", "ik", "relay", "background"}`` → core sets.

    Based on the machine's *physical* core count, NOT the process's current
    affinity: the control process pins itself before spawning the relay/recorder,
    and those children inherit its restricted mask — they must still compute the
    full partition and ``sched_setaffinity`` to their own group (allowed even from
    a restricted mask). Reading the inherited mask would wrongly see only the
    realtime cores.

    8+ cores: CAN 2 / Python control 1 / IK 1 / relay 2 / dataset 2. Each CAN
    arm gets one of the final two cores, away from CPU0: the Jetson routes its
    xHCI interrupt there, and both USB CAN adapters plus the cameras traverse
    that controller. Dataset work gets CPUs 0-1 because it tolerates those
    interrupts. This also avoids the old shared ``realtime`` layout where
    camera/WebRTC bookkeeping made 5-15% of nominal 240 Hz motor ticks late.
    ``ik`` remains a dedicated core so it cannot deschedule CAN or Python.

    The relay gets *two* private cores so :func:`isolate_relay_cpu` can keep its
    Python work (the aiortc WebRTC send + encoded-AU pull loops, all
    GIL-serialized) on one. GStreamer's much wider C thread pool may use the
    other relay core **and** the background cores. That split is the whole point:
    naively letting Python roam just ping-pongs the GIL, while restricting the
    ~70-80 camera/VIC/NVENC/shm tasks to one CPU can deschedule a dataset branch
    for multiple 60 Hz exposures. Sharing throughput cores with the mux-only
    recorder preserves the WebRTC/control isolation without creating that
    single-core bottleneck.

    Below 8 cores there's no room to dedicate an IK core, so ``ik`` shares the
    control group; on 4-5 cores the relay also shares the background group (still
    kept off the control cores). ``None`` when partitioning isn't applicable.
    """
    n = os.cpu_count()
    if not n or n < _MIN_CORES:
        return None
    if n >= 8:
        # CPU0 services the Jetson's xHCI interrupt (both USB CAN adapters and
        # cameras) and is therefore a poor place for a motor deadline. Put the
        # Rust bus loops on the last two cores and leave the housekeeping CPUs
        # to throughput-tolerant dataset work.
        can = {n - 2, n - 1}
        rt = {2}
        ik = {3}
        relay = {4, 5}
        bg = {0, 1}
    elif n >= 6:
        can = {0, 1}
        rt = ik = {2}
        relay = {3, 4}
        bg = set(range(5, n))
    else:  # 4-5 cores: isolate control only; relay + dataset share the rest
        can = rt = ik = {0, 1}
        relay = bg = set(range(2, n))
    return {
        "can": can,
        "realtime": rt,
        "ik": ik,
        "relay": relay,
        "background": bg,
    }


def pin_realtime() -> bool:
    """Pin the calling process to the realtime cores (the control loop + threads)."""
    return _pin("realtime")


def pin_ik() -> bool:
    """Pin the calling process to the IK core(s).

    On 8+ cores this is a single core dedicated to the out-of-process JAX solver,
    isolating it from the control process so recording load can't preempt it
    mid-solve. On smaller hosts the ``ik`` group collapses onto the control cores,
    so this is equivalent to :func:`pin_realtime`.
    """
    return _pin("ik")


def pin_ik_startup() -> bool:
    """Widen the IK worker across the control-side cores for its one-time startup.

    The worker's first act — before it sends its ``ready`` handshake — is a heavy,
    one-shot cost: JAX/XLA compilation plus the up-to-200-iteration rest-pose
    settle and the collision-aware startup trajectory. Confining all of that to the
    single dedicated :func:`pin_ik` core roughly triples its wall time and blows the
    caller's 60s connect handshake (a ``TimeoutError`` that fails robot connect).
    This startup runs before the control loop or any recording has begun, so the
    realtime cores sit idle — let the compile spread across ``realtime`` ∪ ``ik``,
    then :func:`pin_ik` narrows the steady-state solve loop back to the dedicated
    core once ``ready`` is sent. On <8-core hosts ``ik`` already collapses onto the
    realtime cores, so this is the same set :func:`pin_ik` would use.
    """
    groups = core_groups()
    if groups is None:
        return False
    return _apply(groups["realtime"] | groups["ik"], "ik-startup")


def pin_relay() -> bool:
    """Pin the calling process to the relay cores (headset encode + WebRTC send)."""
    return _pin("relay")


def pin_background() -> bool:
    """Pin the calling process to the background cores (dataset recorder + gst)."""
    return _pin("background")


def isolate_relay_cpu() -> bool:
    """Separate relay Python from GStreamer and give gst throughput headroom.

    The relay's latency-critical work — the aiortc WebRTC send (SRTP + sendto for
    every stream) and the encoded-AU pull loops — is all Python, so the GIL
    serializes it onto effectively one core no matter what. What runs *truly* in
    parallel is GStreamer's C thread pool (camera capture, NVENC dispatch, and
    while recording the dataset raw-branch's VIC resize + shmsink copy), which
    holds no GIL. On a shared core those C threads preempt the send thread the
    moment recording starts, and the feed stutters (event-loop maxlag 100-385ms,
    send ~5000->~1400 pkt/s at 0% loss).

    Pin every Python thread — enumerated via :mod:`threading`, and kept together
    so the GIL never crosses cores — to ``relay[0]``. Pin every other thread in
    the process (GStreamer's C workers, which do not surface as Python threads)
    to the remaining relay cores plus ``background``. The send owns a core the
    recording branch cannot touch, while dozens of gst tasks can make forward
    progress through a short scheduler stall instead of overflowing a two-frame
    source queue. Call once after the gst pipelines are PLAYING and before the
    send loop runs. Best-effort: a no-op without ``sched_setaffinity`` or
    ``/proc``, or when the relay group has fewer than two cores.
    """
    if not hasattr(os, "sched_setaffinity"):
        return False
    groups = core_groups()
    if groups is None:
        return False
    relay = sorted(groups["relay"])
    if len(relay) < 2:
        return False
    py_core = {relay[0]}
    # The recorder is mux-only on the production H.264 transport, so sharing
    # its throughput CPUs is far safer than serializing ~80 gst workers onto
    # relay[1].  Exclude py_core for small-host layouts where relay/background
    # intentionally overlap.
    gst_cores = (set(relay[1:]) | groups["background"]) - py_core
    if not gst_cores:
        return False
    import threading

    py_tids = {t.native_id for t in threading.enumerate() if t.native_id is not None}
    try:
        for tid in py_tids:
            os.sched_setaffinity(tid, py_core)  # type: ignore[attr-defined]
    except OSError as exc:
        _logger.debug("could not pin relay python threads to %s: %s", py_core, exc)
        return False
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return False
    moved = 0
    for entry in tasks:
        try:
            tid = int(entry)
        except ValueError:
            continue
        if tid in py_tids:
            continue
        try:
            os.sched_setaffinity(tid, gst_cores)  # type: ignore[attr-defined]
            moved += 1
        except OSError:
            pass  # thread may have exited between listdir and the pin
    _logger.info(
        "isolated relay CPU: python threads -> core %d, %d gst threads -> cores %s",
        relay[0],
        moved,
        sorted(gst_cores),
    )
    return True


def _pin(group: str) -> bool:
    groups = core_groups()
    if groups is None:
        return False
    return _apply(groups[group], group)


def _apply(cores: set[int], label: str) -> bool:
    try:
        os.sched_setaffinity(0, cores)  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:  # AttributeError: no sched_* (macOS)
        _logger.debug("could not set CPU affinity to %s: %s", sorted(cores), exc)
        return False
    _logger.info("pinned to %s cores %s", label, sorted(cores))
    return True
