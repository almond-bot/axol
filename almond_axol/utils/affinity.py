"""CPU-core isolation for the real-time / latency-critical work during recording.

During ``collect-data`` the box runs four kinds of work that contend for cores:

* **realtime** — the control process: the 120 Hz loop plus its web/VR/teleop and
  IK-dispatch threads. Background work landing on its cores stalls it (arm jerk).
* **ik** — the out-of-process JAX IK solver (a ~1-core solve). On 8+ cores it
  gets a dedicated core so recording load can't preempt it mid-solve (which drops
  its rate ~115 -> ~80 Hz); on smaller hosts it shares the realtime cores.
* **relay** — the headset video: gst NVENC encode + the WebRTC (aiortc) *send*
  loop, which is latency-sensitive. In teleop it has its cores to itself and the
  feed is clean; once recording starts, the dataset raw-branch piles onto the
  same cores and starves the send — packets go out late and bursty (0% loss but
  rising jitter), so the live feed gets laggy + grainy.
* **background** — the dataset recorder + its per-camera NVENC encoders. Pure
  throughput; tolerant of an occasional dropped frame.

Partitioning the cores by role keeps each group off the others': the control
loop never gets preempted (no jerk), IK solves at full rate, and the relay's send
gets prompt CPU like it does in teleop (clean feed), while the dataset encode
runs on its own cores.

``pin_realtime`` / ``pin_ik`` / ``pin_relay`` / ``pin_background`` apply the
partition to the calling process (new threads inherit it; ``subprocess`` children
inherit the relay/recorder affinity). Best-effort and self-gating: a no-op on
machines with too few cores or without ``sched_setaffinity`` (e.g. macOS), so
off-Jetson dev is unaffected.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning, so isolation
# is skipped (the groups collapse onto whatever's available).
_MIN_CORES = 4


def core_groups() -> dict[str, set[int]] | None:
    """``{"realtime", "ik", "relay", "background"}`` → core sets, or ``None``.

    Based on the machine's *physical* core count, NOT the process's current
    affinity: the control process pins itself before spawning the relay/recorder,
    and those children inherit its restricted mask — they must still compute the
    full partition and ``sched_setaffinity`` to their own group (allowed even from
    a restricted mask). Reading the inherited mask would wrongly see only the
    realtime cores.

    8+ cores: control 2 / ik 1 / relay 2 / dataset rest. The ``realtime`` group is
    the control process (the 120 Hz loop + web/VR/teleop threads); ``ik`` is a
    *dedicated* core for the out-of-process JAX IK solver. IK is a ~1-core solve,
    and while recording the control cores fill up (loop + status polls + the frame
    bookkeeping), so on a shared core the solver keeps its CPU share but gets
    descheduled mid-solve — its wall-time-per-solve stretches and its rate sags
    (~115 -> ~80 Hz). A dedicated core removes that preemption and, by moving the
    solver off the control cores, also de-contends the control loop.

    The relay gets *two* cores so :func:`isolate_relay_cpu` can split its Python
    work (the aiortc WebRTC send + encoded-AU pull loops, all GIL-serialized) onto
    one core and GStreamer's C thread pool (camera capture, NVENC dispatch, and the
    dataset raw-branch VIC resize + shmsink copy) onto the other. That split is the
    whole point: naively handing the relay a second *roaming* core does nothing —
    the Python side is GIL-bound and just ping-pongs the GIL across cores — but
    physically moving the lock-free C threads off the send's core stops the
    recording raw-branch from preempting the send (which otherwise starves it:
    event-loop maxlag 100-385ms, send collapsing ~5000->~1400 pkt/s at 0% loss).
    The dataset recorder keeps the remaining cores.

    Below 8 cores there's no room to dedicate an IK core, so ``ik`` shares the
    control group; on 4-5 cores the relay also shares the background group (still
    kept off the control cores). ``None`` when partitioning isn't applicable.
    """
    n = os.cpu_count()
    if not n or n < _MIN_CORES:
        return None
    if n >= 8:
        rt = {0, 1}
        ik = {2}
        relay = {3, 4}
        bg = set(range(5, n))
    elif n >= 6:
        rt = ik = {0, 1}
        relay = {2, 3}
        bg = set(range(4, n))
    else:  # 4-5 cores: isolate control only; relay + dataset share the rest
        rt = ik = {0, 1}
        relay = bg = set(range(2, n))
    return {"realtime": rt, "ik": ik, "relay": relay, "background": bg}


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
    """Split the relay's Python threads and GStreamer C threads onto separate cores.

    The relay's latency-critical work — the aiortc WebRTC send (SRTP + sendto for
    every stream) and the encoded-AU pull loops — is all Python, so the GIL
    serializes it onto effectively one core no matter what. What runs *truly* in
    parallel is GStreamer's C thread pool (camera capture, NVENC dispatch, and
    while recording the dataset raw-branch's VIC resize + shmsink copy), which
    holds no GIL. On a shared core those C threads preempt the send thread the
    moment recording starts, and the feed stutters (event-loop maxlag 100-385ms,
    send ~5000->~1400 pkt/s at 0% loss).

    So pin every Python thread — enumerated via :mod:`threading`, and kept together
    so the GIL never crosses cores — to ``relay[0]``, and every other thread in the
    process (GStreamer's C workers, which don't surface as Python threads but do
    appear under ``/proc/self/task``) to ``relay[1]``. The send then owns a core
    the recording raw-branch can't touch, so it stays as clean under recording as
    in teleop. Call once from the relay's event-loop (main) thread after the gst
    pipelines are PLAYING (all their threads exist) and before the send loop runs;
    Python threads spawned later (aiortc helpers) inherit this thread's ``relay[0]``
    affinity. Best-effort: a no-op without ``sched_setaffinity`` or ``/proc``, or
    when the relay group has fewer than two cores.
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
    gst_core = {relay[1]}
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
            os.sched_setaffinity(tid, gst_core)  # type: ignore[attr-defined]
            moved += 1
        except OSError:
            pass  # thread may have exited between listdir and the pin
    _logger.info(
        "isolated relay CPU: python threads -> core %d, %d gst threads -> core %d",
        relay[0],
        moved,
        relay[1],
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
