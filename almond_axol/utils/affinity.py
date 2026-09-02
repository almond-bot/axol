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
from collections.abc import Iterable

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning, so isolation
# is skipped (the groups collapse onto whatever's available).
_MIN_CORES = 4


def core_groups() -> dict[str, set[int]] | None:
    """``{"can", "realtime", "ik", "relay", "background", "irq"}`` → core sets.

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

    ``irq`` names the CPU the kernel delivers that interrupt to by default
    (CPU0 on every Jetson seen so far). Only *CFS* work may run there: a CAN
    reply reaches the socket through the interrupt's bottom half (URB
    giveback + NET_RX softirq) on that CPU, and any ``SCHED_FIFO`` userspace
    thread runnable there delays it for as long as it runs — measured
    2026-09-02 on a customer robot: the FIFO camera capture set on CPU0
    pushed *both arms'* replies past the 240 Hz reply window and faulted the
    core within 10 s of arming, and raising ``ksoftirqd/0`` above the camera
    priorities did not help. What did was moving the interrupt itself onto a
    CAN core (:func:`can_irq_cpu`), which ``jetson.setup`` does at every boot:
    the whole receive path then runs where its consumer already is and no
    camera thread is allowed. :func:`realtime_camera_cores` additionally
    keeps real-time camera work off the ``irq`` CPU, so the two subsystems
    stay decoupled even when that steering has not (yet) been applied.

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
    # The Jetson's xHCI interrupt lands on CPU0 until jetson.setup steers it
    # (its nominal mask says all CPUs; the GIC picks CPU0); every layout keeps
    # SCHED_FIFO camera work off it.
    irq = {0}
    if n >= 8:
        # CPU0 takes the Jetson's xHCI interrupt by default (both USB CAN
        # adapters and cameras) and is therefore a poor place for a motor
        # deadline. Put the Rust bus loops on the last two cores and leave the
        # housekeeping CPUs to throughput-tolerant dataset work.
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
        "irq": irq,
    }


def realtime_camera_cores() -> set[int] | None:
    """Cores where ``SCHED_FIFO`` camera work is allowed to run.

    The relay's throughput cores (everything but its Python core) plus the
    background cores — the same pool :func:`isolate_relay_cpu` gives the CFS
    GStreamer workers — **minus** the ``irq`` CPU. The capture chain the relay
    elevates (:func:`prioritize_capture_threads`) and the Argus daemon
    ``jetson.setup`` elevates both live here, so neither can ever sit on the
    CPU the CAN adapters' interrupt is delivered to before ``jetson.setup``
    steers it away (see :func:`core_groups`), nor land on a control, IK, or
    CAN core. ``None`` when partitioning isn't applicable or the pool would be
    empty.
    """
    groups = core_groups()
    if groups is None:
        return None
    relay = sorted(groups["relay"])
    py_core = {relay[0]} if relay else set()
    cores = (set(relay[1:]) | groups["background"]) - py_core - groups["irq"]
    return cores or None


def can_irq_cpu() -> int | None:
    """CPU the CAN adapters' USB-controller interrupt should be delivered to.

    The highest CAN core: the bus loop pinned there is the interrupt's
    consumer, so the hardirq, URB giveback, NET_RX softirq and socket wake-up
    all run on a core that carries nothing but an ``axol-rt`` ``SCHED_FIFO``
    thread and its own idle time. That thread spends most of each 4.17 ms
    period blocked on exactly this interrupt, so it and the bottom half never
    compete, and no camera or dataset thread is ever scheduled there (see
    :func:`core_groups`). ``None`` when the host has no CAN partition.

    This is the placement that ended the camera-versus-CAN coupling in the
    field (2026-09-02: 160 s of full-load recording with zero missed replies
    beyond load transitions, versus a fault 10 s after arming with the
    interrupt on CPU0). ``jetson.setup`` applies it per boot.
    """
    groups = core_groups()
    if groups is None or not groups["can"]:
        return None
    return max(groups["can"])


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


# The SCHED_FIFO ladder across the stack, lowest to highest:
#   CAPTURE_FIFO_PRIORITY  (5)  relay camera capture chain
#   capture daemon         (6)  nvargus-daemon, see utils.jetson
#   axol-rt CAN loops     (20)  AXOL_RT_FIFO_PRIORITY, rt.link
# Camera work sits above every CFS thread so a capture wake-up never queues
# behind the encode/mux workers; the CAN loops outrank everything and run on
# disjoint cores anyway. The top rung is also what the persistent rtprio grant
# (utils.rtprio, LimitRTPRIO in the service unit) allows a non-root launcher.
CAPTURE_FIFO_PRIORITY = 5
MAX_FIFO_PRIORITY = 20


def prioritize_capture_threads(thread_comms: Iterable[str]) -> int:
    """Move the camera *capture* chain to ``SCHED_FIFO`` so it never misses an exposure.

    The relay's gst pool is ~80 CFS threads (VIC copies, NVENC dispatch, shm
    writers) sharing a few cores with the dataset recorder. CFS hands them out
    round-robin, so whenever a burst of them is runnable together a thread can
    sit runnable-but-unscheduled for most of a scheduling period (~20 ms). For
    the encode/mux threads that is harmless — queues absorb it. For the capture
    chain it is not: the source streaming thread (the SDK ``grab`` + rectify +
    push), the ZED SDK's own worker threads (the V4L2 dequeue / frame assembly
    it spawns unnamed), and the consumers of the two-buffer queues that still
    hold un-copied camera surfaces (the stereo eye crops and the dataset VIC
    copies) must each run within one 60 Hz period or the exposure is gone —
    seen on the robot as ``skipped exposure(s)`` with ``CPU wait since
    previous frame`` of 14-26 ms while every other attribution (SDK/link
    time, GPU clock, arm motion) was clean. A real-time class fixes that at
    the root: a FIFO wake-up preempts the CFS pool immediately and its CPU
    wait is ~0 regardless of how many encoders are dispatching. Their combined
    load is small and bounded (a few percent of a core per camera; the VIC
    consumers mostly sleep on the hardware fence), so they cannot starve the
    recorder, and the kernel's RT throttle caps a runaway at 95 % of a core
    anyway.

    Elevates every thread whose ``comm`` is in ``thread_comms`` (GStreamer
    names task threads ``<element>:<pad>``, truncated to the kernel's 15-char
    limit — the caller passes them already truncated; see
    ``gst_zed.exposure_critical_thread_comms``) plus every non-Python thread
    that still carries the process's own ``comm`` — GStreamer, GLib, NVENC and
    CUDA all rename theirs, so an unrenamed thread in the relay is the SDK's.

    Every thread it elevates is also confined to :func:`realtime_camera_cores`.
    :func:`isolate_relay_cpu` leaves the gst pool free to use the CPU the CAN
    adapters' interrupt lands on by default, which is fine for CFS work but
    not for a FIFO thread: one runnable there delays the interrupt's bottom
    half and with it both arms' CAN feedback (see :func:`core_groups`). Call
    once after the pipelines are PLAYING (the threads exist by then). Returns
    the number of threads moved; ``0`` when the platform has no
    ``sched_setscheduler`` or the process lacks ``CAP_SYS_NICE`` / an rtprio
    allowance — a manual run from a shell without the ``axol provision``
    rtprio grant. That case is logged as a warning (it is the whole story
    behind an otherwise puzzling run of skipped exposures) and the threads
    stay CFS, exactly the previous behaviour.
    """
    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
        return 0
    import threading

    py_tids = {t.native_id for t in threading.enumerate() if t.native_id is not None}
    try:
        with open("/proc/self/comm") as fh:
            process_comm = fh.read().strip()
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return 0
    wanted = set(thread_comms) | {process_comm}
    param = os.sched_param(CAPTURE_FIFO_PRIORITY)  # type: ignore[attr-defined]
    cores = realtime_camera_cores() if hasattr(os, "sched_setaffinity") else None
    moved = 0
    denied: OSError | None = None
    for entry in tasks:
        try:
            tid = int(entry)
        except ValueError:
            continue
        if tid in py_tids:
            continue
        try:
            with open(f"/proc/self/task/{tid}/comm") as fh:
                comm = fh.read().strip()
        except OSError:
            continue  # exited between listdir and here
        if comm not in wanted:
            continue
        try:
            os.sched_setscheduler(tid, os.SCHED_FIFO, param)  # type: ignore[attr-defined]
            moved += 1
        except PermissionError as exc:
            denied = exc
            break
        except OSError:
            continue
        if cores is not None:
            try:
                os.sched_setaffinity(tid, cores)  # type: ignore[attr-defined]
            except OSError:
                pass  # exited; the FIFO call above already succeeded
    if denied is not None:
        _logger.warning(
            "camera capture threads stay SCHED_OTHER (no CAP_SYS_NICE: %s); "
            "expect skipped exposures under recording load. Run `axol provision` "
            "and log in again (or `sudo prlimit --pid $$ --rtprio=%d:%d` in this "
            "shell) so a manual `axol serve` may use SCHED_FIFO",
            denied,
            MAX_FIFO_PRIORITY,
            MAX_FIFO_PRIORITY,
        )
    elif moved:
        _logger.info(
            "camera capture threads -> SCHED_FIFO %d on cores %s (%d threads: %s "
            "+ SDK workers)",
            CAPTURE_FIFO_PRIORITY,
            sorted(cores) if cores is not None else "(unpinned)",
            moved,
            ", ".join(sorted(wanted - {process_comm})),
        )
    return moved


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
