"""CPU-core isolation for the real-time / latency-critical work during recording.

During ``collect-data`` the box runs five kinds of work that contend for cores:

* **can** — the two Rust 240 Hz CAN loops. On every partitioned host (4+
  cores, Jetson or Raspberry Pi 5) each arm owns a dedicated core; the Rust
  process pins the bus threads individually and runs them ``SCHED_FIFO``.
* **realtime** — the Python 120 Hz target loop plus its web/VR/teleop and
  IK-dispatch threads. It has a separate core from CAN, so Python or camera
  activity cannot delay a motor tick. Every control-loop command pins it
  (``teleop`` too, since 2026-09-15 — unpinned, its loop floated onto the
  FIFO CAN cores / the IK core and ran one tick in twelve 15–65 ms late).
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
from pathlib import Path

_logger = logging.getLogger(__name__)

# Below this many cores there's nothing to gain from partitioning, so isolation
# is skipped (the groups collapse onto whatever's available).
_MIN_CORES = 4

# The kernel's online CPU list; the layout is built over these IDs.
_CPU_ONLINE = "/sys/devices/system/cpu/online"


def core_groups() -> dict[str, set[int]] | None:
    """Role → core set: ``can``, ``realtime``, ``ik``, ``relay``, ``background``,
    ``recorder``, ``camera`` and ``irq``.

    Based on the machine's *physical* core count, NOT the process's current
    affinity: the control process pins itself before spawning the relay/recorder,
    and those children inherit its restricted mask — they must still compute the
    full partition and ``sched_setaffinity`` to their own group (allowed even from
    a restricted mask). Reading the inherited mask would wrongly see only the
    realtime cores.

    The supported compute modules, by online core count:

    ===========================  ==  =====  ==  ==  =====  ========  =====  =======
    Host                         n   can    rt  ik  relay  bg        rec    camera
    ===========================  ==  =====  ==  ==  =====  ========  =====  =======
    Raspberry Pi 5               4   2-3    1   1   0      0         0      -
    Orin NX 16GB, AGX Orin 32GB  8   6-7    2   3   4-5    0-1       0-1    0-1,5
    AGX Orin 64GB / Industrial   12  10-11  2   3   4-5    0-1,6-9   8-9    0-1,5-7
    Thor T5000                   14  12-13  2   3   4-5    0-1,6-11  10-11  0-1,5-9
    ===========================  ==  =====  ==  ==  =====  ========  =====  =======

    (``n`` = online cores, ``bg`` = ``background``, ``rec`` = ``recorder``.)
    ``relay`` holds the relay's Python core (its lowest CPU) plus a GStreamer
    core; ``background`` is every throughput core, where the relay's CFS
    GStreamer pool may run (:func:`isolate_relay_cpu`). ``recorder`` is where
    the dataset recorder and the Rust trace writers live, and ``camera`` is
    where ``SCHED_FIFO`` camera work may (:func:`realtime_camera_cores`, which
    also drops CPU0 while the CAN interrupt can still land there).

    ``os.cpu_count()`` is the *online* count, so a Jetson in an ``nvpmodel``
    mode that offlines cores gets the layout for what it actually has;
    ``jetson.setup`` selects the max mode, which onlines every core. The table
    is by position in the online list (:func:`_online_cpus`), which is the
    CPU number itself whenever the online CPUs are ``0..n-1`` — every mode
    seen so far offlines from the top. A mode that offlined a CPU in the
    middle would otherwise hand CAN or control a CPU that is not there, and
    the pin would fail.

    8+ cores: CAN 2 / Python control 1 / IK 1 / relay 2 / dataset the rest.
    Each CAN arm gets one of the final two cores, away from CPU0: the Jetson
    routes its xHCI interrupt there, and both USB CAN adapters plus the
    cameras traverse that controller. Dataset work gets CPUs 0-1 because it
    tolerates those interrupts. This also avoids the old shared ``realtime``
    layout where camera/WebRTC bookkeeping made 5-15% of nominal 240 Hz motor
    ticks late. ``ik`` remains a dedicated core so it cannot deschedule CAN or
    Python.

    On the bigger modules (AGX Orin 64GB, Thor T5000) every core past the
    8-core layout goes to throughput work, keeping the CAN pair last and the
    control, IK and relay cores where they are on the Orin NX. The
    latency-critical roles keep one core each (a FIFO bus loop per arm, the
    GIL-bound control and relay-send loops, a ~1-core IK solve); the side
    that ran short on 8 cores is throughput. There the recorder shares every
    background core with the FIFO camera set (the relay's capture chain and
    the Argus daemon), and FIFO preempts CFS unconditionally, so the recorder
    gets only what the cameras leave — ~5 % of each core on the policy ops
    (2026-09-14), which is why :func:`pin_background_and_ik` lends it the IK
    core. With cores to spare, the two below CAN become ``recorder`` and are
    left out of ``camera``: no FIFO thread is ever scheduled there, so the
    recorder's CPU is guaranteed rather than borrowed. The CFS GStreamer pool
    may still use them — CFS shares fairly with the recorder, and the mux-only
    recorder needs ~40 % of one core, so fencing them off entirely would idle
    most of two CPUs. Every other extra core joins ``camera``. Before this
    layout a 12- or 14-core host used the 8-core groups unchanged and left the
    middle cores unassigned.

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
    keeps real-time camera work off the ``irq`` CPU for as long as the
    interrupt can still land there, so the two subsystems stay decoupled
    when that steering has not (yet) been applied — and hands the CPU back
    to the camera pool once it has.

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

    4-5 cores (Raspberry Pi 5 hosts): the two CAN cores are still carved out
    and kept disjoint from ``realtime``, because that disjointness is what
    lets ``rt.link`` hand the Rust core a CPU per bus *and* request
    ``SCHED_FIFO`` for it. The earlier layout put CAN and Python control on
    the same pair with no pinning and no real-time class, so the 240 Hz bus
    loops ran as ordinary CFS threads next to the target stream, the CSV
    logger and the dashboard. Measured on a Pi 5 ROM soak (2026-09-03): a
    43.8 ms overrun sent the core limp mid-sweep, and every tick more than
    0.5 ms late had already switched the shoulder host damping off for that
    tick (``serve.rs`` gates ``damp_ok`` on on-time ticks), which chops the
    only damper the shoulders have into torque transients. The Pi has no
    room for anything else: control, IK and the CAN cores take three CPUs,
    and the interrupt CPU carries all throughput work (relay + dataset). A
    FIFO camera pool cannot exist there — :func:`realtime_camera_cores`
    returns ``None`` — but Pi hosts run no ZED cameras anyway.
    """
    n = os.cpu_count()
    if not n or n < _MIN_CORES:
        return None
    # The Jetson's xHCI interrupt lands on CPU0 until jetson.setup steers it
    # (its nominal mask says all CPUs; the GIC picks CPU0); every layout keeps
    # SCHED_FIFO camera work off it until then (realtime_camera_cores).
    irq = {0}
    if n >= 8:
        # CPU0 takes the Jetson's xHCI interrupt by default (both USB CAN
        # adapters and cameras) and is therefore a poor place for a motor
        # deadline. Put the Rust bus loops on the last two cores and leave the
        # housekeeping CPUs to throughput-tolerant dataset work. Every core
        # between the relay and CAN (6-9 on an AGX Orin, 6-11 on a Thor
        # T5000) is throughput work too: the latency-critical roles need one
        # core each on every host.
        can = {n - 2, n - 1}
        rt = {2}
        ik = {3}
        relay = {4, 5}
        bg = {0, 1} | set(range(6, n - 2))
        # 12+ cores: the two below CAN are the recorder's, free of FIFO
        # camera work. 8-11 cores have no spare pair; the recorder shares the
        # background cores and borrows IK (pin_background_and_ik).
        recorder = {n - 4, n - 3} if n >= 12 else bg
    elif n >= 6:
        can = {0, 1}
        rt = ik = {2}
        relay = {3, 4}
        bg = set(range(5, n))
        recorder = bg
    else:
        # 4-5 cores: the bus loops get the last two cores (pinned + FIFO via
        # rt.link, exactly as on 8+), Python control/IK one core, and the
        # remaining CPU(s) — including the interrupt CPU, which only CFS
        # work may share — take the relay and dataset throughput work.
        can = {n - 2, n - 1}
        rt = ik = {1}
        relay = bg = recorder = set(range(0, n - 2)) - rt
    # FIFO camera work: the relay's GStreamer core plus the background cores,
    # never the relay's Python core (small layouts overlap relay and
    # background) and never a dedicated recorder core.
    relay_py = {min(relay)}
    camera = (relay | bg) - relay_py
    if recorder != bg:
        camera -= recorder
    ids = _online_cpus(n)
    groups = {
        "can": can,
        "realtime": rt,
        "ik": ik,
        "relay": relay,
        "background": bg,
        "recorder": recorder,
        "camera": camera,
        "irq": irq,
    }
    return {role: {ids[i] for i in cores} for role, cores in groups.items()}


def _online_cpus(n: int) -> list[int]:
    """The online CPU IDs, ascending; ``0..n-1`` when they can't be read.

    ``n`` is ``os.cpu_count()``, which glibc reads from the same kernel list,
    so the two agree except for a CPU changing state between the reads (or a
    ``PYTHON_CPU_COUNT`` override); then the plain ``0..n-1`` numbering is used,
    exactly as before this lookup existed.
    """
    try:
        text = Path(_CPU_ONLINE).read_text().strip()
        online: set[int] = set()
        for part in text.split(","):
            lo, _, hi = part.partition("-")
            online.update(range(int(lo), int(hi or lo) + 1))
    except (OSError, ValueError):
        return list(range(n))
    return sorted(online) if len(online) == n else list(range(n))


def describe_layout() -> str:
    """One line naming this host's core partition, for provisioning logs."""
    groups = core_groups()
    n = os.cpu_count() or 0
    if groups is None:
        return f"{n} cores online: too few to partition, nothing is pinned"
    labels = (
        ("can", "CAN"),
        ("realtime", "control"),
        ("ik", "IK"),
        ("relay", "relay"),
        ("recorder", "recorder"),
        ("camera", "camera"),
    )
    parts = [f"{label} {_cpu_ranges(groups[key]) or '-'}" for key, label in labels]
    return f"{n} cores online: " + ", ".join(parts)


def _cpu_ranges(cores: set[int]) -> str:
    """``{0, 1, 5, 6, 7}`` → ``"0-1,5-7"``."""
    spans: list[list[int]] = []
    for cpu in sorted(cores):
        if spans and cpu == spans[-1][1] + 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in spans)


def realtime_camera_cores() -> set[int] | None:
    """Cores where ``SCHED_FIFO`` camera work is allowed to run.

    The ``camera`` group — the relay's throughput cores (everything but its
    Python core) plus the background cores, i.e. the pool
    :func:`isolate_relay_cpu` gives the CFS GStreamer workers, less any
    dedicated ``recorder`` cores (12+ core hosts, see :func:`core_groups`) —
    **minus** the ``irq`` CPU while the CAN adapters' interrupt can still be
    delivered there. The capture chain the relay
    elevates (:func:`prioritize_capture_threads`) and the Argus daemon
    ``jetson.setup`` elevates both live here, so neither can sit on the CPU
    the CAN replies arrive on (see :func:`core_groups`), nor land on a
    control, IK, or CAN core.

    Once ``jetson.setup`` has steered that interrupt onto a CAN core
    (:func:`can_irq_cpu`, checked live via ``jetson.can_irq_cpus``), the
    ``irq`` CPU is an ordinary throughput core again and rejoins the pool; if
    the interrupt has been moved onto a camera core instead, that core is the
    one left out (:func:`_can_irq_cpus_to_avoid`).
    Confining the whole FIFO set — some 30 relay threads plus the Argus
    daemon — to two cores instead of three left the CFS work sharing those
    cores (the NVENC feed threads, the recorder) preempted for long
    stretches: on 2026-09-03 a recording lost frames at a dataset encoder's
    input queue whose feed thread spent 20 % of its time runnable-but-waiting,
    while no camera source skipped an exposure. Unknown interrupt placement
    (no ``/proc`` row, unreadable affinity, not a Jetson) keeps the CPU
    excluded, exactly as before. ``None`` when partitioning isn't applicable
    or the pool would be empty.
    """
    groups = core_groups()
    if groups is None:
        return None
    cores = set(groups["camera"]) - _can_irq_cpus_to_avoid(groups)
    return cores or None


def _can_irq_cpus_to_avoid(groups: dict[str, set[int]]) -> set[int]:
    """The CPUs the CAN adapters' interrupt is delivered to, conservatively.

    Where it is pinned to a CPU or two (``jetson.setup``'s steering onto a CAN
    core, or wherever ``irqbalance`` or an operator put it since), exactly
    those. Where that is unknown (no ``/proc`` row, unreadable affinity, not a
    Jetson) or the mask is wide (the unsteered default: the GIC then delivers
    to CPU0), the ``irq`` CPU. A FIFO camera thread on the interrupt's CPU
    stalls both arms' feedback, so the pool leaves it out whichever it is —
    before, only CPU0 was ever left out, and an interrupt moved onto a camera
    core (``irqbalance`` rebalances every 10 s) went unnoticed.
    """
    # utils.jetson imports this module at top level; import lazily.
    from .jetson import can_irq_cpus

    delivered = can_irq_cpus()
    if delivered is None or len(delivered) > len(groups["can"]):
        return set(groups["irq"])
    return delivered


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
    """Pin the calling process to the background (throughput) cores."""
    return _pin("background")


def pin_recorder() -> bool:
    """Pin the calling process to the recorder cores (dataset recorder steady state).

    Two cores of its own that no ``SCHED_FIFO`` camera thread may use on 12+
    core hosts (AGX Orin 64GB, Thor T5000); the background cores elsewhere,
    where this is equivalent to :func:`pin_background`.
    """
    return _pin("recorder")


def pin_background_and_ik() -> bool:
    """Pin the recorder subprocess across ``background`` ∪ ``ik``.

    Two uses, both about the recorder being the only *CFS* work left on the
    background cores. Everything else there is real-time: the capture daemon
    (``nvargus-daemon``, SCHED_FIFO 6, pinned to the camera cores by
    ``jetson.setup``; ~75 % of a core for four 60 fps GMSL sources) and the
    relay's exposure-critical GStreamer threads (:func:`prioritize_capture_threads`)
    — so the recorder gets exactly what they leave, which on the policy ops
    (three VIC branches per camera: dataset encode, policy ring, headset
    stream) is ~5 % of each of cores 0/1/5 (measured 2026-09-14 with
    ``run-policy``'s panel configuration, no headset). Two consequences:

    * **Startup.** The recorder's first act is a one-shot ~25 s of CPU:
      importing torch and lerobot. On the 2026-09-14 DAgger session that
      stretched to 56 s and the ``ready`` handshake timed out (the session
      ended before the operator could start an episode); the ZED SDK's own CFS
      grab threads shared the same leftovers and dropped ~10 % of exposures for
      as long as the import ran, which the headset saw as a stuttering feed.
      Every recorder runs its imports widened: the IK core is idle then (the IK
      worker's own compile — :func:`pin_ik_startup` — is over, and the solves
      only run while an operator drives).
    * **Steady state, policy ops only.** The mux-only recorder needs ~40 % of a
      core for four 60 fps H.264 streams (`shmsrc` → Python pull → lerobot row
      → mp4 mux). Narrowed back to the background cores it sustained 55-60
      rows/s with dips to 35, so its per-source AU queue (120 frames = 2 s)
      grew from the first second and tripped ``encoded-AU backlog exceeded``
      25-30 s into every take — the take is discarded and the arms return
      home: the "reset in the middle of the episode" the 2026-09-14 sessions
      showed. Kept on the IK core as well (``share_ik_core`` in the recorder
      config, set by the ops that run the policy ring), CFS wake placement
      puts its threads on the idle core: 3605/3611 rows in two 60 s takes,
      backlog 1-2, cores 0/1 down from 95 % to 83 %. ``collect-data`` keeps
      the narrow steady state (:func:`pin_background`): its relay runs two
      branches per camera, its recorder was never short, and the IK worker on
      that core drives the arms at 100 Hz for the whole session.

    Only a dedicated IK core is borrowed: on hosts where ``ik`` collapses onto
    the control core (<8 cores) the recorder stays on ``background`` — control
    never shares a CPU with throughput work.

    Nothing is borrowed where the recorder has cores of its own (12+ cores,
    see :func:`core_groups`): no FIFO camera thread runs there, so neither the
    import nor the policy-op steady state is starved, and this pins the
    recorder cores — the IK core stays the IK worker's, which on
    ``collect-dagger`` drives the operator's interventions.
    """
    groups = core_groups()
    if groups is None:
        return False
    if groups["recorder"] != groups["background"]:
        return _apply(groups["recorder"], "recorder")
    spare_ik = groups["ik"] - groups["realtime"]
    return _apply(groups["background"] | spare_ik, "background+ik")


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
#   CONTROL_FIFO_PRIORITY (10)  the Python 120 Hz control thread
#   axol-rt CAN loops     (20)  AXOL_RT_FIFO_PRIORITY, rt.link
# Camera work sits above every CFS thread so a capture wake-up never queues
# behind the encode/mux workers; the control thread sits above camera work in
# case a small layout ever puts them on one core (it feeds motor targets); the
# CAN loops outrank everything and run on disjoint cores anyway. The top rung
# is also what the persistent rtprio grant (utils.rtprio, LimitRTPRIO in the
# service unit) allows a non-root launcher.
CAPTURE_FIFO_PRIORITY = 5
CONTROL_FIFO_PRIORITY = 10
MAX_FIFO_PRIORITY = 20

# Opt back into the pre-2026-09-17 behaviour: warn about a CFS control thread
# and run anyway. For a dev box or a demo where the hitching is acceptable and
# arranging the rtprio grant is not worth it; never for data collection.
ALLOW_CFS_CONTROL_ENV = "AXOL_ALLOW_CFS_CONTROL"


class ControlSchedulingError(RuntimeError):
    """The control thread was denied ``SCHED_FIFO`` on a host that offers it.

    Raised instead of quietly continuing because the degraded mode is hard to
    spot and easy to mistake for a code regression: the loop still runs, the
    Rust core still arms, and the only symptom is the arms hitching — which
    also disappears when the cameras are off, so it reads as mechanical.
    ``axol-rt`` already refuses to arm in the same situation (see
    ``configure_bus_scheduling`` in ``rust/axol-rt/src/serve.rs``); this makes
    the Python half consistent with it.

    The two halves fail apart because they get the privilege from different
    places: ``axol rt.install`` puts ``cap_sys_nice`` on the ``axol-rt``
    *binary*, so the core is unaffected by how the operator logged in, while
    the Python control thread has only the launching session's
    ``RLIMIT_RTPRIO``. A login that bypasses PAM therefore produces a box
    whose CAN loops are real-time and whose control loop is not.
    """


def prioritize_control_thread(*, required: bool = True) -> bool:
    """Run the calling thread ``SCHED_FIFO`` so it outranks its core-mates.

    The realtime core is one CPU on every layout, and the control loop shares
    it with everything :func:`pin_realtime` drags along: the VR pose thread
    (pose ingest + SSL websocket), the IK dispatch thread, the diagnostics
    scanners and whatever spills over from the IK worker's XLA pool. All of
    them are CFS ``nice 0`` and CFS shares the core evenly, so the biggest
    consumer — the 120 Hz tick itself at ~35 % of the core — waits as long as
    it runs. Measured on the ZED box on 2026-09-15 with the headset streaming
    (``/proc/<tid>/schedstat``): control thread 320-370 ms/s on CPU and
    300-340 ms/s runnable-but-waiting, the vr-server 140 ms/s, ik-loop 115,
    diag 30, XLA spill 60; the flight recorder saw one command tick in fifty
    15-53 ms late, felt as the arms hitching. A FIFO thread preempts its CFS
    core-mates the moment it wakes, exactly the treatment ``axol-rt`` gives
    the CAN loops and the relay gives camera capture. It still blocks like
    any other thread (select, the GIL, socket writes), so the others run in
    the ~65 % of the core it leaves free.

    Thread-scoped: ``sched_setscheduler(0, ...)`` on Linux acts on the calling
    thread. ``SCHED_RESET_ON_FORK`` is set so threads and processes this one
    spawns afterwards (IK dispatch, flight-recorder dumps, asyncio executors,
    the relay and IK worker processes) start ``SCHED_OTHER``/``nice 0`` — the
    kernel applies the reset on every clone, threads included (verified on
    5.15). A platform with no ``sched_setscheduler``/reset flag stays a silent
    no-op — nothing was on offer there, which matches ``axol-rt``: it too
    insists only when the launcher actually asked for a priority. A host that
    *does* offer real-time scheduling and refuses the request raises
    :exc:`ControlSchedulingError`, because that is the case worth stopping
    for. Pair with :func:`release_control_thread` when the loop ends so a
    long-lived serve worker thread is not left FIFO.

    Args:
        required: Refuse (raise) when the host offers ``SCHED_FIFO`` but
            denies it. ``False`` keeps the old best-effort behaviour — warn
            once and stay CFS — for a caller that genuinely tolerates a
            non-real-time control thread. Operators get the same escape hatch
            without a code change via ``AXOL_ALLOW_CFS_CONTROL=1``.

    Returns:
        True when the thread is now ``SCHED_FIFO``; False when the platform
        had nothing to offer, or when the denial was tolerated.

    Raises:
        ControlSchedulingError: The host offers ``SCHED_FIFO`` and denied it
            while ``required`` and without the env escape hatch.
    """
    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_FIFO"):
        return False
    reset_on_fork = getattr(os, "SCHED_RESET_ON_FORK", None)
    if reset_on_fork is None:
        # Without the reset flag every thread spawned from here would inherit
        # FIFO — the 1 kHz IK-dispatch poll above the control loop itself.
        return False
    param = os.sched_param(CONTROL_FIFO_PRIORITY)  # type: ignore[attr-defined]
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO | reset_on_fork, param)  # type: ignore[attr-defined]
    except PermissionError as exc:
        if required and os.environ.get(ALLOW_CFS_CONTROL_ENV) != "1":
            raise ControlSchedulingError(
                f"control thread cannot enter SCHED_FIFO "
                f"{CONTROL_FIFO_PRIORITY} ({exc}); refusing to run without "
                "real-time scheduling. Pinned but CFS it shares its core with "
                "the VR pose, IK-dispatch and diagnostics threads and lands "
                "about one tick in fifty 15-65 ms late, felt as the arms "
                "hitching and lunging — and masked by turning the cameras "
                "off, which is what makes it look mechanical. The rtprio "
                "grant `axol provision` writes is applied by pam_limits, so "
                "it reaches a PAM login only: a session that bypasses PAM "
                "(Tailscale SSH, or a systemd unit without LimitRTPRIO) never "
                "receives it. Log in again over ssh, or run `sudo prlimit "
                f"--pid $$ --rtprio={MAX_FIFO_PRIORITY}:{MAX_FIFO_PRIORITY}` "
                f"in this shell. Set {ALLOW_CFS_CONTROL_ENV}=1 to run "
                "degraded anyway."
            ) from exc
        _logger.warning(
            "control thread stays SCHED_OTHER (no CAP_SYS_NICE: %s); expect "
            "late control ticks while the VR/IK threads share its core. Run "
            "`axol provision` and log in again (or `sudo prlimit --pid $$ "
            "--rtprio=%d:%d` in this shell) so it may use SCHED_FIFO",
            exc,
            MAX_FIFO_PRIORITY,
            MAX_FIFO_PRIORITY,
        )
        return False
    except OSError as exc:
        _logger.debug("could not make the control thread SCHED_FIFO: %s", exc)
        return False
    _logger.info("control thread -> SCHED_FIFO %d", CONTROL_FIFO_PRIORITY)
    return True


def enter_control_thread(*, required: bool = True) -> bool:
    """Make the calling thread *the* control thread: realtime core + ``SCHED_FIFO``.

    For loops that live on a thread of their own rather than the command's
    calling thread — ``AxolRobot``'s ``axol-event-loop``, which runs
    ``motion_control`` for ``collect-data`` / ``collect-dagger`` (their hot
    loop is scheduled onto it) and ``run-policy`` (whose 60 Hz thread hands
    each action to it). Call it first thing on that thread: the pin is
    thread-scoped (a no-op re-pin when the process already sits on the
    realtime cores, the one thread moved there when it does not — run-policy
    leaves its observation/inference threads free to float), and the FIFO
    policy is thread-scoped with the reset-on-fork flag as in
    :func:`prioritize_control_thread`. Returns True when either took effect,
    and propagates :exc:`ControlSchedulingError` when the host offers
    ``SCHED_FIFO`` but denies it — the control loop of ``collect-data`` /
    ``collect-dagger`` / ``run-policy`` is exactly where a silently CFS thread
    does its damage, so it refuses rather than collect hitched data. Pass
    ``required=False`` to keep the old best-effort behaviour.
    """
    pinned = pin_realtime()
    fifo = prioritize_control_thread(required=required)
    return pinned or fifo


def release_control_thread() -> None:
    """Undo :func:`prioritize_control_thread` for the calling thread."""
    if not hasattr(os, "sched_setscheduler") or not hasattr(os, "SCHED_OTHER"):
        return
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))  # type: ignore[attr-defined]
    except OSError:
        pass


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
