"""Jetson system tweaks for the real-time teleop / data-collection loops.

Tegra defaults trade latency for power/throughput and hurt us:

* **Engine devfreq** — the camera relay's hardware encode path
  (``almond_axol.video.hw_video``) depends on NVENC (the H.264 encoder) and
  the VIC (``nvvidconv``'s colorspace conversion). The default
  ``tegra_wmark`` governor grants just enough clock to keep up with the
  frame rate, so each frame takes nearly a full frame-time to encode
  (~3x worse per-frame latency at the ~25% clock it settles on). The GPU is
  in the same boat: the ZED SDK does its per-frame image processing in CUDA,
  and the default ``nvhost_podgov`` governor parks the GPU at its floor
  (306 of 918 MHz on Orin) and only ramps after the load is already late —
  every camera frame drop we attributed on a station happened at that floor.

* **CPU cpufreq** — the IK solver (JAX/XLA) is a bursty, sleep-heavy
  workload. The default ``schedutil`` governor reads that idle and
  underclocks the cores to ~40-70% of max, which drops the IK rate by a
  matching ~30% (measured 79 Hz vs 113 Hz pinned).

* **Capture daemon scheduling** — on ZED X (GMSL2) every camera's frames
  flow through NVIDIA's Argus daemon, whose capture-request loop runs per
  frame for all sensors in one process. It ships as a plain CFS service, so
  under load it is descheduled behind the control / IK / recorder work the
  cameras feed, and all cameras miss the same frames at once (the relay's
  own capture threads then report ~0 ms CPU wait: they are waiting on the
  daemon, not on the scheduler). :func:`pin_realtime_clocks` runs it
  ``SCHED_FIFO`` one notch above those relay capture threads, confined to
  the same camera cores (``affinity.realtime_camera_cores``) so a
  real-time daemon never lands on a control, IK, CAN, or interrupt core.

* **CAN interrupt placement** — both USB CAN adapters hang off one xHCI
  controller whose interrupt the GIC delivers to CPU0, a core the camera
  relay's worker pool also uses. Every motor reply crosses that CPU's
  interrupt bottom half (URB giveback, NET_RX softirq), and any
  ``SCHED_FIFO`` thread runnable there delays it: with the relay's capture
  chain real-time the core faulted 10 s after arming on a customer robot
  (2026-09-02), and raising ``ksoftirqd/0`` above the camera priorities did
  not help. Steering the interrupt onto the highest CAN core did — the whole
  receive path then runs beside its consumer, where no camera or dataset
  thread is ever scheduled, and a 160 s full-load recording finished with
  zero missed replies outside load transitions. :func:`pin_realtime_clocks`
  applies that steering (``affinity.can_irq_cpu``) at every boot; the
  interrupt number is resolved from the CAN interfaces' USB bus, never
  hard-coded.

The clock ceilings are themselves capped by the ``nvpmodel`` power mode, so
:func:`pin_realtime_clocks` first selects MAXN (mode 0) to uncap them, then
pins the engine and CPU clocks to that max — fixing the latency / rate.
Best-effort and cleared on reboot, so ``axol jetson.setup`` (which calls
:func:`pin_realtime_clocks`) is run at boot from the host installer's systemd
unit — not from teleop / serve.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from ..constants import CAN_LEFT, CAN_RIGHT
from .affinity import CAPTURE_FIFO_PRIORITY, can_irq_cpu, realtime_camera_cores
from .sudo import prime_sudo

_logger = logging.getLogger(__name__)

# Tegra-only encode engines; their presence identifies a Jetson when the L4T
# release file is missing (see :func:`_is_jetson`).
_JETSON_ENGINE_GLOBS = ("*.nvenc", "*.vic")

# Hardware engines whose devfreq clocks the camera path depends on: the
# encode engines plus the GPU the ZED SDK's CUDA processing runs on. The GPU
# is deliberately not a Jetson marker — other ARM SoCs expose a ``*.gpu``
# devfreq node too.
_ENGINE_CLOCK_GLOBS = (*_JETSON_ENGINE_GLOBS, "*.gpu")

# The camera capture daemon(s) in every ZED X frame's path, and the systemd
# drop-in that makes their realtime scheduling survive a daemon restart. The
# live instance is re-scheduled in place with ``chrt`` (threads it spawns
# later inherit), so the daemon is never restarted under running cameras.
_CAPTURE_DAEMON_UNITS = ("nvargus-daemon.service",)
_CAPTURE_DAEMON_DROPIN = "50-axol-realtime.conf"
# One notch above the relay's capture threads (they block on the daemon),
# far below the axol-rt CAN loops (SCHED_FIFO 20).
_CAPTURE_DAEMON_FIFO_PRIORITY = CAPTURE_FIFO_PRIORITY + 1
_SYSTEMD_UNIT_DIR = Path("/etc/systemd/system")

# The arm-hub CAN interfaces. Their ``/sys/class/net/<if>/device`` link names
# the USB bus they hang off, and that bus's host controller owns the
# ``/proc/interrupts`` row (``xhci-hcd:usbN``) every motor reply arrives on.
_CAN_ARM_INTERFACES = (CAN_LEFT, CAN_RIGHT)
_USB_HOST_CONTROLLER = "xhci-hcd"
_PROC_ROOT = Path("/proc")
_SYS_ROOT = Path("/sys")

# ``/proc/<tid>/stat`` policy value for SCHED_FIFO (sched.h SCHED_FIFO == 1).
_SCHED_FIFO = 1

# cpufreq governor that holds the cores at their max clock. The ceiling it
# holds them at is whatever the active ``nvpmodel`` power mode allows, so we
# select MAXN first (see :func:`_set_max_power_mode`) for the full benefit.
_CPU_GOVERNOR = "performance"

# nvpmodel power mode that uncaps the clock ceiling. MAXN is mode 0 on every
# Jetson, so the governor and engine pins can reach the real max clocks.
_MAXN_MODE = "0"

# Where nvpmodel persists the active mode across reboots (``pmode:%.4d``).
# The nvpmodel boot service runs ``nvpmodel -f /etc/nvpmodel.conf`` with no
# ``-m``, which reads the mode from this file and applies it — before the GPU
# golden context exists, so no reboot prompt. Writing the desired mode here is
# therefore the way to make a mode take effect on the next boot without
# rebooting now (see :func:`_set_max_power_mode`).
_NVPMODEL_STATUS = Path("/var/lib/nvpmodel/status")

# Canonical L4T marker present on every Jetson. CPU-governor pinning is gated
# on Jetson detection so ``jetson.setup`` on a non-Tegra Linux host never
# touches that machine's system-wide cpufreq governor (engine pinning is
# already implicitly Jetson-only — its devfreq globs are empty elsewhere).
_JETSON_RELEASE = Path("/etc/nv_tegra_release")


def _combine_output(proc: subprocess.CompletedProcess[str]) -> str:
    """Merge a command's stdout + stderr (a prompt may land on either)."""
    return "\n".join(
        s for s in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if s
    )


def _is_jetson() -> bool:
    """True on NVIDIA Jetson (L4T) hardware, False on any other host."""
    if _JETSON_RELEASE.exists():
        return True
    # Fallback: the encode engines we pin only exist on Tegra, so their
    # presence also identifies a Jetson even if the release file is missing.
    # (``glob`` returns a generator that is always truthy, so it must be
    # consumed — ``any(glob(...))`` — to test whether it actually matched.)
    return any(
        any(Path("/sys/class/devfreq").glob(pattern))
        for pattern in _JETSON_ENGINE_GLOBS
    )


class _RootEscalator:
    """Writes sysfs values / runs commands, escalating to ``sudo`` once.

    The hosted install runs as root (direct writes succeed); a CLI user
    may not be, so the first failed operation primes sudo credentials (a tty
    prompt when ``interactive``) and every operation after that uses
    ``sudo -n``. Priming happens at most once, and only when something
    actually needs changing.

    Each operation returns ``(ok, detail)`` — on failure ``detail`` carries the
    captured error so callers report the real cause (a genuine command/write
    failure under root) instead of always assuming root was missing.
    """

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self._primed = False

    def _prime(self) -> None:
        if self._interactive and not self._primed:
            self._primed = prime_sudo()

    def write(self, path: Path, value: str) -> tuple[bool, str]:
        """Write ``value`` to ``path`` as root; return ``(ok, failure_detail)``.

        ``failure_detail`` is the captured error from the failing attempt so the
        caller can report *why* it failed (a real write error vs. a missing
        privilege) rather than always blaming root.
        """
        try:
            path.write_text(value)
            return True, ""
        except OSError as exc:
            detail = str(exc)
        self._prime()
        proc = subprocess.run(
            ["sudo", "-n", "tee", str(path)],
            input=value,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or "").strip() or detail

    def run(
        self, argv: list[str], *, input_text: str | None = None
    ) -> tuple[bool, str]:
        """Run ``argv`` as root (direct, else ``sudo -n``); return ``(ok, detail)``.

        ``input_text`` is fed to the command's stdin so the caller controls the
        answer to any confirmation prompt (e.g. ``nvpmodel`` asking to reboot
        before a mode switch -- which must be declined, never auto-confirmed)
        instead of the command blocking on a tty or aborting on EOF. ``detail``
        is the failing attempt's captured output, so a non-permission failure is
        reported accurately rather than as "need root".
        """
        try:
            proc = subprocess.run(
                argv, input=input_text, capture_output=True, text=True
            )
            if proc.returncode == 0:
                return True, ""
            detail = _combine_output(proc)
        except OSError as exc:
            detail = str(exc)
        self._prime()
        sudo = subprocess.run(
            ["sudo", "-n", *argv], input=input_text, capture_output=True, text=True
        )
        if sudo.returncode == 0:
            return True, ""
        return False, _combine_output(sudo) or detail


def _query_power_mode(nvpmodel: str) -> str | None:
    """Return the mode id ``nvpmodel -q`` reports, or ``None`` when unreadable.

    ``nvpmodel -q`` prints the active mode id on its last non-empty line.
    """
    try:
        query = subprocess.run([nvpmodel, "-q"], capture_output=True, text=True)
    except OSError:
        return None
    lines = [ln.strip() for ln in query.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _set_max_power_mode(escalator: _RootEscalator) -> None:
    """Select the MAXN ``nvpmodel`` power mode (uncaps the clock ceiling).

    Jetson-only and best-effort. MAXN is the prerequisite for the CPU
    governor and engine pins below: ``nvpmodel`` caps the clock ceiling, so
    without MAXN the ``performance`` governor merely holds the cores at a
    lower mode's max. A no-op when already in MAXN (so no needless sudo
    prompt) or when ``nvpmodel`` is absent.
    """
    if not _is_jetson():
        _logger.debug("not a Jetson; leaving the nvpmodel power mode unchanged")
        return
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel is None:
        _logger.debug("nvpmodel not found; leaving the power mode unchanged")
        return
    # Skip the (root-only) switch when the mode already reports MAXN — either
    # live, or persisted for the next boot by a previous run (see below).
    if _query_power_mode(nvpmodel) == _MAXN_MODE:
        return
    # Answer "n" to any confirmation prompt. Once the GPU golden context
    # exists (always, by the time the installer or the boot ExecStartPre gets
    # here — nvpmodel.service runs earlier in boot), switching to MAXN asks to
    # reboot *now* (``DO YOU WANT TO REBOOT NOW? enter YES/yes to confirm:``).
    # We must never reboot the box here -- jetson.setup runs mid-install (over
    # the operator's SSH session) and at boot, and an in-place reboot would
    # drop that session and restart the robot. Feeding stdin also stops the
    # interactive `axol jetson.setup` run from blocking on the prompt.
    ok, detail = escalator.run([nvpmodel, "-m", _MAXN_MODE], input_text="n\n")
    if _query_power_mode(nvpmodel) == _MAXN_MODE:
        _logger.info("set Jetson power mode to MAXN (nvpmodel -m %s)", _MAXN_MODE)
        return
    if ok or "reboot" in detail.lower():
        # Declining the reboot prompt CANCELS the switch — nvpmodel records
        # nothing, so left alone the mode would never change, on this boot or
        # any later one. Persist the mode ourselves in nvpmodel's status file:
        # the boot service applies the mode saved there before the GPU golden
        # context exists, so MAXN takes effect cleanly on the next *natural*
        # reboot (and the earlier -q short-circuit keeps re-runs quiet).
        pending = f"pmode:{int(_MAXN_MODE):04d}"
        wrote, write_detail = escalator.write(_NVPMODEL_STATUS, pending)
        if wrote:
            _logger.warning(
                "Jetson power mode MAXN needs a reboot to take effect — declined "
                "the in-place reboot so this session/robot isn't restarted, and "
                "recorded MAXN in %s so it applies on the next reboot (the boot "
                "service re-pins the clocks then). The engine/CPU pins below "
                "still help at the current mode's ceiling.",
                _NVPMODEL_STATUS,
            )
        else:
            _logger.warning(
                "Jetson power mode MAXN needs a reboot, and recording it for the "
                "next boot failed (%s) — fix manually with: sudo nvpmodel -m %s "
                "(confirm the reboot prompt, or reboot afterwards).",
                write_detail or "write failed",
                _MAXN_MODE,
            )
    else:
        _logger.warning(
            "cannot set the Jetson power mode to MAXN (nvpmodel -m %s failed%s) — "
            "the active mode caps the clock ceiling the performance governor and "
            "engine pins can reach. Fix manually with: sudo nvpmodel -m %s",
            _MAXN_MODE,
            f": {detail}" if detail else "",
            _MAXN_MODE,
        )


def _pin_engines(writer: _RootEscalator) -> None:
    """Set ``min_freq = max_freq`` on the NVENC/VIC/GPU devfreq nodes."""
    for pattern in _ENGINE_CLOCK_GLOBS:
        for node in Path("/sys/class/devfreq").glob(pattern):
            try:
                max_freq = (node / "max_freq").read_text().strip()
                if (node / "min_freq").read_text().strip() == max_freq:
                    continue
            except OSError as exc:
                _logger.warning("cannot read %s clock state: %s", node.name, exc)
                continue
            ok, detail = writer.write(node / "min_freq", max_freq)
            if ok:
                _logger.info("pinned %s clock to %s Hz", node.name, max_freq)
            else:
                _logger.warning(
                    "cannot pin %s to its max clock (%s) — the camera path "
                    "(hardware encode, ZED SDK CUDA processing) runs at the "
                    "governor's floor and drops frames under load. Fix manually "
                    "with: echo %s | sudo tee %s",
                    node.name,
                    detail or "write failed",
                    max_freq,
                    node / "min_freq",
                )


def _service_main_pid(unit: str) -> int:
    """The unit's MainPID per systemd, 0 when inactive/unknown."""
    try:
        proc = subprocess.run(
            ["systemctl", "show", "-p", "MainPID", "--value", unit],
            capture_output=True,
            text=True,
        )
    except OSError:
        return 0
    try:
        return int(proc.stdout.strip() or 0)
    except ValueError:
        return 0


def _threads_at_fifo(
    pid: int, priority: int, *, proc_root: Path = Path("/proc")
) -> bool | None:
    """True when every thread of ``pid`` already runs SCHED_FIFO at ``priority``.

    ``None`` when the process cannot be inspected (gone, or unreadable).
    """
    tasks = sorted((proc_root / str(pid) / "task").glob("*"))
    if not tasks:
        return None
    for task in tasks:
        try:
            stat = (task / "stat").read_text()
        except OSError:
            return None
        # Fields after the parenthesised comm (which may contain spaces):
        # index 37 is rt_priority, 38 the scheduling policy (0-based from
        # the field following ")", i.e. proc(5) fields 40 and 41).
        fields = stat.rsplit(")", 1)[1].split()
        try:
            rt_priority, policy = int(fields[37]), int(fields[38])
        except (IndexError, ValueError):
            return None
        if policy != _SCHED_FIFO or rt_priority != priority:
            return False
    return True


def _parse_cpu_list(text: str) -> set[int]:
    """``"0-2,5"`` (the kernel's / ``taskset -c`` list syntax) → ``{0,1,2,5}``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _cpu_list(cores: set[int]) -> str:
    """``{1, 5}`` → ``"1,5"`` for ``taskset -c``."""
    return ",".join(str(c) for c in sorted(cores))


def _threads_on_cpus(
    pid: int, cores: set[int], *, proc_root: Path = _PROC_ROOT
) -> bool | None:
    """True when every thread of ``pid`` is confined to exactly ``cores``.

    ``None`` when the process cannot be inspected (gone, or unreadable).
    """
    tasks = sorted((proc_root / str(pid) / "task").glob("*"))
    if not tasks:
        return None
    for task in tasks:
        try:
            status = (task / "status").read_text()
        except OSError:
            return None
        allowed = None
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                allowed = _parse_cpu_list(line.split(":", 1)[1])
                break
        if allowed is None:
            return None
        if allowed != cores:
            return False
    return True


def _can_usb_buses(*, sys_root: Path = _SYS_ROOT) -> set[str]:
    """USB bus numbers the arm-hub CAN interfaces are enumerated on.

    ``/sys/class/net/<if>/device`` resolves to the interface's USB function
    directory, ``<bus>-<port[.port...]>:<config>.<iface>`` (e.g.
    ``1-2.2:1.0``); the leading number is the bus, i.e. the host controller.
    Interfaces that are absent (adapter unplugged, other host) are skipped.
    """
    buses: set[str] = set()
    for iface in _CAN_ARM_INTERFACES:
        try:
            name = (sys_root / "class/net" / iface / "device").resolve().name
        except OSError:
            continue
        bus = name.split(":", 1)[0].split("-", 1)[0]
        if bus.isdigit():
            buses.add(bus)
    return buses


def _can_usb_irqs(
    *, proc_root: Path = _PROC_ROOT, sys_root: Path = _SYS_ROOT
) -> dict[int, str]:
    """``{irq: action}`` for the host controller(s) the CAN adapters hang off.

    Rows of ``/proc/interrupts`` whose action names ``xhci-hcd:usb<bus>`` for
    one of :func:`_can_usb_buses`. When no CAN interface can be resolved to a
    bus (adapters unplugged at setup time, or a host that enumerates them
    differently) every ``xhci-hcd`` row is returned instead, so a Jetson still
    gets its USB interrupts off the camera cores. Empty when the table is
    unreadable or names no such controller.
    """
    try:
        lines = (proc_root / "interrupts").read_text().splitlines()
    except OSError:
        return {}
    buses = _can_usb_buses(sys_root=sys_root)
    wanted = {f"{_USB_HOST_CONTROLLER}:usb{bus}" for bus in buses}
    found: dict[int, str] = {}
    for line in lines[1:]:
        irq, sep, _rest = line.partition(":")
        irq = irq.strip()
        if not sep or not irq.isdigit():
            continue
        action = line.split()[-1]
        if not action.startswith(_USB_HOST_CONTROLLER):
            continue
        if wanted and action not in wanted:
            continue
        found[int(irq)] = action
    return found


def _irq_affinity(irq: int, *, proc_root: Path = _PROC_ROOT) -> set[int] | None:
    """CPUs ``irq`` may currently be delivered to; ``None`` if unreadable."""
    try:
        return _parse_cpu_list(
            (proc_root / "irq" / str(irq) / "smp_affinity_list").read_text()
        )
    except (OSError, ValueError):
        return None


def _irqbalance_active() -> bool:
    """True when the ``irqbalance`` service is running (it would undo our steering)."""
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return False
    try:
        proc = subprocess.run(
            [systemctl, "is-active", "--quiet", "irqbalance.service"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return proc.returncode == 0


def _steer_can_irq(escalator: _RootEscalator) -> None:
    """Deliver the CAN adapters' USB-controller interrupt to a CAN core.

    Jetson-only and best-effort. ``/proc/irq/<n>/smp_affinity_list`` is
    per-boot state, so ``jetson.setup`` re-applies it from the service's
    ``ExecStartPre`` after every reboot; the interrupt number is looked up
    each time (see :func:`_can_usb_irqs`). The target core is
    :func:`affinity.can_irq_cpu`; see :mod:`almond_axol.utils.affinity` for
    why this, and not a ``ksoftirqd`` priority, is what keeps real-time camera
    work from stalling both arms' motor feedback.
    """
    if not _is_jetson():
        _logger.debug("not a Jetson; leaving interrupt affinity alone")
        return
    target = can_irq_cpu()
    if target is None:
        _logger.debug("no CAN core partition on this host; not steering interrupts")
        return
    irqs = _can_usb_irqs(proc_root=_PROC_ROOT, sys_root=_SYS_ROOT)
    if not irqs:
        _logger.warning(
            "no %s interrupt found in /proc/interrupts; the CAN adapters' replies "
            "stay on the kernel's default CPU, where real-time camera work can "
            "delay them",
            _USB_HOST_CONTROLLER,
        )
        return
    for irq, action in sorted(irqs.items()):
        if _irq_affinity(irq, proc_root=_PROC_ROOT) == {target}:
            _logger.info(
                "irq %d (%s) already on CPU %d (a CAN core)", irq, action, target
            )
            continue
        path = _PROC_ROOT / "irq" / str(irq) / "smp_affinity_list"
        ok, detail = escalator.write(path, f"{target}\n")
        if ok and _irq_affinity(irq, proc_root=_PROC_ROOT) == {target}:
            _logger.info(
                "irq %d (%s) -> CPU %d: CAN replies now arrive on a CAN core, "
                "off the camera cores",
                irq,
                action,
                target,
            )
        else:
            _logger.warning(
                "cannot steer irq %d (%s) to CPU %d (%s) — real-time camera work "
                "on its CPU can stall both arms' CAN feedback. Fix manually with: "
                "echo %d | sudo tee %s",
                irq,
                action,
                target,
                detail or "affinity unchanged after write",
                target,
                path,
            )
    if _irqbalance_active():
        _logger.warning(
            "irqbalance is running and may move the CAN adapters' interrupt back "
            "onto a camera core; disable it with: sudo systemctl disable --now "
            "irqbalance"
        )


def _prioritize_capture_daemons(escalator: _RootEscalator) -> None:
    """Run the camera capture daemon(s) SCHED_FIFO, now and after restarts.

    Jetson-only and best-effort. Two halves per unit: a systemd drop-in so
    the policy applies whenever the daemon (re)starts, and ``chrt -a`` /
    ``taskset -a`` on the live process so it applies right now without
    restarting the daemon under running cameras (threads it creates later
    inherit both).

    A real-time daemon must not roam: confined to the camera cores
    (:func:`affinity.realtime_camera_cores`) it cannot preempt the Python
    control loop or IK, nor sit on the CPU the CAN adapters' interrupt lands
    on before :func:`_steer_can_irq` moves it, where a FIFO thread delays the
    interrupt's bottom half and with it both arms' feedback.
    """
    if not _is_jetson():
        _logger.debug("not a Jetson; leaving the camera daemons' scheduling alone")
        return
    cores = realtime_camera_cores()
    dropin_text = (
        "# Installed by `axol jetson.setup`: every ZED X camera frame passes\n"
        "# through this daemon, so it must not be descheduled behind the\n"
        "# control/IK/recorder load the cameras feed. It is confined to the\n"
        "# camera cores: a real-time thread on the CAN adapters' interrupt CPU\n"
        "# would stall both arms' motor feedback.\n"
        "[Service]\n"
        "CPUSchedulingPolicy=fifo\n"
        f"CPUSchedulingPriority={_CAPTURE_DAEMON_FIFO_PRIORITY}\n"
    )
    if cores is not None:
        dropin_text += f"CPUAffinity={' '.join(str(c) for c in sorted(cores))}\n"
    for unit in _CAPTURE_DAEMON_UNITS:
        pid = _service_main_pid(unit)
        dropin = _SYSTEMD_UNIT_DIR / f"{unit}.d" / _CAPTURE_DAEMON_DROPIN
        if pid == 0 and not dropin.exists():
            # Not installed on this station (no Argus daemon => not ZED X).
            _logger.debug("%s not running; skipping its realtime drop-in", unit)
            continue
        try:
            dropin_current = dropin.read_text() == dropin_text
        except OSError:
            dropin_current = False
        if not dropin_current:
            ok, detail = escalator.run(["mkdir", "-p", str(dropin.parent)])
            if ok:
                ok, detail = escalator.write(dropin, dropin_text)
            if ok:
                ok, detail = escalator.run(["systemctl", "daemon-reload"])
            if ok:
                _logger.info(
                    "%s: SCHED_FIFO %d on future starts (%s)",
                    unit,
                    _CAPTURE_DAEMON_FIFO_PRIORITY,
                    dropin,
                )
            else:
                _logger.warning(
                    "cannot install the realtime drop-in for %s (%s) — the "
                    "capture daemon stays CFS after its next restart and all "
                    "cameras drop the same frames under load. Fix manually "
                    "with: sudo systemctl edit %s (CPUSchedulingPolicy=fifo, "
                    "CPUSchedulingPriority=%d)",
                    unit,
                    detail or "failed",
                    unit,
                    _CAPTURE_DAEMON_FIFO_PRIORITY,
                )
        if pid == 0:
            continue
        if not _threads_at_fifo(pid, _CAPTURE_DAEMON_FIFO_PRIORITY):
            ok, detail = escalator.run(
                ["chrt", "-f", "-a", "-p", str(_CAPTURE_DAEMON_FIFO_PRIORITY), str(pid)]
            )
            if ok:
                _logger.info(
                    "%s (pid %d) -> SCHED_FIFO %d",
                    unit,
                    pid,
                    _CAPTURE_DAEMON_FIFO_PRIORITY,
                )
            else:
                _logger.warning(
                    "cannot re-schedule the running %s (pid %d) SCHED_FIFO (%s) — it "
                    "stays CFS until its next restart picks up the drop-in. Fix "
                    "manually with: sudo chrt -f -a -p %d %d",
                    unit,
                    pid,
                    detail or "chrt failed",
                    _CAPTURE_DAEMON_FIFO_PRIORITY,
                    pid,
                )
        if cores is None or _threads_on_cpus(pid, cores):
            continue
        ok, detail = escalator.run(
            ["taskset", "-a", "-c", "-p", _cpu_list(cores), str(pid)]
        )
        if ok:
            _logger.info("%s (pid %d) -> cores %s", unit, pid, sorted(cores))
        else:
            _logger.warning(
                "cannot confine the running %s (pid %d) to cores %s (%s) — a "
                "real-time daemon on the CAN interrupt CPU can stall motor "
                "feedback until its next restart picks up the drop-in. Fix "
                "manually with: sudo taskset -a -c -p %s %d",
                unit,
                pid,
                sorted(cores),
                detail or "taskset failed",
                _cpu_list(cores),
                pid,
            )


def _pin_cpu(writer: _RootEscalator) -> None:
    """Switch every online CPU to the ``performance`` cpufreq governor.

    Jetson-only: gated on :func:`_is_jetson` so running ``jetson.setup`` on a
    non-Tegra Linux host (which may also expose cpufreq) never changes that
    machine's system-wide governor.
    """
    if not _is_jetson():
        _logger.debug("not a Jetson; leaving the CPU cpufreq governor unchanged")
        return
    pinned = 0
    failed = 0
    failed_detail: str | None = None
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        gov = cpu / "cpufreq" / "scaling_governor"
        try:
            if gov.read_text().strip() == _CPU_GOVERNOR:
                continue
        except OSError:
            continue  # offline core or no cpufreq (not a throttled Jetson)
        ok, detail = writer.write(gov, _CPU_GOVERNOR)
        if ok:
            pinned += 1
        else:
            failed += 1
            failed_detail = detail
    if pinned:
        _logger.info("pinned %d CPU core(s) to the %s governor", pinned, _CPU_GOVERNOR)
    if failed:
        # Report the count (not just the last core's error): a single line that
        # named only cpuN understated how many cores actually failed. EINVAL
        # here is usually the clock-ceiling cap of a non-MAXN power mode, which
        # clears once MAXN is active (it may be pending a reboot — the boot
        # service re-pins then), so point there before the manual override.
        _logger.warning(
            "could not set %d CPU core(s) to the %s governor (last error: %s) — "
            "the schedutil default underclocks bursty control loops (~30%% lower "
            "IK rate). This usually clears once the MAXN power mode is active "
            "(it may be pending a reboot; the boot service re-pins then). If it "
            "persists, fix manually with: echo %s | sudo tee "
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
            failed,
            _CPU_GOVERNOR,
            failed_detail or "write failed",
            _CPU_GOVERNOR,
        )


def pin_engine_clocks(*, interactive: bool = False) -> None:
    """Pin NVENC and VIC to their max clock (devfreq ``min_freq = max_freq``).

    Best-effort: direct sysfs write when root, ``sudo -n`` otherwise, and a
    warning with the manual command when neither works. A no-op on machines
    without these devfreq nodes (anything that isn't a Jetson). Cleared on
    reboot, so it runs whenever the hardware encoder is installed.

    With ``interactive=True``, escalation may prompt for the sudo password
    once on the tty (via :func:`prime_sudo`) — only when a clock actually
    needs pinning. Use from CLI entry points; never mid-session.
    """
    _pin_engines(_RootEscalator(interactive=interactive))


def pin_realtime_clocks(*, interactive: bool = False) -> None:
    """Select MAXN, pin clocks, and set the real-time scheduling the loops need.

    Selects the MAXN ``nvpmodel`` power mode (uncaps the clock ceiling), pins
    NVENC/VIC/GPU (encode latency, ZED SDK processing), switches the CPUs to
    the ``performance`` governor (IK rate), steers the CAN adapters'
    USB-controller interrupt onto a CAN core (so real-time camera work can
    never stall motor feedback), and schedules the Argus camera daemon
    ``SCHED_FIFO`` on the camera cores (all-camera frame drops under load).
    All are Jetson-only: MAXN selection, CPU-governor pinning, interrupt
    steering and daemon scheduling are gated on :func:`_is_jetson` so they
    never alter a non-Tegra host, and engine pinning is a no-op without the
    Tegra devfreq nodes. MAXN is selected first because it sets the ceiling
    the governor and engine pins reach; the interrupt step precedes the daemon
    step so the CAN receive path is off the camera cores before any camera
    thread becomes real-time. Same best-effort / ``interactive`` escalation
    semantics as :func:`pin_engine_clocks`; sudo is primed at most once across
    all of them. Invoked via ``axol jetson.setup`` (host installer + boot
    service), not from the teleop / collect-data / serve entry points.
    """
    escalator = _RootEscalator(interactive=interactive)
    _set_max_power_mode(escalator)
    _pin_engines(escalator)
    _pin_cpu(escalator)
    _steer_can_irq(escalator)
    _prioritize_capture_daemons(escalator)
