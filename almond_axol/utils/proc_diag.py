"""Dependency-free ``/proc`` CPU sampler for the control-loop flows.

Shared by ``axol teleop`` and ``axol collect-data`` so both report the same
per-core saturation + hottest process/thread breakdown and can be compared
directly. Runs on its own thread (reading a few hundred ``/proc`` files every
second would perturb the hot control loop) and is Linux-only (best-effort
elsewhere: it simply logs nothing if ``/proc`` is unavailable).
"""

from __future__ import annotations

import logging
import os
import threading
import time

CLK_TCK = float(os.sysconf("SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def read_percpu() -> dict[str, tuple[int, int]]:
    """Per-CPU ``(busy, total)`` jiffies from ``/proc/stat`` (excl. idle+iowait)."""
    out: dict[str, tuple[int, int]] = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line[:3] != "cpu" or line[3] == " ":
                    continue  # skip the "cpu " aggregate and non-cpu lines
                p = line.split()
                vals = [int(x) for x in p[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                total = sum(vals)
                out[p[0]] = (total - idle, total)
    except OSError:
        pass
    return out


def read_proc_cpu(pid: int) -> tuple[int, str] | None:
    """``(utime+stime jiffies, comm)`` for ``pid``, or ``None`` if it's gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return None
    rp = data.rfind(")")
    if rp < 0:
        return None
    comm = data[data.find("(") + 1 : rp]
    rest = data[rp + 2 :].split()
    try:  # fields after comm: state(0) ... utime(11) stime(12) (0-indexed in rest)
        return int(rest[11]) + int(rest[12]), comm
    except (IndexError, ValueError):
        return None


def read_proc_rss(pid: int) -> int:
    """Resident set size in bytes for ``pid`` (field 2 of ``/proc/<pid>/statm``)."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0
    return pages * PAGE_SIZE


def read_children(pid: int) -> list[int]:
    """Direct child pids of ``pid`` from the kernel's ``children`` file.

    ``/proc/<pid>/task/<pid>/children`` is a cheap space-separated pid list (one
    read, no full ``/proc`` walk). Returns ``[]`` when it's unavailable (older
    kernels without ``CONFIG_PROC_CHILDREN``, or the process is gone).
    """
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            return [int(x) for x in f.read().split()]
    except (OSError, ValueError):
        return []


def read_meminfo() -> tuple[int, int, int]:
    """``(mem_available, swap_free, swap_total)`` in bytes from ``/proc/meminfo``."""
    want = {"MemAvailable:": 0, "SwapFree:": 0, "SwapTotal:": 0}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if p and p[0] in want:
                    want[p[0]] = int(p[1]) * 1024  # values are in kB
    except (OSError, IndexError, ValueError):
        pass
    return want["MemAvailable:"], want["SwapFree:"], want["SwapTotal:"]


def _gib(n: float) -> str:
    return f"{n / 1024**3:.1f}G"


class SystemDiag(threading.Thread):
    """Background ``/proc`` sampler: logs per-core saturation, memory, and CPU attribution.

    Two tiers each ~1s window:

    * **INFO (always):** per-core saturation (how many logical CPUs are >85%
      busy, overall busy %), system memory (available + swap), and CPU% / RSS for
      just the *labelled* processes (e.g. main / ik / relay) plus their direct
      children (the gst encode subprocesses). This is cheap — it reads a handful
      of ``/proc`` files, not the whole table — so it can run at the operator's
      default log level alongside the per-second loop-rate line.
    * **DEBUG (only when enabled):** the full system-wide breakdown — the
      highest-CPU process anywhere on the box and the calling process's hottest
      threads (by ``/proc/self/task/*/comm`` name), for naming the in-process GIL
      aggressor. This walks every ``/proc/<pid>``, so it stays gated.

    CPU is in single-core %, so a process using three cores reads ~300%.

    Args:
        labels: ``{pid: label}`` to make known subprocesses legible (mp spawn
            children all report ``comm=python``).
        logger: where to emit the two diag lines (so each flow logs under its
            own module logger).
        period: sample interval in seconds.
    """

    def __init__(
        self,
        labels: dict[int, str],
        logger: logging.Logger,
        period: float = 1.0,
    ) -> None:
        super().__init__(daemon=True, name="diag")
        self._labels = labels
        self._logger = logger
        self._period = period
        self._stop = threading.Event()
        self._main_pid = os.getpid()

    def stop(self) -> None:
        self._stop.set()

    def _scan_pids(self) -> dict[int, tuple[int, str]]:
        out: dict[int, tuple[int, str]] = {}
        try:
            pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
        except OSError:
            return out
        for pid in pids:
            r = read_proc_cpu(pid)
            if r is not None:
                out[pid] = r
        return out

    def _scan_threads(self) -> dict[int, tuple[int, str]]:
        out: dict[int, tuple[int, str]] = {}
        try:
            tids = os.listdir(f"/proc/{self._main_pid}/task")
        except OSError:
            return out
        for tid in tids:
            try:
                with open(f"/proc/{self._main_pid}/task/{tid}/stat") as f:
                    data = f.read()
                rp = data.rfind(")")
                rest = data[rp + 2 :].split()
                jif = int(rest[11]) + int(rest[12])
            except (OSError, IndexError, ValueError):
                continue
            try:
                with open(f"/proc/{self._main_pid}/task/{tid}/comm") as f:
                    name = f.read().strip()
            except OSError:
                name = tid
            out[int(tid)] = (jif, name)
        return out

    def _label(self, pid: int, comm: str) -> str:
        return self._labels.get(pid, comm)

    def _scan_labeled(self) -> dict[int, tuple[int, str]]:
        """``{pid: (jiffies, label)}`` for the labelled pids + their gst children.

        Reads only the known pids and one ``children`` file each — a handful of
        ``/proc`` files, not the whole table — so it is cheap enough for the INFO
        tier. Children inherit a ``<label>-gst`` name so the per-camera encode
        subprocesses are legible without a system-wide walk.
        """
        out: dict[int, tuple[int, str]] = {}
        for pid, label in self._labels.items():
            r = read_proc_cpu(pid)
            if r is not None:
                out[pid] = (r[0], label)
            for child in read_children(pid):
                cr = read_proc_cpu(child)
                if cr is not None:
                    out[child] = (cr[0], f"{label}-{cr[1]}")
        return out

    def run(self) -> None:
        prev_cpu = read_percpu()
        prev_labeled = self._scan_labeled()
        prev_procs = (
            self._scan_pids() if self._logger.isEnabledFor(logging.DEBUG) else {}
        )
        prev_thr = self._scan_threads()
        prev_t = time.perf_counter()
        while not self._stop.wait(self._period):
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt <= 0:
                continue

            cur_cpu = read_percpu()
            busy_cores = 0
            sys_acc = 0.0
            ncpu = 0
            for name, (busy, total) in cur_cpu.items():
                pb, pt = prev_cpu.get(name, (busy, total))
                dtot = total - pt
                if dtot <= 0:
                    continue
                frac = 100.0 * (busy - pb) / dtot
                ncpu += 1
                sys_acc += frac
                if frac > 85.0:
                    busy_cores += 1
            prev_cpu = cur_cpu
            sys_pct = sys_acc / ncpu if ncpu else 0.0

            # INFO tier: labelled procs (+ their gst children) CPU% / RSS + memory.
            cur_labeled = self._scan_labeled()
            lab_pct: list[tuple[float, str]] = []
            for pid, (jif, label) in cur_labeled.items():
                pj = prev_labeled.get(pid)
                if pj is None:
                    continue
                pct = 100.0 * (jif - pj[0]) / CLK_TCK / dt
                if pct >= 5.0:
                    lab_pct.append(
                        (pct, f"{label}={pct:.0f}%/{_gib(read_proc_rss(pid))}")
                    )
            prev_labeled = cur_labeled
            lab_pct.sort(reverse=True)
            top_labeled = "  ".join(s for _, s in lab_pct) or "n/a"

            mem_avail, swap_free, swap_total = read_meminfo()
            self._logger.info(
                "diag: cores>85%%=%d/%d sys=%.0f%%  %s  memavail=%s swap=%s/%s",
                busy_cores,
                ncpu,
                sys_pct,
                top_labeled,
                _gib(mem_avail),
                _gib(swap_total - swap_free),
                _gib(swap_total),
            )

            # DEBUG tier: full system-wide proc + main-thread breakdown (every
            # /proc/<pid> + every thread of this process — too costly for INFO).
            if not self._logger.isEnabledFor(logging.DEBUG):
                continue

            cur_procs = self._scan_pids()
            proc_pct: list[tuple[float, str]] = []
            for pid, (jif, comm) in cur_procs.items():
                pj = prev_procs.get(pid)
                if pj is None:
                    continue
                pct = 100.0 * (jif - pj[0]) / CLK_TCK / dt
                if pct >= 5.0:
                    proc_pct.append((pct, f"{self._label(pid, comm)}[{pid}]"))
            prev_procs = cur_procs
            proc_pct.sort(reverse=True)
            top_procs = "  ".join(f"{n}={p:.0f}%" for p, n in proc_pct[:6]) or "n/a"

            cur_thr = self._scan_threads()
            thr_pct: list[tuple[float, str]] = []
            for tid, (jif, name) in cur_thr.items():
                pj = prev_thr.get(tid)
                if pj is None:
                    continue
                pct = 100.0 * (jif - pj[0]) / CLK_TCK / dt
                if pct >= 5.0:
                    thr_pct.append((pct, name))
            prev_thr = cur_thr
            thr_pct.sort(reverse=True)
            top_thr = "  ".join(f"{n}={p:.0f}%" for p, n in thr_pct[:5]) or "n/a"

            self._logger.debug("diag procs (system-wide): %s", top_procs)
            self._logger.debug("diag main-threads: %s", top_thr)
