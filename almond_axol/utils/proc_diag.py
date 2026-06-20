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


class SystemDiag(threading.Thread):
    """Background ``/proc`` sampler: logs per-core saturation + hottest procs/threads.

    Each ~1s window it reports how many logical CPUs are saturated, overall busy
    %, the highest-CPU processes (labelled where the pid is known — e.g.
    main / ik / relay), and the calling process's hottest threads (by
    ``/proc/self/task/*/comm`` name). CPU is in single-core %, so a process using
    three cores reads ~300%.

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

    def run(self) -> None:
        prev_cpu = read_percpu()
        prev_procs = self._scan_pids()
        prev_thr = self._scan_threads()
        prev_t = time.perf_counter()
        while not self._stop.wait(self._period):
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            if dt <= 0:
                continue
            # Scanning a few hundred /proc files every second is pure overhead
            # unless someone is watching the debug stream — skip it otherwise.
            if not self._logger.isEnabledFor(logging.DEBUG):
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

            self._logger.debug(
                "diag cpu: cores>85%%=%d/%d  sys=%.0f%%  | procs: %s",
                busy_cores,
                ncpu,
                sys_pct,
                top_procs,
            )
            self._logger.debug("diag main-threads: %s", top_thr)
