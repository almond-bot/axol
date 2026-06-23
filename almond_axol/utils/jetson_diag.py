"""Jetson resource sampler built on the L4T ``tegrastats`` stream.

``SystemDiag`` (``almond_axol.utils.proc_diag``) covers CPU and memory from
``/proc``, but the dimensions that decide whether the control loop is starved by
the *GPU/encode engines* rather than the CPU — GR3D (GPU) load, EMC
(memory-bandwidth) load, NVENC clock, per-core frequency (did the pinned
``performance`` governor hold?), temperatures and thermal throttling — live in
version-fragile sysfs/debugfs nodes. ``tegrastats`` ships with every L4T and
normalizes all of them into one line per interval, so we parse that instead.

The sampler runs on its own thread, is Jetson-only (a clean no-op off-Tegra or
when ``tegrastats`` is absent), and logs one compact INFO line per period so the
record-start transition is visible next to the ``loop:`` / ``diag:`` lines. It
also re-checks the engine-clock and CPU-governor pins at runtime (the boot-time
``axol jetson.setup`` is the only thing that sets them, and nothing verified they
held under load).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path

from . import jetson

# Tolerant per-field patterns: each is optional, so an older L4T layout that
# omits a token simply drops it from the line rather than failing the parse.
_RE_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_RE_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_RE_EMC = re.compile(r"\bEMC_FREQ (\d+)%")
_RE_GR3D = re.compile(r"\bGR3D_FREQ (\d+)%")
_RE_NVENC = re.compile(r"\bNVENC (\d+)")
_RE_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_RE_CORE = re.compile(r"(\d+)%@(\d+)")
_RE_TEMP = re.compile(r"\b(\w+)@(-?[\d.]+)C")

# How often (in sample windows) to re-verify the clock/governor pins.
_PIN_CHECK_EVERY = 10


class TegraStatsDiag(threading.Thread):
    """Background ``tegrastats`` reader: logs GPU / EMC / NVENC / mem / freq / temps.

    Args:
        logger: where to emit the ``tegra:`` line (each flow uses its own module
            logger so the output groups with that flow's other diagnostics).
        period: sample interval in seconds (``tegrastats --interval``).
    """

    def __init__(self, logger: logging.Logger, period: float = 1.0) -> None:
        super().__init__(daemon=True, name="tegra-diag")
        self._logger = logger
        self._period = period
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    def run(self) -> None:
        tegrastats = shutil.which("tegrastats")
        if not jetson._is_jetson() or tegrastats is None:
            self._logger.debug("tegrastats unavailable; GPU/thermal diag disabled")
            return
        try:
            self._proc = subprocess.Popen(
                [tegrastats, "--interval", str(int(self._period * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            self._logger.debug("could not start tegrastats: %s", exc)
            return

        assert self._proc.stdout is not None
        window = 0
        try:
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                self._log_line(line)
                window += 1
                if window % _PIN_CHECK_EVERY == 1:
                    self._check_pins()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()

    def _log_line(self, line: str) -> None:
        parts: list[str] = []

        m = _RE_GR3D.search(line)
        parts.append(f"gr3d={m.group(1)}%" if m else "gr3d=?")
        m = _RE_EMC.search(line)
        parts.append(f"emc={m.group(1)}%" if m else "emc=?")
        m = _RE_NVENC.search(line)
        parts.append(f"nvenc={m.group(1)}MHz" if m else "nvenc=off")

        m = _RE_RAM.search(line)
        if m:
            parts.append(
                f"ram={int(m.group(1)) / 1024:.1f}/{int(m.group(2)) / 1024:.1f}G"
            )
        m = _RE_SWAP.search(line)
        if m:
            parts.append(
                f"swap={int(m.group(1)) / 1024:.1f}/{int(m.group(2)) / 1024:.1f}G"
            )

        m = _RE_CPU.search(line)
        if m:
            freqs = [int(f) for _, f in _RE_CORE.findall(m.group(1))]
            if freqs:
                parts.append(f"cpufreq={min(freqs)}-{max(freqs)}MHz")

        temps = [float(v) for _, v in _RE_TEMP.findall(line)]
        if temps:
            parts.append(f"tmax={max(temps):.1f}C")

        parts.append("throttle=yes" if "throttl" in line.lower() else "throttle=none")
        self._logger.info("tegra: %s", "  ".join(parts))

    def _check_pins(self) -> None:
        """WARN if the NVENC/VIC engine clocks or CPU governor aren't pinned.

        The pins are set at boot by ``axol jetson.setup``
        (:func:`jetson.pin_realtime_clocks`) and cleared on reboot; this verifies
        at runtime that they actually held, which directly affects encode latency
        and IK rate. Reuses the exact engine globs / governor from ``jetson``.
        """
        for pattern in jetson._ENGINE_CLOCK_GLOBS:
            for node in Path("/sys/class/devfreq").glob(pattern):
                try:
                    cur = (node / "cur_freq").read_text().strip()
                    mx = (node / "max_freq").read_text().strip()
                except OSError:
                    continue
                if cur != mx:
                    self._logger.warning(
                        "%s clock not pinned: cur=%s max=%s — encode latency ~3x "
                        "worse. Re-run `axol jetson.setup`.",
                        node.name,
                        cur,
                        mx,
                    )
        for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
            gov = cpu / "cpufreq" / "scaling_governor"
            try:
                if gov.read_text().strip() != jetson._CPU_GOVERNOR:
                    self._logger.warning(
                        "%s governor is not %s — bursty control loops underclock "
                        "(~30%% lower IK rate). Re-run `axol jetson.setup`.",
                        cpu.name,
                        jetson._CPU_GOVERNOR,
                    )
                    break  # one warning is enough; they're set together
            except OSError:
                continue
