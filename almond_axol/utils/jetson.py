"""Jetson system tweaks for the real-time teleop / data-collection loops.

Two Tegra defaults trade latency for power/throughput and hurt us:

* **Engine devfreq** — the camera relay's hardware encode path
  (``almond_axol.vr.hw_video``) depends on NVENC (the H.264 encoder) and
  the VIC (``nvvidconv``'s colorspace conversion). The default
  ``tegra_wmark`` governor grants just enough clock to keep up with the
  frame rate, so each frame takes nearly a full frame-time to encode
  (~3x worse per-frame latency at the ~25% clock it settles on).

* **CPU cpufreq** — the IK solver (JAX/XLA) is a bursty, sleep-heavy
  workload. The default ``schedutil`` governor reads that idle and
  underclocks the cores to ~40-70% of max, which drops the IK rate by a
  matching ~30% (measured 79 Hz vs 113 Hz pinned).

Pinning both to max fixes the latency / rate. Best-effort and cleared on
reboot, so ``axol jetson.setup`` (which calls :func:`pin_realtime_clocks`) is
run at boot from the host installer's systemd unit — not from teleop / serve.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .sudo import prime_sudo

_logger = logging.getLogger(__name__)

# Hardware engines whose devfreq clocks the encode path depends on.
_ENGINE_CLOCK_GLOBS = ("*.nvenc", "*.vic")

# cpufreq governor that holds the cores at their max clock. ``nvpmodel``
# still caps the ceiling, so MAXN is assumed for the full benefit.
_CPU_GOVERNOR = "performance"


class _SysfsWriter:
    """Writes sysfs values, escalating to ``sudo`` once if needed.

    The hosted install runs as root (direct writes succeed); a CLI user
    may not be, so the first failed write primes sudo credentials (a tty
    prompt when ``interactive``) and every write after that uses
    ``sudo -n``. Priming happens at most once, and only when a value
    actually needs changing.
    """

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self._primed = False

    def write(self, path: Path, value: str) -> bool:
        try:
            path.write_text(value)
            return True
        except OSError:
            pass
        if self._interactive and not self._primed:
            self._primed = prime_sudo()
        return (
            subprocess.run(
                ["sudo", "-n", "tee", str(path)],
                input=value,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )


def _pin_engines(writer: _SysfsWriter) -> None:
    """Set ``min_freq = max_freq`` on the NVENC/VIC devfreq nodes."""
    for pattern in _ENGINE_CLOCK_GLOBS:
        for node in Path("/sys/class/devfreq").glob(pattern):
            try:
                max_freq = (node / "max_freq").read_text().strip()
                if (node / "min_freq").read_text().strip() == max_freq:
                    continue
            except OSError as exc:
                _logger.warning("cannot read %s clock state: %s", node.name, exc)
                continue
            if writer.write(node / "min_freq", max_freq):
                _logger.info("pinned %s clock to %s Hz", node.name, max_freq)
            else:
                _logger.warning(
                    "cannot pin %s to its max clock (need root) — hardware "
                    "encode latency will be ~3x worse. Fix manually with: "
                    "echo %s | sudo tee %s",
                    node.name,
                    max_freq,
                    node / "min_freq",
                )


def _pin_cpu(writer: _SysfsWriter) -> None:
    """Switch every online CPU to the ``performance`` cpufreq governor."""
    pinned = 0
    failed: Path | None = None
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        gov = cpu / "cpufreq" / "scaling_governor"
        try:
            if gov.read_text().strip() == _CPU_GOVERNOR:
                continue
        except OSError:
            continue  # offline core or no cpufreq (not a throttled Jetson)
        if writer.write(gov, _CPU_GOVERNOR):
            pinned += 1
        else:
            failed = gov
    if pinned:
        _logger.info("pinned %d CPU core(s) to the %s governor", pinned, _CPU_GOVERNOR)
    if failed is not None:
        _logger.warning(
            "cannot set CPU governor to %s (need root) — the schedutil default "
            "underclocks bursty control loops (~30%% lower IK rate). Fix "
            "manually with: echo %s | sudo tee "
            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
            _CPU_GOVERNOR,
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
    _pin_engines(_SysfsWriter(interactive=interactive))


def pin_realtime_clocks(*, interactive: bool = False) -> None:
    """Pin engine **and** CPU clocks for the real-time control loops.

    Pins NVENC/VIC (encode latency) and switches the CPUs to the
    ``performance`` governor (IK rate). Same best-effort / ``interactive``
    escalation semantics as :func:`pin_engine_clocks`; sudo is primed at
    most once across both. Invoked via ``axol jetson.setup`` (host installer +
    boot service), not from the teleop / collect-data / serve entry points.
    """
    writer = _SysfsWriter(interactive=interactive)
    _pin_engines(writer)
    _pin_cpu(writer)
