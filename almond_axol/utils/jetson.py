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

Both ceilings are themselves capped by the ``nvpmodel`` power mode, so
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

from .sudo import prime_sudo

_logger = logging.getLogger(__name__)

# Hardware engines whose devfreq clocks the encode path depends on.
_ENGINE_CLOCK_GLOBS = ("*.nvenc", "*.vic")

# cpufreq governor that holds the cores at their max clock. The ceiling it
# holds them at is whatever the active ``nvpmodel`` power mode allows, so we
# select MAXN first (see :func:`_set_max_power_mode`) for the full benefit.
_CPU_GOVERNOR = "performance"

# nvpmodel power mode that uncaps the clock ceiling. MAXN is mode 0 on every
# Jetson, so the governor and engine pins can reach the real max clocks.
_MAXN_MODE = "0"

# Canonical L4T marker present on every Jetson. CPU-governor pinning is gated
# on Jetson detection so ``jetson.setup`` on a non-Tegra Linux host never
# touches that machine's system-wide cpufreq governor (engine pinning is
# already implicitly Jetson-only — its devfreq globs are empty elsewhere).
_JETSON_RELEASE = Path("/etc/nv_tegra_release")


def _is_jetson() -> bool:
    """True on NVIDIA Jetson (L4T) hardware, False on any other host."""
    if _JETSON_RELEASE.exists():
        return True
    # Fallback: the encode engines we pin only exist on Tegra, so their
    # presence also identifies a Jetson even if the release file is missing.
    return any(
        Path("/sys/class/devfreq").glob(pattern) for pattern in _ENGINE_CLOCK_GLOBS
    )


class _RootEscalator:
    """Writes sysfs values / runs commands, escalating to ``sudo`` once.

    The hosted install runs as root (direct writes succeed); a CLI user
    may not be, so the first failed operation primes sudo credentials (a tty
    prompt when ``interactive``) and every operation after that uses
    ``sudo -n``. Priming happens at most once, and only when something
    actually needs changing.
    """

    def __init__(self, *, interactive: bool) -> None:
        self._interactive = interactive
        self._primed = False

    def _prime(self) -> None:
        if self._interactive and not self._primed:
            self._primed = prime_sudo()

    def write(self, path: Path, value: str) -> bool:
        try:
            path.write_text(value)
            return True
        except OSError:
            pass
        self._prime()
        return (
            subprocess.run(
                ["sudo", "-n", "tee", str(path)],
                input=value,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )

    def run(self, argv: list[str]) -> bool:
        """Run ``argv`` as root: directly when possible, else via ``sudo -n``."""
        try:
            if subprocess.run(argv, capture_output=True, text=True).returncode == 0:
                return True
        except OSError:
            pass
        self._prime()
        return (
            subprocess.run(
                ["sudo", "-n", *argv], capture_output=True, text=True
            ).returncode
            == 0
        )


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
    # ``nvpmodel -q`` prints the active mode id on its last non-empty line;
    # skip the (root-only) switch when it already reports MAXN.
    try:
        query = subprocess.run([nvpmodel, "-q"], capture_output=True, text=True)
        lines = [ln.strip() for ln in query.stdout.splitlines() if ln.strip()]
        if lines and lines[-1] == _MAXN_MODE:
            return
    except OSError:
        pass
    if escalator.run([nvpmodel, "-m", _MAXN_MODE]):
        _logger.info("set Jetson power mode to MAXN (nvpmodel -m %s)", _MAXN_MODE)
    else:
        _logger.warning(
            "cannot set the Jetson power mode to MAXN (need root) — the active "
            "nvpmodel mode caps the clock ceiling the performance governor and "
            "engine pins can reach. Fix manually with: sudo nvpmodel -m %s",
            _MAXN_MODE,
        )


def _pin_engines(writer: _RootEscalator) -> None:
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
    _pin_engines(_RootEscalator(interactive=interactive))


def pin_realtime_clocks(*, interactive: bool = False) -> None:
    """Select MAXN and pin engine **and** CPU clocks for the control loops.

    Selects the MAXN ``nvpmodel`` power mode (uncaps the clock ceiling), pins
    NVENC/VIC (encode latency), and switches the CPUs to the ``performance``
    governor (IK rate). All three are Jetson-only: MAXN selection and
    CPU-governor pinning are gated on :func:`_is_jetson` so they never alter a
    non-Tegra host, and engine pinning is a no-op without the Tegra devfreq
    nodes. MAXN is selected first because it sets the ceiling the governor and
    engine pins reach. Same best-effort / ``interactive`` escalation semantics
    as :func:`pin_engine_clocks`; sudo is primed at most once across all of
    them. Invoked via ``axol jetson.setup`` (host installer + boot service),
    not from the teleop / collect-data / serve entry points.
    """
    escalator = _RootEscalator(interactive=interactive)
    _set_max_power_mode(escalator)
    _pin_engines(escalator)
    _pin_cpu(escalator)
