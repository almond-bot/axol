"""Jetson system tweaks for the real-time teleop / data-collection loops.

Two Tegra defaults trade latency for power/throughput and hurt us:

* **Engine devfreq** — the camera relay's hardware encode path
  (``almond_axol.video.hw_video``) depends on NVENC (the H.264 encoder) and
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
        any(Path("/sys/class/devfreq").glob(pattern)) for pattern in _ENGINE_CLOCK_GLOBS
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
            ok, detail = writer.write(node / "min_freq", max_freq)
            if ok:
                _logger.info("pinned %s clock to %s Hz", node.name, max_freq)
            else:
                _logger.warning(
                    "cannot pin %s to its max clock (%s) — hardware encode "
                    "latency will be ~3x worse. Fix manually with: "
                    "echo %s | sudo tee %s",
                    node.name,
                    detail or "write failed",
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
