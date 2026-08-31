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
:func:`pin_realtime_clocks` first selects MAXN SUPER when the active platform
configuration provides it (otherwise MAXN) to uncap them, then pins the engine
and CPU clocks to that max — fixing the latency / rate.
Best-effort and cleared on reboot, so ``axol jetson.setup`` (which calls
:func:`pin_realtime_clocks`) is run at boot from the host installer's systemd
unit — not from teleop / serve.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from .sudo import prime_sudo

_logger = logging.getLogger(__name__)

# Hardware engines whose devfreq clocks the encode path depends on.
_ENGINE_CLOCK_GLOBS = ("*.nvenc", "*.vic")

# cpufreq governor that holds the cores at their max clock. The ceiling it
# holds them at is whatever the active ``nvpmodel`` power mode allows, so we
# select the strongest configured mode first (see :func:`_set_max_power_mode`)
# for the full benefit.
_CPU_GOVERNOR = "performance"

# Legacy fallback for configurations whose power-mode names cannot be read.
# NVIDIA documents mode 0 as the generic maximum-power mode, but newer Super
# configurations are not numerically uniform (for example, Orin Nano commonly
# exposes MAXN_SUPER as mode 2). Prefer the names in ``nvpmodel.conf`` below.
_MAXN_MODE = "0"

# The active board/SKU-specific configuration is exposed at this canonical
# path (normally a symlink into /etc/nvpmodel/). It is the source of truth for
# which modes the installed BSP actually supports and their numeric IDs.
_NVPMODEL_CONFIG = Path("/etc/nvpmodel.conf")

_POWER_MODEL_RE = re.compile(
    r"^[ \t]*<[ \t]*POWER_MODEL\b(?P<body>[^>\r\n]*)>",
    re.IGNORECASE | re.MULTILINE,
)
_POWER_MODE_ID_RE = re.compile(r"\bID\s*=\s*(\d+)\b", re.IGNORECASE)
_POWER_MODE_NAME_RE = re.compile(
    r"""\bNAME\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""", re.IGNORECASE
)

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


def _preferred_max_power_mode() -> tuple[str, str]:
    """Return ``(id, display_name)`` for the strongest configured mode.

    ``nvpmodel`` mode IDs vary by Jetson module and flash configuration. Read
    the active configuration rather than guessing from the hardware model:
    prefer ``MAXN_SUPER``, then ordinary ``MAXN``. Fall back to the historically
    documented mode 0 when the configuration is absent, unreadable, or does
    not name either mode.
    """
    try:
        config = _NVPMODEL_CONFIG.read_text()
    except OSError as exc:
        _logger.debug("cannot read %s: %s", _NVPMODEL_CONFIG, exc)
        return _MAXN_MODE, "MAXN"

    modes: dict[str, tuple[str, str]] = {}
    for entry in _POWER_MODEL_RE.finditer(config):
        body = entry.group("body")
        mode_id = _POWER_MODE_ID_RE.search(body)
        mode_name = _POWER_MODE_NAME_RE.search(body)
        if mode_id is None or mode_name is None:
            continue
        raw_name = next(group for group in mode_name.groups() if group is not None)
        canonical_name = re.sub(r"[^A-Z0-9]+", "_", raw_name.upper()).strip("_")
        modes.setdefault(
            canonical_name,
            (mode_id.group(1), canonical_name.replace("_", " ")),
        )

    for preferred_name in ("MAXN_SUPER", "MAXN"):
        if preferred_name in modes:
            return modes[preferred_name]
    return _MAXN_MODE, "MAXN"


def _set_max_power_mode(escalator: _RootEscalator) -> None:
    """Select the strongest configured ``nvpmodel`` power mode.

    Jetson-only and best-effort. MAXN SUPER is preferred when the active
    platform configuration supports it, with MAXN as the fallback. This is the
    prerequisite for the CPU governor and engine pins below: ``nvpmodel`` caps
    the clock ceiling, so without the maximum mode the ``performance``
    governor merely holds the cores at a lower mode's max. A no-op when already
    in the selected mode (so no needless sudo prompt) or when ``nvpmodel`` is
    absent.
    """
    if not _is_jetson():
        _logger.debug("not a Jetson; leaving the nvpmodel power mode unchanged")
        return
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel is None:
        _logger.debug("nvpmodel not found; leaving the power mode unchanged")
        return
    max_mode, max_mode_name = _preferred_max_power_mode()
    # Skip the (root-only) switch when the selected maximum mode is active.
    if _query_power_mode(nvpmodel) == max_mode:
        return
    # Answer "n" to any confirmation prompt. Once the GPU golden context
    # exists (always, by the time the installer or the boot ExecStartPre gets
    # here — nvpmodel.service runs earlier in boot), switching maximum modes
    # may ask to reboot *now* (``DO YOU WANT TO REBOOT NOW? enter YES/yes to
    # confirm:``).
    # We must never reboot the box here -- jetson.setup runs mid-install (over
    # the operator's SSH session) and at boot, and an in-place reboot would
    # drop that session and restart the robot. Feeding stdin also stops the
    # interactive `axol jetson.setup` run from blocking on the prompt.
    ok, detail = escalator.run([nvpmodel, "-m", max_mode], input_text="n\n")
    if _query_power_mode(nvpmodel) == max_mode:
        _logger.info(
            "set Jetson power mode to %s (nvpmodel -m %s)",
            max_mode_name,
            max_mode,
        )
        return
    if ok or "reboot" in detail.lower():
        # Declining the reboot prompt CANCELS the switch — nvpmodel records
        # nothing, so left alone the mode would never change, on this boot or
        # any later one. Persist the mode ourselves in nvpmodel's status file:
        # the boot service applies the mode saved there before the GPU golden
        # context exists, so the selected mode takes effect cleanly on the next
        # *natural* reboot (and the earlier -q short-circuit keeps re-runs
        # quiet).
        pending = f"pmode:{int(max_mode):04d}"
        wrote, write_detail = escalator.write(_NVPMODEL_STATUS, pending)
        if wrote:
            _logger.warning(
                "Jetson power mode %s needs a reboot to take effect — declined "
                "the in-place reboot so this session/robot isn't restarted, and "
                "recorded it in %s so it applies on the next reboot (the boot "
                "service re-pins the clocks then). The engine/CPU pins below "
                "still help at the current mode's ceiling.",
                max_mode_name,
                _NVPMODEL_STATUS,
            )
        else:
            _logger.warning(
                "Jetson power mode %s needs a reboot, and recording it for the "
                "next boot failed (%s) — fix manually with: sudo nvpmodel -m "
                "%s (confirm the reboot prompt, or reboot afterwards).",
                max_mode_name,
                write_detail or "write failed",
                max_mode,
            )
    else:
        _logger.warning(
            "cannot set the Jetson power mode to %s (nvpmodel -m %s failed%s) — "
            "the active mode caps the clock ceiling the performance governor "
            "and engine pins can reach. Fix manually with: sudo nvpmodel -m %s",
            max_mode_name,
            max_mode,
            f": {detail}" if detail else "",
            max_mode,
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
        # here is usually the clock-ceiling cap of a lower power mode, which
        # clears once the maximum mode is active (it may be pending a reboot —
        # the boot service re-pins then), so point there before the manual
        # override.
        _logger.warning(
            "could not set %d CPU core(s) to the %s governor (last error: %s) — "
            "the schedutil default underclocks bursty control loops (~30%% lower "
            "IK rate). This usually clears once the maximum power mode is active "
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
    """Select the maximum mode and pin engine and CPU clocks for control loops.

    Selects MAXN SUPER when configured (otherwise MAXN), pins NVENC/VIC (encode
    latency), and switches the CPUs to the ``performance`` governor (IK rate).
    All three are Jetson-only: power-mode selection and CPU-governor pinning
    are gated on :func:`_is_jetson` so they never alter a non-Tegra host, and
    engine pinning is a no-op without the Tegra devfreq nodes. The power mode
    is selected first because it sets the ceiling the governor and engine pins
    reach. Same best-effort / ``interactive`` escalation semantics as
    :func:`pin_engine_clocks`; sudo is primed at most once across all of them.
    Invoked via ``axol jetson.setup`` (host installer + boot service), not from
    the teleop / collect-data / serve entry points.
    """
    escalator = _RootEscalator(interactive=interactive)
    _set_max_power_mode(escalator)
    _pin_engines(escalator)
    _pin_cpu(escalator)
