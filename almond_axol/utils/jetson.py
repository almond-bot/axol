"""Jetson system tweaks.

The camera relay's hardware encode path (``almond_axol.vr.hw_video``)
depends on two Tegra engines: NVENC (the H.264 encoder) and the VIC
(``nvvidconv``'s colorspace conversion). Jetson's default ``tegra_wmark``
devfreq governor scales these engines for *throughput*: it grants just
enough clock to keep up with the frame rate, so each frame takes nearly a
full frame-time to encode (measured ~3x worse per-frame latency at the
~25% clock it settles on). Pinning their clocks to max fixes the latency.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .sudo import prime_sudo

_logger = logging.getLogger(__name__)

# Hardware engines whose devfreq clocks the encode path depends on.
_ENGINE_CLOCK_GLOBS = ("*.nvenc", "*.vic")


def pin_engine_clocks(*, interactive: bool = False) -> None:
    """Pin NVENC and VIC to their max clock (devfreq ``min_freq = max_freq``).

    Best-effort: direct sysfs write when root (the hosted install runs
    ``axol serve`` as root), ``sudo -n`` otherwise, and a warning with the
    manual command when neither works. A no-op on machines without these
    devfreq nodes (anything that isn't a Jetson). Cleared on reboot, so it
    runs at every serve startup and hardware-encoder install.

    With ``interactive=True``, escalation may prompt for the sudo password
    once on the tty (via :func:`prime_sudo`) — only when a clock actually
    needs pinning. Use this from CLI entry points, where the prompt comes
    at startup; never from mid-session contexts.
    """
    primed = False
    for pattern in _ENGINE_CLOCK_GLOBS:
        for node in Path("/sys/class/devfreq").glob(pattern):
            try:
                max_freq = (node / "max_freq").read_text().strip()
                if (node / "min_freq").read_text().strip() == max_freq:
                    continue
            except OSError as exc:
                _logger.warning("cannot read %s clock state: %s", node.name, exc)
                continue
            try:
                (node / "min_freq").write_text(max_freq)
            except OSError:
                if interactive and not primed:
                    # Caches sudo credentials (tty prompt if needed) so the
                    # `sudo -n` below — and any later pins — succeed.
                    primed = prime_sudo()
                done = (
                    subprocess.run(
                        ["sudo", "-n", "tee", str(node / "min_freq")],
                        input=max_freq,
                        capture_output=True,
                        text=True,
                    ).returncode
                    == 0
                )
                if not done:
                    _logger.warning(
                        "cannot pin %s to its max clock (need root) — hardware "
                        "encode latency will be ~3x worse. Fix manually with: "
                        "echo %s | sudo tee %s",
                        node.name,
                        max_freq,
                        node / "min_freq",
                    )
                    continue
            _logger.info("pinned %s clock to %s Hz", node.name, max_freq)
