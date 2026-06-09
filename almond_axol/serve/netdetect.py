"""Best-effort detection of the ethernet interface carrying the ZED link.

The ZED data link (PTP clock sync + camera HEVC streams) runs over an ethernet
connection between the main host and the ZED box. ``zed.sync-clocks`` needs the
interface name (``eth0`` etc.) because the PTP daemon binds to a specific NIC
and its hardware clock. Rather than ask the operator, we derive the interface
from the ZED streamer IP they already provide:

- :func:`iface_for_route` — the local interface with a route to a *remote* IP
  (how the main host reaches the ZED box).
- :func:`iface_owning` — the local interface that *holds* an IP (how the ZED
  box finds the NIC carrying its own streaming address).

:func:`list_eth_ifaces` / :func:`best_eth_iface` remain as fallbacks when the
IP-based lookup can't pin an interface.

Linux-only (reads ``/sys/class/net`` and shells out to ``ip``); returns empty
results elsewhere (e.g. a macOS dev machine) so callers degrade gracefully.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_NET = Path("/sys/class/net")

# Interface name prefixes that are never the wired ZED link.
_SKIP_PREFIXES = (
    "lo",
    "wl",  # wifi
    "docker",
    "veth",
    "br",
    "virbr",
    "tun",
    "tap",
    "ppp",
    "bond",
    "dummy",
)


def _is_ethernet(name: str) -> bool:
    if name.startswith(_SKIP_PREFIXES):
        return False
    # ARPHRD_ETHER == 1; wired NICs report type 1.
    try:
        return (_NET / name / "type").read_text().strip() == "1"
    except OSError:
        return False


def _operstate(name: str) -> str:
    try:
        return (_NET / name / "operstate").read_text().strip()
    except OSError:
        return "unknown"


def _has_carrier(name: str) -> bool:
    try:
        return (_NET / name / "carrier").read_text().strip() == "1"
    except OSError:
        return False


def list_eth_ifaces() -> list[str]:
    """Names of plausible wired interfaces, best candidate first.

    Ranking: carrier + up > up > everything else, with the kernel's own
    ordering as a stable tiebreak.
    """
    if not _NET.exists():
        return []

    names = [p.name for p in sorted(_NET.iterdir()) if _is_ethernet(p.name)]

    def rank(name: str) -> tuple[int, int]:
        up = _operstate(name) == "up"
        carrier = _has_carrier(name)
        # Lower sorts first.
        return (0 if (up and carrier) else 1, 0 if up else 1)

    return sorted(names, key=rank)


def best_eth_iface() -> str | None:
    """The single most likely ZED-link interface, or ``None`` if undetectable."""
    ifaces = list_eth_ifaces()
    return ifaces[0] if ifaces else None


def _run_ip(args: list[str]) -> str | None:
    """Run ``ip <args>`` and return stdout, or ``None`` if unavailable/failed."""
    try:
        out = subprocess.run(["ip", *args], capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def iface_for_route(ip: str) -> str | None:
    """Local interface with a route to ``ip`` (how we reach a remote host).

    Parses ``ip route get <ip>`` (e.g. ``... dev eth0 src 192.168.10.2``).
    Returns ``None`` if ``ip`` is blank, unresolvable, or routes via loopback.
    """
    ip = ip.strip()
    if not ip:
        return None
    out = _run_ip(["route", "get", ip])
    if not out:
        return None
    m = re.search(r"\bdev\s+(\S+)", out)
    if not m:
        return None
    iface = m.group(1)
    return None if iface == "lo" else iface


def iface_owning(ip: str) -> str | None:
    """Local interface that has ``ip`` assigned (our own link address).

    Parses ``ip -o -4 addr show`` lines like
    ``2: eth0    inet 192.168.10.1/24 brd ... scope global eth0``.
    """
    ip = ip.strip()
    if not ip:
        return None
    out = _run_ip(["-o", "-4", "addr", "show"])
    if not out:
        return None
    for line in out.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        addr = parts[parts.index("inet") + 1].split("/")[0]
        if addr == ip:
            name = parts[1]
            return None if name == "lo" else name
    return None
