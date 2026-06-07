"""Best-effort detection of the ethernet interface carrying the ZED link.

The ZED data link (PTP clock sync + camera HEVC streams) runs over a direct
ethernet connection between the main host and the ZED box. Both machines need
to name that interface (``eth0`` etc.) for ``zed.sync-clocks`` / ``zed.stream``
and the receiver's ``--zed_iface``. This module guesses a sensible default and
lists the candidates so the web UI can offer a dropdown with a manual override.

Linux-only (reads ``/sys/class/net``); returns empty results elsewhere (e.g. a
macOS dev machine) so callers degrade to a free-text field.
"""

from __future__ import annotations

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

# Address (with /prefix stripped) the ZED link uses on each side; an interface
# already holding one of these is almost certainly the link, so rank it first.
_LINK_ADDRS = ("192.168.10.1", "192.168.10.2")


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

    Ranking: link IP already assigned > carrier + up > up > everything else,
    with the kernel's own ordering as a stable tiebreak.
    """
    if not _NET.exists():
        return []

    names = [p.name for p in sorted(_NET.iterdir()) if _is_ethernet(p.name)]

    def rank(name: str) -> tuple[int, int, int]:
        up = _operstate(name) == "up"
        carrier = _has_carrier(name)
        has_link_addr = _has_link_address(name)
        # Lower sorts first.
        return (0 if has_link_addr else 1, 0 if (up and carrier) else 1, 0 if up else 1)

    return sorted(names, key=rank)


def _has_link_address(name: str) -> bool:
    """True if ``name`` currently holds one of the static ZED-link IPs."""
    try:
        import subprocess

        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "dev", name],
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout
    except (OSError, ValueError):
        return False
    return any(addr in out for addr in _LINK_ADDRS)


def best_eth_iface() -> str | None:
    """The single most likely ZED-link interface, or ``None`` if undetectable."""
    ifaces = list_eth_ifaces()
    return ifaces[0] if ifaces else None
