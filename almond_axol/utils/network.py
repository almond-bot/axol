"""Small, side-effect-free network discovery helpers."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
from pathlib import Path

_TRUSTED_IP_COMMANDS = ("/usr/sbin/ip", "/usr/bin/ip", "/sbin/ip", "/bin/ip")
_IP_INVENTORY_MAX_CHARS = 1_000_000


def _canonical_ip(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    # Scoped IPv6 link-local URLs are not consistently supported by browsers.
    # Drop the scope for parsing, then reject link-local addresses below; all
    # routable IPv6 interface addresses remain eligible.
    candidate = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return None
    if address.is_unspecified or (
        isinstance(address, ipaddress.IPv6Address) and address.is_link_local
    ):
        return None
    return str(address)


def _trusted_ip_command() -> str | None:
    for raw in _TRUSTED_IP_COMMANDS:
        candidate = Path(raw)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _locally_bindable_ip(address: str) -> bool:
    """Confirm a hostname-resolved fallback is assigned to this machine."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.bind((address, 0))
    except OSError:
        return False
    return True


def local_interface_ips() -> frozenset[str]:
    """Return exact IP addresses configured on this host, best effort.

    Linux's own interface inventory is authoritative for the robot service and
    includes secondary IPv4, VPN, and routable IPv6 addresses. Only fixed
    absolute system paths may supply it to a root service; hostname resolution
    is retained as a portable fallback when ``ip`` is absent.
    Every value is parsed as an IP before it can become a trusted browser
    origin, and hostname fallbacks must also bind locally. Request headers are
    never inputs; custom DNS origin names remain explicit configuration.
    """
    addresses: set[str] = set()
    ip_command = _trusted_ip_command()
    if ip_command is not None:
        try:
            result = subprocess.run(  # noqa: S603 - root-owned absolute executable
                [ip_command, "-json", "address", "show"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if (
            result is not None
            and result.returncode == 0
            and len(result.stdout) <= _IP_INVENTORY_MAX_CHARS
        ):
            try:
                interfaces = json.loads(result.stdout)
            except (TypeError, json.JSONDecodeError):
                interfaces = []
            if isinstance(interfaces, list):
                for interface in interfaces:
                    if not isinstance(interface, dict):
                        continue
                    info_list = interface.get("addr_info")
                    if not isinstance(info_list, list):
                        continue
                    for info in info_list:
                        if not isinstance(info, dict) or info.get("family") not in {
                            "inet",
                            "inet6",
                        }:
                            continue
                        if address := _canonical_ip(info.get("local")):
                            addresses.add(address)
    else:
        try:
            resolved = socket.getaddrinfo(
                socket.gethostname(),
                None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            resolved = []
        for entry in resolved:
            if (
                len(entry) >= 5
                and entry[4]
                and (address := _canonical_ip(entry[4][0]))
                and _locally_bindable_ip(address)
            ):
                addresses.add(address)
    return frozenset(addresses)


def local_ip(*, fallback: str = "127.0.0.1") -> str:
    """Best-effort address another LAN device can use to reach this host.

    UDP ``connect`` selects an interface without sending a packet, but it
    still raises on isolated robot networks with no default route.  Fall back
    to hostname resolution before returning the caller's safe sentinel.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        pass

    try:
        resolved = socket.gethostbyname(socket.gethostname())
        if resolved and not resolved.startswith("127."):
            return resolved
    except OSError:
        pass
    return fallback
