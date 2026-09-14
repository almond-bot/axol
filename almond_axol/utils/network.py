"""Small, side-effect-free network discovery helpers."""

from __future__ import annotations

import socket


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
