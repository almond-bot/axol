"""Browser-origin policy for the robot's LAN-facing HTTP/WebSocket servers.

This is a CSRF/drive-by-browser boundary, not LAN client authentication.
Native clients normally omit ``Origin`` and remain compatible; browser clients
must be same-origin, the hosted Axol UI, a loopback development server, or an
explicit operator-configured origin.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from urllib.parse import urlsplit

_HOSTED_UI_ORIGIN = "https://axol.almond.bot"
_EXTRA_ORIGINS_ENV = "AXOL_ALLOWED_BROWSER_ORIGINS"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_SELF_HOSTED_ORIGINS_ENV = "_AXOL_SELF_HOSTED_UI_ORIGINS"


def _normalize_origin(value: str) -> str | None:
    """Return a canonical HTTP(S) origin, rejecting URLs with extra parts."""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    hostname = parsed.hostname
    if hostname is None:
        return None
    default_port = 80 if parsed.scheme == "http" else 443
    port_suffix = "" if port in (None, default_port) else f":{port}"
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parsed.scheme}://{host.lower()}{port_suffix}"


def allowed_browser_origins() -> frozenset[str]:
    """Configured cross-origin browser clients accepted by Axol services."""
    origins = {_HOSTED_UI_ORIGIN}
    for raw in os.environ.get(_EXTRA_ORIGINS_ENV, "").split(","):
        if not raw.strip():
            continue
        normalized = _normalize_origin(raw)
        if normalized is not None:
            origins.add(normalized)
    return frozenset(origins)


def configure_self_hosted_browser_origins(
    *, scheme: str, port: int, hosts: Iterable[str]
) -> None:
    """Publish the CLI's exact UI origins to in-process VR servers.

    Environment state is used because operation workers may be subprocesses.
    Origins are derived from server-owned startup configuration, never from a
    request's attacker-controlled ``Host`` header (which would permit DNS
    rebinding when plain HTTP is enabled).
    """
    if scheme not in {"http", "https"}:
        raise ValueError("self-hosted browser scheme must be http or https")
    if not 1 <= port <= 65535:
        raise ValueError("self-hosted browser port must be between 1 and 65535")
    origins: set[str] = set()
    for raw_host in hosts:
        host = raw_host.strip()
        if not host or host in {"0.0.0.0", "::", "[::]"}:
            continue
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        normalized = _normalize_origin(f"{scheme}://{host}:{port}")
        if normalized is not None:
            origins.add(normalized)
    os.environ[_SELF_HOSTED_ORIGINS_ENV] = ",".join(sorted(origins))


def _self_hosted_browser_origins() -> frozenset[str]:
    origins: set[str] = set()
    for raw in os.environ.get(_SELF_HOSTED_ORIGINS_ENV, "").split(","):
        if normalized := _normalize_origin(raw):
            origins.add(normalized)
    return frozenset(origins)


def browser_origin_allowed(origin: str | None, *, scheme: str, host: str) -> bool:
    """Whether a browser ``Origin`` may access a server scope.

    A missing header denotes a non-browser/native client and is intentionally
    allowed for the CLI tracker bridge and existing SDK integrations.
    """
    if origin is None:
        return True
    normalized = _normalize_origin(origin)
    if normalized is None:
        return False
    if normalized in allowed_browser_origins():
        return True
    if normalized in _self_hosted_browser_origins():
        return True

    parsed = urlsplit(normalized)
    if parsed.hostname in _LOOPBACK_HOSTS:
        return True
    # ``scheme`` and ``host`` remain parameters for a stable call surface, but
    # neither may grant trust: both ultimately come from the request. In
    # particular, Origin == scheme://Host is still attacker-controlled during
    # DNS rebinding. Self-hosted origins are registered above at CLI startup.
    _ = scheme, host
    return False


# Starlette accepts a regex alongside its exact allowlist. Keep loopback-only
# development permissive across Vite's changing port without trusting arbitrary
# hosted domains.
LOOPBACK_ORIGIN_REGEX = r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$"
