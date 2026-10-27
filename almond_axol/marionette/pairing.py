"""The gateway's pairing details, for the control panel's Marionette page.

The page shows the gateway's QR code and pairing code, so a phone pairs by
scanning the screen. A browser on another computer can't read them from the
gateway itself (it would need the code), but Axol's server usually runs on
the gateway's computer, which the gateway lets in without one. A gateway on
another computer answers when its address is its pairing link, which
carries the code.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

TIMEOUT_S = 2.0
AUTH_MODES = ("on", "off", "strict")
# Straight to the gateway, whatever proxy the environment names.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class PairingError(Exception):
    """The pairing details are unavailable; the message says why."""


def _pairing_url(ws_url: str) -> str:
    parts = urlsplit(ws_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, "/api/pairing", "", ""))


def parse_pairing(body: Any) -> dict[str, Any]:
    """The fields the page shows, checked: the QR code must be an SVG and the
    link a Marionette pairing link."""
    if not isinstance(body, dict):
        raise PairingError("The gateway's pairing details weren't understood.")
    url, svg, code = body.get("url"), body.get("qrSvg"), body.get("code")
    if not (
        isinstance(url, str)
        and url.startswith("marionette://pair?")
        and isinstance(svg, str)
        and svg.lstrip().startswith("<svg")
    ):
        raise PairingError("The gateway's pairing details weren't understood.")
    port = body.get("port")
    return {
        "name": body["name"] if isinstance(body.get("name"), str) else "",
        "auth": body["auth"] if body.get("auth") in AUTH_MODES else "on",
        "code": code if isinstance(code, str) and code else None,
        "url": url,
        "addresses": [a for a in body.get("addresses") or [] if isinstance(a, str)],
        "port": port if isinstance(port, int) and not isinstance(port, bool) else None,
        "qrSvg": svg,
    }


def fetch_pairing(address: str, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """The pairing details of the gateway at ``address`` (any form the app
    takes: an IP or host, host:port, a URL or a pairing link).

    Raises :class:`PairingError` with a sentence for the page.
    """
    try:
        from marionette.address import resolve
    except ImportError as exc:
        raise PairingError("Marionette isn't installed on the Axol computer.") from exc
    try:
        target = resolve(address)
    except ValueError as exc:
        raise PairingError(str(exc)) from exc
    error = f"Can't reach the gateway at {address}."
    for url in target.urls:
        request = urllib.request.Request(
            _pairing_url(url), headers={"Accept": "application/json"}
        )
        if target.token:
            request.add_header("Authorization", f"Bearer {target.token}")
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                return parse_pairing(json.load(response))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise PairingError(
                    "The gateway needs its pairing code. Enter its pairing link "
                    "in Gateway."
                ) from exc
            if exc.code == 404:
                raise PairingError(
                    "This gateway is too old to pair phones. Update Marionette."
                ) from exc
            error = f"The gateway at {address} answered with an error ({exc.code})."
        except (OSError, ValueError):
            continue  # the next of a pairing link's hosts, if any
    raise PairingError(error)
