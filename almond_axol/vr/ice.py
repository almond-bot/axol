"""ICE server (STUN/TURN) configuration for WebRTC, read from the environment.

On a LAN — or over plain Tailscale, where both peers share a routable
``100.x`` overlay — the headset can reach the robot's *host* ICE candidates
directly, so WebRTC media connects with no relay (the default; these env vars
are unset and every function here returns an empty list, preserving the
original behaviour).

When the headset is **off** the robot's network — e.g. an operator reaching the
box through a Tailscale Funnel (or ngrok) tunnel that only proxies the
signaling WebSocket — those host candidates are private and unreachable, and
the tunnel does not carry the WebRTC UDP media. Pointing both peers at a
publicly reachable **TURN** server gives each side a relay candidate that
bridges the media. Set these env vars on the machine running ``axol teleop``
(they are inherited by the out-of-process video relay):

    AXOL_TURN_URL       Comma-separated ICE URL(s): ``turn:``, ``turns:`` or
                        ``stun:``. Example:
                        ``turn:turn.example.com:3478,turns:turn.example.com:5349``
    AXOL_TURN_USERNAME  TURN username (omit for a ``stun:``-only URL).
    AXOL_TURN_PASSWORD  TURN credential / password.

Both the aiortc peer (``ice_servers``) and the browser peer
(``client_ice_servers``, forwarded over the signaling channel) must use the
same servers, so each gathers its own relay candidate.
"""

from __future__ import annotations

import os
from typing import Any

from aiortc import RTCIceServer

_URL_ENV = "AXOL_TURN_URL"
_USER_ENV = "AXOL_TURN_USERNAME"
_PASS_ENV = "AXOL_TURN_PASSWORD"


def _urls() -> list[str]:
    """Parse ``AXOL_TURN_URL`` into a list of ICE URLs (empty when unset)."""
    return [u.strip() for u in os.environ.get(_URL_ENV, "").split(",") if u.strip()]


def ice_servers() -> list[RTCIceServer]:
    """aiortc ``RTCIceServer`` list from the environment; empty when unset.

    An empty list is the signal to construct ``RTCPeerConnection()`` with no
    explicit configuration (aiortc's default), keeping the LAN path untouched.
    """
    urls = _urls()
    if not urls:
        return []
    return [
        RTCIceServer(
            urls=urls,
            username=os.environ.get(_USER_ENV) or None,
            credential=os.environ.get(_PASS_ENV) or None,
        )
    ]


def client_ice_servers() -> list[dict[str, Any]]:
    """Browser-facing ``RTCConfiguration.iceServers`` entries from the env.

    Sent to the headset over the signaling channel so the browser peer gathers
    the same relay candidate. Shape matches the WebRTC ``RTCIceServer``
    dictionary (``{urls, username?, credential?}``); empty when unconfigured.
    """
    urls = _urls()
    if not urls:
        return []
    entry: dict[str, Any] = {"urls": urls}
    username = os.environ.get(_USER_ENV)
    credential = os.environ.get(_PASS_ENV)
    if username:
        entry["username"] = username
    if credential:
        entry["credential"] = credential
    return [entry]
