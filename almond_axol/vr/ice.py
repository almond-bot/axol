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


def summarize_candidates(sdp: str) -> str:
    """One-line tally of ICE candidate *types* embedded in an offer/answer SDP.

    aiortc gathers candidates during ``setLocalDescription`` and embeds them in
    the SDP (non-trickle), so this reveals — without a packet capture — whether a
    peer actually obtained a ``relay`` candidate. Off the robot's network the
    media can only connect through a ``relay`` candidate; if a peer logs
    ``host=… srflx=… relay=0`` here, its TURN gathering failed and that is why
    the stream is stuck.
    """
    counts: dict[str, int] = {}
    for line in sdp.splitlines():
        line = line.strip()
        # e.g. "a=candidate:... typ relay raddr ..." — the token after "typ".
        if not line.startswith("a=candidate:") or " typ " not in line:
            continue
        typ = line.split(" typ ", 1)[1].split()[0]
        counts[typ] = counts.get(typ, 0) + 1
    if not counts:
        return "candidates: none"
    order = ["host", "srflx", "prflx", "relay"]
    keys = [k for k in order if k in counts] + [k for k in counts if k not in order]
    return "candidates: " + " ".join(f"{k}={counts[k]}" for k in keys)


def replicate_candidates_across_mlines(sdp: str) -> str:
    """Copy every embedded ICE candidate onto *all* m-line sections.

    Works around an aiortc BUNDLE bug: when the local offer bundles several media
    onto one transport, ``setRemoteDescription`` routes the answer's remote
    candidates by m-line and then, during BUNDLE collapse, keeps only the
    bundle-tag (``a=group:BUNDLE`` first) m-line's transport. If the browser put
    its candidates on a different bundled m-line than the tag, they are discarded
    and the media transport stalls in ``checking`` with zero remote candidates.

    Rather than guess which m-line is the tag, put the full candidate set on
    every m-line so the surviving transport has them regardless. Candidates on
    discarded m-lines are harmless. No-op for a single m-line or when there are
    no embedded candidates (LAN/trickle). Line endings are preserved.
    """
    lines = sdp.split("\n")
    if sum(1 for line in lines if line.startswith("m=")) < 2:
        return sdp
    candidate_lines = [
        line for line in lines if line.lstrip().startswith("a=candidate:")
    ]
    if not candidate_lines:
        return sdp
    body = [line for line in lines if not line.lstrip().startswith("a=candidate:")]
    out: list[str] = []
    seen_m = False
    for line in body:
        if line.startswith("m="):
            if seen_m:
                out.extend(candidate_lines)  # close the previous media section
            seen_m = True
        out.append(line)
    # Close the final media section, inserting before any trailing blank lines.
    tail = 0
    while tail < len(out) and out[len(out) - 1 - tail].strip() == "":
        tail += 1
    insert_pos = len(out) - tail
    out[insert_pos:insert_pos] = candidate_lines
    return "\n".join(out)


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
