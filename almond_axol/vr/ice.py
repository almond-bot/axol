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


def candidates_by_mline(sdp: str) -> str:
    """Per-m-line tally of ICE candidates in an SDP (which ``mid`` carries them).

    aiortc routes embedded (non-trickle) candidates by m-line and, after BUNDLE
    collapse, only keeps the bundle-tag (first) m-line's candidates. If a browser
    puts its candidates on a non-tag bundled m-line, this shows it: e.g.
    ``mid=0:0 mid=1:0 mid=2:4`` means all 4 candidates are on ``mid=2`` and the
    bundled transport (``mid=0``) gets none.
    """
    sections: list[tuple[str, int]] = []
    cur_mid = "session"
    cur_count = 0
    started = False
    for raw in sdp.splitlines():
        line = raw.strip()
        if line.startswith("m="):
            if started:
                sections.append((cur_mid, cur_count))
            started = True
            cur_mid = "?"
            cur_count = 0
        elif line.startswith("a=mid:"):
            cur_mid = line[len("a=mid:") :]
        elif line.startswith("a=candidate:"):
            cur_count += 1
    if started:
        sections.append((cur_mid, cur_count))
    return " ".join(f"mid={m}:{c}" for m, c in sections) or "no-m-lines"


def bundle_candidates_onto_tag_mline(sdp: str) -> str:
    """Move every embedded ICE candidate onto the first (bundle-tag) m-line.

    Works around an aiortc BUNDLE bug: when the local offer bundles several media
    onto one transport, ``setRemoteDescription`` routes the answer's remote
    candidates by m-line and then, during BUNDLE collapse, discards every
    non-tag m-line — so candidates a browser places on a non-tag bundled m-line
    are dropped and the media transport stalls in ``checking`` with zero remote
    candidates. Consolidating all ``a=candidate:`` lines onto the first m-line
    (which is the one aiortc keeps) makes them survive the collapse.

    No-op when there is a single m-line, or when there are no embedded candidates
    (LAN/trickle), and idempotent when candidates already sit on the first
    m-line. Line endings are preserved.
    """
    lines = sdp.split("\n")
    m_indexes = [i for i, line in enumerate(lines) if line.startswith("m=")]
    if len(m_indexes) < 2:
        return sdp
    candidate_lines = [
        line for line in lines if line.lstrip().startswith("a=candidate:")
    ]
    if not candidate_lines:
        return sdp
    kept = [line for line in lines if not line.lstrip().startswith("a=candidate:")]
    # Re-insert all candidates at the end of the first m-line section (just
    # before the second m-line), so they belong to the bundle-tag media.
    insert_at = [i for i, line in enumerate(kept) if line.startswith("m=")][1]
    return "\n".join(kept[:insert_at] + candidate_lines + kept[insert_at:])


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
