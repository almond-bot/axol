"""WebRTC control channel: low-latency pose transport over a data channel.

Pose frames normally ride the teleop WebSocket. Over a Tailscale Funnel that
WebSocket is a *relayed TCP* path, where head-of-line blocking makes a
high-rate pose stream accumulate latency — teleop feels laggy even while the
WebRTC camera video (UDP) stays smooth.

This carries pose frames over an **unreliable, unordered** WebRTC data channel
on a dedicated peer connection, so they take the same UDP / ICE / TURN path as
the video: no head-of-line blocking, and stale poses are dropped rather than
queued. On a LAN the same channel negotiates a direct host candidate (no TURN),
so it's equal-or-better there too; the WebSocket remains the fallback whenever
the channel isn't open.

The connection is deliberately independent of the camera peer connection (which
may live in the out-of-process video relay): keeping pose handling in the main
server process avoids an extra IPC hop, and the channel is available even when
no cameras are present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from .ice import ice_servers, summarize_candidates

_logger = logging.getLogger(__name__)

# Matches the label the browser listens for in `useAxolControlChannel`.
_POSE_CHANNEL_LABEL = "pose"


class ControlChannelManager:
    """Per-client peer connections that carry pose frames on a data channel.

    The server is the offerer: it creates the unreliable ``pose`` data channel
    and hands the SDP offer to the headset over the signaling WebSocket. Inbound
    channel messages (the headset's pose frames) are passed to ``on_message``,
    which feeds them into the same path as WebSocket pose frames.
    """

    def __init__(self, on_message: Callable[[str], None]) -> None:
        self._on_message = on_message
        self._pcs: dict[int, RTCPeerConnection] = {}

    async def create_offer(self, client_id: int) -> str:
        """Build a fresh peer connection with the pose channel; return the SDP.

        ICE candidates (including the TURN relay candidate when configured) are
        gathered during ``setLocalDescription`` and embedded in the returned SDP
        (non-trickle), matching the video signaling.
        """
        await self.close(client_id)

        servers = ice_servers()
        pc = (
            RTCPeerConnection(RTCConfiguration(iceServers=servers))
            if servers
            else RTCPeerConnection()
        )
        self._pcs[client_id] = pc

        # Unreliable + unordered: drop a late/lost pose rather than stall the
        # stream behind a retransmit (freshness beats completeness for teleop).
        channel = pc.createDataChannel(
            _POSE_CHANNEL_LABEL, ordered=False, maxRetransmits=0
        )

        @channel.on("message")
        def _on_message(message: object) -> None:
            if isinstance(message, str):
                self._on_message(message)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            _logger.info(
                "control[%d] connectionState=%s", client_id, pc.connectionState
            )
            if pc.connectionState in ("failed", "closed"):
                await self.close(client_id)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        if servers:
            _logger.info(
                "control[%d] offer %s",
                client_id,
                summarize_candidates(pc.localDescription.sdp),
            )
        return pc.localDescription.sdp

    async def set_answer(self, client_id: int, sdp: str) -> None:
        """Apply the headset's SDP answer for ``client_id``."""
        pc = self._pcs.get(client_id)
        if pc is None:
            _logger.warning("control answer for unknown client %d", client_id)
            return
        if ice_servers():
            _logger.info("control[%d] answer %s", client_id, summarize_candidates(sdp))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def close(self, client_id: int) -> None:
        """Close and forget the peer connection for ``client_id``, if any."""
        pc = self._pcs.pop(client_id, None)
        if pc is not None:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    async def close_all(self) -> None:
        """Close every active control peer connection."""
        for client_id in list(self._pcs):
            await self.close(client_id)
