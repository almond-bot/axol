"""
WebRTC video relay for streaming wrist cameras to the VR headset.

During data collection the upper computer already decodes the left_arm and
right_arm ZED streams to RGB numpy frames (see ``ZedCamera``). This module
re-encodes those frames as a low-latency WebRTC stream that the Quest WebXR app
can render directly, so the teleoperator can see the grippers.

The existing VR WebSocket (``/ws``) is reused purely for SDP signaling — no new
ports. aiortc gathers ICE candidates during ``setLocalDescription`` and embeds
them in the SDP (non-trickle), so on a LAN no separate candidate exchange is
needed.

aiortc is an optional dependency (the ``video`` extra); importing this module
requires it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from aiortc import (
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame
from numpy.typing import NDArray

from .hw_video import install_hw_encoder

_logger = logging.getLogger(__name__)

# A frame source returns the latest RGB uint8 frame (H, W, 3) or None if no
# frame is available yet.
FrameSource = Callable[[], "NDArray[Any] | None"]


class CameraVideoTrack(VideoStreamTrack):
    """WebRTC video track backed by a numpy RGB frame source.

    Pulls the latest frame from ``source`` on every ``recv`` and hands it to
    aiortc as an ``av.VideoFrame``. Pacing is delegated to the base class's
    :meth:`next_timestamp` (~30 fps), which is plenty for a wrist preview.
    """

    kind = "video"

    def __init__(self, source: FrameSource) -> None:
        super().__init__()
        self._source = source
        self._last: NDArray[Any] | None = None

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        arr: NDArray[Any] | None = None
        try:
            arr = self._source()
        except Exception as exc:  # source is best-effort; never kill the track
            _logger.debug("video source raised: %s", exc)

        if arr is None:
            arr = self._last
        if arr is None:
            # No frame yet — emit black so negotiation and keyframes proceed.
            arr = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            self._last = arr

        frame = VideoFrame.from_ndarray(
            np.ascontiguousarray(arr, dtype=np.uint8), format="rgb24"
        )
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRTCManager:
    """Manages per-client peer connections that send wrist camera video.

    One :class:`RTCPeerConnection` per connected headset, keyed by an opaque
    client id (the WebSocket's ``id()``). The server drives signaling: it
    creates the offer (with the camera tracks attached) and applies the
    headset's answer.
    """

    def __init__(self, sources: dict[str, FrameSource]) -> None:
        self._sources = dict(sources)
        self._pcs: dict[int, RTCPeerConnection] = {}

    @property
    def has_sources(self) -> bool:
        return bool(self._sources)

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a fresh peer connection for ``client_id`` and return the offer.

        Returns ``(sdp, tracks)`` where ``tracks`` maps each negotiated media
        ``mid`` to its camera name so the client can label incoming streams.
        """
        await self.close(client_id)

        pc = RTCPeerConnection()
        self._pcs[client_id] = pc

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            _logger.info("webrtc[%d] connectionState=%s", client_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close(client_id)

        track_names: dict[int, str] = {}
        for name, source in self._sources.items():
            track = CameraVideoTrack(source)
            pc.addTrack(track)
            track_names[id(track)] = name

        # On Jetson, route encoding through NVENC (software VP8/H264 can't
        # keep up at high resolutions) and pin H.264 in the SDP so the
        # hardware path is what actually gets negotiated.
        if install_hw_encoder():
            h264 = [
                c
                for c in RTCRtpSender.getCapabilities("video").codecs
                if c.mimeType.lower() == "video/h264"
            ]
            if h264:
                for transceiver in pc.getTransceivers():
                    transceiver.setCodecPreferences(h264)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        tracks: dict[str, str] = {}
        for transceiver in pc.getTransceivers():
            sender_track = transceiver.sender.track
            if sender_track is not None and id(sender_track) in track_names:
                tracks[transceiver.mid] = track_names[id(sender_track)]

        return pc.localDescription.sdp, tracks

    async def set_answer(self, client_id: int, sdp: str) -> None:
        """Apply the headset's SDP answer for ``client_id``."""
        pc = self._pcs.get(client_id)
        if pc is None:
            _logger.warning("webrtc answer for unknown client %d", client_id)
            return
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

    async def close(self, client_id: int) -> None:
        """Close and forget the peer connection for ``client_id``, if any."""
        pc = self._pcs.pop(client_id, None)
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass

    async def close_all(self) -> None:
        """Close every active peer connection."""
        for client_id in list(self._pcs):
            await self.close(client_id)
