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
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from numpy.typing import NDArray

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


def make_test_pattern_source(width: int = 640, height: int = 480) -> FrameSource:
    """Return a frame source rendering an animated test pattern.

    Useful for exercising the WebRTC path without ZED hardware (e.g. in sim or
    the cloud VM): ``AXOL_VR_VIDEO_TEST=1 uv run axol teleop --sim``. A moving
    bar makes motion and latency visible.
    """
    start = time.monotonic()
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = xs[None, :]
    base[..., 1] = ys[:, None]

    def _source() -> NDArray[Any]:
        t = time.monotonic() - start
        frame = base.copy()
        x = int((t * 120) % width)
        frame[:, max(0, x - 3) : x + 3, 2] = 255
        return frame

    return _source
