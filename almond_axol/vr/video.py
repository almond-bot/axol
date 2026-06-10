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

import asyncio
import logging
import queue
import time
from collections.abc import Callable
from typing import Any

import av
import numpy as np
from aiortc import (
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.mediastreams import (
    VIDEO_CLOCK_RATE,
    VIDEO_TIME_BASE,
    MediaStreamError,
    MediaStreamTrack,
)
from av import VideoFrame
from numpy.typing import NDArray

from .hw_video import install_hw_encoder

_logger = logging.getLogger(__name__)

# A frame source returns the latest RGB uint8 frame (H, W, 3) or None if no
# frame is available yet. Sources may additionally expose
#
#     wait_next(after_ts, timeout_ms) -> (frame, capture_perf_ts) | None
#
# which blocks until a frame captured *after* ``after_ts`` (perf_counter
# seconds; None for "any") is available. When present, the track is
# frame-driven: each new camera frame is encoded the moment it exists,
# instead of being sampled by a fixed-rate timer — pull-sampling leaves a
# frame sitting for up to a full camera interval before encoding even
# starts, which is pure added glass-to-glass latency.
FrameSource = Callable[[], "NDArray[Any] | None"]

# How long the frame-driven path waits for a new frame before emitting a
# keepalive repeat of the previous one (camera stall, etc.).
_WAIT_TIMEOUT_MS = 500.0


class CameraVideoTrack(VideoStreamTrack):
    """WebRTC video track backed by a numpy RGB frame source.

    Sources with ``wait_next`` are frame-driven: ``recv`` blocks until the
    camera produces a new frame and returns it immediately, with the RTP
    timestamp derived from the frame's *capture* time. Plain callables fall
    back to pull-sampling paced by the base class's :meth:`next_timestamp`
    (~30 fps).
    """

    kind = "video"

    def __init__(self, source: FrameSource) -> None:
        super().__init__()
        self._source = source
        self._last: NDArray[Any] | None = None
        self._wait_next = getattr(source, "wait_next", None)
        self._last_cap_ts: float | None = None
        self._ts_origin: float | None = None  # perf_counter of pts == 0

    def _clock_pts(self, perf_ts: float) -> int:
        if self._ts_origin is None:
            self._ts_origin = perf_ts
        return max(0, int((perf_ts - self._ts_origin) * VIDEO_CLOCK_RATE))

    async def _recv_frame_driven(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        assert self._wait_next is not None
        result: tuple[NDArray[Any], float] | None = None
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._wait_next, self._last_cap_ts, _WAIT_TIMEOUT_MS
            )
        except Exception as exc:  # source is best-effort; never kill the track
            _logger.debug("video source wait_next raised: %s", exc)

        if result is not None:
            arr, cap_ts = result
            self._last = arr
            self._last_cap_ts = cap_ts
            pts = self._clock_pts(cap_ts)
        else:
            # Camera stalled: repeat the last frame (or black) as a keepalive
            # so the connection and keyframe machinery stay alive.
            arr = (
                self._last
                if self._last is not None
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            pts = self._clock_pts(time.perf_counter())

        frame = VideoFrame.from_ndarray(
            np.ascontiguousarray(arr, dtype=np.uint8), format="rgb24"
        )
        frame.pts = pts
        frame.time_base = VIDEO_TIME_BASE
        return frame

    async def recv(self) -> VideoFrame:
        if self._wait_next is not None:
            return await self._recv_frame_driven()

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


_NAL_TYPE_IDR = 5
_ANNEXB_START = b"\x00\x00\x00\x01"


class PrecodedVideoTrack(MediaStreamTrack):
    """WebRTC track fed by already-encoded H.264 access units.

    Backed by a ``gst_zed.ZedXOneGstStream``-style source (duck type:
    ``subscribe`` / ``unsubscribe`` / ``alive``) whose pipeline grabs and
    encodes entirely on the GPU. ``recv`` returns ``av.Packet``s, which
    aiortc's sender routes through ``encoder.pack`` — straight to RTP
    packetization, no encode step in Python at all.

    The track stays silent until the first access unit carrying an IDR
    (the stream's ``insert-sps-pps`` puts SPS/PPS on every IDR), so a
    late-joining client always starts on a decodable frame.
    """

    kind = "video"

    def __init__(self, source: Any) -> None:
        super().__init__()
        self._source = source
        self._queue: queue.Queue[list[bytes]] = source.subscribe()
        self._synced = False  # set once the first IDR AU has been sent
        self._ts_origin: float | None = None

    def stop(self) -> None:
        self._source.unsubscribe(self._queue)
        super().stop()

    def _pop(self) -> list[bytes] | None:
        try:
            return self._queue.get(timeout=_WAIT_TIMEOUT_MS / 1000.0)
        except queue.Empty:
            return None

    def _clock_pts(self, perf_ts: float) -> int:
        if self._ts_origin is None:
            self._ts_origin = perf_ts
        return max(0, int((perf_ts - self._ts_origin) * VIDEO_CLOCK_RATE))

    async def recv(self) -> av.Packet:
        loop = asyncio.get_running_loop()
        while True:
            if self.readyState != "live":
                raise MediaStreamError
            au = await loop.run_in_executor(None, self._pop)
            if au is None:
                if not self._source.alive:
                    raise MediaStreamError
                continue  # camera stall — keep waiting
            if not self._synced:
                if not any((nal[0] & 0x1F) == _NAL_TYPE_IDR for nal in au):
                    continue
                self._synced = True
            packet = av.Packet(b"".join(_ANNEXB_START + nal for nal in au))
            packet.pts = self._clock_pts(time.perf_counter())
            packet.time_base = VIDEO_TIME_BASE
            return packet


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
            # Sources exposing subscribe() deliver pre-encoded H.264 AUs
            # (the gst-native camera pipelines); everything else is a raw
            # frame source that aiortc encodes.
            track: MediaStreamTrack
            if hasattr(source, "subscribe"):
                track = PrecodedVideoTrack(source)
            else:
                track = CameraVideoTrack(source)
            pc.addTrack(track)
            track_names[id(track)] = name

        # On Jetson, route encoding through NVENC (software VP8/H264 can't
        # keep up at high resolutions) and pin H.264 in the SDP so the
        # hardware path is what actually gets negotiated. Pre-encoded
        # tracks carry H.264 by construction, so their transceivers must
        # negotiate it regardless of the encoder patch.
        hw = install_hw_encoder()
        h264 = [
            c
            for c in RTCRtpSender.getCapabilities("video").codecs
            if c.mimeType.lower() == "video/h264"
        ]
        if h264:
            for transceiver in pc.getTransceivers():
                if hw or isinstance(transceiver.sender.track, PrecodedVideoTrack):
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
