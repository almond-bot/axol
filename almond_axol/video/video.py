"""aiortc WebRTC relay for streaming ZED camera video to the VR headset.

Each registered source becomes one WebRTC video track the Quest WebXR app
renders directly, so the teleoperator can see the grippers/scene.
:func:`_track_for_source` adapts whatever the caller hands in: a connected ZED
**Python SDK** camera (``ZedCamera`` / stereo eye, grabbed exactly like data
collection) is re-encoded here, while a GPU-resident ``gst_zed`` camera's
pre-encoded H.264 is forwarded with no second encode.

Why aiortc (and not gstreamer ``webrtcbin``): aiortc owns the ICE / DTLS /
SRTP transport in Python, which connects reliably on this multi-homed LAN
(Tailscale + IPv6 link-local + two LAN NICs) where ``webrtcbin``'s libnice
stalls in ICE "checking". On the SDK path, encoding still runs on the Jetson's
hardware NVENC via :mod:`almond_axol.video.hw_video` (a ``gst-launch``
subprocess); pre-encoded sources skip it. Either way aiortc never encodes in
software — it only packetizes and ships RTP.

On the SDK re-encode path, frames are handed in as the SDK's native **BGRA**
(4-channel), so the NVENC pipeline feeds the hardware VIC directly (BGRx ->
NV12) with no CPU colorspace conversion — the same efficiency that keeps the
120 Hz IK loop healthy while three 1920x1200@60 streams encode.

The existing VR WebSocket (``/ws``) is reused purely for SDP signaling — no
new ports. aiortc gathers ICE candidates during ``setLocalDescription`` and
embeds them in the SDP (non-trickle), so on a LAN no separate candidate
exchange is needed.

``aiortc`` is a normal wheel dependency; ``av`` comes with it. The in-Python
NVENC re-encode additionally needs the Jetson's ``nvv4l2h264enc`` (L4T BSP)
plus the host GStreamer tools from ``axol gst.install``; without them, aiortc
falls back to software H.264.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import Any

import av
import numpy as np
from aiortc import (
    RTCConfiguration,
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

from ..vr.ice import (
    ice_servers,
    replicate_candidates_across_mlines,
    summarize_candidates,
)
from .hw_video import install_hw_encoder

_logger = logging.getLogger(__name__)

# How long the frame-driven path waits for a new frame before emitting a
# keepalive repeat of the previous one (camera stall, etc.).
_WAIT_TIMEOUT_MS = 500.0

_NAL_TYPE_IDR = 5
_ANNEXB_START = b"\x00\x00\x00\x01"


def webrtc_available() -> bool:
    """True when aiortc + av import (the WebRTC transport stack is present).

    Hardware NVENC is best-effort on top of this (see
    :func:`~almond_axol.video.hw_video.install_hw_encoder`); when it is missing
    aiortc still streams via software H.264, just slower.
    """
    return True  # importing this module already required aiortc + av


class ZedFrameSource:
    """Frame-driven relay source adapting a ZED SDK camera (or stereo eye).

    Wraps a connected ``ZedCamera`` / stereo eye into the raw-frame ``source``
    that :class:`CameraVideoTrack` consumes: ``wait_next`` blocks until the
    camera produces a frame newer than the last one sent, so every relayed
    frame is encoded the instant it's captured instead of waiting to be sampled
    by a fixed-rate timer. It returns the SDK's **native BGRA** (4-channel) so
    the NVENC pipeline can hand it straight to the hardware via ``nvvidconv`` —
    no CPU colorspace conversion.

    Used by the SDK-grab fallback path (teleop / collect-data / the relay
    subprocess); the GPU-resident ``gst_zed`` cameras instead expose
    ``subscribe()`` and feed :class:`PrecodedVideoTrack` directly.
    """

    def __init__(self, cam: Any) -> None:
        self._cam = cam

    @property
    def width(self) -> int:
        return int(self._cam.width or 0)

    @property
    def height(self) -> int:
        return int(self._cam.height or 0)

    @property
    def fps(self) -> int:
        return int(self._cam.fps or 30)

    def wait_next(self, after_ts: float | None, timeout_ms: float) -> Any:
        target = after_ts + 1e-6 if after_ts is not None else 0.0
        try:
            frame, cap_ts, _recv_ts = self._cam.read_bgra_at_or_after(
                target, timeout_ms=timeout_ms
            )
            return frame, cap_ts
        except Exception:  # noqa: BLE001 - timeout/stall → keepalive in track
            return None


class CameraVideoTrack(VideoStreamTrack):
    """WebRTC video track backed by a connected ZED camera (BGRA frames).

    Frame-driven: ``recv`` blocks on the source's ``wait_next`` until the
    camera produces a frame newer than the last one sent and returns it
    immediately, with the RTP timestamp derived from the frame's *capture*
    time so glass-to-glass latency stays minimal. Frames are the SDK's native
    BGRA (4-channel); the hardware encoder consumes them without a CPU convert.
    """

    kind = "video"

    def __init__(self, source: Any) -> None:
        super().__init__()
        self._source = source
        self._wait_next = source.wait_next
        self._w = int(getattr(source, "width", 0) or 640)
        self._h = int(getattr(source, "height", 0) or 480)
        self._last: NDArray[Any] | None = None
        self._last_cap_ts: float | None = None
        self._ts_origin: float | None = None  # perf_counter of pts == 0

    def _clock_pts(self, perf_ts: float) -> int:
        if self._ts_origin is None:
            self._ts_origin = perf_ts
        return max(0, int((perf_ts - self._ts_origin) * VIDEO_CLOCK_RATE))

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

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
                else np.zeros((self._h, self._w, 4), dtype=np.uint8)
            )
            pts = self._clock_pts(time.perf_counter())

        frame = VideoFrame.from_ndarray(
            np.ascontiguousarray(arr, dtype=np.uint8), format="bgra"
        )
        frame.pts = pts
        frame.time_base = VIDEO_TIME_BASE
        return frame


class PrecodedVideoTrack(MediaStreamTrack):
    """WebRTC track fed by already-encoded H.264 access units.

    Backed by a pre-encoded source (``gst_zed.ZedGstCamera`` / stereo eye;
    duck type: ``subscribe`` / ``unsubscribe`` / ``alive``) whose pipeline
    grabs and encodes entirely on the GPU. ``recv`` returns ``av.Packet``s,
    which aiortc's sender routes through ``encoder.pack`` — straight to RTP
    packetization, with no encode step in Python at all.

    The track stays silent until the first access unit carrying an IDR (the
    pipeline's ``insert-sps-pps`` puts SPS/PPS on every IDR), so a late-joining
    client always starts on a decodable frame.
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


def _track_for_source(source: Any) -> MediaStreamTrack:
    """Build the right WebRTC track for a registered video source.

    The video sources accepted by ``set_video_sources`` are duck-typed so
    callers can hand in whatever they already have:

    * a pre-encoded source exposing ``subscribe()`` (the GPU-resident
      :mod:`~almond_axol.video.gst_zed` cameras) → :class:`PrecodedVideoTrack`,
      whose H.264 access units go straight to RTP with no Python encode;
    * a connected ``ZedCamera`` / stereo eye (exposes ``read_bgra_at_or_after``
      but not ``wait_next``) → wrapped in :class:`ZedFrameSource` so a bare
      camera "just works";
    * anything already exposing ``wait_next`` (e.g. a hand-built
      :class:`ZedFrameSource`) → used as-is.

    The last two feed :class:`CameraVideoTrack`, which aiortc encodes via NVENC.
    """
    if hasattr(source, "subscribe"):
        return PrecodedVideoTrack(source)
    if not hasattr(source, "wait_next") and hasattr(source, "read_bgra_at_or_after"):
        source = ZedFrameSource(source)
    return CameraVideoTrack(source)


class WebRTCManager:
    """Manages per-client peer connections that send ZED camera video.

    One :class:`RTCPeerConnection` per connected headset, keyed by an opaque
    client id. The server drives signaling: it creates the offer (with the
    camera tracks attached) and applies the headset's answer. Implements the
    drop-in signaling interface used by ``VRServer`` and the out-of-process
    relay (``video_proc``): ``create_offer`` / ``set_answer`` / ``close`` /
    ``close_all`` / ``has_sources`` / ``shutdown``.
    """

    def __init__(self, sources: dict[str, Any]) -> None:
        """``sources`` maps each headset-visible track name to a video source.

        A source can be a connected ``ZedCamera`` / stereo eye, a pre-encoded
        ``gst_zed`` camera, or a hand-built :class:`ZedFrameSource`; the right
        WebRTC track is chosen per source (see :func:`_track_for_source`).
        """
        self._sources = dict(sources)
        self._pcs: dict[int, RTCPeerConnection] = {}
        # Per-client (packetsSent, packetsLost) from the last stats poll, for
        # per-interval rates.
        self._prev_stats: dict[int, tuple[int, int]] = {}

    @property
    def has_sources(self) -> bool:
        return bool(self._sources)

    async def log_stats_loop(self, period: float = 1.0) -> None:
        """Log per-client WebRTC send health every ``period`` s (until cancelled).

        Reports the outbound send rate and, from the headset's RTCP reports, the
        packet-loss rate / RTT / jitter — the direct measure of whether the feed
        degrades because packets are being dropped in transit (e.g. the send loop
        not getting CPU) versus the encoder producing bad frames.
        """
        while True:
            await asyncio.sleep(period)
            for client_id, pc in list(self._pcs.items()):
                try:
                    report = await pc.getStats()
                except Exception:  # noqa: BLE001 - stats are best-effort
                    continue
                sent = lost = 0
                rtt = 0.0
                for stat in report.values():
                    kind = getattr(stat, "type", "")
                    if kind == "outbound-rtp":
                        sent += int(getattr(stat, "packetsSent", 0) or 0)
                    elif kind == "remote-inbound-rtp":
                        lost += int(getattr(stat, "packetsLost", 0) or 0)
                        rtt = float(getattr(stat, "roundTripTime", 0.0) or 0.0)
                p_sent, p_lost = self._prev_stats.get(client_id, (sent, lost))
                self._prev_stats[client_id] = (sent, lost)
                d_sent = sent - p_sent
                d_lost = lost - p_lost
                total = d_sent + d_lost
                loss_pct = 100.0 * d_lost / total if total else 0.0
                _logger.info(
                    "webrtc[%d] send: %d pkt/s  lost=%d/s (%.1f%%)  rtt=%.0fms",
                    client_id,
                    d_sent,
                    d_lost,
                    loss_pct,
                    1e3 * rtt,
                )

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a fresh peer connection for ``client_id`` and return the offer.

        Returns ``(sdp, tracks)`` where ``tracks`` maps each negotiated media
        ``mid`` to its camera name so the client can label incoming streams.
        """
        await self.close(client_id)

        # With TURN/STUN configured (off-LAN operator via a tunnel), aiortc
        # gathers a relay candidate and embeds it in the offer SDP below
        # (non-trickle). Unconfigured, this is aiortc's default — the LAN path.
        servers = ice_servers()
        pc = (
            RTCPeerConnection(RTCConfiguration(iceServers=servers))
            if servers
            else RTCPeerConnection()
        )
        self._pcs[client_id] = pc

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            _logger.info("webrtc[%d] connectionState=%s", client_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close(client_id)

        track_names: dict[int, str] = {}
        for name, source in self._sources.items():
            track = _track_for_source(source)
            pc.addTrack(track)
            track_names[id(track)] = name

        # On Jetson, route encoding through NVENC (software H.264 can't keep
        # up at high resolutions) and pin H.264 in the SDP so the hardware
        # path is what actually gets negotiated. Pre-encoded tracks carry
        # H.264 by construction, so their transceivers must negotiate it
        # regardless of the encoder patch.
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
        if servers:
            _logger.info(
                "webrtc[%d] offer %s",
                client_id,
                summarize_candidates(pc.localDescription.sdp),
            )

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
        if ice_servers():
            # Work around an aiortc BUNDLE bug: it keeps only the bundle-tag
            # m-line's transport, so ICE candidates the headset placed on a
            # different bundled m-line get dropped and the media transport
            # stalls in "checking". Replicate candidates onto every m-line so
            # the surviving transport always has them. Only needed off-LAN
            # (when TURN is configured); the LAN path is left untouched.
            fixed = replicate_candidates_across_mlines(sdp)
            if fixed != sdp:
                _logger.info(
                    "webrtc[%d] applied aiortc BUNDLE candidate workaround", client_id
                )
                sdp = fixed
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
        """Close every active peer connection."""
        for client_id in list(self._pcs):
            await self.close(client_id)

    def shutdown(self) -> None:
        """No-op: peer connections are closed via ``close_all`` (async).

        Present so this in-process manager is a drop-in for the out-of-process
        relay (:class:`~almond_axol.video.video_proc.VideoRelayProcess`), whose
        ``shutdown`` tears down the subprocess; callers can invoke ``shutdown``
        unconditionally on either.
        """
