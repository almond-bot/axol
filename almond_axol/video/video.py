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

On the SDK re-encode path, frames are sampled at a fixed 30 fps and handed in
as the SDK's native **BGRA** (4-channel), so the NVENC pipeline feeds the
hardware VIC directly (BGRx -> NV12) with no CPU colorspace conversion. Camera
capture remains independent and may run faster for recording or policy input.

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
from collections.abc import Callable
from dataclasses import dataclass, field
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
from aiortc.rtp import RTCP_PSFB_APP, RtcpPsfbPacket, unpack_remb_fci
from av import VideoFrame
from numpy.typing import NDArray

from ..vr.ice import (
    ice_servers,
    replicate_candidates_across_mlines,
    summarize_candidates,
)
from .constants import HEADSET_STREAM_FPS
from .hw_video import install_hw_encoder
from .link_control import (
    LIVE_EDGE_PERSIST_S,
    LIVE_EDGE_SKIP_IDR_S,
    LIVE_EDGE_SKIP_S,
    BitrateController,
    LinkSample,
    RembState,
    transport_backlog_bytes,
)

_logger = logging.getLogger(__name__)

# How often the link controller re-reads each peer and rescales the encoders,
# and how often it logs one link-health line per peer.
_LINK_TICK_S = 0.5
_LINK_LOG_S = 5.0

# How long a pre-encoded track waits for its next access unit before checking
# whether the camera pipeline is still alive.
_AU_WAIT_TIMEOUT_MS = 500.0

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
    """Latest-frame relay source adapting a ZED SDK camera (or stereo eye).

    Wraps a connected ``ZedCamera`` / stereo eye into the raw-frame ``source``
    that :class:`CameraVideoTrack` consumes. At each 30 Hz stream tick it reads
    the newest frame non-destructively, without clearing the event used by
    recording/policy waiters. It returns the SDK's **native BGRA** (4-channel)
    so the NVENC pipeline can hand it straight to the hardware via
    ``nvvidconv`` — no CPU colorspace conversion.

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
        return HEADSET_STREAM_FPS

    def wait_next(self, after_ts: float | None, timeout_ms: float) -> Any:
        try:
            latest = getattr(self._cam, "read_latest_bgra_with_ts", None)
            if latest is not None:
                frame, cap_ts, _recv_ts = latest()
            else:
                # Compatibility for external camera-like sources implementing
                # the former relay contract. In-repo ZED cameras always take
                # the non-consuming path above.
                target = after_ts + 1e-6 if after_ts is not None else 0.0
                frame, cap_ts, _recv_ts = self._cam.read_bgra_at_or_after(
                    target, timeout_ms=timeout_ms
                )
            if after_ts is not None and cap_ts <= after_ts:
                return None
            return frame, cap_ts
        except Exception:  # noqa: BLE001 - startup/stall → keepalive in track
            return None


class CameraVideoTrack(VideoStreamTrack):
    """WebRTC video track backed by a connected ZED camera (BGRA frames).

    Paced at the fixed headset rate independently of camera capture. Each tick
    samples the newest source frame, repeating the previous frame during a
    camera stall. A per-peer absolute monotonic deadline prevents a delayed
    encode from causing catch-up bursts. Frames are the SDK's native BGRA
    (4-channel); the hardware encoder consumes them without a CPU convert.
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
        self._interval = 1.0 / HEADSET_STREAM_FPS
        self._next_send_at: float | None = None

    async def _pace(self) -> float:
        """Wait for the next 30 Hz slot, resetting after rather than bursting."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_send_at is None or self._next_send_at < now:
            self._next_send_at = now
        delay = self._next_send_at - now
        if delay > 0.0:
            await asyncio.sleep(delay)
        self._next_send_at += self._interval
        return time.perf_counter()

    def _clock_pts(self, perf_ts: float) -> int:
        if self._ts_origin is None:
            self._ts_origin = perf_ts
        return max(0, int((perf_ts - self._ts_origin) * VIDEO_CLOCK_RATE))

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        send_ts = await self._pace()

        result: tuple[NDArray[Any], float] | None = None
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._wait_next, self._last_cap_ts, 0.0
            )
        except Exception as exc:  # source is best-effort; never kill the track
            _logger.debug("video source wait_next raised: %s", exc)

        if result is not None:
            arr, cap_ts = result
            self._last = arr
            self._last_cap_ts = cap_ts
        else:
            # Camera stalled: repeat the last frame (or black) as a keepalive
            # so the connection and keyframe machinery stay alive.
            arr = (
                self._last
                if self._last is not None
                else np.zeros((self._h, self._w, 4), dtype=np.uint8)
            )

        frame = VideoFrame.from_ndarray(
            np.ascontiguousarray(arr, dtype=np.uint8), format="bgra"
        )
        frame.pts = self._clock_pts(send_ts)
        frame.time_base = VIDEO_TIME_BASE
        frame.duration = round(VIDEO_CLOCK_RATE / HEADSET_STREAM_FPS)
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
        # Seconds of video already queued for this track's peer, installed by
        # WebRTCManager once the peer exists (None: never skip). See
        # link_control: past LIVE_EDGE_SKIP_S the track skips to the next IDR
        # rather than adding to a queue the link can't drain.
        self.backlog_s: Callable[[], float] | None = None
        self.frames_skipped = 0
        self._over_since: float | None = None
        self._clock: Callable[[], float] = time.monotonic

    def stop(self) -> None:
        self._source.unsubscribe(self._queue)
        super().stop()

    def _pop(self) -> list[bytes] | None:
        try:
            return self._queue.get(timeout=_AU_WAIT_TIMEOUT_MS / 1000.0)
        except queue.Empty:
            return None

    def _clock_pts(self, perf_ts: float) -> int:
        if self._ts_origin is None:
            self._ts_origin = perf_ts
        return max(0, int((perf_ts - self._ts_origin) * VIDEO_CLOCK_RATE))

    def _should_skip(self, is_idr: bool) -> bool:
        """Whether the link has held too much video for too long to send this AU.

        The backlog must stay over ``LIVE_EDGE_SKIP_S`` for
        ``LIVE_EDGE_PERSIST_S``: an IDR's own burst queues briefly even on a
        link that carries the average rate, and skipping the P-frame after it
        would cost a whole GOP.
        """
        if self.backlog_s is None:
            return False
        try:
            queued = self.backlog_s()
        except Exception:  # noqa: BLE001 - introspection is best-effort
            return False
        if queued <= LIVE_EDGE_SKIP_S:
            self._over_since = None
            return False
        now = self._clock()
        if self._over_since is None:
            self._over_since = now
        if now - self._over_since < LIVE_EDGE_PERSIST_S:
            return False
        return queued > LIVE_EDGE_SKIP_IDR_S if is_idr else True

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
            if getattr(self._queue, "gap", False):
                # The camera dropped an AU for this subscriber: the GOP is
                # broken, so resume on the next IDR.
                self._queue.gap = False  # type: ignore[attr-defined]
                self._synced = False
            is_idr = any((nal[0] & 0x1F) == _NAL_TYPE_IDR for nal in au)
            if self._should_skip(is_idr):
                self.frames_skipped += 1
                self._synced = False
                continue
            if not self._synced:
                if not is_idr:
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
    * a connected ``ZedCamera`` / stereo eye (exposes the latest-BGRA API, or
      the legacy ``read_bgra_at_or_after`` API, but not ``wait_next``) → wrapped
      in :class:`ZedFrameSource` so a bare camera "just works" at the fixed
      headset cadence;
    * anything already exposing ``wait_next`` (e.g. a hand-built
      :class:`ZedFrameSource`) → used as-is.

    The last two feed :class:`CameraVideoTrack`, which aiortc encodes via NVENC.
    """
    if hasattr(source, "subscribe"):
        return PrecodedVideoTrack(source)
    is_zed_like = hasattr(source, "read_latest_bgra_with_ts") or hasattr(
        source, "read_bgra_at_or_after"
    )
    if not hasattr(source, "wait_next") and is_zed_like:
        source = ZedFrameSource(source)
    return CameraVideoTrack(source)


@dataclass
class _PeerLink:
    """Link-control state for one peer connection."""

    pc: RTCPeerConnection
    tracks: list[PrecodedVideoTrack] = field(default_factory=list)
    remb: RembState = field(default_factory=RembState)
    bytes_sent: int | None = None
    bytes_at: float | None = None
    sent_bps: float = 0.0
    last: LinkSample = field(default_factory=LinkSample)

    @property
    def frames_skipped(self) -> int:
        return sum(t.frames_skipped for t in self.tracks)


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
        # Live-edge + adaptive-bitrate state (see link_control).
        self._links: dict[int, _PeerLink] = {}
        self._controller = BitrateController()
        self._link_task: asyncio.Task[None] | None = None
        self._last_link_log = 0.0

    @property
    def has_sources(self) -> bool:
        return bool(self._sources)

    # -- link control (live edge + adaptive bitrate) --------------------------

    def _adjustable_sources(self) -> list[Any]:
        return [
            s
            for s in self._sources.values()
            if int(getattr(s, "stream_bitrate_ceiling", 0) or 0) > 0
        ]

    def _ceiling_bps(self) -> int:
        """Sum of the stream encoders' configured bitrates (scale 1.0)."""
        return sum(int(s.stream_bitrate_ceiling) for s in self._adjustable_sources())

    def _current_bps(self) -> float:
        """What the stream encoders are asked for now, summed (bits/s)."""
        ceiling = self._ceiling_bps()
        if ceiling <= 0:
            return 0.0
        return ceiling * self._controller.scale

    def _backlog_s(self, client_id: int) -> float:
        """Seconds of video queued for ``client_id`` at the current bitrate."""
        link = self._links.get(client_id)
        if link is None:
            return 0.0
        rate = self._current_bps()
        if rate <= 0:
            return 0.0
        return 8.0 * transport_backlog_bytes(link.pc) / rate

    def _hook_remb(self, client_id: int, sender: RTCRtpSender) -> None:
        """Record the receiver's bandwidth estimate (REMB) for ``client_id``.

        aiortc hands REMB to the sender's encoder, which the pre-encoded path
        never uses; wrap the sender's RTCP handler to see it too.
        """
        original = sender._handle_rtcp_packet

        async def handle(packet: Any) -> None:
            if isinstance(packet, RtcpPsfbPacket) and packet.fmt == RTCP_PSFB_APP:
                link = self._links.get(client_id)
                if link is not None:
                    try:
                        bitrate, _ssrcs = unpack_remb_fci(packet.fci)
                    except ValueError:
                        pass
                    else:
                        link.remb.record(bitrate, time.monotonic())
            await original(packet)

        sender._handle_rtcp_packet = handle  # type: ignore[method-assign]

    async def _sample_link(self, link: _PeerLink, now: float) -> LinkSample:
        loss: float | None = None
        rtt: float | None = None
        sent = 0
        try:
            report = await link.pc.getStats()
        except Exception:  # noqa: BLE001 - stats are best-effort
            report = {}
        for stat in report.values():
            kind = getattr(stat, "type", "")
            if kind == "outbound-rtp":
                sent += int(getattr(stat, "bytesSent", 0) or 0)
            elif kind == "remote-inbound-rtp":
                # aiortc reports the RTCP fraction-lost byte (x/256).
                fraction = getattr(stat, "fractionLost", None)
                if fraction is not None:
                    value = float(fraction) / 256.0
                    loss = value if loss is None else max(loss, value)
                rtt_s = getattr(stat, "roundTripTime", None)
                if rtt_s:
                    rtt = float(rtt_s) if rtt is None else max(rtt, float(rtt_s))
        if link.bytes_sent is not None and link.bytes_at is not None:
            dt = now - link.bytes_at
            if dt > 0:
                link.sent_bps = 8.0 * max(0, sent - link.bytes_sent) / dt
        link.bytes_sent, link.bytes_at = sent, now
        link.last = LinkSample(
            backlog_bytes=transport_backlog_bytes(link.pc),
            loss=loss,
            rtt_s=rtt,
            remb_bps=link.remb.usable(now),
        )
        return link.last

    def _apply_scale(self, scale: float) -> None:
        for source in self._adjustable_sources():
            try:
                source.set_stream_bitrate_scale(scale)
            except Exception as exc:  # noqa: BLE001 - keep streaming as is
                _logger.debug("stream bitrate change failed: %s", exc)

    async def _link_tick(self, now: float) -> None:
        # Only connected peers count: one that is still timing out after its
        # headset left must not throttle the stream for everyone else.
        samples = [
            await self._sample_link(link, now)
            for link in self._links.values()
            if getattr(link.pc, "connectionState", "connected") == "connected"
        ]
        ceiling = self._ceiling_bps()
        if ceiling <= 0:
            return
        before = self._controller.scale
        after = self._controller.update(now, ceiling, samples)
        if abs(after - before) > 1e-6:
            self._apply_scale(after)
            if after < before:
                _logger.info(
                    "headset video: lowered to %.1f of %.1f Mbps (%s)",
                    ceiling * after / 1e6,
                    ceiling / 1e6,
                    self._controller.reason or "link congested",
                )
            elif after >= 1.0:
                _logger.info("headset video: back to the full %.1f Mbps", ceiling / 1e6)
        if now - self._last_link_log >= _LINK_LOG_S:
            self._last_link_log = now
            self._log_links(ceiling)

    def _log_links(self, ceiling: int) -> None:
        for client_id, link in self._links.items():
            s = link.last
            _logger.info(
                "webrtc[%d] link: sending %.1f Mbps (encoders %.1f of %.1f Mbps), "
                "loss %s, round trip %s, queued %d KB, receiver estimate %s, "
                "frames skipped %d",
                client_id,
                link.sent_bps / 1e6,
                ceiling * self._controller.scale / 1e6,
                ceiling / 1e6,
                "n/a" if s.loss is None else f"{100.0 * s.loss:.1f}%",
                "n/a" if s.rtt_s is None else f"{1e3 * s.rtt_s:.0f} ms",
                s.backlog_bytes // 1024,
                "n/a" if s.remb_bps is None else f"{s.remb_bps / 1e6:.1f} Mbps",
                link.frames_skipped,
            )

    async def _link_loop(self) -> None:
        while True:
            await asyncio.sleep(_LINK_TICK_S)
            if not self._links:
                continue
            try:
                await self._link_tick(time.monotonic())
            except Exception as exc:  # noqa: BLE001 - never kill the relay
                _logger.debug("link control tick failed: %s", exc)

    def _ensure_link_loop(self) -> None:
        if self._link_task is None or self._link_task.done():
            self._link_task = asyncio.get_running_loop().create_task(
                self._link_loop(), name="webrtc-link-control"
            )

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
        # (non-trickle). Unconfigured (the LAN path), pass an explicitly
        # *empty* server list: a bare RTCPeerConnection() falls back to
        # aiortc's default Google STUN server, and on a robot that can't
        # reach it every offer stalls ~5s in ICE gathering waiting for the
        # STUN timeout — host candidates are all the LAN path needs.
        servers = ice_servers()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self._pcs[client_id] = pc
        link = _PeerLink(pc=pc)
        self._links[client_id] = link

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            _logger.info("webrtc[%d] connectionState=%s", client_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close(client_id)

        track_names: dict[int, str] = {}
        for name, source in self._sources.items():
            track = _track_for_source(source)
            sender = pc.addTrack(track)
            track_names[id(track)] = name
            self._hook_remb(client_id, sender)
            if isinstance(track, PrecodedVideoTrack):
                track.backlog_s = lambda cid=client_id: self._backlog_s(cid)
                link.tracks.append(track)
        self._ensure_link_loop()

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
        self._links.pop(client_id, None)
        self._prev_stats.pop(client_id, None)
        if not self._links and self._controller.scale < 1.0:
            # Nobody is watching: the next viewer starts at the full rate.
            self._controller.reset()
            self._apply_scale(1.0)
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
