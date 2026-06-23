"""
VR network server for the Axol arm.

VRServer accepts secure WebSocket (WSS) connections from a VR headset and
negotiates WebRTC data channels for high-rate pose transport. It surfaces the
latest VRFrame to the caller. IK and motor control are handled separately —
this class is purely the network layer.

Communication is bidirectional:
  - headset → server: VRFrame JSON over an unreliable unordered WebRTC data channel
  - WebSocket: SDP signaling plus arbitrary JSON feedback (e.g. broadcast_text)

Typical usage::

    async with VRServer() as vr:
        while True:
            frame = vr.get_frame()
            if frame is not None:
                print(frame.l_ee, frame.r_ee, frame.l_elbow, frame.r_elbow)
            await asyncio.sleep(0.01)

Or with an on_frame callback::

    def handle(frame: VRFrame) -> None:
        logging.getLogger(__name__).debug("frame: %s", frame)

    async with VRServer(on_frame=handle) as vr:
        await asyncio.sleep(float("inf"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..utils.certs import ACCEPT_PAGE_HTML, CERTFILE, KEYFILE, create_self_signed_cert
from .config import VRServerConfig
from .models import VRFrame

if TYPE_CHECKING:
    from .video import WebRTCManager

_logger = logging.getLogger(__name__)


class VRServer:
    """Secure WebSocket server plus WebRTC pose channel for a VR headset.

    Args:
        config:  Server configuration (port, TLS paths). Defaults to VRServerConfig().
    """

    def __init__(self, config: VRServerConfig = VRServerConfig()) -> None:
        """Configure the VR WebSocket server.

        The server is not started until :meth:`enable` (or ``async with``) is
        called.  A self-signed TLS certificate is auto-generated in
        ``~/.almond/vr/certs/`` on first use if no cert paths are provided.

        Args:
            config: Port, TLS certificate, and private-key paths.
        """
        self._port = config.port
        self._on_frame: Callable[[VRFrame], None] | None = None
        self._certfile = config.certfile or CERTFILE
        self._keyfile = config.keyfile or KEYFILE

        self._latest_frame: VRFrame | None = None
        self._client_count: int = 0
        self._active_clients: set[WebSocket] = set()
        self._server_task: asyncio.Task[None] | None = None
        self._loop_lag_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._webrtc: WebRTCManager | Any | None = None
        self._pose_pcs: dict[int, Any] = {}
        self._pose_stats: dict[str, Any] | None = None
        self._pose_stats_received_at: float | None = None

        self._telemetry_window_start: float | None = None
        self._telemetry_last_arrival: float | None = None
        self._telemetry_last_seq: int | None = None
        self._telemetry_count: int = 0
        self._telemetry_max_arrival_gap_ms: float = 0.0
        self._telemetry_max_client_dt_ms: float | None = None
        self._telemetry_client_dropped: int = 0
        self._telemetry_max_client_buffered_bytes: int = 0
        self._telemetry_max_loop_lag_ms: float = 0.0
        self._telemetry_missing_seq: int = 0
        self._telemetry_out_of_order_seq: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> VRFrame | None:
        """Return the most recent frame received, or None if none yet."""
        return self._latest_frame

    def set_on_frame(self, callback: Callable[[VRFrame], None] | None) -> None:
        """Replace the on_frame callback. Safe to call after construction."""
        self._on_frame = callback

    def set_video_sources(self, sources: dict[str, Any] | None) -> None:
        """Register per-camera video sources to stream to the headset.

        Each value is a connected ``ZedCamera`` / stereo eye (registered
        directly — the relay adapts it to BGRA frames), a pre-encoded
        ``gst_zed`` camera, or any raw-frame source exposing ``width`` /
        ``height`` / ``fps`` + ``wait_next``; the manager picks the right
        WebRTC track per source (see :mod:`almond_axol.vr.video`). The headset
        negotiates a WebRTC connection over the existing ``/ws`` channel and
        receives one video track per source, encoded on the Jetson's hardware
        NVENC and shipped by aiortc.

        This is the in-process fallback; teleop normally runs the relay in a
        dedicated subprocess via :meth:`set_video_manager`. Pass ``None`` or an
        empty dict to disable video. Requires ``aiortc`` (a normal dependency);
        hardware NVENC additionally needs the system GStreamer stack from
        ``axol gst.install``. If aiortc is unavailable this logs a warning and
        leaves video disabled. Safe to call before or after :meth:`enable`.
        """
        if not sources:
            self._webrtc = None
            return
        try:
            from .video import WebRTCManager, webrtc_available

            if not webrtc_available():
                raise RuntimeError("aiortc unavailable")
            self._webrtc = WebRTCManager(sources)
        except Exception as exc:  # noqa: BLE001 - aiortc missing
            _logger.warning(
                "wrist video requested but the WebRTC stack (aiortc) is "
                "unavailable (%s); install the project dependencies (and "
                "`axol gst.install` for hardware NVENC). Continuing without "
                "wrist video.",
                exc,
            )
            self._webrtc = None
            return
        _logger.info("wrist video enabled for: %s", ", ".join(sources))

    def set_video_manager(self, manager: Any | None) -> None:
        """Register a pre-built WebRTC manager (e.g. an out-of-process relay).

        ``manager`` must implement the ``WebRTCManager`` signaling interface
        (``create_offer`` / ``set_answer`` / ``close`` / ``close_all``).
        Used by teleop to keep all video encoding and RTP traffic in a
        separate process (``almond_axol.vr.video_proc``) so it cannot
        contend with the control loops. Pass ``None`` to disable video.
        """
        self._webrtc = manager
        if manager is not None:
            _logger.info("wrist video enabled (external manager)")

    @property
    def connected(self) -> bool:
        """True if at least one VR client is currently connected."""
        return self._client_count > 0

    async def broadcast_text(self, text: str) -> None:
        """Send a text message to all currently connected VR clients."""
        for ws in list(self._active_clients):
            try:
                await ws.send_text(text)
            except Exception as exc:
                _logger.warning("Failed to send feedback to client: %s", exc)

    async def enable(self) -> None:
        """Start the WSS server in the background."""
        if self._server_task is not None:
            return

        if not os.path.isfile(self._certfile) or not os.path.isfile(self._keyfile):
            _logger.info("creating self-signed certificate")
            create_self_signed_cert(self._certfile, self._keyfile)

        app = self._build_app()
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self._port,
            log_level="info",
            ssl_certfile=self._certfile,
            ssl_keyfile=self._keyfile,
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._loop_lag_task = asyncio.create_task(self._sample_loop_lag())
        self._server_task = asyncio.create_task(self._uvicorn_server.serve())
        _logger.info("listening on wss://0.0.0.0:%d/ws", self._port)

    async def disable(self) -> None:
        """Gracefully shut down the WSS server."""
        await self._close_all_pose_channels()

        if self._webrtc is not None:
            await self._webrtc.close_all()

        if self._loop_lag_task is not None:
            self._loop_lag_task.cancel()
            try:
                await self._loop_lag_task
            except asyncio.CancelledError:
                pass
            self._loop_lag_task = None

        if self._uvicorn_server is not None:
            try:
                await self._uvicorn_server.shutdown()
            except Exception:
                pass
            self._uvicorn_server = None

        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            self._server_task = None

        self._client_count = 0
        self._active_clients.clear()

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> VRServer:
        await self.enable()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _sample_loop_lag(self) -> None:
        """Track worst asyncio scheduling delay on the VR server loop."""
        interval = 0.01
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + interval
        while True:
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
            now = loop.time()
            lag_ms = max(0.0, (now - next_tick) * 1000.0)
            self._telemetry_max_loop_lag_ms = max(
                self._telemetry_max_loop_lag_ms, lag_ms
            )
            # If the loop was blocked for multiple periods, resync instead of
            # reporting a train of artificial catch-up ticks.
            next_tick = max(next_tick + interval, now + interval)

    async def _handle_message(
        self, websocket: WebSocket, client_id: int, data: str
    ) -> None:
        """Dispatch one inbound text message.

        Signaling messages carry a ``type`` field. Pose frames normally arrive
        on the WebRTC data channel; untyped WebSocket frames are accepted for
        compatibility with older clients.
        """
        obj = self._parse_json(data)
        if obj is None:
            return

        if isinstance(obj, dict) and "type" in obj:
            await self._handle_signaling(websocket, client_id, obj)
            return

        self._handle_pose_payload(obj)

    @staticmethod
    def _parse_json(data: str | bytes) -> Any | None:
        """Parse an inbound text/bytes JSON payload, logging malformed input."""
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                _logger.warning("invalid utf-8: %s", exc)
                return None
        try:
            return json.loads(data)
        except Exception as exc:
            _logger.warning("invalid json: %s", exc)
            return None

    def _handle_pose_payload(self, obj: Any) -> None:
        """Validate and publish one headset pose frame."""
        try:
            frame = VRFrame.model_validate(obj)
            self._record_frame_telemetry(frame)
            self._latest_frame = frame
            if self._on_frame is not None:
                self._on_frame(frame)
        except Exception as exc:
            _logger.warning("invalid frame: %s", exc)

    def _reset_frame_telemetry(self) -> None:
        """Clear frame-ingress telemetry counters for a fresh headset session."""
        self._telemetry_window_start = None
        self._telemetry_last_arrival = None
        self._telemetry_last_seq = None
        self._telemetry_count = 0
        self._telemetry_max_arrival_gap_ms = 0.0
        self._telemetry_max_client_dt_ms = None
        self._telemetry_client_dropped = 0
        self._telemetry_max_client_buffered_bytes = 0
        self._telemetry_max_loop_lag_ms = 0.0
        self._telemetry_missing_seq = 0
        self._telemetry_out_of_order_seq = 0
        self._pose_stats = None
        self._pose_stats_received_at = None

    def _record_frame_telemetry(self, frame: VRFrame) -> None:
        """Log one-second summaries of headset send cadence vs server receipt."""
        now = time.perf_counter()
        if self._telemetry_window_start is None:
            self._telemetry_window_start = now

        if self._telemetry_last_arrival is not None:
            arrival_gap_ms = (now - self._telemetry_last_arrival) * 1000.0
            self._telemetry_max_arrival_gap_ms = max(
                self._telemetry_max_arrival_gap_ms, arrival_gap_ms
            )
        self._telemetry_last_arrival = now

        if frame.client_dt_ms is not None:
            self._telemetry_max_client_dt_ms = max(
                self._telemetry_max_client_dt_ms or 0.0, frame.client_dt_ms
            )
        self._telemetry_client_dropped += max(0, frame.client_dropped_since_last)
        if frame.client_buffered_amount is not None:
            self._telemetry_max_client_buffered_bytes = max(
                self._telemetry_max_client_buffered_bytes,
                frame.client_buffered_amount,
            )

        if frame.seq is not None:
            if self._telemetry_last_seq is not None:
                seq_delta = frame.seq - self._telemetry_last_seq
                if seq_delta > 1:
                    self._telemetry_missing_seq += seq_delta - 1
                elif seq_delta <= 0:
                    self._telemetry_out_of_order_seq += 1
            self._telemetry_last_seq = frame.seq

        self._telemetry_count += 1
        elapsed = now - self._telemetry_window_start
        if elapsed < 1.0:
            return

        hz = self._telemetry_count / elapsed if elapsed > 0.0 else 0.0
        client_gap = (
            "n/a"
            if self._telemetry_max_client_dt_ms is None
            else f"{self._telemetry_max_client_dt_ms:.1f}ms"
        )
        seq = (
            "n/a" if self._telemetry_last_seq is None else str(self._telemetry_last_seq)
        )
        _logger.info(
            "vr ingress: %.1f Hz  max_arrival_gap=%.1fms  "
            "max_client_dt=%s  max_loop_lag=%.1fms  "
            "client_dropped=%d  max_client_buffered=%dB  missing_seq=%d  "
            "out_of_order_seq=%d  last_seq=%s  %s",
            hz,
            self._telemetry_max_arrival_gap_ms,
            client_gap,
            self._telemetry_max_loop_lag_ms,
            self._telemetry_client_dropped,
            self._telemetry_max_client_buffered_bytes,
            self._telemetry_missing_seq,
            self._telemetry_out_of_order_seq,
            seq,
            self._format_pose_webrtc_stats(),
        )

        self._telemetry_window_start = now
        self._telemetry_count = 0
        self._telemetry_max_arrival_gap_ms = 0.0
        self._telemetry_max_client_dt_ms = None
        self._telemetry_client_dropped = 0
        self._telemetry_max_client_buffered_bytes = 0
        self._telemetry_max_loop_lag_ms = 0.0
        self._telemetry_missing_seq = 0
        self._telemetry_out_of_order_seq = 0

    async def _handle_signaling(
        self, websocket: WebSocket, client_id: int, obj: dict[str, Any]
    ) -> None:
        """Handle a WebRTC signaling message from the headset."""
        msg_type = obj.get("type")

        if msg_type == "pose-webrtc-offer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                await self._handle_pose_webrtc_offer(websocket, client_id, sdp)
            return
        if msg_type == "pose-webrtc-stats":
            self._record_pose_webrtc_stats(obj)
            return

        if self._webrtc is None:
            if msg_type == "webrtc-request":
                await websocket.send_text(json.dumps({"type": "webrtc-unavailable"}))
            return

        if msg_type == "webrtc-request":
            try:
                sdp, tracks = await self._webrtc.create_offer(client_id)
            except Exception as exc:
                _logger.error("failed to create webrtc offer: %s", exc)
                await websocket.send_text(json.dumps({"type": "webrtc-unavailable"}))
                return
            await websocket.send_text(
                json.dumps({"type": "webrtc-offer", "sdp": sdp, "tracks": tracks})
            )
        elif msg_type == "webrtc-answer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                try:
                    await self._webrtc.set_answer(client_id, sdp)
                except Exception as exc:
                    _logger.error("failed to apply webrtc answer: %s", exc)
        else:
            _logger.debug("ignoring unknown signaling type: %s", msg_type)

    async def _handle_pose_webrtc_offer(
        self, websocket: WebSocket, client_id: int, sdp: str
    ) -> None:
        """Answer a client-created WebRTC data channel for pose frames."""
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except Exception as exc:  # noqa: BLE001 - dependency unavailable
            _logger.warning("pose webrtc unavailable: %s", exc)
            await websocket.send_text(json.dumps({"type": "pose-webrtc-unavailable"}))
            return

        await self._close_pose_channel(client_id)

        pc = RTCPeerConnection()
        self._pose_pcs[client_id] = pc

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            _logger.info(
                "pose-webrtc[%d] connectionState=%s",
                client_id,
                pc.connectionState,
            )
            if pc.connectionState in ("failed", "closed"):
                await self._close_pose_channel(client_id)

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            _logger.info(
                "pose-webrtc[%d] data channel opened: %s",
                client_id,
                channel.label,
            )
            if channel.label != "pose":
                _logger.debug("ignoring unexpected data channel: %s", channel.label)
                return

            @channel.on("message")
            def _on_message(message: str | bytes) -> None:
                obj = self._parse_json(message)
                if obj is not None:
                    self._handle_pose_payload(obj)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await websocket.send_text(
                json.dumps(
                    {"type": "pose-webrtc-answer", "sdp": pc.localDescription.sdp}
                )
            )
        except Exception as exc:  # noqa: BLE001 - report and clear failed pc
            _logger.error("failed to answer pose webrtc offer: %s", exc)
            await self._close_pose_channel(client_id)
            await websocket.send_text(json.dumps({"type": "pose-webrtc-unavailable"}))

    @staticmethod
    def _fmt_stat_ms(value: Any) -> str:
        return f"{value:.1f}ms" if isinstance(value, int | float) else "n/a"

    @staticmethod
    def _fmt_stat_int(value: Any, suffix: str = "") -> str:
        return f"{value:.0f}{suffix}" if isinstance(value, int | float) else "n/a"

    def _record_pose_webrtc_stats(self, obj: dict[str, Any]) -> None:
        """Store the latest browser-side WebRTC stats sample."""
        self._pose_stats = dict(obj)
        self._pose_stats_received_at = time.perf_counter()

    def _format_pose_webrtc_stats(self) -> str:
        """Return a compact browser WebRTC stats summary for ingress logs."""
        stats = self._pose_stats
        received = self._pose_stats_received_at
        if stats is None or received is None:
            return "rtc_stats=n/a"
        age_ms = (time.perf_counter() - received) * 1000.0
        local = stats.get("local_candidate") or "?"
        remote = stats.get("remote_candidate") or "?"
        return (
            f"rtc_rtt={self._fmt_stat_ms(stats.get('current_rtt_ms'))}  "
            f"rtc_age={age_ms:.0f}ms  "
            f"rtc_state={stats.get('pc_state', 'n/a')}/{stats.get('ice_state', 'n/a')}/"
            f"{stats.get('channel_state', 'n/a')}  "
            f"rtc_pair={local}->{remote}  "
            f"rtc_pair_bytes={self._fmt_stat_int(stats.get('pair_bytes_sent'))}/"
            f"{self._fmt_stat_int(stats.get('pair_bytes_received'))}  "
            f"dc_msgs={self._fmt_stat_int(stats.get('data_messages_sent'))}/"
            f"{self._fmt_stat_int(stats.get('data_messages_received'))}"
        )

    async def _close_pose_channel(self, client_id: int) -> None:
        """Close the pose WebRTC peer connection for one headset, if present."""
        pc = self._pose_pcs.pop(client_id, None)
        if pc is not None:
            try:
                await pc.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    async def _close_all_pose_channels(self) -> None:
        """Close every active pose WebRTC peer connection."""
        for client_id in list(self._pose_pcs):
            await self._close_pose_channel(client_id)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        server = self

        @app.get("/__accept")
        async def _accept() -> HTMLResponse:
            """Self-closing page the web UI opens to approve the self-signed cert."""
            return HTMLResponse(ACCEPT_PAGE_HTML)

        @app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            await websocket.accept()
            _logger.info("client connected %s", websocket.client)
            server._reset_frame_telemetry()
            server._client_count += 1
            server._active_clients.add(websocket)
            client_id = id(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    await server._handle_message(websocket, client_id, data)
            except WebSocketDisconnect:
                _logger.info("client disconnected %s", websocket.client)
            except Exception as exc:
                _logger.error("connection error: %s", exc)
                try:
                    await websocket.close()
                except Exception:
                    pass
            finally:
                server._active_clients.discard(websocket)
                server._client_count = max(0, server._client_count - 1)
                await server._close_pose_channel(client_id)
                if server._webrtc is not None:
                    await server._webrtc.close(client_id)

        return app
