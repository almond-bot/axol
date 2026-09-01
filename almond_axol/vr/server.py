"""
VR WebSocket server for the Axol arm.

VRServer accepts secure WebSocket (WSS) connections from a VR headset and
surfaces the latest VRFrame to the caller. IK and motor control are handled
separately — this class is purely the network layer.

Communication is bidirectional:
  - headset → server: VRFrame JSON every XR frame
  - server → headset: arbitrary JSON (e.g. state feedback via broadcast_text)

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
import socket
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..constants import URDF_PATH
from ..utils.browser_origin import browser_origin_allowed
from ..utils.certs import (
    ACCEPT_PAGE_HTML,
    CERTFILE,
    KEYFILE,
    PreparedTLSFiles,
    prepare_tls_files,
)
from ..utils.ports import open_listen_socket
from .config import VRServerConfig
from .control_channel import ControlChannelManager
from .ice import client_ice_servers
from .interp import PoseInterpolator
from .models import VRFrame

if TYPE_CHECKING:
    from ..video.video import WebRTCManager

_logger = logging.getLogger(__name__)

_quest_datum_lock = threading.Lock()
_last_quest_datum: dict[str, Any] | None = None
_QUEST_DATUM_FRESH_S = 5.0
# A newly loaded Quest page may connect before uvicorn notices that the old
# socket is half-open. Once the active logical producer has stopped sending for
# this long, another producer of the same kind may take ownership immediately.
_POSE_OWNER_TAKEOVER_STALE_S = 1.0
# Duplicated browser tabs inherit sessionStorage, including the logical pose
# source id, but reserve disjoint sequence ranges. A tab can therefore publish
# a higher range and then leave XR while keeping its signaling socket open. Do
# not let that inactive client's orphaned high-water suppress a surviving tab
# indefinitely; after one pose-silence interval the next active client may
# lower the source-wide high-water (with an interpolation reset).
_POSE_SEQUENCE_CLIENT_STALE_S = 1.0
# A browser/native transport has one logical producer in normal operation. A
# small allowance covers compatibility relays and source rollover, while
# preventing one long-lived unauthenticated LAN socket from retaining an
# unbounded number of attacker-chosen source ids in the arbitration maps.
_MAX_POSE_SOURCES_PER_CLIENT = 8


def get_last_quest_pose_datum() -> dict[str, Any] | None:
    """Current controller datum reported by a connected WebXR client.

    ``live`` becomes false when frames stop even if a dead socket has not yet
    timed out. Internal source ownership fields never leave this process.
    """
    with _quest_datum_lock:
        if _last_quest_datum is None:
            return None
        result = {
            key: value
            for key, value in _last_quest_datum.items()
            if not key.startswith("_")
        }
        age = max(0.0, time.monotonic() - _last_quest_datum["_observedMono"])
        result["ageSeconds"] = round(age, 3)
        result["live"] = age <= _QUEST_DATUM_FRESH_S
        return result


def _remember_quest_pose_datum(
    frame: VRFrame, *, server_token: int, source_id: str
) -> None:
    """Expose the live Quest datum to setup UI without persisting user data."""
    global _last_quest_datum
    left = {"profile": frame.l_pose_profile, "poseSpace": frame.l_pose_space}
    right = {"profile": frame.r_pose_profile, "poseSpace": frame.r_pose_space}
    if not any((*left.values(), *right.values())):
        return
    common_key = None
    if left["profile"] and left == right and left["poseSpace"] == "grip":
        common_key = f"quest:{left['profile']}:{left['poseSpace']}"
    value = {
        "left": left,
        "right": right,
        "commonKey": common_key,
        "observedAt": time.time(),
        "_observedMono": time.monotonic(),
        "_serverToken": server_token,
        "_sourceId": source_id,
    }
    with _quest_datum_lock:
        changed = (
            _last_quest_datum is None
            or _last_quest_datum.get("left") != left
            or _last_quest_datum.get("right") != right
            or _last_quest_datum.get("_serverToken") != server_token
            or _last_quest_datum.get("_sourceId") != source_id
        )
        _last_quest_datum = value
    if changed:
        if common_key is not None:
            _logger.info("Quest controller calibration datum: %s", common_key)
        elif left == right and left["poseSpace"] == "target-ray":
            _logger.error(
                "Quest controllers only exposed target-ray poses for %s; "
                "Mantis calibration and collection require gripSpace",
                left["profile"],
            )
        else:
            _logger.warning(
                "Quest controllers reported different or incomplete pose datums: "
                "left=%s/%s right=%s/%s",
                left["profile"],
                left["poseSpace"],
                right["profile"],
                right["poseSpace"],
            )


def _clear_quest_pose_datum(*, server_token: int, source_id: str | None = None) -> None:
    """Forget a datum once its WebXR source or owning server has stopped."""
    global _last_quest_datum
    with _quest_datum_lock:
        if _last_quest_datum is None:
            return
        if _last_quest_datum.get("_serverToken") != server_token:
            return
        if source_id is not None and _last_quest_datum.get("_sourceId") != source_id:
            return
        _last_quest_datum = None


class VRServer:
    """Secure WebSocket server that receives VRFrame data from a VR headset.

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
        self._quest_datum_token = id(self)
        self._on_frame: Callable[[VRFrame], None] | None = None
        self._certfile = config.certfile or CERTFILE
        self._keyfile = config.keyfile or KEYFILE
        self._tls_files: PreparedTLSFiles | None = None
        if config.pose_source_kind not in (None, "webxr", "tracker"):
            raise ValueError(
                "pose_source_kind must be None, 'webxr', or 'tracker'; got "
                f"{config.pose_source_kind!r}"
            )
        expected_source_id = config.expected_pose_source_id
        if expected_source_id is not None and (
            not isinstance(expected_source_id, str)
            or not expected_source_id.strip()
            or len(expected_source_id) > 128
        ):
            raise ValueError(
                "expected_pose_source_id must be a non-empty string of at most "
                "128 characters"
            )
        self._expected_pose_source_kind = config.pose_source_kind
        self._expected_pose_source_id = expected_source_id

        # Operating mode announced to each headset on connect ("teleop" or
        # "data_collection"). The web UI uses it to lock its HUD to a single
        # mode: teleop can't switch to data collection or record, and data
        # collection can't switch back to plain teleop. ``None`` leaves the UI
        # in its legacy free-toggle behaviour (older backends that never set it).
        self._mode: str | None = None

        # Pose convention announced independently from the HUD/recording mode.
        # Relative is the legacy Axol contract (target-ray controllers + body
        # elbows); absolute is the Mantis contract (calibrated grip datum,
        # elbows ignored). Defaulting here keeps direct/older call sites safe.
        self._pose_mode = "relative"

        # Last tracking state reported by the teleop core. Keep it alongside
        # the other connection-seeded state so a managed tracker bridge that
        # reconnects can distinguish an already-engaged core from one that
        # auto-disengaged while the bridge was away.
        self._tracking: bool | None = None

        # Current episode number to show in the headset HUD during data
        # collection (the 1-based index of the episode being recorded next).
        # Broadcast whenever it changes and re-sent to any client that connects
        # mid-session (see the WebSocket accept handler). ``None`` hides the HUD
        # readout — the default for plain teleop, which never sets it.
        self._episode: int | None = None

        # Latest headset HUD state (armed save/discard confirmation popup,
        # record countdown) published by the driving client via a ``hud``
        # signaling message, and that client's id. Relayed to every *other*
        # client so a dashboard can mirror what the headset would show (the
        # operator may drive with the controllers while the headset is off);
        # cleared — with a null broadcast — when the publisher disconnects.
        self._hud: dict[str, Any] | None = None
        self._hud_client: int | None = None
        # Latest pose sequence carried by the signaling client that published
        # HUD state.  One logical Quest source can span an old and replacement
        # socket briefly during reconnect; their TCP streams have no ordering
        # relative to each other.  Keep HUD updates monotonic with the source's
        # pose stream so a delayed old-socket popup cannot overwrite the new
        # socket's replay and then be cleared when that old socket disconnects.
        self._hud_pose_seq: int | None = None

        # The source-wide sequence high-water deduplicates one logical Quest
        # producer copied over USB, WebRTC, and network. Per-client high-water
        # marks let that source-wide maximum be recomputed when one connection
        # leaves: duplicated browser tabs copy sessionStorage's source id but
        # reserve disjoint 1M sequence blocks, so the higher tab must not leave
        # an older surviving tab permanently below an orphaned maximum.
        self._last_seq: dict[str, int] = {}
        self._client_last_seq: dict[tuple[int, str], int] = {}
        self._client_last_seq_seen: dict[tuple[int, str], float] = {}
        self._pose_owner: str | None = None
        self._pose_owner_kind: str | None = None
        self._pose_owner_last_seen: float | None = None
        self._source_kind: dict[str, str] = {}
        self._source_clients: dict[str, set[int]] = {}
        self._client_sources: dict[int, set[str]] = {}
        # Once a tracker bridge has claimed an unrestricted server, never fail
        # over to a still-connected Quest viewer if that bridge disappears.
        # Managed Mantis servers start latched through pose_source_kind.
        self._tracker_source_latched = config.pose_source_kind == "tracker"

        # Whether camera video is expected to become available this session.
        # Camera bring-up (relay subprocess, ZED open, NVENC init) can take
        # tens of seconds after this server starts accepting connections; a
        # headset that sends ``webrtc-request`` in that window must not be
        # told ``webrtc-unavailable`` (it would give up and show no cameras
        # for the whole session). While True and no manager is registered
        # yet, such requests are parked in ``_video_pending`` and answered
        # with ``webrtc-pending``, then resolved with a real offer — or an
        # honest ``webrtc-unavailable`` — once video setup concludes.
        self._video_expected: bool = False
        # webrtc-requests parked while video is still starting: id → socket.
        self._video_pending: dict[int, WebSocket] = {}
        # Clients whose offer is currently being created. Offer creation
        # awaits (signaling pipe, ICE gathering), so a client's retry request
        # can land while its parked request is being flushed; creating a
        # second offer then would close the in-flight peer connection
        # mid-negotiation and can follow a good offer with an erroneous
        # ``webrtc-unavailable``. Guarded per client in _send_webrtc_offer
        # (single event loop, so a plain set suffices).
        self._video_offering: set[int] = set()
        # Same guard for the control (pose) channel offers.
        self._control_offering: set[int] = set()
        # Offer-creation tasks in flight (offers are built off the WebSocket
        # receive loop — see _handle_signaling); referenced here so the event
        # loop can't garbage-collect a running task.
        self._signaling_tasks: set[asyncio.Task[None]] = set()
        # Event loop this server runs on (captured in enable()) so video
        # registration from another thread can schedule the pending flush.
        self._loop: asyncio.AbstractEventLoop | None = None

        self._latest_frame: VRFrame | None = None
        # Adaptive playout buffer: reconstructs a smooth pose stream from
        # batched/jittered network arrivals. Consumers that want smoothing read
        # via get_render_frame(); get_frame() still returns the raw latest.
        self._interp = PoseInterpolator(
            enabled=config.interp_enabled,
            min_delay_s=config.interp_min_delay_s,
            max_delay_s=config.interp_max_delay_s,
            smooth_window_s=config.interp_smooth_window_s,
            outlier_k=config.interp_outlier_k,
        )
        self._client_count: int = 0
        self._active_clients: set[WebSocket] = set()
        self._server_task: asyncio.Task[None] | None = None
        self._uvicorn_server: uvicorn.Server | None = None
        self._listen_socket: socket.socket | None = None
        self._webrtc: WebRTCManager | Any | None = None
        # Dedicated pose data channel (low-latency control transport). Always
        # available — independent of whether any cameras are streaming.
        self._control = ControlChannelManager(self._ingest_pose_text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> VRFrame | None:
        """Return the most recent frame received, or None if none yet.

        Duplicate/out-of-order frames within each sender's stream are dropped
        before reaching here (see :meth:`_ingest_frame_obj`), so this is the
        newest frame of whichever stream delivered last.
        """
        return self._latest_frame

    def get_render_frame(self) -> VRFrame | None:
        """Return the smoothed playout frame for the current instant.

        Renders the pose from the adaptive interpolation buffer (motion is held
        slightly in the past and interpolated; control state is the latest
        received). Falls back to the raw latest frame when interpolation is
        disabled or there isn't enough history yet. The returned object is
        identity-stable so the IK loop can skip redundant solves while idle.
        """
        return self._interp.sample()

    def set_on_frame(self, callback: Callable[[VRFrame], None] | None) -> None:
        """Replace the on_frame callback. Safe to call after construction."""
        self._on_frame = callback

    def set_mode(self, mode: str | None) -> None:
        """Set the operating mode announced to headsets on connect.

        ``mode`` is ``"teleop"`` for ``axol teleop`` or ``"data_collection"``
        for ``axol collect-data``; the web UI locks its HUD accordingly (see
        ``self._mode``). It is pushed on connect and replayed on request after
        the browser's listener is attached. Safe to call before :meth:`enable`.
        """
        self._mode = mode

    def set_pose_mode(self, mode: str) -> None:
        """Set the controller-pose convention announced to every headset.

        This is deliberately separate from :meth:`set_mode`: teleop and data
        collection can each run either ordinary relative Axol control or the
        absolute Mantis mapping. New web clients default to ``"relative"`` if
        an older server never sends the announcement.
        """
        if mode not in ("relative", "absolute"):
            raise ValueError(
                f"pose mode must be 'relative' or 'absolute'; got {mode!r}"
            )
        self._pose_mode = mode

    def set_episode(self, episode: int | None) -> None:
        """Record the current episode number announced to headsets on connect.

        ``episode`` is the 1-based index shown in the in-headset HUD during data
        collection; ``None`` hides it. Stored so a client connecting mid-session
        gets the current value in the WebSocket accept handler — live updates are
        pushed separately via :meth:`broadcast_text`. Safe to call from any
        thread (a plain attribute assignment).
        """
        self._episode = episode

    def set_video_expected(self, expected: bool) -> None:
        """Declare whether camera video is expected to become available.

        Call (before :meth:`enable`, or any time earlier than the first
        headset connection) when cameras are configured but still starting:
        ``webrtc-request``s that arrive before :meth:`set_video_manager` /
        :meth:`set_video_sources` registers a manager are then parked and
        answered with ``{"type": "webrtc-pending"}`` — the headset keeps its
        "connecting cameras" state — and receive their offer automatically
        the moment video setup concludes (or ``webrtc-unavailable`` if it
        concludes without video). Without this, an early request is told
        video is unavailable and the headset shows no camera screens for the
        rest of the session.
        """
        self._video_expected = expected

    def set_video_sources(self, sources: dict[str, Any] | None) -> None:
        """Register per-camera video sources to stream to the headset.

        Each value is a connected ``ZedCamera`` / stereo eye (registered
        directly — the relay adapts it to BGRA frames), a pre-encoded
        ``gst_zed`` camera, or any raw-frame source exposing ``width`` /
        ``height`` / ``fps`` + ``wait_next``; the manager picks the right
        WebRTC track per source (see :mod:`almond_axol.video.video`). The headset
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
            self._conclude_video_setup()
            return
        try:
            from ..video.video import WebRTCManager, webrtc_available

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
            self._conclude_video_setup()
            return
        _logger.info("wrist video enabled for: %s", ", ".join(sources))
        self._conclude_video_setup()

    def set_video_manager(self, manager: Any | None) -> None:
        """Register a pre-built WebRTC manager (e.g. an out-of-process relay).

        ``manager`` must implement the ``WebRTCManager`` signaling interface
        (``create_offer`` / ``set_answer`` / ``close`` / ``close_all``).
        Used by teleop to keep all video encoding and RTP traffic in a
        separate process (``almond_axol.video.video_proc``) so it cannot
        contend with the control loops. Pass ``None`` to disable video.
        """
        self._webrtc = manager
        if manager is not None:
            _logger.info("wrist video enabled (external manager)")
        self._conclude_video_setup()

    def _conclude_video_setup(self) -> None:
        """Resolve parked ``webrtc-request``s now that video setup finished.

        Called by :meth:`set_video_manager` / :meth:`set_video_sources` from
        any thread. Clients parked on ``webrtc-pending`` are answered on the
        server's event loop: a fresh offer when a manager was registered, or
        ``webrtc-unavailable`` when setup concluded without video. Video is
        no longer "expected" either way — later requests are answered
        directly from the registered manager (or its absence).
        """
        self._video_expected = False
        loop = self._loop
        if loop is None or not self._video_pending:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._flush_video_pending(), loop)
        except RuntimeError:
            pass  # server loop already shut down

    async def _flush_video_pending(self) -> None:
        """Answer every parked video request (runs on the server loop)."""
        pending = list(self._video_pending.items())
        self._video_pending.clear()
        for client_id, ws in pending:
            if ws not in self._active_clients:
                continue
            try:
                if self._webrtc is None:
                    await ws.send_text(json.dumps({"type": "webrtc-unavailable"}))
                else:
                    await self._send_webrtc_offer(ws, client_id)
            except Exception as exc:  # noqa: BLE001 - keep serving other clients
                _logger.warning("failed to resolve pending video request: %s", exc)

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

    async def broadcast_tracking(self, enabled: bool) -> None:
        """Store and broadcast the teleop core's current tracking state."""
        self._tracking = bool(enabled)
        await self.broadcast_text(
            json.dumps({"type": "tracking", "value": self._tracking})
        )

    async def _broadcast_hud(self, exclude: WebSocket | None = None) -> None:
        """Relay the current headset HUD state to (other) connected clients."""
        text = json.dumps({"type": "hud", "value": self._hud})
        for ws in list(self._active_clients):
            if ws is exclude:
                continue
            try:
                await ws.send_text(text)
            except Exception as exc:
                _logger.warning("Failed to relay hud to client: %s", exc)

    def _clear_hud_for_pose_owner_change(self) -> None:
        """Invalidate controls published by the previous pose owner.

        Pose ingestion is synchronous for both WebSocket and WebRTC data-
        channel transports. In the live server both run on its event loop, so
        schedule the null broadcast without delaying the new owner's pose. A
        direct synchronous caller (notably unit tests) still gets the fail-
        closed state clear even when no loop is running.
        """
        if self._hud is None and self._hud_client is None:
            return
        should_broadcast = self._hud is not None
        self._hud = None
        self._hud_client = None
        self._hud_pose_seq = None
        if not should_broadcast:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spawn(self._broadcast_hud())

    def _client_can_publish_hud(self, client_id: int, obj: dict[str, Any]) -> bool:
        """Whether this socket carries the active WebXR pose producer.

        HUD countdowns and confirmations are controls, not passive viewer
        metadata. A Quest that is connected only for cameras during a managed
        tracker run must therefore be unable to overwrite the operator panel's
        state. Requiring an already established pose owner also keeps dashboard
        and signaling-only clients read-only.
        """
        owner = self._pose_owner
        if owner is None or self._pose_owner_kind not in ("webxr", "legacy"):
            return False
        if owner not in self._client_sources.get(client_id, set()):
            return False

        declared_id = obj.get("pose_source_id")
        if declared_id is not None and declared_id != owner:
            return False
        declared_kind = obj.get("pose_source_kind")
        return declared_kind in (None, "webxr")

    async def enable(self) -> None:
        """Start the WSS server in the background.

        The listening socket is bound *here* (reclaiming the port from a stale
        listener if needed) so a bind failure raises synchronously instead of
        being swallowed inside uvicorn's background task. uvicorn then adopts
        the already-bound socket via ``serve(sockets=...)``.
        """
        if self._server_task is not None:
            return

        # Captured so video registration (called from other threads) can
        # schedule the pending-request flush onto this loop.
        self._loop = asyncio.get_running_loop()

        tls_files = prepare_tls_files(self._certfile, self._keyfile)
        if tls_files.generated:
            _logger.info("creating self-signed certificate")
        sock: socket.socket | None = None
        try:
            sock = await asyncio.to_thread(open_listen_socket, "0.0.0.0", self._port)
            self._listen_socket = sock

            app = self._build_app()
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self._port,
                log_level="info",
                ssl_certfile=tls_files.certfile,
                ssl_keyfile=tls_files.keyfile,
                # Keepalives stay at uvicorn's defaults (20s ping / 20s timeout).
                # Tighter pings were tried for faster dead-peer detection, but
                # with the camera RTP saturating the operator's WiFi a pong can
                # take >5s on a perfectly healthy link, and killing the pose
                # socket then tears down video and teleop with it. Prompt
                # dead-peer detection isn't needed for safety: pose silence
                # force-disengages teleop within VRTeleopConfig.disengage_timeout
                # and the arms hold position until the operator acts.
            )
            self._uvicorn_server = uvicorn.Server(config)
            self._server_task = asyncio.create_task(
                self._uvicorn_server.serve(sockets=[sock])
            )
        except BaseException:
            if sock is not None:
                sock.close()
            self._listen_socket = None
            tls_files.close()
            raise
        self._tls_files = tls_files
        _logger.info("listening on wss://0.0.0.0:%d/ws", self._port)

    async def disable(self) -> None:
        """Gracefully shut down the WSS server."""
        _clear_quest_pose_datum(server_token=self._quest_datum_token)
        if self._webrtc is not None:
            await self._webrtc.close_all()
        await self._control.close_all()

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

        # uvicorn closes the adopted socket on a clean shutdown, but close it
        # ourselves too so a cancelled/timed-out shutdown still frees the port
        # for the next ``enable()`` instead of leaking the bind.
        if self._listen_socket is not None:
            try:
                self._listen_socket.close()
            except OSError:
                pass
            self._listen_socket = None

        if self._tls_files is not None:
            self._tls_files.close()
            self._tls_files = None

        self._client_count = 0
        self._active_clients.clear()
        self._video_pending.clear()
        self._video_offering.clear()
        self._control_offering.clear()
        self._signaling_tasks.clear()
        self._pose_owner = None
        self._pose_owner_kind = None
        self._pose_owner_last_seen = None
        self._source_kind.clear()
        self._source_clients.clear()
        self._client_sources.clear()
        self._client_last_seq.clear()
        self._client_last_seq_seen.clear()
        self._hud = None
        self._hud_client = None
        self._hud_pose_seq = None
        self._loop = None
        # Fresh session next enable(): don't gate a reloaded headset's restarted
        # seq counter against a stale high-water mark.
        self._last_seq.clear()

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

    async def _handle_message(
        self, websocket: WebSocket, client_id: int, data: str
    ) -> None:
        """Dispatch one inbound text message.

        Signaling messages carry a ``type`` field; pose frames do not.
        """
        try:
            obj = json.loads(data)
        except Exception as exc:
            _logger.warning("invalid json: %s", exc)
            return

        if isinstance(obj, dict) and "type" in obj:
            await self._handle_signaling(websocket, client_id, obj)
            return

        self._ingest_frame_obj(obj, client_id, client_id)

    def _reset_pose_buffer(self) -> None:
        """Drop pose/interpolation state when ownership changes or disappears."""
        self._latest_frame = None
        self._interp.reset()

    def _register_pose_source(
        self, frame: VRFrame, stream: Any, client_id: int
    ) -> tuple[str, str] | None:
        """Resolve and remember a frame's logical producer and kind."""
        raw_id = frame.pose_source_id
        source_id = (
            raw_id
            if isinstance(raw_id, str) and raw_id.strip() and len(raw_id) <= 128
            else f"legacy:{client_id}"
        )
        raw_kind = frame.pose_source_kind
        source_kind = raw_kind if raw_kind in ("webxr", "tracker") else "legacy"
        client_sources = self._client_sources.get(client_id)
        if (
            client_sources is not None
            and source_id not in client_sources
            and len(client_sources) >= _MAX_POSE_SOURCES_PER_CLIENT
        ):
            return None
        previous_kind = self._source_kind.setdefault(source_id, source_kind)
        if previous_kind != source_kind:
            # A logical id cannot change privilege/kind mid-session. Reject
            # instead of coercing to the first kind: coercion could let a
            # WebXR frame reuse a tracker owner's id and inherit its authority.
            return None
        self._source_clients.setdefault(source_id, set()).add(client_id)
        self._client_sources.setdefault(client_id, set()).add(source_id)
        return source_id, source_kind

    def _pose_source_allowed(
        self, source_id: str, source_kind: str, *, now: float
    ) -> bool:
        """Claim or validate exclusive ownership for one logical producer."""
        if (
            self._expected_pose_source_id is not None
            and source_id != self._expected_pose_source_id
        ):
            return False
        expected = self._expected_pose_source_kind
        if expected == "tracker" and source_kind != "tracker":
            return False
        if expected == "webxr" and source_kind not in ("webxr", "legacy"):
            return False

        if self._pose_owner == source_id:
            self._pose_owner_last_seen = now
            return True
        if self._pose_owner is None:
            if self._tracker_source_latched and source_kind != "tracker":
                return False
            self._pose_owner = source_id
            self._pose_owner_kind = source_kind
            self._pose_owner_last_seen = now
            if source_kind == "tracker":
                self._tracker_source_latched = True
            self._reset_pose_buffer()
            _logger.info("pose source %s claimed control (%s)", source_id, source_kind)
            return True

        owner_kind = self._pose_owner_kind
        same_source_family = source_kind == owner_kind or {
            source_kind,
            owner_kind,
        } <= {"webxr", "legacy"}
        owner_stale = (
            self._pose_owner_last_seen is not None
            and now - self._pose_owner_last_seen > _POSE_OWNER_TAKEOVER_STALE_S
        )
        if same_source_family and owner_stale:
            previous = self._pose_owner
            self._clear_hud_for_pose_owner_change()
            self._pose_owner = source_id
            self._pose_owner_kind = source_kind
            self._pose_owner_last_seen = now
            self._reset_pose_buffer()
            _logger.warning(
                "pose source %s replaced stale owner %s (%s)",
                source_id,
                previous,
                source_kind,
            )
            return True

        # On an unrestricted compatibility server, a tracker bridge may take
        # priority over a WebXR client. Managed Mantis avoids this transition
        # entirely by declaring the expected kind before the server starts.
        if source_kind == "tracker" and self._pose_owner_kind != "tracker":
            _logger.warning(
                "tracker pose source %s superseded %s; resetting pose buffer",
                source_id,
                self._pose_owner,
            )
            self._clear_hud_for_pose_owner_change()
            self._pose_owner = source_id
            self._pose_owner_kind = source_kind
            self._pose_owner_last_seen = now
            self._tracker_source_latched = True
            self._reset_pose_buffer()
            return True
        return False

    def _drop_pose_client(self, client_id: int) -> None:
        """Release a client's logical-source memberships on disconnect."""
        for source_id in self._client_sources.pop(client_id, set()):
            self._client_last_seq.pop((client_id, source_id), None)
            self._client_last_seq_seen.pop((client_id, source_id), None)
            clients = self._source_clients.get(source_id)
            if clients is None:
                continue
            clients.discard(client_id)
            if clients:
                previous_max = self._last_seq.get(source_id)
                remaining = [
                    seq
                    for remaining_client in clients
                    if (seq := self._client_last_seq.get((remaining_client, source_id)))
                    is not None
                ]
                if remaining:
                    self._last_seq[source_id] = max(remaining)
                else:
                    self._last_seq.pop(source_id, None)
                if self._pose_owner == source_id and previous_max != self._last_seq.get(
                    source_id
                ):
                    # The active transport/timing domain left. Do not blend
                    # its buffered capture timestamps with the surviving tab
                    # whose lower sequence window is about to resume.
                    self._reset_pose_buffer()
                continue
            self._source_clients.pop(source_id, None)
            self._source_kind.pop(source_id, None)
            self._last_seq.pop(source_id, None)
            _clear_quest_pose_datum(
                server_token=self._quest_datum_token, source_id=source_id
            )
            if self._pose_owner == source_id:
                self._pose_owner = None
                self._pose_owner_kind = None
                self._pose_owner_last_seen = None
                self._reset_pose_buffer()
                _logger.info("pose source %s released control", source_id)

    def _ingest_frame_obj(self, obj: Any, stream: Any, client_id: int) -> bool:
        """Validate a decoded pose object and publish it to the consumer.

        ``stream`` identifies the physical transport for diagnostics and
        legacy compatibility. Sequence de-duplication and ownership are scoped
        to ``frame.pose_source_id`` across every transport.

        Returns True when the object belongs to the active producer (including
        a duplicate copy); False for invalid or view-only-source frames.
        """
        try:
            frame = VRFrame.model_validate(obj)
        except Exception as exc:
            _logger.warning("invalid frame: %s", exc)
            return False

        source = self._register_pose_source(frame, stream, client_id)
        if source is None:
            return False
        source_id, source_kind = source
        # A Quest remains a useful video/URDF viewer during a managed
        # Lighthouse/Ultimate run. Its pose frames stay view-only, but its
        # controller datum is still setup information worth surfacing.
        if source_kind == "webxr":
            _remember_quest_pose_datum(
                frame,
                server_token=self._quest_datum_token,
                source_id=source_id,
            )
        now = time.monotonic()
        if not self._pose_source_allowed(source_id, source_kind, now=now):
            return False

        # A copy with seq <= this logical producer's high-water mark is a
        # duplicate or delayed straggler, even when it arrived over a different
        # transport. Frames without seq retain legacy latest-wins behavior.
        seq = frame.seq
        if seq is not None:
            client_key = (client_id, source_id)
            client_last = self._client_last_seq.get(client_key)
            client_is_new = client_last is None
            if client_last is not None and seq <= client_last:
                return True
            client_seen = self._client_last_seq_seen.get(client_key)
            client_was_stale = (
                client_seen is not None
                and now - client_seen > _POSE_SEQUENCE_CLIENT_STALE_S
            )
            self._client_last_seq[client_key] = seq
            self._client_last_seq_seen[client_key] = now
            last = self._last_seq.get(source_id)
            if last is not None and seq <= last:
                # A duplicated tab may have used a much higher reserved block,
                # then exited XR without closing its WebSocket. Recompute the
                # source maximum from clients that are still publishing poses.
                # The current frame is already in the per-client maps above;
                # when it establishes the new active maximum, accept it as the
                # first frame of the recovered stream after resetting timing.
                active = [
                    client_seq
                    for remaining_client in self._source_clients.get(source_id, set())
                    if (
                        client_seq := self._client_last_seq.get(
                            (remaining_client, source_id)
                        )
                    )
                    is not None
                    and now
                    - self._client_last_seq_seen.get(
                        (remaining_client, source_id), float("-inf")
                    )
                    <= _POSE_SEQUENCE_CLIENT_STALE_S
                ]
                active_max = max(active) if active else seq
                if active_max < last:
                    self._last_seq[source_id] = active_max
                    if (
                        self._hud_pose_seq is not None
                        and self._hud_pose_seq > active_max
                    ):
                        # The higher-range tab/transport stopped publishing and
                        # this lower range is now the live timing domain.  Its
                        # future HUD updates must not remain pinned below the
                        # departed publisher's sequence watermark.
                        self._clear_hud_for_pose_owner_change()
                    self._reset_pose_buffer()
                    if seq < active_max:
                        return True
                else:
                    return True
            elif last is not None and (client_is_new or client_was_stale):
                # A new or previously inactive tab/transport arrived above the
                # active high-water. A page reload resets performance.now(),
                # and a transport switch can change receive timing, so start a
                # fresh interpolation domain before accepting its first frame.
                self._reset_pose_buffer()
            self._last_seq[source_id] = seq

        self._latest_frame = frame
        self._interp.push(frame)
        if self._on_frame is not None:
            self._on_frame(frame)
        return True

    def _ingest_pose_text(self, client_id: int, data: str) -> None:
        """Ingest a pose frame from the control data channel (text message).

        Mirrors the WebSocket pose path; signaling never arrives here, so a
        message carrying a ``type`` field is ignored rather than validated.
        ``client_id`` is the owning signaling client, so pose-sender tracking
        stays accurate when the channel (not the socket) carries the poses.
        """
        try:
            obj = json.loads(data)
        except Exception as exc:
            _logger.warning("invalid json on pose channel: %s", exc)
            return
        if isinstance(obj, dict) and "type" in obj:
            return
        self._ingest_frame_obj(obj, f"control:{client_id}", client_id)

    async def _handle_signaling(
        self, websocket: WebSocket, client_id: int, obj: dict[str, Any]
    ) -> None:
        """Handle a WebRTC signaling message from the headset."""
        msg_type = obj.get("type")

        # Initial announcements can beat a browser component's message
        # listener on a very fast local socket. Let the client request an
        # idempotent replay after its listener is attached.
        if msg_type == "session-config-request":
            await self._send_session_config(websocket)
            return

        # Control data channel (pose transport): negotiated independently of the
        # cameras, so it's handled before the video-availability check below and
        # works even when no video sources are registered.
        #
        # Offer creation is offloaded to a task: this handler runs on the
        # socket's sequential receive loop, and an inline await (ICE
        # gathering, the relay pipe round-trip) would queue every following
        # message behind it — pose frames stall into bursts, and a queued
        # webrtc/control *answer* applies only after a queued retry has
        # already replaced its peer connection, so negotiation never
        # completes.
        if msg_type == "control-request":
            if client_id not in self._control_offering:
                self._spawn(self._send_control_offer(websocket, client_id))
            return
        if msg_type == "control-answer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                await self._control.set_answer(client_id, sdp)
            return

        # Headset HUD state (armed confirmation popup, record countdown): only
        # the active WebXR pose producer may store and relay it. Camera viewers
        # in a tracker-owned session stay read-only.
        if msg_type == "hud":
            if not self._client_can_publish_hud(client_id, obj):
                return
            owner = self._pose_owner
            publisher_pose_seq = (
                self._client_last_seq.get((client_id, owner))
                if owner is not None
                else None
            )
            if (
                publisher_pose_seq is not None
                and self._hud_pose_seq is not None
                and publisher_pose_seq < self._hud_pose_seq
            ):
                # A replacement signaling socket already replayed newer HUD
                # state for this logical pose source.  Messages delayed on the
                # superseded socket must not win the cross-socket race.
                return
            value = obj.get("value")
            self._hud = value if isinstance(value, dict) else None
            self._hud_client = client_id
            self._hud_pose_seq = publisher_pose_seq
            await self._broadcast_hud(exclude=websocket)
            return

        if self._webrtc is None:
            if msg_type == "webrtc-request":
                if self._video_expected:
                    # Cameras are configured but still starting: park the
                    # request (answered with a real offer when video setup
                    # concludes — see _conclude_video_setup) and tell the
                    # headset to keep its connecting state meanwhile.
                    self._video_pending[client_id] = websocket
                    await websocket.send_text(json.dumps({"type": "webrtc-pending"}))
                else:
                    await websocket.send_text(
                        json.dumps({"type": "webrtc-unavailable"})
                    )
            return

        if msg_type == "webrtc-request":
            # A direct answer supersedes any parked copy of this request.
            self._video_pending.pop(client_id, None)
            if client_id in self._video_offering:
                # An offer for this client is already being built (its retry
                # landed mid-flight): that offer answers this request too.
                # Tell the client to keep waiting rather than starting a
                # competing negotiation.
                await websocket.send_text(json.dumps({"type": "webrtc-pending"}))
                return
            # Built off the receive loop — see the control-request comment.
            self._spawn(self._send_webrtc_offer(websocket, client_id))
        elif msg_type == "webrtc-answer":
            sdp = obj.get("sdp")
            if isinstance(sdp, str):
                try:
                    await self._webrtc.set_answer(client_id, sdp)
                except Exception as exc:
                    _logger.error("failed to apply webrtc answer: %s", exc)
        else:
            _logger.debug("ignoring unknown signaling type: %s", msg_type)

    def _spawn(self, coro: Any) -> None:
        """Run a signaling coroutine as its own task on the server loop.

        The task set holds a strong reference so a running task can't be
        garbage-collected mid-flight.
        """
        task = asyncio.create_task(coro)
        self._signaling_tasks.add(task)
        task.add_done_callback(self._signaling_tasks.discard)

    async def _send_control_offer(self, websocket: WebSocket, client_id: int) -> None:
        """Create and send the control-channel offer. Runs as its own task.

        Guarded per client like the video offers; never raises (a send to a
        just-disconnected socket is normal churn).
        """
        if client_id in self._control_offering:
            return
        self._control_offering.add(client_id)
        try:
            sdp = await self._control.create_offer(client_id)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "control-offer",
                        "sdp": sdp,
                        "iceServers": client_ice_servers(),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - task context; log and move on
            _logger.warning("control offer for client %d failed: %s", client_id, exc)
        finally:
            self._control_offering.discard(client_id)

    async def _send_webrtc_offer(self, websocket: WebSocket, client_id: int) -> None:
        """Create a fresh per-client offer and send it (unavailable on failure).

        Runs as its own task (or from the pending flush); never raises. No-op
        while an offer for the same client is already being created — the
        pending flush and a concurrent request retry would otherwise race,
        with the later ``create_offer`` tearing down the earlier one's peer
        connection mid-negotiation. The in-flight offer (sent to this same
        socket) answers the skipped request.
        """
        if client_id in self._video_offering:
            return
        self._video_offering.add(client_id)
        try:
            webrtc = self._webrtc
            if webrtc is None:
                payload: dict[str, Any] = {"type": "webrtc-unavailable"}
            else:
                try:
                    sdp, tracks = await webrtc.create_offer(client_id)
                    payload = {
                        "type": "webrtc-offer",
                        "sdp": sdp,
                        "tracks": tracks,
                        # Same TURN/STUN servers the aiortc peer used, so the
                        # browser gathers a matching relay candidate. Empty on
                        # a LAN (no env config) — harmless to the headset.
                        "iceServers": client_ice_servers(),
                    }
                except Exception as exc:  # noqa: BLE001 - degrade to no video
                    _logger.error("failed to create webrtc offer: %s", exc)
                    payload = {"type": "webrtc-unavailable"}
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception as exc:  # noqa: BLE001 - client left mid-offer
                _logger.warning("failed to send webrtc reply: %s", exc)
        finally:
            self._video_offering.discard(client_id)

    async def _send_session_config(self, websocket: WebSocket) -> None:
        """Best-effort replay of authoritative per-session client state."""
        announcements: list[tuple[str, Any]] = []
        if self._mode is not None:
            announcements.append(("mode", self._mode))
        # The WebXR client uses this to suppress local record controls when a
        # Lighthouse/Ultimate bridge owns poses. Announce null as well so a
        # current client can distinguish unrestricted policy from an older
        # server that does not expose this contract.
        announcements.append(("pose_source_kind", self._expected_pose_source_kind))
        announcements.append(("pose_mode", self._pose_mode))
        if self._tracking is not None:
            announcements.append(("tracking", self._tracking))
        if self._episode is not None:
            announcements.append(("episode", self._episode))
        if self._hud is not None:
            announcements.append(("hud", self._hud))

        for message_type, value in announcements:
            try:
                await websocket.send_text(
                    json.dumps({"type": message_type, "value": value})
                )
            except Exception as exc:  # noqa: BLE001 - best-effort announce
                _logger.warning("failed to send %s to client: %s", message_type, exc)

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        server = self

        # The web app is served from a different origin (axol.almond.bot or a
        # dev server), so its fetches of the URDF/meshes below need CORS. The
        # WebSocket path is unaffected (WS is not subject to CORS).
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        # Robot model for the headset's URDF overlay (absolute/Mantis mode): the
        # web client fetches /urdf/axol.urdf and resolves its
        # package://assembly/... mesh references against /urdf/.
        app.mount("/urdf", StaticFiles(directory=str(URDF_PATH.parent)), name="urdf")

        @app.get("/__accept")
        async def _accept() -> HTMLResponse:
            """Self-closing page the web UI opens to approve the self-signed cert."""
            return HTMLResponse(ACCEPT_PAGE_HTML)

        @app.websocket("/ws")
        async def _ws(websocket: WebSocket) -> None:
            if not browser_origin_allowed(
                websocket.headers.get("origin"),
                scheme=websocket.url.scheme,
                host=websocket.headers.get("host", websocket.url.netloc),
            ):
                await websocket.close(code=1008, reason="browser origin is not allowed")
                return
            await websocket.accept()
            _logger.info("client connected %s", websocket.client)
            server._client_count += 1
            server._active_clients.add(websocket)
            client_id = id(websocket)
            # Seed HUD and pose conventions for clients joining after setup.
            # The browser requests an idempotent replay once its listener is
            # attached too, closing the fast-connect announcement race.
            await server._send_session_config(websocket)
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
                server._video_pending.pop(client_id, None)
                server._client_count = max(0, server._client_count - 1)
                # The HUD publisher (the headset) left: clear its popups from
                # every mirror so a dashboard doesn't show a stale dialog.
                if server._hud_client == client_id:
                    server._hud_client = None
                    server._hud_pose_seq = None
                    if server._hud is not None:
                        server._hud = None
                        await server._broadcast_hud()
                if server._webrtc is not None:
                    await server._webrtc.close(client_id)
                await server._control.close(client_id)
                # A logical producer may span this socket plus a second USB
                # socket. Release ownership only after its final transport
                # disconnects; view-only clients never keep pose state alive.
                server._drop_pose_client(client_id)
                if server._client_count == 0:
                    server._pose_owner = None
                    server._pose_owner_kind = None
                    server._pose_owner_last_seen = None
                    server._last_seq.clear()
                    server._client_last_seq.clear()
                    server._client_last_seq_seen.clear()
                    server._source_kind.clear()
                    server._source_clients.clear()
                    server._client_sources.clear()
                    server._reset_pose_buffer()

        return app
