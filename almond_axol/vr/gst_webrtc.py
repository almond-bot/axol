"""GStreamer-native WebRTC relay for the VR headset (SDK grab -> NVENC -> webrtcbin).

Both teleop and data collection grab ZED frames with the **Python SDK**
(``ZedCamera`` -> RGB numpy) and hand them to this manager, which pushes
them into a GStreamer pipeline that converts + encodes on the Jetson
(``nvvidconv`` -> ``nvv4l2h264enc``) and sends WebRTC **entirely from
gstreamer** via ``webrtcbin``. No encoded frames ever return to Python —
RTP/SRTP/ICE all run in C inside gstreamer, so the control loops never pay
for video traffic.

Architecture:

* One self-contained pipeline **per connected headset** (keyed by an opaque
  client id). Each pipeline owns one ``appsrc`` per camera, its own encoders,
  and one ``webrtcbin``. Frames are fanned out to every client's ``appsrc``
  in Python (one buffer copy per camera per frame, shared across clients), so
  late joiners and disconnects are just pipeline create/teardown — no dynamic
  pad surgery on a live graph.
* Signaling matches the existing ``/ws`` protocol unchanged: the server is the
  offerer (``create_offer`` -> ``webrtc-offer`` {sdp, tracks}), the headset
  answers (``set_answer``). ICE is **non-trickle**: ``webrtcbin``'s gathered
  candidates are spliced into the offer SDP before it is sent (host candidates
  only, so gathering completes in milliseconds on a LAN).

:class:`GstWebRTCManager` implements the same async interface as the old
aiortc ``WebRTCManager`` (``create_offer`` / ``set_answer`` / ``close`` /
``close_all`` / ``has_sources``) so it is a drop-in for ``VRServer`` and the
out-of-process relay (``video_proc``).

Requires PyGObject and the GStreamer ``webrtcbin`` (gst-plugins-bad), libnice
(``gstreamer1.0-nice``), and the Jetson ``nvv4l2h264enc`` element. These are
installed by ``axol gst.install`` (:mod:`almond_axol.cli.gst.install`), run
once by the host installer — not at teleop/serve startup; see
:func:`gst_webrtc_available`.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Where apt drops the GObject-introspection typelibs. PyGObject is installed
# into the (uv tool) venv but loads these system typelibs; on most systems the
# default search path already covers them, but we prepend defensively so the
# import works regardless of how the venv interpreter was built. This is pure
# process-local config (no privilege), not installation — see ``axol
# gst.install`` (almond_axol.cli.gst.install) for the actual package install.
_TYPELIB_DIRS = (
    "/usr/lib/girepository-1.0",
    f"/usr/lib/{os.uname().machine}-linux-gnu/girepository-1.0",
    "/usr/lib/aarch64-linux-gnu/girepository-1.0",
)


def _set_typelib_path() -> None:
    """Prepend the system typelib dirs to ``GI_TYPELIB_PATH`` for this process."""
    existing = os.environ.get("GI_TYPELIB_PATH", "")
    dirs = [d for d in _TYPELIB_DIRS if Path(d).is_dir()]
    if not dirs:
        return
    parts = dirs + ([existing] if existing else [])
    os.environ["GI_TYPELIB_PATH"] = os.pathsep.join(parts)


# GStreamer modules are imported lazily (PyGObject may be installed on first
# use); _ensure_init() populates these globals.
Gst: Any = None
GstWebRTC: Any = None
GstSdp: Any = None
GLib: Any = None

_init_lock = threading.Lock()
_initialized = False

# How long create_offer waits for the offer SDP + ICE gathering to complete.
_OFFER_TIMEOUT_S = 10.0

# How long a forwarder blocks for a fresh camera frame before looping (so it
# notices the manager shutting down).
_WAIT_TIMEOUT_MS = 500.0


class FrameSource(Protocol):
    """A connected camera (or stereo eye) the manager can pull frames from.

    Mirrors the subset of ``ZedCamera`` the relay uses: fixed ``width`` /
    ``height`` / ``fps`` (known after connect) plus ``wait_next``, which blocks
    until a frame newer than ``after_ts`` is captured so each frame is encoded
    the instant it exists.
    """

    width: int
    height: int
    fps: int

    def wait_next(
        self, after_ts: float | None, timeout_ms: float
    ) -> tuple[NDArray[Any], float] | None: ...


def _ensure_init() -> None:
    """Import PyGObject + GStreamer and ``Gst.init`` exactly once."""
    global Gst, GstWebRTC, GstSdp, GLib, _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        _set_typelib_path()
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstWebRTC", "1.0")
        gi.require_version("GstSdp", "1.0")
        from gi.repository import GLib as _GLib
        from gi.repository import Gst as _Gst
        from gi.repository import GstSdp as _GstSdp
        from gi.repository import GstWebRTC as _GstWebRTC

        _Gst.init(None)
        Gst, GstWebRTC, GstSdp, GLib = _Gst, _GstWebRTC, _GstSdp, _GLib
        _initialized = True


@functools.cache
def gst_webrtc_available() -> bool:
    """True when PyGObject + ``webrtcbin`` + NVENC + payloader are all present."""
    try:
        _ensure_init()
    except Exception as exc:  # noqa: BLE001 - PyGObject/GStreamer missing
        _logger.debug("gst webrtc unavailable: %s", exc)
        return False
    required = ("webrtcbin", "nvv4l2h264enc", "nvvidconv", "h264parse", "rtph264pay")
    missing = [el for el in required if Gst.ElementFactory.find(el) is None]
    if missing:
        _logger.info("gst webrtc unavailable: missing elements %s", ", ".join(missing))
        return False
    _logger.info("gstreamer-native WebRTC (webrtcbin + NVENC) available")
    return True


def _bitrate_for(width: int, height: int, fps: int) -> int:
    """Encoder bitrate (bits/s) for a resolution: ~0.12 bpp, clamped 4-20 Mbps."""
    return max(4_000_000, min(20_000_000, int(width * height * fps * 0.12)))


@dataclass
class _Client:
    """One headset's self-contained pipeline (encoders + webrtcbin)."""

    client_id: int
    pipeline: Any = None
    webrtc: Any = None
    appsrcs: dict[str, Any] = field(default_factory=dict)


class GstWebRTCManager:
    """Per-headset ``webrtcbin`` pipelines fed by SDK camera frames.

    Args:
        sources: Camera name -> frame source (fixed dims/fps + ``wait_next``).
            The cameras must already be connected so their dimensions are known.
    """

    def __init__(self, sources: dict[str, FrameSource]) -> None:
        _ensure_init()
        self._sources = dict(sources)
        # Deterministic camera order: also the m-line / mid order in the offer.
        self._order = list(sources)
        self._specs: dict[str, tuple[int, int, int]] = {}
        for name, src in self._sources.items():
            w, h, fps = int(src.width), int(src.height), int(src.fps)
            if w <= 0 or h <= 0:
                raise ValueError(f"camera {name!r} has unknown dimensions {w}x{h}")
            self._specs[name] = (w, h, fps or 30)

        self._lock = threading.Lock()
        self._clients: dict[int, _Client] = {}
        self._stop = threading.Event()

        # webrtcbin is driven on a dedicated GLib main loop thread; all
        # interaction with the elements is marshaled onto it.
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._loop.run, name="gst-webrtc-loop", daemon=True
        )
        self._loop_thread.start()

        self._forwarders: list[threading.Thread] = []
        for name, src in self._sources.items():
            t = threading.Thread(
                target=self._forward,
                args=(name, src),
                name=f"gst-fwd-{name}",
                daemon=True,
            )
            t.start()
            self._forwarders.append(t)

        _logger.info("gst webrtc manager ready for: %s", ", ".join(self._order))

    @property
    def has_sources(self) -> bool:
        return bool(self._sources)

    # -- frame fan-out --------------------------------------------------------

    def _forward(self, name: str, src: FrameSource) -> None:
        """Pull frames from one camera and push them to every client's appsrc."""
        last_ts: float | None = None
        while not self._stop.is_set():
            try:
                result = src.wait_next(last_ts, _WAIT_TIMEOUT_MS)
            except Exception as exc:  # noqa: BLE001 - source is best-effort
                _logger.debug("camera %s wait_next raised: %s", name, exc)
                result = None
            if result is None:
                continue
            frame, cap_ts = result
            last_ts = cap_ts
            try:
                data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
                buf = Gst.Buffer.new_wrapped(data)
            except Exception as exc:  # noqa: BLE001 - bad frame → skip
                _logger.debug("camera %s buffer wrap failed: %s", name, exc)
                continue
            with self._lock:
                appsrcs = [
                    c.appsrcs[name] for c in self._clients.values() if name in c.appsrcs
                ]
            for appsrc in appsrcs:
                try:
                    appsrc.emit("push-buffer", buf)
                except Exception:  # noqa: BLE001 - client tearing down
                    pass

    # -- pipeline construction ------------------------------------------------

    def _pipeline_str(self) -> str:
        """Build the per-client launch string: appsrc -> NVENC -> webrtcbin."""
        parts = ["webrtcbin name=webrtc latency=0 bundle-policy=max-bundle"]
        for name in self._order:
            w, h, fps = self._specs[name]
            bitrate = _bitrate_for(w, h, fps)
            parts.append(
                f"appsrc name=src_{name} is-live=true do-timestamp=true "
                f"format=time caps=video/x-raw,format=RGB,width={w},height={h},"
                f"framerate={fps}/1 "
                "! queue max-size-buffers=4 leaky=downstream "
                "! videoconvert ! video/x-raw,format=RGBA "
                "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
                f"! nvv4l2h264enc control-rate=1 bitrate={bitrate} preset-level=1 "
                f"insert-sps-pps=true idrinterval={fps} maxperf-enable=true "
                "! h264parse config-interval=-1 "
                "! rtph264pay pt=96 config-interval=-1 aggregate-mode=zero-latency "
                "! application/x-rtp,media=video,encoding-name=H264,payload=96,"
                "clock-rate=90000 "
                "! webrtc."
            )
        return "\n".join(parts)

    # -- WebRTCManager interface (async) -------------------------------------

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a fresh pipeline for ``client_id`` and return ``(sdp, tracks)``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._create_offer_blocking, client_id)

    async def set_answer(self, client_id: int, sdp: str) -> None:
        """Apply the headset's SDP answer for ``client_id``."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._set_answer_blocking, client_id, sdp)

    async def close(self, client_id: int) -> None:
        """Tear down ``client_id``'s pipeline, if any."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._close_blocking, client_id)

    async def close_all(self) -> None:
        """Tear down every active client pipeline."""
        for client_id in list(self._clients):
            await self.close(client_id)

    # -- blocking implementations (run via run_in_executor) ------------------

    def _create_offer_blocking(self, client_id: int) -> tuple[str, dict[str, str]]:
        self._close_blocking(client_id)

        client = _Client(client_id)
        candidates: list[tuple[int, str]] = []
        done = threading.Event()
        box: dict[str, Any] = {}

        def on_ice(_elem: Any, mlineindex: int, candidate: str) -> None:
            candidates.append((mlineindex, candidate))

        def finalize() -> None:
            if done.is_set():
                return
            offer_sdp = box.get("offer_sdp")
            if offer_sdp is None:
                return
            sdp, tracks = self._finalize_offer(offer_sdp, candidates)
            box["sdp"] = sdp
            box["tracks"] = tracks
            done.set()

        def on_gather(elem: Any, _pspec: Any) -> None:
            state = elem.get_property("ice-gathering-state")
            if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
                finalize()

        def on_offer(promise: Any, _user: Any) -> None:
            try:
                reply = promise.get_reply()
                offer = reply.get_value("offer")
                box["offer_sdp"] = offer.sdp.as_text()
                client.webrtc.emit("set-local-description", offer, Gst.Promise.new())
                # Host-only gathering can complete synchronously on set-local.
                if (
                    client.webrtc.get_property("ice-gathering-state")
                    == GstWebRTC.WebRTCICEGatheringState.COMPLETE
                ):
                    finalize()
            except Exception as exc:  # noqa: BLE001 - report upstream
                box["error"] = exc
                done.set()

        def build() -> bool:
            try:
                pipeline = Gst.parse_launch(self._pipeline_str())
                webrtc = pipeline.get_by_name("webrtc")
                client.pipeline = pipeline
                client.webrtc = webrtc
                for name in self._order:
                    client.appsrcs[name] = pipeline.get_by_name(f"src_{name}")
                webrtc.connect("on-ice-candidate", on_ice)
                webrtc.connect("notify::ice-gathering-state", on_gather)
                pipeline.set_state(Gst.State.PLAYING)
                # Register so the forwarders start feeding this pipeline.
                with self._lock:
                    self._clients[client_id] = client
                promise = Gst.Promise.new_with_change_func(on_offer, None)
                webrtc.emit("create-offer", None, promise)
            except Exception as exc:  # noqa: BLE001 - report upstream
                box["error"] = exc
                done.set()
            return False  # one-shot idle callback

        GLib.idle_add(build)

        if not done.wait(_OFFER_TIMEOUT_S):
            self._close_blocking(client_id)
            raise TimeoutError("webrtcbin offer/ICE gathering timed out")
        if "error" in box:
            self._close_blocking(client_id)
            raise box["error"]
        return box["sdp"], box["tracks"]

    def _finalize_offer(
        self, offer_sdp: str, candidates: list[tuple[int, str]]
    ) -> tuple[str, dict[str, str]]:
        """Splice gathered ICE candidates into the offer; map mid -> camera."""
        _res, msg = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(offer_sdp.encode(), msg)

        by_idx: dict[int, list[str]] = {}
        for idx, cand in candidates:
            value = cand
            if value.startswith("a="):
                value = value[2:]
            if value.startswith("candidate:"):
                value = value[len("candidate:") :]
            by_idx.setdefault(idx, []).append(value)

        tracks: dict[str, str] = {}
        for i in range(msg.medias_len()):
            media = msg.get_media(i)
            mid = media.get_attribute_val("mid")
            if mid is not None and i < len(self._order):
                tracks[mid] = self._order[i]
            for value in by_idx.get(i, []):
                media.add_attribute("candidate", value)
            media.add_attribute("end-of-candidates", None)
        return msg.as_text(), tracks

    def _set_answer_blocking(self, client_id: int, sdp: str) -> None:
        with self._lock:
            client = self._clients.get(client_id)
        if client is None or client.webrtc is None:
            _logger.warning("webrtc answer for unknown client %d", client_id)
            return

        def apply() -> bool:
            try:
                _res, msg = GstSdp.SDPMessage.new()
                GstSdp.sdp_message_parse_buffer(sdp.encode(), msg)
                answer = GstWebRTC.WebRTCSessionDescription.new(
                    GstWebRTC.WebRTCSDPType.ANSWER, msg
                )
                client.webrtc.emit("set-remote-description", answer, Gst.Promise.new())
            except Exception as exc:  # noqa: BLE001 - keep serving
                _logger.error("failed to apply answer for %d: %s", client_id, exc)
            return False

        self._run_on_loop(apply)

    def _close_blocking(self, client_id: int) -> None:
        with self._lock:
            client = self._clients.pop(client_id, None)
        if client is None or client.pipeline is None:
            return

        def teardown() -> bool:
            try:
                client.pipeline.set_state(Gst.State.NULL)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                _logger.debug("pipeline teardown for %d: %s", client_id, exc)
            return False

        self._run_on_loop(teardown)

    # -- lifecycle ------------------------------------------------------------

    def _run_on_loop(self, fn: Any) -> None:
        """Run ``fn`` (returns False) on the GLib loop thread and block for it."""
        done = threading.Event()

        def wrapper() -> bool:
            try:
                fn()
            finally:
                done.set()
            return False

        GLib.idle_add(wrapper)
        done.wait(timeout=5.0)

    def shutdown(self) -> None:
        """Stop all client pipelines, forwarders, and the GLib main loop."""
        self._stop.set()
        for client_id in list(self._clients):
            self._close_blocking(client_id)
        for t in self._forwarders:
            t.join(timeout=1.0)
        try:
            self._loop.quit()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
