"""Out-of-process WebRTC video relay (GPU-resident gst grab + aiortc).

Sending video from inside the teleop process measurably starves the
control loops: pushing thousands of RTP packets per second plus encoding
is real CPU/GIL work, and it stretches the IK round-trip the moment a
headset connects. The same isolation pattern used for the IK solver
applies here — video runs in a dedicated subprocess and the control
process never touches it.

The subprocess owns the ZED cameras through :mod:`almond_axol.video.gst_zed`:
the ``zedxonesrc`` / ``zedsrc`` GStreamer elements grab and NVENC-encode
entirely on the GPU, and Python only ever sees the encoded H.264 access
units, which aiortc forwards as pre-encoded packets (no Python encode step).
If the gst stack is unavailable the relay falls back to the ZED SDK grab +
in-Python NVENC path (``hw_video``). Either way aiortc owns the ICE / DTLS /
SRTP transport, which connects reliably on this multi-homed LAN where
gstreamer ``webrtcbin``'s libnice stalls. WebRTC media flows over the
subprocess's own UDP sockets, so the only traffic crossing the process
boundary is SDP signaling (a few messages per headset connection) over a
``multiprocessing`` pipe.

:class:`VideoRelayProcess` is the parent-side handle. It implements the
same async interface as ``WebRTCManager`` (``create_offer`` /
``set_answer`` / ``close`` / ``close_all`` / ``has_sources``), so
``VRServer.set_video_manager`` can use it as a drop-in.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import multiprocessing.connection
import os
import threading
import time

_logger = logging.getLogger(__name__)

# How long the relay subprocess may take to open every camera and report
# readiness (each camera takes a few seconds to start streaming).
_READY_TIMEOUT_S = 60.0

_REQUEST_TIMEOUT_S = 15.0
# ``Future.cancel()`` cannot stop a function already running in an executor.
# Poll the pipe in short slices so cancellation of the owning asyncio task can
# cooperatively release the signaling lock well inside VRTeleop's 5 s teardown
# bound, even when the relay never answers the offer request.
_REQUEST_CANCEL_POLL_S = 0.05


# ---------------------------------------------------------------------------
# Subprocess side
# ---------------------------------------------------------------------------


def _open_sdk_camera(name: str, spec: dict) -> object | None:
    """Open one ZED camera with the Python SDK, preferring 60 fps capture.

    Returns the connected ``ZedCamera`` / ``ZedStereoCamera`` (or ``None`` if
    the camera is absent). 60 fps halves frame staleness vs the GMSL 30 fps
    default; cameras that reject it fall back to their default rate.
    """
    from ..lerobot.camera.camera_zed import ZedCamera, ZedStereoCamera
    from ..lerobot.camera.configuration_zed import ZED_RESOLUTION_DIMS, ZedCameraConfig

    serial = spec["serial"]
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    dims = ZED_RESOLUTION_DIMS.get(resolution)
    width, height = dims if dims is not None else (None, None)
    cls = ZedStereoCamera if stereo else ZedCamera
    for fps in (spec.get("fps", 60), None):
        cam = cls(
            ZedCameraConfig(
                serial=serial,
                fps=fps,
                width=width,
                height=height,
                stereo=stereo,
            )
        )
        try:
            cam.connect(warmup=False)
            return cam
        except RuntimeError as exc:  # live-param mismatch (e.g. 60 fps) → retry
            _logger.info("video relay: %s rejected %s fps (%s)", name, fps, exc)
        except Exception as exc:  # noqa: BLE001 - camera absent → skip it
            _logger.warning("video relay: %s failed to open (%s)", name, exc)
            return None
    return None


def _gsth264_meta(
    socket_path: str,
    width: int,
    height: int,
    fps: int,
    latency_s: float,
    pts_origin_perf: float | None,
) -> dict:
    """Describe the encoded-H.264 shared-memory transport for one dataset source.

    The relay encodes the dataset stream on the GPU and writes AU-aligned H.264
    to ``socket_path`` via GDP + ``shmsink``; the recorder attaches an
    :class:`~almond_axol.video.shm_frames.EncodedAuReader` and muxes it. Shared
    memory alone carries no buffer metadata, so GDP preserves the H.264 caps and
    sensor PTS. ``pts_origin_perf`` maps the relay pipeline's running-time zero
    onto the cross-process ``perf_counter`` clock. ``latency_s`` is retained as
    a conservative fallback when either value is missing or corrupt; it is the
    minimum latency reported by GStreamer's pipeline query and excludes
    recorder-side ``shmsrc`` / Python scheduling latency.
    """
    return {
        "transport": "gstshm-h264",
        "socket_path": socket_path,
        "width": width,
        "height": height,
        "fps": fps,
        "latency_s": latency_s,
        "pts_origin_perf": pts_origin_perf,
    }


def _pyshm_meta(shm_name: str, width: int, height: int, fps: int) -> dict:
    return {
        "transport": "pyshm",
        "shm_name": shm_name,
        "width": width,
        "height": height,
        "fps": fps,
    }


def _gsth264_transport_available() -> bool:
    """Whether both ends of the metadata-preserving shm path are installed."""
    from .gst_zed import _element_available

    return all(
        _element_available(element)
        for element in ("shmsink", "shmsrc", "gdppay", "gdpdepay")
    )


def _plan(name: str, eyes: list[str], suffix: bool) -> list[tuple[str, str]]:
    """``[(eye_side, source_name)]`` from an eye list + naming flag.

    With ``suffix`` the eyes are exposed as ``{name}_left`` / ``{name}_right``
    (the head camera, rendered per-lens); without it a single eye is exposed
    under the plain ``{name}`` so it is indistinguishable from a mono camera
    downstream — one encode/record, one source.
    """
    return [(side, f"{name}_{side}" if suffix else name) for side in eyes]


def _eye_plan(name: str, spec: dict) -> list[tuple[str, str]]:
    """Encoded (headset stream) ``[(eye_side, source_name)]`` for a stereo spec.

    Prefers the stream-specific keys (``stream_eyes`` / ``stream_suffix``) and
    falls back to the legacy coupled keys (``eyes`` / ``eye_suffix``) so an
    encoded-only spec (teleop) keeps working unchanged.
    """
    eyes = spec.get("stream_eyes") or spec.get("eyes") or ["left", "right"]
    suffix = spec.get("stream_suffix", spec.get("eye_suffix", True))
    return _plan(name, eyes, suffix)


def _stream_sbs(spec: dict) -> bool:
    """Whether a stereo spec streams both eyes packed side-by-side (one track).

    Packing both eyes into a single double-width stream halves the headset's
    decoder sessions (its actual bottleneck: two full-res sessions drop the
    Quest's render rate from 120 Hz). Only the gst pipeline can pack; the SDK
    fallback ignores the flag and streams per-eye as before.
    """
    return bool(spec.get("stream_sbs"))


def _raw_plan(name: str, spec: dict) -> list[tuple[str, str]]:
    """Raw (dataset recording) ``[(eye_side, source_name)]`` for a stereo spec.

    Independent of :func:`_eye_plan`: the recorded eyes (``record_eyes`` /
    ``record_suffix``) can differ from the streamed eyes — e.g. stream both
    eyes for depth while recording only the left. Falls back to the legacy
    coupled keys so a spec that only sets ``eyes`` records what it streams.
    """
    eyes = spec.get("record_eyes") or spec.get("eyes") or ["left", "right"]
    suffix = spec.get("record_suffix", spec.get("eye_suffix", True))
    return _plan(name, eyes, suffix)


def _open_gst_camera_raw(
    name: str, spec: dict, cond: object, socket_dir: str | None
) -> tuple[object, dict[str, object], list, dict[str, dict]] | None:
    """Open one camera via the gst pipeline with both encoded + raw branches.

    Like :func:`_open_gst_camera`, but additionally exports each source's
    dataset frames to the recorder process. Two transports:

    * **gstshm-h264** (``socket_dir`` set — gst's shm + GDP elements are
      available): the branch encodes H.264 and ends in native GDP + ``shmsink``
      (pure C), so the relay does **zero** Python per frame and preserves sensor
      PTS for the recorder's :class:`EncodedAuReader`.
    * **pyshm** (fallback): a Python pull loop copies each frame into a
      :class:`RawFrameWriter` shared-memory block (the older path; runs the copy
      in the relay's interpreter).

    A spec can force the pyshm transport with ``raw_transport: "pyshm"`` even
    when gst's shm plugin is available: pyshm blocks are readable by the
    *control* process (``VideoRelayProcess.raw_cameras`` become real
    :class:`RawFrameReader` proxies instead of dims-only stubs), which callers
    need when something in the control process must consume the raw frames —
    e.g. a policy building observations while the recorder subprocess records
    the same frames. The cost is the relay-side per-frame Python copy this
    path always had; the encoded headset branch is unaffected.

    Returns ``(owned_camera, {track: source}, [writers], {source: meta})`` — where
    ``meta`` is the per-source dict from :func:`_gsth264_meta` / :func:`_pyshm_meta`
    — or ``None`` when the gst stack/camera is unavailable (the caller then falls
    back to the in-process camera pipeline).
    """
    from .gst_zed import (
        _RESOLUTION_DIMS,
        ZedGstCamera,
        ZedGstStereoCamera,
        zed_gst_available,
        zed_stereo_gst_available,
    )
    from .shm_frames import RawFrameWriter

    serial = int(spec["serial"])
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    if stereo and not zed_stereo_gst_available():
        return None
    if not stereo and not zed_gst_available():
        return None
    if resolution not in _RESOLUTION_DIMS:
        return None
    width, height = _RESOLUTION_DIMS[resolution]
    # The dataset (raw) frames are what cross to the recorder process and feed the
    # NVENC encoder; downscale them on the relay's VIC when the caller asks for a
    # smaller dataset resolution (clamped here, so it never upscales). The encoded
    # headset stream keeps the full capture resolution. The shm blocks/sockets,
    # raw_meta, and gst raw caps must all agree on these dims.
    raw_w, raw_h = width, height
    ds_name = spec.get("dataset_resolution")
    if ds_name in _RESOLUTION_DIMS:
        dw, dh = _RESOLUTION_DIMS[ds_name]
        if dw < width or dh < height:
            raw_w, raw_h = dw, dh
    raw_dims = (raw_w, raw_h)
    use_shm = socket_dir is not None and spec.get("raw_transport") != "pyshm"
    # A camera can opt out of either branch: stream-only (no raw / dataset) or
    # record-only (no encoded / headset). This path is only entered when the
    # camera records (see _relay_main), so the raw branch is always built; the
    # encoded branch is gated on ``stream``.
    wants_stream = bool(spec.get("stream", True))

    for fps in (int(spec.get("fps", 60)), 30):
        writers: list = []
        try:
            if stereo and use_shm:
                # Encoded eyes feed the headset; raw eyes feed the dataset — they
                # may differ (stream both, record one). Build the union; each eye
                # is cropped once and tee'd into whichever branch wants it. When
                # both eyes stream they are packed into one side-by-side track
                # ({name}_sbs) instead of two per-eye ones (see _stream_sbs).
                sbs = wants_stream and _stream_sbs(spec)
                enc_plan = _eye_plan(name, spec) if wants_stream and not sbs else []
                raw_plan = _raw_plan(name, spec)
                enc_sides = [side for side, _ in enc_plan]
                raw_sides = [side for side, _ in raw_plan]
                socks = {
                    side: os.path.join(socket_dir, f"{src}.sock")
                    for side, src in raw_plan
                }
                eye_kwargs: dict = {}
                if "left" in socks:
                    eye_kwargs["left_raw_socket_path"] = socks["left"]
                if "right" in socks:
                    eye_kwargs["right_raw_socket_path"] = socks["right"]
                cam: object = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    raw_dims=raw_dims,
                    encoded_eyes=enc_sides,
                    raw_eyes=raw_sides,
                    encoded_sbs=sbs,
                    **eye_kwargs,
                )
                cam.connect()
                sources = {
                    src: (cam.left_view if side == "left" else cam.right_view)
                    for side, src in enc_plan
                }
                if sbs:
                    sources[f"{name}_sbs"] = cam.sbs_view
                raw_meta = {
                    src: _gsth264_meta(
                        socks[side],
                        raw_w,
                        raw_h,
                        fps,
                        getattr(cam, "raw_latency_s", 0.0),
                        getattr(cam, "raw_pts_origin_perf", None),
                    )
                    for side, src in raw_plan
                }
                return cam, sources, [], raw_meta
            if stereo:
                sbs = wants_stream and _stream_sbs(spec)
                enc_plan = _eye_plan(name, spec) if wants_stream and not sbs else []
                raw_plan = _raw_plan(name, spec)
                enc_sides = [side for side, _ in enc_plan]
                raw_sides = [side for side, _ in raw_plan]
                eye_writers = {
                    side: RawFrameWriter.create(raw_w, raw_h, cond)
                    for side, _ in raw_plan
                }
                writers = list(eye_writers.values())
                eye_kwargs = {}
                if "left" in eye_writers:
                    eye_kwargs["left_raw_sink"] = eye_writers["left"].publish
                if "right" in eye_writers:
                    eye_kwargs["right_raw_sink"] = eye_writers["right"].publish
                cam = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    raw_dims=raw_dims,
                    encoded_eyes=enc_sides,
                    raw_eyes=raw_sides,
                    encoded_sbs=sbs,
                    **eye_kwargs,
                )
                cam.connect()
                sources = {
                    src: (cam.left_view if side == "left" else cam.right_view)
                    for side, src in enc_plan
                }
                if sbs:
                    sources[f"{name}_sbs"] = cam.sbs_view
                raw_meta = {
                    src: _pyshm_meta(eye_writers[side].name, raw_w, raw_h, fps)
                    for side, src in raw_plan
                }
                return cam, sources, writers, raw_meta
            # Mono: the camera streams only when ``stream`` is set; a record-only
            # mono camera builds the raw branch alone (no encoded source, so it is
            # not exposed to the headset).
            if use_shm:
                sock = os.path.join(socket_dir, f"{name}.sock")
                cam = ZedGstCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=wants_stream,
                    raw_socket_path=sock,
                    raw_dims=raw_dims,
                )
                cam.connect()
                meta = {
                    name: _gsth264_meta(
                        sock,
                        raw_w,
                        raw_h,
                        fps,
                        getattr(cam, "raw_latency_s", 0.0),
                        getattr(cam, "raw_pts_origin_perf", None),
                    )
                }
                return cam, ({name: cam} if wants_stream else {}), [], meta
            writer = RawFrameWriter.create(raw_w, raw_h, cond)
            writers = [writer]
            cam = ZedGstCamera(
                serial,
                resolution,
                fps,
                want_encoded=wants_stream,
                raw_sink=writer.publish,
                raw_dims=raw_dims,
            )
            cam.connect()
            return (
                cam,
                {name: cam} if wants_stream else {},
                writers,
                {name: _pyshm_meta(writer.name, raw_w, raw_h, fps)},
            )
        except Exception as exc:  # noqa: BLE001 - try lower fps, then give up
            for w in writers:
                w.close()
            _logger.info("video relay: gst-raw %s @ %s fps failed (%s)", name, fps, exc)
    return None


def _open_gst_camera(name: str, spec: dict) -> tuple[object, dict[str, object]] | None:
    """Open one camera via the GPU-resident gst pipeline (encoded only).

    Returns ``(owned_camera, {track_name: source})`` where each source exposes
    ``subscribe()`` so the WebRTC manager forwards its pre-encoded H.264 AUs
    directly. The relay never needs raw frames, so the raw branch is omitted
    (lowest cost). Returns ``None`` when the gst stack or camera is unavailable
    so the caller can fall back to the SDK path.
    """
    from .gst_zed import (
        ZedGstCamera,
        ZedGstStereoCamera,
        zed_gst_available,
        zed_stereo_gst_available,
    )

    serial = int(spec["serial"])
    resolution = spec.get("resolution") or "HD1200"
    stereo = bool(spec.get("stereo"))
    if stereo and not zed_stereo_gst_available():
        return None
    if not stereo and not zed_gst_available():
        return None

    for fps in (int(spec.get("fps", 60)), 30):
        try:
            if stereo and _stream_sbs(spec):
                # Both eyes packed into one side-by-side track ({name}_sbs).
                cam: object = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    want_raw=False,
                    encoded_eyes=[],
                    encoded_sbs=True,
                )
                cam.connect()
                return cam, {f"{name}_sbs": cam.sbs_view}
            if stereo:
                plan = _eye_plan(name, spec)
                gst_eyes = "both" if len(plan) == 2 else plan[0][0]
                cam = ZedGstStereoCamera(
                    serial,
                    resolution,
                    fps,
                    want_encoded=True,
                    want_raw=False,
                    eyes=gst_eyes,
                )
                cam.connect()
                sources = {
                    src: (cam.left_view if side == "left" else cam.right_view)
                    for side, src in plan
                }
                return cam, sources
            cam = ZedGstCamera(
                serial,
                resolution,
                fps,
                want_encoded=True,
                want_raw=False,
            )
            cam.connect()
            return cam, {name: cam}
        except Exception as exc:  # noqa: BLE001 - try lower fps, then SDK
            _logger.info("video relay: gst %s @ %s fps failed (%s)", name, fps, exc)
    return None


def _relay_main(
    conn: multiprocessing.connection.Connection,
    cameras: dict[str, dict],
    log_level: int,
    want_raw: bool = False,
    raw_cond: object = None,
) -> None:
    """Relay subprocess entry point: open cameras, serve signaling requests.

    When ``want_raw`` is set (data collection), each camera is opened on the gst
    pipeline with a raw branch whose frames are published to shared memory for
    the control process; ``raw_cond`` is the shared
    :class:`multiprocessing.Condition` guarding those blocks. Cameras that can't
    provide raw via gst still stream to the headset (encoded only) but are
    omitted from ``raw_meta`` so the parent can fall back to the in-process
    camera path.
    """
    logging.basicConfig(level=log_level)

    # Disable the cyclic garbage collector in the relay. aiortc sends WebRTC media
    # on this process's asyncio loop, and a stop-the-world gen2 GC pause freezes
    # that loop for ~100ms — which is exactly what stalls the send during
    # recording (the raw branch's per-frame allocations push GC over its
    # threshold), making the headset feed laggy + grainy. The per-frame objects
    # (numpy frame views, encoded byte buffers) are all refcounted, so they free
    # promptly without the collector; only reference cycles would linger, which is
    # acceptable for a session-scoped subprocess.
    import gc

    gc.disable()

    # Pin the relay to its own cores (away from both the control loop and the
    # dataset recorder/encoders). Its WebRTC send is latency-sensitive; sharing
    # cores with the dataset throughput during recording starves the send and
    # makes the headset feed laggy + grainy. Isolated, it gets prompt CPU like in
    # teleop. Where affinity isn't available, fall back to a positive nice so it
    # at least doesn't preempt the control loop.
    from ..utils import affinity

    if not affinity.pin_relay():
        try:
            os.nice(10)
        except (AttributeError, OSError):
            pass

    from .video import WebRTCManager

    # Keep the camera objects alive for the relay's lifetime; ``sources`` maps
    # the per-track names the headset sees to a video source per camera/eye.
    # Prefer the GPU-resident gst pipeline; fall back to the SDK grab — a bare
    # ZedCamera/eye, which WebRTCManager adapts to a frame-driven NVENC source.
    owned: list[object] = []
    sources: dict[str, object] = {}
    writers: list[object] = []
    raw_meta: dict[str, dict] = {}
    # Prefer the gst-native GDP/shm transport for dataset frames: it exports each
    # encoded AU to the recorder in C, preserving its sensor PTS while the relay
    # does zero Python per frame and WebRTC keeps the GIL it needs. Fall back to
    # the in-relay Python copy (RawFrameWriter) unless every element on both ends
    # is installed. A per-relay-PID dir holds one socket per source; removed on
    # exit.
    socket_dir: str | None = None
    if want_raw:
        if _gsth264_transport_available():
            import tempfile

            socket_dir = tempfile.mkdtemp(prefix="axol-raw-")
    for name, spec in cameras.items():
        # A camera participates per its spec: ``stream`` adds it to the headset
        # feed, ``record`` (only meaningful when the relay wants raw) adds it to
        # the dataset. A camera in neither is skipped — never opened.
        wants_stream = bool(spec.get("stream", True))
        wants_record = want_raw and bool(spec.get("record", True))
        if not (wants_stream or wants_record):
            continue
        if wants_record:
            raw = _open_gst_camera_raw(name, spec, raw_cond, socket_dir)
            if raw is not None:
                cam, gst_sources, cam_writers, cam_meta = raw
                owned.append(cam)
                sources.update(gst_sources)
                writers.extend(cam_writers)
                raw_meta.update(cam_meta)
                continue
            # No gst raw path for this camera. If it also streams, fall through to
            # the encoded-only path below; if it's record-only, there's nothing
            # the relay can supply (SDK has no raw export), so skip it and let
            # collect-data fall back to the in-process camera path.
            if not wants_stream:
                continue
        gst = _open_gst_camera(name, spec)
        if gst is not None:
            cam, gst_sources = gst
            owned.append(cam)
            sources.update(gst_sources)
            continue
        cam = _open_sdk_camera(name, spec)
        if cam is None:
            continue
        owned.append(cam)
        if spec.get("stereo"):
            # One grab, per-eye views. The head camera maps both eyes
            # (overhead_left / overhead_right); a wrist stereo camera maps only
            # its left eye under the plain name (see _eye_plan).
            for side, src in _eye_plan(name, spec):
                sources[src] = cam.left_view if side == "left" else cam.right_view
        else:
            sources[name] = cam

    manager = WebRTCManager(sources) if sources else None
    conn.send(("ready", sorted(sources), raw_meta))
    if manager is None:
        if raw_meta:
            # Record-only: every camera has streaming disabled, so there's no
            # headset feed — but the raw branch still exports frames for the
            # dataset. Not an error.
            _logger.info(
                "video relay: recording only (%d raw source(s), no headset streams)",
                len(raw_meta),
            )
        else:
            _logger.warning(
                "video relay opened no cameras; nothing to stream or record"
            )

    async def _loop_lag_monitor() -> None:
        """Log the relay event-loop's worst scheduling lag each second.

        This is the asyncio analog of the control loop's maxgap: it measures how
        late a fixed-interval wakeup actually fires on the relay's loop — the same
        loop aiortc sends WebRTC media on. A large lag during recording means the
        send is being starved *inside the relay process* (by the dataset raw
        branch), which core isolation can't fix; a small lag means the send is
        prompt and a degraded feed is downstream (network).
        """
        loop = asyncio.get_running_loop()
        period = 0.05
        worst = 0.0
        last = loop.time()
        while True:
            t0 = loop.time()
            await asyncio.sleep(period)
            worst = max(worst, loop.time() - t0 - period)
            now = loop.time()
            if now - last >= 1.0:
                _logger.info("relay event-loop maxlag=%.1fms", 1e3 * worst)
                worst = 0.0
                last = now

    async def serve() -> None:
        loop = asyncio.get_running_loop()
        # The WebRTC send-health logger (packets sent / lost / RTT) and event-loop
        # lag monitor were the recording-jitter instrumentation; only run them at
        # DEBUG so the default output stays quiet (they're dedicated diagnostic
        # tasks — no point spinning them when nothing logs).
        if _logger.isEnabledFor(logging.DEBUG):
            if manager is not None:
                loop.create_task(manager.log_stats_loop())
            loop.create_task(_loop_lag_monitor())
        while True:
            try:
                msg = await loop.run_in_executor(None, conn.recv)
            except (EOFError, OSError):  # parent exited — shut down
                break
            if msg is None:
                break
            kind = msg[0]
            try:
                if kind == "offer":
                    _, request_id, client_id = msg
                    if manager is None:
                        conn.send(("offer_err", request_id, client_id, "no cameras"))
                        continue
                    try:
                        sdp, tracks = await manager.create_offer(client_id)
                        conn.send(("offer_ok", request_id, client_id, sdp, tracks))
                    except Exception as exc:  # noqa: BLE001 - report upstream
                        _logger.error("video relay: offer failed: %s", exc)
                        conn.send(("offer_err", request_id, client_id, str(exc)))
                elif kind == "answer" and manager is not None:
                    _, client_id, sdp = msg
                    await manager.set_answer(client_id, sdp)
                elif kind == "close" and manager is not None:
                    _, client_id = msg
                    await manager.close(client_id)
                elif kind == "close_all" and manager is not None:
                    await manager.close_all()
                elif kind == "raw_enable":
                    _, enabled = msg
                    for cam in owned:
                        if hasattr(cam, "set_raw_enabled"):
                            cam.set_raw_enabled(enabled)
            except Exception as exc:  # noqa: BLE001 - keep serving
                _logger.error("video relay: error handling %s: %s", kind, exc)
        if manager is not None:
            await manager.close_all()

    # Split the relay across its two cores: all Python threads (this WebRTC-send
    # main thread + the encoded-AU pull loops) onto one, GStreamer's C threads
    # (camera/NVENC and the recording raw-branch shm copy) onto the other, so the
    # recording workload can't preempt the send. Done here — after the cameras'
    # gst pipelines are PLAYING (all their threads exist) and right before the
    # event loop runs on this thread.
    affinity.isolate_relay_cpu()

    try:
        asyncio.run(serve())
    finally:
        if manager is not None:
            manager.shutdown()
        for cam in owned:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        for writer in writers:
            try:
                writer.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        if socket_dir is not None:
            import shutil

            shutil.rmtree(socket_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class _RawCameraStub:
    """Dims-only stand-in for a relay raw source on the gst-shm transport.

    On the shmsink path the control process never reads raw frames — the recorder
    owns the ``shmsrc`` consumer — so the control process needs only the camera's
    width/height/fps to size the dataset observation features and to satisfy the
    robot's camera lifecycle (``connect``/``disconnect`` are no-ops on a proxy).
    Reads raise: nothing in the control process should pull frames from these.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, warmup: bool = True) -> None:
        pass

    def disconnect(self) -> None:
        pass

    close = disconnect

    def _no_read(self, *args: object, **kwargs: object):
        raise RuntimeError(
            "raw frames are read by the recorder subprocess on the gst-shm "
            "transport, not the control process."
        )

    read = read_at_or_after = read_latest = read_latest_with_ts = _no_read


class VideoRelayProcess:
    """Parent-side handle for the video relay subprocess.

    Implements the ``WebRTCManager`` interface so it can be handed to
    ``VRServer.set_video_manager``. Signaling requests are serialized over
    one pipe (they are rare — a handful per headset connection), each run
    in an executor so the caller's event loop never blocks.
    """

    def __init__(self, cameras: dict[str, dict], want_raw: bool = False) -> None:
        """Spawn the relay and block until its cameras are streaming.

        Args:
            cameras: Per-source spec: ``{name: {"serial": int,
                "resolution": str, "fps": int, "stereo": bool}}``.
            want_raw: Also publish each camera's raw RGB frames to shared memory
                for the control process (data collection). Successfully exported
                sources appear in :attr:`raw_cameras` as
                :class:`~almond_axol.video.shm_frames.RawFrameReader` proxies.
        """
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        # One Condition guards every source's shared-memory metadata; it must be
        # created here and passed at spawn so parent (readers) and child
        # (writers) share the same underlying primitive.
        self._raw_cond = ctx.Condition() if want_raw else None
        self._proc = ctx.Process(
            target=_relay_main,
            args=(
                child_conn,
                cameras,
                logging.getLogger().level or logging.INFO,
                want_raw,
                self._raw_cond,
            ),
            daemon=True,
            name="video-relay",
        )
        self._lock = threading.Lock()
        self._next_offer_request_id = 0
        self._shutdown_requested = threading.Event()

        self.sources: list[str] = []
        self.raw_cameras: dict[str, object] = {}
        # ``{source: meta}`` describing each raw source's transport (gstshm socket
        # + caps, or pyshm block name) and dims — exposed (with :attr:`raw_cond`)
        # so the recorder subprocess can attach its own consumer per source.
        self.raw_meta: dict[str, dict] = {}
        try:
            # Retain every field shutdown() needs before the fallible spawn. A
            # Process.start() interruption can arrive after the child has already
            # inherited all camera devices but before start() returns.
            self._proc.start()
            child_conn.close()
            if self._conn.poll(_READY_TIMEOUT_S):
                msg = self._conn.recv()
                if isinstance(msg, tuple) and msg[0] == "ready":
                    self.sources = list(msg[1])
                    raw_meta = msg[2] if len(msg) > 2 else {}
                    self.raw_meta = dict(raw_meta)
                    self._attach_raw_readers(raw_meta)
        except BaseException as startup_error:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                startup_error.add_note(
                    "additional video-relay startup cleanup failure: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        finally:
            try:
                child_conn.close()
            except OSError:
                pass
        if not self.sources and not self.raw_cameras:
            _logger.warning("video relay started no camera streams or raw sources")
        elif not self.sources:
            _logger.info(
                "video relay: recording only (%d raw source(s), no headset streams)",
                len(self.raw_cameras),
            )

    @property
    def raw_cond(self) -> object:
        """The shared ``multiprocessing.Condition`` guarding the raw shm blocks.

        Must be passed at spawn time to any other process that attaches a
        :class:`~almond_axol.video.shm_frames.RawFrameReader` to these blocks.
        """
        return self._raw_cond

    def _attach_raw_readers(self, raw_meta: dict[str, dict]) -> None:
        """Expose each relay raw source to the control process.

        On the **gstshm** transport the control process never reads frames (the
        recorder owns the shmsrc consumer), so attach a dims-only
        :class:`_RawCameraStub`. On the **pyshm** fallback, attach a
        :class:`RawFrameReader` over the shared-memory block.
        """
        if not raw_meta:
            return
        from .shm_frames import RawFrameReader

        for source, meta in raw_meta.items():
            try:
                if str(meta["transport"]).startswith("gstshm"):
                    self.raw_cameras[source] = _RawCameraStub(
                        meta["width"], meta["height"], meta["fps"]
                    )
                elif self._raw_cond is not None:
                    self.raw_cameras[source] = RawFrameReader(
                        meta["shm_name"],
                        meta["width"],
                        meta["height"],
                        meta["fps"],
                        self._raw_cond,
                    )
            except Exception as exc:  # noqa: BLE001 - skip a source we can't map
                _logger.warning(
                    "video relay: could not attach raw frames for %s: %s",
                    source,
                    exc,
                )

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)

    def _request_offer(
        self, client_id: int, cancelled: threading.Event
    ) -> tuple[str, dict[str, str]] | None:
        with self._lock:
            self._next_offer_request_id += 1
            request_id = self._next_offer_request_id
            if self._shutdown_requested.is_set():
                raise RuntimeError("video relay is shutting down")
            self._conn.send(("offer", request_id, client_id))
            deadline = time.monotonic() + _REQUEST_TIMEOUT_S
            while True:
                # The asyncio wrapper sets this before propagating cancellation.
                # Returning releases ``_lock``; the result is discarded because
                # its Future is already cancelled.
                if cancelled.is_set():
                    return None
                if self._shutdown_requested.is_set():
                    raise RuntimeError("video relay is shutting down")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("video relay did not answer the offer request")
                if not self._conn.poll(min(_REQUEST_CANCEL_POLL_S, remaining)):
                    continue
                msg = self._conn.recv()
                # A cancelled/timed-out request may receive its response after a
                # replacement has started. Request IDs let the replacement drain
                # that stale response instead of accepting the wrong SDP (the
                # same client ID can legitimately reconnect).
                if (
                    isinstance(msg, tuple)
                    and len(msg) >= 3
                    and msg[0] in {"offer_ok", "offer_err"}
                    and msg[1] != request_id
                ):
                    continue
                break
        if (
            isinstance(msg, tuple)
            and len(msg) == 5
            and msg[0] == "offer_ok"
            and msg[1] == request_id
            and msg[2] == client_id
        ):
            return msg[3], msg[4]
        raise RuntimeError(f"video relay offer failed: {msg}")

    def _send(self, msg: object) -> None:
        with self._lock:
            try:
                self._conn.send(msg)
            except (OSError, ValueError):
                pass  # relay already gone

    # -- WebRTCManager interface --------------------------------------------

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a peer connection in the relay; returns ``(sdp, tracks)``."""
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()
        request = loop.run_in_executor(None, self._request_offer, client_id, cancelled)
        try:
            result = await request
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if result is None:  # only reachable if cancellation raced task delivery
            raise asyncio.CancelledError
        return result

    async def set_answer(self, client_id: int, sdp: str) -> None:
        """Forward the headset's SDP answer to the relay."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("answer", client_id, sdp)
        )

    async def close(self, client_id: int) -> None:
        """Close the relay's peer connection for ``client_id``."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("close", client_id)
        )

    async def close_all(self) -> None:
        """Close every peer connection in the relay."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._send, ("close_all",)
        )

    def set_raw_enabled(self, enabled: bool) -> None:
        """Open/close the raw dataset branch in the relay (recording only).

        The raw RGBA branch (VIC convert + shared-memory copy for every camera)
        is the bulk of the relay's CPU. ``collect-data`` keeps it closed during
        the pre-record teleop phase — where nothing reads raw frames — so the
        control loop keeps the spare cores it needs, then opens it while an
        episode is actually recording. No-op if the relay has no raw sources.
        """
        if not self.raw_cameras:
            return
        self._send(("raw_enable", bool(enabled)))

    # -- Lifecycle ------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the relay subprocess and prove camera ownership was released."""
        # Wake an executor thread polling for an offer response before taking
        # its signaling lock. Some callers own the relay outside VRTeleop's
        # context and shut it down just before VRServer cancels signaling tasks.
        self._shutdown_requested.set()
        failures: list[tuple[str, BaseException]] = []
        remaining_readers: dict[str, object] = {}
        for name, reader in self.raw_cameras.items():
            try:
                reader.disconnect()  # type: ignore[attr-defined]
            except BaseException as error:
                failures.append((f"raw reader {name}", error))
                remaining_readers[name] = reader
        self.raw_cameras = remaining_readers
        try:
            with self._lock:
                self._conn.send(None)
        except (OSError, ValueError):
            pass  # child exit is proved independently below

        process_started = getattr(self._proc, "pid", object()) is not None
        process_alive = False
        if process_started:
            try:
                self._proc.join(timeout=5.0)
                process_alive = self._proc.is_alive()
            except BaseException as error:
                failures.append(("initial process reap", error))
                process_alive = True
        if process_alive:
            try:
                self._proc.terminate()
            except BaseException as error:
                failures.append(("process terminate", error))
            try:
                self._proc.join(timeout=2.0)
                process_alive = self._proc.is_alive()
            except BaseException as error:
                failures.append(("post-terminate reap", error))
                process_alive = True
        if process_alive:
            try:
                self._proc.kill()
            except BaseException as error:
                failures.append(("process kill", error))
            try:
                self._proc.join(timeout=2.0)
                process_alive = self._proc.is_alive()
            except BaseException as error:
                failures.append(("post-kill reap", error))
                process_alive = True
        try:
            self._conn.close()
        except BaseException as error:
            failures.append(("parent pipe close", error))

        if process_alive:
            failure = RuntimeError(
                "video relay did not stop after terminate and kill; camera "
                "ownership remains uncertain"
            )
            for label, error in failures:
                failure.add_note(
                    f"additional {label} failure: {type(error).__name__}: {error}"
                )
            raise failure
        if failures:
            failure = RuntimeError("video relay cleanup did not complete safely")
            for label, error in failures:
                failure.add_note(f"{label}: {type(error).__name__}: {error}")
            raise failure from failures[0][1]
