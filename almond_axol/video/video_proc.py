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
    dataset_fps: int,
    capture_fps: int,
    pts_perf_offset_s: float,
) -> dict:
    """Describe the encoded-H.264 shared-memory transport for one dataset source.

    The relay encodes the dataset stream on the GPU and writes GDP-wrapped,
    AU-aligned H.264 to ``socket_path`` via ``shmsink``; the recorder attaches an
    :class:`~almond_axol.video.shm_frames.EncodedAuReader` and muxes it. GDP
    preserves each GStreamer buffer's nanosecond PTS. ``pts_perf_offset_s`` is
    the sender pipeline's fixed running-time-to-``perf_counter`` mapping, letting
    the recorder convert that PTS to the same system-wide clock as joint/action
    snapshots: ``capture_perf = pts / 1e9 + pts_perf_offset_s``.
    """
    return {
        "transport": "gstshm-h264",
        "socket_path": socket_path,
        "width": width,
        "height": height,
        "fps": dataset_fps,
        "capture_fps": capture_fps,
        "pts_perf_offset_s": pts_perf_offset_s,
    }


def _pyshm_meta(shm_name: str, width: int, height: int, fps: int) -> dict:
    return {
        "transport": "pyshm",
        "shm_name": shm_name,
        "width": width,
        "height": height,
        "fps": fps,
    }


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

    Like :func:`_open_gst_camera`, but additionally exports each source's raw
    frames to the recorder process for the dataset. Two transports:

    * **gstshm** (``socket_dir`` set — gst's ``shm`` + GDP plugins are available):
      the dataset branch GPU-encodes H.264, wraps it with ``gdppay``, and ends in
      a native ``shmsink`` (pure C), so the relay does **zero** Python per frame
      and its interpreter stays free for the WebRTC send. The recorder reads via
      ``shmsrc ! gdpdepay`` (:class:`EncodedAuReader`); GDP retains exposure PTS.
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
    if stereo and not zed_stereo_gst_available(require_sensor_timestamps=True):
        return None
    if not stereo and not zed_gst_available(require_sensor_timestamps=True):
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
    dataset_fps = int(spec.get("dataset_fps", spec.get("fps", 60)))
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
                    dataset_fps=dataset_fps,
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
                pts_perf_offset_s = cam.pts_perf_offset_s
                raw_meta = {
                    src: _gsth264_meta(
                        socks[side],
                        raw_w,
                        raw_h,
                        dataset_fps,
                        fps,
                        pts_perf_offset_s,
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
                    dataset_fps=dataset_fps,
                )
                cam.connect()
                meta = {
                    name: _gsth264_meta(
                        sock,
                        raw_w,
                        raw_h,
                        dataset_fps,
                        fps,
                        cam.pts_perf_offset_s,
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
    # ZedCamera/eye, which WebRTCManager samples on its fixed-rate NVENC track.
    owned: list[object] = []
    sources: dict[str, object] = {}
    writers: list[object] = []
    raw_meta: dict[str, dict] = {}
    # Prefer the gst-native GDP/shmsink transport for dataset video: it encodes
    # and exports every timestamped AU in C, so the relay does zero Python per
    # dataset frame and the WebRTC send keeps the GIL it needs. Falls back to the
    # in-relay Python raw-frame copy (RawFrameWriter) when any required gst
    # element is absent. A per-relay-PID dir holds one socket per source.
    socket_dir: str | None = None
    if want_raw:
        from .gst_zed import _element_available

        # GDP is what preserves caps and sensor-exposure PTS across shm. All four
        # elements are required as one transport; silently falling back to bare
        # shm would recreate the receive-time synchronization regression.
        if all(
            _element_available(element)
            for element in ("shmsink", "shmsrc", "gdppay", "gdpdepay")
        ):
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

    def _set_raw_enabled(enabled: bool) -> None:
        """Gate every dataset branch, coordinating encoded GOP phase."""
        if enabled:
            errors: list[str] = []
            for cam in owned:
                if not hasattr(cam, "set_raw_enabled"):
                    continue
                try:
                    cam.set_raw_enabled(True)
                except Exception as exc:  # noqa: BLE001 - report all branches
                    errors.append(f"{cam}: {exc}")
            if errors:
                raise RuntimeError("; ".join(errors))
            return

        # Two phases are intentional: install the IDR probes on every physical
        # camera before waiting on any one of them. Sequential arm+wait would
        # stop cameras at different GOP phases and their episode-opening IDRs
        # could be hundreds of milliseconds apart.
        armed: list[object] = []
        errors = []
        for cam in owned:
            if hasattr(cam, "begin_raw_disable") and hasattr(cam, "finish_raw_disable"):
                try:
                    cam.begin_raw_disable()
                    armed.append(cam)
                except Exception as exc:  # noqa: BLE001 - finish other cameras
                    errors.append(f"{cam}: {exc}")
            elif hasattr(cam, "set_raw_enabled"):
                try:
                    cam.set_raw_enabled(False)
                except Exception as exc:  # noqa: BLE001 - finish other cameras
                    errors.append(f"{cam}: {exc}")
        # Matches gst_zed's gate bound. At the supported 1 fps minimum a valid
        # next encoder output can arrive almost exactly one second later, so a
        # 1.0 s deadline spuriously fails under any scheduling load.
        deadline = time.perf_counter() + 2.0
        for cam in armed:
            try:
                cam.finish_raw_disable(deadline)
            except Exception as exc:  # noqa: BLE001 - collect every failure
                errors.append(f"{cam}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

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
                    _, client_id = msg
                    if manager is None:
                        conn.send(("offer_err", client_id, "no cameras"))
                        continue
                    try:
                        sdp, tracks = await manager.create_offer(client_id)
                        conn.send(("offer_ok", client_id, sdp, tracks))
                    except Exception as exc:  # noqa: BLE001 - report upstream
                        _logger.error("video relay: offer failed: %s", exc)
                        conn.send(("offer_err", client_id, str(exc)))
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
                    try:
                        # Waiting for the cameras' next natural IDRs can take a
                        # GOP. Keep that wait off aiortc's event loop so RTP,
                        # RTCP, and signaling continue uninterrupted.
                        await loop.run_in_executor(
                            None, _set_raw_enabled, bool(enabled)
                        )
                    except Exception as exc:  # noqa: BLE001 - report to parent
                        _logger.error("video relay: dataset gate failed: %s", exc)
                        conn.send(("raw_enable_err", bool(enabled), str(exc)))
                    else:
                        conn.send(("raw_enable_ok", bool(enabled)))
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
                "resolution": str, "fps": int, "stereo": bool}}``. ``fps``
                is the physical capture/data rate; headset encoding is fixed
                independently at 30 fps.
            want_raw: Also publish each camera's dataset feed to shared memory
                (encoded GDP/H.264 when available, raw RGB as fallback).
                Successfully exported sources appear in :attr:`raw_cameras` as
                lightweight camera proxies.
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
        self._proc.start()
        child_conn.close()
        self._lock = threading.Lock()

        self.sources: list[str] = []
        self.raw_cameras: dict[str, object] = {}
        # ``{source: meta}`` describing each raw source's transport (gstshm socket
        # + caps, or pyshm block name) and dims — exposed (with :attr:`raw_cond`)
        # so the recorder subprocess can attach its own consumer per source.
        self.raw_meta: dict[str, dict] = {}
        try:
            deadline = time.perf_counter() + _READY_TIMEOUT_S
            while True:
                if self._conn.poll(0.1):
                    try:
                        msg = self._conn.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError("video relay exited during startup") from exc
                    if not (isinstance(msg, tuple) and msg[0] == "ready"):
                        raise RuntimeError(
                            f"video relay sent unexpected ready message: {msg!r}"
                        )
                    self.sources = list(msg[1])
                    raw_meta = msg[2] if len(msg) > 2 else {}
                    self.raw_meta = dict(raw_meta)
                    self._attach_raw_readers(raw_meta)
                    break
                if not self._proc.is_alive():
                    raise RuntimeError(
                        "video relay exited during startup "
                        f"(exit code {self._proc.exitcode})"
                    )
                if time.perf_counter() >= deadline:
                    raise RuntimeError("video relay did not become ready in time")
        except BaseException:
            self.shutdown()
            raise
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

    def _request_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        with self._lock:
            self._conn.send(("offer", client_id))
            # Replies are strictly ordered on the pipe; the only inbound
            # messages are responses to "offer" requests.
            if not self._conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError("video relay did not answer the offer request")
            msg = self._conn.recv()
        if msg[0] == "offer_ok" and msg[1] == client_id:
            return msg[2], msg[3]
        raise RuntimeError(f"video relay offer failed: {msg}")

    def _send(self, msg: object) -> None:
        with self._lock:
            try:
                self._conn.send(msg)
            except (OSError, ValueError):
                pass  # relay already gone

    def _request_raw_enabled(self, enabled: bool) -> None:
        """Synchronously gate dataset branches and surface alignment errors."""
        with self._lock:
            try:
                self._conn.send(("raw_enable", enabled))
            except (OSError, ValueError) as exc:
                raise RuntimeError("video relay is unavailable") from exc
            if not self._conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError("video relay did not acknowledge dataset gate")
            try:
                msg = self._conn.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError("video relay closed during dataset gate") from exc
        if msg[:2] == ("raw_enable_ok", enabled):
            return
        if len(msg) >= 3 and msg[:2] == ("raw_enable_err", enabled):
            raise RuntimeError(f"video relay dataset gate failed: {msg[2]}")
        raise RuntimeError(f"unexpected video relay dataset-gate response: {msg}")

    # -- WebRTCManager interface --------------------------------------------

    async def create_offer(self, client_id: int) -> tuple[str, dict[str, str]]:
        """Build a peer connection in the relay; returns ``(sdp, tracks)``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._request_offer, client_id)

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

        The dataset branch (GPU encode + GDP/shared-memory transport, or raw
        fallback) is unnecessary between episodes. ``collect-data`` keeps it
        closed during pre-record teleop and opens it for each episode. No-op if
        the relay has no dataset sources.
        """
        if not self.raw_cameras:
            return
        self._request_raw_enabled(bool(enabled))

    # -- Lifecycle ------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop the relay subprocess (cameras and peer connections included)."""
        for reader in self.raw_cameras.values():
            try:
                reader.disconnect()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self.raw_cameras = {}
        try:
            with self._lock:
                self._conn.send(None)
        except (OSError, ValueError):
            pass
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
