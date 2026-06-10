"""
axol teleop [--sim]

Run a VR teleoperation session. Drives the real Axol robot by default;
pass ``--sim`` to use the browser visualizer instead. Every robot config
field is reachable from the CLI (draccus-style) or from a JSON/YAML file:

    axol teleop                                       # real robot
    axol teleop --sim                                 # browser visualizer
    axol teleop --axol.left_stiffness 0.8
    axol teleop --axol.left.elbow.kp 60 --axol.right.gripper.torque_limit 0.7
    axol teleop --teleop.position_multiplier 2.0      # scale hand motion 2x
    axol teleop --left_channel null                   # disable the left arm
    axol teleop --config_path my_teleop.json          # whole-config file
"""

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Any

from ..utils.jetson import pin_engine_clocks
from .config import TeleopCmdConfig, parse

if TYPE_CHECKING:
    from ..teleop import VRTeleop

_logger = logging.getLogger(__name__)


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def _normalize_sim_flag(argv: list[str]) -> list[str]:
    """Let ``--sim`` be passed as a bare flag.

    draccus parses bool fields as value-taking arguments (``--sim true``),
    so rewrite a standalone ``--sim`` (one that's followed by another flag
    or nothing) into ``--sim true``. An explicit ``--sim true`` / ``--sim
    false`` / ``--sim=...`` is left untouched.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--sim":
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is None or nxt.startswith("-"):
                out.extend(("--sim", "true"))
                i += 1
                continue
        out.append(tok)
        i += 1
    return out


def main(argv: list[str]) -> None:
    """Parse the CLI config and run a VR teleop session."""
    cfg = parse(TeleopCmdConfig, _normalize_sim_flag(argv))
    # force=True: a dependency imported before this point may install a root
    # handler (leaving the level at WARNING), which would make this a no-op
    # and silently drop log_say() / INFO status lines.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)

    if cfg.cameras:
        # Cameras will stream to the headset through the Jetson hardware
        # encoder; pin its engine clocks now, while a sudo prompt can still
        # reach the tty (mid-session escalation is non-interactive).
        pin_engine_clocks(interactive=True)

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


def _start_video_relay(cfg: TeleopCmdConfig) -> Any | None:
    """Start the out-of-process video relay for the configured cameras.

    The relay subprocess owns the gst-native camera pipelines and all
    WebRTC sending, so video traffic can't contend with the teleop control
    loops for the GIL (in-process sending measurably halves the IK rate).
    Returns ``None`` when not applicable — no cameras, stereo overhead
    (SDK-only), or plugins missing — in which case the caller uses the
    in-process path.
    """
    if not cfg.cameras:
        return None
    if cfg.overhead_stereo and "overhead" in cfg.cameras:
        return None  # stereo needs the SDK; mixed relay+SDK isn't supported
    try:
        from ..vr.gst_zed import zed_gst_available
        from ..vr.video_proc import VideoRelayProcess
    except Exception as exc:  # noqa: BLE001 - video extra missing
        _logger.debug("video relay unavailable: %s", exc)
        return None
    if not zed_gst_available():
        return None
    relay = VideoRelayProcess(
        {
            name: {"serial": serial, "resolution": cfg.resolution or "HD1200"}
            for name, serial in cfg.cameras.items()
        }
    )
    if not relay.has_sources:
        relay.shutdown()
        return None
    return relay


def _open_gst_stream(name: str, serial: int, resolution: str | None) -> Any | None:
    """Open a camera through the GPU-resident gst-native pipeline, or None.

    The zed-gstreamer path keeps frames on the GPU end to end (grab ->
    NVENC, ~5 ms) and hands Python pre-encoded H.264, instead of the SDK
    grab -> numpy -> pipe -> NVENC relay (~26 ms). Returns ``None`` when
    the plugins are missing or the camera doesn't start, in which case the
    caller falls back to the SDK path.
    """
    try:
        from ..vr.gst_zed import ZedXOneGstStream, zed_gst_available
    except Exception as exc:  # noqa: BLE001 - vr extras missing
        _logger.debug("gst-native camera path unavailable: %s", exc)
        return None
    if not zed_gst_available():
        return None
    try:
        stream = ZedXOneGstStream(serial, resolution=resolution or "HD1200")
    except Exception as exc:  # noqa: BLE001 - bad resolution / spawn failure
        _logger.info("teleop: %s gst pipeline failed to start (%s)", name, exc)
        return None
    if not stream.wait_ready():
        _logger.info(
            "teleop: %s gst pipeline produced no frames; trying the SDK path", name
        )
        stream.close()
        return None
    return stream


def _connect_zed_cameras(cfg: TeleopCmdConfig) -> list[tuple[str, Any]]:
    """Open the local ZED cameras selected by ``cfg`` → ``(slot, camera)`` pairs.

    Mono cameras prefer the gst-native pre-encoded pipeline (see
    :func:`_open_gst_stream`); stereo and fallback cases open a
    :class:`ZedCamera` through the SDK. Slots whose camera is absent are
    skipped (best-effort preview). Returns an empty list when no cameras
    are configured or no backend is available. Runs synchronously (blocks
    on camera startup), so call it off the event loop.
    """
    if not cfg.cameras:
        return []

    # Resolution validation happens against the SDK's name table when the
    # SDK is importable; the gst path validates names itself.
    sdk_exc: Exception | None = None
    try:
        from ..lerobot.camera.camera_zed import ZedCamera, ZedStereoCamera
        from ..lerobot.camera.configuration_zed import (
            ZED_RESOLUTION_DIMS,
            ZedCameraConfig,
        )
    except Exception as exc:  # noqa: BLE001 - missing pyzed/SDK → gst only
        sdk_exc = exc

    # Capture at the requested resolution; without one, width/height of None
    # adopt each camera's SDK default (HD1200 on GMSL) on connect.
    width: int | None = None
    height: int | None = None
    if cfg.resolution and sdk_exc is None:
        dims = ZED_RESOLUTION_DIMS.get(cfg.resolution)
        if dims is None:
            _logger.warning(
                "unknown ZED resolution %r (expected one of %s); "
                "using the camera default",
                cfg.resolution,
                ", ".join(ZED_RESOLUTION_DIMS),
            )
        else:
            width, height = dims

    def _connect(name: str, serial: int, **kwargs: Any) -> Any | None:
        """Open one camera via the SDK, preferring 60 fps capture.

        At the SDK's default the GMSL cameras capture HD1200 at 30 fps,
        which adds up to a full 33 ms frame interval of staleness to the
        headset feed; 60 fps halves that. Cameras that don't support 60 at
        the requested resolution fall back to their default rate.
        """
        if sdk_exc is not None:
            _logger.warning("teleop: %s camera unavailable (%s)", name, sdk_exc)
            return None
        cls = ZedStereoCamera if kwargs.get("stereo") else ZedCamera
        for fps in (60, None):
            cam = cls(
                ZedCameraConfig(
                    serial=serial, fps=fps, width=width, height=height, **kwargs
                )
            )
            try:
                cam.connect(warmup=False)
                return cam
            except RuntimeError as exc:
                # Live-parameter mismatch (e.g. 60 fps unsupported) → retry
                # at the camera default.
                _logger.info("teleop: %s camera rejected %s fps (%s)", name, fps, exc)
            except Exception as exc:  # noqa: BLE001 - camera absent → skip it
                _logger.info("teleop: %s camera not available (%s)", name, exc)
                return None
        return None

    cameras: list[tuple[str, Any]] = []
    for name, serial in cfg.cameras.items():
        # A stereo overhead carries both eyes on one grab; expose them as
        # overhead_left / overhead_right so the headset can render per-lens.
        # (The gst-native path is mono-only, so stereo stays on the SDK.)
        if name == "overhead" and cfg.overhead_stereo:
            stereo = _connect(name, serial, stereo=True)
            if stereo is None:
                continue
            cameras.append(("overhead_left", stereo.left_view))
            cameras.append(("overhead_right", stereo.right_view))
            continue

        cam = _open_gst_stream(name, serial, cfg.resolution) or _connect(name, serial)
        if cam is None:
            continue
        cameras.append((name, cam))
    if not cameras and sdk_exc is not None:
        _logger.warning("ZED camera preview unavailable: %s", sdk_exc)
    return cameras


class _ZedFrameSource:
    """Frame-driven relay source for a connected ZED camera (or stereo eye).

    Exposes both the pull interface (``__call__``, compatibility) and
    ``wait_next``, which the WebRTC track prefers: it blocks until the camera
    produces a frame newer than the last one sent, so every relayed frame is
    encoded the instant it's captured instead of waiting to be sampled by a
    fixed-rate timer.
    """

    def __init__(self, cam: Any) -> None:
        self._cam = cam

    def __call__(self) -> Any:
        try:
            return self._cam.read_latest(max_age_ms=1000)
        except Exception:  # noqa: BLE001 - stale/dropped frame → black feed
            return None

    def wait_next(self, after_ts: float | None, timeout_ms: float) -> Any:
        target = after_ts + 1e-6 if after_ts is not None else 0.0
        try:
            frame, cap_ts, _recv_ts = self._cam.read_at_or_after(
                target, timeout_ms=timeout_ms
            )
            return frame, cap_ts
        except Exception:  # noqa: BLE001 - timeout/stall → keepalive in track
            return None


def _register_zed_video(teleop: "VRTeleop", cameras: list[tuple[str, Any]]) -> None:
    """Register connected ZED cameras as WebRTC sources for the headset."""
    if not cameras:
        return

    # gst-native streams (pre-encoded; expose subscribe()) go straight to
    # the WebRTC manager; SDK cameras are wrapped as frame-driven sources.
    sources = {
        name: cam if hasattr(cam, "subscribe") else _ZedFrameSource(cam)
        for name, cam in cameras
    }
    try:
        teleop.set_video_sources(sources)
        _logger.info("teleop: streaming cameras to headset: %s", ", ".join(sources))
    except Exception as exc:  # noqa: BLE001
        _logger.warning("failed to enable camera video: %s", exc)


async def _run(cfg: TeleopCmdConfig) -> None:
    from ..robot import Axol, Sim
    from ..teleop import VRTeleop

    if cfg.sim:
        robot = Sim()
    else:
        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel,
            right_channel=cfg.right_channel,
        )
    async with VRTeleop(robot, config=cfg.teleop) as teleop:
        # Prefer the out-of-process relay (gst-native cameras + WebRTC in
        # a subprocess, isolated from the control loops); fall back to
        # in-process sources when it isn't applicable.
        relay = await asyncio.to_thread(_start_video_relay, cfg)
        cameras: list[tuple[str, Any]] = []
        if relay is not None:
            teleop.set_video_manager(relay)
            _logger.info(
                "teleop: streaming cameras to headset (isolated process): %s",
                ", ".join(relay.sources),
            )
        else:
            cameras = await asyncio.to_thread(_connect_zed_cameras, cfg)
            _register_zed_video(teleop, cameras)
        try:
            await teleop.run()
        finally:
            if relay is not None:
                await asyncio.to_thread(relay.shutdown)
            for _name, cam in cameras:
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
