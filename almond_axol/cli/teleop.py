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

from ..utils.jetson import pin_realtime_clocks
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

    # Pin the Jetson clocks now, while a sudo prompt can still reach the tty
    # (mid-session escalation is non-interactive): the CPU governor for the
    # IK loop rate, and — when cameras stream — the NVENC/VIC engines for
    # encode latency.
    pin_realtime_clocks(interactive=True)

    # Ensure the gstreamer WebRTC bindings (PyGObject + webrtcbin) are present
    # so the headset video relay can run; best-effort install if missing.
    if cfg.cameras:
        from ..utils.gst_webrtc_install import ensure_gst_webrtc

        ensure_gst_webrtc()

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


def _overhead_is_stereo(cfg: TeleopCmdConfig) -> bool:
    """Whether the configured overhead camera is a stereo ZED X.

    Auto-detected from the overhead serial's device kind (see
    :func:`almond_axol.zed.stereo_serials`) so operators never flag stereo by
    hand. Best-effort: any enumeration failure treats the overhead as mono.
    """
    serial = cfg.cameras.get("overhead")
    if serial is None:
        return False
    from ..zed import stereo_serials

    return int(serial) in stereo_serials()


def _start_video_relay(cfg: TeleopCmdConfig, overhead_stereo: bool) -> Any | None:
    """Start the out-of-process video relay for the configured cameras.

    The relay subprocess opens the ZED cameras with the Python SDK (exactly
    like data collection), grabs frames on per-camera threads, and encodes
    + sends WebRTC entirely from gstreamer (``webrtcbin``), so video traffic
    can't contend with the teleop control loops (in-process sending
    measurably halves the IK rate). Returns ``None`` when the relay can't be
    used (no cameras, or the gstreamer WebRTC bindings are missing), in
    which case the caller uses the in-process path.
    """
    if not cfg.cameras:
        return None
    try:
        from ..vr.gst_webrtc import gst_webrtc_available
        from ..vr.video_proc import VideoRelayProcess
    except Exception as exc:  # noqa: BLE001 - video extra missing
        _logger.debug("video relay unavailable: %s", exc)
        return None

    if not gst_webrtc_available():
        return None

    resolution = cfg.resolution or "HD1200"
    specs: dict[str, dict[str, Any]] = {}
    for name, serial in cfg.cameras.items():
        spec: dict[str, Any] = {"serial": serial, "resolution": resolution, "fps": 60}
        if name == "overhead" and overhead_stereo:
            spec["stereo"] = True
        specs[name] = spec

    relay = VideoRelayProcess(specs)
    if not relay.has_sources:
        relay.shutdown()
        return None
    return relay


def _connect_zed_cameras(
    cfg: TeleopCmdConfig, overhead_stereo: bool
) -> list[tuple[str, Any]]:
    """Open the local ZED cameras selected by ``cfg`` → ``(slot, camera)`` pairs.

    Opens each camera through the ZED Python SDK (:class:`ZedCamera`, or
    :class:`ZedStereoCamera` for a stereo overhead). Used for the in-process
    fallback when the relay subprocess can't run; the frames are encoded +
    sent by an in-process :class:`GstWebRTCManager`. Slots whose camera is
    absent are skipped (best-effort preview). Returns an empty list when no
    cameras are configured or no backend is available. Runs synchronously
    (blocks on camera startup), so call it off the event loop.
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
        if name == "overhead" and overhead_stereo:
            stereo = _connect(name, serial, stereo=True)
            if stereo is None:
                continue
            cameras.append(("overhead_left", stereo.left_view))
            cameras.append(("overhead_right", stereo.right_view))
            continue

        cam = _connect(name, serial)
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

    @property
    def width(self) -> int:
        return int(self._cam.width or 0)

    @property
    def height(self) -> int:
        return int(self._cam.height or 0)

    @property
    def fps(self) -> int:
        return int(self._cam.fps or 30)

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

    # SDK cameras are wrapped as frame-driven sources for the in-process
    # GstWebRTCManager (NVENC encode + webrtcbin send).
    sources = {name: _ZedFrameSource(cam) for name, cam in cameras}
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
    async with VRTeleop(
        robot,
        config=cfg.teleop,
        kinematics_config=cfg.kinematics,
        vr_server_config=cfg.vr_server,
    ) as teleop:
        # Prefer the out-of-process relay (gst-native cameras + WebRTC in
        # a subprocess, isolated from the control loops); fall back to
        # in-process sources when it isn't applicable.
        overhead_stereo = await asyncio.to_thread(_overhead_is_stereo, cfg)
        relay = await asyncio.to_thread(_start_video_relay, cfg, overhead_stereo)
        cameras: list[tuple[str, Any]] = []
        if relay is not None:
            teleop.set_video_manager(relay)
            _logger.info(
                "teleop: streaming cameras to headset (isolated process): %s",
                ", ".join(relay.sources),
            )
        else:
            cameras = await asyncio.to_thread(
                _connect_zed_cameras, cfg, overhead_stereo
            )
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
