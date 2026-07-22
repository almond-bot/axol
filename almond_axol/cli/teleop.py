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
    axol teleop --cart.enabled true                   # powered cart (base+lift)
    axol teleop --config_path my_teleop.json          # whole-config file
"""

import asyncio
import logging
import socket
from typing import TYPE_CHECKING, Any

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

    # System setup (Jetson clock pinning, the GStreamer NVENC stack) is handled
    # by the host installer + its boot service, not here — see
    # `axol jetson.setup` / `axol gst.install`. This entry point just runs.

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


def _stereo_serials_for(cfg: TeleopCmdConfig) -> set[int]:
    """Serials among the configured cameras that are stereo ZED X units.

    Auto-detected from each serial's device kind (see
    :func:`almond_axol.zed.stereo_serials`) so operators never flag stereo by
    hand — any camera slot (overhead or a wrist) opens through the stereo flow
    when its serial is a stereo ZED X. Best-effort: any enumeration failure
    treats every configured camera as mono.
    """
    if not cfg.cameras:
        return set()
    from ..zed import stereo_serials

    detected = stereo_serials()
    return {int(s) for s in cfg.cameras.values() if int(s) in detected}


def _stereo_eyes_for(name: str) -> tuple[list[str], bool]:
    """``(eyes, suffix)`` for a stereo camera slot under the wrist policy.

    Only the head (``overhead``) camera streams both eyes — packed side-by-side
    into a single ``{name}_sbs`` track on the gst pipeline (see ``stream_sbs``
    in ``_start_video_relay``), or per-lens ``{name}_left`` / ``{name}_right``
    tracks on the SDK fallback. Every other slot (the wrists) streams just its
    left eye under the plain ``{name}``, so a stereo wrist encodes and ships
    exactly one feed, costing the same as a mono one. This is what keeps the
    control loop's spare CPU regardless of wrist camera type.
    """
    if name == "overhead":
        return ["left", "right"], True
    return ["left"], False


def _stream_eyes_for(cfg: TeleopCmdConfig, name: str) -> tuple[list[str], bool]:
    """``(eyes, suffix)`` for a stereo slot's headset stream.

    Honours an explicit per-slot ``--camera_eyes`` override (``both`` streams
    both eyes per-lens, suffixed; ``left`` / ``right`` streams that single eye
    under the plain name), falling back to the default head/wrist policy
    (:func:`_stereo_eyes_for`) for slots the operator left unset.
    """
    eyes = (cfg.camera_eyes or {}).get(name)
    if eyes == "both":
        return ["left", "right"], True
    if eyes in ("left", "right"):
        return [eyes], False
    return _stereo_eyes_for(name)


def _start_video_relay(cfg: TeleopCmdConfig, stereo_set: set[int]) -> Any | None:
    """Start the out-of-process video relay for the configured cameras.

    The relay subprocess opens the ZED cameras through the GPU-resident gst
    pipeline (``gst_zed``: ``zedxonesrc`` grab + NVENC encode on the GPU),
    falling back to the ZED SDK grab + in-Python NVENC path when the gst stack
    is unavailable, and ships WebRTC with aiortc — so video traffic can't
    contend with the teleop control loops (in-process sending measurably halves
    the IK rate). Returns ``None`` when the relay can't be used (no cameras, or
    aiortc isn't installed), in which case the caller uses the in-process path.
    """
    if not cfg.cameras:
        return None
    try:
        from ..video.video import webrtc_available
        from ..video.video_proc import VideoRelayProcess
    except Exception as exc:  # noqa: BLE001 - aiortc missing
        _logger.debug("video relay unavailable: %s", exc)
        return None

    if not webrtc_available():
        return None

    resolution = cfg.resolution or "HD1200"
    specs: dict[str, dict[str, Any]] = {}
    for name, serial in cfg.cameras.items():
        spec: dict[str, Any] = {"serial": serial, "resolution": resolution, "fps": 60}
        if int(serial) in stereo_set:
            spec["stereo"] = True
            spec["eyes"], spec["eye_suffix"] = _stream_eyes_for(cfg, name)
            # Both eyes ship packed side-by-side in one track (one decoder
            # session on the headset); the per-eye keys remain as the SDK
            # fallback, which can't pack.
            if len(spec["eyes"]) == 2:
                spec["stream_sbs"] = True
        specs[name] = spec

    relay = VideoRelayProcess(specs)
    if not relay.has_sources:
        relay.shutdown()
        return None
    return relay


def _connect_zed_cameras(
    cfg: TeleopCmdConfig, stereo_set: set[int]
) -> list[tuple[str, Any]]:
    """Open the local ZED cameras selected by ``cfg`` → ``(slot, camera)`` pairs.

    Opens each camera through the ZED Python SDK (:class:`ZedCamera`, or
    :class:`ZedStereoCamera` for any camera whose serial is a stereo ZED X).
    Used for the in-process fallback when the relay subprocess can't run; the
    frames are encoded + sent by an in-process aiortc :class:`WebRTCManager`.
    Slots whose camera is absent are skipped (best-effort preview). Returns an
    empty list when no cameras are configured or no backend is available. Runs
    synchronously (blocks on camera startup), so call it off the event loop.
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
        # A stereo camera carries both eyes on one grab. The head camera exposes
        # both per-lens ({name}_left / {name}_right); a wrist exposes only its
        # left eye under the plain {name} (see _stereo_eyes_for) so it streams
        # one feed like a mono camera.
        if int(serial) in stereo_set:
            stereo = _connect(name, serial, stereo=True)
            if stereo is None:
                continue
            eyes, suffix = _stream_eyes_for(cfg, name)
            for side in eyes:
                view = stereo.left_view if side == "left" else stereo.right_view
                cameras.append((f"{name}_{side}" if suffix else name, view))
            continue

        cam = _connect(name, serial)
        if cam is None:
            continue
        cameras.append((name, cam))
    if not cameras and sdk_exc is not None:
        _logger.warning("ZED camera preview unavailable: %s", sdk_exc)
    return cameras


def _register_zed_video(teleop: "VRTeleop", cameras: list[tuple[str, Any]]) -> None:
    """Register connected ZED cameras as WebRTC sources for the headset.

    The bare ``ZedCamera`` / stereo eyes are registered directly; the in-process
    aiortc relay adapts each one to a frame-driven source (NVENC encode + aiortc
    RTP send) — see :func:`almond_axol.video.video._track_for_source`.
    """
    if not cameras:
        return

    sources = dict(cameras)
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
    # Powered-cart robots (--cart.enabled true) get the base + lift driven by
    # the headset thumbsticks; VRTeleop owns the cart's lifecycle. Skipped in
    # sim — there's no cart hardware model in the visualizer.
    cart = None
    if cfg.cart.enabled and not cfg.sim:
        from ..robot.cart import Cart

        cart = Cart(cfg.cart)
    async with VRTeleop(
        robot,
        config=cfg.teleop,
        kinematics_config=cfg.kinematics,
        vr_server_config=cfg.vr_server,
        cart=cart,
    ) as teleop:
        # Prefer the out-of-process relay (gst-native cameras + WebRTC in
        # a subprocess, isolated from the control loops); fall back to
        # in-process sources when it isn't applicable.
        stereo_set = await asyncio.to_thread(_stereo_serials_for, cfg)
        relay = await asyncio.to_thread(_start_video_relay, cfg, stereo_set)
        cameras: list[tuple[str, Any]] = []
        if relay is not None:
            teleop.set_video_manager(relay)
            _logger.info(
                "teleop: streaming cameras to headset (isolated process): %s",
                ", ".join(relay.sources),
            )
        else:
            cameras = await asyncio.to_thread(_connect_zed_cameras, cfg, stereo_set)
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
