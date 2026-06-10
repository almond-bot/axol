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

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


def _connect_zed_cameras(cfg: TeleopCmdConfig) -> list[tuple[str, Any]]:
    """Open the local ZED cameras selected by ``cfg`` → ``(slot, camera)`` pairs.

    One :class:`ZedCamera` per configured slot that actually opens; slots
    whose camera is absent are skipped (best-effort preview). Returns an
    empty list when no cameras are configured or the SDK is unavailable.
    Runs synchronously (blocks on the SDK), so call it off the event loop.
    """
    if not cfg.cameras:
        return []
    try:
        from ..lerobot.camera.camera_zed import ZedCamera, ZedStereoCamera
        from ..lerobot.camera.configuration_zed import ZedCameraConfig
    except Exception as exc:  # noqa: BLE001 - missing pyzed/SDK → no preview
        _logger.warning("ZED camera preview unavailable: %s", exc)
        return []

    cameras: list[tuple[str, Any]] = []
    for name, serial in cfg.cameras.items():
        # Teleop only relays frames to the headset, so adapt to whatever the
        # camera captures (fps/width/height of None skip the config-vs-camera
        # mismatch check used for dataset features).
        def _config(**kwargs: Any) -> "ZedCameraConfig":
            return ZedCameraConfig(
                serial=serial, fps=None, width=None, height=None, **kwargs
            )

        # A stereo overhead carries both eyes on one grab; expose them as
        # overhead_left / overhead_right so the headset can render per-lens.
        if name == "overhead" and cfg.overhead_stereo:
            stereo = ZedStereoCamera(_config(stereo=True))
            try:
                stereo.connect(warmup=False)
            except Exception as exc:  # noqa: BLE001 - camera absent → skip it
                _logger.info("teleop: overhead stereo camera not available (%s)", exc)
                continue
            cameras.append(("overhead_left", stereo.left_view))
            cameras.append(("overhead_right", stereo.right_view))
            continue

        cam = ZedCamera(_config())
        try:
            cam.connect(warmup=False)
        except Exception as exc:  # noqa: BLE001 - camera absent → skip it
            _logger.info("teleop: %s camera not available (%s)", name, exc)
            continue
        cameras.append((name, cam))
    return cameras


def _register_zed_video(teleop: "VRTeleop", cameras: list[tuple[str, Any]]) -> None:
    """Register connected ZED cameras as WebRTC sources for the headset."""
    if not cameras:
        return

    def _make_source(cam: Any) -> Any:
        def _source() -> Any:
            try:
                return cam.read_latest(max_age_ms=1000)
            except Exception:  # noqa: BLE001 - stale/dropped frame → black feed
                return None

        return _source

    sources = {name: _make_source(cam) for name, cam in cameras}
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
        cameras = await asyncio.to_thread(_connect_zed_cameras, cfg)
        _register_zed_video(teleop, cameras)
        try:
            await teleop.run()
        finally:
            for _name, cam in cameras:
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
