"""
Teleoperate Axol from one or two iPhones running Marionette.

Connects to a Marionette gateway as its consumer. Each phone drives the arm
it selects while its hold-to-move pad is held, closes and opens that arm's
gripper, and drives the Jelly base and lift from its Drive screen. The
control panel's Marionette page shows the robot and the cameras, and drives
the base and lift from the keyboard. With ``--sim`` nothing touches the
hardware: a simulated Axol on Jelly moves instead, with rendered camera
views.

Start the gateway first, with the Axol phone screen: in the Marionette
repository run ``npm run dev:axol``. The session is started and stopped by
the control panel (``axol serve``), which owns the page's connection to it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ..constants import CAN_LEFT, CAN_RIGHT
from ..kinematics.config import KinematicsConfig
from ..robot.config import AxolConfig
from .config import LogLevel

_logger = logging.getLogger(__name__)


@dataclass
class MarionetteCmdConfig:
    """Config for the ``marionette`` operation.

    Arm gains and channels come from the shared robot settings, the same as
    jogging; the lift and base use the shared ``jelly.*`` settings.
    """

    axol: AxolConfig = field(default_factory=AxolConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    sim: bool = False
    """Drive a simulated Axol instead of the hardware."""
    gateway: str = "localhost:8787"
    """The Marionette gateway: an IP or host name, host:port, or a ws:// URL."""
    max_speed: float = 0.5
    """Fastest a gripper follows its phone (m/s)."""
    max_turn_rate: float = 3.0
    """Fastest a gripper turns to follow its phone (rad/s)."""
    joint_speed: float = 2.0
    """Fastest any arm joint turns while following a phone (rad/s)."""
    rotation_scale: float = 1.0
    """Gripper rotation per phone rotation (1 turns the gripper as far as the phone)."""
    pose_cutoff: float = 4.0
    """Smoothing of the phone's motion, as a low-pass cutoff in Hz."""
    camera_fps: float = 12.0
    """Camera frames per second sent to the phones and the page."""
    camera_width: int = 640
    """Width in pixels of the camera frames sent to the phones and the page."""
    camera_quality: int = 60
    """JPEG quality (10 to 95) of the camera frames."""
    cameras: dict[str, int] = field(default_factory=dict)
    """Camera serials by slot (``overhead``, ``left_arm``, ``right_arm``). The
    control panel fills this in from Settings."""
    camera_eyes: dict[str, str] = field(default_factory=dict)
    """Set by the control panel; Marionette sends each stereo camera's left eye."""
    resolution: str | None = None
    """Camera capture resolution (SVGA, HD1080 or HD1200)."""
    rate_hz: float = 125.0
    """Control-loop rate."""
    kd: float = 0.25
    """Gravity compensation's hand-guiding damping. Shared with gravity
    compensation's settings; Marionette never hand-guides."""
    log_level: LogLevel = "INFO"


def build_session(cfg: MarionetteCmdConfig) -> Any:
    """The :class:`~almond_axol.marionette.session.MarionetteSession` this
    config describes."""
    from ..constants import urdf_path
    from ..marionette.cameras import CameraHub, SyntheticCameraSource, ZedCameraSource
    from ..marionette.session import MarionetteSession
    from ..marionette.teleop import TeleopLimits
    from ..settings import resolve_robot_model
    from .jog import build_backend, planner_factory

    planner = planner_factory(cfg.kinematics)
    backend = build_backend(
        cfg, left_channel=cfg.left_channel, right_channel=cfg.right_channel
    )
    session_ref: list[Any] = []
    if cfg.sim:
        from ..marionette.sim_cameras import SimCameraSource

        model = resolve_robot_model(
            cfg.kinematics.robot_model, mount=cfg.kinematics.mount
        )
        gripper = cfg.kinematics.gripper or cfg.axol.gripper or "parallel"
        source: Any = _FallbackSource(
            SimCameraSource(
                str(urdf_path(model, gripper=gripper)),
                lambda: session_ref[0].sim_camera_state(),
            ),
            SyntheticCameraSource(),
        )
    elif cfg.cameras:
        source = ZedCameraSource(
            cfg.cameras, resolution=cfg.resolution, fps=cfg.camera_fps
        )
    else:
        source = None
    hub = CameraHub(
        source,
        fps=cfg.camera_fps,
        width=cfg.camera_width,
        quality=cfg.camera_quality,
        error=None
        if source is not None
        else "No cameras are assigned. Assign them in Settings, Cameras.",
    )
    session = MarionetteSession(
        backend,
        planner,
        gateway=cfg.gateway,
        cameras=hub,
        rate=cfg.rate_hz,
        limits=TeleopLimits(
            max_speed=cfg.max_speed,
            max_turn_rate=cfg.max_turn_rate,
            joint_speed=cfg.joint_speed,
            rotation_scale=cfg.rotation_scale,
            pose_cutoff=cfg.pose_cutoff,
        ),
    )
    session_ref.append(session)
    return session


class _FallbackSource:
    """The first source that starts; the second when it can't."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        self._sources = (primary, fallback)
        self._active: Any = primary
        self.error: str | None = None

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def available(self) -> tuple[str, ...]:
        return self._active.available

    def start(self) -> None:
        try:
            self._sources[0].start()
        except Exception as exc:  # noqa: BLE001 - fall back to test pictures
            _logger.warning("marionette: sim cameras unavailable (%s)", exc)
            self._active = self._sources[1]
            self._active.start()

    def stop(self) -> None:
        self._active.stop()

    def grab(self, streams: Any, width: int, quality: int) -> dict[str, bytes]:
        try:
            return self._active.grab(streams, width, quality)
        except Exception as exc:
            if self._active is self._sources[1]:
                raise
            _logger.warning(
                "marionette: sim cameras failed (%s); using test images", exc
            )
            self._active.stop()
            self._active = self._sources[1]
            self._active.start()
            return self._active.grab(streams, width, quality)


def _validate(cfg: MarionetteCmdConfig) -> None:
    from ..marionette.gateway import gateway_url

    gateway_url(cfg.gateway)
    speeds = (cfg.max_speed, cfg.max_turn_rate, cfg.joint_speed, cfg.rate_hz)
    if min(speeds) <= 0 or cfg.rotation_scale < 0 or cfg.pose_cutoff <= 0:
        raise ValueError(
            "rate_hz, max_speed, max_turn_rate, joint_speed and pose_cutoff must "
            "be positive, and rotation_scale not negative."
        )
    if cfg.camera_fps <= 0 or cfg.camera_width < 64:
        raise ValueError("camera_fps must be positive and camera_width at least 64.")


def _run(cfg: MarionetteCmdConfig, stop_event: threading.Event | None = None) -> None:
    """Run a Marionette session until the panel stops it."""
    _validate(cfg)
    session = build_session(cfg)
    asyncio.run(session.run(stop_event))


def main(argv: list[str]) -> None:
    from .config import normalize_bool_flags, parse

    cfg = parse(
        MarionetteCmdConfig, normalize_bool_flags(argv, "sim"), settings_op="marionette"
    )
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        _run(cfg)
    except KeyboardInterrupt:
        print("\nExiting Marionette ...")
