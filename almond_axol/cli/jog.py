"""
Jog Axol from the control panel's Jog page.

Runs a jog session that holds the arms where they are and waits for the
page: plan an IK move for one gripper and execute it, raise or lower the
lift while a button is held, and drive the Jelly base from the drive pad.
With ``--sim`` nothing touches the hardware, and the lift and base are
simulated with the real controllers' speeds and ramps.

The session is started and stopped by the control panel (``axol serve``),
which owns the page's connection to it. Run it from the panel.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from ..constants import CAN_LEFT, CAN_RIGHT
from ..kinematics.config import KinematicsConfig, apply_mantis_kinematics_profile
from ..robot.config import AxolConfig
from ..settings import resolve_robot_model
from .config import LogLevel

_logger = logging.getLogger(__name__)


@dataclass
class JogCmdConfig:
    """Config for the ``jog`` operation.

    Arm gains and channels come from the shared robot settings, the same as
    gravity compensation and waypoints; the lift and base use the shared
    ``jelly.*`` settings, the same as teleop.
    """

    axol: AxolConfig = field(default_factory=AxolConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    sim: bool = False
    """Jog a simulated robot instead of the hardware."""
    arm_speed: float = 0.6
    """Average speed (rad/s) of the furthest-moving joint along a planned move."""
    min_duration: float = 1.0
    """Shortest planned move, in seconds."""
    rate_hz: float = 125.0
    """Control-loop rate: arm commands and planned moves are streamed at it."""
    kd: float = 0.25
    """Gravity compensation's hand-guiding damping. Shared with gravity
    compensation's settings; jogging never hand-guides."""
    log_level: LogLevel = "INFO"


def build_session(cfg: JogCmdConfig) -> Any:
    """The :class:`~almond_axol.jog.session.JogSession` this config describes."""
    from ..jog.backends import HardwareBackend, SimBackend
    from ..jog.planner import JogPlanner
    from ..robot.jelly import JellyConfig
    from ..settings import shared_config

    jelly_cfg = shared_config(JellyConfig, "teleop", "jelly")
    kinematics = cfg.kinematics
    apply_mantis_kinematics_profile(kinematics)

    def planner() -> JogPlanner:
        from ..kinematics.solver import KinematicsSolver
        from ..jog.backends import rest_q

        built = JogPlanner(KinematicsSolver(kinematics))
        # Plan one short move so the trajectory graph compiles now, not on
        # the operator's first Plan.
        q = rest_q()
        left, _ = built.tips(q)
        built.plan(
            q,
            "left",
            (left[0] + 0.01, left[1]),
            speed=cfg.arm_speed,
            rate=cfg.rate_hz,
            min_duration=cfg.min_duration,
        )
        return built

    if cfg.sim:
        model = resolve_robot_model(kinematics.robot_model, mount=kinematics.mount)
        backend: Any = SimBackend(model, jelly_cfg)
    else:
        from ..robot import Axol
        from ..robot.jelly import Jelly, detect_jelly

        axol = None
        if cfg.left_channel is not None or cfg.right_channel is not None:
            axol = Axol(
                config=cfg.axol,
                left_channel=cfg.left_channel,
                right_channel=cfg.right_channel,
            )
        detected = detect_jelly(jelly_cfg)
        jelly = Jelly(detected) if detected is not None else None
        if axol is None and jelly is None:
            raise ValueError(
                "Both arms are turned off and no Jelly is attached, so there is "
                "nothing to jog."
            )
        backend = HardwareBackend(axol, jelly)

    from ..jog.session import JogSession

    return JogSession(
        backend,
        planner,
        rate=cfg.rate_hz,
        arm_speed=cfg.arm_speed,
        min_duration=cfg.min_duration,
    )


def _run(cfg: JogCmdConfig, stop_event: threading.Event | None = None) -> None:
    """Run a jog session until the panel stops it."""
    import asyncio

    if cfg.rate_hz <= 0 or cfg.arm_speed <= 0 or cfg.min_duration < 0:
        raise ValueError(
            "rate_hz and arm_speed must be positive and min_duration non-negative."
        )
    session = build_session(cfg)
    asyncio.run(session.run(stop_event))


def main(argv: list[str]) -> None:
    from .config import normalize_bool_flags, parse

    cfg = parse(JogCmdConfig, normalize_bool_flags(argv, "sim"), settings_op="jog")
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        _run(cfg)
    except KeyboardInterrupt:
        print("\nExiting jog ...")
