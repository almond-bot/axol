"""
axol gravity-comp

Put the Axol arms into gravity-compensation mode so the operator can move them
by hand. Each free arm joint is sent ``set_impedance(p_des=current, v_des=0,
kp=0, kd=KD, t_ff=gravity)`` at the configured rate; joints not in the free
set are held rigidly at their current position with their configured
``ArmConfig`` gains; the gripper is held softly at its current position.

Every field is reachable from the CLI (draccus-style) or a JSON/YAML file:

    axol gravity-comp
    axol gravity-comp --right_channel null
    axol gravity-comp --kd 1.0
    axol gravity-comp --free_joints [WRIST_3]
    axol gravity-comp --right_channel null --free_joints [SHOULDER_1,WRIST_3]
    axol gravity-comp --config_path my_gravity.json
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..robot import Axol
from ..utils.shared import ARM_JOINTS, Joint
from .config import GravityCompCmdConfig, parse


def _resolve_free_joints(names: list[str] | None) -> set[Joint] | None:
    """Convert a list of joint names into a set of arm ``Joint`` enums.

    Names are case-insensitive and must name one of the seven arm joints
    (``GRIPPER`` is rejected — gravity comp only applies to arm joints).
    ``None`` means "all seven arm joints" and is passed through unchanged.
    """
    if names is None:
        return None
    valid_names = [j.name for j in ARM_JOINTS]
    out: set[Joint] = set()
    for raw in names:
        name = raw.strip().upper()
        if not name:
            continue
        try:
            j = Joint[name]
        except KeyError:
            raise SystemExit(f"unknown joint {name!r}; valid: {', '.join(valid_names)}")
        if j not in ARM_JOINTS:
            raise SystemExit(
                f"{name!r} cannot be gravity-compensated; valid: {', '.join(valid_names)}"
            )
        out.add(j)
    if not out:
        raise SystemExit("free_joints is empty")
    return out


def main(argv: list[str]) -> None:
    """Parse the CLI config and run gravity-compensation mode."""
    cfg = parse(GravityCompCmdConfig, argv)
    # force=True: a dependency imported before this point may install a root
    # handler (leaving the level at WARNING), which would make this a no-op
    # and silently drop log_say() / INFO status lines.
    logging.basicConfig(level=getattr(logging, cfg.log_level), force=True)
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        print("\nExiting gravity comp ...")


async def _run(cfg: GravityCompCmdConfig) -> None:
    if cfg.left_channel is None and cfg.right_channel is None:
        raise SystemExit("Both arms disabled — nothing to do.")

    free_joints = _resolve_free_joints(cfg.free_joints)
    free_str = (
        "all 7 joints"
        if free_joints is None
        else ", ".join(j.name for j in ARM_JOINTS if j in free_joints)
    )
    print(
        f"Gravity comp: free={free_str}; kd={cfg.kd:.2f} Nm·s/rad, "
        f"rate={cfg.rate_hz:.0f} Hz (telemetry={cfg.telemetry_hz:.0f} Hz). "
        f"Press Ctrl-C to exit."
    )

    async with Axol(
        config=cfg.axol,
        left_channel=cfg.left_channel,
        right_channel=cfg.right_channel,
    ) as axol:
        # ``enable()`` (called by ``__aenter__``) leaves arm joints in IMPEDANCE
        # and the gripper in POSITION_FORCE — both of which are the modes
        # ``gravity_compensate`` expects, so we don't touch control modes here.
        await axol.start_telemetry(cfg.telemetry_hz)
        # Settle a few cycles so positions cache is populated before we drive.
        await asyncio.sleep(max(0.05, 5.0 / cfg.telemetry_hz))

        dt = 1.0 / cfg.rate_hz
        while True:
            loop_start = time.monotonic()
            await axol.gravity_compensate(kd=cfg.kd, free_joints=free_joints)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)
