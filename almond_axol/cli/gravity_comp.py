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
import math

from ..robot import Axol
from ..utils.shared import ARM_JOINTS, Joint
from .config import GravityCompCmdConfig, parse


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

    print(
        f"Gravity comp READ-ONLY: no torque commanded — move the joints by hand. "
        f"Logging shoulder_2 every 0.5 s (telemetry={cfg.telemetry_hz:.0f} Hz). "
        f"Press Ctrl-C to exit."
    )

    async with Axol(
        left_channel=cfg.left_channel, right_channel=cfg.right_channel
    ) as axol:
        # ``enable()`` (called by ``__aenter__``) releases the brakes but sends
        # no motion command, so with no ``gravity_compensate`` call below the
        # motors apply no torque and the joints are free to move by hand.
        # torque=True so we can log the measured shoulder_2 torque alongside
        # the model's gravity feedforward.
        await axol.start_telemetry(cfg.telemetry_hz, torque=True)
        # Motors may still be rebooting from set_control_mode(); block until
        # every motor has answered at least one telemetry poll before reading.
        await axol.wait_for_telemetry()

        # Diagnostic: every 0.5 s, log shoulder_2's joint angle, the gravity
        # feedforward the model *would* command at that pose (computed, not
        # sent), and the measured motor torque. Prefer the left arm; fall back
        # to the right.
        arm = axol.left if axol.left is not None else axol.right
        s2_i = ARM_JOINTS.index(Joint.SHOULDER_2)

        while True:
            arm_q = arm.positions[: len(ARM_JOINTS)]
            grav_ff = float(
                arm._gravity_comp.gravity_arm(arm_q, is_left=arm._is_left)[s2_i]
            )
            pos = float(arm_q[s2_i])
            meas = float(arm.torques[s2_i])
            print(
                f"shoulder_2  pos={math.degrees(pos):+7.2f}° ({pos:+.4f} rad)  "
                f"grav_ff={grav_ff:+.3f} Nm  meas={meas:+.3f} Nm",
                flush=True,
            )
            await asyncio.sleep(0.5)
