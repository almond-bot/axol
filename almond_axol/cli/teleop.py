"""
axol teleop --robot [axol|sim]

Run a VR teleoperation session. Every robot config field is reachable
from the CLI (draccus-style) or from a JSON/YAML file:

    axol teleop --robot sim
    axol teleop --robot axol --axol.left_stiffness 0.8
    axol teleop --robot axol --axol.left.elbow.kp 60 --axol.right.gripper.torque_limit 0.7
    axol teleop --robot axol --left_channel null            # disable the left arm
    axol teleop --robot axol --config_path my_teleop.json   # whole-config file
"""

import asyncio
import logging
import socket

from .config import TeleopCmdConfig, parse


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def main(argv: list[str]) -> None:
    cfg = parse(TeleopCmdConfig, argv)
    logging.basicConfig(level=getattr(logging, cfg.log_level))

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(_run(cfg))


async def _run(cfg: TeleopCmdConfig) -> None:
    from ..robot import Axol, Sim
    from ..teleop import VRTeleop

    if cfg.robot == "sim":
        robot = Sim()
    else:
        robot = Axol(
            config=cfg.axol,
            left_channel=cfg.left_channel,
            right_channel=cfg.right_channel,
        )
    async with VRTeleop(robot) as teleop:
        await teleop.run()
