"""
axol teleop --robot [axol|sim]

Run a VR teleoperation session with default parameters.
"""

import argparse
import asyncio
import logging
import socket

from ..shared import (
    STIFFNESS_HELP_TEMPLATE,
    STIFFNESS_METAVAR,
    parse_stiffness,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("teleop", help="Run a VR teleoperation session.")
    p.add_argument(
        "--robot",
        choices=["axol", "sim"],
        required=True,
        help="Robot backend: 'axol' for hardware, 'sim' for visualizer.",
    )
    p.add_argument(
        "--no-left",
        action="store_true",
        help="Disable the left arm.",
    )
    p.add_argument(
        "--no-right",
        action="store_true",
        help="Disable the right arm.",
    )
    p.add_argument(
        "--left-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the left gripper in POSITION_FORCE mode (default: 1.0).",
    )
    p.add_argument(
        "--right-gripper-torque-limit",
        type=float,
        default=1.0,
        help="Max output torque (Nm) for the right gripper in POSITION_FORCE mode (default: 1.0).",
    )
    p.add_argument(
        "--left-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=STIFFNESS_METAVAR,
        help=STIFFNESS_HELP_TEMPLATE.format(side="left", attr="left_stiffness"),
    )
    p.add_argument(
        "--right-stiffness",
        type=parse_stiffness,
        default=0.0,
        metavar=STIFFNESS_METAVAR,
        help=STIFFNESS_HELP_TEMPLATE.format(side="right", attr="right_stiffness"),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))

    hostname = socket.gethostname()
    local_ip = _get_local_ip()
    print("Connect the VR app (https://axol.almond.bot) to this machine:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")

    asyncio.run(
        _run(
            args.robot,
            no_left=args.no_left,
            no_right=args.no_right,
            left_gripper_torque_limit=args.left_gripper_torque_limit,
            right_gripper_torque_limit=args.right_gripper_torque_limit,
            left_stiffness=args.left_stiffness,
            right_stiffness=args.right_stiffness,
        )
    )


async def _run(
    robot_type: str,
    *,
    no_left: bool = False,
    no_right: bool = False,
    left_gripper_torque_limit: float = 1.0,
    right_gripper_torque_limit: float = 1.0,
    left_stiffness: float | tuple[float, ...] = 0.0,
    right_stiffness: float | tuple[float, ...] = 0.0,
) -> None:
    from ..robot import Axol, Sim
    from ..robot.config import AxolConfig
    from ..teleop import VRTeleop

    if robot_type == "sim":
        robot = Sim()
    else:
        kwargs = {}
        if no_left:
            kwargs["left_channel"] = None
        if no_right:
            kwargs["right_channel"] = None
        axol_config = AxolConfig(
            left_stiffness=left_stiffness,
            right_stiffness=right_stiffness,
        )
        axol_config.left.gripper.torque_limit = left_gripper_torque_limit
        axol_config.right.gripper.torque_limit = right_gripper_torque_limit
        robot = Axol(config=axol_config, **kwargs)
    async with VRTeleop(robot) as teleop:
        await teleop.run()
