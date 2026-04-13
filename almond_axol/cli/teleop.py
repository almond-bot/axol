"""
almond-axol teleop --robot [axol|sim]

Run a VR teleoperation session with default parameters.
"""

import argparse
import asyncio
import logging
import socket


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

    asyncio.run(_run(args.robot, no_left=args.no_left, no_right=args.no_right))


async def _run(
    robot_type: str, *, no_left: bool = False, no_right: bool = False
) -> None:
    from ..robot import Axol, Sim
    from ..teleop import VRTeleop

    if robot_type == "sim":
        robot = Sim()
    else:
        kwargs = {}
        if no_left:
            kwargs["left_channel"] = None
        if no_right:
            kwargs["right_channel"] = None
        robot = Axol(**kwargs)
    async with VRTeleop(robot) as teleop:
        await teleop.run()
