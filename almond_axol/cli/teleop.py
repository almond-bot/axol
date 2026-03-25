"""
almond-axol teleop [axol|sim]

Run a VR teleoperation session with default parameters.
"""

import argparse
import asyncio
import logging


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("teleop", help="Run a VR teleoperation session.")
    p.add_argument(
        "--robot",
        choices=["axol", "sim"],
        required=True,
        help="Robot backend: 'axol' for hardware, 'sim' for visualizer.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    asyncio.run(_run(args.robot))


async def _run(robot_type: str) -> None:
    from ..robot import Axol, Sim
    from ..teleop import VRTeleop

    robot = Sim() if robot_type == "sim" else Axol()
    async with VRTeleop(robot) as teleop:
        await teleop.run()
