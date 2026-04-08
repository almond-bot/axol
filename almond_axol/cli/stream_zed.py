"""
almond-axol stream-zed

Start HEVC HD720 streaming for all three ZED cameras.
"""

from __future__ import annotations

import argparse
import asyncio
import logging


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "stream-zed", help="Stream all three ZED cameras over the local network."
    )
    p.add_argument(
        "--overhead",
        type=int,
        default=None,
        metavar="SERIAL",
        help="Serial number of the overhead camera.",
    )
    p.add_argument(
        "--left-arm",
        type=int,
        default=None,
        metavar="SERIAL",
        help="Serial number of the left-arm camera.",
    )
    p.add_argument(
        "--right-arm",
        type=int,
        default=None,
        metavar="SERIAL",
        help="Serial number of the right-arm camera.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.overhead is None and args.left_arm is None and args.right_arm is None:
        raise SystemExit("error: at least one of --overhead, --left-arm, --right-arm must be provided")
    logging.basicConfig(level=getattr(logging, args.log_level))
    asyncio.run(_run(args.overhead, args.left_arm, args.right_arm))


async def _run(overhead: int, left_arm: int, right_arm: int) -> None:
    from ..zed import ZedConfig, ZedStreamer

    config = ZedConfig(
        overhead_serial=overhead,
        left_arm_serial=left_arm,
        right_arm_serial=right_arm,
    )
    async with ZedStreamer(config):
        await asyncio.sleep(float("inf"))
