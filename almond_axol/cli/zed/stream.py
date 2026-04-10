"""
almond-axol zed.stream

Stream ZED-X One cameras over the local network using HEVC.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

_SENDER_IP = "192.168.10.1/24"

_VALID_RESOLUTIONS = ["HD1080", "HD1200", "SVGA"]


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "zed.stream", help="Stream ZED-X One cameras over the local network."
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
        "--resolution",
        default="HD1080",
        choices=_VALID_RESOLUTIONS,
        help="Capture resolution for all cameras (default: HD1080).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        metavar="FPS",
        help="Capture frame rate for all cameras (default: 60).",
    )
    p.add_argument(
        "--bitrate",
        type=int,
        default=8000,
        metavar="KBPS",
        help="HEVC encoding bitrate in kbits/s (default: 8000).",
    )
    p.add_argument(
        "--setup-ip",
        metavar="IFACE",
        default=None,
        help=f"Assign the sender IP ({_SENDER_IP}) to IFACE before streaming (e.g. eth0). Requires sudo.",
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
        raise SystemExit(
            "error: at least one of --overhead, --left-arm, --right-arm must be provided"
        )
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.setup_ip:
        from ...shared import setup_link_ip

        setup_link_ip(args.setup_ip, _SENDER_IP)
    try:
        asyncio.run(
            _run(
                args.overhead,
                args.left_arm,
                args.right_arm,
                args.resolution,
                args.fps,
                args.bitrate,
            )
        )
    except KeyboardInterrupt:
        pass


async def _run(
    overhead: int | None,
    left_arm: int | None,
    right_arm: int | None,
    resolution: str,
    fps: int,
    bitrate: int,
) -> None:
    import pyzed.sl as sl

    from ...zed import ZedConfig, ZedStreamer

    config = ZedConfig(
        overhead_serial=overhead,
        left_arm_serial=left_arm,
        right_arm_serial=right_arm,
        resolution=getattr(sl.RESOLUTION, resolution),
        fps=fps,
        bitrate=bitrate,
    )
    async with ZedStreamer(config):
        await asyncio.sleep(float("inf"))
