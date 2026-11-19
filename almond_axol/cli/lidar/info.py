"""
axol lidar.info

Show the lidar's model, firmware, address, temperature and faults, then
measure its data rate for two seconds.

Usage:
    axol lidar.info
    axol lidar.info --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from ...lidar import LidarConfig, LivoxClient
from . import format_stats, format_status, measure


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser(
        "lidar.info", help="Show the lidar's status and measure its data rate."
    )
    p.add_argument("--json", action="store_true", help="Print JSON.")
    p.add_argument(
        "--seconds", type=float, default=2.0, help="How long to measure (default 2)."
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    cfg = LidarConfig.load()
    if not cfg.lidar_ip or not cfg.host_ip:
        raise SystemExit("The lidar isn't set up. Run `axol lidar.setup` first.")
    try:
        with LivoxClient(cfg.lidar_ip, cfg.host_ip) as client:
            status = client.status()
    except (TimeoutError, OSError) as exc:
        raise SystemExit(
            f"The lidar at {cfg.lidar_ip} doesn't answer ({exc}). Check its power "
            "and cable, then run `axol lidar.setup`."
        ) from None
    stats = measure(cfg, args.seconds) if args.seconds > 0 else None
    if args.json:
        print(
            json.dumps(
                {
                    "status": asdict(status),
                    "stream": asdict(stats) if stats else None,
                },
                indent=2,
            )
        )
        return
    print(format_status(status))
    if stats is not None:
        print(format_stats(stats))
