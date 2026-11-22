"""``axol mcap.check``: check MCAP episodes recorded by Collect data."""

from __future__ import annotations

import argparse
import sys


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mcap.check",
        help="Check MCAP episodes recorded by Collect data.",
        description=(
            "Check MCAP episodes: video, timestamps, joints, base, transforms, "
            "camera calibration and metadata. Exits with status 1 if a required "
            "part is missing."
        ),
    )
    parser.add_argument("files", nargs="+", help="MCAP episode files")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from ..recording.mcap.check import main

    sys.exit(main(args.files))
