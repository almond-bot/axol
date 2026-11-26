"""``axol mcap.synthetic``: write sample MCAP episodes from the sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RESOLUTIONS = {"HD1200": (1920, 1200), "SVGA": (960, 600)}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mcap.synthetic",
        help="Write sample MCAP episodes from the sim, without a robot.",
        description=(
            "Write two sample MCAP episodes of Axol on Jelly with the Camera Kit, "
            "made in the sim: the right arm puts a cube in a bin with a camera "
            "calibration, and the left arm moves a cup without one. The files "
            "have the same topics, metadata and format as Collect data's."
        ),
    )
    parser.add_argument("output", type=Path, help="folder for the episodes")
    parser.add_argument(
        "--duration", type=float, default=15.0, help="seconds per episode (15)"
    )
    parser.add_argument("--fps", type=int, default=60, help="camera frame rate (60)")
    parser.add_argument(
        "--resolution",
        choices=sorted(_RESOLUTIONS),
        default="HD1200",
        help="camera resolution (HD1200)",
    )
    parser.add_argument(
        "--codec", choices=("h264", "h265"), default="h264", help="video codec (h264)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=24,
        help="encoder CRF: lower is sharper and larger (24)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="processes that render video (4)"
    )
    parser.add_argument(
        "--topic_names",
        default="",
        help="topic renames, as default=new pairs (like MCAP topic names)",
    )
    parser.add_argument(
        "--frame_names",
        default="",
        help="frame renames, as default=new pairs (like MCAP frame names)",
    )
    parser.add_argument(
        "--episode_metadata",
        default="",
        help="key=value pairs saved in each episode (like Episode metadata)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from ..recording.mcap.custom_metadata import parse_episode_metadata
    from ..recording.mcap.topics import parse_frame_names, parse_topic_names
    from ..synthetic.episode import EpisodeSpec, SyntheticOptions, make_episodes

    try:
        topics = parse_topic_names(args.topic_names)
        frames = parse_frame_names(args.frame_names)
        metadata = parse_episode_metadata(args.episode_metadata)
    except ValueError as exc:
        sys.exit(str(exc))
    width, height = _RESOLUTIONS[args.resolution]
    options = SyntheticOptions(
        output=args.output,
        fps=args.fps,
        width=width,
        height=height,
        resolution=args.resolution,
        codec=args.codec,
        quality=args.quality,
        workers=max(1, args.workers),
        topic_names=topics,
        frame_names=frames,
        episode_metadata={**metadata, "synthetic": "true"},
        duration_s=args.duration,
    )
    specs = [
        EpisodeSpec(
            "Put the red cube in the bin.",
            "cube_to_bin",
            calibrated=True,
            seed=1,
        ),
        EpisodeSpec(
            "Move the blue cup next to the bin.",
            "cup_to_table",
            calibrated=False,
            seed=2,
        ),
    ]
    paths = make_episodes(options, specs)
    for path in paths:
        print(path)
