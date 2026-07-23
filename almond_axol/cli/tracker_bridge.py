"""
axol tracker.bridge

Stream Vive tracker poses into a running teleop session as VRFrame JSON.

Run it next to ``axol teleop --umi`` / ``collect-data --umi`` (or against
``axol teleop --sim`` for a dry run): it opens the configured tracker
backend, composes VRFrames at 120 Hz, and connects to the VR WebSocket
server exactly like a headset would — nothing downstream changes.

Backend + left/right binding come from ``~/.almond/tracker/config.json``
(written by ``axol tracker.identify``); every field can be overridden on
the command line. Interim controls until the rig's button PCB exists:
Enter toggles engage, ``r`` resets, ``q`` quits.
"""

from __future__ import annotations

import asyncio
import logging

from ..utils.ports import VR_PORT


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.bridge`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.bridge",
        help="Stream Vive tracker poses to the VR server (headset-free teleop).",
    )
    parser.add_argument(
        "--backend",
        choices=("survive", "ultimate", "synthetic"),
        default=None,
        help="Tracker backend (default: the saved config, else survive).",
    )
    parser.add_argument(
        "--left",
        default=None,
        help="Left-side device key (libsurvive codename / Ultimate MAC); "
        "overrides the saved binding.",
    )
    parser.add_argument(
        "--right",
        default=None,
        help="Right-side device key; overrides the saved binding.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="VR server host (the teleop machine). Default: localhost.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VR_PORT,
        help=f"VR server port. Default: {VR_PORT}.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=120.0,
        help="Frame streaming rate. Default: 120.",
    )
    parser.set_defaults(func=run)


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Open the tracker backend and stream frames until quit."""
    from ..tracker import create_source, load_tracker_config
    from ..tracker.bridge import TrackerBridge
    from ..tracker.synthetic import LEFT_KEY, RIGHT_KEY

    logging.basicConfig(level=logging.INFO, force=True)

    config = load_tracker_config()
    if args.backend is not None:
        config.backend = args.backend
    left = args.left if args.left is not None else config.left
    right = args.right if args.right is not None else config.right
    if config.backend == "synthetic" and left is None and right is None:
        left, right = LEFT_KEY, RIGHT_KEY

    source = create_source(config)
    source.start()
    try:
        bridge = TrackerBridge(
            source,
            left=left,
            right=right,
            host=args.host,
            port=args.port,
            hz=args.hz,
        )
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
