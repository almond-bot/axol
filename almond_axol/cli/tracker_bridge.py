"""
axol tracker.bridge

Stream Vive tracker poses into a running teleop session as VRFrame JSON.

Run it next to ``axol teleop --umi`` / ``collect-data --umi`` (or against
``axol teleop --sim`` for a dry run): it opens the configured tracker
backend, composes VRFrames at 120 Hz, and connects to the VR WebSocket
server exactly like a headset would — nothing downstream changes.

Backend + left/right binding come from ``~/.almond/tracker/config.json``
(written by ``axol tracker.identify``); every field can be overridden on
the command line. When a rig's CAN trigger node is configured
(``--trigger-can-left`` / ``--trigger-can-right``), its binary trigger
switch drives the grip command (pressed = close, released = open);
engage/reset stay on stdin (Enter toggles engage, ``r`` resets, ``q``
quits) until the button PCB exists.
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
    parser.add_argument(
        "--trigger-can-left",
        default=None,
        help="SocketCAN interface of the left rig's trigger node "
        "(e.g. can_alm_umi_l); overrides the saved config.",
    )
    parser.add_argument(
        "--trigger-can-right",
        default=None,
        help="SocketCAN interface of the right rig's trigger node; "
        "overrides the saved config.",
    )
    parser.add_argument(
        "--allow-single-side",
        action="store_true",
        help="Run with only one side's tracker bound. WARNING: absolute-mode "
        "(UMI) engagement fits the base transform from BOTH controller "
        "positions, so the placeholder pose streamed for the unbound side "
        "corrupts it.",
    )
    parser.set_defaults(func=run)


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Open the tracker backend (and trigger PCBs) and stream frames until quit."""
    from ..tracker import create_source, load_tracker_config
    from ..tracker.bridge import TrackerBridge
    from ..tracker.synthetic import LEFT_KEY, RIGHT_KEY
    from ..tracker.trigger import TriggerReader

    logging.basicConfig(level=logging.INFO, force=True)

    config = load_tracker_config()
    if args.backend is not None:
        config.backend = args.backend
    left = args.left if args.left is not None else config.left
    right = args.right if args.right is not None else config.right
    if config.backend == "synthetic" and left is None and right is None:
        left, right = LEFT_KEY, RIGHT_KEY

    if args.trigger_can_left is not None:
        config.trigger_can_left = args.trigger_can_left
    if args.trigger_can_right is not None:
        config.trigger_can_right = args.trigger_can_right
    allow_single_side = args.allow_single_side or config.allow_single_side

    triggers: dict[str, TriggerReader] = {}
    source = create_source(config)
    source.start()
    try:
        for side, channel in (
            ("left", config.trigger_can_left),
            ("right", config.trigger_can_right),
        ):
            if channel:
                triggers[side] = TriggerReader(channel)
        bridge = TrackerBridge(
            source,
            left=left,
            right=right,
            host=args.host,
            port=args.port,
            hz=args.hz,
            left_trigger=triggers.get("left"),
            right_trigger=triggers.get("right"),
            allow_single_side=allow_single_side,
        )
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        for reader in triggers.values():
            reader.close()
        source.stop()
