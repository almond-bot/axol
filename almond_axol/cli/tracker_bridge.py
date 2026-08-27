"""
axol tracker.bridge

Stream Vive tracker poses into a running teleop session as VRFrame JSON.

Run it next to ``axol teleop --umi`` / ``collect-data --umi`` (or against
``axol teleop --sim`` for a dry run): it opens the configured tracker
backend, composes VRFrames at 120 Hz, and connects to the VR WebSocket
server exactly like a headset would — nothing downstream changes.

With ``--backend static`` it needs no tracker hardware at all: the arms
hold a fixed pose and the rig's CAN trigger node is the only live input,
which is the quickest way to bring up or debug a Mantis gripper.

Backend + left/right binding come from ``~/.almond/tracker/config.json``
(written by ``axol tracker.identify``); every field can be overridden on
the command line. The rig's CAN trigger nodes default to the Mantis
gripper buses (override with ``--trigger-can-left`` /
``--trigger-can-right``), and their analog trigger position drives the
grip command proportionally (squeeze = close, release = open); engage/
reset stay on stdin (Enter toggles engage, ``r`` resets, ``q`` quits)
until the button PCB exists.
"""

from __future__ import annotations

import asyncio
import logging

from ..utils.ports import VR_PORT

_logger = logging.getLogger(__name__)


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.bridge`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.bridge",
        help="Stream Vive tracker poses to the VR server (headset-free teleop).",
    )
    parser.add_argument(
        "--backend",
        choices=("survive", "ultimate", "synthetic", "static"),
        default=None,
        help="Tracker backend (default: the saved config, else survive). "
        "Use static for gripper-only Mantis teleop with no tracker hardware: "
        "the arms hold still and only the trigger node drives the gripper.",
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
        help="SocketCAN interface of the left rig's trigger node; overrides "
        "the saved config, which defaults to the Mantis gripper bus.",
    )
    parser.add_argument(
        "--trigger-can-right",
        default=None,
        help="SocketCAN interface of the right rig's trigger node; overrides "
        "the saved config, which defaults to the Mantis gripper bus.",
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
    from ..tracker import HARDWARE_FREE_BINDINGS, create_source, load_tracker_config
    from ..tracker.bridge import TrackerBridge
    from ..tracker.trigger import TriggerReader

    logging.basicConfig(level=logging.INFO, force=True)

    config = load_tracker_config()
    if args.backend is not None:
        config.backend = args.backend
    left = args.left if args.left is not None else config.left
    right = args.right if args.right is not None else config.right
    binding = HARDWARE_FREE_BINDINGS.get(config.backend)
    if binding is not None and left is None and right is None:
        left, right = binding

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
            if not channel:
                continue
            try:
                triggers[side] = TriggerReader(channel)
            except Exception:
                # No trigger node reachable on this host (sim dry run, a rig
                # without the PCB, or the CAN bus not brought up). That side
                # streams fully open rather than failing the whole session.
                _logger.warning(
                    "%s trigger node on %s could not be opened — that side's "
                    "gripper will stream fully open",
                    side,
                    channel,
                    exc_info=True,
                )
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
