"""
axol tracker.bridge

Stream Vive tracker poses into a running teleop session as VRFrame JSON.

Run it next to ``axol teleop --mantis`` / ``collect-data --mantis`` (or against
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
from typing import Any, Callable

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
        "(Mantis) engagement fits the base transform from BOTH controller "
        "positions, so the placeholder pose streamed for the unbound side "
        "corrupts it.",
    )
    parser.set_defaults(func=run)


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Open the tracker backend (and trigger PCBs) and stream frames until quit."""
    logging.basicConfig(level=logging.INFO, force=True)

    from ..tracker import load_tracker_config

    config = load_tracker_config()
    if args.backend is not None:
        config.backend = args.backend
    if args.left is not None:
        config.left = args.left
    if args.right is not None:
        config.right = args.right

    if args.trigger_can_left is not None:
        config.trigger_can_left = args.trigger_can_left
    if args.trigger_can_right is not None:
        config.trigger_can_right = args.trigger_can_right
    config.allow_single_side = args.allow_single_side or config.allow_single_side

    run_configured_bridge(
        config,
        host=args.host,
        port=args.port,
        hz=args.hz,
    )


def run_configured_bridge(
    config: Any,
    *,
    host: str = "localhost",
    port: int = VR_PORT,
    hz: float = 120.0,
    controls: Any = None,
    on_ready: Callable[[], None] | None = None,
) -> None:
    """Run one configured bridge, optionally under headless lifecycle controls.

    ``on_ready`` fires after the tracker backend, trigger readers, and bridge
    object are ready. The WebSocket connection may not exist yet: the bridge
    deliberately starts before the operation's VR server and reconnects until
    that server begins listening.
    """
    from ..tracker import HARDWARE_FREE_BINDINGS, create_source
    from ..tracker.bridge import TrackerBridge
    from ..tracker.trigger import TriggerReader

    left, right = config.left, config.right
    binding = HARDWARE_FREE_BINDINGS.get(config.backend)
    if binding is not None and left is None and right is None:
        left, right = binding

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
            host=host,
            port=port,
            hz=hz,
            controls=controls,
            left_trigger=triggers.get("left"),
            right_trigger=triggers.get("right"),
            allow_single_side=config.allow_single_side,
        )
        if on_ready is not None:
            on_ready()
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        for reader in triggers.values():
            reader.close()
        source.stop()
