"""Per-motor CLI commands (info, health, CAN ID, zero-position calibration)."""

import argparse

from ...constants import CAN_LEFT, CAN_RIGHT


def add_side_and_channel_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the shared arm selector plus the CAN interface override.

    ``--l`` / ``--r`` pick which arm (and therefore which Axol hub CAN
    interface) the command talks to; ``--channel`` overrides the interface
    name for setups without the Axol hub CAN adapter (e.g. a generic
    single-channel adapter enumerated as ``can0``).
    """
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help=f"Left arm ({CAN_LEFT})")
    side.add_argument("--r", action="store_true", help=f"Right arm ({CAN_RIGHT})")
    parser.add_argument(
        "--channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface to use instead of the selected arm's Axol "
        "hub interface, for setups without the Axol hub CAN adapter "
        "(e.g. can0).",
    )


def resolve_channel(args: argparse.Namespace) -> str:
    """The CAN interface to use: ``--channel`` if given, else the arm's."""
    channel = getattr(args, "channel", None)
    if channel:
        return channel
    return CAN_LEFT if args.l else CAN_RIGHT
