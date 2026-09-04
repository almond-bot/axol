"""Per-motor CLI commands (info, health, CAN ID, zero-position calibration)."""

import argparse

from ...constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT

TARGETS = ("axol", "mantis")


def add_side_and_channel_arguments(
    parser: argparse.ArgumentParser, *, supports_mantis: bool = True
) -> None:
    """Register the shared arm selector plus the CAN interface override.

    ``--l`` / ``--r`` pick which arm (and therefore which Axol hub CAN
    interface) the command talks to; ``--channel`` overrides the interface
    name for setups without the Axol hub CAN adapter (e.g. a generic
    single-channel adapter enumerated as ``can0``).
    """
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--l", action="store_true", help=f"Left side ({CAN_LEFT})")
    side.add_argument("--r", action="store_true", help=f"Right side ({CAN_RIGHT})")
    if supports_mantis:
        parser.add_argument(
            "--target",
            choices=TARGETS,
            default="axol",
            help="Hardware whose left/right bus to use when --channel is omitted "
            "(default: %(default)s).",
        )
    parser.add_argument(
        "--channel",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface to use instead of the selected hardware's "
        "left/right default (e.g. can0).",
    )


def resolve_channel(args: argparse.Namespace) -> str:
    """The explicit channel, or the selected hardware's left/right bus."""
    channel = getattr(args, "channel", None)
    if channel:
        return channel
    if getattr(args, "target", "axol") == "mantis":
        return CAN_MANTIS_LEFT if args.l else CAN_MANTIS_RIGHT
    return CAN_LEFT if args.l else CAN_RIGHT
