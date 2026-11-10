"""Per-motor CLI commands (info, health, CAN ID, zero-position calibration)."""

import argparse

from ...constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT

TARGETS = ("axol", "mantis")


def add_side_and_channel_arguments(
    parser: argparse.ArgumentParser,
    *,
    supports_mantis: bool = True,
    channel_replaces_side: bool = False,
) -> None:
    """Register the shared arm selector plus the CAN interface override.

    ``--l`` / ``--r`` pick which arm (and therefore which Axol hub CAN
    interface) the command talks to; ``--channel`` overrides the interface
    name for setups without the Axol hub CAN adapter (e.g. a generic
    single-channel adapter enumerated as ``can0``).

    ``channel_replaces_side`` is for per-motor tools that also serve a motor
    outside any arm (a test stand): ``--channel`` alone is then enough, and
    :func:`resolve_channel` asks for one of the two when both are missing.
    """
    side = parser.add_mutually_exclusive_group(required=not channel_replaces_side)
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
        help=(
            "SocketCAN interface the motor is on (e.g. can0). Use it instead of "
            "--l / --r for a motor outside the arms, such as on a test stand."
            if channel_replaces_side
            else "SocketCAN interface to use instead of the selected hardware's "
            "left/right default (e.g. can0)."
        ),
    )


def resolve_channel(args: argparse.Namespace) -> str:
    """The explicit channel, or the selected hardware's left/right bus."""
    channel = getattr(args, "channel", None)
    if channel:
        return channel
    if not (getattr(args, "l", False) or getattr(args, "r", False)):
        raise SystemExit("error: pick an arm with --l or --r, or a bus with --channel.")
    if getattr(args, "target", "axol") == "mantis":
        return CAN_MANTIS_LEFT if args.l else CAN_MANTIS_RIGHT
    return CAN_LEFT if args.l else CAN_RIGHT
