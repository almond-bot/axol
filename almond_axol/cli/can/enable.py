"""
axol can.enable

Runs the CAN startup script to bring up the Almond Axol CAN interfaces —
every configured bus the host has: the arm hub pair (can_alm_axol_l/r), the
cart wheel bus (can_alm_axol_b), and the chest bus (can_alm_axol_c), each
skipped when its adapter isn't attached. Requires can.setup to have been run
at least once to generate the script.
``--umi`` brings up the Mantis rig's interfaces instead of the arm's.

For setups without the Axol-named adapters, pass ``--channels`` to bring up
other SocketCAN interfaces directly instead (no startup script needed):

    axol can.enable --channels can0
    axol can.enable --channels can0 can1
"""

import argparse

from .setup import _AXOL_PROFILE, _UMI_PROFILE, bring_up_can, bring_up_interfaces


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.enable`` subcommand."""
    parser = subparsers.add_parser(
        "can.enable",
        help="Bring up CAN interfaces using the startup script.",
    )
    parser.add_argument(
        "--umi",
        action="store_true",
        help="Bring up the Mantis interfaces instead of the arm's.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface(s) to bring up instead of the Axol hub's "
        "(e.g. --channels can0 can1), for setups without the Axol hub CAN "
        "adapter. Skips the hub startup script.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace | None = None) -> None:
    """Bring up the CAN interfaces.

    Default: the Axol hub's saved startup script (with its RX-wedge recovery);
    ``--umi`` selects the Mantis rig's script instead. With ``--channels``: plain
    per-interface bring-up of the named interfaces.
    """
    channels = getattr(args, "channels", None)
    if channels:
        print(f"Bringing up CAN interfaces: {', '.join(channels)}")
        try:
            bring_up_interfaces(channels)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1) from None
        return

    profile = _UMI_PROFILE if getattr(args, "umi", False) else _AXOL_PROFILE
    if not profile.cron_script.exists():
        print(f"ERROR: Startup script not found at {profile.cron_script}.")
        setup_cmd = (
            "axol can.setup --umi" if profile is _UMI_PROFILE else "axol can.setup"
        )
        print(
            f"Run '{setup_cmd}' first"
            + (
                ", or pass --channels for a non-hub adapter."
                if profile is _AXOL_PROFILE
                else "."
            )
        )
        raise SystemExit(1)

    bring_up_can(profile)
