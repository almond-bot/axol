"""
axol can.enable

Runs the CAN startup script to bring up the Almond Axol CAN interfaces —
every configured bus the host has: the arm hub pair (can_alm_axol_l/r), the
cart wheel bus (can_alm_axol_b), and the chest bus (can_alm_axol_c), each
skipped when its adapter isn't attached. Requires can.setup to have been run
at least once to generate the scripts. If both Axol and Mantis are configured,
both are brought up automatically.

For setups without the Axol-named adapters, pass ``--channels`` to bring up
other SocketCAN interfaces directly instead (no startup script needed):

    axol can.enable --channels can0
    axol can.enable --channels can0 can1
"""

import argparse

from .setup import _AXOL_PROFILE, _MANTIS_PROFILE, bring_up_can, bring_up_interfaces


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.enable`` subcommand."""
    parser = subparsers.add_parser(
        "can.enable",
        help="Bring up CAN interfaces using the startup script.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        metavar="IFACE",
        help="SocketCAN interface(s) to bring up directly "
        "(e.g. --channels can0 can1). Skips all saved hub startup scripts.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace | None = None) -> None:
    """Bring up the CAN interfaces.

    Default: every saved Axol/Mantis startup script (with RX-wedge recovery).
    With ``--channels``: plain per-interface bring-up of the named interfaces.
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

    profiles = [
        profile
        for profile in (_AXOL_PROFILE, _MANTIS_PROFILE)
        if profile.cron_script.exists()
    ]
    if not profiles:
        print("ERROR: No configured Axol or Mantis CAN startup scripts found.")
        print("Run 'axol can.setup' first, or pass --channels for another adapter.")
        raise SystemExit(1)

    for profile in profiles:
        bring_up_can(profile)
