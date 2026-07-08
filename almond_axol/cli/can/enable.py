"""
axol can.enable

Runs the CAN startup script to bring up the Almond Axol CAN interfaces.
Requires can.setup to have been run at least once to generate the script.
"""

from .setup import _AXOL_PROFILE, _UMI_PROFILE, _bring_up_can


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``can.enable`` subcommand."""
    parser = subparsers.add_parser(
        "can.enable",
        help="Bring up CAN interfaces using the startup script.",
    )
    parser.add_argument(
        "--umi",
        action="store_true",
        help="Bring up the handheld UMI rig interfaces instead of the arm's.",
    )
    parser.set_defaults(func=run)


def run(args: object = None) -> None:
    """Bring up the CAN interfaces using the saved startup script."""
    profile = _UMI_PROFILE if getattr(args, "umi", False) else _AXOL_PROFILE
    if not profile.cron_script.exists():
        print(f"ERROR: Startup script not found at {profile.cron_script}.")
        setup_cmd = (
            "axol can.setup --umi" if profile is _UMI_PROFILE else "axol can.setup"
        )
        print(f"Run '{setup_cmd}' first.")
        raise SystemExit(1)

    _bring_up_can(profile)
