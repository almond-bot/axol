"""
almond-axol can.enable

Runs the CAN startup script to bring up the Almond Axol CAN interfaces.
Requires can.setup to have been run at least once to generate the script.
"""

from .setup import _CRON_SCRIPT, _bring_up_can


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    subparsers.add_parser(
        "can.enable",
        help="Bring up CAN interfaces using the startup script.",
    ).set_defaults(func=run)


def run(_args: object = None) -> None:
    if not _CRON_SCRIPT.exists():
        print(f"ERROR: Startup script not found at {_CRON_SCRIPT}.")
        print("Run 'almond-axol can.setup' first.")
        raise SystemExit(1)

    _bring_up_can()
