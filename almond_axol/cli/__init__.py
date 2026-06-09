"""CLI entry point for the axol command registered via pyproject.toml."""

import argparse
import importlib
import sys

from . import serve as serve_cmd
from .can import enable as can_enable
from .can import setup as can_setup
from .motor import info as motor_info
from .motor import set_can_id, set_zero_pos
from .tune import friction, pid, repeatability
from .zed import install as zed_install
from .zed import stream as zed_stream
from .zed import sync_clocks as zed_sync_clocks

# Commands that parse their config with draccus instead of argparse. Their
# dotted ``--section.field`` overrides aren't compatible with argparse
# subparsers, so we intercept them before argparse runs and hand the raw
# argv tail to the module's ``main(argv)``. They're imported lazily so that
# e.g. ``axol teleop --sim`` never pulls in the lerobot/camera stack
# that ``collect-data`` / ``run-policy`` import at module load.
_DRACCUS_COMMANDS: dict[str, tuple[str, str]] = {
    "teleop": ("teleop", "Run a VR teleoperation session."),
    "gravity-comp": ("gravity_comp", "Hold the Axol in gravity-compensation mode."),
    "collect-data": ("collect_data", "Record teleoperation episodes."),
    "run-policy": ("run_policy", "Run a trained policy on the robot."),
}


def _dispatch_draccus(command: str, argv: list[str]) -> None:
    module_name, _ = _DRACCUS_COMMANDS[command]
    module = importlib.import_module(f".{module_name}", __name__)
    module.main(argv)


def main() -> None:
    """Dispatch ``axol <command>`` to the matching CLI handler."""
    argv = sys.argv[1:]
    if argv and argv[0] in _DRACCUS_COMMANDS:
        _dispatch_draccus(argv[0], argv[1:])
        return

    parser = argparse.ArgumentParser(prog="axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    can_setup.add_parser(subparsers)
    can_enable.add_parser(subparsers)
    set_can_id.add_parser(subparsers)
    set_zero_pos.add_parser(subparsers)
    motor_info.add_parser(subparsers)
    zed_stream.add_parser(subparsers)
    zed_sync_clocks.add_parser(subparsers)
    zed_install.add_parser(subparsers)
    pid.add_parser(subparsers)
    friction.add_parser(subparsers)
    repeatability.add_parser(subparsers)
    serve_cmd.add_parser(subparsers)

    # Register the draccus commands as bare subparsers purely so they show
    # up in ``axol --help``; their real parsing happens in the interceptor
    # above (``axol <cmd> --help`` is handled by the command's own parser).
    for name, (_, help_text) in _DRACCUS_COMMANDS.items():
        subparsers.add_parser(name, help=help_text, add_help=False)

    args = parser.parse_args()
    args.func(args)
