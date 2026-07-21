"""CLI entry point for the axol command registered via pyproject.toml."""

import argparse
import importlib
import sys

from ..utils.dotenv import load_local_env
from . import provision as provision_cmd
from . import serve as serve_cmd
from .can import driver as can_driver
from .can import enable as can_enable
from .can import setup as can_setup
from .gst import build_zed as gst_build_zed
from .gst import install as gst_install
from .jetson import setup as jetson_setup
from .motor import health as motor_health
from .motor import info as motor_info
from .motor import set_can_id, set_zero_pos
from .tune import friction, pid, repeatability
from .zed import driver as zed_driver
from .zed import install as zed_install

# Commands that parse their config with draccus instead of argparse. Their
# dotted ``--section.field`` overrides aren't compatible with argparse
# subparsers, so we intercept them before argparse runs and hand the raw
# argv tail to the module's ``main(argv)``. They're imported lazily so that
# e.g. ``axol teleop --sim`` never pulls in the lerobot/camera stack
# that ``collect-data`` / ``run-policy`` import at module load.
# Diagnostics scripts under ``almond_axol.diagnostics``. Like the draccus
# commands they are dispatched lazily, before argparse: their modules import
# the full robot stack (numpy, mujoco via the gravity compensator), which
# would slow every ``axol`` invocation if imported here for registration.
# Each module's ``main(argv)`` owns its own argparse parser.
_DIAG_COMMANDS: dict[str, tuple[str, str]] = {
    "diag.rom-enable": (
        "almond_axol.diagnostics.rom.enable",
        "Range-of-motion soak test: sweep every joint for two hours.",
    ),
    "diag.rom-disable": (
        "almond_axol.diagnostics.rom.disable",
        "Open the grippers left clamped by the ROM test and power down.",
    ),
    "diag.zed-cable": (
        "almond_axol.diagnostics.zed.cable",
        "Verify a ZED camera cable by validating captured frames.",
    ),
}

_DRACCUS_COMMANDS: dict[str, tuple[str, str]] = {
    "teleop": ("teleop", "Run a VR teleoperation session."),
    "gravity-comp": ("gravity_comp", "Hold the Axol in gravity-compensation mode."),
    "collect-data": ("collect_data", "Record teleoperation episodes."),
    "replay-dataset": (
        "replay_dataset",
        "Replay a recorded dataset episode on the robot.",
    ),
    "run-policy": ("run_policy", "Run a trained policy on the robot."),
    "inference-server": (
        "inference_server",
        "Serve policy inference for run-policy --server_host.",
    ),
}


def _dispatch_draccus(command: str, argv: list[str]) -> None:
    module_name, _ = _DRACCUS_COMMANDS[command]
    module = importlib.import_module(f".{module_name}", __name__)
    module.main(argv)


def main() -> None:
    """Dispatch ``axol <command>`` to the matching CLI handler."""
    # Load .env / .env.local (TURN credentials, etc.) before any command runs so
    # the values are in os.environ for in-process ops and child subprocesses.
    load_local_env()

    argv = sys.argv[1:]
    if argv and argv[0] in _DRACCUS_COMMANDS:
        _dispatch_draccus(argv[0], argv[1:])
        return
    if argv and argv[0] in _DIAG_COMMANDS:
        module_name, _ = _DIAG_COMMANDS[argv[0]]
        importlib.import_module(module_name).main(argv[1:])
        return

    parser = argparse.ArgumentParser(prog="axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    can_setup.add_parser(subparsers)
    can_enable.add_parser(subparsers)
    can_driver.add_parser(subparsers)
    set_can_id.add_parser(subparsers)
    set_zero_pos.add_parser(subparsers)
    motor_info.add_parser(subparsers)
    motor_health.add_parser(subparsers)
    zed_install.add_parser(subparsers)
    zed_driver.add_parser(subparsers)
    gst_install.add_parser(subparsers)
    gst_build_zed.add_parser(subparsers)
    provision_cmd.add_parser(subparsers)
    jetson_setup.add_parser(subparsers)
    pid.add_parser(subparsers)
    friction.add_parser(subparsers)
    repeatability.add_parser(subparsers)
    serve_cmd.add_parser(subparsers)

    # Register the draccus + diagnostics commands as bare subparsers purely so
    # they show up in ``axol --help``; their real parsing happens in the
    # interceptors above (``axol <cmd> --help`` is handled by the command's
    # own parser).
    for name, (_, help_text) in (*_DRACCUS_COMMANDS.items(), *_DIAG_COMMANDS.items()):
        subparsers.add_parser(name, help=help_text, add_help=False)

    args = parser.parse_args()
    args.func(args)
