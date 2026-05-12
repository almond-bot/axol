"""CLI entry point for the axol command registered via pyproject.toml."""

import argparse

from . import collect_data, gravity_comp, run_policy, teleop
from .can import enable as can_enable
from .can import setup as can_setup
from .motor import info as motor_info
from .motor import set_can_id, set_zero_pos
from .tune import friction, pid, repeatability
from .zed import install as zed_install
from .zed import stream as zed_stream


def main() -> None:
    parser = argparse.ArgumentParser(prog="axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    can_setup.add_parser(subparsers)
    can_enable.add_parser(subparsers)
    set_can_id.add_parser(subparsers)
    set_zero_pos.add_parser(subparsers)
    motor_info.add_parser(subparsers)
    collect_data.add_parser(subparsers)
    gravity_comp.add_parser(subparsers)
    run_policy.add_parser(subparsers)
    teleop.add_parser(subparsers)
    zed_stream.add_parser(subparsers)
    zed_install.add_parser(subparsers)
    pid.add_parser(subparsers)
    friction.add_parser(subparsers)
    repeatability.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
