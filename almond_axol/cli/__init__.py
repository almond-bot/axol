import argparse

from . import (
    collect_data,
    identify_friction,
    install_zed,
    motor_info,
    run_policy,
    set_can_id,
    setup_can,
    stream_zed,
    teleop,
    tune_pid,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="almond-axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_can.add_parser(subparsers)
    set_can_id.add_parser(subparsers)
    motor_info.add_parser(subparsers)
    collect_data.add_parser(subparsers)
    run_policy.add_parser(subparsers)
    teleop.add_parser(subparsers)
    stream_zed.add_parser(subparsers)
    install_zed.add_parser(subparsers)
    tune_pid.add_parser(subparsers)
    identify_friction.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
