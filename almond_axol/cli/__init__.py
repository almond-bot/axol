import argparse

from . import (
    collect_data,
    enable_can,
    identify_feedforward,
    install_zed,
    motor_info,
    run_policy,
    set_can_id,
    set_zero_pos,
    setup_can,
    stream_zed,
    teleop,
    tune_pid,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="almond-axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_can.add_parser(subparsers)
    enable_can.add_parser(subparsers)
    set_can_id.add_parser(subparsers)
    set_zero_pos.add_parser(subparsers)
    motor_info.add_parser(subparsers)
    collect_data.add_parser(subparsers)
    run_policy.add_parser(subparsers)
    teleop.add_parser(subparsers)
    stream_zed.add_parser(subparsers)
    install_zed.add_parser(subparsers)
    tune_pid.add_parser(subparsers)
    identify_feedforward.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
