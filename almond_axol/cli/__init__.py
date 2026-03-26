import argparse

from . import collect_data, install_zed, run_policy, setup_can, stream_zed, teleop


def main() -> None:
    parser = argparse.ArgumentParser(prog="almond-axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_can.add_parser(subparsers)
    collect_data.add_parser(subparsers)
    run_policy.add_parser(subparsers)
    teleop.add_parser(subparsers)
    stream_zed.add_parser(subparsers)
    install_zed.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
