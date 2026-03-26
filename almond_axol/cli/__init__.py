import argparse

from . import install_zed, setup_can, teleop


def main() -> None:
    parser = argparse.ArgumentParser(prog="almond-axol")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_can.add_parser(subparsers)
    teleop.add_parser(subparsers)
    install_zed.add_parser(subparsers)

    args = parser.parse_args()
    args.func(args)
