"""``python -m almond_axol.hub flash --port PORT``: the root half of a flash.

:func:`almond_axol.hub.firmware.ensure_hub_firmware` runs this through
``sudo`` when the bring-up itself is not root. Same arguments as
``axol hub.flash`` / ``axol hub.info``, with the bare verbs ``flash`` and
``info``.
"""

import argparse
import sys

from ..cli import hub as hub_cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m almond_axol.hub")
    hub_cli.add_parser(parser.add_subparsers(dest="command", required=True))
    argv = [f"hub.{a}" if a in ("flash", "info") else a for a in sys.argv[1:]]
    args = parser.parse_args(argv)
    args.func(args)
