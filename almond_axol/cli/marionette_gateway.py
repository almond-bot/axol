"""``axol marionette.gateway``: run the Marionette gateway on this computer.

The gateway the phones connect to ships inside the Marionette SDK (the
``marionette`` extra), with a Node runtime, so the Host Computer needs no copy
of the Marionette repository. Every argument is the gateway's own:
``--port``, ``--name``, ``--auth``, ``--token``, ``--no-mdns`` and so on
(``axol marionette.gateway --help``).
"""

from __future__ import annotations


def main(argv: list[str]) -> None:
    try:
        from marionette import gateway
    except ImportError:
        raise SystemExit(
            "Marionette isn't installed. Install it with: "
            "pip install 'almond-axol[marionette]'"
        ) from None
    raise SystemExit(gateway.main(argv))
