"""
axol serve

Run the Axol web control panel: a small local server that wraps the CLI so the
robot can be driven from a browser instead of a terminal. It serves the built
web UI (when present) and a JSON/WebSocket API that launches, streams, and
stops ``axol`` commands as subprocesses.

    axol serve                  # serve on http://localhost:8001
    axol serve --port 9000
    axol serve --open           # also open a browser window on startup
    axol serve --host 127.0.0.1 # localhost only
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

from ..utils.certs import CERTFILE, KEYFILE, create_self_signed_cert

# The VR server and this control-panel API share one self-signed certificate
# (see ``almond_axol.utils.certs``) so a single browser cert acceptance covers both.


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``serve`` subcommand."""
    parser = subparsers.add_parser(
        "serve",
        help="Run the web control panel + API server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0, reachable on the LAN).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to listen on (default: 8001).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open a browser window on startup (off by default).",
    )
    parser.add_argument(
        "--no-tls",
        action="store_true",
        help=(
            "Serve plain HTTP instead of HTTPS. TLS is on by default so a "
            "browser on an HTTPS site (e.g. axol.almond.bot) can reach this "
            "machine without mixed-content blocking."
        ),
    )
    parser.set_defaults(func=run)


def _find_static_dir() -> Path | None:
    """Locate the built web bundle (web/app/dist), if it exists."""
    # almond_axol/cli/serve.py -> repo root is two parents up from the package.
    repo_root = Path(__file__).resolve().parents[2]
    dist = repo_root / "web" / "app" / "dist"
    return dist if (dist / "index.html").is_file() else None


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def run(args: argparse.Namespace) -> None:
    """Start the control-panel server."""
    import uvicorn

    from ..serve import create_app

    static_dir = _find_static_dir()
    app = create_app(static_dir)

    tls = not args.no_tls
    ssl_kwargs: dict[str, str] = {}
    if tls:
        _ensure_cert()
        ssl_kwargs = {"ssl_certfile": CERTFILE, "ssl_keyfile": KEYFILE}
    scheme = "https" if tls else "http"

    local = f"{scheme}://localhost:{args.port}"
    print("Axol control panel:")
    print(f"  Local : {local}")
    if args.host == "0.0.0.0":
        print(f"  LAN   : {scheme}://{_local_ip()}:{args.port}")
    if tls:
        print(
            "  (self-signed TLS — to connect from a browser on another machine, "
            "open the LAN URL once and accept the certificate; --no-tls disables)"
        )
    if static_dir is None:
        print(
            "  (no local web bundle — drive this machine from "
            "https://axol.almond.bot, or build one with `npm install && "
            "npm run build` in web/)"
        )

    if args.open:
        _open_browser_when_ready(local)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **ssl_kwargs)


def _ensure_cert() -> None:
    """Generate the shared self-signed cert on first use (idempotent)."""
    if os.path.isfile(CERTFILE) and os.path.isfile(KEYFILE):
        return
    print("Generating self-signed TLS certificate ...")
    create_self_signed_cert(CERTFILE, KEYFILE)


def _open_browser_when_ready(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
