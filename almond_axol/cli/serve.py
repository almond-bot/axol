"""
axol serve

Run the Axol web control panel: a small local server that wraps the CLI so the
robot can be driven from a browser instead of a terminal. It serves the built
web UI (when present) and a JSON/WebSocket API that launches robot operations,
streams their output, and stops them. The four core operations (teleop,
gravity-comp, collect-data, run-policy) run in-process so they share one robot
connection; the setup/calibration commands run as ``axol`` subprocesses.

    axol serve                  # serve on https://localhost:8001
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

from ..utils.certs import CERTFILE, KEYFILE, PreparedTLSFiles, prepare_tls_files

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
    parser.add_argument(
        "--operator",
        metavar="USER",
        help=(
            "Non-root account that may read datasets recorded by a manual root "
            "serve (default: SUDO_USER). Ignored by non-root serves."
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
    # Explicit process marker: security gates must remain active for a manual
    # root ``axol serve`` even when the installer did not set ALMOND_HOME.
    from ..utils.state_files import (
        configure_root_service_dataset,
        mark_privileged_service,
    )

    mark_privileged_service()
    if os.geteuid() == 0:
        # Third-party dataset writers must never create group-writable entries
        # inside the immutable hosted store, including on manual serve runs
        # outside the installed systemd unit.
        os.umask(0o027)
        configure_root_service_dataset(getattr(args, "operator", None))

    import uvicorn

    from ..serve import create_app

    tls = not args.no_tls
    scheme = "https" if tls else "http"
    lan_ip = _local_ip() if args.host in {"0.0.0.0", "::"} else args.host

    static_dir = _find_static_dir()
    app = create_app(static_dir)

    local = f"{scheme}://localhost:{args.port}"
    print("Axol control panel:")
    print(f"  Local : {local}")
    if args.host == "0.0.0.0":
        print(f"  LAN   : {scheme}://{lan_ip}:{args.port}")
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

    # Bind the port ourselves — reclaiming it from a stale/leftover listener —
    # before handing the socket to uvicorn, so a restart (or a crashed previous
    # instance that didn't release the socket) doesn't fail with "address
    # already in use". uvicorn still installs its own SIGINT/SIGTERM handlers
    # and closes the adopted socket on exit.
    from ..utils.ports import open_listen_socket

    tls_files: PreparedTLSFiles | None = None
    sock = open_listen_socket(args.host, args.port)
    try:
        ssl_kwargs: dict[str, str] = {}
        if tls:
            tls_files = prepare_tls_files(CERTFILE, KEYFILE)
            if tls_files.generated:
                print("Generating self-signed TLS certificate ...")
            ssl_kwargs = {
                "ssl_certfile": tls_files.certfile,
                "ssl_keyfile": tls_files.keyfile,
            }
        config = uvicorn.Config(
            app, host=args.host, port=args.port, log_level="info", **ssl_kwargs
        )
        server = uvicorn.Server(config)
        server.run(sockets=[sock])
    finally:
        sock.close()
        if tls_files is not None:
            tls_files.close()


def _open_browser_when_ready(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
