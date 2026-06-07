"""
axol serve

Run the Axol web control panel: a small local server that wraps the CLI so the
robot can be driven from a browser instead of a terminal. It serves the built
web UI (when present) and a JSON/WebSocket API that launches, streams, and
stops ``axol`` commands as subprocesses.

    axol serve                  # http://localhost:8090, opens a browser
    axol serve --port 9000
    axol serve --no-open        # don't auto-open a browser
    axol serve --host 127.0.0.1 # localhost only
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path


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
        default=8090,
        help="Port to listen on (default: 8090).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser window on startup.",
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

    local = f"http://localhost:{args.port}"
    print("Axol control panel:")
    print(f"  Local : {local}")
    if args.host == "0.0.0.0":
        print(f"  LAN   : http://{_local_ip()}:{args.port}")
    if static_dir is None:
        print(
            "  (web UI not built — run `npm install && npm run build` in web/; "
            "the API is still available)"
        )

    if not args.no_open:
        _open_browser_when_ready(local)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _open_browser_when_ready(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
