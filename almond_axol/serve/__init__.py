"""Local web control panel + API server for the axol CLI.

``axol serve`` exposes a small FastAPI app that the bundled web UI talks to.
It wraps the existing CLI: each "run" spawns ``axol <command> ...`` as a
subprocess, streams its stdout to connected log WebSockets, and can stop it.
"""

from .app import create_app

__all__ = ["create_app"]
