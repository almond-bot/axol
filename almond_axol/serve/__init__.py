"""Local web control panel + API server for the axol CLI.

``axol serve`` exposes a small FastAPI app that the bundled web UI talks to.
The four core operations (teleop, gravity-comp, collect-data, run-policy) run
*in-process* via :class:`~almond_axol.serve.runner.OperationRunner`, sharing
one persistent robot connection; the remaining setup/calibration commands
(``can.*``, ``motor.*``, ``tune.*``, …) are spawned as ``axol <command>``
subprocesses by :class:`~almond_axol.serve.manager.SessionManager`. Either
way the output streams to connected log WebSockets and the run can be stopped.
"""

from .app import create_app

__all__ = ["create_app"]
