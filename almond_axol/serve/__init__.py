"""Local web control panel + API server for the axol CLI.

``axol serve`` exposes a small FastAPI app that the bundled web UI talks to.
The core operations (teleop, gravity-comp, collect-data, run-policy,
replay-dataset) run *in-process* via
:class:`~almond_axol.serve.runner.OperationRunner`, sharing one persistent
robot connection; the remaining setup/calibration commands (``can.*``,
``motor.*``, ``tune.*``, …) are spawned as ``axol <command>`` subprocesses by
:class:`~almond_axol.serve.manager.SessionManager`. Either way the output
streams to connected log WebSockets and the run can be stopped.

Which commands exist is a registry, not a fixed list. A package built on
``almond-axol`` adds its own by calling :func:`register` before
:func:`create_app`; they then appear in the API and the web panel like the
built-ins, with no changes needed here or in ``web/``::

    from almond_axol.serve import CommandDef, create_app, register

    register(CommandDef(
        id="my-op", cli="my-op", label="My op", description="…",
        category="Operate", kind="draccus", loader=lambda: MyOpConfig,
        entrypoint=lambda: my_op_run, requires_hardware=True,
        per_run_fields=("repo_id", "task"), settings_like="collect-data",
    ))
    app = create_app(static_dir)

See :class:`~almond_axol.serve.commands.CommandDef` for the full set of
declarations and the entrypoint protocol.

A package can also extend the app itself — extra ``/api/...`` routes or a
page it serves — with :func:`register_app_extension`, whose callable runs
inside ``create_app`` ahead of the web bundle's SPA catch-all, and advertise
such a page to the panel's navigation with :func:`register_page` (reported
by ``GET /api/info`` as ``pages``; see :mod:`almond_axol.serve.extensions`)::

    from almond_axol.serve import PanelPage, register_app_extension, register_page

    register_app_extension(lambda app: app.include_router(my_router))
    register_page(PanelPage("My page", "/my-page"))
"""

from .app import create_app
from .commands import CommandDef, operation_ids, register
from .extensions import PanelPage, register_app_extension, register_page

__all__ = [
    "CommandDef",
    "PanelPage",
    "create_app",
    "operation_ids",
    "register",
    "register_app_extension",
    "register_page",
]
