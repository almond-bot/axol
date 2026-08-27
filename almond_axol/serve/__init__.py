"""Local web control panel + API server for the axol CLI.

``axol serve`` exposes a small FastAPI app that the bundled web UI talks to.
The core operations (teleop, gravity-comp, collect-data, run-policy,
replay-dataset) run via :class:`~almond_axol.serve.runner.OperationRunner` —
the high-rate control ops (teleop, gravity-comp) in their own subprocess so
their control loops never share the serve interpreter, the rest in-process;
the remaining setup/calibration commands (``can.*``, ``motor.*``, ``tune.*``,
…) are spawned as ``axol <command>`` subprocesses by
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
"""

from .app import create_app
from .commands import CommandDef, operation_ids, register

__all__ = ["CommandDef", "create_app", "operation_ids", "register"]
