"""The curated catalog of ``axol`` CLI commands the web UIs expose.

Each command maps to a real ``axol <cli>`` invocation, grouped for the UI
(Operate / Diagnostics / Calibrate / Setup). Its configuration surface is
introspected on demand (see :mod:`.introspect`): draccus commands expose their
nested config dataclass; argparse commands expose their flags/options. Commands
whose imports fail (missing ``lerobot``, ZED SDK, mujoco, …) are simply marked
unavailable so the rest of the catalog still loads.

Only commands some UI surface actually launches belong here: the control
panel's five operations, and the diagnostics dashboard's tests, CAN bring-up
buttons, and motor calibration tools. Everything else — install-time commands
(``gst.*``, ``jetson.setup``, ``can.driver``), the tuning suite (``tune.*``),
one-off checks (``motor.info`` / ``motor.health``, whose read set the
dashboard's motor tiles show live), the remote ``inference-server``, and
``serve`` itself — stays CLI-only.

:data:`COMMANDS` is a registry rather than a fixed table: a package that builds
on ``almond-axol`` can :func:`register` its own commands before calling
:func:`~almond_axol.serve.create_app`, and they flow through the API and the
web panel like the built-in ones. Everything the serve layer needs to launch a
command — whether it runs in-process, what it does with cameras, which settings
it inherits — is declared on its :class:`CommandDef` rather than branched on
its id, so registration is the only integration point.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .introspect import Schema, build_argparse_schema, build_schema

# Display order for the catalog's category groups.
CATEGORY_ORDER = ["Operate", "Diagnostics", "Calibrate", "Setup"]


class CommandDef:
    """A launchable command and how to introspect its configuration.

    A command with an ``entrypoint`` is an *operation*: the serve layer runs it
    in-process (:class:`~almond_axol.serve.runner.OperationRunner`) so it shares
    the persistent robot connection, and the control panel gives it a panel of
    its own. Everything else is spawned as a ``python -m <module> <cli>``
    subprocess by :class:`~almond_axol.serve.manager.SessionManager`.

    An operation's entrypoint must match one of::

        def _run(cfg, *, stop_event=None, control=None) -> None:   # thread
        async def _run(cfg) -> None:                               # async

    ``control`` is passed only when ``episode_control`` is set; see that
    argument for the object's protocol.
    """

    def __init__(
        self,
        id: str,
        cli: str,
        label: str,
        description: str,
        category: str,
        kind: str,
        loader: Callable[[], Any],
        *,
        sim_capable: bool = False,
        requires_hardware: bool = False,
        uses_can_bus: bool = True,
        drives_motors: bool = False,
        entrypoint: Callable[[], Callable[..., Any]] | None = None,
        execution: str = "thread",
        requires_cameras: bool = False,
        camera_mode: str = "none",
        sim_flag: str | None = None,
        robot_free_flags: tuple[str, ...] = (),
        uses_headset: bool = False,
        episode_control: Callable[[], Callable[..., Any]] | None = None,
        per_run_fields: tuple[str, ...] = (),
        settings_like: str | None = None,
        module: str = "almond_axol",
    ) -> None:
        self.id = id
        self.cli = cli
        self.label = label
        self.description = description
        self.category = category
        self.kind = kind  # "draccus" | "argparse"
        self.sim_capable = sim_capable
        self.requires_hardware = requires_hardware
        # Whether launching this command needs sole ownership of the CAN bus.
        # A camera-only diagnostic (the ZED cable test) doesn't, so the serve
        # layer keeps the idle motor telemetry streaming while it runs.
        self.uses_can_bus = uses_can_bus
        # Motor-driving diagnostics must honor the same fault gate as core
        # operations. Connectivity repair commands intentionally leave this
        # false so they remain available when a motor cannot be reached.
        self.drives_motors = drives_motors
        # Lazy loader for the in-process entrypoint; None makes this a
        # subprocess command. Deferred like ``loader`` so a missing optional
        # extra only marks this one command unavailable.
        self._entrypoint = entrypoint
        # "thread" | "async". Async ops own an event loop for their whole run
        # (teleop's VR server, gravity-comp's telemetry) and are awaited;
        # thread ops are called synchronously on a worker thread.
        self.execution = execution
        # Needs at least one camera serial configured before it can start.
        self.requires_cameras = requires_cameras
        # How the operator's camera spec reaches the config: "argv" folds
        # serials into the argv-style args (the cameras are required draccus
        # inputs), "teleop" attaches them to a built config's camera dict
        # (unreachable via flat argv), "none" ignores it.
        self.camera_mode = camera_mode
        # Arg name that means "no hardware" for this op, so a sim run skips the
        # robot link and the motor-fault gate. None means it always needs the
        # robot.
        self.sim_flag = sim_flag
        # Arg names that mean "doesn't touch the arms" without being sim
        # (teleop's cart_only): the run skips the robot link and the
        # motor-fault gate but still drives real, non-arm hardware.
        self.robot_free_flags = robot_free_flags
        # Driven from the VR headset, so the panel tells the operator to point
        # the headset at this machine once the op is running.
        self.uses_headset = uses_headset
        # Lazy loader for an episode-control class, constructed as
        # ``cls(stop_event)`` and handed to the entrypoint as ``control``. It
        # must expose ``push(command: str)`` for API-pushed decisions and
        # ``snapshot() -> dict`` for the phase the panel renders.
        self._episode_control = episode_control
        # Config keys the panel surfaces per run; everything else comes from
        # the shared settings, folded in server-side.
        self.per_run_fields = per_run_fields
        # Borrow another op's settings targets. Settings are declared as dotted
        # config paths per op, so an op embedding the same config dataclasses
        # inherits the whole mapping instead of re-declaring it.
        self.settings_like = settings_like
        # ``python -m <module>`` target for the subprocess path, so a command
        # registered by a downstream package runs out of that package's CLI.
        self.module = module
        self._loader = loader

    @property
    def is_operation(self) -> bool:
        """Runs in-process (and gets a control-panel operation of its own)."""
        return self._entrypoint is not None

    @property
    def has_episode_control(self) -> bool:
        """Drives episodes the panel can save / rerecord / quit."""
        return self._episode_control is not None

    def load(self) -> Any:
        """Return the config class (draccus) or ``add_parser`` fn (argparse)."""
        return self._loader()

    def load_entrypoint(self) -> Callable[..., Any]:
        """Return the in-process ``_run`` callable (operations only)."""
        if self._entrypoint is None:
            raise ValueError(f"{self.id} is not an in-process operation")
        return self._entrypoint()

    def load_episode_control(self) -> Callable[..., Any] | None:
        """Return the episode-control class, or None when the op has none."""
        return None if self._episode_control is None else self._episode_control()


# -- draccus config-class loaders -------------------------------------------


def _teleop() -> type:
    from ..cli.config import TeleopCmdConfig

    return TeleopCmdConfig


def _gravity_comp() -> type:
    from ..cli.config import GravityCompCmdConfig

    return GravityCompCmdConfig


def _waypoints() -> type:
    from ..cli.waypoints import WaypointsCmdConfig

    return WaypointsCmdConfig


def _collect_data() -> type:
    from ..cli.collect_data import CollectDataConfig

    return CollectDataConfig


def _replay_dataset() -> type:
    from ..cli.replay_dataset import ReplayDatasetConfig

    return ReplayDatasetConfig


def _run_policy() -> type:
    from ..cli.run_policy import RunPolicyConfig

    return RunPolicyConfig


# -- in-process entrypoint loaders ------------------------------------------


def _teleop_run() -> Callable[..., Any]:
    from ..cli.teleop import _run

    return _run


def _gravity_comp_run() -> Callable[..., Any]:
    from ..cli.gravity_comp import _run

    return _run


def _waypoints_run() -> Callable[..., Any]:
    from ..cli.waypoints import _run

    return _run


def _waypoints_control() -> Callable[..., Any]:
    from ..cli.waypoints import _QueueWaypointControl

    return _QueueWaypointControl


def _collect_data_run() -> Callable[..., Any]:
    from ..cli.collect_data import _run

    return _run


def _replay_dataset_run() -> Callable[..., Any]:
    from ..cli.replay_dataset import _run

    return _run


def _run_policy_run() -> Callable[..., Any]:
    from ..cli.run_policy import _run

    return _run


def _run_policy_control() -> Callable[..., Any]:
    from ..cli.run_policy import _QueuePolicyControl

    return _QueuePolicyControl


# -- argparse add_parser loaders --------------------------------------------


def _argparse_loader(module: str, attr: str = "add_parser") -> Callable[[], Any]:
    def load() -> Any:
        import importlib

        # ``module`` is relative to this package (``almond_axol.serve``); e.g.
        # ``..cli.zed.install`` resolves to ``almond_axol.cli.zed.install``.
        mod = importlib.import_module(module, __package__)
        return getattr(mod, attr)

    return load


COMMANDS: dict[str, CommandDef] = {
    # -- Operate ------------------------------------------------------------
    "teleop": CommandDef(
        "teleop",
        "teleop",
        "Teleoperation",
        "Drive the Axol from a VR headset. Enable simulation to preview in the "
        "browser without hardware, or cart-only to drive just the powered cart.",
        "Operate",
        "draccus",
        _teleop,
        sim_capable=True,
        entrypoint=_teleop_run,
        execution="async",
        camera_mode="teleop",
        sim_flag="sim",
        robot_free_flags=("cart_only",),
        uses_headset=True,
        per_run_fields=("sim", "cart_only"),
    ),
    "gravity-comp": CommandDef(
        "gravity-comp",
        "gravity-comp",
        "Gravity compensation",
        "Hold the arms in gravity-comp so they can be moved by hand.",
        "Operate",
        "draccus",
        _gravity_comp,
        requires_hardware=True,
        entrypoint=_gravity_comp_run,
        execution="async",
        per_run_fields=("free_joints",),
    ),
    "waypoints": CommandDef(
        "waypoints",
        "waypoints",
        "Waypoints",
        "Hand-guide the arms in gravity comp to record waypoints, then replay "
        "them as straight-line moves solved with inverse kinematics. Enable "
        "simulation to preview a saved path in the browser.",
        "Operate",
        "draccus",
        _waypoints,
        sim_capable=True,
        entrypoint=_waypoints_run,
        episode_control=_waypoints_control,
        sim_flag="sim",
        # The gravity-comp side of a session takes the same config shape
        # (axol.*, channels, kd, rates), so the settings table is inherited.
        settings_like="gravity-comp",
        per_run_fields=("file", "loops", "play_only", "sim"),
    ),
    "collect-data": CommandDef(
        "collect-data",
        "collect-data",
        "Collect data",
        "Record teleoperation episodes to a LeRobot dataset with the local ZED cameras.",
        "Operate",
        "draccus",
        _collect_data,
        requires_hardware=True,
        entrypoint=_collect_data_run,
        requires_cameras=True,
        camera_mode="argv",
        per_run_fields=("repo_id", "task"),
    ),
    "replay-dataset": CommandDef(
        "replay-dataset",
        "replay-dataset",
        "Replay dataset",
        "Replay a recorded episode of a LeRobot dataset on the robot, then "
        "return to rest.",
        "Operate",
        "draccus",
        _replay_dataset,
        requires_hardware=True,
        entrypoint=_replay_dataset_run,
        per_run_fields=("repo_id", "episode", "loop", "interpolate"),
    ),
    "run-policy": CommandDef(
        "run-policy",
        "run-policy",
        "Run policy",
        "Run a trained policy on the robot via LeRobot async inference.",
        "Operate",
        "draccus",
        _run_policy,
        requires_hardware=True,
        entrypoint=_run_policy_run,
        requires_cameras=True,
        camera_mode="argv",
        episode_control=_run_policy_control,
        per_run_fields=("policy_path", "policy_type", "task", "repo_id"),
    ),
    # -- Diagnostics ----------------------------------------------------------
    "diag.rom-enable": CommandDef(
        "diag.rom-enable",
        "diag.rom-enable",
        "ROM enable",
        "Sweep every joint through its full range of motion for two hours; "
        "telemetry is captured for the dashboard.",
        "Diagnostics",
        "argparse",
        _argparse_loader("..diagnostics.rom.enable"),
        requires_hardware=True,
        drives_motors=True,
    ),
    "diag.rom-disable": CommandDef(
        "diag.rom-disable",
        "diag.rom-disable",
        "ROM disable",
        "Open the grippers left clamped by the ROM test and power down.",
        "Diagnostics",
        "argparse",
        _argparse_loader("..diagnostics.rom.disable"),
        requires_hardware=True,
        drives_motors=True,
    ),
    "diag.zed-cable": CommandDef(
        "diag.zed-cable",
        "diag.zed-cable",
        "ZED cable check",
        "Capture and validate frames from a connected ZED-X One camera to "
        "verify its GMSL cable. Camera-only — does not touch the arms.",
        "Diagnostics",
        "argparse",
        _argparse_loader("..diagnostics.zed.cable"),
        requires_hardware=True,
        uses_can_bus=False,
    ),
    # -- Calibrate ----------------------------------------------------------
    "motor.set-zero-pos": CommandDef(
        "motor.set-zero-pos",
        "motor.set-zero-pos",
        "Set zero position",
        "Set a motor's zero, or walk every joint with guided end-stop zeroing.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.motor.set_zero_pos"),
        requires_hardware=True,
        drives_motors=True,
    ),
    "motor.set-can-id": CommandDef(
        "motor.set-can-id",
        "motor.set-can-id",
        "Set CAN ID",
        "Change a motor's CAN ID and persist it to flash.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.motor.set_can_id"),
        requires_hardware=True,
    ),
    # -- Setup --------------------------------------------------------------
    "can.setup": CommandDef(
        "can.setup",
        "can.setup",
        "CAN setup",
        "Name the CAN interfaces and register a @reboot bring-up entry.",
        "Setup",
        "argparse",
        _argparse_loader("..cli.can.setup"),
        requires_hardware=True,
    ),
    "can.enable": CommandDef(
        "can.enable",
        "can.enable",
        "CAN enable",
        "Bring up the CAN interfaces using the saved startup script.",
        "Setup",
        "argparse",
        _argparse_loader("..cli.can.enable"),
        requires_hardware=True,
    ),
}


_schema_cache: dict[str, Schema] = {}


def register(command: CommandDef) -> None:
    """Add (or replace) a command in the catalog.

    The integration point for packages built on ``almond-axol``: register
    before :func:`~almond_axol.serve.create_app` and the command shows up in
    the API and the web panel alongside the built-in ones. Re-registering an
    id replaces it — a downstream package can substitute its own variant of a
    built-in operation — and drops the stale schema so the new config is
    introspected on next use.
    """
    COMMANDS[command.id] = command
    _schema_cache.pop(command.id, None)


def operation_ids() -> set[str]:
    """Ids of the commands the serve layer runs in-process."""
    return {cmd.id for cmd in COMMANDS.values() if cmd.is_operation}


def get_schema(command_id: str) -> Schema:
    """Return (and memoize) the form schema for a command.

    May raise ``ImportError`` (missing hardware extra) or other errors while
    building the config — callers listing commands should catch those.
    """
    if command_id not in _schema_cache:
        cmd = COMMANDS[command_id]
        loaded = cmd.load()
        if cmd.kind == "draccus":
            _schema_cache[command_id] = build_schema(loaded)
        else:
            _schema_cache[command_id] = build_argparse_schema(loaded)
    return _schema_cache[command_id]


def command_specs() -> list[dict[str, Any]]:
    """Serializable specs (including the full form schema) for every command."""
    specs: list[dict[str, Any]] = []
    for cmd in COMMANDS.values():
        spec: dict[str, Any] = {
            "id": cmd.id,
            "cli": cmd.cli,
            "label": cmd.label,
            "description": cmd.description,
            "category": cmd.category,
            "simCapable": cmd.sim_capable,
            "requiresHardware": cmd.requires_hardware,
            "usesCanBus": cmd.uses_can_bus,
            # Everything the panel needs to build an operation's tile and form
            # without knowing the command: older panels ignore these and fall
            # back to their built-in list.
            "isOperation": cmd.is_operation,
            "requiresCameras": cmd.requires_cameras,
            "perRunFields": list(cmd.per_run_fields),
            "episodeControl": cmd.has_episode_control,
            "simFlag": cmd.sim_flag,
            "robotFreeFlags": list(cmd.robot_free_flags),
            "usesHeadset": cmd.uses_headset,
        }
        try:
            schema = get_schema(cmd.id)
            spec["available"] = True
            spec["error"] = None
            spec["schema"] = schema.nodes
            spec["required"] = schema.required
        except Exception as exc:  # noqa: BLE001 - report any build failure to UI
            spec["available"] = False
            spec["error"] = f"{type(exc).__name__}: {exc}"
            spec["schema"] = []
            spec["required"] = []
        specs.append(spec)
    return specs


def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _format_value(value: Any) -> str | None:
    """Render a submitted form value as a CLI token (or omit it)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    if text == "":
        return None
    return text


def build_argv(command_id: str, args: dict[str, Any]) -> list[str]:
    """Translate submitted form values into an argv tail for the command.

    Each key is emitted per its schema recipe (see :class:`Schema`): dotted
    draccus options, argparse flags/options/lists, choice switches, and
    positionals. Leaves inside a dict-typed draccus field (``dictleaf``) are
    folded back into one inline ``--root {…}`` value, since draccus has no
    per-leaf options for dicts. Keys not present in the schema are ignored so
    the UI cannot inject arbitrary arguments.
    """
    if command_id not in COMMANDS:
        raise KeyError(command_id)
    emit = get_schema(command_id).emit

    options: list[str] = []
    positionals: list[str] = []
    # ``root`` -> nested value tree assembled from its submitted leaves.
    dict_values: dict[str, dict[str, Any]] = {}
    for key, raw in args.items():
        spec = emit.get(key)
        if spec is None:
            continue
        kind = spec["t"]
        if kind == "flag":
            if _truthy(raw):
                options.append(spec["flag"])
        elif kind == "flag_off":
            if not _truthy(raw):
                options.append(spec["flag"])
        elif kind == "choice":
            flag = spec["map"].get(str(raw).strip())
            if flag:
                options.append(flag)
        elif kind == "optlist":
            text = str(raw).strip()
            if text:
                options.extend([spec["flag"], *text.split()])
        elif kind == "pos":
            token = _format_value(raw)
            if token is not None:
                positionals.append(token)
        elif kind == "dictleaf":
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                continue
            node = dict_values.setdefault(spec["root"], {})
            parts = spec["sub"].split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = raw
        else:  # "opt"
            token = _format_value(raw)
            if token is not None:
                options.extend([spec["flag"], token])
    for root, tree in dict_values.items():
        # JSON is valid YAML, so draccus's value parser loads it as a dict and
        # deep-merges it over the dict field's defaults.
        options.extend([f"--{root}", json.dumps(tree)])
    return [*options, *positionals]
