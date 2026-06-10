"""Every ``axol`` CLI command the web control panel can launch.

Each command maps to a real ``axol <cli>`` invocation. Its configuration surface
is introspected on demand (see :mod:`.introspect`): draccus commands expose their
nested config dataclass; argparse commands expose their flags/options. Commands
whose imports fail (missing ``lerobot``, ZED SDK, mujoco, …) are simply marked
unavailable so the rest of the catalog still loads.

``serve`` itself is intentionally absent — it is the server hosting this UI.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .introspect import Schema, build_argparse_schema, build_schema

# Display order for the catalog's category groups.
CATEGORY_ORDER = ["Operate", "Cameras", "Calibrate", "Setup"]


class CommandDef:
    """A launchable command and how to introspect its configuration."""

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
    ) -> None:
        self.id = id
        self.cli = cli
        self.label = label
        self.description = description
        self.category = category
        self.kind = kind  # "draccus" | "argparse"
        self.sim_capable = sim_capable
        self.requires_hardware = requires_hardware
        self._loader = loader

    def load(self) -> Any:
        """Return the config class (draccus) or ``add_parser`` fn (argparse)."""
        return self._loader()


# -- draccus config-class loaders -------------------------------------------


def _teleop() -> type:
    from ..cli.config import TeleopCmdConfig

    return TeleopCmdConfig


def _gravity_comp() -> type:
    from ..cli.config import GravityCompCmdConfig

    return GravityCompCmdConfig


def _collect_data() -> type:
    from ..cli.collect_data import CollectDataConfig

    return CollectDataConfig


def _run_policy() -> type:
    from ..cli.run_policy import RunPolicyConfig

    return RunPolicyConfig


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
        "browser without hardware.",
        "Operate",
        "draccus",
        _teleop,
        sim_capable=True,
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
    ),
    # -- Cameras ------------------------------------------------------------
    "zed.install": CommandDef(
        "zed.install",
        "zed.install",
        "Install pyzed",
        "Download and install the pyzed wheel matching the installed ZED SDK.",
        "Cameras",
        "argparse",
        _argparse_loader("..cli.zed.install"),
        requires_hardware=True,
    ),
    # -- Calibrate ----------------------------------------------------------
    "tune.pid": CommandDef(
        "tune.pid",
        "tune.pid",
        "Tune PID",
        "Tune Kp/Kd for a single joint via sine or step tracking.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.tune.pid"),
        requires_hardware=True,
    ),
    "tune.friction": CommandDef(
        "tune.friction",
        "tune.friction",
        "Tune friction",
        "Identify the friction-model parameters (Fc, k, Fv, Fo) for one joint.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.tune.friction"),
        requires_hardware=True,
    ),
    "tune.repeatability": CommandDef(
        "tune.repeatability",
        "tune.repeatability",
        "Repeatability",
        "Drive between rest and a crossed-arms touching pose to measure repeatability.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.tune.repeatability"),
        requires_hardware=True,
    ),
    "motor.set-zero-pos": CommandDef(
        "motor.set-zero-pos",
        "motor.set-zero-pos",
        "Set zero position",
        "Set a motor's zero, or walk every joint with guided end-stop zeroing.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.motor.set_zero_pos"),
        requires_hardware=True,
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
    "motor.info": CommandDef(
        "motor.info",
        "motor.info",
        "Motor info",
        "Read a motor's status to verify it is reachable at a CAN ID.",
        "Calibrate",
        "argparse",
        _argparse_loader("..cli.motor.info"),
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
