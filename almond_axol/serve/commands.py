"""The CLI commands the web control panel can launch, and how to configure them.

Each command maps to a real ``axol <cli>`` invocation. Its full configuration
surface is introspected on demand from the command's draccus config dataclass
(see :mod:`.introspect`) — every nested field the CLI accepts is exposed to the
UI. ``collect-data`` / ``run-policy`` import lerobot, so their config classes
are loaded lazily and a command is simply marked unavailable if the import
fails (e.g. the ``lerobot`` extra isn't installed).
"""

from __future__ import annotations

from typing import Any, Callable

from .introspect import Schema, build_schema


class CommandDef:
    """A launchable command: its CLI name and a lazy loader for its config."""

    def __init__(
        self,
        id: str,
        cli: str,
        label: str,
        description: str,
        loader: Callable[[], type],
        *,
        sim_capable: bool = False,
        requires_hardware: bool = False,
    ) -> None:
        self.id = id
        self.cli = cli
        self.label = label
        self.description = description
        self.sim_capable = sim_capable
        self.requires_hardware = requires_hardware
        self._loader = loader

    def load_config_class(self) -> type:
        return self._loader()


def _load_teleop() -> type:
    from ..cli.config import TeleopCmdConfig

    return TeleopCmdConfig


def _load_gravity_comp() -> type:
    from ..cli.config import GravityCompCmdConfig

    return GravityCompCmdConfig


def _load_collect_data() -> type:
    from ..cli.collect_data import CollectDataConfig

    return CollectDataConfig


def _load_run_policy() -> type:
    from ..cli.run_policy import RunPolicyConfig

    return RunPolicyConfig


COMMANDS: dict[str, CommandDef] = {
    "teleop": CommandDef(
        "teleop",
        "teleop",
        "Teleoperation",
        "Drive the Axol from a VR headset. Enable simulation to preview in the "
        "browser without hardware.",
        _load_teleop,
        sim_capable=True,
    ),
    "gravity-comp": CommandDef(
        "gravity-comp",
        "gravity-comp",
        "Gravity compensation",
        "Hold the arms in gravity-comp so they can be moved by hand.",
        _load_gravity_comp,
        requires_hardware=True,
    ),
    "collect-data": CommandDef(
        "collect-data",
        "collect-data",
        "Collect data",
        "Record teleoperation episodes to a LeRobot dataset with the ZED cameras.",
        _load_collect_data,
        requires_hardware=True,
    ),
    "run-policy": CommandDef(
        "run-policy",
        "run-policy",
        "Run policy",
        "Run a trained policy on the robot via LeRobot async inference.",
        _load_run_policy,
        requires_hardware=True,
    ),
}


_schema_cache: dict[str, Schema] = {}


def get_schema(command_id: str) -> Schema:
    """Return (and memoize) the form schema for a command.

    May raise ``ImportError`` (lerobot missing) or other errors while building
    the config — callers that just want to list commands should catch those.
    """
    if command_id not in _schema_cache:
        cmd = COMMANDS[command_id]
        _schema_cache[command_id] = build_schema(cmd.load_config_class())
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


def _format_value(value: Any) -> str | None:
    """Render a submitted form value as a draccus CLI token (or omit it)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    if text == "":
        return None
    return text


def build_argv(command_id: str, args: dict[str, Any]) -> list[str]:
    """Translate submitted form values into a draccus-style argv tail.

    Only keys that exist as leaves in the command's schema are forwarded, so
    the UI can't inject arbitrary CLI arguments.
    """
    if command_id not in COMMANDS:
        raise KeyError(command_id)
    allowed = get_schema(command_id).leaf_keys

    argv: list[str] = []
    for key, raw in args.items():
        if key not in allowed:
            continue
        token = _format_value(raw)
        if token is None:
            continue
        argv.extend((f"--{key}", token))
    return argv
