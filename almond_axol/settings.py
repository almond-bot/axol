"""One robot, one settings file — shared by the control panel, the CLI, and the SDK.

The control panel persists every robot-level tunable (stiffness, per-joint
gains and link masses, gripper limits, teleop rates, CAN channels, …) in
``~/.almond/settings.json`` (``$ALMOND_HOME/settings.json``) and folds it into
every operation it launches. This module makes the *same* file the default
for the other two ways of driving the robot, so a value saved once applies
everywhere without being re-entered:

- The draccus CLI commands (``axol teleop``, ``gravity-comp``, ``collect-data``,
  …) read it by default through :func:`shared_overlay`, layered between the
  built-in defaults and ``--config_path`` / CLI flags. ``--no_settings``
  skips it; ``--settings_path PATH`` reads a different settings file.
- The SDK constructors (:class:`~almond_axol.robot.Axol`,
  :class:`~almond_axol.robot.Mantis`, :class:`~almond_axol.robot.Jelly`,
  :class:`~almond_axol.teleop.VRTeleop`) build their default configs through
  :func:`shared_config` when no explicit config is passed, and resolve their
  CAN channels through :func:`shared_can_channels` /
  :func:`shared_mantis_can_channels`. Pass a config or channel explicitly to
  override.

The file is a nested tree keyed by canonical sections (see
:mod:`almond_axol.serve.settings`): its ``axol``, ``teleop``, ``kinematics``,
``jelly`` and ``vr_server`` objects have exactly the shape of the matching
config dataclasses, so the same vocabulary serves the panel, a
``--config_path`` file and the SDK. The translation is exactly the control
panel's: every stored value is mapped to the operation's dotted config keys by
:meth:`SettingsStore.merged_args`, emitted as CLI tokens, and parsed the way
an op start parses them — so the three surfaces cannot disagree about a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .serve.settings import SettingsStore

__all__ = [
    "SHARED",
    "load_store",
    "shared_axol_config",
    "shared_can_channels",
    "shared_config",
    "shared_mantis_can_channels",
    "shared_overlay",
    "store_path",
]

T = TypeVar("T")


class _Shared:
    """Sentinel type for "resolve this argument from the shared settings"."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "SHARED"


SHARED: Any = _Shared()
"""Default for SDK channel arguments: read the value from the shared settings.

Channels use ``None`` to mean "this arm is disabled", so a distinct sentinel
marks "not given" — :class:`~almond_axol.robot.Axol` resolves it to the
``robot.left_channel`` / ``robot.right_channel`` settings, Mantis to the
``mantis.*`` ones. Pass a string or ``None`` explicitly to override.
"""


def store_path() -> Path:
    """Where the robot's settings live (``$ALMOND_HOME/settings.json``)."""
    from .serve import settings as serve_settings

    return serve_settings.SETTINGS_PATH


def load_store(path: str | Path | None = None, *, strict: bool = True) -> SettingsStore:
    """Open the shared settings (``~/.almond/settings.json`` unless ``path``).

    A missing file yields an empty store, so every loader below degrades to
    the built-in defaults on a fresh host. A file that exists but cannot be
    read (corrupt JSON, permissions, a symlinked path component) raises
    rather than quietly running the defaults — the caller asked for the
    robot's settings and must not get a different robot. Pass
    ``strict=False`` for serve-style tolerance. Imported lazily: the store
    lives in the serve package, which this module must stay cheap to import
    from.
    """
    from .serve import settings as serve_settings

    resolved = serve_settings.SETTINGS_PATH if path is None else Path(path)
    return serve_settings.SettingsStore(resolved, strict=strict)


def shared_overlay(
    op_id: str,
    args: dict[str, Any] | None = None,
    *,
    store: SettingsStore | None = None,
) -> dict[str, Any]:
    """The shared settings as a nested config overlay for one operation.

    Returns the nested dict :func:`almond_axol.cli.config.parse` merges above
    the dataclass defaults — the same values the control panel would fold
    into a start of ``op_id``. ``args`` are the request-style args the panel
    would send (only ``mantis`` / ``mantis_source`` change the fold: they
    select the Mantis rig channel map and the Quest tracker key). An
    operation the catalog doesn't know has no settings and yields ``{}``.

    The stored values go through ``build_argv`` and draccus's own string
    parser rather than straight into the overlay, so ``"null"`` channels,
    list-valued stiffness / rest poses and inline dict fields decode exactly
    as they do when the panel launches the operation.
    """
    from draccus import cfgparsing
    from draccus import utils as draccus_utils

    from .serve.commands import COMMANDS, build_argv

    if op_id not in COMMANDS:
        return {}
    if store is None:
        store = load_store()
    merged = store.merged_args(op_id, dict(args or {}))
    argv = build_argv(op_id, merged)
    flat: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--") and i + 1 < len(argv):
            flat[token[2:]] = cfgparsing.parse_string(argv[i + 1])
            i += 2
        else:  # pragma: no cover - draccus commands emit only ``--key value``
            i += 1
    return draccus_utils.deflatten(flat, sep=".")


def shared_config(
    config_class: type[T],
    op_id: str,
    prefix: str | None = None,
    *,
    store: SettingsStore | None = None,
) -> T:
    """Build ``config_class`` from its defaults plus the shared settings.

    ``prefix`` names the dotted subtree of ``op_id``'s config the class lives
    at (``"axol"`` for :class:`AxolConfig` on ``teleop``); ``None`` decodes
    the operation's whole config. This is what the SDK constructors call
    when no explicit config is given.
    """
    import mergedeep
    from draccus.parsers import decoding

    from .cli.config import _default_overlay

    overlay = shared_overlay(op_id, store=store)
    if prefix is not None:
        for part in prefix.split("."):
            overlay = overlay.get(part, {}) if isinstance(overlay, dict) else {}
    merged = mergedeep.merge({}, _default_overlay(config_class), overlay)
    return decoding.decode(config_class, merged)


def shared_axol_config(store: SettingsStore | None = None) -> Any:
    """The :class:`~almond_axol.robot.AxolConfig` the control panel would run.

    Stiffness, gripper limits, ``has_gripper`` and every per-joint gain /
    friction / mass / centre-of-mass saved in the panel (Robot tab and
    Advanced → Axol), over the calibrated defaults.
    """
    from .robot.config import AxolConfig

    return shared_config(AxolConfig, "teleop", "axol", store=store)


def shared_can_channels(
    store: SettingsStore | None = None,
) -> tuple[str | None, str | None]:
    """The Axol arms' (left, right) CAN interfaces from the shared settings.

    Unset values are the hub's persistent names; a saved ``null`` disables
    that arm, exactly as it does for the panel and the CLI.
    """
    if store is None:
        store = load_store()
    return store.can_channels()


def shared_mantis_can_channels(
    store: SettingsStore | None = None,
) -> tuple[str | None, str | None]:
    """The Mantis rig's (left, right) CAN interfaces from the shared settings."""
    if store is None:
        store = load_store()
    return store.mantis_can_channels()
