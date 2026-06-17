"""Shared draccus plumbing for the rich-config CLI commands.

The ``teleop``, ``gravity-comp``, ``collect-data``, and ``run-policy``
commands expose their full configuration via draccus, so every nested
field is reachable two ways:

- Dotted CLI overrides, lerobot-style: ``--axol.left.elbow.kp 200``.
  Dict-typed fields take one inline YAML/JSON value instead:
  ``--robot_config.cameras "{overhead: {serial: 41234567}}"``.
- A whole-config file: ``--config_path run.json`` (JSON or YAML), with
  CLI overrides layered on top.

This module provides the pieces shared by all four commands:

- :func:`parse`, a thin wrapper around :class:`draccus.ArgumentParser`
  that injects the *full* default config as the lowest-priority layer of
  the merge. draccus on its own builds a partially-specified nested
  dataclass from only the leaf(s) you override, which fails for configs
  whose nested dataclasses have required fields with per-instance
  defaults (see :class:`AxolConfig`'s seven differently-defaulted
  ``JointConfig`` fields). Seeding the encoded default config as the base
  of draccus's ``mergedeep`` step restores correct partial-override
  semantics (defaults -> ``--config_path`` file -> CLI flags).
- :data:`LogLevel` / :data:`PolicyType` / :data:`AggregateFn`, ``Literal``
  aliases registered with draccus so it validates choices the way
  ``argparse``'s ``choices=`` used to.
- :class:`TeleopCmdConfig` and :class:`GravityCompCmdConfig`, the two
  command configs that do not touch ``lerobot`` (kept here so importing
  them stays cheap on the sim/teleop-only path). The ``collect-data`` and
  ``run-policy`` configs live in their own command modules where the
  ``lerobot`` imports already belong.

This module intentionally imports no ``lerobot`` code so ``axol teleop
--sim`` keeps working in environments without the ``lerobot`` extra
installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal, TypeVar, get_args

import draccus
import mergedeep
import numpy as np

from ..kinematics.config import KinematicsConfig
from ..robot.config import AxolConfig
from ..teleop.config import VRTeleopConfig
from ..utils.shared import CAN_LEFT, CAN_RIGHT
from ..vr.config import VRServerConfig

T = TypeVar("T")


# ----------------------------------------------------------------------
# numpy.ndarray codec.
#
# draccus has no built-in encoder/decoder for ``numpy.ndarray`` (used by
# the VR teleop rest-pose fields). Encode to a plain list and decode back
# to a float32 array — the only ndarray config fields are joint vectors.
# ----------------------------------------------------------------------


@draccus.encode.register
def _encode_ndarray(arr: np.ndarray) -> list[float]:
    return arr.tolist()


draccus.decode.register(
    np.ndarray, lambda raw, _path=(): np.asarray(raw, dtype=np.float32)
)


# draccus 0.10.0's attribute-docstring extraction (used only to populate
# --help text) raises IndexError on some source layouts (e.g. a config
# whose last field is the final line of its module). The help text is
# cosmetic, so wrap the extractor to degrade gracefully to "no docstring"
# instead of crashing the whole parse.
from draccus.wrappers import docstring as _draccus_docstring  # noqa: E402

_orig_get_attribute_docstring = _draccus_docstring.get_attribute_docstring


def _safe_get_attribute_docstring(some_dataclass: type, field_name: str) -> Any:
    try:
        return _orig_get_attribute_docstring(some_dataclass, field_name)
    except Exception:  # noqa: BLE001
        return _draccus_docstring.AttributeDocString()


_draccus_docstring.get_attribute_docstring = _safe_get_attribute_docstring


# ----------------------------------------------------------------------
# Literal choice decoders.
#
# draccus 0.10.0 has no built-in decoder for ``typing.Literal`` and its
# registry only accepts concrete type objects (not the bare
# ``typing.Literal`` origin), so we register one decoder per concrete
# alias. Each rejects out-of-set values, mirroring argparse ``choices=``.
# ----------------------------------------------------------------------


def _register_literal(lit: T) -> T:
    """Register a draccus decoder for a concrete ``Literal[...]`` alias."""
    allowed = get_args(lit)

    def _decode(raw: Any, _path: Any = ()) -> Any:
        if raw not in allowed:
            raise ValueError(f"{raw!r} is not one of {list(allowed)}")
        return raw

    draccus.decode.register(lit, _decode)
    return lit


LogLevel = _register_literal(Literal["DEBUG", "INFO", "WARNING", "ERROR"])
PolicyType = _register_literal(
    Literal["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"]
)
AggregateFn = _register_literal(
    Literal[
        "temporal_ensemble",
        "weighted_average",
        "latest_only",
        "average",
        "conservative",
    ]
)


# ----------------------------------------------------------------------
# Parser: draccus + full-default overlay.
# ----------------------------------------------------------------------


def _strip_required_inputs(instance: Any, node: Any) -> None:
    """Recursively drop ``required_input``-marked fields from the overlay.

    A nested field can be made a *required user input* — no usable default,
    even when its parent dataclass is reachable via a ``default_factory`` —
    by marking it ``field(metadata={"required_input": True})``. Its owning
    factory still supplies a placeholder so the default config is
    constructible for encoding; removing the placeholder here keeps it out
    of the overlay so draccus raises "missing required field" unless the
    user supplies it (on the CLI or via ``--config_path``).
    """
    if not dataclasses.is_dataclass(instance) or not isinstance(node, dict):
        return
    for f in dataclasses.fields(instance):
        if f.metadata.get("required_input"):
            node.pop(f.name, None)
            continue
        value = getattr(instance, f.name, None)
        if dataclasses.is_dataclass(value) and isinstance(node.get(f.name), dict):
            _strip_required_inputs(value, node[f.name])
        elif isinstance(value, dict) and isinstance(node.get(f.name), dict):
            # Dicts of dataclasses (e.g. AxolRobotConfig.cameras) — recurse
            # into each entry so per-camera required fields (serial) are
            # stripped from the overlay too.
            for key, item in value.items():
                if dataclasses.is_dataclass(item) and isinstance(
                    node[f.name].get(key), dict
                ):
                    _strip_required_inputs(item, node[f.name][key])


def _default_overlay(config_class: type) -> dict[str, Any]:
    """Encode ``config_class``'s full default config into a nested dict.

    Required fields (no default and no ``default_factory``) are filled
    with ``None`` only so the instance can be constructed for encoding,
    then dropped from the overlay — the user must still supply them on the
    CLI or in ``--config_path`` (and draccus raises "missing required
    field" if they don't). Nested fields marked
    ``field(metadata={"required_input": True})`` are likewise dropped (see
    :func:`_strip_required_inputs`).
    """
    sentinel_kwargs: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(config_class):
        if f.default is MISSING and f.default_factory is MISSING:
            sentinel_kwargs[f.name] = None
            required.append(f.name)
    instance = config_class(**sentinel_kwargs)
    overlay = draccus.encode(instance)
    for name in required:
        overlay.pop(name, None)
    _strip_required_inputs(instance, overlay)
    return overlay


class _OverlayArgumentParser(draccus.argparsing.ArgumentParser):  # type: ignore[misc]
    """``draccus.ArgumentParser`` that seeds the full default config.

    Overrides ``_postprocessing`` to deep-merge in ``self._overlay`` as
    the lowest-priority layer (below the ``--config_path`` file and below
    the explicit CLI flags), so a single deep override like
    ``--axol.left.elbow.kp 200`` keeps the elbow's other per-joint
    defaults instead of demanding the whole ``JointConfig``. Kept faithful
    to draccus 0.10.0's own ``_postprocessing`` (pinned in pyproject).
    """

    def __init__(self, *args: Any, overlay: dict[str, Any], **kwargs: Any) -> None:
        self._overlay = overlay
        super().__init__(*args, **kwargs)

    def _postprocessing(self, parsed_args: Any) -> Any:
        import warnings

        from draccus import cfgparsing, utils
        from draccus.parsers import decoding

        parsed_arg_values = vars(parsed_args)
        for key in parsed_arg_values:
            parsed_value = cfgparsing.parse_string(parsed_arg_values[key])
            if isinstance(parsed_value, str) and parsed_value.startswith("include"):
                with open(parsed_value[len("include ") :]) as f:
                    parsed_arg_values[key] = cfgparsing.load_config(f)
            else:
                parsed_arg_values[key] = parsed_value

        config_path = self.config_path
        if utils.CONFIG_ARG in parsed_arg_values:
            new_config_path = parsed_arg_values[utils.CONFIG_ARG]
            if config_path is not None:
                warnings.warn(
                    UserWarning(
                        f"Overriding default {config_path} with {new_config_path}"
                    ),
                    stacklevel=2,
                )
            config_path = new_config_path
            del parsed_arg_values[utils.CONFIG_ARG]

        if config_path is not None:
            with open(config_path) as f:
                file_args = cfgparsing.load_config(f, file=config_path)
        else:
            file_args = {}

        deflat_d = utils.deflatten(parsed_arg_values, sep=".")
        # Precedence (later wins): defaults -> --config_path file -> CLI.
        deflat_d = mergedeep.merge({}, self._overlay, file_args, deflat_d)
        return decoding.decode(self.config_class, deflat_d)


# Per-joint arm fields (``kp`` / ``kd`` / ``friction.*`` / ``mass`` / ``com``
# / ``j_eff`` / ``kd_soft`` for the seven arm joints). For a config that
# embeds ``AxolConfig`` these are ~140 of the ~165 generated options and
# flood ``--help`` into illegibility. Matched anywhere in a dotted option
# string so it works for both ``--axol.left.elbow.kp`` (teleop) and
# ``--robot_config.axol_config.left.elbow.kp`` (collect-data / run-policy).
_JOINT_FIELD_RE = re.compile(
    r"\.(shoulder_1|shoulder_2|shoulder_3|elbow|wrist_1|wrist_2|wrist_3)\."
)

# draccus auto-generates a ``--<name> str`` "Config file for <name>" include
# option for every nested dataclass *type* (e.g. ``--axol``, ``--left``,
# ``--shoulder_1``, ``--friction``, ``--gripper``). They duplicate the single
# top-level ``--config_path`` at every level of the tree and add nothing but
# noise to ``--help``. ``--config_path`` itself is help "Path for a config
# file ..." so it's not caught by this prefix.
_INCLUDE_HELP_PREFIX = "Config file for "

# Clean, accurate help for the handful of nested fields kept visible in
# ``--help`` (keyed by the option's leaf segment, so it covers both
# ``--axol.left_stiffness`` and ``--robot_config.axol_config.left_stiffness``).
# draccus's inline-docstring extraction mis-renders some of these as raw
# source (e.g. the ``left_stiffness`` line dumps the ``field(...)`` defaults),
# so we override them outright.
_FIELD_HELP: dict[str, str] = {
    "left_stiffness": (
        "Compliance<->stiffness blend in [0, 1]: a scalar (all arm joints) "
        "or a 7-element list, one per joint."
    ),
    "right_stiffness": (
        "Compliance<->stiffness blend in [0, 1]: a scalar (all arm joints) "
        "or a 7-element list, one per joint."
    ),
    "max_step_rad": "Max change (rad) in any arm joint between consecutive commands.",
    "torque_limit": "Peak gripper output torque (Nm) in POSITION_FORCE mode.",
    "max_speed": "Max gripper joint speed (rad/s).",
    "serial": "Required. Serial number of the ZED camera to open.",
}

# Substrings that mark draccus inline help as mis-extracted source code.
_GARBLED_HELP_MARKERS = ("field(", "default_factory", "def ", "lambda")


def _is_help_noise(action: argparse.Action) -> bool:
    """True if ``action`` should be hidden from ``--help`` (still parseable).

    ``argparse.SUPPRESS`` on ``action.help`` only affects the help listing;
    the option is still parsed normally, so every field stays overridable.
    Hides the per-joint arm gains and draccus's per-nested-dataclass config-
    file include options, leaving the handful of common top-level / stiffness
    / gripper fields visible.
    """
    if any(_JOINT_FIELD_RE.search(opt) for opt in action.option_strings):
        return True
    return (action.help or "").startswith(_INCLUDE_HELP_PREFIX)


def _condense_help(ap: argparse.ArgumentParser) -> None:
    """Trim a draccus-built parser's ``--help`` down to the common fields.

    A config that embeds :class:`AxolConfig` expands to ~165 options across
    ~36 argument groups; the per-joint gains and draccus's per-dataclass
    "Config file for X" includes make ``--help`` unreadable. This:

    - Suppresses the noisy options (see :func:`_is_help_noise`). draccus
      registers the include options on the argument *groups* but not on
      ``parser._actions``, so both are scanned.
    - Drops the per-nested-dataclass section docstrings and any section
      left with no visible option, so only the command summary plus the
      common top-level / stiffness / gripper fields remain.

    Purely cosmetic: every suppressed field is still fully overridable on
    the CLI. A no-op for configs without nested dataclasses (e.g.
    ``gravity-comp``), whose help is already short.
    """
    actions: dict[int, argparse.Action] = {id(a): a for a in ap._actions}
    for group in ap._action_groups:
        for a in group._group_actions:
            actions[id(a)] = a

    suppressed = 0
    for a in actions.values():
        if _is_help_noise(a):
            a.help = argparse.SUPPRESS
            suppressed += 1
    if not suppressed:
        return

    # Clean up the help shown for the fields that remain visible.
    for a in actions.values():
        if a.help == argparse.SUPPRESS:
            continue
        opt = next((o for o in a.option_strings if o.startswith("--")), "")
        leaf = opt.lstrip("-").split(".")[-1]
        if leaf in _FIELD_HELP:
            a.help = _FIELD_HELP[leaf]
        elif any(marker in (a.help or "") for marker in _GARBLED_HELP_MARKERS):
            a.help = None

    for group in ap._action_groups:
        # Nested-dataclass groups are titled like ``AxolConfig ['axol']`` /
        # ``JointConfig ['axol.left.elbow']``; their docstrings are the bulk
        # of the noise. The top command-config group (no ``[`` in the title)
        # keeps its docstring as the command summary.
        if "[" in (group.title or ""):
            group.description = None
        if not any(a.help != argparse.SUPPRESS for a in group._group_actions):
            group.title = None
            group.description = None

    ap.epilog = (
        "Only common fields are shown above. Every nested config field is "
        "still overridable from the CLI — e.g. per-joint gains like "
        "--axol.left.elbow.kp 60 (or --robot_config.axol_config.* for "
        "collect-data / run-policy) — or load a whole-config file with "
        "--config_path. Full reference: "
        "https://docs.almond.bot/cli/configuration"
    )


def parse(config_class: type[T], argv: list[str]) -> T:
    """Parse ``argv`` into ``config_class`` with full-default overlay.

    draccus auto-adds ``--config_path PATH`` for a whole-config JSON/YAML
    file; every nested field is also overridable via ``--dotted.path
    VALUE``. Unspecified fields fall back to the dataclass defaults.

    Deeply-nested per-joint gains and draccus's per-dataclass config-file
    includes are hidden from ``--help`` (but remain fully overridable) so
    the listing stays scannable; an epilog points at the full reference.
    """
    overlay = _default_overlay(config_class)
    parser = _OverlayArgumentParser(config_class=config_class, overlay=overlay)
    _condense_help(parser.parser)
    try:
        return parser.parse_args(argv)
    except (draccus.ParsingError, draccus.utils.DecodingError) as exc:
        # Surface config errors (missing required field, bad choice, type
        # mismatch) as a clean usage error instead of a traceback.
        # draccus wraps the underlying argparse parser as ``parser.parser``.
        parser.parser.error(str(exc))


# ----------------------------------------------------------------------
# Command configs that don't touch lerobot (kept import-cheap for sim).
# ----------------------------------------------------------------------


@dataclass
class TeleopCmdConfig:
    """Config for ``axol teleop``.

    Runs on the real Axol robot by default; pass ``--sim`` to drive the
    browser visualizer instead (the ``axol`` config and CAN channels are
    ignored in sim). Gripper torque limits and the compliance/stiffness
    blend live on the nested ``axol`` config — override them via
    ``--axol.left.gripper.torque_limit`` and ``--axol.left_stiffness``.
    Disable an arm with ``--left_channel null`` / ``--right_channel null``.
    Teleop session parameters (e.g. the position multiplier) live on the
    nested ``teleop`` config — e.g. ``--teleop.position_multiplier 2.0``.
    IK solver cost weights live on the nested ``kinematics`` config — e.g.
    ``--kinematics.pos_weight 100`` or ``--kinematics.max_joint_delta 0.02``.

    Map camera slots to local ZED serial numbers via ``--cameras`` to also
    stream them to the headset (overhead as the main feed, wrist cameras
    switched with the right thumbstick) — e.g. ``--cameras "{overhead:
    41234567, left_arm: 41234568}"``. Requires the ZED SDK and pyzed
    (``axol zed.install``) plus the ``video`` extra installed locally.

    A stereo ZED X overhead is detected automatically from its serial: its
    two eyes are relayed as ``overhead_left`` / ``overhead_right`` and
    rendered per-lens in the headset for true stereo. ``--resolution`` picks
    the capture resolution for all cameras (``SVGA`` / ``HD1080`` /
    ``HD1200``); ``null`` (the default) keeps each camera's SDK default.

    The VR WebSocket server (port, TLS certs) lives on the nested
    ``vr_server`` config — e.g. ``--vr_server.port 9000``.
    """

    sim: bool = False
    axol: AxolConfig = field(default_factory=AxolConfig)
    teleop: VRTeleopConfig = field(default_factory=VRTeleopConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    vr_server: VRServerConfig = field(default_factory=VRServerConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    cameras: dict[str, int] = field(default_factory=dict)
    resolution: str | None = None
    log_level: LogLevel = "INFO"


@dataclass
class GravityCompCmdConfig:
    """Config for ``axol gravity-comp``.

    ``free_joints`` is a list of arm-joint names (e.g.
    ``[WRIST_3, ELBOW]``) to gravity-compensate; ``null`` (the default)
    frees all seven arm joints. Disable an arm with ``--left_channel
    null`` / ``--right_channel null``.

    The gravity feed-forward torque (per-joint mass / centre-of-mass) and
    the impedance gains used to hold non-free joints both come from the
    nested ``axol`` config — override them via e.g.
    ``--axol.left.elbow.kp 60`` or ``--axol.left_stiffness 0.8``.
    """

    axol: AxolConfig = field(default_factory=AxolConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    free_joints: list[str] | None = None
    kd: float = 0.25
    rate_hz: float = 250.0
    telemetry_hz: float = 500.0
    log_level: LogLevel = "INFO"
