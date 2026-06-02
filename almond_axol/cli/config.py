"""Shared draccus plumbing for the rich-config CLI commands.

The ``teleop``, ``gravity-comp``, ``collect-data``, and ``run-policy``
commands expose their full configuration via draccus, so every nested
field is reachable two ways:

- Dotted CLI overrides, lerobot-style: ``--axol.left.elbow.kp 200``,
  ``--robot_config.zed_host 10.0.0.5``.
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

import dataclasses
from dataclasses import MISSING, dataclass, field
from typing import Any, Literal, TypeVar, get_args

import draccus
import mergedeep
import numpy as np

from ..robot.config import AxolConfig
from ..shared import CAN_LEFT, CAN_RIGHT

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


def _default_overlay(config_class: type) -> dict[str, Any]:
    """Encode ``config_class``'s full default config into a nested dict.

    Required fields (no default and no ``default_factory``) are filled
    with ``None`` only so the instance can be constructed for encoding,
    then dropped from the overlay — the user must still supply them on the
    CLI or in ``--config_path`` (and draccus raises "missing required
    field" if they don't).
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


def parse(config_class: type[T], argv: list[str]) -> T:
    """Parse ``argv`` into ``config_class`` with full-default overlay.

    draccus auto-adds ``--config_path PATH`` for a whole-config JSON/YAML
    file; every nested field is also overridable via ``--dotted.path
    VALUE``. Unspecified fields fall back to the dataclass defaults.
    """
    overlay = _default_overlay(config_class)
    parser = _OverlayArgumentParser(config_class=config_class, overlay=overlay)
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
    """

    sim: bool = False
    axol: AxolConfig = field(default_factory=AxolConfig)
    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    log_level: LogLevel = "INFO"


@dataclass
class GravityCompCmdConfig:
    """Config for ``axol gravity-comp``.

    ``free_joints`` is a list of arm-joint names (e.g.
    ``[WRIST_3, ELBOW]``) to gravity-compensate; ``null`` (the default)
    frees all seven arm joints. Disable an arm with ``--left_channel
    null`` / ``--right_channel null``.
    """

    left_channel: str | None = CAN_LEFT
    right_channel: str | None = CAN_RIGHT
    free_joints: list[str] | None = None
    kd: float = 0.25
    rate_hz: float = 250.0
    telemetry_hz: float = 500.0
    log_level: LogLevel = "INFO"
