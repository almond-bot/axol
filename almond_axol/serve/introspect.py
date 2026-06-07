"""Turn a draccus command config dataclass into a UI form schema.

The CLI exposes the *entire* nested config via draccus dotted overrides
(``--axol.left.elbow.kp 60``). To render that in a web form we walk the
config's encoded default tree: ``draccus.encode(default_instance)`` flattens
every nested dataclass — including lerobot ``ChoiceRegistry`` subconfigs and
numpy fields (encoders registered in ``cli.config``) — into plain JSON
(dicts / lists / scalars). Dicts become collapsible groups; scalars become
fields whose type is inferred from the default value.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import MISSING
from typing import Any

import draccus

# Importing cli.config registers draccus's numpy + Literal codecs (needed so
# encode() of configs with ndarray / Literal fields doesn't blow up). It's a
# cheap, lerobot-free import.
from ..cli import config as _config  # noqa: F401

# Leaf fields whose allowed values we know up front, keyed by the leaf segment
# of the dotted path, so they render as dropdowns instead of free text.
_KNOWN_OPTIONS: dict[str, list[str]] = {
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR"],
    "policy_type": [
        "act",
        "smolvla",
        "diffusion",
        "tdmpc",
        "vqbet",
        "pi0",
        "pi05",
        "groot",
    ],
    "aggregate_fn": [
        "temporal_ensemble",
        "weighted_average",
        "latest_only",
        "average",
        "conservative",
    ],
}


class Schema:
    """A command's form schema plus the data needed to validate submissions."""

    def __init__(
        self, nodes: list[dict[str, Any]], leaf_keys: set[str], required: list[str]
    ) -> None:
        self.nodes = nodes
        self.leaf_keys = leaf_keys
        self.required = required


def _humanize(key: str) -> str:
    return key.replace("_", " ")


def _leaf_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def _make_node(prefix: str, key: str, value: Any, required: set[str]) -> dict[str, Any]:
    full = f"{prefix}.{key}" if prefix else key

    if isinstance(value, dict):
        return {
            "kind": "group",
            "key": full,
            "label": _humanize(key),
            "children": [_make_node(full, k, v, set()) for k, v in value.items()],
        }

    is_required = bool(prefix == "" and key in required)
    options = _KNOWN_OPTIONS.get(key)

    if options is not None:
        ftype = "select"
        default: Any = value
    elif isinstance(value, list):
        ftype = "text"
        default = json.dumps(value)
    else:
        ftype = _leaf_type(value)
        default = value

    return {
        "kind": "field",
        "key": full,
        "label": _humanize(key),
        "type": ftype,
        "default": None if is_required else default,
        "options": options,
        "required": is_required,
    }


def _collect_leaf_keys(nodes: list[dict[str, Any]], out: set[str]) -> None:
    for node in nodes:
        if node["kind"] == "group":
            _collect_leaf_keys(node["children"], out)
        else:
            out.add(node["key"])


def build_schema(config_class: type) -> Schema:
    """Build a form :class:`Schema` from a draccus command config dataclass.

    Required fields (no default) are encoded with a ``None`` sentinel so the
    instance can be built, then surfaced as required, value-less fields.
    """
    sentinel: dict[str, Any] = {}
    required: set[str] = set()
    for f in dataclasses.fields(config_class):
        if f.default is MISSING and f.default_factory is MISSING:
            sentinel[f.name] = None
            required.add(f.name)

    instance = config_class(**sentinel)
    encoded = draccus.encode(instance)
    if not isinstance(encoded, dict):  # pragma: no cover - configs are dataclasses
        raise TypeError(f"unexpected encoded config: {type(encoded)!r}")

    nodes = [_make_node("", k, v, required) for k, v in encoded.items()]
    leaf_keys: set[str] = set()
    _collect_leaf_keys(nodes, leaf_keys)
    return Schema(nodes=nodes, leaf_keys=leaf_keys, required=sorted(required))
