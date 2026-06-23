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

import argparse
import dataclasses
import inspect
import json
import re
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
    "dataset_resolution": ["SVGA", "HD1080", "HD1200"],
    "eyes": ["both", "left", "right"],
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
    """A command's form schema plus the data needed to rebuild its argv.

    ``emit`` maps each leaf key to a recipe describing how that form value is
    turned into CLI tokens (see :func:`build_argv`). draccus commands use dotted
    ``--section.field value`` options; argparse commands use their own flags,
    store_true switches, positionals, and choice groups.
    """

    def __init__(
        self,
        nodes: list[dict[str, Any]],
        required: list[str],
        emit: dict[str, dict[str, Any]],
    ) -> None:
        self.nodes = nodes
        self.required = required
        self.emit = emit


def _humanize(key: str) -> str:
    return key.replace("_", " ")


_SECTION_HEADERS = {"attributes", "args", "arguments", "parameters"}
_DOC_FIELD_RE = re.compile(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_RST_ROLE_RE = re.compile(r":[a-z]+:`([^`]*)`")


def _clean_doc(text: str) -> str:
    """Strip reST roles / backticks and collapse whitespace for a tooltip."""
    text = _RST_ROLE_RE.sub(r"\1", text)
    text = text.replace("``", "").replace("`", "").replace("**", "")
    return " ".join(text.split())


def _parse_doc_fields(doc: str) -> dict[str, str]:
    """Parse a Google/NumPy-style ``Attributes:`` / ``Args:`` block.

    Returns ``{field_name: help}``. Entries start at a fixed indent as
    ``name: text``; more-indented lines are continuations of the current field.
    """
    lines = doc.splitlines()
    out: dict[str, str] = {}
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.endswith(":") and stripped[:-1].lower() in _SECTION_HEADERS:
            i += 1
            entry_indent: int | None = None
            name: str | None = None
            parts: list[str] = []
            while i < n:
                line = lines[i]
                if not line.strip():
                    i += 1
                    continue
                indent = len(line) - len(line.lstrip())
                if entry_indent is None:
                    entry_indent = indent
                if indent < entry_indent:
                    break
                match = _DOC_FIELD_RE.match(line.strip())
                if indent == entry_indent and match:
                    if name is not None:
                        out[name] = _clean_doc(" ".join(parts))
                    name = match.group(1)
                    parts = [match.group(2)]
                elif name is not None:
                    parts.append(line.strip())
                i += 1
            if name is not None:
                out[name] = _clean_doc(" ".join(parts))
        else:
            i += 1
    return out


# Curated leaf help and garbled-source markers mirror the CLI ``--help``
# (see :mod:`almond_axol.cli.config`); reuse them so UI tooltips match it.
_CURATED_FIELD_HELP: dict[str, str] = getattr(_config, "_FIELD_HELP", {})
_GARBLED_HELP_MARKERS: tuple[str, ...] = getattr(
    _config, "_GARBLED_HELP_MARKERS", ("field(", "default_factory", "def ", "lambda")
)


def _attribute_help(cls: type, name: str) -> str | None:
    """Per-field help from draccus's attribute-docstring extraction.

    Mirrors how the CLI sources nested-field help: an inline / preceding
    comment or an attribute docstring. Mis-extracted source (draccus sometimes
    dumps the ``field(...)`` literal) is dropped, matching ``--help``.
    """
    try:
        from draccus.wrappers import docstring as _dd

        ad = _dd.get_attribute_docstring(cls, name)
    except Exception:  # noqa: BLE001 - help is cosmetic
        return None
    for cand in (ad.docstring_below, ad.comment_inline, ad.comment_above):
        text = (cand or "").strip()
        if text and not any(marker in text for marker in _GARBLED_HELP_MARKERS):
            return _clean_doc(text)
    return None


def _field_docs(instance: Any) -> dict[str, str]:
    """Per-field help for a dataclass, sourced like the CLI ``--help``.

    Precedence (later wins): class ``Attributes:`` / ``Args:`` docstring →
    draccus attribute docstrings (inline/preceding comments) → the CLI's
    curated overrides.
    """
    if not dataclasses.is_dataclass(instance):
        return {}
    cls = type(instance)
    out: dict[str, str] = {}
    doc = inspect.getdoc(cls)
    if doc:
        out.update(_parse_doc_fields(doc))
    for field in dataclasses.fields(cls):
        if field.name not in out:
            comment = _attribute_help(cls, field.name)
            if comment:
                out[field.name] = comment
        curated = _CURATED_FIELD_HELP.get(field.name)
        if curated:
            out[field.name] = curated
    return out


def _leaf_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def _children(
    prefix: str,
    values: dict[str, Any],
    instance: Any,
    required: set[str],
    dict_roots: set[str],
) -> list[dict[str, Any]]:
    """Build the child nodes for a group, pulling per-field help from ``instance``."""
    docs = _field_docs(instance)
    out: list[dict[str, Any]] = []
    is_dc = dataclasses.is_dataclass(instance)
    for key, value in values.items():
        if is_dc:
            child = getattr(instance, key, None)
        elif isinstance(instance, dict):
            child = instance.get(key)
        else:
            child = None
        out.append(
            _make_node(prefix, key, value, required, child, docs.get(key), dict_roots)
        )
    return out


def _make_node(
    prefix: str,
    key: str,
    value: Any,
    required: set[str],
    instance: Any,
    help_text: str | None,
    dict_roots: set[str],
) -> dict[str, Any]:
    full = f"{prefix}.{key}" if prefix else key

    if isinstance(value, dict):
        # A dict-typed field (e.g. AxolRobotConfig.cameras) has no per-leaf
        # draccus options — its leaves must be folded back into one inline
        # ``--<root> {…}`` value. Record the outermost dict boundary.
        if isinstance(instance, dict) and not any(
            full == r or full.startswith(r + ".") for r in dict_roots
        ):
            dict_roots.add(full)
        return {
            "kind": "group",
            "key": full,
            "label": _humanize(key),
            "help": help_text,
            "children": _children(full, value, instance, set(), dict_roots),
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
        "help": help_text,
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

    dict_roots: set[str] = set()
    nodes = _children("", encoded, instance, required, dict_roots)
    leaf_keys: set[str] = set()
    _collect_leaf_keys(nodes, leaf_keys)
    # A leaf is a dotted ``--key value`` override — except inside dict-typed
    # fields, where draccus only accepts a single inline value for the whole
    # dict; those leaves are folded back together at argv-build time.
    emit: dict[str, dict[str, Any]] = {}
    for key in leaf_keys:
        root = next((r for r in dict_roots if key.startswith(r + ".")), None)
        if root is not None:
            emit[key] = {
                "t": "dictleaf",
                "flag": f"--{root}",
                "root": root,
                "sub": key[len(root) + 1 :],
            }
        else:
            emit[key] = {"t": "opt", "flag": f"--{key}"}
    return Schema(nodes=nodes, required=sorted(required), emit=emit)


# ---------------------------------------------------------------------------
# argparse commands
# ---------------------------------------------------------------------------


def build_argparse_schema(add_parser: Any) -> Schema:
    """Build a form :class:`Schema` from an argparse subcommand.

    ``add_parser`` is the module's ``add_parser(subparsers)`` registrar; we run
    it against a throwaway parser, then introspect the resulting subparser's
    actions. Each flag/option/positional becomes a flat field (no nesting), and
    a required mutually-exclusive group of switches (e.g. ``--l`` / ``--r``)
    collapses into a single required dropdown.
    """
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_parser(sub)
    cmd_parser = next(iter(sub.choices.values()))

    nodes: list[dict[str, Any]] = []
    required: list[str] = []
    emit: dict[str, dict[str, Any]] = {}

    grouped: dict[int, argparse._MutuallyExclusiveGroup] = {}
    for group in cmd_parser._mutually_exclusive_groups:
        for action in group._group_actions:
            grouped[id(action)] = group
    handled_groups: set[int] = set()

    for action in cmd_parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        group = grouped.get(id(action))
        if group is not None:
            if id(group) in handled_groups:
                continue
            handled_groups.add(id(group))
            node, key, spec, is_req = _make_group_field(group)
        else:
            node, key, spec, is_req = _make_action_field(action)
        if node is None:
            continue
        nodes.append(node)
        emit[key] = spec
        if is_req:
            required.append(key)

    return Schema(nodes=nodes, required=required, emit=emit)


def _make_group_field(
    group: argparse._MutuallyExclusiveGroup,
) -> tuple[dict[str, Any] | None, str, dict[str, Any], bool]:
    """Collapse a mutually-exclusive switch group into one dropdown field."""
    actions = group._group_actions
    dests = [a.dest for a in actions]
    flags = {a.dest: a.option_strings[0] for a in actions if a.option_strings}

    # The common left/right arm selector gets friendly labels.
    if set(dests) == {"l", "r"}:
        key, label = "arm", "Arm"
        options = ["left", "right"]
        value_to_flag = {"left": flags["l"], "right": flags["r"]}
    else:
        key = "__" + "_".join(dests)
        label = " / ".join(dests)
        options = dests
        value_to_flag = {d: flags[d] for d in dests}

    required = bool(group.required)
    node = {
        "kind": "field",
        "key": key,
        "label": label,
        "type": "select",
        "default": None if required else "",
        "options": options,
        "required": required,
        "help": None,
    }
    return node, key, {"t": "choice", "map": value_to_flag}, required


def _make_action_field(
    action: argparse.Action,
) -> tuple[dict[str, Any] | None, str, dict[str, Any], bool]:
    key = action.dest
    label = _humanize(key)
    required = bool(getattr(action, "required", False))
    help_text = action.help or None

    # store_true / store_false switches → boolean toggle.
    if isinstance(action, argparse._StoreTrueAction):
        node = _field(
            key, label, "boolean", bool(action.default), None, False, help_text
        )
        return node, key, {"t": "flag", "flag": action.option_strings[0]}, False
    if isinstance(action, argparse._StoreFalseAction):
        node = _field(
            key, label, "boolean", bool(action.default), None, False, help_text
        )
        return node, key, {"t": "flag_off", "flag": action.option_strings[0]}, False

    options = list(action.choices) if action.choices else None
    ftype = "select" if options else _arg_value_type(action)
    is_list = action.nargs in ("+", "*")
    default = _arg_default(action, ftype, is_list, required)

    if not action.option_strings:
        # Positional argument.
        node = _field(key, label, ftype, default, options, required, help_text)
        return node, key, {"t": "pos"}, required

    flag = action.option_strings[0]
    spec = {"t": "optlist", "flag": flag} if is_list else {"t": "opt", "flag": flag}
    node = _field(key, label, ftype, default, options, required, help_text)
    return node, key, spec, required


def _field(
    key: str,
    label: str,
    ftype: str,
    default: Any,
    options: list[str] | None,
    required: bool,
    help_text: str | None,
) -> dict[str, Any]:
    return {
        "kind": "field",
        "key": key,
        "label": label,
        "type": ftype,
        "default": None if required else default,
        "options": options,
        "required": required,
        "help": help_text,
    }


def _arg_value_type(action: argparse.Action) -> str:
    if action.type is int or action.type is float:
        return "number"
    return "text"


def _arg_default(action: argparse.Action, ftype: str, is_list: bool, req: bool) -> Any:
    if req:
        return None
    value = action.default
    if value is None:
        return None
    if is_list and isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if ftype == "boolean":
        return bool(value)
    if ftype == "number":
        return value
    return str(value)
