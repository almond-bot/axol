"""Read the configuration dialect emitted by libsurvive's native writer.

Despite its ``config.json`` filename, ``config_save`` writes root-level members
without enclosing braces or commas before/between lighthouse groups. Values
inside those groups are JSON. Keep the file itself unchanged: this parser only
checks that a calibration transaction reads and publishes complete content.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid configuration value: {value}")


def parse_survive_config(text: str) -> dict[str, Any]:
    """Accept a JSON object or native rootless members, rejecting partial files.

    Only the native writer's root-level group separators are relaxed. Nested
    arrays and objects must remain complete JSON; no input is repaired or
    reserialized before it is passed back to libsurvive.
    """
    text = text.strip()
    if not text:
        raise ValueError("Native configuration is empty")
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_members, parse_constant=_reject_constant
    )
    if text.startswith("{"):
        return decoder.decode(text)

    def skip_space(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    members: list[tuple[str, Any]] = []
    position = 0
    comma = False
    while position < len(text):
        key, position = decoder.raw_decode(text, position)
        if not isinstance(key, str) or not key:
            raise ValueError("Expected a quoted configuration key")
        position = skip_space(position)
        if text[position : position + 1] != ":":
            raise ValueError(f"Expected ':' after configuration key {key}")
        position = skip_space(position + 1)
        value, position = decoder.raw_decode(text, position)
        if members and not comma:
            if not re.fullmatch(r"lighthouse\d+", key) or not isinstance(value, dict):
                raise ValueError(f"Missing comma before configuration key {key}")
        members.append((key, value))
        position = skip_space(position)
        comma = text[position : position + 1] == ","
        if comma:
            position = skip_space(position + 1)
            if position == len(text):
                raise ValueError("Native configuration ends with a comma")
    return _unique_members(members)


def native_seed(contents: bytes | None) -> bytes:
    """The bytes to start a native run from, given a saved configuration.

    survive-cli segfaults on a file holding only an empty object (``{}``), while
    an empty file starts it from scratch exactly like a missing one. Seed a
    fresh calibration with no content instead.
    """
    if contents is None or not contents.strip():
        return b""
    if parse_survive_config(contents.decode("utf-8")) == {}:
        return b""
    return contents
