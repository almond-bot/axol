"""Shared framing and validation for Axol's custom policy interface.

Every WebSocket message between the robot and a custom policy endpoint is one
binary frame::

    uint32 big-endian N | N bytes of UTF-8 JSON header | raw payload

The header is an object whose ``"type"`` names the message. The message
schemas live in :mod:`almond_axol.policy.plan_protocol`; this module holds the
framing, limits, error types and the action-chunk coercion both ends share.
Everything here is pure numpy + JSON.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOL = "axol-policy"

MAX_HEADER_BYTES = 256 * 1024
# A few raw 1080p stereo frames fit comfortably; anything bigger is a bug.
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_ACTIONS_PER_CHUNK = 1024
MAX_DIMENSIONS = 256
MAX_CAMERAS = 31
MAX_NAME_BYTES = 128
MAX_TEXT_BYTES = 4096
MAX_ABSOLUTE_VALUE = 1_000_000.0
MAX_CAMERA_DIMENSION = 8192

_LENGTH = struct.Struct(">I")


class PolicyProtocolError(ValueError):
    """A custom-policy message is malformed or violates the protocol."""


class PolicyRemoteError(RuntimeError):
    """The policy answered a request with an ``error`` message."""


# ----------------------------------------------------------------------
# Framing
# ----------------------------------------------------------------------


def encode_message(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    """Frame one message: length-prefixed JSON header, then the raw payload."""
    try:
        header_bytes = json.dumps(
            dict(header), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PolicyProtocolError(f"Message header is not JSON: {exc}") from exc
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise PolicyProtocolError("Message header exceeds its byte limit.")
    total = _LENGTH.size + len(header_bytes) + len(payload)
    if total > MAX_MESSAGE_BYTES:
        raise PolicyProtocolError("Message exceeds its byte limit.")
    return _LENGTH.pack(len(header_bytes)) + header_bytes + payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyProtocolError(f"Duplicate header key {key!r}.")
        result[key] = value
    return result


def _reject_constant(constant: str) -> Any:
    raise PolicyProtocolError(f"Non-finite number {constant!r} in header.")


def decode_message(data: Any) -> tuple[dict[str, Any], memoryview]:
    """Split one framed message into its header dict and payload view."""
    if isinstance(data, str):
        raise PolicyProtocolError("Expected a binary WebSocket message, got text.")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PolicyProtocolError("Expected a binary message.")
    view = memoryview(data)
    if not _LENGTH.size < len(view) <= MAX_MESSAGE_BYTES:
        raise PolicyProtocolError("Message size is out of bounds.")
    (header_size,) = _LENGTH.unpack_from(view)
    if not 1 <= header_size <= MAX_HEADER_BYTES:
        raise PolicyProtocolError("Message header size is out of bounds.")
    end = _LENGTH.size + header_size
    if end > len(view):
        raise PolicyProtocolError("Message header is truncated.")
    try:
        header = json.loads(
            bytes(view[_LENGTH.size : end]).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except PolicyProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise PolicyProtocolError("Message header is not UTF-8 JSON.") from exc
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise PolicyProtocolError("Message header must be an object with a 'type'.")
    return header, view[end:]


# ----------------------------------------------------------------------
# Shared types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    """One camera the robot streams: its name and ``(height, width, 3)`` shape."""

    name: str
    shape: tuple[int, int, int]


def encode_error(message: str) -> bytes:
    # Keep the header bounded no matter how long the exception text is.
    return encode_message({"type": "error", "message": str(message)[:MAX_TEXT_BYTES]})


def as_action_chunk(value: Any, action_names: Sequence[str]) -> np.ndarray:
    """Coerce a policy's return value into a validated ``(T, D)`` float32 chunk.

    Accepts a ``(T, D)`` array-like (a torch tensor works via ``numpy()``), a
    single ``(D,)`` action, or a list of ``{action_name: value}`` dicts.
    """
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
        try:
            rows = [[float(row[name]) for name in action_names] for row in value]
        except KeyError as exc:
            raise PolicyProtocolError(
                f"Action dict is missing {exc.args[0]!r}."
            ) from None
        chunk = np.asarray(rows, dtype=np.float32)
    else:
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy) and not isinstance(value, np.ndarray):
            # torch tensors: detach/move to CPU first when that API exists.
            for method in ("detach", "cpu"):
                bound = getattr(value, method, None)
                if callable(bound):
                    value = bound()
            value = value.numpy()
        try:
            chunk = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise PolicyProtocolError(f"Action chunk is not numeric: {exc}") from None
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] != len(action_names):
        raise PolicyProtocolError(
            f"Action chunk must have shape (T, {len(action_names)}), got "
            f"{tuple(chunk.shape)}."
        )
    if not 1 <= chunk.shape[0] <= MAX_ACTIONS_PER_CHUNK:
        raise PolicyProtocolError(
            f"Action chunk must have 1-{MAX_ACTIONS_PER_CHUNK} rows, got "
            f"{chunk.shape[0]}."
        )
    if not np.isfinite(chunk).all() or np.abs(chunk).max() > MAX_ABSOLUTE_VALUE:
        raise PolicyProtocolError("Action chunk contains non-finite/huge values.")
    return chunk
