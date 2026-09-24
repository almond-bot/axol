"""Wire format for Axol's custom policy protocol.

A custom policy is any process that speaks this protocol over a WebSocket:
``axol run-policy --policy_type custom`` (the robot) connects to it, sends the
robot's joint state + camera frames, and executes the action chunks it sends
back. The Python side of both ends lives in :mod:`almond_axol.policy`, but the
format is small enough to implement in any language.

Every WebSocket message is one binary frame::

    uint32 big-endian N | N bytes of UTF-8 JSON header | raw payload

The header is an object whose ``"type"`` names the message. Only
``observation`` and ``actions`` carry a payload; the rest are header-only.

Robot → policy:

- ``hello`` opens a session: the robot's ordered ``state_names`` and
  ``action_names``, its ``cameras`` (``name`` + ``[height, width, 3]``
  shape), the control ``fps``, ``actions_per_chunk``, the ``task`` and,
  when the operator gave one, a ``policy_path`` (free-form: a checkpoint or
  model name for a server that hosts several).
- ``reset`` marks an episode boundary (``episode`` counts from 1).
- ``observation``: ``timestep``, ``timestamp`` (Unix seconds), ``task``,
  ``state`` (floats, ``state_names`` order) and ``images`` (``name`` +
  ``shape``). The payload is each image's uint8 RGB bytes in C order,
  concatenated in ``images`` order.

Policy → robot (exactly one reply per request):

- ``ready`` answers ``hello``: the ``action_names`` the policy emits (must
  equal the robot's exactly), plus an optional ``name`` and ``fps``.
- ``reset_ok`` answers ``reset``.
- ``actions`` answers ``observation``: the ``timestep`` it was predicted from
  and the chunk ``shape`` ``[T, D]``. The payload is ``T × D`` little-endian
  float32 values, row-major; row ``k`` is the command for timestep
  ``timestep + k``.
- ``error`` (``message``) answers any request the policy cannot serve; the
  robot stops the rollout.

Everything here is pure numpy + JSON (no LeRobot/torch), so a policy process
only needs ``almond-axol``'s base install — or none of it.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PROTOCOL = "axol-policy"
PROTOCOL_VERSION = 1

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
MAX_TIMESTAMP = 100_000_000_000.0

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
# Field validation
# ----------------------------------------------------------------------


def _text(value: Any, source: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PolicyProtocolError(f"{source}: expected a non-empty trimmed string.")
    if len(value.encode("utf-8")) > max_bytes:
        raise PolicyProtocolError(f"{source}: exceeds {max_bytes} bytes.")
    if not value.isprintable():
        raise PolicyProtocolError(f"{source}: contains control characters.")
    return value


def _names(value: Any, source: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_DIMENSIONS:
        raise PolicyProtocolError(f"{source}: expected 1-{MAX_DIMENSIONS} names.")
    names = tuple(
        _text(name, f"{source} entry", max_bytes=MAX_NAME_BYTES) for name in value
    )
    if len(set(names)) != len(names):
        raise PolicyProtocolError(f"{source}: names must be unique.")
    return names


def _int(value: Any, source: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PolicyProtocolError(
            f"{source}: expected an integer in [{minimum}, {maximum}]."
        )
    return value


def _float(value: Any, source: str, *, maximum: float = MAX_ABSOLUTE_VALUE) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyProtocolError(f"{source}: expected a number.")
    result = float(value)
    if not math.isfinite(result) or abs(result) > maximum:
        raise PolicyProtocolError(
            f"{source}: expected a finite number within ±{maximum:g}."
        )
    return result


def _keys(
    header: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required = frozenset(required) | {"type"}
    allowed = required | frozenset(optional)
    missing = sorted(required - header.keys())
    unknown = sorted(header.keys() - allowed)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise PolicyProtocolError(
            f"Malformed {header.get('type')!r} message: {'; '.join(detail)}."
        )


def _shape(value: Any, source: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(d) is not int for d in value)
    ):
        raise PolicyProtocolError(f"{source}: expected [height, width, channels].")
    height, width, channels = value
    if not (
        1 <= height <= MAX_CAMERA_DIMENSION
        and 1 <= width <= MAX_CAMERA_DIMENSION
        and channels in (1, 3)
    ):
        raise PolicyProtocolError(f"{source}: shape {value} is out of bounds.")
    return height, width, channels


def expect_type(header: Mapping[str, Any], expected: str) -> None:
    """Raise unless ``header`` is an ``expected`` message (or relay an error)."""
    kind = header.get("type")
    if kind == "error":
        message = header.get("message")
        raise PolicyRemoteError(
            message if isinstance(message, str) and message else "Policy error."
        )
    if kind != expected:
        raise PolicyProtocolError(f"Expected a {expected!r} message, got {kind!r}.")


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    """One camera the robot streams: its name and ``(height, width, 3)`` shape."""

    name: str
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class PolicySpec:
    """What the robot announced in ``hello`` — the contract for this session.

    Attributes:
        state_names: Order of :attr:`Observation.state` (e.g.
            ``left_shoulder_pitch.pos`` … ``right_gripper.pos``).
        action_names: Order of the columns every action chunk must have.
        cameras: The cameras each observation carries, in payload order.
        fps: Control rate; row ``k`` of a chunk executes ``k / fps`` seconds
            after the observation it was predicted from.
        actions_per_chunk: The most rows the robot will use from one chunk
            (extra rows are dropped).
        task: The task instruction the run was started with.
        robot: Robot type (``"axol"``).
        policy_path: The operator's ``--policy_path`` (``None`` when blank).
            Axol doesn't interpret it; use it to pick a checkpoint/model.
    """

    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    cameras: tuple[CameraSpec, ...]
    fps: int
    actions_per_chunk: int
    task: str
    robot: str = "axol"
    policy_path: str | None = None

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)


def encode_hello(spec: PolicySpec) -> bytes:
    extra = {} if spec.policy_path is None else {"policy_path": spec.policy_path}
    return encode_message(
        {
            **extra,
            "type": "hello",
            "protocol": PROTOCOL,
            "version": PROTOCOL_VERSION,
            "robot": spec.robot,
            "fps": spec.fps,
            "actions_per_chunk": spec.actions_per_chunk,
            "task": spec.task,
            "state_names": list(spec.state_names),
            "action_names": list(spec.action_names),
            "cameras": [
                {"name": camera.name, "shape": list(camera.shape)}
                for camera in spec.cameras
            ],
        }
    )


def decode_hello(header: Mapping[str, Any]) -> PolicySpec:
    expect_type(header, "hello")
    _keys(
        header,
        (
            "protocol",
            "version",
            "robot",
            "fps",
            "actions_per_chunk",
            "task",
            "state_names",
            "action_names",
            "cameras",
        ),
        ("policy_path",),
    )
    if header["protocol"] != PROTOCOL:
        raise PolicyProtocolError(f"Unknown protocol {header['protocol']!r}.")
    if header["version"] != PROTOCOL_VERSION or type(header["version"]) is not int:
        raise PolicyProtocolError(
            f"Robot speaks protocol version {header['version']!r}; this policy "
            f"server speaks {PROTOCOL_VERSION}. Upgrade almond-axol on both ends."
        )
    cameras_raw = header["cameras"]
    if not isinstance(cameras_raw, list) or len(cameras_raw) > MAX_CAMERAS:
        raise PolicyProtocolError(f"hello cameras: expected 0-{MAX_CAMERAS} entries.")
    cameras = []
    for raw in cameras_raw:
        if not isinstance(raw, dict) or set(raw) != {"name", "shape"}:
            raise PolicyProtocolError("hello camera: expected {name, shape}.")
        name = _text(raw["name"], "camera name", max_bytes=MAX_NAME_BYTES)
        cameras.append(CameraSpec(name, _shape(raw["shape"], f"camera {name!r}")))
    if len({camera.name for camera in cameras}) != len(cameras):
        raise PolicyProtocolError("hello cameras: names must be unique.")
    return PolicySpec(
        state_names=_names(header["state_names"], "state_names"),
        action_names=_names(header["action_names"], "action_names"),
        cameras=tuple(cameras),
        fps=_int(header["fps"], "fps", minimum=1, maximum=1000),
        actions_per_chunk=_int(
            header["actions_per_chunk"],
            "actions_per_chunk",
            minimum=1,
            maximum=MAX_ACTIONS_PER_CHUNK,
        ),
        task=_text(header["task"], "task"),
        robot=_text(header["robot"], "robot", max_bytes=MAX_NAME_BYTES),
        policy_path=(
            None
            if header.get("policy_path") is None
            else _text(header["policy_path"], "policy_path")
        ),
    )


@dataclass(frozen=True)
class ReadyInfo:
    """What the policy answered ``hello`` with."""

    action_names: tuple[str, ...]
    name: str | None = None
    fps: int | None = None


def encode_ready(info: ReadyInfo) -> bytes:
    header: dict[str, Any] = {"type": "ready", "action_names": list(info.action_names)}
    if info.name is not None:
        header["name"] = info.name
    if info.fps is not None:
        header["fps"] = info.fps
    return encode_message(header)


def decode_ready(header: Mapping[str, Any]) -> ReadyInfo:
    expect_type(header, "ready")
    _keys(header, ("action_names",), ("name", "fps"))
    name = header.get("name")
    fps = header.get("fps")
    return ReadyInfo(
        action_names=_names(header["action_names"], "action_names"),
        name=None if name is None else _text(name, "name", max_bytes=MAX_NAME_BYTES),
        fps=None if fps is None else _int(fps, "fps", minimum=1, maximum=1000),
    )


def encode_reset(episode: int) -> bytes:
    return encode_message({"type": "reset", "episode": episode})


def decode_reset(header: Mapping[str, Any]) -> int:
    expect_type(header, "reset")
    _keys(header, ("episode",))
    return _int(header["episode"], "episode", minimum=0, maximum=1_000_000_000)


RESET_OK = {"type": "reset_ok"}


def encode_error(message: str) -> bytes:
    # Keep the header bounded no matter how long the exception text is.
    return encode_message({"type": "error", "message": str(message)[:MAX_TEXT_BYTES]})


@dataclass
class Observation:
    """One robot observation, as handed to :meth:`Policy.infer`.

    Attributes:
        state: Joint/EE state, float32, in :attr:`PolicySpec.state_names` order.
        state_names: Names for each entry of ``state``.
        images: Camera name → ``(height, width, 3)`` uint8 RGB frame.
        task: Current task instruction (can change mid-episode with subtasks).
        timestep: Control tick this observation was taken at; the chunk you
            return starts executing here.
        timestamp: Unix time (seconds) the observation was taken.
    """

    state: np.ndarray
    state_names: tuple[str, ...]
    images: dict[str, np.ndarray]
    task: str
    timestep: int
    timestamp: float
    _state_dict: dict[str, float] | None = field(default=None, repr=False)

    @property
    def joints(self) -> dict[str, float]:
        """``state`` keyed by name, e.g. ``obs.joints["left_gripper.pos"]``."""
        if self._state_dict is None:
            self._state_dict = {
                name: float(value)
                for name, value in zip(self.state_names, self.state, strict=True)
            }
        return self._state_dict


def encode_observation(
    *,
    spec: PolicySpec,
    state: Sequence[float],
    images: Mapping[str, np.ndarray],
    task: str,
    timestep: int,
    timestamp: float,
) -> bytes:
    """Frame an observation in the camera/state order ``spec`` announced."""
    if len(state) != len(spec.state_names):
        raise PolicyProtocolError(
            f"Observation state has {len(state)} values, expected "
            f"{len(spec.state_names)}."
        )
    image_headers = []
    parts = []
    for camera in spec.cameras:
        image = images.get(camera.name)
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != camera.shape
        ):
            raise PolicyProtocolError(
                f"Camera {camera.name!r} frame must be uint8 with shape {camera.shape}."
            )
        image_headers.append({"name": camera.name, "shape": list(camera.shape)})
        parts.append(np.ascontiguousarray(image).tobytes(order="C"))
    return encode_message(
        {
            "type": "observation",
            "timestep": int(timestep),
            "timestamp": float(timestamp),
            "task": task,
            "state": [float(value) for value in state],
            "images": image_headers,
        },
        b"".join(parts),
    )


def decode_observation(
    header: Mapping[str, Any], payload: memoryview, spec: PolicySpec
) -> Observation:
    expect_type(header, "observation")
    _keys(header, ("timestep", "timestamp", "task", "state", "images"))
    state_raw = header["state"]
    if not isinstance(state_raw, list) or len(state_raw) != len(spec.state_names):
        raise PolicyProtocolError("Observation state width does not match hello.")
    state = np.array(
        [_float(value, "state value") for value in state_raw], dtype=np.float32
    )
    images_raw = header["images"]
    if not isinstance(images_raw, list) or len(images_raw) != len(spec.cameras):
        raise PolicyProtocolError("Observation camera count does not match hello.")
    images: dict[str, np.ndarray] = {}
    cursor = 0
    for raw, camera in zip(images_raw, spec.cameras, strict=True):
        if (
            not isinstance(raw, dict)
            or raw.get("name") != camera.name
            or raw.get("shape") != list(camera.shape)
            or set(raw) != {"name", "shape"}
        ):
            raise PolicyProtocolError(
                f"Observation camera {camera.name!r} does not match hello."
            )
        size = math.prod(camera.shape)
        if cursor + size > len(payload):
            raise PolicyProtocolError("Observation image data is truncated.")
        images[camera.name] = (
            np.frombuffer(payload[cursor : cursor + size], dtype=np.uint8)
            .reshape(camera.shape)
            .copy()
        )
        cursor += size
    if cursor != len(payload):
        raise PolicyProtocolError("Observation has trailing image bytes.")
    return Observation(
        state=state,
        state_names=spec.state_names,
        images=images,
        task=_text(header["task"], "task"),
        timestep=_int(header["timestep"], "timestep", minimum=0, maximum=10**9),
        timestamp=_float(header["timestamp"], "timestamp", maximum=MAX_TIMESTAMP),
    )


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


def encode_actions(chunk: np.ndarray, timestep: int) -> bytes:
    matrix = np.ascontiguousarray(chunk, dtype="<f4")
    return encode_message(
        {"type": "actions", "timestep": int(timestep), "shape": list(matrix.shape)},
        matrix.tobytes(order="C"),
    )


def decode_actions(
    header: Mapping[str, Any],
    payload: memoryview,
    action_names: Sequence[str],
) -> tuple[int, np.ndarray]:
    """Return ``(timestep, chunk)`` from an ``actions`` reply."""
    expect_type(header, "actions")
    _keys(header, ("timestep", "shape"))
    timestep = _int(header["timestep"], "timestep", minimum=0, maximum=10**9)
    shape = header["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(d) is not int for d in shape)
    ):
        raise PolicyProtocolError("actions shape: expected [T, D].")
    rows, width = shape
    if width != len(action_names):
        raise PolicyProtocolError(
            f"Action chunk has {width} columns, expected {len(action_names)}."
        )
    if not 1 <= rows <= MAX_ACTIONS_PER_CHUNK:
        raise PolicyProtocolError(
            f"Action chunk must have 1-{MAX_ACTIONS_PER_CHUNK} rows."
        )
    if len(payload) != rows * width * 4:
        raise PolicyProtocolError("Action chunk payload has the wrong byte length.")
    chunk = np.frombuffer(payload, dtype="<f4").reshape(rows, width).astype(np.float32)
    if not np.isfinite(chunk).all() or np.abs(chunk).max() > MAX_ABSOLUTE_VALUE:
        raise PolicyProtocolError("Action chunk contains non-finite/huge values.")
    return timestep, chunk
