"""Safe, bounded wire codecs for LeRobot async inference.

LeRobot 0.6.1 sends Python pickles in all three opaque protobuf byte fields.
Those payloads cross a plaintext, unauthenticated gRPC connection, so unpickling
them would give either peer code execution.  Axol uses explicit framed JSON and
raw little-endian numeric/image buffers instead.  Every field, shape, dtype,
name, count, and total byte size is checked before constructing local LeRobot
objects.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable, Iterator, Mapping
from numbers import Real
from typing import Any

import numpy as np
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.transport.utils import CHUNK_SIZE, TransferState

OBSERVATION_PROTOCOL = "almond-axol-observation"
ACTION_PROTOCOL = "almond-axol-actions"
INFERENCE_WIRE_VERSION = 1

MAX_WIRE_HEADER_BYTES = 64 * 1024
MAX_OBSERVATION_WIRE_BYTES = 64 * 1024 * 1024
MAX_ACTION_WIRE_BYTES = 2 * 1024 * 1024
MAX_TASK_BYTES = 4096
MAX_TIMESTEP = 1_000_000_000
MAX_ACTIONS_PER_CHUNK = 1024
MAX_ACTION_DIMENSIONS = 256
MAX_ABSOLUTE_NUMERIC_VALUE = 1_000_000.0

_OBSERVATION_MAGIC = b"AXOLOBS1"
_ACTION_MAGIC = b"AXOLACT1"
_FRAME_PREFIX = struct.Struct(">8sI")


class InferenceWireError(ValueError):
    """An inference wire payload is malformed or incompatible."""


class InferenceWireSizeError(InferenceWireError):
    """An inference payload or chunk exceeded its hard byte limit."""


def _bounded_text(value: Any, source: str, *, max_bytes: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InferenceWireError(
            f"Malformed {source}: expected a non-empty string without surrounding whitespace."
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InferenceWireError(f"Malformed {source}: invalid Unicode.") from exc
    if len(encoded) > max_bytes:
        raise InferenceWireError(
            f"Malformed {source}: exceeds the {max_bytes}-byte limit."
        )
    if any(not character.isprintable() for character in value):
        raise InferenceWireError(f"Malformed {source}: contains control characters.")
    return value


def _exact_keys(value: Any, expected: frozenset[str], source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceWireError(f"Malformed {source}: expected an object.")
    keys = frozenset(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise InferenceWireError(f"Malformed {source}: {'; '.join(detail)}.")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceWireError(f"Malformed wire header: duplicate key {key!r}.")
        result[key] = value
    return result


def _wire_int(value: Any, source: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InferenceWireError(
            f"Malformed {source}: expected an integer in [{minimum}, {maximum}]."
        )
    return value


def _wire_float(
    value: Any,
    source: str,
    *,
    minimum: float = -MAX_ABSOLUTE_NUMERIC_VALUE,
    maximum: float = MAX_ABSOLUTE_NUMERIC_VALUE,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InferenceWireError(f"Malformed {source}: expected a finite number.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise InferenceWireError(
            f"Malformed {source}: expected a finite number."
        ) from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise InferenceWireError(
            f"Malformed {source}: expected a finite value in [{minimum}, {maximum}]."
        )
    return result


def _ordered_names(value: Any, source: str, *, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise InferenceWireError(
            f"Malformed {source}: expected 1-{maximum} ordered names."
        )
    names = tuple(
        _bounded_text(name, f"{source} name", max_bytes=128) for name in value
    )
    if len(names) != len(set(names)):
        raise InferenceWireError(f"Malformed {source}: names must be unique.")
    return names


def _observation_layout(
    features: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, int, int]], ...]]:
    """Extract the already-negotiated state and camera layout."""
    if not isinstance(features, Mapping):
        raise InferenceWireError("Malformed negotiated observation features.")
    state = features.get("observation.state")
    if not isinstance(state, Mapping):
        raise InferenceWireError("Negotiated observation.state feature is missing.")
    state_names = _ordered_names(
        state.get("names"), "negotiated state schema", maximum=MAX_ACTION_DIMENSIONS
    )
    shape = state.get("shape")
    if (
        state.get("dtype") != "float32"
        or not isinstance(shape, (list, tuple))
        or tuple(shape) != (len(state_names),)
    ):
        raise InferenceWireError("Malformed negotiated observation.state feature.")

    cameras: list[tuple[str, tuple[int, int, int]]] = []
    prefix = "observation.images."
    for feature_key, spec in features.items():
        if feature_key == "observation.state":
            continue
        if (
            not isinstance(feature_key, str)
            or not feature_key.startswith(prefix)
            or not isinstance(spec, Mapping)
        ):
            raise InferenceWireError("Malformed negotiated image feature.")
        camera_name = _bounded_text(
            feature_key.removeprefix(prefix), "camera name", max_bytes=128
        )
        raw_shape = spec.get("shape")
        if (
            spec.get("dtype") != "image"
            or not isinstance(raw_shape, (list, tuple))
            or len(raw_shape) != 3
            or any(type(dimension) is not int for dimension in raw_shape)
        ):
            raise InferenceWireError(f"Malformed negotiated camera {camera_name!r}.")
        camera_shape = tuple(raw_shape)
        height, width, channels = camera_shape
        if not (1 <= height <= 8192 and 1 <= width <= 8192 and channels in (1, 3)):
            raise InferenceWireError(f"Malformed negotiated camera {camera_name!r}.")
        cameras.append((camera_name, camera_shape))

    if not cameras or len(cameras) > 31:
        raise InferenceWireError("Negotiated observation requires 1-31 cameras.")
    if len(cameras) != len({name for name, _ in cameras}):
        raise InferenceWireError("Negotiated camera names must be unique.")
    collisions = {name for name, _ in cameras} & (set(state_names) | {"task"})
    if collisions or "task" in state_names:
        raise InferenceWireError(
            "Negotiated state/camera names collide with reserved raw keys."
        )
    return state_names, tuple(cameras)


def _encode_frame(
    magic: bytes,
    header: dict[str, Any],
    payload: bytes,
    *,
    maximum: int,
) -> bytes:
    try:
        header_bytes = json.dumps(
            header,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InferenceWireError("Wire header contains non-JSON data.") from exc
    if not 1 <= len(header_bytes) <= MAX_WIRE_HEADER_BYTES:
        raise InferenceWireSizeError("Wire header exceeds its byte limit.")
    total = _FRAME_PREFIX.size + len(header_bytes) + len(payload)
    if total > maximum:
        raise InferenceWireSizeError(f"Wire payload exceeds its {maximum}-byte limit.")
    return _FRAME_PREFIX.pack(magic, len(header_bytes)) + header_bytes + payload


def _decode_frame(
    data: Any, magic: bytes, *, maximum: int
) -> tuple[dict[str, Any], memoryview]:
    if not isinstance(data, bytes) or not _FRAME_PREFIX.size < len(data) <= maximum:
        raise InferenceWireSizeError(f"Wire payload must fit within {maximum} bytes.")
    actual_magic, header_size = _FRAME_PREFIX.unpack_from(data)
    if actual_magic != magic:
        raise InferenceWireError("Wire payload has the wrong protocol magic.")
    if not 1 <= header_size <= MAX_WIRE_HEADER_BYTES:
        raise InferenceWireSizeError("Wire header exceeds its byte limit.")
    payload_offset = _FRAME_PREFIX.size + header_size
    if payload_offset > len(data):
        raise InferenceWireError("Wire header is truncated.")
    try:
        header = json.loads(
            data[_FRAME_PREFIX.size : payload_offset].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                InferenceWireError(f"Malformed wire header constant {constant!r}.")
            ),
        )
    except InferenceWireError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InferenceWireError("Wire header is not valid UTF-8 JSON.") from exc
    if not isinstance(header, dict):
        raise InferenceWireError("Wire header must be an object.")
    return header, memoryview(data)[payload_offset:]


def encode_timed_observation(
    observation: TimedObservation,
    features: Mapping[str, Mapping[str, Any]],
) -> bytes:
    """Encode one local Axol observation as bounded JSON plus raw uint8 images."""
    if not isinstance(observation, TimedObservation):
        raise InferenceWireError("Expected a TimedObservation.")
    state_names, cameras = _observation_layout(features)
    raw = observation.get_observation()
    if not isinstance(raw, dict):
        raise InferenceWireError("Timed observation data must be a dictionary.")
    expected_keys = {*state_names, *(name for name, _ in cameras), "task"}
    if set(raw) != expected_keys:
        raise InferenceWireError(
            "Observation keys do not exactly match the negotiated feature layout."
        )

    state = [_wire_float(raw[name], f"state value {name!r}") for name in state_names]
    task = _bounded_text(raw["task"], "task", max_bytes=MAX_TASK_BYTES)
    image_headers: list[dict[str, Any]] = []
    image_parts: list[bytes] = []
    image_bytes = 0
    for name, shape in cameras:
        image = raw[name]
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.shape != shape
        ):
            raise InferenceWireError(
                f"Observation camera {name!r} must be uint8 with shape {shape}."
            )
        contiguous = np.ascontiguousarray(image)
        image_bytes += contiguous.nbytes
        if image_bytes > MAX_OBSERVATION_WIRE_BYTES:
            raise InferenceWireSizeError("Observation image data exceeds its limit.")
        image_headers.append({"name": name, "shape": list(shape)})
        image_parts.append(contiguous.tobytes(order="C"))

    timestamp = _wire_float(
        observation.get_timestamp(),
        "observation timestamp",
        minimum=0.0,
        maximum=100_000_000_000.0,
    )
    timestep = _wire_int(
        observation.get_timestep(),
        "observation timestep",
        minimum=0,
        maximum=MAX_TIMESTEP,
    )
    if type(observation.must_go) is not bool:
        raise InferenceWireError("Observation must_go must be a boolean.")
    header = {
        "protocol": OBSERVATION_PROTOCOL,
        "version": INFERENCE_WIRE_VERSION,
        "timestamp": timestamp,
        "timestep": timestep,
        "must_go": observation.must_go,
        "state": state,
        "task": task,
        "images": image_headers,
    }
    return _encode_frame(
        _OBSERVATION_MAGIC,
        header,
        b"".join(image_parts),
        maximum=MAX_OBSERVATION_WIRE_BYTES,
    )


def decode_timed_observation(
    data: bytes,
    features: Mapping[str, Mapping[str, Any]],
) -> TimedObservation:
    """Decode an untrusted observation without invoking a general deserializer."""
    state_names, cameras = _observation_layout(features)
    header, payload = _decode_frame(
        data, _OBSERVATION_MAGIC, maximum=MAX_OBSERVATION_WIRE_BYTES
    )
    header = _exact_keys(
        header,
        frozenset(
            {
                "protocol",
                "version",
                "timestamp",
                "timestep",
                "must_go",
                "state",
                "task",
                "images",
            }
        ),
        "observation header",
    )
    if header["protocol"] != OBSERVATION_PROTOCOL:
        raise InferenceWireError("Observation protocol does not match.")
    if (
        type(header["version"]) is not int
        or header["version"] != INFERENCE_WIRE_VERSION
    ):
        raise InferenceWireError("Observation wire version does not match.")
    timestamp = _wire_float(
        header["timestamp"],
        "observation timestamp",
        minimum=0.0,
        maximum=100_000_000_000.0,
    )
    timestep = _wire_int(
        header["timestep"],
        "observation timestep",
        minimum=0,
        maximum=MAX_TIMESTEP,
    )
    if type(header["must_go"]) is not bool:
        raise InferenceWireError("Observation must_go must be a boolean.")
    task = _bounded_text(header["task"], "task", max_bytes=MAX_TASK_BYTES)
    if not isinstance(header["state"], list) or len(header["state"]) != len(
        state_names
    ):
        raise InferenceWireError("Observation state width does not match.")
    state = [
        _wire_float(value, f"state value {name!r}")
        for name, value in zip(state_names, header["state"], strict=True)
    ]

    image_headers = header["images"]
    if not isinstance(image_headers, list) or len(image_headers) != len(cameras):
        raise InferenceWireError("Observation camera count does not match.")
    raw: dict[str, Any] = dict(zip(state_names, state, strict=True))
    raw["task"] = task
    cursor = 0
    for image_header, (expected_name, expected_shape) in zip(
        image_headers, cameras, strict=True
    ):
        image_header = _exact_keys(
            image_header, frozenset({"name", "shape"}), "image header"
        )
        if image_header["name"] != expected_name:
            raise InferenceWireError("Observation camera order/name does not match.")
        shape = image_header["shape"]
        if not isinstance(shape, list) or tuple(shape) != expected_shape:
            raise InferenceWireError(
                f"Observation camera {expected_name!r} shape does not match."
            )
        size = math.prod(expected_shape)
        end = cursor + size
        if end > len(payload):
            raise InferenceWireError("Observation image data is truncated.")
        raw[expected_name] = (
            np.frombuffer(payload[cursor:end], dtype=np.uint8)
            .reshape(expected_shape)
            .copy()
        )
        cursor = end
    if cursor != len(payload):
        raise InferenceWireError("Observation contains trailing image bytes.")
    return TimedObservation(
        timestamp=timestamp,
        timestep=timestep,
        observation=raw,
        must_go=header["must_go"],
    )


def encode_timed_actions(
    actions: Iterable[TimedAction], action_schema: Iterable[str]
) -> bytes:
    """Encode a local policy action chunk as metadata plus float32 rows."""
    import torch

    schema = _ordered_names(
        tuple(action_schema), "action schema", maximum=MAX_ACTION_DIMENSIONS
    )
    action_list = list(actions)
    if not 1 <= len(action_list) <= MAX_ACTIONS_PER_CHUNK:
        raise InferenceWireError(
            f"Action chunk must contain 1-{MAX_ACTIONS_PER_CHUNK} actions."
        )

    rows: list[np.ndarray] = []
    timestamps: list[float] = []
    timesteps: list[int] = []
    for index, action in enumerate(action_list):
        if not isinstance(action, TimedAction):
            raise InferenceWireError("Action chunk contains a non-TimedAction value.")
        tensor = action.get_action()
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 1
            or not tensor.is_floating_point()
            or tensor.is_complex()
        ):
            raise InferenceWireError(
                "Timed action must contain a one-dimensional tensor."
            )
        if tensor.shape[0] != len(schema):
            raise InferenceWireError("Timed action width does not match action schema.")
        row = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        if (
            not np.isfinite(row).all()
            or np.abs(row).max(initial=0.0) > MAX_ABSOLUTE_NUMERIC_VALUE
        ):
            raise InferenceWireError(
                "Timed action contains a non-finite/out-of-range value."
            )
        rows.append(row)
        timestamps.append(
            _wire_float(
                action.get_timestamp(),
                f"action timestamp {index}",
                minimum=0.0,
                maximum=100_000_000_000.0,
            )
        )
        timesteps.append(
            _wire_int(
                action.get_timestep(),
                f"action timestep {index}",
                minimum=0,
                maximum=MAX_TIMESTEP,
            )
        )
    if any(right != left + 1 for left, right in zip(timesteps, timesteps[1:])):
        raise InferenceWireError("Action timesteps must be strictly contiguous.")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise InferenceWireError("Action timestamps must be nondecreasing.")

    matrix = np.stack(rows).astype("<f4", copy=False)
    header = {
        "protocol": ACTION_PROTOCOL,
        "version": INFERENCE_WIRE_VERSION,
        "action_schema": list(schema),
        "count": len(rows),
        "timestamps": timestamps,
        "timesteps": timesteps,
    }
    return _encode_frame(
        _ACTION_MAGIC,
        header,
        matrix.tobytes(order="C"),
        maximum=MAX_ACTION_WIRE_BYTES,
    )


def decode_timed_actions(
    data: bytes, action_schema: Iterable[str]
) -> list[TimedAction]:
    """Decode an untrusted action chunk into CPU float32 tensors."""
    import torch

    expected_schema = _ordered_names(
        tuple(action_schema), "action schema", maximum=MAX_ACTION_DIMENSIONS
    )
    header, payload = _decode_frame(data, _ACTION_MAGIC, maximum=MAX_ACTION_WIRE_BYTES)
    header = _exact_keys(
        header,
        frozenset(
            {
                "protocol",
                "version",
                "action_schema",
                "count",
                "timestamps",
                "timesteps",
            }
        ),
        "action header",
    )
    if header["protocol"] != ACTION_PROTOCOL:
        raise InferenceWireError("Action protocol does not match.")
    if (
        type(header["version"]) is not int
        or header["version"] != INFERENCE_WIRE_VERSION
    ):
        raise InferenceWireError("Action wire version does not match.")
    schema = _ordered_names(
        header["action_schema"], "received action schema", maximum=MAX_ACTION_DIMENSIONS
    )
    if schema != expected_schema:
        raise InferenceWireError("Received action schema does not exactly match.")
    count = _wire_int(
        header["count"],
        "action count",
        minimum=1,
        maximum=MAX_ACTIONS_PER_CHUNK,
    )
    timestamps = header["timestamps"]
    timesteps = header["timesteps"]
    if (
        not isinstance(timestamps, list)
        or not isinstance(timesteps, list)
        or len(timestamps) != count
        or len(timesteps) != count
    ):
        raise InferenceWireError("Action metadata count does not match.")
    decoded_timestamps = [
        _wire_float(
            value,
            f"action timestamp {index}",
            minimum=0.0,
            maximum=100_000_000_000.0,
        )
        for index, value in enumerate(timestamps)
    ]
    decoded_timesteps = [
        _wire_int(
            value,
            f"action timestep {index}",
            minimum=0,
            maximum=MAX_TIMESTEP,
        )
        for index, value in enumerate(timesteps)
    ]
    if any(
        right != left + 1
        for left, right in zip(decoded_timesteps, decoded_timesteps[1:])
    ):
        raise InferenceWireError("Action timesteps must be strictly contiguous.")
    if any(
        right < left for left, right in zip(decoded_timestamps, decoded_timestamps[1:])
    ):
        raise InferenceWireError("Action timestamps must be nondecreasing.")

    expected_bytes = count * len(schema) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise InferenceWireError("Action numeric buffer has the wrong byte length.")
    matrix = np.frombuffer(payload, dtype="<f4").reshape(count, len(schema))
    if (
        not np.isfinite(matrix).all()
        or np.abs(matrix).max(initial=0.0) > MAX_ABSOLUTE_NUMERIC_VALUE
    ):
        raise InferenceWireError(
            "Action buffer contains non-finite/out-of-range values."
        )
    return [
        TimedAction(
            timestamp=timestamp,
            timestep=timestep,
            action=torch.from_numpy(row.copy()),
        )
        for timestamp, timestep, row in zip(
            decoded_timestamps, decoded_timesteps, matrix, strict=True
        )
    ]


def receive_bounded_chunks(
    iterator: Iterator[Any],
    shutdown_event: Any,
    *,
    maximum: int = MAX_OBSERVATION_WIRE_BYTES,
) -> bytes:
    """Collect one protobuf chunk stream without upstream's unbounded buffer."""
    buffer = bytearray()
    started = False
    for item in iterator:
        if shutdown_event.is_set():
            raise InferenceWireError("Observation receiver is shutting down.")
        try:
            chunk = item.data
            state = item.transfer_state
        except (AttributeError, TypeError) as exc:
            raise InferenceWireError(
                "Malformed inference stream chunk object."
            ) from exc
        if not isinstance(chunk, bytes) or not chunk or len(chunk) > CHUNK_SIZE:
            raise InferenceWireSizeError("Invalid inference stream chunk size.")
        if state == TransferState.TRANSFER_BEGIN:
            if started or buffer:
                raise InferenceWireError("Duplicate inference stream begin marker.")
            started = True
        elif state == TransferState.TRANSFER_MIDDLE:
            if not started:
                raise InferenceWireError(
                    "Inference stream middle arrived before begin."
                )
        elif state == TransferState.TRANSFER_END:
            # A one-chunk message is emitted as END without a BEGIN by LeRobot.
            pass
        else:
            raise InferenceWireError("Unknown inference stream transfer state.")
        if len(buffer) + len(chunk) > maximum:
            raise InferenceWireSizeError(
                f"Inference stream exceeds its {maximum}-byte limit."
            )
        buffer.extend(chunk)
        if state == TransferState.TRANSFER_END:
            return bytes(buffer)
    raise InferenceWireError("Inference stream ended without a final marker.")
