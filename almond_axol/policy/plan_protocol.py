"""Custom policy interface (v2): compressed observations and accepted plans.

Uses v1's length-prefixed JSON / binary framing, but has its own strict schema.
An inference request has no task, robot tick, or absolute action prefix: the
desktop owns the task and caches its predictions; ``continuation`` identifies
the robot's accepted prediction and first remaining row. Row zero of a reply
belongs to the request's local scheduler origin, never its arrival time.

Images are lossless, 8-bit RGB PNG. State and image times are nanoseconds in
one robot monotonic clock. They are metadata, not desktop-clock deadlines.
Optional negotiated feedback identifies the most recently confirmed dispatch
in the current reset generation, independently of the accepted continuation.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import (
    MAX_ABSOLUTE_VALUE,
    MAX_ACTIONS_PER_CHUNK,
    MAX_CAMERA_DIMENSION,
    MAX_CAMERAS,
    MAX_DIMENSIONS,
    MAX_MESSAGE_BYTES,
    MAX_NAME_BYTES,
    PROTOCOL,
    CameraSpec,
    PolicyProtocolError,
    PolicyRemoteError,
    as_action_chunk,
    encode_message,
)

PLAN_PROTOCOL_VERSION = 2
_MAX_NS = 2**63 - 1
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PlanSpec:
    """Frozen session contract; independent ordered observation/action layouts.

    The initial v2 transport supports PNG only, with no raw-image fallback.
    ``max_adoption_offset_steps`` is a maximum reply age, not a minimum wait.
    ``None`` accepts replies until their final scheduled row has passed.
    """

    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    cameras: tuple[CameraSpec, ...]
    fps: int = 30
    actions_per_chunk: int = 30
    request_interval: int = 10
    max_adoption_offset_steps: int | None = 6
    dispatch_feedback: bool = False

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)


@dataclass(frozen=True)
class Continuation:
    """An unchanged suffix of one prediction the robot actually accepted."""

    prediction_id: str
    from_row: int


@dataclass(frozen=True)
class LastDispatched:
    """Most recent confirmed dispatch in the current reset generation.

    This identifies the original row before downstream execution filtering,
    after the dispatch call returned successfully. It is not a hardware
    acknowledgement, measured pose, or proof that the target was reached.
    """

    prediction_id: str
    row: int


@dataclass(frozen=True)
class PlanObservation:
    request_id: str
    state: np.ndarray
    images: Mapping[str, np.ndarray]
    state_sample_time_ns: int
    image_capture_time_ns: Mapping[str, int]
    continuation: Continuation | None = None
    delay_steps: int | None = None
    last_dispatched: LastDispatched | None = None


@dataclass(frozen=True)
class PlanActions:
    request_id: str
    actions: np.ndarray
    max_adoption_offset_steps: int | None = None


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, Mapping):
        raise PolicyProtocolError("Expected an object.")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise PolicyProtocolError(
            f"Malformed object: missing {sorted(missing)}, unknown {sorted(unknown)}."
        )


def _int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PolicyProtocolError(
            f"{name}: expected an integer in [{minimum}, {maximum}]."
        )
    return value


def _name(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isprintable()
        or len(value.encode("utf-8")) > MAX_NAME_BYTES
    ):
        raise PolicyProtocolError(f"{name}: expected a non-empty bounded name.")
    return value


def _names(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not 1 <= len(value) <= MAX_DIMENSIONS:
        raise PolicyProtocolError(f"{name}: expected 1-{MAX_DIMENSIONS} names.")
    names = tuple(_name(item, name) for item in value)
    if len(set(names)) != len(names):
        raise PolicyProtocolError(f"{name}: duplicate names.")
    return names


def _empty_payload(payload: memoryview) -> None:
    if payload:
        raise PolicyProtocolError("Control message has unexpected payload bytes.")


def expect_plan_type(
    header: Mapping[str, Any], payload: memoryview, expected: str
) -> None:
    """Validate errors too: they cannot hide extra fields or payload data."""
    if header.get("type") == "error":
        _keys(header, {"type", "message"})
        _empty_payload(payload)
        message = header["message"]
        if not isinstance(message, str) or not message:
            raise PolicyProtocolError("Malformed error message.")
        raise PolicyRemoteError(message)
    if header.get("type") != expected:
        raise PolicyProtocolError(f"Expected {expected!r}, got {header.get('type')!r}.")


def _spec_fields(spec: PlanSpec) -> dict[str, Any]:
    if type(spec.dispatch_feedback) is not bool:
        raise PolicyProtocolError("dispatch_feedback must be a boolean.")
    fields = {
        "protocol": PROTOCOL,
        "version": PLAN_PROTOCOL_VERSION,
        "state_names": list(spec.state_names),
        "action_names": list(spec.action_names),
        "cameras": [
            {"name": camera.name, "shape": list(camera.shape), "codec": "png"}
            for camera in spec.cameras
        ],
        "fps": spec.fps,
        "actions_per_chunk": spec.actions_per_chunk,
        "request_interval": spec.request_interval,
        "max_adoption_offset_steps": spec.max_adoption_offset_steps,
    }
    # Legacy v2 peers reject unknown fields; advertise only when requested.
    if spec.dispatch_feedback:
        fields["dispatch_feedback"] = True
    return fields


def _decode_spec(header: Mapping[str, Any], kind: str) -> PlanSpec:
    _keys(
        header,
        {
            "type",
            "protocol",
            "version",
            "state_names",
            "action_names",
            "cameras",
            "fps",
            "actions_per_chunk",
            "request_interval",
            "max_adoption_offset_steps",
        },
        {"dispatch_feedback"},
    )
    if "dispatch_feedback" in header and header["dispatch_feedback"] is not True:
        raise PolicyProtocolError(
            "Omit disabled dispatch_feedback; otherwise send true."
        )
    if (
        header["type"] != kind
        or header["protocol"] != PROTOCOL
        or type(header["version"]) is not int
        or header["version"] != PLAN_PROTOCOL_VERSION
    ):
        raise PolicyProtocolError("Expected custom policy interface version 2.")
    raw_cameras = header["cameras"]
    if not isinstance(raw_cameras, list) or len(raw_cameras) > MAX_CAMERAS:
        raise PolicyProtocolError(f"cameras: expected at most {MAX_CAMERAS} entries.")
    cameras = []
    for raw in raw_cameras:
        _keys(raw, {"name", "shape", "codec"})
        shape = raw["shape"]
        if not isinstance(shape, list) or len(shape) != 3:
            raise PolicyProtocolError("camera shape must be [height, width, 3].")
        height = _int(shape[0], "camera height", 1, MAX_CAMERA_DIMENSION)
        width = _int(shape[1], "camera width", 1, MAX_CAMERA_DIMENSION)
        if type(shape[2]) is not int or shape[2] != 3 or raw["codec"] != "png":
            raise PolicyProtocolError("Version 2 requires lossless RGB PNG cameras.")
        cameras.append(
            CameraSpec(_name(raw["name"], "camera name"), (height, width, 3))
        )
    if len({camera.name for camera in cameras}) != len(cameras):
        raise PolicyProtocolError("Duplicate camera names.")
    if sum(math.prod(camera.shape) for camera in cameras) > MAX_MESSAGE_BYTES:
        raise PolicyProtocolError("Total decoded image size exceeds the session limit.")
    horizon = _int(
        header["actions_per_chunk"], "actions_per_chunk", 1, MAX_ACTIONS_PER_CHUNK
    )
    max_age = header["max_adoption_offset_steps"]
    if max_age is not None:
        max_age = _int(max_age, "max_adoption_offset_steps", 0, horizon - 1)
    return PlanSpec(
        state_names=_names(header["state_names"], "state_names"),
        action_names=_names(header["action_names"], "action_names"),
        cameras=tuple(cameras),
        fps=_int(header["fps"], "fps", 1, 1000),
        actions_per_chunk=horizon,
        request_interval=_int(
            header["request_interval"], "request_interval", 1, horizon
        ),
        max_adoption_offset_steps=max_age,
        dispatch_feedback=header.get("dispatch_feedback", False),
    )


def encode_hello(spec: PlanSpec) -> bytes:
    header = {"type": "hello", **_spec_fields(spec)}
    _decode_spec(header, "hello")
    return encode_message(header)


def decode_hello(header: Mapping[str, Any], payload: memoryview) -> PlanSpec:
    _empty_payload(payload)
    return _decode_spec(header, "hello")


def encode_ready(spec: PlanSpec) -> bytes:
    header = {"type": "ready", **_spec_fields(spec)}
    _decode_spec(header, "ready")
    return encode_message(header)


def decode_ready(header: Mapping[str, Any], payload: memoryview) -> PlanSpec:
    expect_plan_type(header, payload, "ready")
    _empty_payload(payload)
    return _decode_spec(header, "ready")


def encode_reset(episode: int, *, reply: bool = False) -> bytes:
    _int(episode, "episode", 0, 1_000_000_000)
    return encode_message(
        {"type": "reset_ok" if reply else "reset", "episode": episode}
    )


def decode_reset(
    header: Mapping[str, Any], payload: memoryview, *, reply: bool = False
) -> int:
    expect_plan_type(header, payload, "reset_ok" if reply else "reset")
    _keys(header, {"type", "episode"})
    _empty_payload(payload)
    return _int(header["episode"], "episode", 0, 1_000_000_000)


def _decode_continuation(value: Any) -> Continuation | None:
    if value is None:
        return None
    _keys(value, {"prediction_id", "from_row"})
    return Continuation(
        prediction_id=_name(value["prediction_id"], "prediction_id"),
        from_row=_int(value["from_row"], "from_row", 0, MAX_ACTIONS_PER_CHUNK - 1),
    )


def _decode_last_dispatched(value: Any) -> LastDispatched | None:
    if value is None:
        return None
    _keys(value, {"prediction_id", "row"})
    return LastDispatched(
        prediction_id=_name(value["prediction_id"], "prediction_id"),
        row=_int(value["row"], "row", 0, MAX_ACTIONS_PER_CHUNK - 1),
    )


def _state(values: Any, width: int) -> np.ndarray:
    if not isinstance(values, list) or len(values) != width:
        raise PolicyProtocolError("Observation state width does not match hello.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or abs(value) > MAX_ABSOLUTE_VALUE
        or not math.isfinite(value)
        for value in values
    ):
        raise PolicyProtocolError("Observation contains invalid state values.")
    return np.array(values, dtype=np.float32)


def _check_png(data: bytes | memoryview, shape: tuple[int, int, int]) -> None:
    # Check dimensions BEFORE cv2 allocates decoded pixels. Restrict the codec
    # to the exact 8-bit RGB format the encoder emits, not palette/gray/16-bit.
    if (
        len(data) < 33
        or bytes(data[:8]) != _PNG_SIGNATURE
        or bytes(data[8:16]) != b"\x00\x00\x00\rIHDR"
    ):
        raise PolicyProtocolError("Image is not a PNG with a valid IHDR header.")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", bytes(data[16:29])
    )
    if (
        (height, width, 3) != shape
        or depth != 8
        or color != 2
        or compression != 0
        or filtering != 0
        or interlace not in (0, 1)
    ):
        raise PolicyProtocolError("PNG dimensions/format do not match the session.")


def encode_observation(obs: PlanObservation, spec: PlanSpec) -> bytes:
    import cv2

    request_id = _name(obs.request_id, "request_id")
    # Coercing here is convenient for the public numpy API, but validation below
    # still rejects NaN, incorrect dimensions, or booleans.
    values = np.asarray(obs.state).tolist()
    state = _state(values, len(spec.state_names))
    _int(obs.state_sample_time_ns, "state_sample_time_ns", 0, _MAX_NS)
    if set(obs.images) != set(spec.camera_names) or set(
        obs.image_capture_time_ns
    ) != set(spec.camera_names):
        raise PolicyProtocolError("Observation camera names do not match hello.")
    continuation = None
    if obs.continuation is not None:
        continuation = {
            "prediction_id": obs.continuation.prediction_id,
            "from_row": obs.continuation.from_row,
        }
    _decode_continuation(continuation)
    header: dict[str, Any] = {
        "type": "infer",
        "request_id": request_id,
        "observation": {
            "state": state.tolist(),
            "state_sample_time_ns": obs.state_sample_time_ns,
            "images": [],
        },
        "continuation": continuation,
    }
    if spec.dispatch_feedback:
        dispatched = obs.last_dispatched
        header["last_dispatched"] = (
            None
            if dispatched is None
            else {"prediction_id": dispatched.prediction_id, "row": dispatched.row}
        )
        _decode_last_dispatched(header["last_dispatched"])
    elif obs.last_dispatched is not None:
        raise PolicyProtocolError(
            "last_dispatched requires negotiated dispatch_feedback."
        )
    if obs.delay_steps is not None:
        header["delay_steps"] = _int(
            obs.delay_steps, "delay_steps", 0, spec.actions_per_chunk
        )
    parts = []
    total = 0
    for camera in spec.cameras:
        frame = obs.images[camera.name]
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != camera.shape
        ):
            raise PolicyProtocolError(
                f"Camera {camera.name!r} must be uint8 RGB {camera.shape}."
            )
        capture_ns = _int(
            obs.image_capture_time_ns[camera.name], "capture_time_ns", 0, _MAX_NS
        )
        ok, encoded = cv2.imencode(
            ".png",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if not ok:
            raise PolicyProtocolError("PNG encoding failed.")
        part = encoded.tobytes()
        total += len(part)
        if total > MAX_MESSAGE_BYTES:
            raise PolicyProtocolError("Compressed images exceed the message limit.")
        header["observation"]["images"].append(
            {
                "name": camera.name,
                "capture_time_ns": capture_ns,
                "byte_length": len(part),
            }
        )
        parts.append(part)
    return encode_message(header, b"".join(parts))


def decode_observation(
    header: Mapping[str, Any], payload: memoryview, spec: PlanSpec
) -> PlanObservation:
    import cv2

    expect_plan_type(header, payload, "infer")
    required = {"type", "request_id", "observation", "continuation"}
    if spec.dispatch_feedback:
        required.add("last_dispatched")
    _keys(header, required, {"delay_steps"})
    request_id = _name(header["request_id"], "request_id")
    observation = header["observation"]
    _keys(observation, {"state", "state_sample_time_ns", "images"})
    state = _state(observation["state"], len(spec.state_names))
    state_ns = _int(
        observation["state_sample_time_ns"], "state_sample_time_ns", 0, _MAX_NS
    )
    continuation = _decode_continuation(header["continuation"])
    last_dispatched = (
        _decode_last_dispatched(header["last_dispatched"])
        if spec.dispatch_feedback
        else None
    )
    delay = header.get("delay_steps")
    if "delay_steps" in header:
        delay = _int(delay, "delay_steps", 0, spec.actions_per_chunk)
    image_headers = observation["images"]
    if not isinstance(image_headers, list) or len(image_headers) != len(spec.cameras):
        raise PolicyProtocolError("Observation camera count does not match hello.")
    images: dict[str, np.ndarray] = {}
    capture_times: dict[str, int] = {}
    cursor = 0
    for raw, camera in zip(image_headers, spec.cameras, strict=True):
        _keys(raw, {"name", "capture_time_ns", "byte_length"})
        if raw["name"] != camera.name:
            raise PolicyProtocolError("Observation camera order does not match hello.")
        capture_times[camera.name] = _int(
            raw["capture_time_ns"], "capture_time_ns", 0, _MAX_NS
        )
        size = _int(raw["byte_length"], "image byte_length", 33, MAX_MESSAGE_BYTES)
        if cursor + size > len(payload):
            raise PolicyProtocolError("Truncated PNG image payload.")
        part = payload[cursor : cursor + size]
        _check_png(part, camera.shape)
        try:
            decoded = cv2.imdecode(
                np.frombuffer(part, dtype=np.uint8), cv2.IMREAD_COLOR
            )
        except cv2.error as exc:
            raise PolicyProtocolError("PNG image could not be decoded.") from exc
        if (
            decoded is None
            or decoded.dtype != np.uint8
            or decoded.shape != camera.shape
        ):
            raise PolicyProtocolError(
                "PNG image could not be decoded at its declared shape."
            )
        images[camera.name] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        cursor += size
    if cursor != len(payload):
        raise PolicyProtocolError("Observation has trailing image bytes.")
    return PlanObservation(
        request_id,
        state,
        images,
        state_ns,
        capture_times,
        continuation,
        delay,
        last_dispatched,
    )


def validate_actions(reply: PlanActions, spec: PlanSpec) -> PlanActions:
    request_id = _name(reply.request_id, "request_id")
    chunk = as_action_chunk(reply.actions, spec.action_names)
    if len(chunk) > spec.actions_per_chunk:
        raise PolicyProtocolError("Action horizon exceeds the negotiated maximum.")
    bound = reply.max_adoption_offset_steps
    if bound is not None:
        bound = _int(bound, "max_adoption_offset_steps", 0, len(chunk) - 1)
        if (
            spec.max_adoption_offset_steps is not None
            and bound > spec.max_adoption_offset_steps
        ):
            raise PolicyProtocolError(
                "Reply cannot relax the negotiated adoption deadline."
            )
    return PlanActions(request_id, np.array(chunk, dtype=np.float32, copy=True), bound)


def encode_actions(reply: PlanActions, spec: PlanSpec) -> bytes:
    reply = validate_actions(reply, spec)
    header: dict[str, Any] = {
        "type": "actions",
        "request_id": reply.request_id,
        "shape": list(reply.actions.shape),
    }
    if reply.max_adoption_offset_steps is not None:
        header["max_adoption_offset_steps"] = reply.max_adoption_offset_steps
    return encode_message(
        header, reply.actions.astype("<f4", copy=False).tobytes(order="C")
    )


def decode_actions(
    header: Mapping[str, Any], payload: memoryview, spec: PlanSpec
) -> PlanActions:
    expect_plan_type(header, payload, "actions")
    _keys(header, {"type", "request_id", "shape"}, {"max_adoption_offset_steps"})
    shape = header["shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise PolicyProtocolError("actions shape must be [rows, width].")
    rows = _int(shape[0], "action rows", 1, spec.actions_per_chunk)
    width = _int(
        shape[1], "action width", len(spec.action_names), len(spec.action_names)
    )
    if len(payload) != rows * width * 4:
        raise PolicyProtocolError("Action payload has the wrong byte length.")
    if (
        "max_adoption_offset_steps" in header
        and header["max_adoption_offset_steps"] is None
    ):
        raise PolicyProtocolError(
            "Omit an unspecified adoption bound; do not send null."
        )
    return validate_actions(
        PlanActions(
            request_id=header["request_id"],
            actions=np.frombuffer(payload, dtype="<f4").reshape(rows, width),
            max_adoption_offset_steps=header.get("max_adoption_offset_steps"),
        ),
        spec,
    )
