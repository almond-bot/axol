from __future__ import annotations

import json
import os
import pickle
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.transport import services_pb2
from lerobot.transport.utils import TransferState, send_bytes_in_chunks

from almond_axol.cli.inference_server import InferenceServerConfig
from almond_axol.lerobot.inference_wire import (
    MAX_ACTION_WIRE_BYTES,
    InferenceWireError,
    InferenceWireSizeError,
    decode_timed_actions,
    decode_timed_observation,
    encode_timed_actions,
    encode_timed_observation,
    receive_bounded_chunks,
)


STATE_SCHEMA = ("left.pos", "right.pos")
ACTION_SCHEMA = ("left.target", "right.target")
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": list(STATE_SCHEMA),
    },
    "observation.images.overhead": {
        "dtype": "image",
        "shape": (2, 3, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    },
}
_PREFIX = struct.Struct(">8sI")


def _observation() -> TimedObservation:
    return TimedObservation(
        timestamp=1_800_000_000.25,
        timestep=42,
        must_go=True,
        observation={
            "left.pos": np.float32(1.25),
            "right.pos": np.float64(-2.5),
            "overhead": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "task": "Pick the red cube",
        },
    )


def _actions() -> list[TimedAction]:
    return [
        TimedAction(
            timestamp=1_800_000_000.0 + index / 60,
            timestep=42 + index,
            action=torch.tensor([index + 0.25, -index], dtype=torch.float32),
        )
        for index in range(3)
    ]


def _rewrite_header(data: bytes, update) -> bytes:  # type: ignore[no-untyped-def]
    magic, header_size = _PREFIX.unpack_from(data)
    offset = _PREFIX.size + header_size
    header = json.loads(data[_PREFIX.size : offset])
    update(header)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return _PREFIX.pack(magic, len(encoded)) + encoded + data[offset:]


class InferenceObservationWireTest(unittest.TestCase):
    def test_standalone_server_defaults_to_loopback(self) -> None:
        self.assertEqual(InferenceServerConfig().host, "127.0.0.1")

    def test_round_trip_preserves_numeric_state_text_and_images(self) -> None:
        encoded = encode_timed_observation(_observation(), FEATURES)
        decoded = decode_timed_observation(encoded, FEATURES)

        self.assertEqual(decoded.get_timestamp(), 1_800_000_000.25)
        self.assertEqual(decoded.get_timestep(), 42)
        self.assertTrue(decoded.must_go)
        self.assertEqual(decoded.observation["task"], "Pick the red cube")
        self.assertEqual(decoded.observation["left.pos"], 1.25)
        np.testing.assert_array_equal(
            decoded.observation["overhead"],
            _observation().observation["overhead"],
        )
        self.assertTrue(decoded.observation["overhead"].flags.writeable)

    def test_real_protobuf_chunk_round_trip_is_bounded(self) -> None:
        encoded = encode_timed_observation(_observation(), FEATURES)
        chunks = send_bytes_in_chunks(encoded, services_pb2.Observation)
        collected = receive_bounded_chunks(
            chunks, SimpleNamespace(is_set=lambda: False)
        )
        self.assertEqual(collected, encoded)

    def test_malicious_pickle_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "observation-pickle-executed"

            class Payload:
                def __reduce__(self):  # type: ignore[no-untyped-def]
                    return (os.system, (f"touch {sentinel}",))

            with self.assertRaisesRegex(InferenceWireError, "protocol magic"):
                decode_timed_observation(pickle.dumps(Payload()), FEATURES)
            self.assertFalse(sentinel.exists())

    def test_shape_unknown_field_trailing_bytes_and_bool_are_rejected(self) -> None:
        encoded = encode_timed_observation(_observation(), FEATURES)
        malformed = (
            _rewrite_header(
                encoded,
                lambda header: header["images"][0].update({"shape": [3, 2, 3]}),
            ),
            _rewrite_header(
                encoded, lambda header: header.update({"unknown": "field"})
            ),
            encoded + b"trailing",
        )
        for payload in malformed:
            with self.subTest(size=len(payload)), self.assertRaises(InferenceWireError):
                decode_timed_observation(payload, FEATURES)

        observation = _observation()
        observation.observation["left.pos"] = True
        with self.assertRaisesRegex(InferenceWireError, "finite number"):
            encode_timed_observation(observation, FEATURES)

        prefix_magic, header_size = _PREFIX.unpack_from(encoded)
        body_offset = _PREFIX.size + header_size
        huge_integer_header = encoded[_PREFIX.size : body_offset].replace(
            b'"timestep":42', b'"timestep":' + b"9" * 5000
        )
        huge_integer = (
            _PREFIX.pack(prefix_magic, len(huge_integer_header))
            + huge_integer_header
            + encoded[body_offset:]
        )
        with self.assertRaisesRegex(InferenceWireError, "UTF-8 JSON"):
            decode_timed_observation(huge_integer, FEATURES)

    def test_chunk_collector_rejects_oversize_and_invalid_sequence(self) -> None:
        event = SimpleNamespace(is_set=lambda: False)
        oversized = iter(
            [
                services_pb2.Observation(
                    transfer_state=TransferState.TRANSFER_END,
                    data=b"12345",
                )
            ]
        )
        with self.assertRaisesRegex(InferenceWireSizeError, "exceeds"):
            receive_bounded_chunks(oversized, event, maximum=4)

        middle_first = iter(
            [
                services_pb2.Observation(
                    transfer_state=TransferState.TRANSFER_MIDDLE,
                    data=b"x",
                )
            ]
        )
        with self.assertRaisesRegex(InferenceWireError, "before begin"):
            receive_bounded_chunks(middle_first, event)


class InferenceActionWireTest(unittest.TestCase):
    def test_round_trip_returns_cpu_float32_tensors(self) -> None:
        encoded = encode_timed_actions(_actions(), ACTION_SCHEMA)
        decoded = decode_timed_actions(encoded, ACTION_SCHEMA)

        self.assertEqual([action.get_timestep() for action in decoded], [42, 43, 44])
        self.assertTrue(all(action.action.dtype == torch.float32 for action in decoded))
        torch.testing.assert_close(decoded[2].action, torch.tensor([2.25, -2.0]))

    def test_malicious_pickle_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "action-pickle-executed"

            class Payload:
                def __reduce__(self):  # type: ignore[no-untyped-def]
                    return (os.system, (f"touch {sentinel}",))

            with self.assertRaisesRegex(InferenceWireError, "protocol magic"):
                decode_timed_actions(pickle.dumps(Payload()), ACTION_SCHEMA)
            self.assertFalse(sentinel.exists())

    def test_oversize_schema_shape_and_trailing_bytes_are_rejected(self) -> None:
        encoded = encode_timed_actions(_actions(), ACTION_SCHEMA)
        with self.assertRaisesRegex(InferenceWireSizeError, "fit within"):
            decode_timed_actions(b"x" * (MAX_ACTION_WIRE_BYTES + 1), ACTION_SCHEMA)
        with self.assertRaisesRegex(InferenceWireError, "schema"):
            decode_timed_actions(encoded, tuple(reversed(ACTION_SCHEMA)))
        with self.assertRaisesRegex(InferenceWireError, "byte length"):
            decode_timed_actions(encoded + b"trailing", ACTION_SCHEMA)

    def test_boolean_integer_complex_and_nonfinite_tensors_are_rejected(self) -> None:
        for dtype, values in (
            (torch.bool, [True, False]),
            (torch.int64, [1, 2]),
            (torch.complex64, [1 + 0j, 2 + 0j]),
            (torch.float32, [float("nan"), 0.0]),
        ):
            action = TimedAction(
                timestamp=1.0,
                timestep=0,
                action=torch.tensor(values, dtype=dtype),
            )
            with self.subTest(dtype=dtype), self.assertRaises(InferenceWireError):
                encode_timed_actions([action], ACTION_SCHEMA)

        bool_timestamp = _actions()[0]
        bool_timestamp.timestamp = True
        with self.assertRaisesRegex(InferenceWireError, "finite number"):
            encode_timed_actions([bool_timestamp], ACTION_SCHEMA)

    def test_noncontiguous_timesteps_are_rejected(self) -> None:
        actions = _actions()
        actions[1].timestep = 99
        with self.assertRaisesRegex(InferenceWireError, "contiguous"):
            encode_timed_actions(actions, ACTION_SCHEMA)


if __name__ == "__main__":
    unittest.main()
