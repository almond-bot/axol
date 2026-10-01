"""Hardware-free protocol conformance and real WebSocket session tests."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from almond_axol.policy.plan_client import PlanPolicyClient
from almond_axol.policy.plan_protocol import (
    Continuation,
    LastDispatched,
    PlanActions,
    PlanObservation,
    PlanSpec,
    decode_actions,
    decode_hello,
    decode_observation,
    decode_ready,
    decode_reset,
    encode_actions,
    encode_hello,
    encode_observation,
    encode_ready,
    encode_reset,
)
from almond_axol.policy.protocol import (
    CameraSpec,
    PolicyProtocolError,
    PolicyRemoteError,
    decode_message,
    encode_error,
)
from tests.plan_peer import PlanPeer


@pytest.fixture
def spec():
    return PlanSpec(
        state_names=("joint.left", "joint.right"),
        action_names=("x", "y", "grip"),
        cameras=(CameraSpec("overhead", (12, 18, 3)), CameraSpec("wrist", (8, 9, 3))),
        fps=30,
        actions_per_chunk=12,
        request_interval=4,
        max_adoption_offset_steps=6,
    )


def observation(
    spec,
    request_id="request-1",
    continuation=None,
    delay_steps=None,
    last_dispatched=None,
):
    rng = np.random.default_rng(17)
    return PlanObservation(
        request_id=request_id,
        state=np.array([0.1, -0.2], dtype=np.float32),
        images={
            camera.name: rng.integers(0, 256, camera.shape, dtype=np.uint8)
            for camera in spec.cameras
        },
        state_sample_time_ns=5_000_000_001,
        image_capture_time_ns={
            camera.name: 4_999_999_000 + i for i, camera in enumerate(spec.cameras)
        },
        continuation=continuation,
        delay_steps=delay_steps,
        last_dispatched=last_dispatched,
    )


class TestCodec:
    def test_negotiated_spec_is_frozen_and_has_no_task(self, spec):
        hello = encode_hello(spec)
        assert decode_hello(*decode_message(hello)) == spec
        assert decode_ready(*decode_message(encode_ready(spec))) == spec
        header, _ = decode_message(hello)
        assert header["version"] == 2
        assert "task" not in header
        assert header["cameras"][0]["codec"] == "png"
        assert len(spec.state_names) != len(spec.action_names)
        assert "dispatch_feedback" not in header

    def test_dispatch_feedback_negotiation_roundtrip(self, spec):
        enabled = replace(spec, dispatch_feedback=True)
        for encode, decode in (
            (encode_hello, decode_hello),
            (encode_ready, decode_ready),
        ):
            header, payload = decode_message(encode(enabled))
            assert header["dispatch_feedback"] is True
            assert decode(header, payload) == enabled

    @pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
    def test_dispatch_feedback_wire_capability_must_be_true(self, spec, value):
        for encode, decode in (
            (encode_hello, decode_hello),
            (encode_ready, decode_ready),
        ):
            header, payload = decode_message(encode(spec))
            header["dispatch_feedback"] = value
            with pytest.raises(PolicyProtocolError, match="dispatch_feedback"):
                decode(header, payload)

    @pytest.mark.parametrize("value", [None, 0, 1, "true"])
    def test_dispatch_feedback_api_capability_must_be_boolean(self, spec, value):
        with pytest.raises(PolicyProtocolError, match="dispatch_feedback"):
            encode_hello(replace(spec, dispatch_feedback=value))

    @pytest.mark.parametrize("reference", [None, LastDispatched("older-plan", 2)])
    def test_last_dispatched_roundtrip_is_independent_of_continuation(
        self, spec, reference
    ):
        enabled = replace(spec, dispatch_feedback=True)
        obs = observation(
            enabled,
            continuation=Continuation("newer-plan", 0),
            last_dispatched=reference,
        )
        header, payload = decode_message(encode_observation(obs, enabled))
        assert "last_dispatched" in header
        decoded = decode_observation(header, payload, enabled)
        assert decoded.last_dispatched == reference
        assert decoded.continuation == obs.continuation

    def test_last_dispatched_requires_capability_and_is_mandatory_when_enabled(
        self, spec
    ):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        enabled = replace(spec, dispatch_feedback=True)
        with pytest.raises(PolicyProtocolError, match="last_dispatched"):
            decode_observation(header, payload, enabled)
        with pytest.raises(PolicyProtocolError, match="last_dispatched"):
            decode_observation({**header, "last_dispatched": None}, payload, spec)
        with pytest.raises(PolicyProtocolError, match="dispatch_feedback"):
            encode_observation(
                observation(spec, last_dispatched=LastDispatched("previous", 0)),
                spec,
            )

    @pytest.mark.parametrize(
        "value",
        [
            {},
            {"prediction_id": "previous"},
            {"prediction_id": "previous", "row": -1},
            {"prediction_id": "previous", "row": True},
            {"prediction_id": "previous", "row": 0.5},
            {"prediction_id": "previous", "row": 1024},
            {"prediction_id": "", "row": 0},
            {"prediction_id": "previous", "row": 0, "actions": []},
            [],
        ],
    )
    def test_invalid_last_dispatched(self, spec, value):
        enabled = replace(spec, dispatch_feedback=True)
        header, payload = decode_message(
            encode_observation(observation(enabled), enabled)
        )
        header["last_dispatched"] = value
        with pytest.raises(PolicyProtocolError):
            decode_observation(header, payload, enabled)

    def test_rgb_png_roundtrip_and_minimal_request(self, spec):
        obs = observation(spec, continuation=Continuation("accepted-0", 3))
        header, payload = decode_message(encode_observation(obs, spec))
        assert set(header) == {"type", "request_id", "observation", "continuation"}
        assert bytes(payload[:8]) == b"\x89PNG\r\n\x1a\n"
        decoded = decode_observation(header, payload, spec)
        np.testing.assert_array_equal(decoded.state, obs.state)
        assert decoded.request_id == obs.request_id
        assert decoded.continuation == obs.continuation
        assert decoded.image_capture_time_ns == obs.image_capture_time_ns
        assert decoded.state_sample_time_ns == obs.state_sample_time_ns
        for name in spec.camera_names:
            np.testing.assert_array_equal(decoded.images[name], obs.images[name])
        assert decoded.delay_steps is None

    def test_adaptive_delay_is_explicit_and_optional(self, spec):
        obs = observation(spec, delay_steps=0)
        packet = encode_observation(obs, spec)
        assert decode_message(packet)[0]["delay_steps"] == 0
        assert decode_observation(*decode_message(packet), spec).delay_steps == 0

    def test_action_reply_is_exact_float32(self, spec):
        actions = np.linspace(-0.8, 0.9, 36, dtype=np.float32).reshape(12, 3)
        reply = PlanActions("request-1", actions)
        header, payload = decode_message(encode_actions(reply, spec))
        assert set(header) == {"type", "request_id", "shape"}
        decoded = decode_actions(header, payload, spec)
        np.testing.assert_array_equal(decoded.actions, actions)
        assert decoded.actions.dtype == np.float32
        assert decoded.request_id == reply.request_id

    def test_reply_can_tighten_but_not_relax_adoption_bound(self, spec):
        reply = PlanActions("request-1", np.zeros((12, 3)), 4)
        assert (
            decode_actions(
                *decode_message(encode_actions(reply, spec)), spec
            ).max_adoption_offset_steps
            == 4
        )
        with pytest.raises(PolicyProtocolError, match="relax"):
            encode_actions(replace(reply, max_adoption_offset_steps=7), spec)

    @pytest.mark.parametrize(
        "update",
        [
            {"version": 1},
            {"version": True},
            {"task": "desktop only"},
            {"fps": True},
            {"fps": 0},
            {"state_names": ["same", "same"]},
            {"actions_per_chunk": 0},
            {"request_interval": 13},
            {"max_adoption_offset_steps": 12},
        ],
    )
    def test_invalid_hello(self, spec, update):
        header, payload = decode_message(encode_hello(spec))
        header.update(update)
        with pytest.raises(PolicyProtocolError):
            decode_hello(header, payload)

    @pytest.mark.parametrize(
        "shape,codec",
        [([12, 18, 1], "png"), ([12, 18, 3], "raw"), ([8193, 18, 3], "png")],
    )
    def test_camera_codec_and_shape_are_restricted(self, spec, shape, codec):
        header, payload = decode_message(encode_hello(spec))
        header["cameras"][0].update(shape=shape, codec=codec)
        with pytest.raises(PolicyProtocolError):
            decode_hello(header, payload)

    def test_total_decoded_image_budget(self, spec):
        large = replace(spec, cameras=(CameraSpec("big", (8192, 8192, 3)),))
        with pytest.raises(PolicyProtocolError, match="decoded image size"):
            encode_hello(large)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("state", [float("nan"), 0]),
            ("state", [True, 0]),
            ("state", [float("inf"), 0]),
            ("state", [1e20, 0]),
            ("state", [0]),
            ("state_sample_time_ns", 1.5),
            ("state_sample_time_ns", -1),
            ("state_sample_time_ns", True),
            ("task", "not here"),
        ],
    )
    def test_invalid_observation(self, spec, key, value):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        header["observation"][key] = value
        with pytest.raises(PolicyProtocolError):
            decode_observation(header, payload, spec)

    @pytest.mark.parametrize(
        "value",
        [
            {"prediction_id": "previous", "from_row": -1},
            {"prediction_id": "previous", "from_row": True},
            {"prediction_id": "", "from_row": 0},
            {"prediction_id": "previous", "from_row": 0, "actions": []},
        ],
    )
    def test_invalid_continuation(self, spec, value):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        header["continuation"] = value
        with pytest.raises(PolicyProtocolError):
            decode_observation(header, payload, spec)

    def test_unknown_request_fields_and_null_delay_rejected(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        with pytest.raises(PolicyProtocolError):
            decode_observation({**header, "delay_steps": None}, payload, spec)
        with pytest.raises(PolicyProtocolError):
            decode_observation({**header, "task": "no"}, payload, spec)

    def test_truncated_and_trailing_images(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        for bad in (payload[:-1], memoryview(bytes(payload) + b"x")):
            with pytest.raises(PolicyProtocolError):
                decode_observation(header, bad, spec)

    def test_png_size_checked_before_decoder_allocation(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        corrupt = bytearray(payload)
        corrupt[16:20] = (1_000_000).to_bytes(4, "big")
        with patch("cv2.imdecode") as decoder:
            with pytest.raises(PolicyProtocolError, match="dimensions/format"):
                decode_observation(header, memoryview(corrupt), spec)
            decoder.assert_not_called()

    def test_png_signature_and_camera_order_checked(self, spec):
        header, payload = decode_message(encode_observation(observation(spec), spec))
        with pytest.raises(PolicyProtocolError, match="PNG"):
            decode_observation(header, memoryview(b"x" + bytes(payload[1:])), spec)
        header["observation"]["images"].reverse()
        with pytest.raises(PolicyProtocolError, match="camera order"):
            decode_observation(header, payload, spec)

    def test_control_messages_reject_payload(self, spec):
        for decode, encoded in (
            (decode_hello, encode_hello(spec)),
            (decode_ready, encode_ready(spec)),
            (decode_reset, encode_reset(2)),
        ):
            header, _ = decode_message(encoded)
            with pytest.raises(PolicyProtocolError, match="payload"):
                decode(header, memoryview(b"unexpected"))

    @pytest.mark.parametrize(
        "chunk", [np.full((12, 3), np.nan), np.zeros((13, 3)), np.zeros((12, 4))]
    )
    def test_invalid_model_output(self, spec, chunk):
        with pytest.raises(PolicyProtocolError):
            encode_actions(PlanActions("r", chunk), spec)

    def test_invalid_action_payload_and_nonfinite_wire(self, spec):
        header = {"type": "actions", "request_id": "r", "shape": [12, 3]}
        with pytest.raises(PolicyProtocolError, match="byte length"):
            decode_actions(header, memoryview(b""), spec)
        payload = memoryview(np.full((12, 3), np.inf, dtype="<f4").tobytes())
        with pytest.raises(PolicyProtocolError, match="non-finite"):
            decode_actions(header, payload, spec)


class ReplyFixture:
    """Deterministic wire data, with no inference or prediction cache."""

    def __init__(self):
        self.received = []
        self.resets = 0

    def setup(self, spec):
        self.spec = spec

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        self.received.append(obs)
        return np.full(
            (self.spec.actions_per_chunk, len(self.spec.action_names)),
            len(self.received),
            dtype=np.float32,
        )


class TestRobotClient:
    def test_loopback_transmits_exact_images_and_distinct_plan_references(self, spec):
        enabled = replace(spec, dispatch_feedback=True)
        fixture = ReplyFixture()
        with PlanPeer(fixture) as peer, PlanPolicyClient(peer.url) as client:
            assert client.connect(enabled) == enabled
            client.reset(1)
            first_obs = observation(enabled)
            first = client.infer(first_obs)
            second = client.infer(observation(enabled, "second"))
            third_obs = observation(
                enabled,
                "third",
                continuation=Continuation(second.request_id, 3),
                last_dispatched=LastDispatched(first.request_id, 8),
                delay_steps=2,
            )
            third = client.infer(third_obs)
            assert third.request_id == "third"
            assert fixture.received[-1].continuation == third_obs.continuation
            assert fixture.received[-1].last_dispatched == third_obs.last_dispatched
            assert fixture.received[-1].delay_steps == 2
            np.testing.assert_array_equal(
                fixture.received[0].images["overhead"], first_obs.images["overhead"]
            )
            assert fixture.received[0].state_sample_time_ns == 5_000_000_001
            client.reset(2)
            client.infer(observation(enabled, "fresh"))
            assert fixture.resets == 2
            assert fixture.received[-1].continuation is None
            assert fixture.received[-1].last_dispatched is None

    def test_inference_requires_connect_and_reset(self, spec):
        with PlanPolicyClient("ws://unused") as client:
            with pytest.raises(PolicyProtocolError, match="Connect and reset"):
                client.infer(observation(spec))
            with pytest.raises(PolicyProtocolError, match="connect"):
                client.reset(1)
        with PlanPeer(ReplyFixture()) as peer, PlanPolicyClient(peer.url) as client:
            client.connect(spec)
            with pytest.raises(PolicyProtocolError, match="Connect and reset"):
                client.infer(observation(spec))

    def test_handshake_cannot_silently_drop_dispatch_feedback(self, spec):
        ws = Mock()
        ws.recv.return_value = encode_ready(spec)
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            with pytest.raises(PolicyProtocolError, match="differs from hello"):
                client.connect(replace(spec, dispatch_feedback=True))
            assert client.ready is None
            ws.close.assert_called_once()

    def test_wrong_reset_acknowledgement_closes_session(self, spec):
        ws = Mock()
        ws.recv.side_effect = [encode_ready(spec), encode_reset(2, reply=True)]
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            client.connect(spec)
            with pytest.raises(PolicyProtocolError, match="wrong episode"):
                client.reset(1)
            assert client.ready is None
            ws.close.assert_called_once()

    @pytest.mark.parametrize("failure", ["wrong_id", "timeout", "remote_error"])
    def test_failed_inference_closes_stream_before_another_request(self, spec, failure):
        ws = Mock()
        bad_reply = {
            "wrong_id": encode_actions(
                PlanActions("wrong", np.zeros((2, 3), np.float32)), spec
            ),
            "timeout": TimeoutError(),
            "remote_error": encode_error("unknown continuation"),
        }[failure]
        ws.recv.side_effect = [
            encode_ready(spec),
            encode_reset(1, reply=True),
            bad_reply,
        ]
        error_type = {
            "wrong_id": PolicyProtocolError,
            "timeout": TimeoutError,
            "remote_error": PolicyRemoteError,
        }[failure]
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            client.connect(spec)
            client.reset(1)
            with pytest.raises(error_type):
                client.infer(observation(spec))
            assert client.ready is None
            ws.close.assert_called_once()
            with pytest.raises(PolicyProtocolError, match="Connect and reset"):
                client.infer(observation(spec, "next"))
            assert ws.send.call_count == 3

    def test_refusal_before_hello_send_is_reported_and_closed(self, spec):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        ws = Mock()
        ws.send.side_effect = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        ws.recv.return_value = encode_error("Policy endpoint is busy.")
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            with pytest.raises(PolicyRemoteError, match="busy"):
                client.connect(spec)
            assert client.ready is None
            ws.close.assert_called_once()

    @pytest.mark.parametrize("queued_success", [True, False])
    def test_send_failure_never_consumes_queued_success(self, spec, queued_success):
        from websockets.exceptions import ConnectionClosedOK
        from websockets.frames import Close

        closed = ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)
        ws = Mock()
        ws.send.side_effect = closed
        if queued_success:
            ws.recv.return_value = encode_ready(spec)
        else:
            ws.recv.side_effect = closed
        with (
            patch("websockets.sync.client.connect", return_value=ws),
            PlanPolicyClient("ws://unused") as client,
        ):
            with pytest.raises(ConnectionClosedOK) as error:
                client.connect(spec)
            assert error.value is closed
            assert client.ready is None
