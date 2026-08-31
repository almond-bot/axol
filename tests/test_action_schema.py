from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import grpc
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.transport import services_pb2
from lerobot.transport.utils import send_bytes_in_chunks

from almond_axol.cli.collect_dagger import _LocalPolicy
from almond_axol.cli.run_policy import _build_axol_robot_client
from almond_axol.lerobot import action_schema as action_schema_module
from almond_axol.lerobot.action_schema import (
    ACTION_SCHEMA_METADATA_KEY,
    ActionSchemaError,
    AxolRemotePolicyConfig,
    MAX_POLICY_SETUP_BYTES,
    decode_axol_policy_setup,
    encode_axol_policy_setup,
    encode_action_schema_confirmation,
    require_exact_action_schema,
    resolve_policy_action_schema,
)
from almond_axol.lerobot.inference_patch import (
    enable_action_schema_handshake,
    import_robot_client_preserving_logging,
)
from almond_axol.lerobot.inference_wire import (
    InferenceWireError,
    decode_timed_actions,
    encode_timed_observation,
)


JOINT_SCHEMA = tuple(f"joint_{index}.pos" for index in range(14))
CARTESIAN_SCHEMA = tuple(
    [
        "left_ee.x",
        "left_ee.y",
        "left_ee.z",
        "left_ee.rx",
        "left_ee.ry",
        "left_ee.rz",
        "left_gripper.pos",
        "right_ee.x",
        "right_ee.y",
        "right_ee.z",
        "right_ee.rx",
        "right_ee.ry",
        "right_ee.rz",
        "right_gripper.pos",
    ]
)


def _policy_config(schema: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        action_feature_names=list(schema),
        dataset_feature_names=None,
        input_features={},
        output_features={"action": SimpleNamespace(shape=(len(schema),))},
        use_amp=False,
    )


def _wire_features(state_schema: tuple[str, ...]) -> dict[str, dict]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_schema),),
            "names": list(state_schema),
        },
        "observation.images.overhead": {
            "dtype": "image",
            "shape": (2, 3, 3),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
    }


def _remote_config(
    action_schema: tuple[str, ...] = JOINT_SCHEMA,
) -> AxolRemotePolicyConfig:
    return AxolRemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="org/policy",
        lerobot_features=_wire_features(JOINT_SCHEMA),
        actions_per_chunk=50,
        device="cuda:0",
        action_schema=action_schema,
    )


class PolicySetupWireTest(unittest.TestCase):
    def test_versioned_json_round_trip_rebuilds_trusted_config(self) -> None:
        encoded = encode_axol_policy_setup(_remote_config())
        decoded = decode_axol_policy_setup(encoded)

        self.assertTrue(encoded.startswith(b"{"))
        self.assertEqual(decoded.policy_type, "act")
        self.assertEqual(decoded.pretrained_name_or_path, "org/policy")
        self.assertEqual(decoded.device, "cuda:0")
        self.assertEqual(decoded.actions_per_chunk, 50)
        self.assertEqual(decoded.action_schema, JOINT_SCHEMA)
        self.assertEqual(decoded.rename_map, {})
        self.assertEqual(decoded.lerobot_features, _wire_features(JOINT_SCHEMA))

    def test_malicious_pickle_is_rejected_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "pickle-executed"

            class Payload:
                def __reduce__(self):  # type: ignore[no-untyped-def]
                    return (os.system, (f"touch {sentinel}",))

            malicious = pickle.dumps(Payload(), protocol=0)
            with self.assertRaisesRegex(ActionSchemaError, "UTF-8 JSON"):
                decode_axol_policy_setup(malicious)

            self.assertFalse(sentinel.exists())

    def test_oversized_and_unknown_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(ActionSchemaError, "1-65536"):
            decode_axol_policy_setup(b"x" * (MAX_POLICY_SETUP_BYTES + 1))

        payload = json.loads(encode_axol_policy_setup(_remote_config()))
        payload["surprise"] = "extension"
        with self.assertRaisesRegex(ActionSchemaError, "unknown.*surprise"):
            decode_axol_policy_setup(json.dumps(payload).encode())

    def test_duplicate_keys_and_nonstandard_constants_are_rejected(self) -> None:
        with self.assertRaisesRegex(ActionSchemaError, "duplicate key"):
            decode_axol_policy_setup(b'{"protocol":1,"protocol":2}')

        payload = json.loads(encode_axol_policy_setup(_remote_config()))
        payload["actions_per_chunk"] = float("nan")
        with self.assertRaisesRegex(ActionSchemaError, "constant 'NaN'"):
            decode_axol_policy_setup(json.dumps(payload).encode())

        valid = encode_axol_policy_setup(_remote_config()).decode()
        huge_integer = valid.replace(
            '"actions_per_chunk":50', '"actions_per_chunk":' + "9" * 5000
        )
        with self.assertRaisesRegex(ActionSchemaError, "UTF-8 JSON"):
            decode_axol_policy_setup(huge_integer.encode())

    def test_malformed_or_dangerous_fields_are_rejected(self) -> None:
        base = json.loads(encode_axol_policy_setup(_remote_config()))
        cases = (
            ("unsupported policy", {**base, "policy_type": "custom"}, "Unsupported"),
            ("shell-like device", {**base, "device": "cuda;sh"}, "Malformed device"),
            ("boolean chunk", {**base, "actions_per_chunk": True}, "integer"),
            ("oversized chunk", {**base, "actions_per_chunk": 1025}, "integer"),
            ("boolean version", {**base, "version": True}, "version"),
            (
                "oversized path",
                {**base, "pretrained_name_or_path": "p" * 1025},
                "1024-byte",
            ),
            (
                "unknown feature",
                {
                    **base,
                    "lerobot_features": {
                        **base["lerobot_features"],
                        "observation.secret": {
                            "dtype": "float32",
                            "shape": [1],
                            "names": ["secret"],
                        },
                    },
                },
                "only Axol state/images",
            ),
        )
        for label, payload, message in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ActionSchemaError, message),
            ):
                decode_axol_policy_setup(json.dumps(payload).encode())

    def test_nonempty_rename_map_never_crosses_wire(self) -> None:
        config = _remote_config()
        config.rename_map = {"observation.state": "other"}
        with self.assertRaisesRegex(ActionSchemaError, "does not permit"):
            encode_axol_policy_setup(config)

    def test_camera_name_cannot_collide_with_state_or_task(self) -> None:
        payload = json.loads(encode_axol_policy_setup(_remote_config()))
        camera = payload["lerobot_features"].pop("observation.images.overhead")
        payload["lerobot_features"]["observation.images.joint_0.pos"] = camera
        with self.assertRaisesRegex(ActionSchemaError, "collide"):
            decode_axol_policy_setup(json.dumps(payload).encode())

        payload["lerobot_features"].pop("observation.images.joint_0.pos")
        payload["lerobot_features"]["observation.images.task"] = camera
        with self.assertRaisesRegex(ActionSchemaError, "collide"):
            decode_axol_policy_setup(json.dumps(payload).encode())


class ActionSchemaResolutionTest(unittest.TestCase):
    def test_resolves_ordered_names_from_training_dataset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            model = base / "pretrained_model"
            dataset = base / "dataset"
            model.mkdir()
            (dataset / "meta").mkdir(parents=True)
            (model / "train_config.json").write_text(
                json.dumps({"dataset": {"repo_id": "local/set", "root": str(dataset)}})
            )
            (dataset / "meta" / "info.json").write_text(
                json.dumps({"features": {"action": {"names": list(JOINT_SCHEMA)}}})
            )

            resolved = resolve_policy_action_schema(
                str(model),
                policy_config=SimpleNamespace(
                    output_features={"action": SimpleNamespace(shape=(14,))}
                ),
            )

        self.assertEqual(resolved, JOINT_SCHEMA)

    def test_hub_dataset_revision_is_used_for_schema_metadata(self) -> None:
        def hub_json(repo_id, filename, *, repo_type="model", revision=None):
            if repo_id == "org/policy" and filename == "train_config.json":
                return {
                    "dataset": {
                        "repo_id": "org/training-data",
                        "repo_type": "dataset",
                        "revision": "schema-v2",
                        "root": "/training-host/private/dataset",
                    }
                }
            if repo_id == "org/training-data" and filename == "meta/info.json":
                self.assertEqual(repo_type, "dataset")
                self.assertEqual(revision, "schema-v2")
                return {"features": {"action": {"names": list(JOINT_SCHEMA)}}}
            return None

        with (
            mock.patch.object(
                action_schema_module, "_hub_json", side_effect=hub_json
            ) as fetch,
            mock.patch.object(
                action_schema_module,
                "_read_json",
                side_effect=AssertionError(
                    "Hub metadata must not choose local filesystem paths"
                ),
            ),
        ):
            resolved = resolve_policy_action_schema(
                "org/policy",
                policy_config=SimpleNamespace(
                    output_features={"action": SimpleNamespace(shape=(14,))}
                ),
            )

        self.assertEqual(resolved, JOINT_SCHEMA)
        self.assertTrue(
            any(
                call.args[:2] == ("org/training-data", "meta/info.json")
                for call in fetch.call_args_list
            )
        )

    def test_loaded_processor_schema_needs_no_hub_round_trip(self) -> None:
        processor = SimpleNamespace(
            steps=[SimpleNamespace(action_names=list(JOINT_SCHEMA))]
        )
        with mock.patch.object(
            action_schema_module,
            "_hub_json",
            side_effect=AssertionError("Hub lookup should be unnecessary"),
        ):
            resolved = resolve_policy_action_schema(
                "org/policy",
                policy_config=SimpleNamespace(
                    output_features={"action": SimpleNamespace(shape=(14,))}
                ),
                processors=(processor,),
            )

        self.assertEqual(resolved, JOINT_SCHEMA)

    def test_same_width_different_semantics_are_rejected(self) -> None:
        with self.assertRaisesRegex(ActionSchemaError, "first mismatch at index 0"):
            require_exact_action_schema(
                JOINT_SCHEMA,
                CARTESIAN_SCHEMA,
                policy_label="Test policy",
            )

    def test_width_only_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "config.json").write_text(
                json.dumps(
                    {"output_features": {"action": {"type": "ACTION", "shape": [14]}}}
                )
            )
            with self.assertRaisesRegex(
                ActionSchemaError, "only exposes an action width"
            ):
                resolve_policy_action_schema(
                    temp,
                    policy_config=SimpleNamespace(
                        output_features={"action": SimpleNamespace(shape=(14,))}
                    ),
                )


class DaggerActionSchemaTest(unittest.TestCase):
    def test_mismatch_fails_without_touching_hardware_or_motion(self) -> None:
        loaded_policy = SimpleNamespace(config=_policy_config(JOINT_SCHEMA))
        loaded_policy.to = mock.Mock(return_value=loaded_policy)
        loaded_policy.eval = mock.Mock()
        policy_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_policy)
        )
        pipeline = SimpleNamespace(steps=[])
        robot = SimpleNamespace(
            action_features=dict.fromkeys(CARTESIAN_SCHEMA, float),
            observation_features={},
            name="axol",
            connect=mock.Mock(),
            send_action=mock.Mock(),
        )

        with (
            mock.patch("lerobot.policies.get_policy_class", return_value=policy_class),
            mock.patch(
                "lerobot.policies.make_pre_post_processors",
                return_value=(pipeline, pipeline),
            ),
        ):
            backend = _LocalPolicy("unused", "act", "cpu", "task")
            with self.assertRaisesRegex(ActionSchemaError, "does not exactly match"):
                backend.connect(robot)

        robot.connect.assert_not_called()
        robot.send_action.assert_not_called()


class _Call:
    def __init__(self, metadata: tuple[tuple[str, bytes], ...]) -> None:
        self._metadata = metadata

    def initial_metadata(self) -> tuple[tuple[str, bytes], ...]:
        return self._metadata


class _UnarySetup:
    def __init__(self, metadata: tuple[tuple[str, bytes], ...]) -> None:
        self.metadata = metadata
        self.last_request = None

    def with_call(self, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return services_pb2.Empty(), _Call(self.metadata)


class _ClientStub:
    def __init__(self, metadata: tuple[tuple[str, bytes], ...]) -> None:
        self.SendPolicyInstructions = _UnarySetup(metadata)

    def Ready(self, request):  # noqa: N802
        return services_pb2.Empty()


class _RpcFailure(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class AsyncActionSchemaClientTest(unittest.TestCase):
    @staticmethod
    def _client(robot):  # type: ignore[no-untyped-def]
        import_robot_client_preserving_logging()
        config = SimpleNamespace(
            fps=60,
            environment_dt=1 / 60,
            server_address="127.0.0.1:1",
            policy_type="act",
            pretrained_name_or_path="unused",
            actions_per_chunk=2,
            policy_device="cpu",
            client_device="cpu",
        )
        return _build_axol_robot_client(config=config, robot=robot, publisher=None)

    @staticmethod
    def _robot() -> SimpleNamespace:
        observations = dict.fromkeys(CARTESIAN_SCHEMA, float)
        observations["overhead"] = (2, 3, 3)
        return SimpleNamespace(
            action_features=dict.fromkeys(CARTESIAN_SCHEMA, float),
            observation_features=observations,
            config=SimpleNamespace(observe_cartesian=True),
            connect=mock.Mock(),
            send_action=mock.Mock(return_value={}),
            torque_residuals=mock.Mock(return_value={}),
        )

    def test_missing_confirmation_fails_before_hardware_or_send(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        client.stub = _ClientStub(())
        try:
            with self.assertRaisesRegex(
                ActionSchemaError, "did not provide exactly one"
            ):
                client.start()
            with self.assertRaisesRegex(ActionSchemaError, "before exact"):
                client._shape_and_send(np.zeros(14))
        finally:
            client.stop()

        robot.connect.assert_not_called()
        robot.send_action.assert_not_called()

    def test_same_width_remote_schema_mismatch_fails_before_send(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        client.stub = _ClientStub(
            (
                (
                    ACTION_SCHEMA_METADATA_KEY,
                    encode_action_schema_confirmation(JOINT_SCHEMA),
                ),
            )
        )
        try:
            with self.assertRaisesRegex(ActionSchemaError, "first mismatch at index 0"):
                client.start()
        finally:
            client.stop()

        robot.connect.assert_not_called()
        robot.send_action.assert_not_called()

    def test_exact_confirmation_allows_named_send(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        stub = _ClientStub(
            (
                (
                    ACTION_SCHEMA_METADATA_KEY,
                    encode_action_schema_confirmation(CARTESIAN_SCHEMA),
                ),
            )
        )
        client.stub = stub
        try:
            self.assertTrue(client.start())
            decoded = decode_axol_policy_setup(
                stub.SendPolicyInstructions.last_request.data
            )
            self.assertEqual(decoded.action_schema, CARTESIAN_SCHEMA)
            client._shape_and_send(np.arange(14, dtype=float))
        finally:
            client.stop()

        sent = robot.send_action.call_args.args[0]
        self.assertEqual(tuple(sent), CARTESIAN_SCHEMA)
        self.assertEqual(sent[CARTESIAN_SCHEMA[-1]], 13.0)

    def test_malicious_action_pickle_is_fatal_without_execution(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "client-pickle-executed"

            class Payload:
                def __reduce__(self):  # type: ignore[no-untyped-def]
                    return (os.system, (f"touch {sentinel}",))

            response = services_pb2.Actions(data=pickle.dumps(Payload()))
            client.stub = SimpleNamespace(GetActions=mock.Mock(return_value=response))
            client.start_barrier = SimpleNamespace(wait=lambda: None)
            client._action_schema_confirmed = True
            try:
                client.receive_actions()
            finally:
                client.stop()

            self.assertIsInstance(client.fatal_error, InferenceWireError)
            self.assertFalse(sentinel.exists())
            robot.send_action.assert_not_called()

    def test_server_protocol_rejection_during_observation_is_fatal(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        client.stub = SimpleNamespace(
            SendObservations=mock.Mock(
                side_effect=_RpcFailure(
                    grpc.StatusCode.INVALID_ARGUMENT, "bad observation frame"
                )
            )
        )
        client._action_schema_confirmed = True
        raw = dict.fromkeys(CARTESIAN_SCHEMA, 0.0)
        raw["overhead"] = np.zeros((2, 3, 3), dtype=np.uint8)
        raw["task"] = "test"
        observation = TimedObservation(
            timestamp=1_800_000_000.0,
            timestep=0,
            observation=raw,
            must_go=True,
        )
        try:
            self.assertFalse(client.send_observation(observation))
            self.assertTrue(client.shutdown_event.is_set())
            self.assertIsInstance(client.fatal_error, InferenceWireError)
        finally:
            client.stop()
        robot.send_action.assert_not_called()

    def test_local_action_install_failure_is_fatal(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        client.stub = SimpleNamespace(
            GetActions=mock.Mock(return_value=services_pb2.Actions(data=b"nonempty"))
        )
        client.start_barrier = SimpleNamespace(wait=lambda: None)
        client._action_schema_confirmed = True
        client._accept_action_payload = mock.Mock(
            side_effect=RuntimeError("queue install failed")
        )
        try:
            client.receive_actions()
            self.assertTrue(client.shutdown_event.is_set())
            self.assertIsInstance(client.fatal_error, RuntimeError)
        finally:
            client.stop()
        robot.send_action.assert_not_called()

    def test_server_internal_action_failure_is_fatal(self) -> None:
        robot = self._robot()
        client = self._client(robot)
        client.stub = SimpleNamespace(
            GetActions=mock.Mock(
                side_effect=_RpcFailure(
                    grpc.StatusCode.INTERNAL, "inspect the inference-server logs"
                )
            )
        )
        client.start_barrier = SimpleNamespace(wait=lambda: None)
        client._action_schema_confirmed = True
        try:
            client.receive_actions()
            self.assertTrue(client.shutdown_event.is_set())
            self.assertIsInstance(client.fatal_error, InferenceWireError)
        finally:
            client.stop()
        robot.send_action.assert_not_called()


class _Abort(RuntimeError):
    pass


class _Context:
    def __init__(self, peer: str = "peer-1") -> None:
        self._peer = peer
        self.metadata: tuple[tuple[str, bytes], ...] = ()
        self.abort_code = None

    def peer(self) -> str:
        return self._peer

    def send_initial_metadata(self, metadata):  # type: ignore[no-untyped-def]
        self.metadata = tuple(metadata)

    def abort(self, code, details):  # type: ignore[no-untyped-def]
        self.abort_code = code
        raise _Abort(details)


class AsyncActionSchemaServerTest(unittest.TestCase):
    def test_server_confirms_loaded_policy_not_client_assertion(self) -> None:
        from lerobot.async_inference import policy_server as server_module

        server_cls = server_module.PolicyServer
        originals = {
            name: getattr(server_cls, name)
            for name in (
                "Ready",
                "SendPolicyInstructions",
                "SendObservations",
                "GetActions",
            )
        }
        marker = "_axol_safe_policy_setup_v2"
        had_marker = hasattr(server_cls, marker)
        old_marker = getattr(server_cls, marker, None)
        if had_marker:
            delattr(server_cls, marker)

        loaded_policy = SimpleNamespace(config=_policy_config(JOINT_SCHEMA))
        loaded_policy.to = mock.Mock(return_value=loaded_policy)
        policy_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_policy)
        )
        pipeline = SimpleNamespace(steps=[])
        try:
            with (
                mock.patch.object(
                    server_module, "get_policy_class", return_value=policy_class
                ),
                mock.patch.object(
                    server_module,
                    "make_pre_post_processors",
                    return_value=(pipeline, pipeline),
                ),
            ):
                enable_action_schema_handshake()
                server = server_cls(PolicyServerConfig(host="127.0.0.1", port=1))
                context = _Context()
                server.Ready(services_pb2.Empty(), context)
                with self.assertRaisesRegex(_Abort, "not confirmed"):
                    server.GetActions(services_pb2.Empty(), context)

                mismatch = AxolRemotePolicyConfig(
                    "act",
                    "unused",
                    _wire_features(JOINT_SCHEMA),
                    2,
                    action_schema=CARTESIAN_SCHEMA,
                )
                with self.assertRaisesRegex(_Abort, "does not exactly match"):
                    server.SendPolicyInstructions(
                        services_pb2.PolicySetup(
                            data=encode_axol_policy_setup(mismatch)
                        ),
                        context,
                    )

                compatible = AxolRemotePolicyConfig(
                    "act",
                    "unused",
                    _wire_features(JOINT_SCHEMA),
                    2,
                    action_schema=JOINT_SCHEMA,
                )
                server.SendPolicyInstructions(
                    services_pb2.PolicySetup(data=encode_axol_policy_setup(compatible)),
                    context,
                )
                self.assertEqual(context.metadata[0][0], ACTION_SCHEMA_METADATA_KEY)

                with self.assertRaisesRegex(_Abort, "Malformed inference stream"):
                    server.SendObservations(iter([object()]), context)
                self.assertEqual(context.abort_code, grpc.StatusCode.INVALID_ARGUMENT)

                with tempfile.TemporaryDirectory() as temp:
                    sentinel = Path(temp) / "server-pickle-executed"

                    class Payload:
                        def __reduce__(self):  # type: ignore[no-untyped-def]
                            return (os.system, (f"touch {sentinel}",))

                    hostile = send_bytes_in_chunks(
                        pickle.dumps(Payload()), services_pb2.Observation
                    )
                    with self.assertRaisesRegex(_Abort, "protocol magic"):
                        server.SendObservations(hostile, context)
                    self.assertFalse(sentinel.exists())

                raw_observation = dict.fromkeys(JOINT_SCHEMA, 0.25)
                raw_observation["task"] = "test"
                raw_observation["overhead"] = np.zeros((2, 3, 3), dtype=np.uint8)
                observation = TimedObservation(
                    timestamp=1_800_000_000.0,
                    timestep=4,
                    observation=raw_observation,
                    must_go=True,
                )
                safe_observation = encode_timed_observation(
                    observation, compatible.lerobot_features
                )
                server.SendObservations(
                    send_bytes_in_chunks(safe_observation, services_pb2.Observation),
                    context,
                )
                server._predict_action_chunk = lambda _observation: [
                    TimedAction(
                        timestamp=1_800_000_000.0 + index / 60,
                        timestep=4 + index,
                        action=torch.arange(14, dtype=torch.float32) + index,
                    )
                    for index in range(2)
                ]
                response = server.GetActions(services_pb2.Empty(), context)
                actions = decode_timed_actions(response.data, JOINT_SCHEMA)
                self.assertEqual([action.timestep for action in actions], [4, 5])
                torch.testing.assert_close(
                    actions[1].action, torch.arange(14, dtype=torch.float32) + 1
                )

                later = TimedObservation(
                    timestamp=1_800_000_001.0,
                    timestep=6,
                    observation=raw_observation,
                    must_go=True,
                )
                server.SendObservations(
                    send_bytes_in_chunks(
                        encode_timed_observation(later, compatible.lerobot_features),
                        services_pb2.Observation,
                    ),
                    context,
                )

                def fail_inference(_observation):  # type: ignore[no-untyped-def]
                    raise RuntimeError("secret internal model error")

                server._predict_action_chunk = fail_inference
                with self.assertRaisesRegex(_Abort, "inspect the inference-server"):
                    server.GetActions(services_pb2.Empty(), context)
                self.assertEqual(context.abort_code, grpc.StatusCode.INTERNAL)
        finally:
            for name, original in originals.items():
                setattr(server_cls, name, original)
            if had_marker:
                setattr(server_cls, marker, old_marker)
            elif hasattr(server_cls, marker):
                delattr(server_cls, marker)


if __name__ == "__main__":
    unittest.main()
