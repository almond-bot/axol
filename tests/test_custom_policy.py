"""Custom policy support: the almond_axol.policy SDK and run-policy's client for it."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.policy import (
    CameraSpec,
    Observation,
    Policy,
    PolicyClient,
    PolicyProtocolError,
    PolicyRemoteError,
    PolicyServer,
    PolicySpec,
    policy_url,
)
from almond_axol.policy import protocol

STATE = tuple(f"joint_{i}.pos" for i in range(4))
ACTIONS = tuple(f"joint_{i}.pos" for i in range(4))
CAMERA = CameraSpec("overhead", (2, 3, 3))


def _spec(**overrides) -> PolicySpec:  # type: ignore[no-untyped-def]
    fields = dict(
        state_names=STATE,
        action_names=ACTIONS,
        cameras=(CAMERA,),
        fps=30,
        actions_per_chunk=8,
        task="pick the cube",
    )
    fields.update(overrides)
    return PolicySpec(**fields)


def _frame(value: int = 7) -> np.ndarray:
    return np.full(CAMERA.shape, value, dtype=np.uint8)


class ProtocolTest(unittest.TestCase):
    def test_hello_round_trip(self) -> None:
        spec = _spec(policy_path="org/model")
        header, payload = protocol.decode_message(protocol.encode_hello(spec))
        self.assertEqual(len(payload), 0)
        self.assertEqual(protocol.decode_hello(header), spec)

    def test_hello_without_policy_path(self) -> None:
        header, _ = protocol.decode_message(protocol.encode_hello(_spec()))
        self.assertNotIn("policy_path", header)
        self.assertIsNone(protocol.decode_hello(header).policy_path)

    def test_hello_rejects_other_versions(self) -> None:
        header, _ = protocol.decode_message(protocol.encode_hello(_spec()))
        header["version"] = protocol.PROTOCOL_VERSION + 1
        with self.assertRaisesRegex(PolicyProtocolError, "Upgrade almond-axol"):
            protocol.decode_hello(header)

    def test_observation_round_trip(self) -> None:
        spec = _spec()
        message = protocol.encode_observation(
            spec=spec,
            state=[0.1, 0.2, 0.3, 0.4],
            images={"overhead": _frame(9)},
            task="pick",
            timestep=12,
            timestamp=1_700_000_000.5,
        )
        header, payload = protocol.decode_message(message)
        obs = protocol.decode_observation(header, payload, spec)
        np.testing.assert_allclose(obs.state, [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
        self.assertEqual(obs.state.dtype, np.float32)
        np.testing.assert_array_equal(obs.images["overhead"], _frame(9))
        self.assertEqual((obs.task, obs.timestep), ("pick", 12))
        self.assertAlmostEqual(obs.joints["joint_2.pos"], 0.3, places=6)

    def test_observation_frame_shape_is_enforced(self) -> None:
        with self.assertRaisesRegex(PolicyProtocolError, "overhead"):
            protocol.encode_observation(
                spec=_spec(),
                state=[0.0] * 4,
                images={"overhead": np.zeros((3, 3, 3), np.uint8)},
                task="t",
                timestep=0,
                timestamp=0.0,
            )

    def test_truncated_observation_is_rejected(self) -> None:
        spec = _spec()
        message = protocol.encode_observation(
            spec=spec,
            state=[0.0] * 4,
            images={"overhead": _frame()},
            task="t",
            timestep=0,
            timestamp=0.0,
        )
        header, payload = protocol.decode_message(message[:-1])
        with self.assertRaisesRegex(PolicyProtocolError, "truncated"):
            protocol.decode_observation(header, payload, spec)

    def test_actions_round_trip(self) -> None:
        chunk = np.arange(12, dtype=np.float32).reshape(3, 4)
        header, payload = protocol.decode_message(protocol.encode_actions(chunk, 5))
        step, decoded = protocol.decode_actions(header, payload, ACTIONS)
        self.assertEqual(step, 5)
        np.testing.assert_array_equal(decoded, chunk)

    def test_action_width_must_match(self) -> None:
        header, payload = protocol.decode_message(
            protocol.encode_actions(np.zeros((2, 3), np.float32), 0)
        )
        with self.assertRaisesRegex(PolicyProtocolError, "3 columns, expected 4"):
            protocol.decode_actions(header, payload, ACTIONS)

    def test_as_action_chunk_accepts_common_shapes(self) -> None:
        single = protocol.as_action_chunk([1, 2, 3, 4], ACTIONS)
        self.assertEqual(single.shape, (1, 4))
        rows = protocol.as_action_chunk(
            [dict(zip(ACTIONS, [1, 2, 3, 4])), dict(zip(ACTIONS, [5, 6, 7, 8]))],
            ACTIONS,
        )
        np.testing.assert_array_equal(rows[1], [5, 6, 7, 8])

    def test_as_action_chunk_rejects_bad_values(self) -> None:
        with self.assertRaisesRegex(PolicyProtocolError, r"\(T, 4\)"):
            protocol.as_action_chunk(np.zeros((2, 5)), ACTIONS)
        with self.assertRaisesRegex(PolicyProtocolError, "non-finite"):
            protocol.as_action_chunk([[np.nan, 0, 0, 0]], ACTIONS)
        with self.assertRaisesRegex(PolicyProtocolError, "missing"):
            protocol.as_action_chunk([{"joint_0.pos": 1.0}], ACTIONS)

    def test_malformed_frames_are_rejected(self) -> None:
        for bad in (b"", b"\x00\x00\x00\x05{}", "text", b"\x00\x00\x00\x02[]"):
            with self.assertRaises(PolicyProtocolError):
                protocol.decode_message(bad)
        with self.assertRaisesRegex(PolicyProtocolError, "Duplicate"):
            raw = b'{"type":"a","type":"b"}'
            protocol.decode_message(len(raw).to_bytes(4, "big") + raw)

    def test_error_reply_is_raised_as_remote_error(self) -> None:
        header, _ = protocol.decode_message(protocol.encode_error("boom"))
        with self.assertRaisesRegex(PolicyRemoteError, "boom"):
            protocol.expect_type(header, "actions")

    def test_policy_url(self) -> None:
        self.assertEqual(policy_url("10.0.0.2", 8765), "ws://10.0.0.2:8765")
        self.assertEqual(policy_url("::1", 9000), "ws://[::1]:9000")
        self.assertEqual(policy_url("wss://gpu.lan/axol", 1), "wss://gpu.lan/axol")


class _Recorder(Policy):
    """Echo-style policy: a chunk whose row k is the state plus k."""

    name = "recorder"

    def __init__(self, rows: int = 3) -> None:
        self.rows = rows
        self.specs: list[PolicySpec] = []
        self.resets = 0
        self.observations: list[Observation] = []

    def setup(self, spec: PolicySpec) -> None:
        self.specs.append(spec)

    def reset(self) -> None:
        self.resets += 1

    def infer(self, obs: Observation) -> np.ndarray:
        self.observations.append(obs)
        return obs.state[None, :] + np.arange(self.rows, dtype=np.float32)[:, None]


class _Served:
    """A PolicyServer on an ephemeral port, on a background thread."""

    def __init__(self, policy) -> None:  # type: ignore[no-untyped-def]
        self.server = PolicyServer(policy, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return policy_url("127.0.0.1", self.server.port)

    def __enter__(self) -> _Served:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)


class ServerClientTest(unittest.TestCase):
    def test_session_round_trip(self) -> None:
        policy = _Recorder()
        with _Served(policy) as served, PolicyClient(served.url) as client:
            ready = client.connect(_spec(policy_path="org/model"))
            self.assertEqual(ready.action_names, ACTIONS)
            self.assertEqual(ready.name, "recorder")
            client.reset(1)
            chunk = client.infer(
                state=[1.0, 2.0, 3.0, 4.0],
                images={"overhead": _frame()},
                task="pick",
                timestep=40,
                timestamp=time.time(),
            )
        np.testing.assert_array_equal(chunk[2], [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(policy.specs[0].policy_path, "org/model")
        self.assertEqual(policy.resets, 1)
        self.assertEqual(policy.observations[0].timestep, 40)
        np.testing.assert_array_equal(
            policy.observations[0].images["overhead"], _frame()
        )

    def test_plain_function_policy(self) -> None:
        with _Served(lambda obs: np.zeros((2, 4))) as served:
            with PolicyClient(served.url) as client:
                client.connect(_spec())
                chunk = client.infer(
                    state=[0.0] * 4,
                    images={"overhead": _frame()},
                    task="t",
                    timestep=0,
                    timestamp=0.0,
                )
        self.assertEqual(chunk.shape, (2, 4))

    def test_infer_exception_is_relayed(self) -> None:
        def broken(obs: Observation) -> None:
            raise RuntimeError("CUDA out of memory")

        with _Served(broken) as served, PolicyClient(served.url) as client:
            client.connect(_spec())
            with self.assertRaisesRegex(PolicyRemoteError, "CUDA out of memory"):
                client.infer(
                    state=[0.0] * 4,
                    images={"overhead": _frame()},
                    task="t",
                    timestep=0,
                    timestamp=0.0,
                )

    def test_wrong_width_chunk_is_relayed(self) -> None:
        with _Served(lambda obs: np.zeros((2, 3))) as served:
            with PolicyClient(served.url) as client:
                client.connect(_spec())
                with self.assertRaisesRegex(PolicyRemoteError, r"\(T, 4\)"):
                    client.infer(
                        state=[0.0] * 4,
                        images={"overhead": _frame()},
                        task="t",
                        timestep=0,
                        timestamp=0.0,
                    )

    def test_setup_can_refuse_the_session(self) -> None:
        class Picky(Policy):
            def setup(self, spec: PolicySpec) -> None:
                if "wrist" not in spec.camera_names:
                    raise ValueError("needs a wrist camera")

        with _Served(Picky()) as served, PolicyClient(served.url) as client:
            with self.assertRaisesRegex(PolicyRemoteError, "needs a wrist camera"):
                client.connect(_spec())

    def test_declared_action_names_are_reported(self) -> None:
        class Cartesian(Policy):
            action_names = ("left_ee.x", "left_ee.y")
            fps = 15

        with _Served(Cartesian()) as served, PolicyClient(served.url) as client:
            ready = client.connect(_spec())
        self.assertEqual(ready.action_names, ("left_ee.x", "left_ee.y"))
        self.assertEqual(ready.fps, 15)

    def test_second_robot_is_refused(self) -> None:
        with _Served(_Recorder()) as served:
            with PolicyClient(served.url) as first:
                first.connect(_spec())
                with PolicyClient(served.url) as second:
                    with self.assertRaisesRegex(
                        PolicyRemoteError, "already has a robot"
                    ):
                        second.connect(_spec())


# ----------------------------------------------------------------------
# run-policy's robot-side client (needs the lerobot extra)
# ----------------------------------------------------------------------

ROBOT_ACTIONS = tuple(f"joint_{i}.pos" for i in range(14))


class CustomRobotClientTest(unittest.TestCase):
    @staticmethod
    def _robot() -> SimpleNamespace:
        observations = dict.fromkeys(ROBOT_ACTIONS, float)
        observations["overhead"] = CAMERA.shape
        return SimpleNamespace(
            action_features=dict.fromkeys(ROBOT_ACTIONS, float),
            observation_features=observations,
            config=SimpleNamespace(observe_cartesian=True),
            connect=mock.Mock(),
            send_action=mock.Mock(return_value={}),
            torque_residuals=mock.Mock(return_value={}),
        )

    def _client(self, url: str, **kwargs):  # type: ignore[no-untyped-def]
        from almond_axol.cli.run_policy import _build_axol_robot_client
        from almond_axol.lerobot.inference_patch import (
            import_robot_client_preserving_logging,
        )

        import_robot_client_preserving_logging()
        config = SimpleNamespace(
            fps=30,
            environment_dt=1 / 30,
            server_address="unused",
            policy_type="custom",
            pretrained_name_or_path="custom",
            actions_per_chunk=4,
            policy_device="cpu",
            client_device="cpu",
            task="stack the cups",
            aggregate_fn=None,
        )
        return _build_axol_robot_client(
            config=config,
            robot=self._robot(),
            publisher=None,
            custom_policy_url=url,
            **kwargs,
        )

    @staticmethod
    def _observation(timestep: int):  # type: ignore[no-untyped-def]
        from lerobot.async_inference.helpers import TimedObservation

        raw = {name: float(i) for i, name in enumerate(ROBOT_ACTIONS)}
        raw["overhead"] = _frame()
        raw["task"] = "stack the cups"
        return TimedObservation(
            timestamp=time.time(), timestep=timestep, observation=raw
        )

    def test_rollout_round_trip(self) -> None:
        policy = _Recorder(rows=6)
        with _Served(policy) as served:
            client = self._client(served.url, custom_policy_path="org/cups")
            try:
                self.assertTrue(client.start())
                spec = policy.specs[0]
                self.assertEqual(spec.action_names, ROBOT_ACTIONS)
                self.assertEqual(spec.camera_names, ("overhead",))
                self.assertEqual(spec.policy_path, "org/cups")
                self.assertEqual(spec.task, "stack the cups")

                client.reset_episode_state()
                self.assertEqual(policy.resets, 1)
                client.start_barrier = threading.Barrier(1)
                receiver = threading.Thread(target=client.receive_actions, daemon=True)
                receiver.start()
                client.send_observation(self._observation(timestep=10))

                deadline = time.time() + 5
                while client.action_queue.qsize() < 4 and time.time() < deadline:
                    time.sleep(0.01)
                client.shutdown_event.set()
                receiver.join(timeout=5)
                self.assertFalse(receiver.is_alive())
                self.assertIsNone(client.fatal_error)

                queued = list(client.action_queue.queue)
                # Truncated to actions_per_chunk and stamped from the obs timestep.
                self.assertEqual([a.get_timestep() for a in queued], [10, 11, 12, 13])
                np.testing.assert_allclose(
                    queued[1].get_action().numpy()[:3], [1, 2, 3]
                )
                obs = policy.observations[0]
                self.assertEqual(obs.state_names, ROBOT_ACTIONS)
                self.assertEqual(obs.task, "stack the cups")
            finally:
                client.stop()

    def test_action_layout_mismatch_is_refused(self) -> None:
        from almond_axol.lerobot.action_schema import ActionSchemaError

        class JointOnly(Policy):
            action_names = tuple(f"other_{i}.pos" for i in range(14))

        with _Served(JointOnly()) as served:
            client = self._client(served.url)
            try:
                with self.assertRaisesRegex(ActionSchemaError, "Custom policy action"):
                    client.start()
                with self.assertRaisesRegex(ActionSchemaError, "before"):
                    client.send_observation(self._observation(0))
            finally:
                client.stop()

    def test_declared_fps_mismatch_is_refused(self) -> None:
        class Slow(Policy):
            fps = 15

        with _Served(Slow()) as served:
            client = self._client(served.url)
            try:
                with self.assertRaisesRegex(ValueError, "--fps 15"):
                    client.start()
            finally:
                client.stop()
            client = self._client(served.url, allow_fps_mismatch=True)
            try:
                self.assertTrue(client.start())
            finally:
                client.stop()

    def test_policy_failure_is_fatal(self) -> None:
        def broken(obs: Observation) -> None:
            raise RuntimeError("model crashed")

        with _Served(broken) as served:
            client = self._client(served.url)
            try:
                client.start()
                client.reset_episode_state()
                client.start_barrier = threading.Barrier(1)
                receiver = threading.Thread(target=client.receive_actions, daemon=True)
                receiver.start()
                client.send_observation(self._observation(0))
                receiver.join(timeout=5)
                self.assertIsInstance(client.fatal_error, PolicyRemoteError)
                self.assertIn("model crashed", str(client.fatal_error))
                self.assertFalse(client.running)
            finally:
                client.stop()

    def test_unreachable_server_names_the_fix(self) -> None:
        client = self._client("ws://127.0.0.1:1")
        try:
            with self.assertRaisesRegex(RuntimeError, "almond_axol.policy.serve"):
                client.start()
        finally:
            client.stop()


class RunPolicyConfigTest(unittest.TestCase):
    def test_lerobot_policy_still_needs_a_path(self) -> None:
        from almond_axol.cli import run_policy

        cfg = run_policy.RunPolicyConfig(policy_type="act", task="t")
        with self.assertRaisesRegex(ValueError, "--policy_path is required"):
            run_policy._run(cfg)

    def test_form_schema_offers_custom_for_run_policy_only(self) -> None:
        from almond_axol.cli.collect_dagger import DaggerConfig
        from almond_axol.cli.run_policy import RunPolicyConfig
        from almond_axol.serve.introspect import build_schema

        def field(config_class, key):  # type: ignore[no-untyped-def]
            nodes = list(build_schema(config_class).nodes)
            while nodes:
                node = nodes.pop()
                if node.get("key") == key:
                    return node
                nodes.extend(node.get("children", []))
            raise AssertionError(key)

        run_type = field(RunPolicyConfig, "policy_type")
        self.assertEqual(run_type["type"], "select")
        self.assertIn("custom", run_type["options"])
        self.assertTrue(run_type["required"])
        self.assertFalse(field(RunPolicyConfig, "policy_path")["required"])
        self.assertNotIn("custom", field(DaggerConfig, "policy_type")["options"])
        self.assertTrue(field(DaggerConfig, "policy_path")["required"])


if __name__ == "__main__":
    unittest.main()
