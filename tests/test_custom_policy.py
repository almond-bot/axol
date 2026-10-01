"""The almond_axol.policy SDK (version-2 endpoint) and run-policy's custom wiring."""

from __future__ import annotations

import io
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.policy import (
    CameraSpec,
    Continuation,
    Observation,
    PlanObservation,
    PlanPolicyClient,
    PlanSpec,
    Policy,
    PolicyRemoteError,
    PolicyServer,
    PolicySpec,
    check_policy,
    default_spec,
    policy_url,
)
from almond_axol.policy.check import axol_action_names

NAMES = ("left_shoulder_1.pos", "left_gripper.pos")
CAMERA = CameraSpec("overhead", (2, 3, 3))


def _spec(**overrides) -> PlanSpec:  # type: ignore[no-untyped-def]
    fields = dict(
        state_names=NAMES,
        action_names=NAMES,
        cameras=(CAMERA,),
        fps=30,
        actions_per_chunk=8,
        request_interval=2,
        max_adoption_offset_steps=3,
        dispatch_feedback=True,
    )
    fields.update(overrides)
    return PlanSpec(**fields)


def _frame(value: int = 7) -> np.ndarray:
    return np.full(CAMERA.shape, value, dtype=np.uint8)


def _request(
    request_id: str,
    continuation: Continuation | None = None,
    state: tuple[float, float] = (0.0, 1.0),
) -> PlanObservation:
    return PlanObservation(
        request_id=request_id,
        state=np.asarray(state, np.float32),
        images={"overhead": _frame()},
        state_sample_time_ns=1,
        image_capture_time_ns={"overhead": 1},
        continuation=continuation,
    )


class _Recorder(Policy):
    """Row k of every chunk is (base + k, 1); base counts requests."""

    name = "recorder"

    def __init__(self, rows: int = 6) -> None:
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
        base = 10.0 * len(self.observations)
        chunk = np.ones((self.rows, 2), np.float32)
        chunk[:, 0] = base + np.arange(self.rows)
        return chunk


class _Served:
    """A PolicyServer on an ephemeral port, on a background thread."""

    def __init__(self, policy, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.server = PolicyServer(policy, host="127.0.0.1", port=0, **kwargs)
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


class ServerSessionTest(unittest.TestCase):
    def test_session_echoes_contract_and_resolves_continuation(self) -> None:
        policy = _Recorder()
        with _Served(policy) as served, PlanPolicyClient(served.url) as client:
            self.assertEqual(client.connect(_spec()), _spec())
            client.reset(1)
            first = client.infer(_request("plan-1"))
            np.testing.assert_array_equal(first.actions[:, 0], 10 + np.arange(6))
            second = client.infer(_request("plan-2", Continuation("plan-1", 2)))
        self.assertEqual(policy.resets, 1)
        self.assertIsNone(policy.observations[0].plan)
        # The rows of plan-1 still to run, aligned with plan-2's row 0.
        np.testing.assert_array_equal(
            policy.observations[1].plan[:, 0], 10 + np.arange(2, 6)
        )
        self.assertFalse(policy.observations[1].plan.flags.writeable)
        self.assertEqual(second.request_id, "plan-2")
        self.assertEqual(policy.observations[1].joints["left_gripper.pos"], 1.0)
        np.testing.assert_array_equal(
            policy.observations[1].images["overhead"], _frame()
        )

    def test_reset_forgets_published_plans(self) -> None:
        with _Served(_Recorder()) as served, PlanPolicyClient(served.url) as client:
            client.connect(_spec())
            client.reset(1)
            client.infer(_request("plan-1"))
            client.reset(2)
            with self.assertRaisesRegex(PolicyRemoteError, "did not publish"):
                client.infer(_request("plan-2", Continuation("plan-1", 1)))

    def test_plain_function_and_dict_rows(self) -> None:
        def infer(obs: Observation) -> list[dict[str, float]]:
            return [dict(zip(NAMES, (0.5, 0.0)))] * 20  # longer than the horizon

        with _Served(infer) as served, PlanPolicyClient(served.url) as client:
            client.connect(_spec())
            client.reset(1)
            reply = client.infer(_request("plan-1"))
        self.assertEqual(reply.actions.shape, (8, 2))  # truncated to actions_per_chunk

    def test_infer_exception_is_relayed(self) -> None:
        def broken(obs: Observation) -> None:
            raise RuntimeError("CUDA out of memory")

        with _Served(broken) as served, PlanPolicyClient(served.url) as client:
            client.connect(_spec())
            client.reset(1)
            with self.assertRaisesRegex(PolicyRemoteError, "CUDA out of memory"):
                client.infer(_request("plan-1"))

    def test_declared_layout_and_fps_are_enforced(self) -> None:
        class Cartesian(Policy):
            action_names = ("left_ee.x", "left_gripper.pos")

        class Slow(Policy):
            fps = 15

        for policy, message in ((Cartesian(), "configured for"), (Slow(), "--fps 15")):
            with self.subTest(policy=type(policy).__name__):
                with _Served(policy) as served, PlanPolicyClient(served.url) as client:
                    with self.assertRaisesRegex(PolicyRemoteError, message):
                        client.connect(_spec())

    def test_setup_can_refuse_the_session(self) -> None:
        class Picky(Policy):
            def setup(self, spec: PolicySpec) -> None:
                if "wrist" not in spec.camera_names:
                    raise ValueError("needs a wrist camera")

        with _Served(Picky()) as served, PlanPolicyClient(served.url) as client:
            with self.assertRaisesRegex(PolicyRemoteError, "needs a wrist camera"):
                client.connect(_spec())

    def test_second_robot_is_refused(self) -> None:
        with _Served(_Recorder()) as served:
            with PlanPolicyClient(served.url) as first:
                first.connect(_spec())
                with PlanPolicyClient(served.url) as second:
                    with self.assertRaisesRegex(
                        PolicyRemoteError, "already has a robot"
                    ):
                        second.connect(_spec())

    def test_invalid_ensemble_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolicyServer(_Recorder(), host="127.0.0.1", port=0, ensemble=float("nan"))


class EnsembleTest(unittest.TestCase):
    def test_overlapping_chunks_are_ensembled_but_grippers_snap(self) -> None:
        # Chunk 1 is all 0s, chunk 2 (continuing chunk 1 at row 2) all 1s.
        chunks = iter([np.zeros((6, 2)), np.ones((6, 2))])
        with _Served(lambda obs: next(chunks), ensemble=0.0) as served:
            with PlanPolicyClient(served.url) as client:
                client.connect(_spec())
                client.reset(1)
                client.infer(_request("plan-1"))
                reply = client.infer(_request("plan-2", Continuation("plan-1", 2)))
        # Rows 0-3 overlap chunk 1 (uniform weights at k=0), rows 4-5 don't.
        np.testing.assert_allclose(reply.actions[:, 0], [0.5] * 4 + [1.0] * 2)
        # The gripper follows the newest prediction.
        np.testing.assert_allclose(reply.actions[:, 1], 1.0)

    def test_ensemble_off_publishes_chunks_unchanged(self) -> None:
        chunks = iter([np.zeros((6, 2)), np.ones((6, 2))])
        with _Served(lambda obs: next(chunks)) as served:
            with PlanPolicyClient(served.url) as client:
                client.connect(_spec())
                client.reset(1)
                client.infer(_request("plan-1"))
                reply = client.infer(_request("plan-2", Continuation("plan-1", 2)))
        np.testing.assert_allclose(reply.actions, 1.0)


class _Ramp(Policy):
    """A smooth ramp that continues from the executing plan when there is one."""

    def infer(self, obs: Observation) -> np.ndarray:
        if obs.plan is not None:
            # Row 0 is the tick plan[0] targets: keep it, then keep ramping.
            return obs.plan[0] + 0.01 * np.arange(30)[:, None]
        return obs.state + 0.01 * np.arange(1, 31)[:, None]


class CheckPolicyTest(unittest.TestCase):
    def test_default_spec_matches_axol_layouts(self) -> None:
        joint = default_spec()
        self.assertEqual(len(joint.action_names), 16)
        self.assertEqual(joint.action_names[0], "left_shoulder_1.pos")
        self.assertEqual(joint.action_names[7], "left_gripper.pos")
        self.assertEqual(len(axol_action_names(cartesian=True)), 14)
        self.assertEqual(len(axol_action_names(gripper=False)), 14)

    def test_continuing_policy_executes_without_boundary_jumps(self) -> None:
        with _Served(_Ramp()) as served:
            report = check_policy(served.url, ticks=120, delay_steps=3)
        self.assertEqual(report.recoveries, [])
        self.assertGreater(report.adopted, 10)
        executed = report.targets[~np.isnan(report.targets).any(axis=1)]
        np.testing.assert_allclose(np.diff(executed[:, 0]), 0.01, atol=1e-5)
        self.assertLess(report.max_step_at_switch.max(), 0.011)
        self.assertIn("Largest per-tick jumps", report.summary())

    def test_restarting_policy_shows_switch_jumps(self) -> None:
        def restart(obs: Observation) -> np.ndarray:
            # Ignores the motion in progress and replays the same ramp from a
            # fixed pose every time, so each hand-over jumps backwards.
            return np.repeat(0.01 * np.arange(1, 31)[:, None], 16, axis=1)

        with _Served(restart) as served:
            report = check_policy(served.url, ticks=120, delay_steps=3)
        self.assertGreater(report.max_step_at_switch.max(), 0.02)

    def test_late_replies_are_reported_as_recoveries(self) -> None:
        with _Served(_Ramp()) as served:
            report = check_policy(served.url, ticks=60, delay_steps=10)
        self.assertTrue(report.recoveries)
        self.assertIn("arrived at row", report.recoveries[0])


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
            actions_per_chunk=30,
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

    def test_sdk_policy_is_accepted_and_refusal_blocks_start(self) -> None:
        with _Served(_Recorder()) as served:
            client = self._client(served.url)
            try:
                self.assertIsInstance(client._policy_client, PlanPolicyClient)
                self.assertTrue(client.start())
                self.assertTrue(client._action_schema_confirmed)
            finally:
                client.stop()

        class JointOnly(Policy):
            action_names = ("other.pos",)

        with _Served(JointOnly()) as served:
            client = self._client(served.url)
            try:
                with self.assertRaisesRegex(PolicyRemoteError, "configured for"):
                    client.start()
                self.assertFalse(client._action_schema_confirmed)
                client.robot.send_action.assert_not_called()
                client.robot.connect.assert_not_called()
            finally:
                client.stop()

    def test_unreachable_endpoint_cannot_confirm_action_schema(self) -> None:
        client = self._client("ws://127.0.0.1:1")
        try:
            with self.assertRaises(OSError):
                client.start()
            self.assertFalse(client._action_schema_confirmed)
            client.robot.send_action.assert_not_called()
        finally:
            client.stop()


class RunPolicyConfigTest(unittest.TestCase):
    def test_cli_selects_custom_without_a_version_and_rejects_old_selector(self):
        from almond_axol.cli.collect_dagger import DaggerConfig
        from almond_axol.cli.config import parse
        from almond_axol.cli.run_policy import RunPolicyConfig

        for config_class in (RunPolicyConfig, DaggerConfig):
            args = [
                "--policy_type",
                "custom",
                "--task",
                "test",
                "--robot_config.cameras",
                "{overhead: {serial: 1234}}",
            ]
            if config_class is DaggerConfig:
                args += ["--repo_id", "local/test", "--hold_to_intervene", "true"]
            with self.subTest(command=config_class.__name__):
                self.assertEqual(parse(config_class, args).policy_type, "custom")
                for old_version in ("1", "2"):
                    with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                        with self.assertRaises(SystemExit) as error:
                            parse(
                                config_class, args + ["--custom_protocol", old_version]
                            )
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_lerobot_policy_still_needs_a_path(self) -> None:
        from almond_axol.cli import run_policy

        cfg = run_policy.RunPolicyConfig(policy_type="act", task="t")
        with self.assertRaisesRegex(ValueError, "--policy_path is required"):
            run_policy._run(cfg)

    def test_form_schema_offers_custom_for_run_policy_and_dagger(self) -> None:
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
        self.assertIn("custom", field(DaggerConfig, "policy_type")["options"])
        self.assertFalse(field(DaggerConfig, "policy_path")["required"])
        for config_class in (RunPolicyConfig, DaggerConfig):
            with self.subTest(command=config_class.__name__):
                with self.assertRaises(AssertionError):
                    field(config_class, "custom_protocol")
                self.assertIsNotNone(field(config_class, "plan_config"))


if __name__ == "__main__":
    unittest.main()
