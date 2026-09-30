"""Custom policy interface client integration with mocked hardware and real loopback WebSockets."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.constants import Joint
from almond_axol.policy.plan_protocol import (
    LastDispatched,
    PlanActions,
    PlanObservation,
)
from almond_axol.policy.plan_scheduler import (
    PlanRuntimeConfig,
    PlanSchedulingError,
    SensorTimingError,
)
from tests.plan_peer import PlanPeer

STATE = tuple(
    f"{side}_{joint.value}.pos" for side in ("left", "right") for joint in Joint
)
ACTIONS = tuple(
    name
    for side in ("left", "right")
    for name in (
        *(f"{side}_ee.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")),
        f"{side}_gripper.pos",
    )
)


class _Policy:
    def __init__(self) -> None:
        self.observations: list[PlanObservation] = []
        self.resets = 0

    def setup(self, spec) -> None:
        self.spec = spec

    def reset(self) -> None:
        self.resets += 1

    def infer(self, obs):
        self.observations.append(obs)
        result = np.zeros((30, 14), dtype=np.float32)
        result[:, 0] = np.arange(30) * 0.001
        return result


class PlanRobotClientTest(unittest.TestCase):
    def make_client(self, url: str, **options):
        from almond_axol.cli.run_policy import _build_axol_robot_client
        from almond_axol.lerobot.inference_patch import (
            import_robot_client_preserving_logging,
        )

        import_robot_client_preserving_logging()
        self.sent: list[tuple[float, dict]] = []
        observations = dict.fromkeys(STATE, float)
        observations["overhead"] = (6, 10, 3)

        def capture():
            now = time.perf_counter_ns()
            raw = dict.fromkeys(STATE, 0.0)
            raw["overhead"] = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
            return raw, now - 1_000_000, {"overhead": now - 2_000_000}

        def send(action):
            self.sent.append((time.perf_counter(), action))
            return action

        robot = SimpleNamespace(
            action_features=dict.fromkeys(ACTIONS, float),
            observation_features=observations,
            config=SimpleNamespace(observe_cartesian=False, action_space="cartesian"),
            cartesian_actions=True,
            reset_cartesian_seed=mock.Mock(),
            positions=(np.zeros(8), np.zeros(8)),
            _joints_to_cartesian=lambda *args: dict.fromkeys(ACTIONS, 0.0),
            get_observation_with_sensor_timestamps=capture,
            connect=mock.Mock(
                side_effect=AssertionError("hardware connect is forbidden")
            ),
            send_action=send,
            torque_residuals=lambda: (np.zeros(7), np.zeros(7)),
        )
        config = SimpleNamespace(
            fps=30,
            environment_dt=1 / 30,
            server_address="unused",
            policy_type="custom",
            pretrained_name_or_path="custom",
            actions_per_chunk=30,
            policy_device="cpu",
            client_device="cpu",
            task="local recording label",
            aggregate_fn=None,
        )
        return _build_axol_robot_client(
            config=config,
            robot=robot,
            publisher=None,
            custom_policy_url=url,
            custom_protocol=2,
            plan_config=PlanRuntimeConfig(**options),
        )

    def test_real_transport_drives_independent_layout_without_task_or_blending(self):
        policy = _Policy()
        server = PlanPeer(policy, host="127.0.0.1", port=0)
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        client = self.make_client(
            f"ws://127.0.0.1:{server.port}", output_width=5, output_height=3
        )
        workers = []
        try:
            self.assertTrue(client.start())
            client.reset_episode_state()
            for target, args in [
                (client.receive_actions, ()),
                (client.control_loop, ("ignored",)),
                (client.observation_loop, ("ignored",)),
            ]:
                worker = threading.Thread(target=target, args=args, daemon=True)
                workers.append(worker)
                worker.start()
            deadline = time.perf_counter() + 5
            while (
                len(self.sent) < 22
                and client.fatal_error is None
                and time.perf_counter() < deadline
            ):
                time.sleep(0.01)
            self.assertIsNone(client.fatal_error)
            self.assertGreaterEqual(len(self.sent), 22)
            self.assertEqual(policy.spec.state_names, STATE)
            self.assertEqual(policy.spec.action_names, ACTIONS)
            self.assertEqual(policy.spec.cameras[0].shape, (3, 5, 3))
            self.assertTrue(policy.spec.dispatch_feedback)
            self.assertEqual(policy.resets, 1)
            self.assertIsNone(policy.observations[0].continuation)
            self.assertIsNone(policy.observations[0].last_dispatched)
            continuation = policy.observations[1].continuation
            self.assertEqual(
                continuation.prediction_id, policy.observations[0].request_id
            )
            self.assertGreaterEqual(continuation.from_row, 10)
            dispatched = policy.observations[1].last_dispatched
            self.assertEqual(
                dispatched.prediction_id, policy.observations[0].request_id
            )
            # A row reserved by pop() may still be inside hardware dispatch.
            self.assertIn(
                dispatched.row, (continuation.from_row - 2, continuation.from_row - 1)
            )
            self.assertIsNone(policy.observations[1].delay_steps)
            self.assertFalse(hasattr(policy.observations[1], "task"))
            # The first targets pass unchanged; inherited temporal ensemble and
            # arrival alignment would alter this trace.
            np.testing.assert_allclose(
                [a["left_ee.x"] for _, a in self.sent[:10]],
                np.arange(10) * 0.001,
                atol=1e-8,
            )
            intervals = np.diff([stamp for stamp, _ in self.sent[:10]])
            self.assertTrue(np.all(intervals > 0.010), intervals)  # no catch-up burst
            client.robot.connect.assert_not_called()
        finally:
            client.stop()
            for worker in workers:
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())
            server.shutdown()
            serving.join(timeout=3)

    def test_cartesian_jump_is_checked_against_dispatched_predecessor(self):
        client = self.make_client("ws://unused")
        client._plan_last_target = np.zeros(14)
        row = np.zeros(14)
        row[0] = 0.051
        with self.assertRaises(PlanSchedulingError):
            client._check_plan_step(row)
        self.assertEqual(self.sent, [])

    def dispatch_once(self, client):
        """Run one real dispatch iteration; stop at the next clock wait."""
        client._action_schema_confirmed = True
        client.start_barrier = threading.Barrier(1)
        # Avoid a machine-load-dependent deadline fault in these identity tests.
        client.config.environment_dt = 10.0

        def stop_at_next_tick(timeout):
            client.shutdown_event.set()
            return True

        with mock.patch.object(client.shutdown_event, "wait", stop_at_next_tick):
            client.control_loop("ignored")
        client.shutdown_event.clear()

    def bootstrap_plan(self, client, value=0.0):
        pending = client._scheduler.begin_request(time.perf_counter_ns())
        self.assertTrue(
            client._scheduler.adopt(
                pending.request_id,
                np.full((30, 14), value, dtype=np.float32),
                time.perf_counter_ns(),
            )
        )
        return pending.request_id

    def capture_request(self, client):
        self.observation_worker(client)
        with client._plan_ready:
            self.assertTrue(
                client._plan_ready.wait_for(
                    lambda: client._plan_slot is not None
                    or client.fatal_error is not None,
                    timeout=2,
                )
            )
            self.assertIsNone(client.fatal_error)
            return client._plan_slot

    def test_dispatch_reference_survives_adoption_before_hardware_send_returns(self):
        client = self.observation_client(request_interval=1)
        original = self.bootstrap_plan(client)
        client._scheduler.pop()
        replacement = client._scheduler.begin_request(time.perf_counter_ns())
        client._scheduler.pop()
        client._scheduler.pop()
        original_send = client.robot.send_action

        def send_and_adopt(action):
            self.assertIsNone(client._plan_last_dispatched)
            with client._plan_ready:
                self.assertTrue(
                    client._scheduler.adopt(
                        replacement.request_id,
                        np.full((30, 14), 0.01, dtype=np.float32),
                        time.perf_counter_ns(),
                    )
                )
            return original_send(action)

        client.robot.send_action = send_and_adopt
        self.dispatch_once(client)
        self.assertIsNone(client.fatal_error)
        self.assertEqual(len(self.sent), 1)
        observation = self.capture_request(client)
        self.assertEqual(observation.continuation.prediction_id, replacement.request_id)
        self.assertEqual(observation.continuation.from_row, 3)
        self.assertEqual(observation.last_dispatched, LastDispatched(original, 3))

    def test_failed_hardware_send_never_publishes_dispatch_reference(self):
        client = self.observation_client(request_interval=1)
        self.bootstrap_plan(client)

        def fail(action):
            self.assertIsNone(client._plan_last_dispatched)
            raise RuntimeError("mock hardware dispatch failed")

        client.robot.send_action = fail
        self.dispatch_once(client)
        self.assertIsInstance(client.fatal_error, RuntimeError)
        self.assertIsNone(client._plan_last_dispatched)
        self.assertEqual(self.sent, [])

    def test_recovery_during_send_does_not_resurrect_reference_after_reset(self):
        client = self.observation_client(request_interval=1)
        self.bootstrap_plan(client)
        original_send = client.robot.send_action

        def recover_during_send(action):
            with client._plan_ready:
                client._scheduler.recover("hold while dispatching")
            return original_send(action)

        client.robot.send_action = recover_during_send
        self.dispatch_once(client)
        self.assertEqual(len(self.sent), 1)
        self.assertIsNone(client._plan_last_dispatched)
        observation = self.capture_request(client)
        self.assertIsNone(observation.continuation)
        self.assertIsNone(observation.last_dispatched)

    def test_feedback_is_frozen_before_image_preparation(self):
        from almond_axol.policy.plan_scheduler import prepare_plan_images

        client = self.observation_client(request_interval=1)
        prediction = self.bootstrap_plan(client)
        self.dispatch_once(client)

        def dispatch_during_preparation(images, config):
            client._plan_last_dispatched = (
                client._scheduler.generation,
                LastDispatched(prediction, 1),
            )
            return prepare_plan_images(images, config)

        with mock.patch(
            "almond_axol.policy.plan_scheduler.prepare_plan_images",
            side_effect=dispatch_during_preparation,
        ):
            observation = self.capture_request(client)
        self.assertEqual(observation.last_dispatched, LastDispatched(prediction, 0))

    def test_recovery_omits_previous_generation_dispatch_reference(self):
        client = self.observation_client(request_interval=1)
        self.bootstrap_plan(client)
        self.dispatch_once(client)
        self.assertIsNotNone(client._plan_last_dispatched)
        client._scheduler.recover("late reply")
        observation = self.capture_request(client)
        self.assertIsNone(observation.continuation)
        self.assertIsNone(observation.last_dispatched)

    def test_final_blocking_row_cannot_reference_a_cleared_prediction_cache(self):
        client = self.observation_client(request_interval=1)
        client._scheduler.recover("blocking refresh")
        self.bootstrap_plan(client)
        self.dispatch_once(client)
        self.assertEqual(len(self.sent), 1)
        self.assertIsNone(client._plan_last_dispatched)
        observation = self.capture_request(client)
        self.assertIsNone(observation.continuation)
        self.assertIsNone(observation.last_dispatched)

    def observation_client(self, **options):
        """Exercise the real acquisition loop without network or action workers."""
        client = self.make_client("ws://unused", **options)
        client.start_barrier = threading.Barrier(1)
        client._camera_names = ("overhead",)
        client._state_names = STATE
        client._policy_client.infer = mock.Mock(
            side_effect=AssertionError("observation test reached transport")
        )
        self.addCleanup(client.stop)
        return client

    def observation_worker(self, client):
        worker = threading.Thread(
            target=client.observation_loop, args=("ignored",), daemon=True
        )

        def stop():
            client.stop()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

        self.addCleanup(stop)
        worker.start()
        return worker

    def test_initial_stale_sample_is_reacquired_without_reserving_or_redating(self):
        from almond_axol.lerobot.robot.robot_axol import PolicyObservationNotReady

        for initial_failure in ("stale", "not_ready"):
            with self.subTest(initial_failure=initial_failure):
                client = self.observation_client()
                original_capture = client.robot.get_observation_with_sensor_timestamps
                captured = []

                def capture():
                    self.assertIsNone(client._scheduler.pending)
                    self.assertIsNone(client._plan_slot)
                    self.assertEqual(self.sent, [])
                    client._policy_client.infer.assert_not_called()
                    sample = original_capture()
                    captured.append(sample)
                    if len(captured) == 1:
                        if initial_failure == "not_ready":
                            raise PolicyObservationNotReady(
                                "telemetry not bracketed yet"
                            )
                        return sample[0], sample[1] - 150_000_000, sample[2]
                    return sample

                client.robot.get_observation_with_sensor_timestamps = mock.Mock(
                    side_effect=capture
                )
                with mock.patch.object(
                    client._scheduler,
                    "begin_request",
                    wraps=client._scheduler.begin_request,
                ) as begin_request:
                    worker = self.observation_worker(client)
                    with client._plan_ready:
                        self.assertTrue(
                            client._plan_ready.wait_for(
                                lambda: client._plan_slot is not None
                                or client.fatal_error is not None,
                                timeout=2,
                            )
                        )
                        observation = client._plan_slot
                    self.assertIsNone(client.fatal_error)
                    self.assertEqual(len(captured), 2)
                    self.assertEqual(begin_request.call_count, 1)
                    self.assertEqual(observation.state_sample_time_ns, captured[1][1])
                    self.assertEqual(observation.image_capture_time_ns, captured[1][2])
                    self.assertIsNone(observation.continuation)
                    client._policy_client.infer.assert_not_called()
                    self.assertEqual(self.sent, [])
                    client.stop()
                    worker.join(timeout=2)

    def test_initial_staleness_times_out_without_request_or_motion(self):
        client = self.observation_client(startup_observation_timeout_s=0.03)
        original_capture = client.robot.get_observation_with_sensor_timestamps

        def capture():
            raw, stamp, cameras = original_capture()
            return raw, stamp - 150_000_000, cameras

        client.robot.get_observation_with_sensor_timestamps = mock.Mock(
            side_effect=capture
        )
        with mock.patch.object(
            client._scheduler, "begin_request", wraps=client._scheduler.begin_request
        ) as begin_request:
            started = time.perf_counter()
            client.observation_loop("ignored")
            self.assertLess(time.perf_counter() - started, 1)
            begin_request.assert_not_called()
        self.assertIsInstance(client.fatal_error, TimeoutError)
        self.assertIn("state_age=", str(client.fatal_error))
        self.assertGreaterEqual(
            client.robot.get_observation_with_sensor_timestamps.call_count, 2
        )
        self.assertIsNone(client._scheduler.pending)
        self.assertIsNone(client._plan_slot)
        client._policy_client.infer.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_initial_freshness_wait_is_cancelled_promptly(self):
        from almond_axol.lerobot.robot.robot_axol import PolicyObservationNotReady

        client = self.observation_client(startup_observation_timeout_s=3)
        entered = threading.Event()

        def capture():
            entered.set()
            raise PolicyObservationNotReady("camera has not published yet")

        client.robot.get_observation_with_sensor_timestamps = capture
        worker = self.observation_worker(client)
        self.assertTrue(entered.wait(timeout=2))
        started = time.perf_counter()
        client.shutdown_event.set()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertIsNone(client.fatal_error)
        self.assertIsNone(client._scheduler.pending)
        client._policy_client.infer.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_initial_corrupt_or_future_timestamp_is_immediately_fatal(self):
        for invalid in ("state_future", "camera_future", "camera_malformed"):
            with self.subTest(invalid=invalid):
                client = self.observation_client()
                raw, stamp, cameras = (
                    client.robot.get_observation_with_sensor_timestamps()
                )
                if invalid == "state_future":
                    stamp += 10_000_000_000
                else:
                    # A stale state must not hide the invalid camera stamp.
                    stamp -= 150_000_000
                    cameras = {
                        "overhead": True
                        if invalid == "camera_malformed"
                        else time.perf_counter_ns() + 10_000_000_000
                    }
                client.robot.get_observation_with_sensor_timestamps = mock.Mock(
                    return_value=(raw, stamp, cameras)
                )
                with mock.patch.object(
                    client._scheduler,
                    "begin_request",
                    wraps=client._scheduler.begin_request,
                ) as begin_request:
                    client.observation_loop("ignored")
                    begin_request.assert_not_called()
                self.assertIsInstance(client.fatal_error, SensorTimingError)
                self.assertFalse(client.fatal_error.retriable)
                client.robot.get_observation_with_sensor_timestamps.assert_called_once()
                client._policy_client.infer.assert_not_called()
                self.assertEqual(self.sent, [])

    def test_stale_observation_after_bootstrap_remains_fatal(self):
        client = self.observation_client()
        original_capture = client.robot.get_observation_with_sensor_timestamps
        calls = 0

        def capture():
            nonlocal calls
            calls += 1
            raw, stamp, cameras = original_capture()
            return raw, stamp - (150_000_000 if calls > 1 else 0), cameras

        client.robot.get_observation_with_sensor_timestamps = capture
        worker = self.observation_worker(client)
        with client._plan_ready:
            self.assertTrue(
                client._plan_ready.wait_for(
                    lambda: client._plan_slot is not None, timeout=2
                )
            )
            pending = client._scheduler.pending
            client._plan_slot = None
            self.assertTrue(
                client._scheduler.adopt(
                    pending.request_id, np.zeros((30, 14)), time.perf_counter_ns()
                )
            )
            for _ in range(10):
                client._scheduler.pop()
            client._plan_ready.notify_all()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(calls, 2)
        self.assertIsInstance(client.fatal_error, SensorTimingError)
        self.assertIn("state observation is stale", str(client.fatal_error))
        self.assertIsNone(client._scheduler.actions)
        self.assertIsNone(client._scheduler.pending)
        self.assertEqual(self.sent, [])

    def test_initial_capture_returning_after_deadline_does_not_reserve_request(self):
        client = self.observation_client(startup_observation_timeout_s=0.02)
        original_capture = client.robot.get_observation_with_sensor_timestamps

        def capture():
            time.sleep(0.04)
            return original_capture()

        client.robot.get_observation_with_sensor_timestamps = capture
        with mock.patch.object(
            client._scheduler, "begin_request", wraps=client._scheduler.begin_request
        ) as begin_request:
            worker = self.observation_worker(client)
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: client._plan_slot is not None
                        or client.fatal_error is not None,
                        timeout=2,
                    )
                )
            client.stop()
            worker.join(timeout=2)
            begin_request.assert_not_called()
        self.assertIsInstance(client.fatal_error, TimeoutError)
        client._policy_client.infer.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_cancellation_during_capture_does_not_accept_returned_sample(self):
        client = self.observation_client()
        original_capture = client.robot.get_observation_with_sensor_timestamps

        def capture():
            sample = original_capture()
            client.shutdown_event.set()
            return sample

        client.robot.get_observation_with_sensor_timestamps = capture
        with mock.patch.object(client.logger, "info") as info:
            captured = client._capture_plan_observation(
                time.perf_counter_ns() + 3_000_000_000
            )
        self.assertIsNone(captured)
        info.assert_not_called()
        self.assertIsNone(client.fatal_error)
        self.assertIsNone(client._scheduler.pending)
        client._policy_client.infer.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_stop_invalidates_pending_without_waiting_for_network(self):
        client = self.make_client("ws://unused")
        request = client._scheduler.begin_request(time.perf_counter_ns())
        checked = []

        def close():
            checked.append(client.shutdown_event.is_set())
            checked.append(client._scheduler.pending is None)

        client._policy_client.close = close
        client.stop()
        self.assertEqual(checked, [True, True])
        self.assertFalse(
            client._scheduler.adopt(request.request_id, np.zeros((30, 14)), 0)
        )

    def test_missed_dispatch_slot_recovers_instead_of_silently_retiming_plan(self):
        client = self.make_client("ws://unused", late_policy="abort")
        client._action_schema_confirmed = True
        client.start_barrier = threading.Barrier(1)
        request = client._scheduler.begin_request(time.perf_counter_ns())
        client._scheduler.adopt(
            request.request_id, np.zeros((30, 14)), time.perf_counter_ns()
        )
        original_send = client.robot.send_action

        def delayed_send(action):
            time.sleep(0.045)  # misses the next 30 Hz slot, less than two ticks
            return original_send(action)

        client.robot.send_action = delayed_send
        client.control_loop("ignored")
        self.assertEqual(len(self.sent), 1)
        self.assertIsInstance(client.fatal_error, PlanSchedulingError)
        self.assertIn("dispatch exceeded", str(client.fatal_error))
        self.assertIsNone(client._scheduler.actions)

    def queue_request(self, client, *, state_age_s=0):
        now = time.perf_counter_ns()
        with client._plan_ready:
            pending = client._scheduler.begin_request(now)
            client._plan_slot = PlanObservation(
                request_id=pending.request_id,
                state=np.zeros(16, dtype=np.float32),
                images={"overhead": np.zeros((6, 10, 3), dtype=np.uint8)},
                state_sample_time_ns=now - round(state_age_s * 1e9),
                image_capture_time_ns={"overhead": now},
            )
            client._plan_ready.notify_all()
        return pending

    def test_receiver_failure_invalidates_targets_and_stops_dispatch(self):
        for error in (TimeoutError("transport timed out"), ValueError("bad reply")):
            with self.subTest(error=error):
                client = self.make_client("ws://unused")
                client.start_barrier = threading.Barrier(1)
                client._wire_generation = client._scheduler.generation
                pending = self.queue_request(client)
                client._policy_client.infer = mock.Mock(side_effect=error)
                client.receive_actions()
                self.assertIs(client.fatal_error, error)
                self.assertTrue(client.shutdown_event.is_set())
                self.assertIsNone(client._scheduler.actions)
                self.assertIsNone(client._scheduler.pending)
                self.assertFalse(client._network_busy)
                self.assertFalse(
                    client._scheduler.adopt(
                        pending.request_id, np.zeros((30, 14)), time.perf_counter_ns()
                    )
                )
                self.assertEqual(self.sent, [])

    def test_stale_unsent_observation_is_discarded_before_transport(self):
        client = self.make_client("ws://unused")
        client.start_barrier = threading.Barrier(1)
        client._wire_generation = client._scheduler.generation
        self.queue_request(client, state_age_s=1)
        client._policy_client.infer = mock.Mock(
            side_effect=AssertionError("stale observation reached transport")
        )
        worker = threading.Thread(target=client.receive_actions, daemon=True)
        worker.start()
        try:
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: client._scheduler.pending is None, timeout=2
                    )
                )
            self.assertIsNone(client.fatal_error)
            client._policy_client.infer.assert_not_called()
            self.assertTrue(client._scheduler.request_due)
        finally:
            client.stop()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())

    def test_recovery_drains_old_reply_before_reset_and_fresh_inference(self):
        client = self.make_client("ws://unused")
        client.start_barrier = threading.Barrier(1)
        client._wire_generation = client._scheduler.generation
        old_request = self.queue_request(client)
        entered = threading.Event()
        release = threading.Event()
        events = []

        def infer(observation):
            events.append(("infer", observation.request_id))
            if observation.request_id == old_request.request_id:
                entered.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release old inference")
            events.append(("reply", observation.request_id))
            return PlanActions(observation.request_id, np.zeros((30, 14)))

        client._policy_client.infer = infer
        client._policy_client.reset = lambda episode: events.append(("reset", episode))
        worker = threading.Thread(target=client.receive_actions, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            with client._plan_ready:
                client._scheduler.recover("test hold during inference")
                self.assertTrue(client._network_busy)
            self.assertFalse(any(kind == "reset" for kind, _ in events))
            release.set()
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: not client._network_busy, timeout=2
                    )
                )
                self.assertIsNone(client._scheduler.actions)
            fresh = self.queue_request(client)
            self.assertTrue(fresh.bootstrap)
            self.assertIsNone(fresh.prediction_id)
            with client._plan_ready:
                self.assertTrue(
                    client._plan_ready.wait_for(
                        lambda: client._scheduler.prediction_id == fresh.request_id,
                        timeout=2,
                    )
                )
            self.assertEqual(
                events,
                [
                    ("infer", old_request.request_id),
                    ("reply", old_request.request_id),
                    ("reset", 1),
                    ("infer", fresh.request_id),
                    ("reply", fresh.request_id),
                ],
            )
            self.assertIsNone(client.fatal_error)
            self.assertEqual(self.sent, [])
        finally:
            release.set()
            client.stop()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())


class RobotLayoutAndTimestampTest(unittest.TestCase):
    def robot(self, **kwargs):
        from almond_axol.lerobot.robot.config_axol import AxolRobotConfig
        from almond_axol.lerobot.robot.robot_axol import AxolRobot

        return AxolRobot(AxolRobotConfig(cameras={}, **kwargs))

    def test_independent_observation_and_action_layout_matrix(self):
        for observation_cartesian in (False, True):
            for action_space in (None, "joint", "cartesian"):
                robot = self.robot(
                    observe_cartesian=observation_cartesian, action_space=action_space
                )
                expected_action_cartesian = (
                    observation_cartesian
                    if action_space is None
                    else action_space == "cartesian"
                )
                self.assertEqual(
                    tuple(robot.observation_features),
                    ACTIONS if observation_cartesian else STATE,
                )
                self.assertEqual(
                    tuple(robot.action_features),
                    ACTIONS if expected_action_cartesian else STATE,
                )

    def test_recorded_actions_follow_action_space_not_observation_space(self):
        robot = self.robot(observe_cartesian=False, action_space="cartesian")
        values = dict.fromkeys(ACTIONS, 0.5)
        robot._joints_to_cartesian = mock.Mock(return_value=values)
        self.assertEqual(robot.action_to_dataset(dict.fromkeys(STATE, 0.0)), values)
        robot = self.robot(observe_cartesian=True, action_space="joint")
        joints = dict.fromkeys(STATE, 0.0)
        self.assertIs(robot.action_to_dataset(joints), joints)

    def test_sensor_timestamps_survive_observation_cache_reserve(self):
        robot = self.robot()
        now = time.perf_counter()
        state_ts = now - 0.010
        camera_times = {"a": now - 0.012, "b": now - 0.008}
        for name, stamp in camera_times.items():
            frame = np.full((2, 3, 3), 7, dtype=np.uint8)
            robot.cameras[name] = SimpleNamespace(
                fps=60,
                latest_capture_ts=lambda s=stamp: (s, now),
                read_latest_with_ts=lambda s=stamp, f=frame: (f, s, now),
            )
        robot._axol = SimpleNamespace(
            state_nearest=mock.Mock(
                return_value=(
                    np.zeros(8),
                    np.zeros(8),
                    np.zeros(8),
                    np.zeros(8),
                    state_ts,
                )
            )
        )
        first, first_state, first_cameras = (
            robot.get_observation_with_sensor_timestamps()
        )
        second, second_state, second_cameras = (
            robot.get_observation_with_sensor_timestamps()
        )
        self.assertEqual(first_state, round(state_ts * 1e9))
        self.assertEqual(
            first_cameras,
            {name: round(stamp * 1e9) for name, stamp in camera_times.items()},
        )
        self.assertEqual((second_state, second_cameras), (first_state, first_cameras))
        self.assertEqual(first.keys(), second.keys())
        robot._axol.state_nearest.assert_called_once()


if __name__ == "__main__":
    unittest.main()
