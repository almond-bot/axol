"""Real SDK transport through rich-prefix, run-policy and DAgger workflows."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest

from almond_axol.policy import (
    CameraSpec,
    Continuation,
    LastDispatched,
    PlanObservation,
    PlanPolicyClient,
    PlanSpec,
    Policy,
    PolicyProtocolError,
    PolicyRemoteError,
    PolicyServer,
    Prediction,
)
from almond_axol.policy.plan_dagger import PlanDaggerPolicy
from almond_axol.policy.plan_scheduler import PlanRuntimeConfig


def wait_until(predicate, timeout=5):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.002)
    raise AssertionError("Timed out waiting for the SDK workflow")


@dataclass
class RichPlan:
    actions: np.ndarray
    rotations: np.ndarray
    openings: np.ndarray


class RichPrefixPolicy(Policy):
    """Adapter-owned float64 matrices and unclipped gripper openings.

    Published float32 commands cannot reconstruct this state. Only the raw
    accepted-plan identity and offset can select its exact cached prefix.
    """

    def __init__(self, *, ramp=False, block_call=None, reply_bound=None):
        self.ramp = ramp
        self.block_call = block_call
        self.reply_bound = reply_bound
        self.entered = threading.Event()
        self.release = threading.Event()
        self.observations = []
        self.cache = {}
        self.history = {}
        self.prefixes = []
        self.resets = 0

    def setup(self, spec):
        self.spec = spec
        self.cache.clear()

    def reset(self):
        self.resets += 1
        self.cache.clear()

    def infer(self, obs):
        self.observations.append(obs)
        call = len(self.observations)
        assert obs.state_sample_time_ns == obs.state_time_ns
        assert obs.image_capture_time_ns == obs.image_time_ns
        reference = obs.continuation
        prior = None if reference is None else self.cache[reference.prediction_id]
        if prior is None:
            assert obs.plan is None
        else:
            np.testing.assert_array_equal(obs.plan, prior.actions[reference.from_row :])
            assert not obs.plan.flags.writeable
        if obs.last_dispatched is None:
            assert obs.last_dispatched_action is None
        else:
            dispatched = obs.last_dispatched
            np.testing.assert_array_equal(
                obs.last_dispatched_action,
                self.cache[dispatched.prediction_id].actions[dispatched.row],
            )
            assert not obs.last_dispatched_action.flags.writeable
        if call == self.block_call:
            self.entered.set()
            assert self.release.wait(5), "test did not release the model"

        n, width = self.spec.actions_per_chunk, len(self.spec.action_names)
        actions = np.zeros((n, width), dtype=np.float32)
        start = 0.0 if prior is None else float(prior.actions[reference.from_row, 0])
        actions[:, 0] = start + np.arange(n) * 0.001 if self.ramp else call
        openings = 1.234567890123 + call / 100 + np.arange(n) / 10000
        actions[:, -1] = np.clip(openings, 0, 1)
        angle = 0.1234567890123 + call / 100 + np.arange(n) / 1000
        rotations = np.zeros((n, 3, 3), dtype=np.float64)
        rotations[:, 0, 0] = rotations[:, 1, 1] = np.cos(angle)
        rotations[:, 1, 0] = np.sin(angle)
        rotations[:, 0, 1] = -np.sin(angle)
        rotations[:, 2, 2] = 1
        if prior is not None:
            offset = reference.from_row
            count = min(2, len(prior.actions) - offset)
            actions[:count] = prior.actions[offset : offset + count]
            rotations[:count] = prior.rotations[offset : offset + count]
            openings[:count] = prior.openings[offset : offset + count]
            self.prefixes.append(
                (reference, rotations[:count].copy(), openings[:count].copy())
            )
        keep = {
            ref.prediction_id
            for ref in (obs.continuation, obs.last_dispatched)
            if ref is not None
        }
        self.cache = {key: self.cache[key] for key in keep}
        plan = RichPlan(actions.copy(), rotations, openings)
        self.cache[obs.request_id] = plan
        self.history[obs.request_id] = plan
        return Prediction(actions, max_adoption_offset_steps=self.reply_bound)


@contextmanager
def serve(policy):
    server = PolicyServer(policy, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.port}"
    finally:
        policy.release.set()
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()


def request(identity, continuation=None, dispatched=None):
    return PlanObservation(
        request_id=identity,
        state=np.array([0.2], dtype=np.float32),
        images={"eye": np.zeros((2, 3, 3), dtype=np.uint8)},
        state_sample_time_ns=123,
        image_capture_time_ns={"eye": 122},
        continuation=continuation,
        last_dispatched=dispatched,
    )


def spec():
    return PlanSpec(
        state_names=("joint.pos",),
        action_names=("target", "gripper"),
        cameras=(CameraSpec("eye", (2, 3, 3)),),
        actions_per_chunk=8,
        request_interval=2,
        max_adoption_offset_steps=3,
        dispatch_feedback=True,
    )


def test_rich_prefix_and_distinct_dispatch_survive_expired_plan_timeline():
    policy = RichPrefixPolicy()
    with serve(policy) as url, PlanPolicyClient(url) as client:
        client.connect(spec())
        client.reset(1)
        first = client.infer(request("first"))
        client.infer(
            request("second", Continuation("first", 6), LastDispatched("first", 5))
        )
        client.infer(
            request("third", Continuation("second", 2), LastDispatched("first", 7))
        )
        # The first plan's timeline has ended, but dispatch still names it.
        client.infer(
            request("fourth", Continuation("third", 2), LastDispatched("first", 7))
        )
        observed = policy.observations[-1]
        assert observed.continuation == Continuation("third", 2)
        assert observed.last_dispatched == LastDispatched("first", 7)
        np.testing.assert_array_equal(observed.last_dispatched_action, first.actions[7])
        assert set(policy.cache) == {"first", "third", "fourth"}
        for reference, rotations, openings in policy.prefixes:
            prior = policy.history[reference.prediction_id]
            offset = reference.from_row
            np.testing.assert_array_equal(
                rotations, prior.rotations[offset : offset + len(rotations)]
            )
            np.testing.assert_array_equal(
                openings, prior.openings[offset : offset + len(openings)]
            )
            assert np.all(
                openings > 1
            )  # Published gripper rows have lost this information.
        client.reset(2)
        assert not policy.cache
        with pytest.raises(PolicyRemoteError):
            client.infer(
                request("stale", Continuation("fourth", 0), LastDispatched("first", 7))
            )
        assert (
            len(policy.observations) == 4
        )  # Invalid references never reach the model.


@contextmanager
def dagger_session(policy):
    from tests.test_plan_dagger import make_robot

    robot = make_robot()
    config = PlanRuntimeConfig(request_interval=2, max_adoption_offset_steps=3)
    with serve(policy) as url:
        backend = PlanDaggerPolicy(
            url, fps=100, horizon=8, config=config, shadow_inference=False
        )
        try:
            backend.connect(robot)
            yield backend, robot
        finally:
            policy.release.set()
            backend.close()
            assert backend._worker is None or not backend._worker.is_alive()
            robot.connect.assert_not_called()
            robot.send_action.assert_not_called()


def pop_and_confirm(backend):
    action = wait_until(backend.act)
    backend.note_dispatched()
    return action


def test_sdk_reply_deadline_is_enforced_by_real_dagger_scheduler():
    policy = RichPrefixPolicy(block_call=2, reply_bound=0)
    with dagger_session(policy) as (backend, _robot):
        backend.reset()
        assert pop_and_confirm(backend)["target"] == 1
        pop_and_confirm(backend)
        assert policy.entered.wait(3)
        assert policy.observations[1].continuation.from_row == 2
        # One elapsed dispatch tick is below the negotiated limit of three,
        # but above this prediction's zero-row adoption deadline.
        pop_and_confirm(backend)
        policy.release.set()
        wait_until(
            lambda: len(policy.observations) >= 3
            and backend._scheduler.prediction_id == policy.observations[2].request_id
        )
        assert pop_and_confirm(backend)["target"] == 3
        assert backend._scheduler.blocking
        assert "limit 0" in backend._scheduler.last_recovery
        assert policy.resets == 2
        assert policy.observations[2].continuation is None
        assert policy.observations[2].last_dispatched is None
        assert policy.observations[2].last_dispatched_action is None


def test_sdk_dagger_takeover_discards_blocked_reply_and_bootstraps_handback():
    policy = RichPrefixPolicy(block_call=2)
    with dagger_session(policy) as (backend, _robot):
        backend.reset()
        pop_and_confirm(backend)
        pop_and_confirm(backend)
        assert policy.entered.wait(3)
        assert policy.observations[1].continuation is not None
        assert policy.observations[1].last_dispatched is not None
        started = time.perf_counter()
        backend.set_intervention(True)
        backend.set_intervention(False)
        assert time.perf_counter() - started < 0.1
        assert backend.act() is None
        policy.release.set()
        assert pop_and_confirm(backend)["target"] == 3
        handback = policy.observations[2]
        assert handback.continuation is None
        assert handback.last_dispatched is None
        assert handback.last_dispatched_action is None
        assert set(policy.cache) == {handback.request_id}
        assert policy.resets == 2


def test_socket_reply_timeout_closes_stream_before_stale_reply_can_be_reused():
    policy = RichPrefixPolicy(block_call=1)
    with serve(policy) as url, PlanPolicyClient(url, reply_timeout=0.05) as client:
        client.connect(spec())
        client.reset(1)
        try:
            with pytest.raises(TimeoutError, match="did not reply"):
                client.infer(request("slow"))
            assert policy.entered.is_set()
            assert client.spec is None
            with pytest.raises(PolicyProtocolError, match="Connect and reset"):
                client.infer(request("next"))
        finally:
            policy.release.set()


def test_run_policy_sdk_references_and_private_prefix_survive_episode_reset():
    from tests.test_plan_robot_client import PlanRobotClientTest

    policy = RichPrefixPolicy(ramp=True)
    harness = PlanRobotClientTest()
    with serve(policy) as url:
        client = harness.make_client(url, request_interval=2)
        workers = []
        try:
            assert client.start()
            for _episode in range(2):
                first_observation = len(policy.observations)
                first_send = len(harness.sent)
                client.reset_episode_state()
                workers = [
                    threading.Thread(target=target, args=args, daemon=True)
                    for target, args in (
                        (client.receive_actions, ()),
                        (client.control_loop, ("ignored",)),
                        (client.observation_loop, ("ignored",)),
                    )
                ]
                for worker in workers:
                    worker.start()
                wait_until(
                    lambda: len(harness.sent) >= first_send + 10 or client.fatal_error
                )
                assert client.fatal_error is None
                client.shutdown_event.set()
                for worker in workers:
                    worker.join(timeout=3)
                    assert not worker.is_alive()
                episode = policy.observations[first_observation:]
                assert episode[0].continuation is None
                assert episode[0].last_dispatched is None
                assert episode[0].last_dispatched_action is None
                assert any(obs.continuation is not None for obs in episode[1:])
                assert any(
                    obs.last_dispatched_action is not None for obs in episode[1:]
                )
                executed = [
                    row["left_ee.x"]
                    for _, row in harness.sent[first_send : first_send + 10]
                ]
                np.testing.assert_allclose(np.diff(executed), 0.001, atol=1e-6)
            assert policy.resets == 2
            assert policy.prefixes
            client.robot.connect.assert_not_called()
        finally:
            client.stop()
            for worker in workers:
                worker.join(timeout=3)
