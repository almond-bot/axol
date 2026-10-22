"""SDK observations preserve the robot's two independent prediction references."""

from dataclasses import replace

import numpy as np
import pytest

from almond_axol.policy import (
    Continuation,
    LastDispatched,
    PlanPolicyClient,
    Policy,
    PolicyProtocolError,
)
from almond_axol.policy.plan_protocol import (
    decode_actions,
    encode_observation,
)
from almond_axol.policy.protocol import decode_message
from almond_axol.policy.server import PolicyServer, _Session
from tests.test_custom_policy import _Recorder, _request, _Served, _spec


class _Endpoint:
    """Run the real server request codec/wrapper while retaining failed sessions.

    The production client closes its connection on an error, so an in-process
    endpoint lets these tests inspect transactional cache behavior after errors.
    """

    def __init__(self, policy=None, *, ensemble=None, spec=None):
        self.policy = _Recorder() if policy is None else policy
        self.spec = _spec() if spec is None else spec
        self.session = _Session(self.spec, ensemble)
        self.server = object.__new__(PolicyServer)
        self.server.policy = self.policy
        self.policy.setup(self.spec)
        self.reset()

    def reset(self):
        self.session.clear()
        self.policy.reset()

    def infer(self, request_id, continuation=None, dispatched=None):
        request = replace(
            _request(request_id, continuation),
            last_dispatched=dispatched,
            delay_steps=2,
        )
        message = self.server._infer(
            self.session, *decode_message(encode_observation(request, self.spec))
        )
        return decode_actions(*decode_message(message), self.spec)


def test_websocket_sdk_exposes_raw_references_and_exact_published_dispatch_row():
    policy = _Recorder()
    continuation = Continuation("accepted", 2)
    dispatched = LastDispatched("earlier", 4)
    with _Served(policy) as served, PlanPolicyClient(served.url) as client:
        client.connect(_spec())
        client.reset(1)
        earlier = client.infer(_request("earlier"))
        client.infer(_request("accepted", Continuation("earlier", 1)))
        client.infer(
            replace(_request("next", continuation), last_dispatched=dispatched)
        )
    observation = policy.observations[-1]
    assert observation.continuation == continuation
    assert observation.last_dispatched == dispatched
    np.testing.assert_array_equal(
        observation.last_dispatched_action, earlier.actions[4]
    )
    np.testing.assert_array_equal(observation.plan[:, 0], 20 + np.arange(2, 6))
    assert not observation.last_dispatched_action.flags.writeable
    assert not observation.plan.flags.writeable
    assert observation.state_sample_time_ns == observation.state_time_ns == 1
    assert (
        observation.image_capture_time_ns
        == observation.image_time_ns
        == {"overhead": 1}
    )


def test_bootstrap_observation_keeps_existing_defaults():
    endpoint = _Endpoint()
    endpoint.infer("first")
    obs = endpoint.policy.observations[-1]
    assert obs.continuation is None
    assert obs.last_dispatched is None
    assert obs.last_dispatched_action is None
    assert obs.plan is None
    assert obs.delay_steps == 2


def test_dispatch_only_keeps_reference_without_old_ensemble_conditioning():
    endpoint = _Endpoint(ensemble=0.0)
    first = endpoint.infer("first")
    second = endpoint.infer("second", dispatched=LastDispatched("first", 5))
    obs = endpoint.policy.observations[-1]
    assert obs.continuation is None
    assert obs.plan is None
    assert obs.last_dispatched == LastDispatched("first", 5)
    np.testing.assert_array_equal(obs.last_dispatched_action, first.actions[5])
    # Bootstrap resets conditioning; retained dispatch history is not an
    # overlapping model prediction on the new timeline.
    np.testing.assert_array_equal(second.actions[:, 0], 20 + np.arange(6))
    endpoint.infer("third", Continuation("second", 1), LastDispatched("first", 5))
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].last_dispatched_action, first.actions[5]
    )


def test_dispatch_feedback_resolves_published_ensemble_not_raw_prediction():
    endpoint = _Endpoint(ensemble=0.0)
    endpoint.infer("first")
    second = endpoint.infer("second", Continuation("first", 2))
    assert second.actions[1, 0] == (13 + 21) / 2
    endpoint.infer("third", Continuation("second", 2), LastDispatched("second", 1))
    obs = endpoint.policy.observations[-1]
    np.testing.assert_array_equal(obs.last_dispatched_action, second.actions[1])
    assert obs.last_dispatched_action[0] != 21


@pytest.mark.parametrize("kind", ["continuation", "dispatched"])
@pytest.mark.parametrize("unknown", [False, True])
def test_unknown_or_past_end_reference_is_rejected_before_policy_inference(
    kind, unknown
):
    endpoint = _Endpoint()
    first = endpoint.infer("first")
    prediction_id, row = ("unpublished", 0) if unknown else ("first", 6)
    continuation = Continuation("first", 1)
    dispatched = LastDispatched("first", 0)
    if kind == "continuation":
        continuation = Continuation(prediction_id, row)
    else:
        dispatched = LastDispatched(prediction_id, row)
    with pytest.raises(PolicyProtocolError):
        endpoint.infer("invalid", continuation, dispatched)
    assert len(endpoint.policy.observations) == 1
    assert "invalid" not in endpoint.session.plans
    endpoint.infer("retry", Continuation("first", 1), LastDispatched("first", 0))
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].plan, first.actions[1:]
    )


def test_active_continuation_and_older_dispatch_survive_many_declined_candidates():
    endpoint = _Endpoint()
    previous = endpoint.infer("previous")
    accepted = endpoint.infer("accepted", Continuation("previous", 5))
    for i in range(_Session.MAX_PLANS + 16):
        endpoint.infer(
            f"declined-{i}",
            Continuation("accepted", 5),
            LastDispatched("previous", 5),
        )
    obs = endpoint.policy.observations[-1]
    np.testing.assert_array_equal(obs.plan, accepted.actions[5:])
    np.testing.assert_array_equal(obs.last_dispatched_action, previous.actions[5])
    assert obs.continuation == Continuation("accepted", 5)
    assert obs.last_dispatched == LastDispatched("previous", 5)
    # Limits may reserve slots for the two protected plans; unreferenced
    # candidates must remain bounded independently of session length.
    assert len(endpoint.session.plans) <= _Session.MAX_PLANS + 2


@pytest.mark.parametrize("ensemble", [None, 0.0])
def test_history_is_snapshotted_when_policy_reuses_its_inference_buffer(ensemble):
    class Reusing(Policy):
        def __init__(self):
            self.buffer = np.ones((6, 2), np.float32)
            self.observations = []

        def infer(self, obs):
            self.observations.append(obs)
            self.buffer[:, 0] = 10 * len(self.observations) + np.arange(6)
            self.buffer[:, 1] = 1
            return self.buffer

    policy = Reusing()
    endpoint = _Endpoint(policy, ensemble=ensemble)
    first = endpoint.infer("first")
    policy.buffer[:] = 999  # The GPU adapter reuses its output allocation.
    endpoint.infer("second", Continuation("first", 2), LastDispatched("first", 0))
    second_obs = policy.observations[-1]
    np.testing.assert_array_equal(second_obs.plan, first.actions[2:])
    np.testing.assert_array_equal(second_obs.last_dispatched_action, first.actions[0])
    endpoint.infer("third", Continuation("first", 3), LastDispatched("first", 1))
    np.testing.assert_array_equal(policy.observations[-1].plan, first.actions[3:])
    np.testing.assert_array_equal(
        policy.observations[-1].last_dispatched_action, first.actions[1]
    )


def test_observation_views_cannot_mutate_published_history():
    endpoint = _Endpoint()
    first = endpoint.infer("first")
    endpoint.infer("second", Continuation("first", 1), LastDispatched("first", 0))
    obs = endpoint.policy.observations[-1]
    with pytest.raises(ValueError):
        obs.plan[0, 0] = 999
    with pytest.raises(ValueError):
        obs.last_dispatched_action[0] = 999
    # Even deliberately re-enabling writes on the convenience copies must
    # never alter the server's original published target identities.
    obs.plan.flags.writeable = True
    obs.last_dispatched_action.flags.writeable = True
    obs.plan[:] = 999
    obs.last_dispatched_action[:] = 999
    endpoint.infer("third", Continuation("first", 1), LastDispatched("first", 0))
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].plan, first.actions[1:]
    )
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].last_dispatched_action, first.actions[0]
    )


def test_prediction_wrapper_carries_tighter_adoption_deadline_over_websocket():
    from almond_axol.policy import Prediction

    def infer(obs):
        return Prediction(np.ones((6, 2), np.float32), max_adoption_offset_steps=2)

    with _Served(infer) as served, PlanPolicyClient(served.url) as client:
        client.connect(_spec())
        client.reset(1)
        reply = client.infer(_request("bounded"))
    assert reply.request_id == "bounded"
    assert reply.max_adoption_offset_steps == 2
    np.testing.assert_array_equal(reply.actions, np.ones((6, 2)))


@pytest.mark.parametrize("bound", [-1, 4, 6, True])
def test_invalid_endpoint_deadline_is_rejected_before_cache_publication(bound):
    from almond_axol.policy import Prediction

    class Bounded(_Recorder):
        def infer(self, obs):
            chunk = super().infer(obs)
            return Prediction(chunk, bound) if obs.request_id == "invalid" else chunk

    endpoint = _Endpoint(Bounded())
    first = endpoint.infer("first")
    with pytest.raises(PolicyProtocolError):
        endpoint.infer("invalid", Continuation("first", 1))
    assert "invalid" not in endpoint.session.plans
    assert set(endpoint.session.plans) == {"first"}
    endpoint.infer("retry", Continuation("first", 1))
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].plan, first.actions[1:]
    )


def test_deadline_is_checked_against_horizon_after_sdk_truncation():
    from almond_axol.policy import Prediction

    class Oversized(Policy):
        def infer(self, obs):
            return Prediction(np.zeros((20, 2)), max_adoption_offset_steps=8)

    endpoint = _Endpoint(Oversized(), spec=_spec(max_adoption_offset_steps=None))
    with pytest.raises(PolicyProtocolError):
        endpoint.infer("invalid")
    assert not endpoint.session.plans


@pytest.mark.parametrize("dispatched", [False, True])
def test_reset_forgets_both_reference_kinds_and_allows_fresh_id_reuse(dispatched):
    endpoint = _Endpoint()
    endpoint.infer("reusable")
    endpoint.reset()
    with pytest.raises(PolicyProtocolError):
        endpoint.infer(
            "stale",
            None if dispatched else Continuation("reusable", 0),
            LastDispatched("reusable", 0) if dispatched else None,
        )
    reply = endpoint.infer("reusable")
    assert reply.request_id == "reusable"
    obs = endpoint.policy.observations[-1]
    assert obs.plan is None
    assert obs.continuation is None
    assert obs.last_dispatched is None
    assert endpoint.policy.resets == 2


def test_duplicate_request_id_is_rejected_before_inference_even_after_eviction():
    endpoint = _Endpoint()
    endpoint.infer("accepted")
    for i in range(_Session.MAX_PLANS + 16):
        endpoint.infer(f"candidate-{i}", Continuation("accepted", 1))
    calls = len(endpoint.policy.observations)
    for duplicate in ("accepted", "candidate-0"):
        with pytest.raises(PolicyProtocolError):
            endpoint.infer(duplicate, Continuation("accepted", 1))
    assert len(endpoint.policy.observations) == calls


def test_nonfinite_ensemble_is_rejected_before_publishing_history():
    endpoint = _Endpoint(ensemble=-1000.0)
    first = endpoint.infer("first")
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(PolicyProtocolError):
            endpoint.infer("invalid", Continuation("first", 1))
    assert set(endpoint.session.plans) == {"first"}
    calls = len(endpoint.policy.observations)
    with pytest.raises(PolicyProtocolError):
        endpoint.infer("bad-reference", Continuation("invalid", 0))
    assert len(endpoint.policy.observations) == calls
    # A fresh timeline can still retrieve the previously published row.
    endpoint.infer("retry", dispatched=LastDispatched("first", 0))
    np.testing.assert_array_equal(
        endpoint.policy.observations[-1].last_dispatched_action, first.actions[0]
    )
