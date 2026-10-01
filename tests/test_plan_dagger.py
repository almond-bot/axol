"""Real loopback transport with fake sensors; never commands hardware."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from almond_axol.policy.plan_dagger import PlanDaggerPolicy
from almond_axol.policy.plan_scheduler import (
    PlanRuntimeConfig,
    PlanSchedulingError,
    SensorTimingError,
)
from tests.plan_peer import PlanPeer


def wait_until(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("Timed out waiting for background work")


class RecordingPolicy:
    def __init__(self):
        self.observations = []
        self.resets = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_call = None

    def setup(self, spec):
        self.spec = spec

    def reset(self):
        self.resets += 1

    def infer(self, observation):
        self.observations.append(observation)
        call = len(self.observations)
        if call == self.block_call:
            self.entered.set()
            assert self.release.wait(timeout=5), "test forgot to release inference"
        return np.full((8, 2), call, dtype=np.float32)


def make_robot():
    def capture():
        now = time.perf_counter_ns()
        return (
            {"joint.pos": 0.2, "effort": 0.4, "eye": np.zeros((2, 3, 3), np.uint8)},
            now - 1_000_000,
            {"eye": now - 2_000_000},
        )

    return SimpleNamespace(
        observation_features={"joint.pos": float, "effort": float, "eye": (2, 3, 3)},
        action_features={"target": float, "gripper": float},
        get_observation_with_sensor_timestamps=mock.Mock(side_effect=capture),
        connect=mock.Mock(side_effect=AssertionError("no hardware connect")),
        send_action=mock.Mock(side_effect=AssertionError("no hardware commands")),
    )


@contextmanager
def session(policy=None, robot=None, **options):
    policy = policy or RecordingPolicy()
    robot = robot or make_robot()
    server = PlanPeer(policy, host="127.0.0.1", port=0)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    config = options.pop(
        "config",
        PlanRuntimeConfig(
            request_interval=2,
            max_adoption_offset_steps=2,
            startup_observation_timeout_s=0.1,
        ),
    )
    backend = PlanDaggerPolicy(
        f"ws://127.0.0.1:{server.port}", fps=100, horizon=8, config=config, **options
    )
    try:
        backend.connect(robot)
        yield backend, policy, robot
    finally:
        policy.release.set()
        backend.close()
        server.shutdown()
        serving.join(timeout=3)
        assert not serving.is_alive()
        assert backend._worker is None or not backend._worker.is_alive()
        robot.connect.assert_not_called()
        robot.send_action.assert_not_called()


def next_action(backend):
    return wait_until(backend.act)


def test_connect_is_inactive_and_schema_is_independent():
    with session() as (backend, policy, robot):
        time.sleep(0.04)
        robot.get_observation_with_sensor_timestamps.assert_not_called()
        assert not policy.resets
        assert not policy.observations
        assert backend.act() is None
        assert policy.spec.state_names == ("joint.pos", "effort")
        assert policy.spec.action_names == ("target", "gripper")
        assert policy.spec.dispatch_feedback
        backend.set_instruction("recording label, desktop chooses actual task")
        backend.reset()
        assert next_action(backend) == {"target": 1.0, "gripper": 1.0}
        assert not hasattr(policy.observations[0], "task")
        backend.note_dispatched()


def test_takeover_and_handback_do_not_wait_for_inflight_inference():
    policy = RecordingPolicy()
    policy.block_call = 1
    with session(policy) as (backend, policy, _robot):
        backend.reset()
        assert policy.entered.wait(timeout=3)
        started = time.perf_counter()
        backend.set_intervention(True)
        assert time.perf_counter() - started < 0.1
        assert backend.act() is None
        backend.set_intervention(False)
        assert backend.act() is None
        # No reset can overtake the first infer on this stream.
        assert policy.resets == 1
        policy.release.set()
        action = next_action(backend)
        assert action["target"] == 2.0
        backend.note_dispatched()
        assert policy.resets == 2
        assert len(policy.observations) == 2
        assert policy.observations[-1].continuation is None
        assert policy.observations[-1].last_dispatched is None


def test_shadow_repeats_without_dispatch_and_handback_uses_fresh_prediction():
    with session() as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        backend.note_dispatched()
        backend.set_intervention(True)
        wait_until(lambda: len(policy.observations) >= 4)
        assert backend.act() is None
        shadow = policy.observations[1:]
        assert all(
            obs.continuation is None and obs.last_dispatched is None for obs in shadow
        )
        assert len({obs.request_id for obs in shadow}) == len(shadow)
        assert len({obs.state_sample_time_ns for obs in shadow}) == len(shadow)
        backend.set_intervention(False)
        handback_generation = backend._scheduler.generation
        action = next_action(backend)
        assert action["target"] >= 5.0
        assert backend._scheduler.generation == handback_generation
        assert policy.resets == 3
        backend.note_dispatched()


def test_feedback_is_confirmed_reserved_row_even_after_a_new_plan_is_adopted():
    policy = RecordingPolicy()
    policy.block_call = 2
    with session(policy) as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        backend.note_dispatched()
        backend.act()
        backend.note_dispatched()
        assert policy.entered.wait(timeout=3)
        old_id = policy.observations[0].request_id
        assert policy.observations[1].continuation.prediction_id == old_id
        assert policy.observations[1].continuation.from_row == 2
        assert policy.observations[1].last_dispatched.prediction_id == old_id
        assert policy.observations[1].last_dispatched.row == 1
        # Reserve row 2 before the second plan arrives. Complete the local
        # hardware send only after the accepted-plan ID has changed.
        assert backend.act()["target"] == 1.0
        policy.release.set()
        wait_until(
            lambda: backend._scheduler.prediction_id
            == policy.observations[1].request_id
        )
        backend.note_dispatched()
        assert backend._last_dispatched[1].prediction_id == old_id
        assert backend._last_dispatched[1].row == 2


def test_unconfirmed_dispatch_cannot_be_mislabeled_or_resurrected_after_takeover():
    with session(shadow_inference=False) as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        assert backend._last_dispatched is None
        with pytest.raises(RuntimeError, match="note_dispatched"):
            backend.act()
        backend.set_intervention(True)
        backend.note_dispatched()
        assert backend._last_dispatched is None
        time.sleep(0.05)
        assert len(policy.observations) == 1
        backend.set_intervention(False)
        next_action(backend)
        backend.note_dispatched()
        assert policy.observations[-1].last_dispatched is None


def test_pause_discards_pending_reply_and_stops_sensor_acquisition():
    policy = RecordingPolicy()
    policy.block_call = 1
    with session(policy) as (backend, policy, robot):
        backend.reset()
        assert policy.entered.wait(timeout=3)
        backend.pause()
        captures = robot.get_observation_with_sensor_timestamps.call_count
        policy.release.set()
        time.sleep(0.05)
        assert backend.act() is None
        assert robot.get_observation_with_sensor_timestamps.call_count == captures
        assert backend._scheduler.actions is None
        backend.reset()
        assert next_action(backend)["target"] == 2.0
        backend.note_dispatched()


def test_close_cancels_a_blocked_receive_with_bounded_wait():
    policy = RecordingPolicy()
    policy.block_call = 1
    with session(policy) as (backend, policy, _robot):
        backend.reset()
        assert policy.entered.wait(timeout=3)
        started = time.perf_counter()
        backend.close()
        assert time.perf_counter() - started < 2.5
        assert backend.act() is None
        assert not backend._worker.is_alive()


def test_initial_stale_sensors_are_reacquired_without_changing_timestamps():
    robot = make_robot()
    original = robot.get_observation_with_sensor_timestamps.side_effect
    calls = 0

    def capture():
        nonlocal calls
        calls += 1
        raw, state, cameras = original()
        return raw, state - 10_000_000_000 if calls == 1 else state, cameras

    robot.get_observation_with_sensor_timestamps.side_effect = capture
    with session(robot=robot) as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        backend.note_dispatched()
        assert calls == 2
        assert len(policy.observations) == 1
        assert (
            time.perf_counter_ns() - policy.observations[0].state_sample_time_ns < 1e9
        )


def test_invalid_sensor_timestamp_is_fatal_in_shadow_mode():
    robot = make_robot()
    original = robot.get_observation_with_sensor_timestamps.side_effect

    def capture():
        raw, state, cameras = original()
        return raw, state + 10_000_000_000, cameras

    robot.get_observation_with_sensor_timestamps.side_effect = capture
    with session(robot=robot) as (backend, policy, _robot):
        backend.set_intervention(True)
        wait_until(lambda: backend._fatal_error is not None)
        with pytest.raises(RuntimeError, match="Remote DAgger") as error:
            backend.check_error()
        assert isinstance(error.value.__cause__, SensorTimingError)
        assert not policy.observations


def test_image_preparation_cannot_block_takeover_or_publish_old_generation():
    entered, release = threading.Event(), threading.Event()

    def prepare(images, _config):
        entered.set()
        assert release.wait(timeout=3)
        return images

    with session(shadow_inference=False) as (backend, policy, _robot):
        with mock.patch("almond_axol.policy.plan_dagger.prepare_plan_images", prepare):
            backend.reset()
            assert entered.wait(timeout=3)
            started = time.perf_counter()
            backend.set_intervention(True)
            assert time.perf_counter() - started < 0.1
            assert backend.act() is None
            release.set()
            time.sleep(0.04)
        assert not policy.observations


def test_late_reply_enters_blocking_refresh_without_retiming_old_rows():
    policy = RecordingPolicy()
    policy.block_call = 2
    with session(policy) as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        backend.note_dispatched()
        backend.act()
        backend.note_dispatched()
        assert policy.entered.wait(timeout=3)
        for _ in range(3):
            assert backend.act()["target"] == 1.0
            backend.note_dispatched()
        policy.release.set()
        wait_until(lambda: len(policy.observations) >= 3)
        assert backend._scheduler.blocking
        assert policy.resets == 2
        assert policy.observations[2].continuation is None
        assert policy.observations[2].last_dispatched is None
        assert next_action(backend)["target"] == 3.0
        backend.note_dispatched()
        assert backend.act()["target"] == 3.0
        backend.note_dispatched()
        # Exhausting a blocking segment invalidates before the local send
        # completes, so its final row must not repopulate a cleared cache.
        assert backend._last_dispatched is None
        wait_until(lambda: len(policy.observations) >= 4)
        assert policy.resets == 3
        assert policy.observations[3].continuation is None
        assert policy.observations[3].last_dispatched is None


def test_control_deadline_overrun_invalidates_locally_before_remote_reset():
    policy = RecordingPolicy()
    policy.block_call = 2
    with session(policy) as (backend, policy, _robot):
        backend.reset()
        next_action(backend)
        backend.note_dispatched()
        backend.act()
        backend.note_dispatched()
        assert policy.entered.wait(timeout=3)
        backend.note_control_overrun("test missed dispatch")
        assert backend.act() is None
        assert backend._scheduler.blocking
        assert backend._scheduler.last_recovery == "test missed dispatch"
        assert backend._last_dispatched is None
        assert policy.resets == 1
        policy.release.set()
        assert next_action(backend)["target"] == 3.0
        backend.note_dispatched()
        assert policy.resets == 2


class CartesianPolicy(RecordingPolicy):
    def infer(self, observation):
        super().infer(observation)
        actions = np.zeros((8, 3), dtype=np.float32)
        actions[:, 0] = 1.0 if len(self.observations) == 1 else 20.0
        actions[1, 0] += 0.001
        actions[2, 0] += 1.0
        return actions


def cartesian_robot():
    robot = make_robot()
    robot.action_features = dict.fromkeys(
        ("left_ee.x", "left_ee.y", "left_ee.z"), float
    )
    return robot


def test_cartesian_jump_is_rejected_before_consuming_a_row():
    policy = CartesianPolicy()
    policy.block_call = 2
    with session(policy, cartesian_robot()) as (backend, policy, _robot):
        backend.reset()
        assert next_action(backend)["left_ee.x"] == 1.0
        assert backend._last_target is None
        backend.note_dispatched()
        assert backend.act()["left_ee.x"] == pytest.approx(1.001)
        backend.note_dispatched()
        assert policy.entered.wait(timeout=3)
        tick = backend._scheduler.next_tick
        with pytest.raises(PlanSchedulingError, match="Cartesian step"):
            backend.act()
        assert backend._scheduler.next_tick == tick
        assert backend._reserved is None


def test_handback_clears_prior_policy_step_reference_for_downstream_joint_ramp():
    with session(CartesianPolicy(), cartesian_robot(), shadow_inference=False) as (
        backend,
        _policy,
        _robot,
    ):
        backend.reset()
        assert next_action(backend)["left_ee.x"] == 1.0
        backend.note_dispatched()
        assert backend._last_target is not None
        backend.set_intervention(True)
        assert backend._last_target is None
        backend.set_intervention(False)
        assert next_action(backend)["left_ee.x"] == 20.0
        backend.note_dispatched()
