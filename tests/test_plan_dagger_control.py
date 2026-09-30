"""Command arbitration, transition shaping and recorder fences without motors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from lerobot.teleoperators.utils import TeleopEvents

from almond_axol.cli import plan_dagger_control
from almond_axol.cli.collect_dagger import _STATE_POLICY, _STATE_TELEOP
from almond_axol.cli.plan_dagger_control import PlanDaggerControlLoop
from almond_axol.policy.handover import JointHandover

ANCHOR = {"left_joint.pos": 0.2, "right_joint.pos": -0.1, "left_gripper.pos": 0.3}
RESOLVED = {"left_joint.pos": 2.0, "right_joint.pos": -1.0, "left_gripper.pos": 0.8}
CARTESIAN = {"left_ee.x": 0.45}


@pytest.fixture(autouse=True)
def synchronous_recorder_fence():
    # The helper's periodic hold machinery has separate control-loop tests.
    # Keep these checks about source arbitration and fence ordering.
    with mock.patch.object(
        plan_dagger_control,
        "run_blocking_with_sync_control_ticks",
        side_effect=lambda fn, *_args, **_kwargs: fn(),
    ):
        yield


def make_loop():
    events, sent, published = [], [], []
    teleop = SimpleNamespace(
        teleop_engaged=False,
        grips_held_raw=False,
        get_action=mock.Mock(return_value=dict(RESOLVED)),
        get_teleop_events=mock.Mock(return_value={}),
    )
    policy = SimpleNamespace(
        check_error=mock.Mock(),
        act=mock.Mock(return_value=dict(CARTESIAN)),
        set_intervention=mock.Mock(
            side_effect=lambda value: events.append(("source", value))
        ),
        note_dispatched=mock.Mock(side_effect=lambda: events.append(("confirmed",))),
        note_control_overrun=mock.Mock(),
        pause=mock.Mock(),
    )

    def publish(state, action, stamp, *, intervention):
        events.append(("publish", intervention))
        published.append((state, dict(action), stamp, intervention))

    recorder = SimpleNamespace(
        pause_episode=mock.Mock(side_effect=lambda: events.append(("pause",)) or 60),
        resume_episode=mock.Mock(side_effect=lambda: events.append(("resume",)) or 60),
        publish=mock.Mock(side_effect=publish),
        poll_capture_error=mock.Mock(return_value=None),
    )
    robot = SimpleNamespace(
        reset_cartesian_seed=mock.Mock(),
        get_joint_observation=mock.Mock(return_value={"left_joint.pos": 0.19}),
        action_to_dataset=mock.Mock(
            side_effect=lambda action: {"left_ee.x": action["left_joint.pos"]}
        ),
        before_transform=None,
    )

    def send(action, *, joint_transform=None):
        resolved = dict(RESOLVED) if "left_ee.x" in action else dict(action)
        if joint_transform is not None:
            if robot.before_transform is not None:
                robot.before_transform()
            resolved = joint_transform(resolved)
        sent.append(dict(resolved))
        events.append(("sent",))
        return resolved

    robot.send_action = mock.Mock(side_effect=send)
    loop = PlanDaggerControlLoop(
        robot=robot,
        policy=policy,
        teleop=teleop,
        recorder=recorder,
        fps=30,
        teleop_hz=120,
        initial_action=ANCHOR,
        handover_duration_s=8 / 30,
    )
    return SimpleNamespace(
        loop=loop,
        robot=robot,
        policy=policy,
        teleop=teleop,
        recorder=recorder,
        events=events,
        sent=sent,
        published=published,
    )


def test_policy_dispatch_records_resolved_sent_joints_without_mutating_plan():
    env = make_loop()
    requested = env.policy.act.return_value
    env.loop.tick(12.0)
    assert env.sent == [ANCHOR]
    assert requested == CARTESIAN
    assert env.published[0][1] == env.sent[0]
    assert env.published[0][3] is False
    assert env.loop.hold_action == ANCHOR
    env.policy.act.assert_called_once_with(None)
    env.policy.note_dispatched.assert_called_once_with()
    env.robot.action_to_dataset.assert_not_called()
    assert env.events.index(("sent",)) < env.events.index(("confirmed",))
    env.teleop.get_action.assert_not_called()


def test_pending_inference_holds_and_records_without_advancing_handover():
    env = make_loop()
    env.policy.act.return_value = None
    env.loop.tick(12.0)
    env.loop.tick(12.1)
    assert env.sent == [ANCHOR, ANCHOR]
    assert env.loop.handover.step == 0
    env.policy.note_dispatched.assert_not_called()
    assert all(not row[3] for row in env.published)


def test_raw_grip_takes_ownership_before_ik_engages_and_fences_recording():
    env = make_loop()
    env.teleop.grips_held_raw = True
    env.loop.tick(12.0)
    assert env.loop.state == _STATE_TELEOP
    env.policy.set_intervention.assert_called_once_with(True)
    env.policy.act.assert_not_called()
    env.teleop.get_action.assert_not_called()
    assert env.sent == [ANCHOR]
    assert env.published[0][3] is True
    assert env.events == [
        ("source", True),
        ("pause",),
        ("sent",),
        ("publish", True),
        ("resume",),
    ]
    assert env.loop.open_span_start == 2.0


def test_forced_disengagement_retains_operator_until_grips_release():
    env = make_loop()
    env.teleop.grips_held_raw = env.teleop.teleop_engaged = True
    env.loop.tick(12.0)
    assert env.sent[-1] == RESOLVED
    env.teleop.teleop_engaged = False
    env.loop.tick(12.1)
    assert env.loop.state == _STATE_TELEOP
    assert env.sent[-1] == RESOLVED
    assert env.teleop.get_action.call_count == 1
    env.policy.act.assert_not_called()
    assert env.policy.set_intervention.call_args_list == [mock.call(True)]
    # Only an explicit release requests a fresh policy generation. While its
    # bootstrap is pending, hold the final operator command unchanged.
    env.teleop.grips_held_raw = False
    env.policy.act.return_value = None
    env.loop.tick(12.2)
    assert env.loop.state == _STATE_POLICY
    assert env.policy.set_intervention.call_args_list == [
        mock.call(True),
        mock.call(False),
    ]
    assert env.sent[-1] == RESOLVED
    assert env.published[-1][3] is False
    assert env.loop.handover.anchor == RESOLVED
    assert env.robot.reset_cartesian_seed.call_count == 2
    assert env.loop.intervention_spans == [(2.0, 2.0)]
    assert env.events[-5:] == [
        ("source", False),
        ("pause",),
        ("sent",),
        ("publish", False),
        ("resume",),
    ]


def test_takeover_arriving_during_ik_prevents_policy_motor_send_and_confirmation():
    env = make_loop()
    env.robot.before_transform = lambda: setattr(env.teleop, "grips_held_raw", True)
    env.loop.tick(12.0)
    assert env.sent == [ANCHOR]
    assert env.loop.state == _STATE_TELEOP
    env.policy.note_dispatched.assert_not_called()
    assert env.published[0][3] is True
    assert env.loop.handover.step == 0


def test_stop_arriving_during_ik_prevents_all_sends_and_confirmation():
    env = make_loop()
    env.robot.before_transform = env.loop.shutdown_event.set
    env.loop.tick(12.0)
    assert not env.sent
    assert not env.published
    env.policy.note_dispatched.assert_not_called()
    env.policy.set_intervention.assert_not_called()


def test_failed_motor_send_cannot_confirm_or_record_dispatch():
    env = make_loop()
    env.robot.send_action.side_effect = RuntimeError("motor rejected send")
    with pytest.raises(RuntimeError, match="motor rejected"):
        env.loop.tick(12.0)
    env.policy.note_dispatched.assert_not_called()
    env.recorder.publish.assert_not_called()


@pytest.mark.parametrize(
    ("event", "choice"),
    [(TeleopEvents.TERMINATE_EPISODE, "s"), (TeleopEvents.RERECORD_EPISODE, "r")],
)
def test_run_honors_episode_events_and_always_pauses_backend(event, choice):
    env = make_loop()
    env.teleop.get_teleop_events.return_value = {event: True}
    env.recorder.poll_capture_error.side_effect = [None, "unexpected second tick"]
    with mock.patch.object(plan_dagger_control.affinity, "enter_control_thread"):
        env.loop.run()
    assert env.loop.fatal_error is None
    assert env.loop.vr_choice == choice
    env.policy.pause.assert_called_once_with()
    assert not env.sent


@pytest.mark.parametrize("failure", ["capture", "policy", "stop"])
def test_all_shutdown_paths_pause_background_inference(failure):
    env = make_loop()
    if failure == "capture":
        env.recorder.poll_capture_error.return_value = "sensor mismatch"
    elif failure == "policy":
        env.policy.check_error.side_effect = RuntimeError("background inference failed")
    else:
        env.loop.shutdown_event.set()
    with mock.patch.object(plan_dagger_control.affinity, "enter_control_thread"):
        env.loop.run()
    env.policy.pause.assert_called_once_with()
    assert not env.sent
    if failure == "capture":
        assert env.loop.capture_error == "sensor mismatch"
    elif failure == "policy":
        assert isinstance(env.loop.fatal_error, RuntimeError)


def test_handover_anchors_first_output_preserves_inputs_and_bounds_convergence():
    transition = JointHandover(fps=30, duration_s=8 / 30, max_vel=0.8, max_accel=2.0)
    held, target = dict(ANCHOR), dict(RESOLVED)
    transition.seed(held)
    held["left_joint.pos"] = -99.0
    outputs = [transition.apply(target) for _ in range(250)]
    assert outputs[0] == ANCHOR
    assert target == RESOLVED
    assert outputs[8]["left_joint.pos"] != pytest.approx(RESOLVED["left_joint.pos"])
    trace = np.asarray(
        [ANCHOR["left_joint.pos"], *[row["left_joint.pos"] for row in outputs]]
    )
    velocity = np.diff(trace) * 30
    acceleration = np.diff(velocity) * 30
    assert np.max(np.abs(velocity)) <= 0.8 + 1e-5
    assert np.max(np.abs(acceleration)) <= 2.0 + 1e-3
    assert outputs[-1] == RESOLVED
    assert transition.anchor is None


def test_handover_reseeding_uses_last_operator_command_and_validates_schema():
    transition = JointHandover(fps=30, duration_s=8 / 30)
    transition.seed(ANCHOR)
    for _ in range(4):
        transition.apply(RESOLVED)
    operator = {key: value + 0.1 for key, value in ANCHOR.items()}
    transition.seed(operator)
    assert transition.apply(RESOLVED) == operator
    with pytest.raises(ValueError, match="schema"):
        transition.apply({"wrong_joint.pos": 0.0})
    with pytest.raises(ValueError, match="finite"):
        transition.seed({"left_joint.pos": float("nan")})
