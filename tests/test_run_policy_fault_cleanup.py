"""Session failure must not turn a healthy core's supporting torque off."""

from contextlib import ExitStack
from threading import Event
from types import SimpleNamespace
from unittest import mock

import pytest

from almond_axol.cli import run_policy
from almond_axol.policy.plan_scheduler import PlanSchedulingError
from almond_axol.robot.base import HardwareCleanupError


def run_session(
    *,
    fault=None,
    acknowledgment=True,
    workers_stopped=True,
    stop_at_cleanup=False,
    connect_error=None,
    initial_continue=True,
    soft_park=False,
    gate_choices=None,
    episode_choice="q",
    timeout_choice="q",
    park_error=None,
    quit_requested=False,
    late_episode_choice=None,
    pre_episode_error=None,
):
    events = []
    stopped = Event()
    robot = mock.Mock(config=SimpleNamespace(observe_cartesian=False))
    robot.connect.side_effect = connect_error
    robot.disconnect.side_effect = lambda: events.append("disable")
    robot.disconnect_preserving_position.side_effect = lambda: events.append("preserve")
    reset = mock.Mock()
    reset.return_to_rest.return_value = True

    def park(*args, **kwargs):
        assert events[-1] == "client stopped"
        assert workers_stopped
        events.append("park")
        if park_error is not None:
            raise park_error
        return True

    reset.park.side_effect = park
    client = mock.Mock(fatal_error=fault, contact_tripped=None)
    client.start.return_value = True
    if soft_park:
        client.stop.side_effect = lambda: events.append("client stopped")
    control = mock.Mock()
    control.quit_requested = quit_requested
    control.poll_choice.return_value = None if fault else episode_choice
    control.resolve_timeout.return_value = timeout_choice
    gates = iter(gate_choices) if gate_choices is not None else None

    def await_continue(message, **kwargs):
        if "Policy failed" not in message:
            if gates is not None:
                decision = next(gates)
                if isinstance(decision, BaseException):
                    raise decision
                control.quit_requested = decision == "q"
                return decision == "continue"
            return initial_continue
        events.append("acknowledge")
        # The core is still connected during the acknowledgment gate.
        robot.disconnect.assert_not_called()
        robot.disconnect_preserving_position.assert_not_called()
        if isinstance(acknowledgment, BaseException):
            raise acknowledgment
        return acknowledgment

    control.await_continue.side_effect = await_continue

    def stop_workers(**kwargs):
        events.append("workers stopped" if workers_stopped else "workers alive")
        if late_episode_choice is not None:
            control.poll_choice.return_value = late_episode_choice
        if stop_at_cleanup:
            stopped.set()
        return (
            workers_stopped,
            None
            if workers_stopped
            else HardwareCleanupError("live observation worker"),
        )

    cfg = run_policy.RunPolicyConfig(
        policy_type="custom",
        task="test",
        custom_protocol=2,
        robot_config=object(),
        actions_per_chunk=30,
        soft_park_on_quit=soft_park,
        rest_pose_left=[-0.8708, 0, 0, 1.395, 0, 0, 0.3442],
        rest_pose_right=[0.8784, 0, 0, -1.403, 0, 0, -0.3408],
    )
    with ExitStack() as stack:
        for target, value in (
            ("almond_axol.lerobot.robot.robot_axol.AxolRobot", robot),
            ("lerobot.processor.make_default_processors", (None, None, None)),
            ("lerobot.async_inference.configs.RobotClientConfig", object()),
        ):
            stack.enter_context(mock.patch(target, return_value=value))
        reset_constructor = stack.enter_context(
            mock.patch.object(run_policy, "IKResetController", return_value=reset)
        )
        stack.enter_context(
            mock.patch.object(
                run_policy, "_build_axol_robot_client", return_value=client
            )
        )
        stack.enter_context(mock.patch.object(run_policy, "ActionPublisher"))
        stack.enter_context(mock.patch.object(run_policy.threading, "Thread"))
        stack.enter_context(
            mock.patch.object(
                run_policy, "_stop_episode_workers", side_effect=stop_workers
            )
        )
        stack.enter_context(mock.patch.object(run_policy.time, "sleep"))
        if episode_choice is None and fault is None:
            from itertools import count

            stack.enter_context(
                mock.patch.object(
                    run_policy.time, "perf_counter", side_effect=count(0, 1000)
                )
            )
        stack.enter_context(
            mock.patch.object(run_policy.gc, "collect", side_effect=pre_episode_error)
        )
        stack.enter_context(mock.patch.object(run_policy.gc, "disable"))
        stack.enter_context(mock.patch.object(run_policy.gc, "enable"))
        stack.enter_context(mock.patch("signal.signal"))
        raised = None
        try:
            run_policy._run(cfg, stop_event=stopped, control=control)
        except BaseException as exc:
            raised = exc
        reset_constructor.assert_called_once_with(
            rest_pose_left=cfg.rest_pose_left, rest_pose_right=cfg.rest_pose_right
        )
    return SimpleNamespace(
        robot=robot,
        reset=reset,
        client=client,
        control=control,
        events=events,
        raised=raised,
    )


@pytest.mark.parametrize(
    "acknowledgment",
    [True, False, EOFError(), KeyboardInterrupt(), RuntimeError("panel failed")],
)
@pytest.mark.parametrize(
    "fault",
    [
        PlanSchedulingError("state observation is stale"),
        ConnectionError("server disconnected"),
    ],
)
def test_policy_fault_preserves_support_and_original_failure(fault, acknowledgment):
    result = run_session(fault=fault, acknowledgment=acknowledgment)
    assert result.raised is fault
    assert result.events == ["workers stopped", "acknowledge", "preserve"]
    result.robot.disconnect.assert_not_called()
    result.robot.disconnect_preserving_position.assert_called_once_with()
    result.robot.send_action.assert_not_called()
    result.reset.return_to_rest.assert_called_once()  # initial setup only
    result.reset.hold_limp.assert_not_called()


def test_stop_request_after_fault_preserves_support_without_waiting_for_input():
    fault = PlanSchedulingError("stale")
    result = run_session(fault=fault, stop_at_cleanup=True)
    assert result.raised is fault
    assert result.events == ["workers stopped", "preserve"]
    result.robot.disconnect.assert_not_called()


def test_live_worker_prevents_gate_and_both_forms_of_hardware_teardown():
    result = run_session(fault=PlanSchedulingError("stale"), workers_stopped=False)
    assert isinstance(result.raised, HardwareCleanupError)
    assert result.events == ["workers alive"]
    result.robot.disconnect.assert_not_called()
    result.robot.disconnect_preserving_position.assert_not_called()


def test_deliberate_normal_exit_retains_existing_cleanup():
    result = run_session(initial_continue=False)
    assert result.raised is None
    assert result.events == ["disable"]
    result.robot.disconnect_preserving_position.assert_not_called()


def test_partial_connect_failure_uses_existing_startup_rollback():
    error = RuntimeError("partial bring-up failed")
    result = run_session(connect_error=error)
    assert result.raised is error
    assert result.events == ["disable"]
    result.robot.disconnect_preserving_position.assert_not_called()
