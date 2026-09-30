"""Explicit quit parks only after every policy writer has relinquished ownership."""

from threading import Event
from unittest import mock

import pytest

from almond_axol.cli.run_policy import _QueuePolicyControl, _StdinPolicyControl
from almond_axol.lerobot.rollout import stdin_watcher
from almond_axol.policy.plan_scheduler import PlanSchedulingError

from .test_run_policy_fault_cleanup import run_session


@pytest.mark.parametrize("entry", ["first gate", "next gate", "episode", "time cap"])
def test_explicit_quit_parks_after_client_quiescence(entry):
    kwargs = {"soft_park": True}
    if entry == "first gate":
        kwargs["gate_choices"] = ["q"]
    elif entry == "next gate":
        kwargs.update(gate_choices=["continue", "q"], episode_choice="s")
    elif entry == "time cap":
        kwargs.update(episode_choice=None, timeout_choice="q")
    result = run_session(**kwargs)
    assert result.raised is None
    expected = [] if entry == "first gate" else ["workers stopped"]
    assert result.events == expected + ["client stopped", "park", "disable"]
    result.reset.park.assert_called_once()
    result.robot.disconnect_preserving_position.assert_not_called()


@pytest.mark.parametrize("entry", ["first gate", "episode", "time cap", "interrupt"])
def test_eof_or_interrupt_preserves_without_automatic_motion(entry):
    kwargs = {"soft_park": True}
    if entry == "first gate":
        kwargs["gate_choices"] = ["abort"]
    elif entry == "episode":
        kwargs["episode_choice"] = "abort"
    elif entry == "time cap":
        kwargs.update(episode_choice=None, timeout_choice="abort")
    else:
        kwargs["gate_choices"] = [KeyboardInterrupt()]
    result = run_session(**kwargs)
    assert result.raised is None
    assert result.events[-1] == "preserve"
    result.reset.park.assert_not_called()
    result.robot.disconnect.assert_not_called()


@pytest.mark.parametrize(
    "error", [RuntimeError("contact"), TimeoutError("stale"), KeyboardInterrupt()]
)
def test_failed_or_interrupted_park_cannot_release_torque(error):
    result = run_session(soft_park=True, park_error=error)
    assert result.raised is error
    assert result.events.index("workers stopped") < result.events.index("park")
    assert result.events[-1] == "preserve"
    result.robot.disconnect.assert_not_called()


def test_fault_and_quit_cannot_trigger_park():
    fault = PlanSchedulingError("camera/state skew")
    result = run_session(soft_park=True, fault=fault)
    assert result.raised is fault
    result.reset.park.assert_not_called()
    result.robot.disconnect.assert_not_called()
    assert result.events[-1] == "preserve"


def test_live_worker_cannot_park_or_disconnect():
    result = run_session(soft_park=True, workers_stopped=False)
    result.reset.park.assert_not_called()
    result.robot.disconnect.assert_not_called()
    result.robot.disconnect_preserving_position.assert_not_called()


def test_external_stop_does_not_turn_queued_q_into_powered_move():
    result = run_session(soft_park=True, stop_at_cleanup=True)
    result.reset.park.assert_not_called()
    assert result.events[-1] == "preserve"


def test_interrupt_takes_precedence_over_an_already_observed_q():
    result = run_session(
        soft_park=True, quit_requested=True, gate_choices=[KeyboardInterrupt()]
    )
    result.reset.park.assert_not_called()
    assert result.events[-1] == "preserve"


def test_q_arriving_at_time_cap_is_honored_without_another_prompt():
    result = run_session(soft_park=True, episode_choice=None, late_episode_choice="q")
    assert result.raised is None
    result.control.resolve_timeout.assert_not_called()
    assert result.events == ["workers stopped", "client stopped", "park", "disable"]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("GC failed")])
def test_pre_episode_gc_failure_stops_input_reader_before_cleanup(error):
    result = run_session(soft_park=True, pre_episode_error=error)
    result.control.begin_episode.assert_called_once()
    result.control.end_episode.assert_called_once()
    assert "workers stopped" not in result.events  # None were started yet.
    result.reset.park.assert_not_called()
    result.robot.disconnect.assert_not_called()
    assert result.events[-1] == "preserve"


@pytest.mark.parametrize("kind", ["stdin", "queue"])
@pytest.mark.parametrize("entry", ["ready", "contact", "limp", "episode", "time cap"])
def test_control_surfaces_distinguish_normal_quit_from_contact_abort(kind, entry):
    control = _StdinPolicyControl() if kind == "stdin" else _QueuePolicyControl(Event())
    if kind == "queue":
        control.push("q")
    elif entry == "episode":
        control._result["choice"] = "q"
    with mock.patch("builtins.input", return_value=" q "):
        if entry == "episode":
            assert control.poll_choice() == "q"
        elif entry == "time cap":
            assert control.resolve_timeout(10) == "q"
        else:
            assert control.await_continue("Scene", phase=entry) is False
    assert control.quit_requested is (entry not in {"contact", "limp"})


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_terminal_aborts_are_not_explicit_quit(error):
    control = _StdinPolicyControl()
    with mock.patch("builtins.input", side_effect=error):
        control.quit_requested = True  # A late q from the previous reader.
        assert not control.await_continue("Scene")
        assert not control.quit_requested
        control.quit_requested = True
        assert control.resolve_timeout(10) == "abort"
    assert not control.quit_requested


def test_episode_stdin_eof_aborts_without_impersonating_q():
    result = {"choice": None}
    with (
        mock.patch("select.select", return_value=([object()], [], [])),
        mock.patch("sys.stdin.readline", return_value=""),
    ):
        stdin_watcher(Event(), result, eof_choice="abort")
    assert result["choice"] == "abort"


def test_stdin_reader_is_joined_before_next_prompt():
    control = _StdinPolicyControl()
    control._stop = Event()
    control._thread = mock.Mock()
    control._thread.is_alive.return_value = False
    control.end_episode()
    assert control._stop.is_set()
    control._thread.join.assert_called_once_with(timeout=1.0)
