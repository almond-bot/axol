"""Remote DAgger terminal ownership and parking without hardware or stdin."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from almond_axol.cli import collect_dagger
from almond_axol.cli.dagger_terminal import DaggerStdinControl
from almond_axol.cli.run_policy import _GATE_CONTACT, _GATE_READY


class ScriptedTerminal(DaggerStdinControl):
    def __init__(self, mode, events):
        super().__init__()
        self.mode = mode
        self.events = events

    def _start_reader(
        self, on_subtask=None, num_subtasks=0, *, allowed_choices=("s", "r", "q")
    ):
        self.events.append(("reader", self._gate_phase, allowed_choices))
        self._result = {"choice": None}
        if allowed_choices == ("q",):
            if self.mode == "idle_q":
                self._result["choice"] = "q"
            elif self.mode == "idle_eof":
                self._result["choice"] = "abort"
            elif self.mode == "contact_q" and self._gate_phase == _GATE_CONTACT:
                self._result["choice"] = "q"

    def end_gate(self):
        if self._gate_active:
            self.events.append(("end gate", self._gate_phase))
            if self.mode == "record_race" and self._gate_phase == _GATE_READY:
                self._result["choice"] = "q"
            if self.mode == "contact_race" and self._gate_phase == _GATE_CONTACT:
                self._result["choice"] = "q"
        super().end_gate()


def test_idle_quit_wins_over_simultaneous_vr_record():
    teleop = Mock()
    teleop.get_teleop_events.return_value = {"start_recording": True}
    robot, home = Mock(), Mock()
    control = ScriptedTerminal("idle_q", [])
    control.begin_gate("ready")

    assert collect_dagger._idle_teleop_until_record(
        teleop, robot, home, 120, control, threading.Event()
    ) == (False, False)

    teleop.get_teleop_events.assert_not_called()
    robot.send_action.assert_not_called()
    home.assert_not_called()


def test_idle_teleop_remains_live_until_quit():
    teleop = Mock()
    teleop.get_teleop_events.return_value = {}
    teleop.consume_idle_reset.return_value = False
    teleop.teleop_engaged = True
    teleop.get_action.return_value = {"left_joint.pos": 0.2}
    robot, home = Mock(), Mock()
    control = Mock()
    control.poll_gate.side_effect = [None, "quit"]

    with patch.object(collect_dagger.time, "sleep"):
        assert collect_dagger._idle_teleop_until_record(
            teleop, robot, home, 120, control, threading.Event()
        ) == (True, False)

    robot.send_action.assert_called_once_with({"left_joint.pos": 0.2})
    home.assert_not_called()


def test_idle_reader_join_keeps_measured_joint_hold_alive():
    held = threading.Event()
    control = ScriptedTerminal("record_race", [])
    control.begin_gate("ready")
    end_gate = control.end_gate

    def wait_for_hold():
        assert held.wait(timeout=1)
        end_gate()

    control.end_gate = wait_for_hold
    robot = SimpleNamespace(
        positions=([0.25], [-0.5]),
        _left_pos_keys=["left_joint.pos"],
        _right_pos_keys=["right_joint.pos"],
        send_action=Mock(side_effect=lambda action: held.set()),
    )

    result = collect_dagger._finish_idle_terminal_gate(control, robot, 120)

    assert result == "quit"
    assert control.quit_requested
    robot.send_action.assert_called_with(
        {"left_joint.pos": 0.25, "right_joint.pos": -0.5}
    )


@pytest.mark.parametrize(
    ("mode", "should_park"),
    [
        ("idle_q", True),
        ("idle_eof", False),
        ("record_race", True),
        ("contact_q", False),
        ("contact_race", False),
        ("active_q", True),
        ("active_q_capture_error", True),
        ("active_q_finish_error", True),
        ("active_eof", False),
    ],
)
def test_remote_collector_quit_lifecycle(tmp_path, mode, should_park):
    config = collect_dagger.DaggerConfig(
        policy_type="custom",
        task="test task",
        repo_id="local/test",
        root=str(tmp_path / "dataset"),
        hold_to_intervene=True,
        record_joint_actions=True,
        start_from_current_pose=True,
    )
    config.robot_config.cameras["overhead"].serial = 1234
    config.robot_config.action_space = "cartesian"
    events = []
    control = ScriptedTerminal(mode, events)
    robot, reset, policy, teleop, recorder, worker = (Mock() for _ in range(6))
    robot._left_pos_keys = ["left_joint.pos"]
    robot._right_pos_keys = ["right_joint.pos"]
    robot.positions = ([0.2], [-0.2])
    robot.get_joint_observation.return_value = {}
    robot.disconnect.side_effect = lambda: events.append("disable")
    robot.disconnect_preserving_position.side_effect = lambda: events.append("preserve")
    robot.send_action.side_effect = lambda action: events.append("hold")
    reset.wait_ready.return_value = True
    reset.park.side_effect = lambda *args, **kwargs: events.append("park") or True
    teleop.teleop_engaged = False
    teleop.consume_idle_reset.side_effect = [
        mode.startswith("contact"),
        mode == "contact_race",
    ]
    teleop.get_teleop_events.side_effect = [
        {},  # discard events from startup
        {"start_recording": mode == "record_race" or mode.startswith("active")},
    ]
    recorder.episode_count.return_value = 0
    recorder.finish_episode.return_value = 1
    if mode == "active_q_finish_error":
        recorder.finish_episode.side_effect = collect_dagger.RecorderCaptureError(
            "camera alignment failed while finishing"
        )
    worker.capture_error = (
        "camera alignment failed" if mode == "active_q_capture_error" else None
    )
    worker.fatal_error = None
    worker.open_span_start = None
    worker.vr_choice = None
    worker.ident = None
    worker.is_alive.return_value = False
    worker.shutdown_event = threading.Event()

    def start_worker():
        events.append("start policy control")
        control._result["choice"] = "q" if mode.startswith("active_q") else "abort"

    worker.start.side_effect = start_worker

    def contact_home(*args, **kwargs):
        events.append("contact")
        assert kwargs["wait_retry"]() is False
        return False

    reset.return_to_rest.side_effect = contact_home
    relay = Mock()
    relay.readable_raw_cameras = {"overhead"}
    relay.raw_cameras = {"overhead": Mock()}
    with (
        patch("almond_axol.zed.stereo_serials", return_value=set()),
        patch("almond_axol.lerobot.robot.robot_axol.AxolRobot", return_value=robot),
        patch(
            "almond_axol.lerobot.teleop.teleop_vr_dagger.DaggerVRTeleop",
            return_value=teleop,
        ),
        patch("almond_axol.policy.plan_dagger.PlanDaggerPolicy", return_value=policy),
        patch(
            "almond_axol.recording.datasets.dataset_features_for_robot", return_value={}
        ),
        patch(
            "almond_axol.cli.plan_dagger_control.PlanDaggerControlLoop",
            return_value=worker,
        ),
        patch.object(collect_dagger, "IKResetController", return_value=reset),
        patch.object(collect_dagger, "_start_video_relay", return_value=relay),
        patch.object(collect_dagger, "DatasetRecorderProcess", return_value=recorder),
        patch.object(collect_dagger, "restore_dataset_ownership"),
        patch.object(collect_dagger.affinity, "pin_realtime"),
        patch("os.sched_getaffinity", return_value={0}),
        patch("os.sched_setaffinity"),
        patch("signal.signal"),
    ):
        collect_dagger._run(config, stop_event=threading.Event(), control=control)

    if should_park:
        assert events.index("park") < events.index("disable")
        robot.disconnect_preserving_position.assert_not_called()
    else:
        robot.disconnect_preserving_position.assert_called_once_with()
        reset.park.assert_not_called()
        robot.disconnect.assert_not_called()
    if mode.startswith("active"):
        worker.start.assert_called_once_with()
        recorder.cancel_episode.assert_called_once_with()
        reset.return_to_rest.assert_not_called()
    else:
        policy.reset.assert_not_called()
        worker.start.assert_not_called()
        recorder.start_episode.assert_not_called()
    if mode.startswith("contact"):
        # A contact abort must retain limp support, without even the idle
        # reader's measured-joint heartbeat restoring impedance afterward.
        assert "hold" not in events[events.index("contact") :]
        assert control.abort_requested
        assert not control.quit_requested
    policy.close.assert_called_once_with()
    assert control._gate_active is False
