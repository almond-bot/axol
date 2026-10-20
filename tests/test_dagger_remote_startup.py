"""Remote collector startup reaches a ready parking worker before enabling CAN."""

from __future__ import annotations

import threading
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from almond_axol.cli import collect_dagger
from almond_axol.cli.dagger_terminal import DaggerStdinControl


class _ReachedConnect(RuntimeError):
    pass


@pytest.mark.parametrize(
    "ready_result", [True, False, RuntimeError("reset worker failed")]
)
def test_remote_startup_uses_selected_rest_pose_and_waits_before_connect(
    tmp_path, ready_result
):
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
    config.teleop_config.vr_teleop_config.rest_pose_left = [0.1] * 7
    config.teleop_config.vr_teleop_config.rest_pose_right = [-0.2] * 7
    events = []
    robot, reset, policy, teleop = Mock(), Mock(), Mock(), Mock()
    robot._left_pos_keys = [f"left_joint_{index}.pos" for index in range(8)]
    robot._right_pos_keys = [f"right_joint_{index}.pos" for index in range(8)]
    relay = Mock()
    relay.readable_raw_cameras = {"overhead"}
    relay.raw_cameras = {"overhead": Mock()}

    def ready(**kwargs):
        events.append("reset ready")
        if isinstance(ready_result, Exception):
            raise ready_result
        return ready_result

    def connect():
        events.append("robot connect")
        raise _ReachedConnect()

    reset.wait_ready.side_effect = ready
    robot.prepare_cartesian_actions.side_effect = lambda: events.append("IK ready")
    robot.connect.side_effect = connect
    expected_error = (
        type(ready_result)
        if isinstance(ready_result, Exception)
        else _ReachedConnect
        if ready_result
        else None
    )
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
        patch.object(
            collect_dagger, "IKResetController", return_value=reset
        ) as construct,
        patch.object(collect_dagger, "_start_video_relay", return_value=relay),
        patch.object(collect_dagger.affinity, "pin_realtime"),
        patch("os.sched_getaffinity", return_value={0}),
        patch("os.sched_setaffinity"),
        pytest.raises(expected_error) if expected_error else nullcontext(),
    ):
        collect_dagger._run(config, stop_event=threading.Event(), control=Mock())

    construct.assert_called_once_with(
        rest_pose_left=[0.1] * 7,
        rest_pose_right=[-0.2] * 7,
    )
    reset.start.assert_called_once_with()
    reset.wait_ready.assert_called_once()
    assert events == (
        ["reset ready", "IK ready", "robot connect"]
        if ready_result is True
        else ["reset ready"]
    )
    # A handshake failure tears down setup resources before any robot connect.
    if ready_result is not True:
        robot.connect.assert_not_called()
        robot.prepare_cartesian_actions.assert_not_called()
    reset.stop.assert_called_once_with()
    policy.close.assert_called_once_with()


@pytest.fixture
def remote_startup_session(tmp_path):
    """Run the collector supervisor with all device/process owners mocked."""
    config = collect_dagger.DaggerConfig(
        policy_type="custom",
        task="test task",
        repo_id="local/test",
        root=str(tmp_path / "dataset"),
        hold_to_intervene=True,
        record_joint_actions=True,
        home_on_start=True,
        start_from_current_pose=True,
    )
    config.robot_config.cameras["overhead"].serial = 1234
    config.robot_config.action_space = "cartesian"
    events = []
    robot, reset, policy, teleop, recorder, worker = (Mock() for _ in range(6))
    robot._left_pos_keys = ["left_joint.pos"]
    robot._right_pos_keys = ["right_joint.pos"]
    robot.positions = ([0.2], [-0.2])
    robot.get_joint_observation.return_value = {}
    robot.prepare_cartesian_actions.side_effect = lambda: events.append("IK ready")
    robot.connect.side_effect = lambda: events.append("robot connect")
    robot.disconnect.side_effect = lambda: events.append("disable")
    robot.disconnect_preserving_position.side_effect = lambda: events.append("preserve")
    reset.wait_ready.side_effect = lambda **kwargs: events.append("reset ready") or True
    reset.park.side_effect = lambda *args, **kwargs: events.append("park") or True

    def home(*args, **kwargs):
        events.append("home")
        robot.positions = ([0.1], [-0.1])
        return True

    reset.return_to_rest.side_effect = home
    teleop.teleop_engaged = True
    teleop.consume_idle_reset.return_value = False
    teleop.get_teleop_events.side_effect = [
        {},  # clear stale events after startup
        {},  # one idle teleop tick before Record
        {"start_recording": True},
    ]
    teleop.get_action.return_value = {"left_joint.pos": 0.35, "right_joint.pos": -0.45}

    def send_action(action):
        robot.positions = ([action["left_joint.pos"]], [action["right_joint.pos"]])

    robot.send_action.side_effect = send_action
    recorder.episode_count.return_value = 0
    recorder.finish_episode.return_value = 1
    recorder.start_episode.side_effect = lambda task: events.append("start recording")
    worker.capture_error = None
    worker.fatal_error = None
    worker.open_span_start = None
    worker.vr_choice = None
    worker.ident = None
    worker.is_alive.return_value = False
    worker.shutdown_event = threading.Event()
    worker.start.side_effect = lambda: events.append("start policy control")
    relay = Mock()
    relay.readable_raw_cameras = {"overhead"}
    relay.raw_cameras = {"overhead": Mock()}

    with ExitStack() as stack:
        patches = [
            patch("almond_axol.zed.stereo_serials", return_value=set()),
            patch("almond_axol.lerobot.robot.robot_axol.AxolRobot", return_value=robot),
            patch(
                "almond_axol.lerobot.teleop.teleop_vr_dagger.DaggerVRTeleop",
                return_value=teleop,
            ),
            patch(
                "almond_axol.policy.plan_dagger.PlanDaggerPolicy", return_value=policy
            ),
            patch(
                "almond_axol.recording.datasets.dataset_features_for_robot",
                return_value={},
            ),
            patch.object(collect_dagger, "IKResetController", return_value=reset),
            patch.object(collect_dagger, "_start_video_relay", return_value=relay),
            patch.object(
                collect_dagger, "DatasetRecorderProcess", return_value=recorder
            ),
            patch.object(collect_dagger, "restore_dataset_ownership"),
            patch.object(collect_dagger.affinity, "pin_realtime"),
            patch("os.sched_getaffinity", return_value={0}),
            patch("os.sched_setaffinity"),
            patch("signal.signal"),
        ]
        for mocked in patches:
            stack.enter_context(mocked)
        construct_worker = stack.enter_context(
            patch(
                "almond_axol.cli.plan_dagger_control.PlanDaggerControlLoop",
                return_value=worker,
            )
        )
        yield SimpleNamespace(
            config=config,
            robot=robot,
            reset=reset,
            policy=policy,
            teleop=teleop,
            recorder=recorder,
            worker=worker,
            construct_worker=construct_worker,
            events=events,
            stop_event=threading.Event(),
        )


@pytest.mark.parametrize(
    ("home_on_start", "start_from_current_pose", "expected_homes"),
    [(True, True, 1), (False, True, 0), (False, False, 2)],
)
def test_startup_home_is_separate_from_current_pose_episode_start(
    remote_startup_session, home_on_start, start_from_current_pose, expected_homes
):
    session = remote_startup_session
    session.config.home_on_start = home_on_start
    session.config.start_from_current_pose = start_from_current_pose
    control = Mock()
    control.quit_requested = False
    control.abort_requested = False
    control.poll_gate.return_value = None
    control.poll_choice.return_value = "q"
    control.begin_gate.side_effect = lambda message: session.events.append("idle gate")

    collect_dagger._run(session.config, stop_event=session.stop_event, control=control)

    assert session.reset.return_to_rest.call_count == expected_homes
    assert (
        session.events.index("reset ready")
        < session.events.index("IK ready")
        < session.events.index("robot connect")
    )
    if expected_homes:
        assert (
            session.events.index("robot connect")
            < session.events.index("home")
            < session.events.index("idle gate")
        )
    assert (
        session.events.index("idle gate")
        < session.events.index("start recording")
        < session.events.index("start policy control")
    )
    session.teleop.get_action.assert_called_once_with()
    initial_action = session.construct_worker.call_args.kwargs["initial_action"]
    assert initial_action == (
        {"left_joint.pos": 0.35, "right_joint.pos": -0.45}
        if start_from_current_pose
        else {"left_joint.pos": 0.1, "right_joint.pos": -0.1}
    )
    session.policy.reset.assert_called_once_with()
    session.worker.start.assert_called_once_with()
    assert session.events.index("park") < session.events.index("disable")


@pytest.mark.parametrize(
    "outcome", ["failed", "contact_q", "interrupted", "stopped", "error"]
)
def test_unsuccessful_startup_home_preserves_support_without_starting_policy(
    remote_startup_session, outcome
):
    session = remote_startup_session
    control = DaggerStdinControl()

    def unsuccessful_home(*args, **kwargs):
        session.events.append("home")
        assert kwargs["stopped"]() is False
        if outcome == "contact_q":
            assert kwargs["wait_retry"]() is False
            assert control.abort_requested
            assert not control.quit_requested
        elif outcome == "interrupted":
            raise KeyboardInterrupt
        elif outcome == "stopped":
            session.stop_event.set()
            assert kwargs["stopped"]() is True
        elif outcome == "error":
            raise RuntimeError("startup home failed")
        return False

    session.reset.return_to_rest.side_effect = unsuccessful_home
    with (
        patch("builtins.input", return_value="q"),
        patch.object(control, "begin_gate") as begin_gate,
        pytest.raises(RuntimeError, match="startup home failed")
        if outcome == "error"
        else nullcontext(),
    ):
        collect_dagger._run(
            session.config, stop_event=session.stop_event, control=control
        )

    session.reset.return_to_rest.assert_called_once()
    begin_gate.assert_not_called()
    session.teleop.get_action.assert_not_called()
    session.recorder.start_episode.assert_not_called()
    session.policy.reset.assert_not_called()
    session.construct_worker.assert_not_called()
    session.robot.send_action.assert_not_called()
    session.reset.park.assert_not_called()
    session.robot.disconnect.assert_not_called()
    session.robot.disconnect_preserving_position.assert_called_once_with()
    session.policy.close.assert_called_once_with()
    session.reset.stop.assert_called_once_with()


def test_cartesian_warmup_failure_never_enables_motors(remote_startup_session):
    session = remote_startup_session
    session.robot.prepare_cartesian_actions.side_effect = RuntimeError(
        "IK warmup failed"
    )

    with pytest.raises(RuntimeError, match="IK warmup failed"):
        collect_dagger._run(
            session.config, stop_event=session.stop_event, control=Mock()
        )

    session.robot.connect.assert_not_called()
    session.robot.send_action.assert_not_called()
    session.recorder.start_episode.assert_not_called()
    session.policy.reset.assert_not_called()
    session.construct_worker.assert_not_called()
    session.reset.park.assert_not_called()
    session.policy.close.assert_called_once_with()
    session.reset.stop.assert_called_once_with()
