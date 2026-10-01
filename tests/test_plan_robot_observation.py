"""Generic robot execution seams use fake feedback and never open hardware."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from almond_axol.kinematics import fk
from almond_axol.lerobot.robot import robot_axol
from almond_axol.lerobot.robot.config_axol import AxolRobotConfig
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.lerobot.rollout import IKResetController


@pytest.fixture
def robot_loop():
    robot = AxolRobot(AxolRobotConfig(cameras={}, action_space="cartesian"))
    robot._axol = SimpleNamespace(
        left=SimpleNamespace(positions=np.zeros(8)),
        right=SimpleNamespace(positions=np.zeros(8)),
        motion_control=AsyncMock(),
    )
    robot._loop = asyncio.new_event_loop()
    thread = threading.Thread(target=robot._loop.run_forever)
    thread.start()
    try:
        yield robot
    finally:
        robot._loop.call_soon_threadsafe(robot._loop.stop)
        thread.join(timeout=2)
        assert not thread.is_alive()
        robot._loop.close()
        robot._axol = None
        robot._loop = None
        robot._loop_thread = None


@pytest.mark.parametrize("cartesian", [False, True])
def test_post_ik_transform_returns_and_records_actual_joint_command(
    robot_loop, cartesian
):
    robot = robot_loop
    events = []
    left, right = np.arange(8) / 20, -np.arange(8) / 20
    right[-1] = 0.8
    resolved = robot._joint_action(left, right)
    robot._ensure_ik = Mock()

    def solve(action):
        events.append("ik")
        return left, right

    robot._cartesian_action_to_targets = solve

    def transform(action):
        events.append("transform")
        assert action == resolved
        return {key: value / 2 for key, value in action.items()}

    async def send(**kwargs):
        events.append("send")

    robot._axol.motion_control.side_effect = send
    source = dict.fromkeys(robot.action_features, 0.0) if cartesian else resolved
    performed = robot.send_action(source, joint_transform=transform)

    assert events == (["ik"] if cartesian else []) + ["transform", "send"]
    assert performed == {key: value / 2 for key, value in resolved.items()}
    sent = robot._axol.motion_control.call_args.kwargs
    assert robot._joint_action(sent["left"], sent["right"]) == pytest.approx(performed)
    np.testing.assert_allclose(robot._last_joint_command, np.r_[left, right] / 2)
    assert source == (
        dict.fromkeys(robot.action_features, 0.0) if cartesian else resolved
    )


def test_rejected_transform_never_dispatches(robot_loop):
    robot = robot_loop
    rejected = Mock(side_effect=RuntimeError("policy revoked"))
    with pytest.raises(RuntimeError, match="policy revoked"):
        robot.send_action(
            robot._joint_action(np.zeros(8), np.zeros(8)), joint_transform=rejected
        )
    robot._axol.motion_control.assert_not_awaited()
    assert robot._last_joint_command is None


def test_generic_cartesian_handover_reseeds_real_shapers_from_measured_pose(
    monkeypatch,
):
    robot = AxolRobot(AxolRobotConfig(cameras={}, action_space="cartesian"))
    robot._axol = SimpleNamespace(
        left=SimpleNamespace(positions=np.zeros(8)),
        right=SimpleNamespace(positions=np.zeros(8)),
    )
    solver = SimpleNamespace(
        num_joints=14,
        left_indices=list(range(7)),
        right_indices=list(range(7, 14)),
        ik=Mock(return_value=np.full(14, 1.0)),
    )
    robot._ik = solver
    monkeypatch.setattr(fk, "pose6_to_pos_rot", lambda pose: pose)
    monkeypatch.setattr(robot_axol.time, "monotonic", lambda: 100.0)
    action = dict.fromkeys(robot.action_features, 0.0)
    robot._cartesian_action_to_targets(action)

    # A different control owner moved the measured arms far from the old
    # shaped command. Resetting must not replay that old position/velocity.
    robot._axol.left.positions[:7] = -0.5
    robot._axol.right.positions[:7] = 0.5
    robot.reset_cartesian_seed()
    left, right = robot._cartesian_action_to_targets(action)

    np.testing.assert_array_equal(
        solver.ik.call_args.args[0], np.r_[np.full(7, -0.5), np.full(7, 0.5)]
    )
    assert np.max(np.abs(left[:7] + 0.5)) < 0.02
    assert np.max(np.abs(right[:7] - 0.5)) < 0.02


def test_timed_out_joint_dispatch_joins_cancellation_before_handover(robot_loop):
    robot = robot_loop
    finalized = threading.Event()

    async def blocked_send(**kwargs):
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0.01)
            finalized.set()

    robot._axol.motion_control.side_effect = blocked_send
    with pytest.raises(TimeoutError):
        robot.send_action(robot._joint_action(np.zeros(8), np.zeros(8)))
    assert finalized.is_set()
    assert robot._last_joint_command is None
    assert not robot._dispatch_untrusted


def test_reset_readiness_cancels_without_consuming_the_handshake():
    # The shared wait (teleop.core.wait_for_ik_ready) polls, then honours the
    # stop request before reading anything from the worker pipe.
    controller = IKResetController()
    controller._conn = Mock(poll=Mock(return_value=False))
    assert controller.wait_ready(stopped=lambda: True) is False
    controller._conn.recv.assert_not_called()


def test_reset_readiness_checks_worker_death_and_timeout():
    controller = IKResetController()
    controller._conn = Mock(poll=Mock(return_value=False))
    controller._proc = SimpleNamespace(is_alive=lambda: False, exitcode=17)
    with pytest.raises(RuntimeError, match="exit code 17"):
        controller.wait_ready()
    controller._proc = SimpleNamespace(is_alive=lambda: True, exitcode=None)
    with pytest.raises(TimeoutError, match="did not become ready"):
        controller.wait_ready(timeout=0)


def test_reset_readiness_validates_and_caches_handshake():
    controller = IKResetController()
    controller._conn = Mock(poll=Mock(return_value=True), recv=Mock(return_value=()))
    with pytest.raises(RuntimeError, match="Unexpected IK worker handshake"):
        controller.wait_ready()
    controller._conn.recv.return_value = (
        "ready",
        np.zeros(14),
        range(7),
        range(7, 14),
        [],
    )
    assert controller.wait_ready()
    controller._conn.reset_mock()
    assert controller.wait_ready()
    controller._conn.poll.assert_not_called()
