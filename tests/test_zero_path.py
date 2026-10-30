"""Sessions start through zero and park back to zero before torque-off."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from almond_axol.lerobot import rollout
from almond_axol.teleop.worker import IKWorker

_REST = np.full(14, 0.2, dtype=np.float32)


def _line(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
    return [start + (goal - start) * (i / 4) for i in range(5)]


def _worker() -> IKWorker:
    """An IKWorker with a stand-in solver and a straight-line planner."""
    worker = object.__new__(IKWorker)
    worker._solver = SimpleNamespace(
        num_joints=14, left_indices=list(range(7)), right_indices=list(range(7, 14))
    )
    worker._rest_pose_left = _REST[:7].copy()
    worker._rest_pose_right = _REST[7:].copy()
    worker.compute_reset_trajectory = mock.Mock(side_effect=_line)
    return worker


class TestPathPlanning:
    def test_legs_chain_through_named_goals_in_order(self) -> None:
        worker = _worker()
        start = np.full(14, 0.3, dtype=np.float32)

        traj = worker.compute_path_trajectory(start, ("zero", "rest"))

        legs = worker.compute_reset_trajectory.call_args_list
        assert len(legs) == 2
        np.testing.assert_array_equal(legs[0].args[0], start)
        np.testing.assert_array_equal(legs[0].args[1], np.zeros(14))
        # The second leg starts where the first one ends.
        np.testing.assert_array_equal(legs[1].args[0], np.zeros(14))
        np.testing.assert_array_equal(legs[1].args[1], _REST)
        assert any(np.allclose(q, 0.0) for q in traj)
        np.testing.assert_allclose(traj[-1], _REST)

    def test_a_leg_already_at_its_goal_is_skipped(self) -> None:
        worker = _worker()

        traj = worker.compute_path_trajectory(np.zeros(14), ("zero", "rest"))

        worker.compute_reset_trajectory.assert_called_once()
        np.testing.assert_allclose(traj[-1], _REST)

    def test_an_unknown_goal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown path goal"):
            _worker().compute_path_trajectory(np.zeros(14), ("home",))


def _run_worker(*requests: object) -> tuple[mock.Mock, list[tuple]]:
    """Run the worker loop against a mocked IKWorker; return it and replies."""
    from almond_axol.teleop import worker as module

    worker = mock.Mock(left_indices=list(range(7)), right_indices=list(range(7, 14)))
    worker.get_rest_q.return_value = _REST.copy()
    worker.path_goal.side_effect = lambda name: (
        _REST.copy() if name == "rest" else np.zeros(14, dtype=np.float32)
    )
    worker.compute_path_trajectory.side_effect = lambda q, goals: [
        np.asarray(q, dtype=np.float32),
        worker.path_goal(goals[-1]),
    ]
    conn = mock.Mock()
    conn.recv.side_effect = [*requests, None]
    with (
        mock.patch.object(module, "IKWorker", return_value=worker),
        mock.patch.object(module.signal, "signal"),
        mock.patch.object(module.os, "nice"),
        mock.patch("almond_axol.utils.affinity.pin_ik_startup"),
        mock.patch("almond_axol.utils.affinity.pin_ik"),
        mock.patch.dict(module.os.environ),
    ):
        module.run_ik_worker(
            conn,
            SimpleNamespace(absolute_mode=False),
            object(),
            np.full(8, 0.1, dtype=np.float32),
            np.full(8, -0.1, dtype=np.float32),
        )
    return worker, [call.args[0] for call in conn.send.call_args_list]


class TestWorkerProtocol:
    def test_the_startup_move_goes_through_zero_to_rest(self) -> None:
        worker, replies = _run_worker()

        q_start, goals = worker.compute_path_trajectory.call_args_list[0].args
        assert goals == ("zero", "rest")
        np.testing.assert_allclose(q_start[:7], 0.1)
        np.testing.assert_allclose(q_start[7:], -0.1)
        assert replies[0][0] == "ready"
        np.testing.assert_array_equal(replies[0][4][-1], _REST)

    def test_a_reset_path_request_answers_with_its_last_goal(self) -> None:
        current = np.full(14, 0.4, dtype=np.float32)
        worker, replies = _run_worker(("reset_path", current, ("rest", "zero")))

        assert worker.compute_path_trajectory.call_args.args[1] == ("rest", "zero")
        kind, goal, traj = replies[1]
        assert kind == "reset_traj"
        np.testing.assert_array_equal(goal, np.zeros(14))
        np.testing.assert_array_equal(traj[-1], np.zeros(14))

    @pytest.mark.parametrize(
        "request_",
        [
            ("reset_path", np.zeros(14), ()),
            ("reset_path", np.zeros(14), ("home",)),
            ("reset_path", np.full(14, np.nan), ("zero",)),
            ("reset_path", np.zeros(14)),
        ],
    )
    def test_a_malformed_reset_path_is_refused(self, request_) -> None:
        with pytest.raises(ValueError, match="reset_path"):
            _run_worker(request_)


class _Plant:
    """Synthetic ideal joint plant for the controller's unguarded play path."""

    def __init__(self) -> None:
        self.q = np.full(14, 0.3)
        self.actions: list[dict[str, float]] = []

    @property
    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return np.r_[self.q[:7], 0.5], np.r_[self.q[7:], 0.5]

    def send_action(self, action: dict[str, float]) -> None:
        self.actions.append(action)

    def torque_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(7), np.zeros(7)

    def parking_gripper_hold(self) -> tuple[float, float]:
        return 0.2, 0.8  # The last commanded grasp, not the measured 0.5.


class _Pipe:
    def __init__(self) -> None:
        self.requests: list[tuple] = []

    def send(self, request: tuple) -> None:
        self.requests.append(request)

    def recv(self) -> tuple:
        request = self.requests[-1]
        goal = np.zeros(14) if request[0] == "reset_path" else _REST.copy()
        return "reset_traj", goal, _line(np.asarray(request[1]), goal)


def _controller(monkeypatch) -> rollout.IKResetController:
    monkeypatch.setattr(rollout.time, "sleep", lambda _s: None)
    controller = object.__new__(rollout.IKResetController)
    controller._ready = True
    controller._arms_limp = False
    controller._proc = mock.Mock()
    controller._q_init = np.zeros(14, dtype=np.float32)
    controller._left_indices = list(range(7))
    controller._right_indices = list(range(7, 14))
    controller._vr_cfg = SimpleNamespace(frequency=1000.0)
    controller._conn = _Pipe()
    return controller


def test_the_rollout_controller_sends_its_path_to_the_worker(monkeypatch) -> None:
    controller = _controller(monkeypatch)
    plant = _Plant()

    assert controller.return_to_rest(plant, path=("rest", "zero"))

    (request,) = controller._conn.requests
    assert request[0] == "reset_path"
    assert request[2] == ("rest", "zero")
    np.testing.assert_allclose(request[1], 0.3)
    assert plant.actions[-1]["left_elbow.pos"] == pytest.approx(0.0)


def test_a_teardown_park_holds_the_commanded_grasp(monkeypatch) -> None:
    controller = _controller(monkeypatch)
    plant = _Plant()

    assert controller.return_to_rest(plant, path=("rest", "zero"), hold_grasp=True)

    assert {a["left_gripper.pos"] for a in plant.actions} == {0.2}
    assert {a["right_gripper.pos"] for a in plant.actions} == {0.8}


def test_a_plain_return_still_ramps_the_grippers_open(monkeypatch) -> None:
    controller = _controller(monkeypatch)
    plant = _Plant()

    assert controller.return_to_rest(plant)

    assert plant.actions[-1]["left_gripper.pos"] == pytest.approx(1.0)
