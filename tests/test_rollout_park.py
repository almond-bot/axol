"""Guarded parking uses live fake feedback, a real interpolator and no hardware."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from almond_axol.constants import ARM_JOINTS
from almond_axol.lerobot import rollout
from almond_axol.lerobot.robot.robot_axol import AxolRobot
from almond_axol.robot import control as robot_control


class Clock:
    now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, duration):
        self.now += max(duration, 0.001)


class Plant:
    """Synthetic ideal joint plant; follow<1 adds measurable tracking lag."""

    def __init__(self, follow=1.0):
        self.q = np.zeros(14)
        self.follow = follow
        self.actions = []
        self.reset_command_state = mock.Mock()
        self.residual = 0.0
        self.stale_after = None

    def parking_positions(self):
        if self.stale_after is not None and len(self.actions) >= self.stale_after:
            raise RuntimeError("stale motor feedback")
        return np.r_[self.q[:7], 0.5], np.r_[self.q[7:], 0.5]

    def parking_gripper_hold(self):
        self.parking_positions()
        return 0.2, 0.8  # Last commanded grasp differs from measured grippers.

    def send_action(self, action):
        self.actions.append(action.copy())
        goal = np.array(
            [
                action[f"{side}_{joint.value}.pos"]
                for side in ("left", "right")
                for joint in ARM_JOINTS
            ]
        )
        self.q += self.follow * (goal - self.q)

    def torque_residuals(self):
        return np.full(7, self.residual), np.zeros(7)


class Pipe:
    def __init__(self, plant, clock):
        self.plant, self.clock = plant, clock
        self.requests = []
        self.respond = True
        self.drift = 0.0
        self.invalid = False
        self.empty = False

    def send(self, request):
        self.requests.append(request)
        self.goal = np.full(14, 0.2) if len(request) == 2 else request[2].copy()
        self.traj = list(np.linspace(request[1], self.goal, 21))

    def poll(self, timeout):
        if not self.respond:
            self.clock.sleep(timeout)
        return self.respond

    def recv(self):
        self.plant.q += self.drift
        if self.invalid:
            self.traj[-1][:] = np.nan
        return "reset_traj", self.goal.copy(), [] if self.empty else self.traj


@pytest.fixture
def parking(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(rollout, "time", clock)
    monkeypatch.setattr(robot_control, "time", clock)
    plant = Plant()
    controller = object.__new__(rollout.IKResetController)
    controller._ready = True
    controller._proc = mock.Mock()
    controller._proc.is_alive.return_value = True
    controller._q_init = np.zeros(14, dtype=np.float32)
    controller._left_indices = list(range(7))
    controller._right_indices = list(range(7, 14))
    controller._vr_cfg = SimpleNamespace(frequency=100.0)
    controller._conn = Pipe(plant, clock)
    return SimpleNamespace(
        controller=controller, plant=plant, clock=clock, pipe=controller._conn
    )


def test_two_leg_park_holds_commanded_grasp_and_waits_for_shaping_lag(parking):
    parking.plant.follow = 0.15
    assert parking.controller.park(parking.plant)
    requests = parking.pipe.requests
    assert len(requests) == 2
    assert len(requests[0]) == 2  # Configured REST, not the zero goal.
    assert len(requests[1]) == 3
    np.testing.assert_array_equal(requests[1][2], np.zeros(14))
    assert np.max(np.abs(requests[1][1] - 0.2)) < 0.05
    assert np.max(np.abs(parking.plant.q)) < 0.05
    assert len(parking.plant.actions) > 42  # Includes both measured settle holds.
    for action in parking.plant.actions:
        assert action["left_gripper.pos"] == 0.2
        assert action["right_gripper.pos"] == 0.8
    assert parking.plant.reset_command_state.call_count == 2


@pytest.mark.parametrize(
    "failure",
    [
        "dead",
        "plan timeout",
        "drift",
        "invalid",
        "empty distant",
        "stale",
        "contact",
        "stop",
        "lag timeout",
    ],
)
def test_park_failures_never_continue_to_zero(parking, failure):
    stopped = None
    if failure == "dead":
        parking.controller._proc.is_alive.return_value = False
    elif failure == "plan timeout":
        parking.pipe.respond = False
    elif failure == "drift":
        parking.pipe.drift = 0.06
    elif failure == "invalid":
        parking.pipe.invalid = True
    elif failure == "empty distant":
        parking.pipe.empty = True
    elif failure == "stale":
        parking.plant.stale_after = 3
    elif failure == "contact":
        parking.plant.residual = 10
    elif failure == "stop":

        def stopped():
            return len(parking.plant.actions) >= 3
    elif failure == "lag timeout":
        parking.plant.follow = 0
    with pytest.raises((RuntimeError, TimeoutError)):
        parking.controller.park(parking.plant, stopped=stopped)
    assert len(parking.pipe.requests) <= 1
    if failure in {"dead", "plan timeout", "drift", "invalid", "empty distant"}:
        assert parking.plant.actions == []
    if failure == "lag timeout":
        assert parking.clock.now >= 30


def test_keyboard_interrupt_during_send_stops_park(parking):
    parking.plant.send_action = mock.Mock(side_effect=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        parking.controller.park(parking.plant)
    assert len(parking.pipe.requests) == 1


@pytest.mark.parametrize("failure", ["stale", "stop"])
def test_fault_on_last_settle_dispatch_cannot_report_success(parking, failure):
    # Learn the deterministic fake plant's final send, then fault precisely
    # after that send. The final successful settle sample precedes the send.
    assert parking.controller.park(parking.plant)
    final_count = len(parking.plant.actions)
    parking.clock.now = 0.0
    parking.pipe.requests.clear()
    parking.plant.q[:] = 0
    parking.plant.actions.clear()
    stopped = None
    if failure == "stale":
        parking.plant.stale_after = final_count
    else:

        def stopped():
            return len(parking.plant.actions) >= final_count

    with pytest.raises(RuntimeError):
        parking.controller.park(parking.plant, stopped=stopped)


def test_real_feedback_guard_checks_every_motor_and_core_health():
    robot = object.__new__(AxolRobot)
    robot._dispatch_untrusted = False
    robot._last_joint_command = np.r_[np.zeros(7), 0.2, np.zeros(7), 0.8]
    left = SimpleNamespace(
        positions=np.r_[np.zeros(7), 0.5],
        motors={i: SimpleNamespace(feedback_ts=100.0) for i in range(8)},
    )
    right = SimpleNamespace(
        positions=left.positions.copy(),
        motors={i: SimpleNamespace(feedback_ts=100.0) for i in range(8)},
    )
    robot._axol = SimpleNamespace(left=left, right=right, fault=None, limp=None)
    with mock.patch(
        "almond_axol.lerobot.robot.robot_axol.time.time", return_value=100.1
    ):
        assert robot.parking_gripper_hold() == (0.2, 0.8)
        left.motors[0].feedback_ts = 99.0
        with pytest.raises(RuntimeError, match="stale"):
            robot.parking_positions()
        left.motors[0].feedback_ts = 100.0
        for fault in ("fault", "limp"):
            setattr(robot._axol, fault, "core stopped")
            with pytest.raises(RuntimeError, match="faulted or limp"):
                robot.parking_positions()
            setattr(robot._axol, fault, None)
        robot._dispatch_untrusted = True
        with pytest.raises(RuntimeError, match="completion is unknown"):
            robot.parking_positions()


def test_worker_honors_explicit_zero_target_without_changing_rest_protocol():
    from almond_axol.kinematics.config import KinematicsConfig
    from almond_axol.teleop import worker as module

    rest = np.full(14, 0.2, dtype=np.float32)
    current = np.full(14, 0.1, dtype=np.float32)
    zero = np.zeros(14, dtype=np.float32)
    worker = mock.Mock(left_indices=list(range(7)), right_indices=list(range(7, 14)))
    worker.get_rest_q.return_value = rest
    worker.compute_reset_trajectory.side_effect = lambda start, goal: [
        start.copy(),
        goal.copy(),
    ]
    conn = mock.Mock()
    conn.recv.side_effect = [("reset", current), ("reset", rest, zero), None]
    with (
        mock.patch.object(module, "IKWorker", return_value=worker),
        mock.patch.object(module.signal, "signal"),
        mock.patch.object(module.os, "nice"),
        mock.patch("almond_axol.utils.affinity.pin_ik_startup"),
        mock.patch("almond_axol.utils.affinity.pin_ik"),
        mock.patch.dict(module.os.environ),
    ):
        module.run_ik_worker(conn, SimpleNamespace(), KinematicsConfig())
    replies = [args[0][0] for args in conn.send.call_args_list]
    assert replies[0][0] == "ready"
    np.testing.assert_array_equal(replies[1][1], rest)
    np.testing.assert_array_equal(replies[2][1], zero)
    np.testing.assert_array_equal(replies[2][2][-1], zero)
    np.testing.assert_array_equal(
        worker.compute_reset_trajectory.call_args.args[1], zero
    )
