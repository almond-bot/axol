"""Jogging: lift and base models, IK planning, the session and the API."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from almond_axol.jog.backends import HardwareBackend, SimBackend, rest_q
from almond_axol.jog.session import (
    HOLD_TIMEOUT_S,
    JogError,
    JogSession,
    active_session,
)
from almond_axol.jog.sim import (
    LIFT_FULL_TRAVEL_S,
    LIFT_RAMP_S,
    BaseSim,
    LiftSim,
    body_velocity,
)
from almond_axol.robot.jelly import JellyConfig

DT = 0.02  # Jelly's 50 Hz command period


def _run_lift(lift: LiftSim, direction: int, seconds: float) -> list[float]:
    speeds = []
    for _ in range(round(seconds / DT)):
        lift.step(direction, DT)
        speeds.append(lift.velocity)
    return speeds


class LiftSimTest(unittest.TestCase):
    def test_held_jog_ramps_to_full_speed_then_holds_it(self) -> None:
        lift = LiftSim(raised=0.0)
        speeds = _run_lift(lift, 1, 2.0)
        full = 1.0 / LIFT_FULL_TRAVEL_S
        ramp_ticks = next(i for i, v in enumerate(speeds) if v >= full - 1e-9)
        self.assertAlmostEqual(ramp_ticks * DT, LIFT_RAMP_S, delta=2 * DT)
        self.assertAlmostEqual(speeds[-1], full, places=9)
        # Monotonic ramp, no overshoot.
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(speeds, speeds[1:])))
        self.assertLessEqual(max(speeds), full + 1e-12)

    def test_release_decelerates_at_the_same_rate(self) -> None:
        lift = LiftSim(raised=0.2)
        _run_lift(lift, 1, 2.0)
        at_release = lift.raised
        speeds = _run_lift(lift, 0, 1.0)
        stop_tick = next(i for i, v in enumerate(speeds) if v == 0.0)
        self.assertAlmostEqual(stop_tick * DT, LIFT_RAMP_S, delta=2 * DT)
        # Stopping distance of a linear ramp: v²/2a.
        full = 1.0 / LIFT_FULL_TRAVEL_S
        expected = full * full / (2 * lift.accel)
        self.assertAlmostEqual(lift.raised - at_release, expected, delta=full * DT)
        self.assertFalse(lift.moving)

    def test_lowering_mirrors_raising(self) -> None:
        lift = LiftSim(raised=1.0)
        _run_lift(lift, -1, 1.0)
        self.assertLess(lift.velocity, 0.0)
        self.assertLess(lift.raised, 1.0)

    def test_full_travel_takes_the_firmware_time_and_stops_at_the_end(self) -> None:
        lift = LiftSim(raised=0.0)
        elapsed = 0.0
        while lift.raised < 1.0 and elapsed < 30.0:
            lift.step(1, DT)
            elapsed += DT
        self.assertEqual(lift.raised, 1.0)
        self.assertEqual(lift.velocity, 0.0)
        self.assertAlmostEqual(elapsed, LIFT_FULL_TRAVEL_S + LIFT_RAMP_S, delta=0.2)
        # Holding up at the top does nothing.
        _run_lift(lift, 1, 1.0)
        self.assertEqual(lift.raised, 1.0)

    def test_slower_jog_speed_setting_slows_the_lift(self) -> None:
        lift = LiftSim(raised=0.0, jog_speed=325)
        _run_lift(lift, 1, 3.0)
        self.assertAlmostEqual(lift.velocity, 0.5 / LIFT_FULL_TRAVEL_S, places=9)


class BaseSimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = JellyConfig()

    def test_command_slews_at_the_configured_rate(self) -> None:
        base = BaseSim(self.cfg)
        for _ in range(50):  # 1 s
            base.step((1.0, 0.0, 0.0), DT)
        self.assertAlmostEqual(base.command[0], self.cfg.slew * 1.0, places=6)
        self.assertGreater(base.x, 0.0)
        self.assertAlmostEqual(base.y, 0.0, places=9)

    def test_release_ramps_down_but_a_dead_source_stops_at_once(self) -> None:
        base = BaseSim(self.cfg)
        for _ in range(100):
            base.step((1.0, 0.0, 0.0), DT)
        held = base.command[0]
        base.step((0.0, 0.0, 0.0), DT)
        self.assertAlmostEqual(base.command[0], held - self.cfg.slew * DT, places=6)
        base.step(None, DT)
        self.assertEqual(base.command, (0.0, 0.0, 0.0))
        self.assertFalse(base.moving)

    def test_near_axis_translation_snaps_onto_the_axis(self) -> None:
        base = BaseSim(self.cfg)
        heading = math.radians(10.0)
        for _ in range(100):
            base.step((math.cos(heading), math.sin(heading), 0.0), DT)
        self.assertAlmostEqual(base.command[1], 0.0, places=9)
        self.assertAlmostEqual(base.y, 0.0, places=9)

    def test_full_command_speeds_follow_the_wheels(self) -> None:
        cfg = self.cfg
        vx, vy, wz = body_velocity(1.0, 0.0, 0.0, cfg)
        rim = cfg.max_speed * 0.06335
        self.assertAlmostEqual(vx, rim * math.sqrt(2.0), places=6)
        self.assertEqual((vy, wz), (0.0, 0.0))
        # A combined command is scaled down so no wheel exceeds max_speed.
        cx, _, cw = body_velocity(1.0, 0.0, 1.0, cfg)
        self.assertAlmostEqual(cx, vx / 2.0, places=6)
        self.assertAlmostEqual(cw, body_velocity(0.0, 0.0, 1.0, cfg)[2] / 2.0, places=6)

    def test_turning_rotates_the_heading_counter_clockwise(self) -> None:
        base = BaseSim(self.cfg)
        for _ in range(100):
            base.step((0.0, 0.0, 1.0), DT)
        self.assertGreater(base.yaw, 0.0)
        self.assertAlmostEqual(base.x, 0.0, places=9)

    def test_driving_forward_after_a_turn_follows_the_heading(self) -> None:
        base = BaseSim(self.cfg, yaw=math.pi / 2)
        for _ in range(100):
            base.step((1.0, 0.0, 0.0), DT)
        self.assertAlmostEqual(base.x, 0.0, places=6)
        self.assertGreater(base.y, 0.0)


# -- IK planning on the real solver ----------------------------------------


@pytest.fixture(scope="module")
def planner() -> Any:
    from almond_axol.jog.planner import JogPlanner
    from almond_axol.kinematics.config import (
        KinematicsConfig,
        apply_mantis_kinematics_profile,
    )
    from almond_axol.kinematics.solver import KinematicsSolver

    config = KinematicsConfig(robot_model="mobile")
    apply_mantis_kinematics_profile(config)
    return JogPlanner(KinematicsSolver(config))


def test_ik_reaches_a_reachable_target_within_two_mm(planner: Any) -> None:
    q = rest_q()
    left, right = planner.tips(q)
    target = (left[0] + np.array([0.1, 0.05, 0.12], np.float32), left[1])
    solution = planner.solve(q, "left", target)
    assert solution.reached
    assert solution.position_error < 2e-3
    reached = planner.tips(solution.q)[0]
    assert np.linalg.norm(reached[0] - target[0]) < 2e-3
    # The other arm does not move.
    np.testing.assert_array_equal(solution.q[7:], q[7:])
    np.testing.assert_allclose(planner.tips(solution.q)[1][0], right[0], atol=1e-6)


def test_ik_reaches_a_rotated_target(planner: Any) -> None:
    q = rest_q()
    _, right = planner.tips(q)
    rotation = (
        Rotation.from_euler("y", 40, degrees=True).as_matrix() @ right[1]
    ).astype(np.float32)
    target = (right[0] + np.array([0.12, -0.03, 0.1], np.float32), rotation)
    solution = planner.solve(q, "right", target)
    assert solution.reached
    assert solution.orientation_error < math.radians(2.0)
    np.testing.assert_array_equal(solution.q[:7], q[:7])


def test_ik_returns_the_closest_pose_for_an_unreachable_target(planner: Any) -> None:
    q = rest_q()
    left, _ = planner.tips(q)
    target = (left[0] + np.array([0.9, 0.4, 0.0], np.float32), left[1])
    start_error = float(np.linalg.norm(left[0] - target[0]))
    solution = planner.solve(q, "left", target)
    assert not solution.reached
    # Much closer than where it started, and pointed at the target.
    assert solution.position_error < start_error - 0.3
    moved = planner.tips(solution.q)[0][0] - left[0]
    direction = (target[0] - left[0]) / start_error
    assert float(moved @ direction) > 0.3


def test_plan_starts_at_the_current_pose_and_ends_on_the_solution(planner: Any) -> None:
    q = rest_q()
    left, _ = planner.tips(q)
    target = (left[0] + np.array([0.05, 0.0, 0.1], np.float32), left[1])
    plan = planner.plan(q, "left", target, speed=0.6, rate=125.0, min_duration=1.0)
    assert plan.duration >= 1.0
    np.testing.assert_allclose(plan.trajectory[0], q, atol=0.03)
    np.testing.assert_array_equal(plan.trajectory[-1], plan.solution.q)
    # Small steps the realtime core accepts, and the other arm is pinned.
    steps = np.abs(np.diff(np.stack(plan.trajectory), axis=0))
    assert float(steps.max()) < 0.05
    assert all(np.array_equal(p[7:], q[7:]) for p in plan.trajectory)
    np.testing.assert_allclose(plan.path[-1], plan.solution.tip[0], atol=1e-5)
    np.testing.assert_allclose(plan.path[0], left[0], atol=5e-3)


# -- the session -----------------------------------------------------------


class _FakePlanner:
    """Planner stand-in: solves exactly, with a straight joint-space path."""

    def __init__(self) -> None:
        from almond_axol.jog.planner import IkSolution, JogPlan

        self._solution = IkSolution
        self._plan = JogPlan
        self.calls = 0

    def indices(self, arm: str) -> list[int]:
        return list(range(7)) if arm == "left" else list(range(7, 14))

    def tips(self, q: np.ndarray, *, blocking: bool = True) -> Any:
        pose = (np.zeros(3, np.float32), np.eye(3, dtype=np.float32))
        return pose, pose

    def joint_trajectory(self, q_from: np.ndarray, q_to: np.ndarray, **kw: Any) -> list:
        steps = 50
        return [q_from + (q_to - q_from) * (i + 1) / steps for i in range(steps)]

    def plan(self, q_now: np.ndarray, arm: str, target: Any, **kw: Any) -> Any:
        self.calls += 1
        q_to = q_now.copy()
        q_to[self.indices(arm)[0]] += 0.2
        solution = self._solution(q_to, target, 0.0, 0.0)
        steps = 100  # half a second at the tests' 200 Hz loop
        trajectory = [q_now + (q_to - q_now) * (i + 1) / steps for i in range(steps)]
        return self._plan(
            arm, target, solution, q_now.copy(), trajectory, kw["rate"], []
        )


def _start(session: JogSession) -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(session.run(stop)), daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5
    while active_session() is not session and time.monotonic() < deadline:
        time.sleep(0.01)
    while session.state()["arms"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    return thread, stop


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


QUAT = [0.0, 0.0, 0.0, 1.0]


class JogSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = SimBackend("mobile")
        self.planner = _FakePlanner()
        self.session = JogSession(self.backend, lambda: self.planner, rate=200.0)
        self.thread, self.stop = _start(self.session)

    def tearDown(self) -> None:
        self.stop.set()
        self.thread.join(5)
        self.assertIsNone(active_session())

    def test_state_reports_arms_lift_and_base(self) -> None:
        state = self.session.state()
        self.assertEqual(state["mode"], "sim")
        self.assertEqual(len(state["arms"]["left"]), 7)
        self.assertEqual(state["lift"]["raised"], 1.0)
        self.assertEqual(state["base"]["x"], 0.0)
        self.assertFalse(state["hold"]["active"])

    def test_plan_then_execute_moves_the_arm_to_the_solution(self) -> None:
        plan = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        before = self.session.state()["arms"]["left"][0]
        self.assertAlmostEqual(plan["joints"][0], before + 0.2, places=5)
        # Planning alone moves nothing.
        time.sleep(0.05)
        self.assertEqual(self.session.state()["arms"]["left"][0], before)
        self.session.execute(plan["id"])
        self.assertTrue(
            _wait(lambda: self.session.state()["lastExecution"] is not None)
        )
        state = self.session.state()
        self.assertEqual(state["lastExecution"]["status"], "done")
        self.assertAlmostEqual(state["arms"]["left"][0], before + 0.2, places=5)

    def test_a_plan_runs_once(self) -> None:
        plan = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        self.session.execute(plan["id"])
        with self.assertRaises(JogError):
            self.session.execute(plan["id"])

    def test_only_the_latest_plan_can_run(self) -> None:
        first = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        self.session.plan("right", [0.3, -0.2, 0.1], QUAT)
        with self.assertRaisesRegex(JogError, "out of date"):
            self.session.execute(first["id"])

    def test_execute_refuses_a_plan_the_arm_has_moved_away_from(self) -> None:
        stale = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        with self.session._lock:
            self.session._command[0] += 0.1
        with self.assertRaisesRegex(JogError, "moved after this plan"):
            self.session.execute(stale["id"])

    def test_cancel_stops_the_arm_where_it_is(self) -> None:
        plan = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        before = self.session.state()["arms"]["left"][0]
        self.session.execute(plan["id"])
        time.sleep(0.1)
        self.assertTrue(self.session.cancel())
        self.assertTrue(
            _wait(lambda: self.session.state()["lastExecution"] is not None)
        )
        state = self.session.state()
        self.assertEqual(state["lastExecution"]["status"], "stopped")
        stopped_at = state["arms"]["left"][0]
        self.assertGreater(stopped_at, before)
        self.assertLess(stopped_at, before + 0.2)
        time.sleep(0.05)
        self.assertEqual(self.session.state()["arms"]["left"][0], stopped_at)
        self.assertFalse(self.session.cancel())

    def test_bad_plan_requests_are_refused(self) -> None:
        with self.assertRaises(JogError):
            self.session.plan("left", [0.3, 0.2], QUAT)
        with self.assertRaises(JogError):
            self.session.plan("left", [0.3, float("nan"), 0.1], QUAT)
        with self.assertRaises(JogError):
            self.session.plan("left", [0.3, 0.2, 0.1], [0.0, 0.0, 0.0, 0.0])
        with self.assertRaises(JogError):
            self.session.plan("middle", [0.3, 0.2, 0.1], QUAT)  # type: ignore[arg-type]

    def _hold_for(self, seconds: float, vx: float = 0.0, lift: int = 0) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.session.hold(vx, 0.0, 0.0, lift)
            time.sleep(0.05)

    def test_stopping_returns_moved_arms_to_rest(self) -> None:
        plan = self.session.plan("left", [0.3, 0.2, 0.1], QUAT)
        self.session.execute(plan["id"])
        self.assertTrue(
            _wait(lambda: self.session.state()["lastExecution"] is not None)
        )
        self.stop.set()
        self.thread.join(5)
        left, right = asyncio.run(self.backend.arm_positions())
        np.testing.assert_allclose(
            np.concatenate([left[:7], right[:7]]), rest_q(), atol=1e-6
        )

    def test_lift_moves_only_while_held(self) -> None:
        self._hold_for(1.0, lift=-1)
        held = self.session.state()["lift"]
        self.assertLess(held["raised"], 1.0)
        self.assertLess(held["velocity"], 0.0)
        self.session.release()
        self.assertTrue(_wait(lambda: not self.session.state()["lift"]["moving"], 2.0))
        settled = self.session.state()["lift"]["raised"]
        time.sleep(0.2)
        self.assertEqual(self.session.state()["lift"]["raised"], settled)

    def test_base_drives_while_held_and_stops_when_the_page_goes_quiet(self) -> None:
        self._hold_for(1.0, vx=1.0)
        state = self.session.state()
        self.assertTrue(state["hold"]["active"])
        self.assertGreater(state["base"]["x"], 0.0)
        self.assertGreater(state["base"]["command"][0], 0.0)
        # No more hold messages: after the timeout the base stops at once.
        time.sleep(HOLD_TIMEOUT_S + 0.1)
        state = self.session.state()
        self.assertFalse(state["hold"]["active"])
        self.assertEqual(state["base"]["command"], [0.0, 0.0, 0.0])

    def test_disconnect_stops_everything_immediately(self) -> None:
        self._hold_for(0.5, vx=1.0, lift=1)
        self.session.disconnect()
        time.sleep(0.05)
        self.assertEqual(self.session.state()["base"]["command"], [0.0, 0.0, 0.0])

    def test_an_idle_page_does_not_cancel_the_page_that_is_driving(self) -> None:
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            self.session.hold(0.0, 0.0, 0.0, 0, source="observer")
            self.session.hold(0.0, 0.0, 0.0, -1, source="operator")
            time.sleep(0.03)
        state = self.session.state()
        self.assertEqual(state["hold"]["lift"], -1)
        self.assertLess(state["lift"]["velocity"], -0.9 * self.backend.lift.max_speed)

    def test_the_latest_page_to_press_a_control_wins(self) -> None:
        self.session.hold(1.0, 0.0, 0.0, 0, source="a")
        time.sleep(0.01)
        self.session.hold(0.0, 1.0, 0.0, 0, source="b")
        self.session.hold(1.0, 0.0, 0.0, 0, source="a")  # still holding
        self.assertEqual(self.session.state()["hold"]["drive"], [0.0, 1.0, 0.0])

    def test_driving_page_vanishing_stops_at_once_despite_an_idle_page(self) -> None:
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            self.session.hold(0.0, 0.0, 0.0, 0, source="observer")
            self.session.hold(1.0, 0.0, 0.0, 0, source="operator")
            time.sleep(0.03)
        self.assertGreater(self.session.state()["base"]["command"][0], 0.1)
        self.session.disconnect("operator")
        self.session.hold(0.0, 0.0, 0.0, 0, source="observer")
        time.sleep(0.05)
        self.assertEqual(self.session.state()["base"]["command"], [0.0, 0.0, 0.0])

    def test_drive_commands_are_linear_and_invalid_holds_are_refused(self) -> None:
        session = JogSession(self.backend, lambda: self.planner)
        # A 20 % speed limit must reach the wheels as 20 %, not vanish into
        # a stick deadzone.
        session.hold(0.2, -0.1, 0.05, 0)
        self.assertEqual(session.state()["hold"]["drive"], [0.2, -0.1, 0.05])
        session.hold(3.0, -3.0, 0.0, 0)
        self.assertEqual(session.state()["hold"]["drive"], [1.0, -1.0, 0.0])
        with self.assertRaises(JogError):
            session.hold(float("nan"), 0.0, 0.0, 0)
        with self.assertRaises(JogError):
            session.hold(0.0, 0.0, 0.0, 2)
        with self.assertRaises(JogError):
            session.hold(True, 0.0, 0.0, 0)  # type: ignore[arg-type]


def test_classic_sim_has_no_lift_or_base() -> None:
    backend = SimBackend("classic")
    assert backend.lift_state() is None
    assert backend.base_state() is None
    backend.drive((1.0, 0.0, 0.0, 1), DT)  # ignored, no error


# -- hardware backend with fake devices ------------------------------------


class _FakeArm:
    def __init__(self, positions: np.ndarray) -> None:
        self.positions = positions


class _FakeAxol:
    def __init__(self) -> None:
        left, right = rest_q()[:7], rest_q()[7:]
        self.left = _FakeArm(np.append(left, 0.3).astype(np.float32))
        self.right = _FakeArm(np.append(right, 0.0).astype(np.float32))
        self.sent: list[tuple[Any, Any]] = []
        self.entered = self.exited = False

    async def __aenter__(self) -> _FakeAxol:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.exited = True

    async def motion_control(self, left: Any = None, right: Any = None) -> None:
        self.sent.append((left, right))


class _FakeStatus:
    velocity = -325
    moving = True


class _FakeJelly:
    def __init__(self) -> None:
        self.config = JellyConfig()
        self.has_wheels = True
        self.has_lift = True
        self.body_cmd = (0.0, 0.0, 0.0)
        self.commands: list[tuple[float, float, float, int]] = []
        self.lift_raised = 0.5
        self.lift_status = _FakeStatus()
        self.wheel_fault = False
        self.enabled = self.disabled = False

    async def enable(self) -> None:
        self.enabled = True

    async def disable(self) -> None:
        self.disabled = True

    def set_command(self, vx: float, vy: float, wz: float, lift: int = 0) -> None:
        self.commands.append((vx, vy, wz, lift))
        self.body_cmd = (vx, vy, wz)


class HardwareBackendTest(unittest.TestCase):
    def test_session_holds_the_measured_pose_and_feeds_jelly(self) -> None:
        axol, jelly = _FakeAxol(), _FakeJelly()
        session = JogSession(HardwareBackend(axol, jelly), _FakePlanner, rate=200.0)
        thread, stop = _start(session)
        try:
            self.assertTrue(axol.entered and jelly.enabled)
            self.assertTrue(_wait(lambda: len(axol.sent) > 10))
            left, right = axol.sent[-1]
            # Streams the measured pose, gripper included.
            np.testing.assert_allclose(left, axol.left.positions)
            np.testing.assert_allclose(right, axol.right.positions)
            # No page connected: Jelly is never commanded, so its own
            # timeout keeps the wheels and lift stopped.
            self.assertEqual(jelly.commands, [])
            session.hold(0.5, 0.0, 0.0, 1)
            self.assertTrue(_wait(lambda: len(jelly.commands) > 0))
            self.assertEqual(jelly.commands[-1], (0.5, 0.0, 0.0, 1))
            state = session.state()
            self.assertEqual(state["mode"], "hardware")
            self.assertEqual(state["lift"]["raised"], 0.5)
            self.assertLess(state["lift"]["velocity"], 0.0)
            self.assertTrue(state["base"]["estimated"])
            time.sleep(HOLD_TIMEOUT_S + 0.1)
            count = len(jelly.commands)
            time.sleep(0.1)
            self.assertEqual(len(jelly.commands), count)
        finally:
            stop.set()
            thread.join(5)
        self.assertTrue(axol.exited and jelly.disabled)

    def test_stop_drives_the_arms_to_rest_before_disabling(self) -> None:
        axol = _FakeAxol()
        # Start away from rest, as an arm left somewhere by a jog would be.
        axol.left.positions = axol.left.positions + np.array(
            [0.3] * 7 + [0.0], np.float32
        )
        session = JogSession(HardwareBackend(axol, None), _FakePlanner, rate=200.0)
        thread, stop = _start(session)
        self.assertTrue(_wait(lambda: session.state()["ready"]))
        stop.set()
        thread.join(5)
        self.assertTrue(axol.exited)
        left, right = axol.sent[-1]
        np.testing.assert_allclose(
            np.concatenate([left[:7], right[:7]]), rest_q(), atol=1e-6
        )
        self.assertAlmostEqual(float(left[7]), 0.3, places=6)  # grip kept

    def test_absent_arm_is_never_commanded(self) -> None:
        axol = _FakeAxol()
        axol.right = None  # type: ignore[assignment]
        session = JogSession(HardwareBackend(axol, None), _FakePlanner, rate=200.0)
        thread, stop = _start(session)
        try:
            self.assertTrue(_wait(lambda: len(axol.sent) > 3))
            self.assertIsNone(axol.sent[-1][1])
            self.assertIsNone(session.state()["lift"])
        finally:
            stop.set()
            thread.join(5)


# -- the jog operation and API ---------------------------------------------


def test_jog_is_a_sim_capable_operation_using_gravity_comp_settings() -> None:
    from almond_axol.serve.commands import COMMANDS, build_argv
    from almond_axol.cli.config import parse
    from almond_axol.cli.jog import JogCmdConfig

    cmd = COMMANDS["jog"]
    assert cmd.is_operation and cmd.sim_capable and cmd.sim_flag == "sim"
    argv = build_argv(
        "jog",
        {
            "sim": True,
            "kinematics.robot_model": "mobile",
            "kd": 0.4,
            "rate_hz": 100,
            "left_channel": "can_alm_axol_l",
        },
    )
    cfg = parse(JogCmdConfig, argv)
    assert cfg.sim is True
    assert cfg.kinematics.robot_model == "mobile"
    assert cfg.rate_hz == 100


class _StubSession:
    def __init__(self) -> None:
        self.holds: list[tuple[Any, ...]] = []
        self.disconnected = 0

    def state(self) -> dict[str, Any]:
        return {"mode": "sim", "ready": True}

    def plan(
        self, arm: str, position: list[float], quaternion: list[float]
    ) -> dict[str, Any]:
        if arm == "right":
            raise JogError("The arms are still starting.")
        return {"id": 7, "arm": arm}

    def execute(self, plan_id: int) -> dict[str, Any]:
        if plan_id != 7:
            raise JogError("That plan is out of date. Plan the move again.")
        return {"id": 7, "duration": 1.0}

    def cancel(self) -> bool:
        return True

    def hold(
        self, vx: float, vy: float, wz: float, lift: int, source: str = ""
    ) -> None:
        assert source.startswith("ws-")
        self.holds.append((vx, vy, wz, lift))

    def release(self, source: str = "") -> None:
        self.holds.append(("release",))

    def disconnect(self, source: str = "") -> None:
        self.disconnected += 1


def _app(tmp_path: Path) -> Any:
    from almond_axol.serve import app as app_module

    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html>app</html>")
    return app_module.create_app(static_dir=static)


def test_jog_api_without_a_session(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/jog/state").json() == {"running": False}
        body = {"arm": "left", "position": [0, 0, 0], "quaternion": QUAT}
        assert client.post("/api/jog/plan", json=body).status_code == 409
        assert client.post("/api/jog/execute", json={"id": 1}).status_code == 409
        assert client.post("/api/jog/cancel").status_code == 409


def test_jog_api_routes_to_the_session(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    stub = _StubSession()
    with (
        patch("almond_axol.jog.session.active_session", return_value=stub),
        TestClient(_app(tmp_path)) as client,
    ):
        assert client.get("/api/jog/state").json() == {
            "running": True,
            "mode": "sim",
            "ready": True,
        }
        body = {"arm": "left", "position": [0.1, 0.2, 0.3], "quaternion": QUAT}
        assert client.post("/api/jog/plan", json=body).json()["id"] == 7
        refused = client.post("/api/jog/plan", json={**body, "arm": "right"})
        assert refused.status_code == 409
        assert refused.json()["error"] == "The arms are still starting."
        assert (
            client.post("/api/jog/plan", json={**body, "arm": "up"}).status_code == 422
        )
        assert client.post("/api/jog/execute", json={"id": 7}).json()["duration"] == 1.0
        assert client.post("/api/jog/execute", json={"id": 3}).status_code == 409
        assert client.post("/api/jog/cancel").json() == {"stopped": True}
        with client.websocket_connect("/api/jog/ws") as ws:
            assert ws.receive_json()["type"] == "state"
            ws.send_json({"type": "hold", "drive": [0.5, 0, 0], "lift": 1})
            ws.send_json({"type": "hold", "drive": [0.5, 0], "lift": 1})
            message = ws.receive_json()
            while message["type"] != "error":
                message = ws.receive_json()
            ws.send_json({"type": "release"})
            ws.receive_json()
        assert _wait(lambda: stub.disconnected >= 1)
        assert (0.5, 0, 0, 1) in stub.holds
        assert ("release",) in stub.holds


def test_jog_operation_end_to_end_in_sim(tmp_path: Path) -> None:
    """Start the real jog op through serve, plan, execute, lift and drive."""
    from fastapi.testclient import TestClient

    with TestClient(_app(tmp_path)) as client:
        started = client.post(
            "/api/op/start",
            json={
                "op": "jog",
                "args": {"sim": True, "kinematics.robot_model": "mobile"},
            },
        )
        assert started.status_code == 200, started.text
        try:
            assert _wait(lambda: client.get("/api/jog/state").json().get("ready"), 240)
            assert _wait(lambda: client.get("/api/jog/state").json().get("tips"), 10)
            state = client.get("/api/jog/state").json()
            assert state["mode"] == "sim" and state["lift"]["raised"] == 1.0
            tip = state["tips"]["left"]
            goal = [p + d for p, d in zip(tip["position"], (0.08, -0.04, 0.1))]
            plan = client.post(
                "/api/jog/plan",
                json={"arm": "left", "position": goal, "quaternion": tip["quaternion"]},
            ).json()
            assert plan["reachable"], plan
            assert plan["positionError"] < 2e-3
            assert len(plan["path"]) > 2
            assert (
                client.post("/api/jog/execute", json={"id": plan["id"]}).status_code
                == 200
            )
            assert _wait(
                lambda: (
                    client.get("/api/jog/state").json().get("lastExecution") or {}
                ).get("status")
                == "done",
                30,
            )
            time.sleep(0.3)
            reached = client.get("/api/jog/state").json()["tips"]["left"]["position"]
            assert np.linalg.norm(np.subtract(reached, goal)) < 2e-3
            with client.websocket_connect("/api/jog/ws") as ws:
                for _ in range(20):
                    ws.send_json({"type": "hold", "drive": [0.6, 0.0, 0.0], "lift": -1})
                    time.sleep(0.05)
                state = client.get("/api/jog/state").json()
                assert state["lift"]["velocity"] < 0
                assert state["base"]["x"] > 0
            assert _wait(
                lambda: client.get("/api/jog/state").json()["base"]["command"]
                == [0, 0, 0],
                2,
            )
        finally:
            client.post("/api/op/stop")
        assert _wait(
            lambda: client.get("/api/jog/state").json() == {"running": False}, 20
        )
