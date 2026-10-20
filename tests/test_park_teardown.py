"""Teardown park: rest, then zero, before the torque comes off.

Every flow plays a guarded return-to-rest on the way out, because disabling
raised arms drops them under gravity, and then (``park_to_zero``) continues
to the all-zero joint pose with the grasp held. These tests pin the decision
that guards it — park while the arms are position-controlled, skip while they
are limp or the bus is silent, stop at rest when the zero leg is off or a leg
fails — driving the real engine against a stand-in for the robot driver
rather than against mocked collaborators.
"""

from __future__ import annotations

import asyncio
import collections
import signal
import threading
import time
import unittest
from unittest import mock

import numpy as np

from almond_axol.lerobot.rollout import arms_reporting
from almond_axol.teleop import shutdown
from almond_axol.teleop import teleop as teleop_module
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.teleop import VRTeleop

# Long enough for the park to plan and play the short test trajectory, short
# enough that a test that expects a park to run out its cap does not stall the
# suite.
_TEST_PARK_TIMEOUT_S = 0.5

# Where the arms are left before teardown, in radians on every arm joint.
_RAISED = 0.5
# The stand-in planner's rest pose, distinct from zero so the two park legs
# are told apart in the command stream.
_REST = 0.25


class _FakeArm:
    """One arm of the stand-in driver: a settled position cache."""

    def __init__(self) -> None:
        self.positions = np.full(8, _RAISED, dtype=np.float32)


class _FakeRobot:
    """Stands in for the Axol hardware driver at the CAN boundary.

    Records what teardown commanded (``commands``, ``gravity_cycles``) and can
    play the failure modes the park has to survive: a bus that never answers a
    position read, and a bus that refuses commands.
    """

    def __init__(
        self,
        *,
        positions_error: BaseException | None = None,
        positions_hang: bool = False,
        command_error: BaseException | None = None,
        residual: float = 0.0,
    ) -> None:
        self.left = _FakeArm()
        self.right = _FakeArm()
        self.commands: list[np.ndarray] = []
        self.gravity_cycles = 0
        self.disabled = False
        self._positions_error = positions_error
        self._positions_hang = positions_hang
        self._command_error = command_error
        self._residual = residual

    async def get_positions(self) -> tuple[np.ndarray, np.ndarray]:
        if self._positions_hang:
            await asyncio.sleep(60.0)
        if self._positions_error is not None:
            raise self._positions_error
        return self.left.positions, self.right.positions

    async def motion_control(self, *, left: np.ndarray, right: np.ndarray) -> None:
        if self._command_error is not None:
            raise self._command_error
        self.commands.append(np.asarray(left, dtype=np.float32))

    async def gravity_compensate(self, kd: float = 0.5) -> None:
        self.gravity_cycles += 1

    def torque_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        res = np.full(7, self._residual, dtype=np.float32)
        return res, res.copy()

    def reset_command_state(self) -> None:
        pass

    async def disable(self) -> None:
        self.disabled = True


class _FakeIKProcess:
    """Stands in for the IK worker subprocess.

    Live while the park needs it to plan a path, and exited once teardown
    joins it — the real worker leaves on its shutdown sentinel — so
    ``_disable`` runs end to end without spawning anything.
    """

    pid = 4242
    exitcode = 0

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self._alive = False


class _FakeIKPipe:
    """Stands in for the parent end of the IK worker pipe.

    Answers a reset request the way the worker's planner does — a short
    straight-line trajectory from the requested joints to the rest pose
    (``_REST`` on every joint), or to the explicit goal of a park's zero
    leg — so the real IK dispatch loop can plan the park. With
    ``honour_goal=False`` the zero leg is answered with the rest pose, as a
    planner that ignored the goal would.
    """

    def __init__(self, *, honour_goal: bool = True) -> None:
        self._replies: collections.deque[object] = collections.deque()
        self._honour_goal = honour_goal
        self.goals: list[np.ndarray | None] = []

    def send(self, msg: object) -> None:
        if isinstance(msg, tuple) and msg[0] == "reset":
            start = np.asarray(msg[1], dtype=np.float32)
            goal = np.asarray(msg[2], dtype=np.float32) if len(msg) > 2 else None
            self.goals.append(goal)
            target = np.full_like(start, _REST) if goal is None else goal.copy()
            if goal is not None and not self._honour_goal:
                target = np.full_like(start, _REST)
            traj = [start + (target - start) * (i / 20) for i in range(21)]
            self._replies.append(("reset_traj", target, traj))

    def recv(self) -> object:
        return self._replies.popleft()

    def poll(self, timeout: float = 0.0) -> bool:
        return bool(self._replies)


class _GatedIKPipe(_FakeIKPipe):
    """A planner whose first reply waits for ``release`` — a slow plan."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def recv(self) -> object:
        self.release.wait(timeout=2.0)
        return super().recv()


def _teleop(
    robot: _FakeRobot,
    *,
    ik_worker_alive: bool = True,
    ik_thread: bool = True,
    pipe: _FakeIKPipe | None = None,
    **config: object,
) -> VRTeleop:
    """A teleop session seated on ``robot``, ready to park.

    Skips ``enable()`` (VR server, IK subprocess) and instead seats the pieces
    a park reads: the solver's joint index maps, filters seeded at the arms'
    measured pose, a live IK worker, and the real IK dispatch loop running
    against a stand-in planner.
    """
    with mock.patch.object(teleop_module, "VRServer", return_value=mock.MagicMock()):
        session = VRTeleop(robot, config=VRTeleopConfig(**config))
    core = session._core  # noqa: SLF001 - seating engine state a park reads
    core.set_solution(
        np.full(14, _RAISED, dtype=np.float32), list(range(7)), list(range(7, 14))
    )
    core.seed_filters(robot.left.positions, robot.right.positions)
    session._ik_process = _FakeIKProcess(alive=ik_worker_alive)  # noqa: SLF001
    if ik_thread:
        stop = session._ik_stop  # noqa: SLF001
        thread = threading.Thread(
            target=core.run_ik_loop,
            args=(
                pipe or _FakeIKPipe(),
                lambda: None,
                stop,
                lambda: True,
                lambda _t: None,
            ),
            daemon=True,
        )
        thread.start()
        session._ik_thread = thread  # noqa: SLF001
    return session


def _stop_ik_thread(session: VRTeleop) -> None:
    thread = session._ik_thread  # noqa: SLF001
    if thread is not None:
        session._ik_stop.set()  # noqa: SLF001
        thread.join(timeout=2.0)


def _leave_rest(session: VRTeleop) -> None:
    """Mark the arms as driven away from the rest pose.

    The flag is normally cleared by a tracking engage, which needs a live
    headset frame stream; setting it directly keeps the test on the teardown
    behaviour it is about.
    """
    session._core._at_rest = False  # noqa: SLF001


def _visited(robot: _FakeRobot, value: float) -> bool:
    """Whether some command put every left arm joint at ``value``."""
    return any(np.allclose(cmd[:7], value, atol=1e-3) for cmd in robot.commands)


class ParkTeardownTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for name in ("PARK_TIMEOUT_S", "PARK_ZERO_TIMEOUT_S"):
            patcher = mock.patch.object(teleop_module, name, _TEST_PARK_TIMEOUT_S)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _session(self, robot: _FakeRobot, **kwargs: object) -> VRTeleop:
        session = _teleop(robot, **kwargs)
        self.addCleanup(_stop_ik_thread, session)
        return session

    async def test_position_controlled_arms_park_through_rest_to_zero(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot)
        _leave_rest(session)

        started = time.perf_counter()
        await session._park_arms()  # noqa: SLF001
        elapsed = time.perf_counter() - started

        # Both legs have to be driven all the way, not merely started: a
        # park that held position until its deadline would leave the arms
        # raised. Rest comes first, zero last.
        self.assertGreater(len(robot.commands), 1, "teardown did not drive the arms")
        self.assertTrue(_visited(robot, _REST), "the park skipped the rest pose")
        np.testing.assert_allclose(robot.commands[-1][:7], 0.0, atol=1e-5)
        self.assertLess(elapsed, 2 * _TEST_PARK_TIMEOUT_S)

    async def test_park_to_zero_off_stops_at_rest(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot, park_to_zero=False)
        _leave_rest(session)

        await session._park_arms()  # noqa: SLF001

        self.assertGreater(len(robot.commands), 1)
        np.testing.assert_allclose(robot.commands[-1][:7], _REST, atol=1e-5)
        self.assertFalse(_visited(robot, 0.0))

    async def test_the_grasp_is_held_through_both_legs(self) -> None:
        # A parked arm may still be holding something: every rest move
        # ramps the grippers open, but the park keeps the commanded grasp
        # until the torque comes off.
        robot = _FakeRobot()
        session = self._session(robot)
        _leave_rest(session)
        core = session._core  # noqa: SLF001
        core.set_initial_grips(0.3, 0.6)
        core.seed_filters(robot.left.positions, robot.right.positions)

        await session._park_arms()  # noqa: SLF001

        self.assertTrue(_visited(robot, 0.0))
        grips = {round(float(cmd[7]), 6) for cmd in robot.commands}
        self.assertEqual(grips, {0.3}, "the park let the gripper ramp open")

    async def test_the_zero_leg_asks_the_planner_for_zero(self) -> None:
        robot = _FakeRobot()
        pipe = _FakeIKPipe()
        session = self._session(robot, pipe=pipe)
        _leave_rest(session)

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(len(pipe.goals), 2)
        self.assertIsNone(pipe.goals[0])
        np.testing.assert_allclose(pipe.goals[1], 0.0)

    async def test_a_refused_zero_plan_leaves_the_arms_at_rest(self) -> None:
        # Zero is a torque-off pose: a plan that does not honour it is not
        # played, and the park ends at rest instead.
        robot = _FakeRobot()
        session = self._session(robot, pipe=_FakeIKPipe(honour_goal=False))
        _leave_rest(session)

        with self.assertLogs(teleop_module._logger, level="WARNING") as logs:
            await session._park_arms()  # noqa: SLF001

        np.testing.assert_allclose(robot.commands[-1][:7], _REST, atol=1e-5)
        self.assertFalse(_visited(robot, 0.0))
        self.assertTrue(any("refused" in line for line in logs.output), logs.output)

    async def test_a_second_ctrl_c_aborts_the_park(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot)
        _leave_rest(session)
        core = session._core  # noqa: SLF001

        outcome = await shutdown.park_arms(
            core,
            park_to_zero=True,
            send_step=session._guard_send_step,  # noqa: SLF001
            gravity_step=session._guard_gravity_step,  # noqa: SLF001
            torque_residuals=robot.torque_residuals,
            reset_command_state=robot.reset_command_state,
            get_positions=session._guard_positions,  # noqa: SLF001
            announce=lambda _msg: None,
            # "Ctrl+C" lands a few commands into the rest leg.
            abort=lambda: len(robot.commands) >= 3,
            rest_timeout_s=_TEST_PARK_TIMEOUT_S,
            zero_timeout_s=_TEST_PARK_TIMEOUT_S,
        )

        self.assertEqual(outcome, "aborted")
        self.assertLess(len(robot.commands), 10)
        self.assertFalse(_visited(robot, _REST))

    def test_abort_on_sigint_turns_ctrl_c_into_a_flag(self) -> None:
        abort = threading.Event()
        previous = signal.getsignal(signal.SIGINT)
        with shutdown.abort_on_sigint(abort):
            signal.raise_signal(signal.SIGINT)  # no KeyboardInterrupt
        self.assertTrue(abort.is_set())
        self.assertIs(signal.getsignal(signal.SIGINT), previous)

    async def test_a_stop_mid_reset_finishes_that_move(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot)
        # An X reset was planned and started playing when the stop landed:
        # the core already counts the arms as at rest, but they are not yet.
        session._core.request_reset()  # noqa: SLF001
        deadline = time.perf_counter() + 1.0
        while not session._core.reset_interp.is_active():  # noqa: SLF001
            self.assertLess(time.perf_counter(), deadline, "reset never planned")
            await asyncio.sleep(0.001)
        self.assertTrue(session._core.at_rest)  # noqa: SLF001

        await session._park_arms()  # noqa: SLF001

        self.assertGreater(len(robot.commands), 1)
        self.assertTrue(_visited(robot, _REST))
        np.testing.assert_allclose(robot.commands[-1][:7], 0.0, atol=1e-5)

    async def test_a_stop_mid_reset_holds_the_grasp_it_found(self) -> None:
        # An X reset ramps the grippers open as it plays. A park that lands
        # mid-ramp must freeze them where they are for the rest of that move
        # and the zero leg — not let them finish opening (dropping what they
        # hold) and then close them back to the park's grasp.
        robot = _FakeRobot()
        session = self._session(robot)
        core = session._core  # noqa: SLF001
        core.set_initial_grips(0.3, 0.6)
        core.seed_filters(robot.left.positions, robot.right.positions)
        core.request_reset()
        deadline = time.perf_counter() + 1.0
        while not core.reset_interp.is_active():
            self.assertLess(time.perf_counter(), deadline, "reset never planned")
            await asyncio.sleep(0.001)
        for _ in range(4):  # a few control ticks into the reset's ramp
            core.compute_output()
        found = float(core._last_cmd[7])  # noqa: SLF001
        self.assertGreater(found, 0.3, "the reset's ramp never started opening")

        await session._park_arms()  # noqa: SLF001

        self.assertTrue(_visited(robot, 0.0))
        grips = {round(float(cmd[7]), 6) for cmd in robot.commands}
        self.assertEqual(grips, {round(found, 6)}, "the grasp moved during the park")

    async def test_a_stop_while_a_reset_plans_holds_the_grasp(self) -> None:
        # The X reset is still in its planning round trip when the stop
        # lands, so nothing is playing yet for the park to freeze: the plan
        # must pick the park's grasp up when it is adopted, or the rest leg
        # ramps the grippers open and the zero leg closes them again.
        robot = _FakeRobot()
        pipe = _GatedIKPipe()
        session = self._session(robot, pipe=pipe)
        core = session._core  # noqa: SLF001
        core.set_initial_grips(0.3, 0.6)
        core.seed_filters(robot.left.positions, robot.right.positions)
        core.request_reset()
        deadline = time.perf_counter() + 1.0
        while not core._reset_dispatching:  # noqa: SLF001
            self.assertLess(time.perf_counter(), deadline, "reset never dispatched")
            await asyncio.sleep(0.001)

        park = asyncio.create_task(session._park_arms())  # noqa: SLF001
        while not core.is_parking:
            self.assertLess(time.perf_counter(), deadline, "park never began")
            await asyncio.sleep(0.001)
        pipe.release.set()
        await park

        self.assertTrue(_visited(robot, _REST))
        self.assertTrue(_visited(robot, 0.0))
        grips = {round(float(cmd[7]), 6) for cmd in robot.commands}
        self.assertEqual(grips, {0.3}, "the grasp moved during the park")

    async def test_contact_ends_the_park_without_a_limp_hold(self) -> None:
        # Nobody is left to hand-guide the arms and press reset, so holding
        # them limp until the cap would only delay the same torque-off — and
        # a rest leg that hit something must not be followed by the zero leg.
        robot = _FakeRobot(residual=100.0)
        session = self._session(robot)
        _leave_rest(session)

        with (
            mock.patch.object(teleop_module, "PARK_TIMEOUT_S", 5.0),
            mock.patch.object(teleop_module, "PARK_ZERO_TIMEOUT_S", 5.0),
        ):
            started = time.perf_counter()
            await session._park_arms()  # noqa: SLF001
            elapsed = time.perf_counter() - started

        self.assertGreater(len(robot.commands), 0)
        self.assertEqual(robot.gravity_cycles, 0)
        self.assertLess(elapsed, 1.0)
        self.assertFalse(_visited(robot, 0.0))

    async def test_missing_ik_dispatch_thread_skips_the_park(self) -> None:
        # Without the dispatch thread a reset request is never planned, and
        # the "park" would only hold the raised pose until its deadline.
        robot = _FakeRobot()
        session = self._session(robot, ik_thread=False)
        _leave_rest(session)

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(robot.commands, [])

    async def test_limp_arms_are_left_where_the_operator_has_them(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot)
        _leave_rest(session)
        # A contact hold freezes the IK pipeline for as long as the arms are
        # hand-guidable; parking then would pull against the operator.
        session._core.pause_ik()  # noqa: SLF001

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(robot.commands, [])
        self.assertEqual(robot.gravity_cycles, 0)

    async def test_stalled_bus_skips_the_park(self) -> None:
        robot = _FakeRobot(positions_hang=True)
        session = self._session(robot)
        _leave_rest(session)

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(robot.commands, [])

    async def test_arms_already_at_rest_still_park_to_zero(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot)

        await session._park_arms()  # noqa: SLF001

        self.assertGreater(len(robot.commands), 1)
        np.testing.assert_allclose(robot.commands[-1][:7], 0.0, atol=1e-5)

    async def test_arms_already_at_rest_are_not_moved_without_the_zero_leg(
        self,
    ) -> None:
        robot = _FakeRobot()
        session = self._session(robot, park_to_zero=False)

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(robot.commands, [])

    async def test_dead_ik_worker_skips_the_park(self) -> None:
        robot = _FakeRobot()
        session = self._session(robot, ik_worker_alive=False)
        _leave_rest(session)

        await session._park_arms()  # noqa: SLF001

        self.assertEqual(robot.commands, [])

    async def test_a_failing_park_still_torques_the_arms_off(self) -> None:
        robot = _FakeRobot(command_error=OSError("CAN write failed"))
        session = self._session(robot)
        _leave_rest(session)

        await session.disable()

        self.assertEqual(robot.commands, [])
        self.assertTrue(robot.disabled, "a failed park swallowed the torque-off")


class ArmsReportingTest(unittest.TestCase):
    """The liveness check the LeRobot flows' teardown park starts from."""

    def test_a_reporting_bus_allows_the_park(self) -> None:
        robot = mock.Mock()
        robot.positions = (np.zeros(8), np.zeros(8))

        self.assertTrue(arms_reporting(robot))

    def test_a_silent_bus_blocks_the_park(self) -> None:
        robot = mock.Mock()
        type(robot).positions = mock.PropertyMock(
            side_effect=RuntimeError("no telemetry sample")
        )

        self.assertFalse(arms_reporting(robot))


if __name__ == "__main__":
    unittest.main()
