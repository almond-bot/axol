"""Real-model reset geometry, failure behavior, and JAX-free worker startup."""

from pathlib import Path
import subprocess
import sys
import textwrap
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("mink")

from almond_axol.constants import GRIPPER_TIP_OFFSET
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.mink_backend import MinkKinematicsSolver
from almond_axol.kinematics.mink_trajectory import (
    MinkPlanningError,
    _ResetProblem,
    plan_mink_trajectory,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop import worker as worker_module


@pytest.fixture(scope="module")
def solver():
    return MinkKinematicsSolver(KinematicsConfig(backend="mink"))


@pytest.fixture(scope="module")
def recorded():
    with np.load(Path(__file__).parent / "data/mink_ik_reference/stream.npz") as data:
        return {key: data[key].copy() for key in data.files}


def plan(solver, start, goal, **kwargs):
    return np.asarray(
        plan_mink_trajectory(
            solver, start, goal, speed=0.6, rate=120.0, min_duration=1.5, **kwargs
        )
    )


def tips(solver, q):
    return [p + r @ GRIPPER_TIP_OFFSET for p, r in solver.fk(q)]


@pytest.mark.parametrize("custom_rest", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_zero_rest_round_trip_reaches_exact_joint_goal_on_straight_tip_path(
    solver, recorded, custom_rest, reverse
):
    cfg = VRTeleopConfig()
    rest = (
        recorded["rest14"]
        if custom_rest
        else np.r_[cfg.rest_pose_left, cfg.rest_pose_right]
    )
    start, goal = (rest, np.zeros(14)) if reverse else (np.zeros(14), rest)
    model_flags = solver._ik.model.opt.disableflags
    before = solver.posture_pose
    path = plan(solver, start, goal)
    np.testing.assert_array_equal(path[0], np.asarray(start, dtype=np.float32))
    np.testing.assert_array_equal(path[-1], np.asarray(goal, dtype=np.float32))
    assert np.max(np.abs(np.diff(path, axis=0))) * 120 <= 1.875 * 0.6 + 1e-5
    np.testing.assert_array_equal(solver.posture_pose, before)
    assert solver._ik.model.opt.disableflags == model_flags

    # FK is the independently tested teleop adapter, not planner output metadata.
    for arm, (first, last) in enumerate(zip(tips(solver, start), tips(solver, goal))):
        direction = last - first
        for q in path[::7]:
            tip = tips(solver, q)[arm]
            fraction = np.clip(
                np.dot(tip - first, direction) / np.dot(direction, direction), 0, 1
            )
            assert np.linalg.norm(tip - first - fraction * direction) < 0.0051


@pytest.mark.parametrize("index", [0, 10, 30, 50, 100])
def test_recorded_policy_poses_return_including_start_clearance_recovery(
    solver, recorded, index
):
    start = recorded["expected_joints"][index]
    path = plan(solver, start, recorded["rest14"])
    np.testing.assert_array_equal(path[0], start)
    np.testing.assert_array_equal(path[-1], recorded["rest14"])
    # The recovery model checks every returned tick and its midpoint, never
    # permitting clearance to worsen while restoring the normal margin.
    problem = _ResetProblem(solver, 0.6, 1 / 120)
    problem.allow_start_recovery(start)
    for previous, q in zip(path, path[1:]):
        problem.validate((previous + q) / 2, "midpoint")
        problem.validate(q, "tick")
        problem.advance_clearance(q)


def test_measured_zero_limit_noise_recovers_without_snapping(solver):
    start = np.zeros(14)
    start[[3, 10]] = [-0.001, 0.001]
    path = plan(solver, start, np.zeros(14))
    np.testing.assert_array_equal(path[0], start.astype(np.float32))
    np.testing.assert_array_equal(path[-1], np.zeros(14))
    assert np.all(np.diff(np.minimum(path[:, 3], 0)) >= -1e-6)
    assert np.all(np.diff(np.maximum(path[:, 10], 0)) <= 1e-6)
    start[3] = -0.011
    with pytest.raises(MinkPlanningError, match="recovery exceeds"):
        plan(solver, start, np.zeros(14))
    with pytest.raises(MinkPlanningError, match="joint limit"):
        plan(solver, np.zeros(14), start)


def test_unchanged_arm_is_frozen_inside_reset_solve(solver, recorded):
    start = recorded["rest14"]
    goal = start.copy()
    goal[:7] = 0
    path = plan(solver, start, goal)
    np.testing.assert_array_equal(
        path[:, 7:], np.broadcast_to(start[7:], path[:, 7:].shape)
    )


def test_valid_endpoints_do_not_imply_a_safe_straight_reset(solver, recorded):
    start = np.array(
        [
            0.5043427856,
            -0.1204992376,
            0.5312146842,
            1.8067360357,
            -0.6825031830,
            0.2745272676,
            -0.1175720145,
            1.2648777445,
            0.6559060422,
            0.0266693508,
            -0.9321955047,
            0.7651608513,
            -0.6793303125,
            -0.6542348377,
        ]
    )
    problem = _ResetProblem(solver, 0.6, 1 / 120)
    problem.validate(start, "start")
    problem.validate(recorded["rest14"], "goal")
    with pytest.raises(MinkPlanningError):
        plan(solver, start, recorded["rest14"])


def test_qp_failure_never_retries_without_collision_limits(solver, recorded):
    with patch(
        "almond_axol.kinematics.mink_trajectory.mink.solve_ik",
        side_effect=RuntimeError("infeasible"),
    ) as solve:
        with pytest.raises(MinkPlanningError, match="collision limits enabled"):
            plan(solver, np.zeros(14), recorded["rest14"])
    assert solve.call_count == 1
    assert any(
        type(limit).__name__ == "CollisionAvoidanceLimit"
        for limit in solve.call_args.kwargs["limits"]
    )


def test_solver_hold_never_becomes_a_successful_partial_plan(solver, recorded):
    with patch(
        "almond_axol.kinematics.mink_trajectory.mink.solve_ik",
        return_value=np.zeros(solver._ik.model.nv),
    ):
        with pytest.raises(MinkPlanningError, match="straight Cartesian path"):
            plan(solver, np.zeros(14), recorded["rest14"])


def test_invalid_timing_or_goal_is_rejected(solver):
    with pytest.raises(ValueError, match="positive and finite"):
        plan(solver, np.zeros(14), np.zeros(14), linear_speed=0)
    with pytest.raises(MinkPlanningError, match="finite"):
        plan(solver, np.zeros(14), np.full(14, np.nan))


def test_planning_failure_keeps_worker_alive_for_retry():
    worker = Mock(left_indices=list(range(7)), right_indices=list(range(7, 14)))
    zero = np.zeros(14, dtype=np.float32)
    worker.get_rest_q.return_value = zero
    worker.compute_reset_trajectory.side_effect = [
        [zero],
        MinkPlanningError("blocked"),
        [zero],
    ]
    conn = Mock()
    conn.recv.side_effect = [
        ("reset", np.full(14, np.nan)),
        ("reset", zero),
        ("reset", zero),
        None,
    ]
    with (
        patch.object(worker_module, "IKWorker", return_value=worker),
        patch.object(worker_module.signal, "signal"),
        patch.object(worker_module.os, "nice"),
        patch("almond_axol.utils.affinity.pin_ik_startup"),
        patch("almond_axol.utils.affinity.pin_ik"),
    ):
        worker_module.run_ik_worker(
            conn, VRTeleopConfig(), KinematicsConfig(backend="mink")
        )
    assert [call.args[0][0] for call in conn.send.call_args_list] == [
        "ready",
        "reset_error",
        "reset_error",
        "reset_traj",
    ]
    worker.reset.assert_called_once()


def test_real_spawned_worker_startup_reset_and_park_leg_need_no_jax(tmp_path):
    script = tmp_path / "spawn_mink.py"
    script.write_text(
        textwrap.dedent("""
        import importlib.abc
        import multiprocessing as mp
        import sys
        class NoJax(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in {'jax', 'jaxlib', 'jaxlie', 'jaxls', 'pyroki'}:
                    raise ModuleNotFoundError('JAX is unavailable: ' + name, name=name)
        sys.meta_path.insert(0, NoJax())
        import numpy as np
        from almond_axol.kinematics.config import KinematicsConfig
        from almond_axol.teleop.config import VRTeleopConfig
        from almond_axol.teleop.worker import run_ik_worker
        def checked_worker(*args):
            run_ik_worker(*args)
            assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'jaxlie', 'jaxls', 'pyroki'} for name in sys.modules)
        if __name__ == '__main__':
            ctx = mp.get_context('spawn')
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=checked_worker,
                args=(child, VRTeleopConfig(), KinematicsConfig(backend='mink')))
            proc.start()
            child.close()
            try:
                assert parent.poll(20), 'worker startup timeout'
                ready = parent.recv()
                assert ready[0] == 'ready'
                rest = ready[1]
                for goal in (rest, np.zeros_like(rest)):
                    parent.send(('reset', rest, goal))
                    assert parent.poll(20), 'worker reset timeout'
                    result = parent.recv()
                    assert result[0] == 'reset_traj', result
                    np.testing.assert_array_equal(result[2][-1], goal)
                parent.send(None)
                proc.join(5)
                assert proc.exitcode == 0, proc.exitcode
            finally:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(5)
                parent.close()
    """)
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_teleop_rejected_plan_holds_command_and_does_not_claim_rest():
    import logging
    import threading
    from almond_axol.teleop.core import VRTeleopCore

    core = VRTeleopCore(
        VRTeleopConfig(), logging.getLogger("test"), broadcast_tracking=lambda _: None
    )
    q = np.linspace(0.0, 0.2, 14, dtype=np.float32)
    core.set_solution(q, list(range(7)), list(range(7, 14)))
    core.request_reset()
    stop = threading.Event()
    conn = Mock()

    def reject():
        stop.set()
        return "reset_error", "straight path is blocked"

    conn.recv.side_effect = reject
    core.run_ik_loop(conn, lambda: None, stop, lambda: True, lambda _: None)
    np.testing.assert_array_equal(core.q, q)
    assert not core._at_rest
    assert not core.reset_interp.is_active()
    assert not core.is_resetting
    assert not core.left_enabled and not core.right_enabled
    assert core._reset_error == "straight path is blocked"


def test_teleop_guarded_return_enters_retry_hold_after_rejected_plan():
    import asyncio
    import logging
    from unittest.mock import AsyncMock
    from almond_axol.teleop.core import VRTeleopCore

    core = VRTeleopCore(
        VRTeleopConfig(), logging.getLogger("test"), broadcast_tracking=lambda _: None
    )
    core._reset_error = "blocked"
    announce = Mock()
    with patch.object(
        core, "_contact_hold_until_reset", new=AsyncMock(return_value="stopped")
    ) as hold:
        asyncio.run(
            core.guarded_return(
                send_step=AsyncMock(),
                gravity_step=AsyncMock(),
                torque_residuals=lambda: (np.zeros(7), np.zeros(7)),
                reset_command_state=lambda: None,
                get_positions=lambda: (np.zeros(8), np.zeros(8)),
                stopped=lambda: False,
                announce=announce,
            )
        )
    hold.assert_awaited_once()
    assert "blocked" in announce.call_args.args[0]
