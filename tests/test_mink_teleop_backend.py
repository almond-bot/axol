"""Current-frame Mink teleop, policy parity, and worker backend selection."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yourdfpy

pytest.importorskip("mink")

from almond_axol.constants import URDF_PATH, Joint, urdf_body_name
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.mink_backend import MinkKinematicsSolver
from almond_axol.policy.mink_ik import MinkIK
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.worker import IKWorker


@pytest.fixture
def solver():
    return MinkKinematicsSolver(KinematicsConfig(backend="mink"))


@pytest.fixture(scope="module")
def saved():
    path = Path(__file__).with_name("data") / "mink_ik_reference" / "stream.npz"
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def test_current_frame_fk_elbows_and_shoulders_match_current_urdf(solver, saved):
    urdf = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False)
    assert solver.num_joints == 14
    assert solver.left_indices == list(range(7))
    assert solver.right_indices == list(range(7, 14))
    for q in saved["fk_joints"][::3]:
        urdf.update_cfg(dict(zip(solver.joint_names, q, strict=True)))
        elbows = solver.elbow_positions(q)
        for index, pose in enumerate(solver.fk(q)):
            is_left = index == 0
            ee = urdf.get_transform(urdf_body_name(Joint.GRIPPER, is_left=is_left))
            elbow = urdf.get_transform(urdf_body_name(Joint.ELBOW, is_left=is_left))
            shoulder = urdf.get_transform(
                urdf_body_name(Joint.SHOULDER_1, is_left=is_left)
            )
            np.testing.assert_allclose(pose[0], ee[:3, 3], rtol=0, atol=1e-6)
            np.testing.assert_allclose(pose[1], ee[:3, :3], rtol=0, atol=1e-6)
            np.testing.assert_allclose(elbows[index], elbow[:3, 3], rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                solver.shoulder_positions["left" if is_left else "right"],
                shoulder[:3, 3],
                rtol=0,
                atol=1e-6,
            )


def test_two_arm_tracking_matches_policy_solver_with_same_frame_inputs(solver, saved):
    reference = MinkIK()
    q = saved["rest14"].copy()
    solver.set_posture_pose(q)
    reference.set_rest_posture(q)
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    root = np.array([-2.77556e-17, -6.93889e-18, 0.86])
    for tick in range(24):
        targets, model_frame = [], []
        for side in ("left", "right"):
            position = root + quarter_turn @ (saved[f"{side}_pos"][tick] - root)
            rotation = quarter_turn @ saved[f"{side}_rot"][tick]
            targets.append((position, rotation))
            model_frame.append(
                (
                    (root + quarter_turn.T @ (position - root)).astype(np.float32),
                    (quarter_turn.T @ rotation).astype(np.float32),
                )
            )
        expected = reference.solve(q, *model_frame)
        q = solver.ik(q, *targets)
        np.testing.assert_array_equal(q, expected, err_msg=f"tick {tick}")


def test_single_arm_tracking_constrains_and_preserves_inactive_arm(solver, saved):
    q = saved["rest14"].copy()
    solver.set_posture_pose(q)
    left, right = solver.fk(q)
    left = (left[0] + [0.015, 0.0, 0.01], left[1])
    right = (right[0] + [0.1, 0.1, 0.1], right[1])
    original_constraints = solver._ik.tracker._constraints
    original_solve = solver._ik.solve
    seen_constraints = []

    def capture(*args):
        seen_constraints.append(solver._ik.tracker._constraints)
        assert args[2] is None
        return original_solve(*args)

    with patch.object(solver._ik, "solve", side_effect=capture):
        result = solver.ik(q, left, right, active_sides=("left",))
    np.testing.assert_array_equal(result[7:], q[7:])
    assert np.max(np.abs(result[:7] - q[:7])) > 1e-6
    assert len(seen_constraints[0]) == len(original_constraints) + 1
    assert solver._ik.tracker._constraints is original_constraints
    assert np.max(np.abs(result - q)) <= solver.config.max_joint_delta + 1e-7


def test_constraints_restore_after_failed_solve(solver, saved):
    q = saved["rest14"]
    constraints = solver._ik.tracker._constraints
    with (
        patch.object(solver._ik, "solve", side_effect=RuntimeError("failed")),
        pytest.raises(RuntimeError, match="failed"),
    ):
        solver.ik(q, *solver.fk(q), active_sides=("right",))
    assert solver._ik.tracker._constraints is constraints


def test_posture_copy_and_explicit_tracking_reset(solver, saved):
    q = saved["rest14"].copy()
    solver.set_posture_pose(q)
    q[:] = 0
    posture = solver.posture_pose
    np.testing.assert_array_equal(posture, saved["rest14"])
    posture[:] = 0
    np.testing.assert_array_equal(solver.posture_pose, saved["rest14"])
    solver.ik(saved["rest14"], *solver.fk(saved["rest14"]))
    assert solver._ik.tracker._gate_scale is not None
    solver.reset_tracking_state()
    assert solver._ik.tracker._gate_scale is None
    assert all(value is None for value in solver._ik.tracker._last_ee_target.values())


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_timing_scale_fails_before_solve(solver, scale):
    with patch.object(solver._ik, "solve") as solve:
        with pytest.raises(ValueError, match="delta_scale"):
            solver.ik(np.zeros(14), delta_scale=scale)
        solve.assert_not_called()


def test_invalid_inputs_and_no_active_arm(solver, saved):
    q = saved["rest14"]
    with pytest.raises(ValueError, match="14 finite"):
        solver.fk(np.full(14, np.nan))
    with pytest.raises(ValueError, match="14 finite"):
        solver.ik(np.zeros(16))
    with pytest.raises(ValueError, match="active_sides"):
        solver.ik(q, active_sides=("wrong",))
    with pytest.raises(ValueError, match="targets must be finite"):
        solver.ik(q, left_pose=(np.full(3, np.nan), np.eye(3)))
    with patch.object(solver._ik, "solve") as solve:
        np.testing.assert_array_equal(solver.ik(q, *solver.fk(q), active_sides=()), q)
        solve.assert_not_called()


def test_unsupported_elbow_configuration_fails_before_building_model():
    with patch("almond_axol.kinematics.mink_backend.MinkIK") as create:
        with pytest.raises(ValueError, match="elbow_weight=0"):
            MinkKinematicsSolver(KinematicsConfig(backend="mink", elbow_weight=1))
        create.assert_not_called()


def test_default_worker_keeps_jax_and_unknown_backend_is_rejected():
    fake = Mock(
        num_joints=14, left_indices=list(range(7)), right_indices=list(range(7, 14))
    )
    with (
        patch(
            "almond_axol.teleop.worker._make_jax_solver", return_value=fake
        ) as create,
        patch.object(IKWorker, "_settle_rest_pose", return_value=np.zeros(14)),
    ):
        config = KinematicsConfig()
        worker = IKWorker(VRTeleopConfig(), config)
        create.assert_called_once_with(config)
        assert worker._solver is fake
        assert not worker._mink_backend
    with patch("almond_axol.teleop.worker._make_jax_solver") as create:
        with pytest.raises(ValueError, match="backend must"):
            IKWorker(VRTeleopConfig(), KinematicsConfig(backend="unknown"))
        create.assert_not_called()


def test_mink_worker_keeps_rest_posture_and_uses_mink_for_reset(saved):
    cfg = VRTeleopConfig(
        rest_pose_left=saved["rest14"][:7],
        rest_pose_right=saved["rest14"][7:],
        ik_frequency=30,
    )
    with (
        patch("almond_axol.teleop.worker._make_jax_solver") as create_jax,
        patch.object(
            IKWorker, "_settle_rest_pose", side_effect=AssertionError("JAX only")
        ),
        patch("almond_axol.kinematics.mink_trajectory.plan_mink_trajectory") as plan,
    ):
        worker = IKWorker(cfg, KinematicsConfig(backend="mink"))
        assert isinstance(worker._solver, MinkKinematicsSolver)
        np.testing.assert_array_equal(worker.get_rest_q(), saved["rest14"])
        create_jax.assert_not_called()
        for _ in range(2):
            worker.compute_reset_trajectory(saved["rest14"], saved["rest14"])
        create_jax.assert_not_called()
        assert plan.call_args.args[0] is worker._solver
        assert plan.call_count == 2
        worker.reset()
        np.testing.assert_array_equal(worker._solver.posture_pose, saved["rest14"])


def test_mink_worker_engage_keeps_policy_rest_and_passes_active_sides():
    from tests.test_ik_freeze_clutch import _frame, _step_worker

    worker = _step_worker()
    worker._mink_backend = True
    q = np.zeros(14, dtype=np.float32)
    rest = worker._solver.posture_pose
    worker._active["left"] = False
    worker.step(_frame(left_forward=0, t_ms=1000), q)
    np.testing.assert_array_equal(worker._solver.posture_pose, rest)
    with patch.object(worker._solver, "ik", return_value=q.copy()) as solve:
        worker.step(_frame(left_forward=0, t_ms=1010, r_lock=False), q)
        assert solve.call_args.kwargs["active_sides"] == ("left",)
