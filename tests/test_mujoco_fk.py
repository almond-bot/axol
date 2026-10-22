"""Validate current-world FK against independent frozen model-frame FK samples."""

from pathlib import Path

import numpy as np
import pytest

from almond_axol.kinematics.mujoco_fk import (
    AxolForwardKinematics,
    pose6_to_pos_rot,
)


@pytest.fixture(scope="module")
def samples():
    with np.load(Path(__file__).parent / "data/mink_ik_reference/stream.npz") as source:
        yield {name: source[name] for name in source.files}


def test_current_world_fk_matches_independent_reference_samples(samples):
    fk = AxolForwardKinematics()
    # The current URDF adds +90 degrees about the translated root origin.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.0, 0.0, 0.86])
    for joints, positions, rotations in zip(
        samples["fk_joints"],
        samples["fk_positions"],
        samples["fk_rotations"],
        strict=True,
    ):
        poses = fk.ee_poses(joints[:7], joints[7:])
        for pose, model_position, model_rotation in zip(
            poses, positions, rotations, strict=True
        ):
            position, actual_rotation = pose6_to_pos_rot(pose)
            np.testing.assert_allclose(
                position, origin + rotation @ (model_position - origin), atol=3e-7
            )
            np.testing.assert_allclose(
                actual_rotation, rotation @ model_rotation, atol=1e-6
            )
            assert pose.shape == (6,)
            assert pose.dtype == np.float32


def test_pose_decoder_matches_independent_rotation_samples(samples):
    for pose, position, rotation in zip(
        samples["pose6"],
        samples["pose_positions"],
        samples["pose_rotations"],
        strict=True,
    ):
        actual_position, actual_rotation = pose6_to_pos_rot(pose)
        np.testing.assert_array_equal(actual_position, position)
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-6)


def test_gripper_openings_do_not_move_the_recorded_mount(samples):
    fk = AxolForwardKinematics()
    joints = samples["fk_joints"][3]
    expected = fk.ee_poses(joints[:7], joints[7:])
    actual = fk.ee_poses(np.r_[joints[:7], 0.0], np.r_[joints[7:], 1.0])
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("bad", [np.zeros(6), np.zeros((7, 1)), np.full(7, np.nan)])
def test_fk_refuses_invalid_joint_angles(bad):
    with pytest.raises(ValueError, match="seven finite joint angles"):
        AxolForwardKinematics().ee_poses(bad, np.zeros(7))


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((6, 1)), np.full(6, np.nan)])
def test_pose_decoder_refuses_invalid_targets(bad):
    with pytest.raises(ValueError, match="six finite values"):
        pose6_to_pos_rot(bad)
