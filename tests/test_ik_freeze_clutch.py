from __future__ import annotations

import math
import types
import unittest
from unittest.mock import patch

import numpy as np

from almond_axol.teleop.worker import IKWorker, _relative_target_np
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), np.float32)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), np.float32)


def _freeze_tracker() -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._freeze_since = {}
    worker._freeze_targets = {}
    return worker


class _IdentityFilter:
    def update(self, value: np.ndarray, t: float | None = None) -> np.ndarray:
        del t
        return np.asarray(value, dtype=np.float32).copy()

    def nudge(self, delta: np.ndarray) -> None:
        del delta

    def reset(self, seed: np.ndarray | None = None) -> None:
        del seed


class _SeedLeftSolver:
    left_indices = list(range(7))
    right_indices = list(range(7, 14))
    num_joints = 14

    def __init__(self) -> None:
        self.left_pose = (
            np.array((0.40, 0.20, 0.30), np.float32),
            np.eye(3, dtype=np.float32),
        )
        self.right_pose = (
            np.array((0.40, -0.20, 0.30), np.float32),
            np.eye(3, dtype=np.float32),
        )
        self.left_elbow = np.array((0.10, 0.20, 0.30), np.float32)
        self.right_elbow = np.array((0.10, -0.20, 0.30), np.float32)
        self._posture = np.arange(14, dtype=np.float32)

    @property
    def posture_pose(self) -> np.ndarray:
        return self._posture.copy()

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture = np.asarray(q, dtype=np.float32).copy()

    def fk(
        self, q: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
        del q
        return self.left_pose, self.right_pose

    def elbow_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del q
        return self.left_elbow, self.right_elbow

    def ik(self, q: np.ndarray, **kwargs: object) -> np.ndarray:
        del kwargs
        out = np.asarray(q, dtype=np.float32).copy()
        # The left arm is stuck at its seed while the right arm remains healthy.
        out[self.right_indices[0]] += 0.01
        return out


def _step_worker() -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._config = types.SimpleNamespace(
        ik_frequency=120.0,
        position_multiplier=1.0,
        rotation_multiplier=1.0,
        reengage="clutch",
    )
    worker._solver = _SeedLeftSolver()
    worker._use_elbow = False
    worker._active = {"left": True, "right": True}
    worker._hold_fk = {}
    worker._hold_elbow_fk = {}
    worker._ramp = {}
    worker._box = None
    worker._box_leader = None
    worker._freeze_since = {}
    worker._freeze_targets = {}
    worker._snap_ctrl = {
        "left": (np.zeros(3, np.float32), np.eye(3, dtype=np.float32)),
        "right": (np.zeros(3, np.float32), np.eye(3, dtype=np.float32)),
    }
    worker._snap_fk = {
        "left": worker._solver.left_pose,
        "right": worker._solver.right_pose,
    }
    worker._snap_elbow_ctrl = {}
    worker._snap_elbow_fk = {}
    worker._prev_raw = {}
    worker._prev_raw_t = None
    worker._raw_vel = {}
    worker._suspect = None
    worker._last_solve_t = None
    worker._rec = None
    worker._f_l_pos = _IdentityFilter()
    worker._f_l_quat = _IdentityFilter()
    worker._f_r_pos = _IdentityFilter()
    worker._f_r_quat = _IdentityFilter()
    worker._f_l_elbow = _IdentityFilter()
    worker._f_r_elbow = _IdentityFilter()
    return worker


def _frame(
    *,
    left_forward: float,
    t_ms: float,
    l_lock: bool = True,
    r_lock: bool = True,
) -> VRFrame:
    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        # VR +z is robot +x (forward).
        l_ee=VRPose(
            position=VRPosition(x=0.0, y=0.0, z=left_forward),
            quaternion=identity,
        ),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=l_lock,
        r_lock=r_lock,
        t=t_ms,
    )


class RelativeTargetReanchorTest(unittest.TestCase):
    def test_current_controller_pose_maps_exactly_to_reanchored_fk(self) -> None:
        ctrl_pos = np.array((0.2, -0.1, 0.7), np.float32)
        ctrl_rot = _rot_z(0.4)
        fk_pos = np.array((0.5, 0.25, 0.35), np.float32)
        fk_rot = _rot_y(-0.3)

        target_pos, target_rot = _relative_target_np(
            ctrl_pos,
            ctrl_rot,
            ctrl_pos,
            ctrl_rot,
            fk_pos,
            fk_rot,
            position_multiplier=2.0,
            rotation_multiplier=1.5,
        )

        np.testing.assert_allclose(target_pos, fk_pos, atol=1e-7)
        np.testing.assert_allclose(target_rot, fk_rot, atol=1e-7)

        # A new controller delta starts from the new origin; no old backlog is
        # hidden in the mapping. Controller-local +x maps to EE-local +z.
        moved_ctrl = ctrl_pos + ctrl_rot @ np.array((0.01, 0.0, 0.0), np.float32)
        moved_pos, _ = _relative_target_np(
            moved_ctrl,
            ctrl_rot,
            ctrl_pos,
            ctrl_rot,
            fk_pos,
            fk_rot,
            position_multiplier=2.0,
            rotation_multiplier=1.5,
        )
        np.testing.assert_allclose(moved_pos, fk_pos + 0.02 * fk_rot[:, 2], atol=1e-7)


class FreezeTrackerTest(unittest.TestCase):
    def test_duration_and_translation_thresholds_gate_clutch(self) -> None:
        worker = _freeze_tracker()
        pos = np.zeros(3, np.float32)
        rot = np.eye(3, dtype=np.float32)

        with patch(
            "almond_axol.teleop.worker.time.monotonic",
            side_effect=(10.0, 10.49, 10.51),
        ):
            self.assertFalse(worker._note_solve("left", True, pos, rot, None))
            self.assertFalse(
                worker._note_solve(
                    "left", True, pos + np.array((0.006, 0.0, 0.0)), rot, None
                )
            )
            self.assertTrue(
                worker._note_solve(
                    "left", True, pos + np.array((0.006, 0.0, 0.0)), rot, None
                )
            )

        self.assertNotIn("left", worker._freeze_since)
        self.assertNotIn("left", worker._freeze_targets)

    def test_rotation_only_freeze_requests_clutch(self) -> None:
        worker = _freeze_tracker()
        pos = np.zeros(3, np.float32)
        rot = np.eye(3, dtype=np.float32)

        with patch("almond_axol.teleop.worker.time.monotonic", side_effect=(3.0, 3.51)):
            self.assertFalse(worker._note_solve("right", True, pos, rot, None))
            self.assertTrue(
                worker._note_solve("right", True, pos, _rot_z(math.radians(6)), None)
            )

    def test_elbow_motion_counts_and_progress_clears_only_one_arm(self) -> None:
        worker = _freeze_tracker()
        pos = np.zeros(3, np.float32)
        rot = np.eye(3, dtype=np.float32)
        elbow = np.zeros(3, np.float32)

        with patch(
            "almond_axol.teleop.worker.time.monotonic", side_effect=(1.0, 1.0, 1.51)
        ):
            self.assertFalse(worker._note_solve("left", True, pos, rot, elbow))
            self.assertFalse(worker._note_solve("right", True, pos, rot, elbow))
            worker._note_solve("left", False, pos, rot, elbow)
            self.assertTrue(
                worker._note_solve(
                    "right",
                    True,
                    pos,
                    rot,
                    elbow + np.array((0.0, 0.006, 0.0)),
                )
            )

        self.assertNotIn("left", worker._freeze_since)
        self.assertNotIn("right", worker._freeze_since)

    def test_clear_without_side_resets_all_arms(self) -> None:
        worker = _freeze_tracker()
        worker._freeze_since = {"left": 1.0, "right": 2.0}
        pose = (np.zeros(3), np.eye(3), None)
        worker._freeze_targets = {"left": pose, "right": pose}

        worker._clear_freeze()

        self.assertEqual(worker._freeze_since, {})
        self.assertEqual(worker._freeze_targets, {})


class FreezeClutchStepTest(unittest.TestCase):
    def test_stalled_arm_reanchors_without_blocking_healthy_arm(self) -> None:
        worker = _step_worker()
        right_snap_ctrl = tuple(value.copy() for value in worker._snap_ctrl["right"])
        right_snap_fk = tuple(value.copy() for value in worker._snap_fk["right"])
        right_posture = worker._solver.posture_pose[7:].copy()
        q0 = np.zeros(14, np.float32)

        with patch(
            "almond_axol.teleop.worker.time.monotonic", side_effect=(20.0, 20.51)
        ):
            q1 = worker.step(_frame(left_forward=0.0, t_ms=0.0), q0)
            q2 = worker.step(_frame(left_forward=0.006, t_ms=8.33), q1)

        # The stalled arm has no command step on the clutch frame. The healthy
        # arm's result from that same bimanual solve is retained.
        np.testing.assert_array_equal(q2[:7], q1[:7])
        self.assertAlmostEqual(float(q2[7] - q1[7]), 0.01, places=6)

        np.testing.assert_array_equal(worker._snap_ctrl["right"][0], right_snap_ctrl[0])
        np.testing.assert_array_equal(worker._snap_ctrl["right"][1], right_snap_ctrl[1])
        np.testing.assert_array_equal(worker._snap_fk["right"][0], right_snap_fk[0])
        np.testing.assert_array_equal(worker._snap_fk["right"][1], right_snap_fk[1])
        np.testing.assert_array_equal(worker._solver.posture_pose[7:], right_posture)

        # The current filtered left-controller sample now maps exactly to FK of
        # the held left joints; the six millimetres accumulated while frozen is
        # absent from the next solve target.
        np.testing.assert_allclose(
            worker._snap_ctrl["left"][0], np.array((0.006, 0.0, 0.0)), atol=1e-7
        )
        target_pos, target_rot = _relative_target_np(
            *worker._snap_ctrl["left"],
            *worker._snap_ctrl["left"],
            *worker._snap_fk["left"],
        )
        np.testing.assert_allclose(target_pos, worker._solver.left_pose[0], atol=1e-7)
        np.testing.assert_allclose(target_rot, worker._solver.left_pose[1], atol=1e-7)

    def test_single_arm_reengage_repins_only_that_arms_posture(self) -> None:
        # Both arms tracking; the right arm has travelled since its posture
        # was pinned (posture is arange(14), q_current sits elsewhere).
        worker = _step_worker()
        posture_before = worker._solver.posture_pose
        q0 = np.full(14, 0.5, np.float32)

        with patch(
            "almond_axol.teleop.worker.time.monotonic", side_effect=(30.0, 30.01)
        ):
            # Freeze the left arm, then re-engage it: the rising lock is an
            # engage snap for the left arm only.
            q1 = worker.step(_frame(left_forward=0.0, t_ms=0.0, l_lock=False), q0)
            q2 = worker.step(_frame(left_forward=0.0, t_ms=8.33), q1)

        # The snap frame itself produces no motion.
        np.testing.assert_array_equal(q2, q1)
        posture_after = worker._solver.posture_pose
        # The re-engaging arm is pinned to its held joints...
        np.testing.assert_array_equal(posture_after[:7], q1[:7])
        # ...while the arm that kept tracking keeps its existing attractor, so
        # its null-space equilibrium is untouched by the other arm's unfreeze.
        np.testing.assert_array_equal(posture_after[7:], posture_before[7:])
        self.assertFalse(np.array_equal(posture_after[7:], q1[7:]))

    def test_snap_arm_reanchors_elbow_origin_too(self) -> None:
        worker = _freeze_tracker()
        worker._ramp = {}
        worker._snap_ctrl = {}
        worker._snap_fk = {}
        worker._snap_elbow_ctrl = {}
        worker._snap_elbow_fk = {}
        ctrl_pos = np.array((0.1, 0.2, 0.3), np.float32)
        ctrl_rot = _rot_z(0.2)
        ctrl_elbow = np.array((0.0, 0.3, 0.4), np.float32)
        fk_pose = (np.array((0.4, 0.2, 0.3), np.float32), _rot_y(0.2))
        fk_elbow = np.array((0.2, 0.3, 0.4), np.float32)

        worker._snap_arm("left", ctrl_pos, ctrl_rot, ctrl_elbow, fk_pose, fk_elbow)

        np.testing.assert_array_equal(worker._snap_elbow_ctrl["left"], ctrl_elbow)
        np.testing.assert_array_equal(worker._snap_elbow_fk["left"], fk_elbow)


if __name__ == "__main__":
    unittest.main()
