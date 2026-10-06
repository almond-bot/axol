from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from almond_axol.kinematics.solver import KinematicsSolver, _base_collision_safe_step


class KinematicsBoundaryRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = KinematicsSolver()

    def test_boundary_projection_preserves_feasible_motion(self) -> None:
        solver = self.solver
        q_from = np.zeros(solver.num_joints, dtype=np.float32)
        q_from_pyroki = solver.to_pyroki_order(q_from)
        q_candidate = q_from_pyroki.copy()
        actuated_names = list(solver.robot.joints.actuated_names)
        boundary_index = actuated_names.index("left_e1_0")
        feasible_index = actuated_names.index("left_e2_0")
        q_candidate[boundary_index] = -0.01  # At its lower limit at home.
        q_candidate[feasible_index] = 0.01  # A separate feasible movement.

        q_out, guard_active = _base_collision_safe_step(
            solver.robot,
            solver.robot_coll,
            jnp.asarray(q_from_pyroki),
            jnp.asarray(q_candidate),
            solver._base_clearance_floor,
            jnp.asarray(solver.config.max_joint_delta, dtype=jnp.float32),
        )
        q_out = np.asarray(q_out, dtype=np.float32)

        np.testing.assert_array_less(
            np.asarray(solver.robot.joints.lower_limits) - 1e-6,
            q_out,
        )
        np.testing.assert_array_less(
            q_out,
            np.asarray(solver.robot.joints.upper_limits) + 1e-6,
        )
        self.assertFalse(bool(guard_active))
        self.assertAlmostEqual(float(q_out[boundary_index]), 0.0, places=6)
        self.assertGreater(float(q_out[feasible_index]), 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(q_out - q_from_pyroki))),
            solver.config.max_joint_delta + 1e-6,
        )

    def test_step_after_reaching_limit_keeps_feasible_motion(self) -> None:
        # A step scaled to hit a limit lands a rounding error short of it,
        # not exactly on it; the next step must still treat that joint as
        # on its limit instead of letting the sliver of room scale every
        # other joint's motion to ~zero.
        solver = self.solver
        actuated_names = list(solver.robot.joints.actuated_names)
        boundary_index = actuated_names.index("left_e1_0")
        feasible_index = actuated_names.index("left_e2_0")
        q = solver.to_pyroki_order(np.zeros(solver.num_joints, dtype=np.float32))
        q[boundary_index] = 0.0037
        max_delta = jnp.asarray(solver.config.max_joint_delta, dtype=jnp.float32)

        for _ in range(2):
            q_candidate = q.copy()
            q_candidate[boundary_index] -= 0.01
            q_candidate[feasible_index] += 0.01
            q_out, _ = _base_collision_safe_step(
                solver.robot,
                solver.robot_coll,
                jnp.asarray(q),
                jnp.asarray(q_candidate),
                solver._base_clearance_floor,
                max_delta,
            )
            q_prev, q = q, np.asarray(q_out, dtype=np.float32)

        self.assertGreaterEqual(
            float(q[boundary_index]),
            float(solver.robot.joints.lower_limits[boundary_index]),
        )
        self.assertAlmostEqual(
            float(q[feasible_index] - q_prev[feasible_index]), 0.01, places=5
        )

    def test_zero_seed_public_ik_keeps_finite_progress(self) -> None:
        solver = self.solver
        q = np.zeros(solver.num_joints, dtype=np.float32)

        # The zero configuration is the documented straight-arm singular seed.
        q_goal = q.copy()
        q_goal[:7] = np.array((0.05, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0), np.float32)
        target = solver.fk(q_goal)[0]
        q_out = solver.ik(q, left_pose=target)

        self.assertTrue(np.isfinite(q_out).all())
        self.assertTrue(np.any(np.abs(q_out) > 1e-6))
        q_out_pyroki = solver.to_pyroki_order(q_out)
        np.testing.assert_array_less(
            np.asarray(solver.robot.joints.lower_limits) - 1e-6,
            q_out_pyroki,
        )
        np.testing.assert_array_less(
            q_out_pyroki,
            np.asarray(solver.robot.joints.upper_limits) + 1e-6,
        )
        self.assertLessEqual(
            float(np.max(np.abs(q_out - q))),
            solver.config.max_joint_delta + 1e-6,
        )


class ElbowSwivelTest(unittest.TestCase):
    """The elbow's free motion with the gripper mount held."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = KinematicsSolver()
        q = np.zeros(cls.solver.num_joints, dtype=np.float32)
        # Box mode's angled grasp, wrist_2 turned well off straight.
        q[:7] = np.radians((32.8, -12.1, -25.8, 31.4, 7.4, -30.8, 34.5))
        cls.q = q

    def test_the_shoulder_and_wrist_centres_are_fixed_on_the_arm(self) -> None:
        solver = self.solver
        rng = np.random.default_rng(0)
        reach, forearm = [], []
        for _ in range(4):
            q = rng.uniform(-0.6, 0.6, solver.num_joints).astype(np.float32)
            arm = solver.elbow_swivel(q)["left"]
            pos, rot = solver.fk(q)[0]
            wrist = pos + rot @ arm.wrist_in_mount
            reach.append(np.linalg.norm(arm.elbow - arm.shoulder))
            forearm.append(np.linalg.norm(wrist - arm.elbow))
            np.testing.assert_allclose(
                arm.shoulder, solver.elbow_swivel(self.q)["left"].shoulder
            )
        self.assertLess(np.ptp(reach), 1e-5)
        self.assertLess(np.ptp(forearm), 1e-5)

    def test_the_direction_swings_the_elbow_with_the_mount_held(self) -> None:
        solver = self.solver
        q = self.q
        arm = solver.elbow_swivel(q)["left"]
        self.assertAlmostEqual(float(np.linalg.norm(arm.direction)), 1.0, places=6)
        pose = solver.fk(q)[0]
        elbow = np.asarray(solver.elbow_positions(q)[0], np.float64)
        hint = (elbow + 0.01 * arm.direction).astype(np.float32)
        q_out = q.copy()
        for _ in range(30):
            solver.set_posture_pose(q_out)
            q_out = solver.ik(
                q_out,
                left_pose=pose,
                left_elbow_pos=hint,
                elbow_weight=10.0,
                pose_weight_scale=(2.0, 4.0),
            )
        pos, rot = solver.fk(q_out)[0]
        moved = solver.elbow_positions(q_out)[0] - elbow
        # Each solve is damped toward its seed, so the swing is slow; the
        # mount must not move with it.
        self.assertGreater(float(moved @ arm.direction), 3e-4)
        self.assertLess(float(np.linalg.norm(pos - pose[0])), 1e-3)
        cos = np.clip((np.trace(rot.T @ pose[1]) - 1.0) / 2.0, -1.0, 1.0)
        self.assertLess(float(np.degrees(np.arccos(cos))), 0.05)


if __name__ == "__main__":
    unittest.main()
