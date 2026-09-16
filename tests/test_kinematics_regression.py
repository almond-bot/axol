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


if __name__ == "__main__":
    unittest.main()
