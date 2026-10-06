from __future__ import annotations

import math
import unittest

import jax.numpy as jnp
import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.kinematics.solver import KinematicsSolver, _base_collision_safe_step
from almond_axol.teleop.box import turn_about_line


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

    def test_turning_a_joint_turns_the_mount_about_its_axis_line(self) -> None:
        solver = self.solver
        q = np.random.default_rng(7).uniform(-0.6, 0.6, solver.num_joints)
        q = q.astype(np.float32)
        index = ARM_JOINTS.index(Joint.WRIST_2)
        turn = math.radians(39.0)
        lines = solver.joint_axes(q, Joint.WRIST_2, mount_frame=True)
        turned = q.copy()
        turned[solver.left_indices[index]] += turn
        turned[solver.right_indices[index]] -= turn
        before = dict(zip(("left", "right"), solver.fk(q)))
        after = dict(zip(("left", "right"), solver.fk(turned)))
        for side, sign in (("left", 1.0), ("right", -1.0)):
            pos, rot = turn_about_line(before[side], *lines[side], sign * turn)
            np.testing.assert_allclose(pos, after[side][0], atol=1e-5)
            np.testing.assert_allclose(rot, after[side][1], atol=1e-5)
        # The mount-frame line is the same after the turn.
        again = solver.joint_axes(turned, Joint.WRIST_2, mount_frame=True)
        for side in ("left", "right"):
            np.testing.assert_allclose(again[side][0], lines[side][0], atol=1e-5)
            np.testing.assert_allclose(again[side][1], lines[side][1], atol=1e-5)
        limits = solver.joint_limits(Joint.WRIST_2)
        self.assertLess(limits["left"][0], 0.0)
        self.assertGreater(limits["left"][1], turn)


if __name__ == "__main__":
    unittest.main()
