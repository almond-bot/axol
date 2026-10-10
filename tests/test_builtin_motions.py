"""Built-in reference motions (slow_osc, fast_swing, hold): generated in code,
inside the joint limits, clear of the base, and always loadable by name."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from almond_axol.tuning import motion as motion_mod
from almond_axol.tuning.motion import (
    BUILTIN_MOTIONS,
    ReferenceMotion,
    _motion_limits,
    list_motions,
    load_motion,
    save_motion,
    start_pose,
)


def _peak_dps(m: ReferenceMotion, cols: slice = slice(None)) -> float:
    v = np.abs(np.diff(m.q[:, cols].astype(float), axis=0)) * m.rate
    return math.degrees(float(v.max()))


class BuiltinMotionTest(unittest.TestCase):
    def test_every_builtin_inside_the_limits_and_at_rest_at_both_ends(self) -> None:
        lo, hi = _motion_limits()
        for name, make in BUILTIN_MOTIONS.items():
            with self.subTest(name):
                m = make()
                self.assertEqual(m.name, name)
                self.assertEqual(m.q.shape[1], 14)
                q = m.q.astype(float)
                self.assertTrue(np.all(q >= lo) and np.all(q <= hi))
                np.testing.assert_allclose(q[0], start_pose(), atol=1e-6)
                np.testing.assert_allclose(q[-1], start_pose(), atol=1e-6)
                self.assertTrue(m.meta["builtin"])

    def test_slow_osc_is_slow_and_fast_swing_is_fast(self) -> None:
        slow = BUILTIN_MOTIONS["slow_osc"]()
        self.assertLess(_peak_dps(slow), 25.0)
        # The sweep itself (after the 1 s still + 7 s approach): shoulder_1 at
        # ~9°/s, the elbow coupled at 1.2x.
        sweep = slice(
            round(8 * slow.rate), round(8 * slow.rate) + round(28 * slow.rate)
        )
        sweep_q = ReferenceMotion("s", slow.rate, slow.q[sweep])
        self.assertLess(_peak_dps(sweep_q), 12.0)
        fast = BUILTIN_MOTIONS["fast_swing"]()
        self.assertGreater(_peak_dps(fast), 120.0)
        self.assertLess(_peak_dps(fast), 180.0)

    def test_right_arm_motions_hold_the_left_arm_and_mirror_onto_it(self) -> None:
        for name in ("slow_osc", "fast_swing"):
            with self.subTest(name):
                right = BUILTIN_MOTIONS[name]()
                left = BUILTIN_MOTIONS[f"{name}_left"]()
                self.assertEqual(_peak_dps(right, slice(0, 7)), 0.0)
                self.assertEqual(_peak_dps(left, slice(7, 14)), 0.0)
                np.testing.assert_array_equal(left.q[:, :7], -right.q[:, 7:])

    def test_hold_never_moves(self) -> None:
        self.assertEqual(_peak_dps(BUILTIN_MOTIONS["hold"]()), 0.0)

    def test_builtins_load_by_name_and_a_file_overrides_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(motion_mod, "MOTIONS_DIR", Path(tmp)):
                self.assertTrue(load_motion("slow_osc").meta["builtin"])
                names = [m.name for m in list_motions()]
                self.assertEqual(names, sorted(BUILTIN_MOTIONS))
                own = ReferenceMotion(
                    "slow_osc", 240.0, np.zeros((10, 14), np.float32), {"source": "x"}
                )
                save_motion(own)
                self.assertNotIn("builtin", load_motion("slow_osc").meta)
                listed = {m.name: m for m in list_motions()}
                self.assertEqual(len(listed["slow_osc"].q), 10)
                with self.assertRaises(FileNotFoundError):
                    load_motion("no_such_motion")


class BuiltinMotionCollisionTest(unittest.TestCase):
    """Every sample stays outside the collision solver's activation shell —
    the margin the teleop return-to-rest planner keeps from the base."""

    def test_no_builtin_enters_the_collision_shell(self) -> None:
        import jax
        import jax.numpy as jnp

        from almond_axol.kinematics.model import collision_cost_params
        from almond_axol.kinematics.solver import KinematicsSolver

        solver = KinematicsSolver()
        starts, _ = collision_cost_params(solver.robot, solver.robot_coll, 0.025)
        dist = jax.jit(
            jax.vmap(
                lambda c: solver.robot_coll.compute_self_collision_distance(
                    solver.robot, c
                )
            )
        )
        for name, make in BUILTIN_MOTIONS.items():
            with self.subTest(name):
                m = make()
                q = m.q[:: int(m.rate // 60)]
                full = np.zeros((len(q), solver.num_joints), np.float32)
                full[:, solver.left_indices] = q[:, :7]
                full[:, solver.right_indices] = q[:, 7:]
                cfg = np.stack([solver.to_pyroki_order(r) for r in full])
                d = np.asarray(dist(jnp.asarray(cfg)))
                self.assertLessEqual(float((starts[None, :] - d).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
