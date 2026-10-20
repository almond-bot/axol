"""Whole-body IK: the Jelly base and lift as joints, behind a switch.

Arm IK (the default) is covered by the rest of the suite; this pins the
whole-body view of axol_jelly.urdf and its switch: it agrees with arm IK at
every body joint's zero, the solver keeps the body still inside the arms' reach and
moves it outside, and every surface that cannot follow the body refuses.
"""

from __future__ import annotations

import unittest

import numpy as np

from almond_axol.constants import (
    BODY_JOINTS,
    AxolModel,
    check_whole_body,
    torso_links,
    urdf_arm_joint_names,
    urdf_path,
)
from almond_axol.kinematics.model import ik_urdf

ARM_JOINT_NAMES = urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(
    is_left=False
)


class WholeBodyUrdfTest(unittest.TestCase):
    """Whole-body and arm IK load the same file (axol_jelly.urdf), with
    different joints frozen (:func:`ik_urdf`)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.path = urdf_path(AxolModel.MOBILE)
        cls.body = ik_urdf(cls.path, whole_body=True)
        cls.arm = ik_urdf(cls.path)

    def test_only_mobile_has_whole_body_joints(self) -> None:
        for model in (AxolModel.CLASSIC, AxolModel.OX_CART, AxolModel.OWL):
            with self.assertRaises(ValueError):
                check_whole_body(model)
            with self.assertRaises(ValueError):
                torso_links(model, whole_body=True)

    def test_arm_and_body_joints_move(self) -> None:
        self.assertEqual(
            set(self.body.actuated_joint_names), set(ARM_JOINT_NAMES) | set(BODY_JOINTS)
        )
        # The second lift stage follows the first.
        mimic = self.body.joint_map["lift_2"].mimic
        self.assertEqual(mimic.joint, "lift")
        lift = self.body.joint_map["lift"]
        self.assertEqual((lift.limit.lower, lift.limit.upper), (0.0, 0.34))
        self.assertEqual(set(self.arm.actuated_joint_names), set(ARM_JOINT_NAMES))

    def test_zero_body_pose_is_the_arm_model(self) -> None:
        # Same world frame at startup: every link where arm IK has it, for
        # arbitrary arm angles.
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = dict(zip(ARM_JOINT_NAMES, rng.uniform(-1.0, 1.0, 14)))
            self.arm.update_cfg(q)
            self.body.update_cfg({**q, **{n: 0.0 for n in BODY_JOINTS}})
            for link in self.arm.link_map:
                np.testing.assert_allclose(
                    self.body.get_transform(link),
                    self.arm.get_transform(link),
                    atol=1e-6,
                    err_msg=link,
                )

    def test_body_joints_move_the_robot_as_named(self) -> None:
        zero = {n: 0.0 for n in self.body.actuated_joint_names}
        self.body.update_cfg(zero)
        s1 = self.body.get_transform("s1")[:3, 3].copy()
        floor = self.body.get_transform("floor")[:3, 3].copy()
        self.body.update_cfg({**zero, "base_x": 0.5, "base_y": -0.2, "lift": 0.1})
        moved = self.body.get_transform("s1")[:3, 3]
        # Forward, right, and the torso drops both stages' worth.
        np.testing.assert_allclose(moved - s1, [0.5, -0.2, -0.2], atol=1e-9)
        # The ground stays put.
        np.testing.assert_allclose(self.body.get_transform("floor")[:3, 3], floor)


class WholeBodyCollisionTest(unittest.TestCase):
    def test_arms_collide_with_every_body_section(self) -> None:
        from almond_axol.kinematics.model import shared_robot_collision

        rc = shared_robot_collision(AxolModel.MOBILE, whole_body=True)
        torso = set(torso_links(AxolModel.MOBILE, whole_body=True))
        self.assertTrue({"lift_stage", "jelly", "deck_0", "deck_1", "deck_2"} <= torso)
        pairs = {
            frozenset((rc.link_names[int(i)], rc.link_names[int(j)]))
            for i, j in zip(rc.active_idx_i, rc.active_idx_j)
        }
        # Frame-only links (camera frames, finger frames) take part in no pair.
        solid = {n for n, r in zip(rc.link_names, np.asarray(rc.coll.radius)) if r > 0}
        arms = {
            n
            for n in solid
            if n.startswith(("left_", "right_")) and not n.endswith(("_s2", "_s3"))
        }
        self.assertEqual(pairs, {frozenset((t, a)) for t in torso for a in arms})


class WholeBodySolverTest(unittest.TestCase):
    """One solver (JIT ~20 s): the body holds in reach, moves out of it."""

    @classmethod
    def setUpClass(cls) -> None:
        from almond_axol.kinematics.config import KinematicsConfig
        from almond_axol.kinematics.solver import KinematicsSolver
        from almond_axol.teleop.config import VRTeleopConfig

        cls.solver = KinematicsSolver(KinematicsConfig(whole_body=True))
        rest = VRTeleopConfig()
        cls.q0 = np.concatenate(
            [rest.rest_pose_left, rest.rest_pose_right, np.zeros(len(BODY_JOINTS))]
        ).astype(np.float32)
        cls.poses = cls.solver.fk(cls.q0)

    def _drive(self, offset, steps: int) -> np.ndarray:
        (lp, lr), (rp, rr) = self.poses
        d = np.asarray(offset, dtype=np.float32)
        q = self.q0.copy()
        for _ in range(steps):
            q = self.solver.ik(q, left_pose=(lp + d, lr), right_pose=(rp + d, rr))
        return q

    def test_layout(self) -> None:
        s = self.solver
        self.assertIs(s.robot_model, AxolModel.MOBILE)  # implied
        self.assertEqual(s.num_joints, 14 + len(BODY_JOINTS))
        self.assertEqual(s.joint_names[14:], list(BODY_JOINTS))
        self.assertEqual(s.body_indices, list(range(14, 18)))

    def test_body_holds_while_the_arms_can_reach(self) -> None:
        q = self._drive([0.2, 0.0, 0.15], 150)
        self.assertLess(np.abs(q[14:]).max(), 0.01)
        # ...and the arms did move (the lift on its limit must not freeze
        # the step).
        self.assertGreater(np.abs(q[:14] - self.q0[:14]).max(), 0.05)

    def test_base_drives_toward_an_out_of_reach_target(self) -> None:
        q = self._drive([1.0, 0.0, 0.2], 500)
        self.assertGreater(q[14], 0.2)  # base_x forward
        self.assertLess(abs(q[15]), 0.05)
        (lp, _), _ = self.solver.fk(q)
        target = self.poses[0][0] + np.array([1.0, 0.0, 0.2], dtype=np.float32)
        self.assertLess(float(np.linalg.norm(lp - target)), 0.05)


class WholeBodySwitchTest(unittest.TestCase):
    def test_panel_switch_reaches_teleop_only(self) -> None:
        from almond_axol.serve.settings import targets_for

        self.assertEqual(
            targets_for("kinematics.whole_body", "teleop"), ("kinematics.whole_body",)
        )
        self.assertEqual(targets_for("kinematics.whole_body", "collect-data"), ())

    def test_sim_carries_the_body_joints(self) -> None:
        from almond_axol.robot.sim import Sim

        sim = Sim(whole_body=True, robot_model="mobile")
        sim.set_body_joints(np.array([0.1, 0.2, 0.3, 0.04]))
        np.testing.assert_allclose(sim._build_q()[14:], [0.1, 0.2, 0.3, 0.04])
        arm_only = Sim(robot_model="mobile")
        arm_only.set_body_joints(np.array([0.1, 0.2, 0.3, 0.04]))
        self.assertEqual(arm_only._build_q().shape, (14,))

    def test_teleop_refuses_what_cannot_follow_the_body(self) -> None:
        from almond_axol.kinematics.config import KinematicsConfig
        from almond_axol.teleop.config import VRTeleopConfig
        from almond_axol.teleop.teleop import VRTeleop
        from almond_axol.vr.config import VRServerConfig

        class ArmsOnly:  # no set_body_joints: real hardware
            async def motion_control(self, left=None, right=None) -> None: ...

        whole = KinematicsConfig(whole_body=True)
        with self.assertRaisesRegex(ValueError, "sim-only"):
            VRTeleop(
                ArmsOnly(),  # type: ignore[arg-type]
                config=VRTeleopConfig(),
                kinematics_config=whole,
                vr_server_config=VRServerConfig(),
            )
        with self.assertRaisesRegex(ValueError, "relative"):
            VRTeleop(
                ArmsOnly(),  # type: ignore[arg-type]
                config=VRTeleopConfig(absolute_mode=True, hold_to_engage=False),
                kinematics_config=whole,
                vr_server_config=VRServerConfig(),
            )


if __name__ == "__main__":
    unittest.main()
