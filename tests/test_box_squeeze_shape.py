"""Box mode's squeeze shaping: the clamp force shared over the tool's contacts.

An impedance arm pressing its gripper into a box exerts its force at the
gripper *mount*; the parcel gripper touches the box at its folded blade's
face beside the wrist and the fixed blade's tip 13 cm further along. A plain
lateral run-ahead is a force through the mount, so through the face alone —
the tip lifts off as the squeeze grows. ``almond_axol.robot.squeeze`` fits
the run-ahead's contact part, replaces it by the same force split evenly
over the contacts (the moment that takes rides the wrists) and saturates it
at the force cap, consistently across poses.

Covered here: the shaping math on a synthetic arm (even split, the rest of
the run-ahead untouched, saturation by the force cap and by the spring caps'
equivalent, pass-through while not pressing, single-contact tools), the
MuJoCo mount Jacobian (against finite differences and the URDF's fixed
gripper joint), the arm applying it to its commands ahead of the spring-cap
back-off, the robot deriving each arm's normal from the measured pair, the
tool geometry's contact points, the core's decision, the live setting and
the teleop loop handing the shaping to the robot on change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.axol import Axol
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.robot.squeeze import SqueezeSpec, orient_contacts, shape_squeeze
from almond_axol.teleop.box import (
    PARCEL_TIP_FWD_M,
    PARCEL_TIP_IN_M,
    URDF_TOOL,
    parcel_tool,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LiveSettings


def _core(**overrides) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**overrides),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


def _contact_split(jac, kp, rotation, normal, contacts, dq_cmd):
    """Two-contact equilibrium: the force each contact carries when the arm
    is commanded ``dq_cmd`` past a rigid box that holds every contact on its
    plane (the other wrench components free)."""
    comp = jac @ np.diag(1.0 / kp) @ jac.T
    r = contacts @ rotation.T
    wr = np.hstack((np.tile(normal, (len(r), 1)), np.cross(r, normal)))
    d_cmd = jac @ dq_cmd

    def disp(d, k):
        return float((d[:3] + np.cross(d[3:], r[k])) @ normal)

    a = np.array(
        [[disp(comp @ wr[m], k) for m in range(len(r))] for k in range(len(r))]
    )
    b = np.array([disp(d_cmd, k) for k in range(len(r))])
    return np.linalg.solve(a, b)


class ShapeMathTest(unittest.TestCase):
    """The shaping with the left arm's real Jacobian at a box-carrying pose."""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = AxolConfig()
        gc = GravityCompensator(cfg)
        # Elbow bent, hand ahead of the shoulder at about box height.
        q = np.array([0.4, -0.15, 0.1, 1.2, 0.0, 0.3, 0.0])
        _p, cls.rot, cls.jac = gc.mount_jacobian(q, is_left=True)
        cls.kp = np.array([float(getattr(cfg.left, j.value).kp) for j in ARM_JOINTS])
        cls.n = np.array([0.0, -1.0, 0.0])  # left gripper: the box is at -y
        tool = parcel_tool(141.5)
        cls.pts = orient_contacts(tool.contacts("flush"), cls.rot, cls.n)

    def _translation_run_ahead(self, delta: float) -> np.ndarray:
        """What the IK does when the width is jogged in: the whole gripper
        translated ``delta`` into the box, joint space via the pseudo-inverse."""
        return np.linalg.pinv(self.jac) @ np.concatenate([delta * self.n, np.zeros(3)])

    def test_a_translation_squeeze_loads_the_face_and_shaping_shares_it(self) -> None:
        dq = self._translation_run_ahead(0.01)
        before = _contact_split(self.jac, self.kp, self.rot, self.n, self.pts, dq)
        # Unshaped the split is nowhere near even: one contact carries far
        # more than the total and the other is in tension (lifts off) —
        # which one depends on the pose (the face at the arm-down poses the
        # operator squeezes at, the tip here with the elbow well bent).
        self.assertGreater(abs(before[0] - before[1]), abs(before.sum()))
        res = shape_squeeze(self.kp * dq, self.kp, self.jac, self.rot, self.n, self.pts)
        after = _contact_split(
            self.jac, self.kp, self.rot, self.n, self.pts, res.tau / self.kp
        )
        self.assertAlmostEqual(after[0], after[1], places=6)
        self.assertAlmostEqual(after.sum(), before.sum(), places=6)
        self.assertGreater(res.estimate, 0.0)
        self.assertEqual(res.force, res.estimate)  # no cap in the way
        self.assertEqual(res.limit, float("inf"))

    def test_the_rest_of_the_run_ahead_is_kept(self) -> None:
        """Servo lag / payload: a run-ahead orthogonal (in the compliance
        metric) to what the contacts can push back with passes through."""
        r = self.pts @ self.rot.T
        wr = np.hstack((np.tile(self.n, (2, 1)), np.cross(r, self.n)))
        basis = self.jac.T @ wr.T
        rng = np.random.default_rng(3)
        rest = rng.normal(size=7)
        # Remove its component along the contact basis in the 1/kp metric.
        d = 1.0 / self.kp
        coef = np.linalg.solve(basis.T @ (d[:, None] * basis), basis.T @ (d * rest))
        rest -= basis @ coef
        tau = rest + basis @ np.array([5.0, -1.0])  # 4 N total, unevenly
        res = shape_squeeze(tau, self.kp, self.jac, self.rot, self.n, self.pts)
        self.assertAlmostEqual(res.estimate, 4.0, places=6)
        np.testing.assert_allclose(res.contact_forces, [5.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(
            res.tau, rest + basis @ np.array([2.0, 2.0]), atol=1e-6
        )

    def test_force_cap_saturates_and_the_shape_is_kept(self) -> None:
        dq = self._translation_run_ahead(0.05)  # far past contact
        res = shape_squeeze(
            self.kp * dq, self.kp, self.jac, self.rot, self.n, self.pts, force_cap=8.0
        )
        self.assertGreater(res.estimate, 8.0)
        self.assertEqual(res.force, 8.0)
        self.assertEqual(res.limit, 8.0)
        split = _contact_split(
            self.jac, self.kp, self.rot, self.n, self.pts, res.tau / self.kp
        )
        np.testing.assert_allclose(split, [4.0, 4.0], atol=1e-6)

    def test_spring_caps_bound_the_force_through_the_even_split(self) -> None:
        dq = self._translation_run_ahead(0.05)
        r = self.pts @ self.rot.T
        wr = np.hstack((np.tile(self.n, (2, 1)), np.cross(r, self.n)))
        even = (self.jac.T @ wr.T).mean(axis=1)
        caps = {1: 4.0, 2: 4.0}
        expected = min(4.0 / abs(even[1]), 4.0 / abs(even[2]))
        res = shape_squeeze(
            self.kp * dq, self.kp, self.jac, self.rot, self.n, self.pts, caps, 100.0
        )
        self.assertAlmostEqual(res.limit, expected, places=9)
        self.assertAlmostEqual(res.force, expected, places=9)
        # The tighter of the two wins.
        res2 = shape_squeeze(
            self.kp * dq, self.kp, self.jac, self.rot, self.n, self.pts, caps, 1.0
        )
        self.assertEqual(res2.limit, 1.0)

    def test_pulling_off_or_no_contact_passes_through(self) -> None:
        dq = self._translation_run_ahead(-0.01)  # opening the width
        tau = self.kp * dq
        res = shape_squeeze(tau, self.kp, self.jac, self.rot, self.n, self.pts, {}, 8.0)
        self.assertLess(res.estimate, 0.0)
        self.assertEqual(res.force, 0.0)
        np.testing.assert_array_equal(res.tau, tau)
        res = shape_squeeze(np.zeros(7), self.kp, self.jac, self.rot, self.n, self.pts)
        np.testing.assert_array_equal(res.tau, np.zeros(7))
        # No contacts at all: nothing to do.
        res = shape_squeeze(tau, self.kp, self.jac, self.rot, self.n, np.zeros((0, 3)))
        np.testing.assert_array_equal(res.tau, tau)

    def test_single_contact_only_saturates(self) -> None:
        pts = self.pts[:1]
        dq = self._translation_run_ahead(0.01)
        res = shape_squeeze(self.kp * dq, self.kp, self.jac, self.rot, self.n, pts)
        # One contact: the contact part is replaced by itself.
        np.testing.assert_allclose(res.tau, self.kp * dq, atol=1e-9)
        res = shape_squeeze(
            self.kp * dq, self.kp, self.jac, self.rot, self.n, pts, force_cap=1.0
        )
        self.assertEqual(res.force, 1.0)
        split = _contact_split(
            self.jac, self.kp, self.rot, self.n, pts, res.tau / self.kp
        )
        self.assertAlmostEqual(float(split[0]), 1.0, places=6)

    def test_orient_contacts_mirrors_onto_the_box_side(self) -> None:
        pts = [np.array([0.02, 0.0, -0.1])]
        # Mount +x toward the box: unchanged.
        same = orient_contacts(pts, np.eye(3), np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(same, [[0.02, 0.0, -0.1]])
        # Mount +x away from the box: x mirrored.
        flipped = orient_contacts(pts, np.eye(3), np.array([-1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(flipped, [[-0.02, 0.0, -0.1]])


class MountJacobianTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gc = GravityCompensator(AxolConfig())

    def test_mount_is_the_urdf_gripper_link(self) -> None:
        # The fixed gripper joint on each wrist: 28.5 mm across (mirrored per
        # side) and 32.3 mm along the fingers, no rotation.
        for is_left, x in ((True, -0.0285), (False, 0.0285)):
            offset, r_off = self.gc._mount_offset[is_left]
            np.testing.assert_allclose(offset, [x, 0.0, -0.032251], atol=1e-6)
            np.testing.assert_allclose(r_off, np.eye(3), atol=1e-9)

    def test_jacobian_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(1)
        for is_left in (True, False):
            q = rng.uniform(-0.8, 0.8, len(ARM_JOINTS))
            _p, rot, jac = self.gc.mount_jacobian(q, is_left=is_left)
            self.assertEqual(jac.shape, (6, 7))
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
            eps = 1e-6
            fd = np.zeros((6, 7))
            for i in range(7):
                dq = np.zeros(7)
                dq[i] = eps
                p1, r1, _ = self.gc.mount_jacobian(q + dq, is_left=is_left)
                p0, r0, _ = self.gc.mount_jacobian(q - dq, is_left=is_left)
                fd[:3, i] = (p1 - p0) / (2 * eps)
                skew = ((r1 - r0) / (2 * eps)) @ rot.T
                fd[3:, i] = (skew[2, 1], skew[0, 2], skew[1, 0])
            np.testing.assert_allclose(jac, fd, atol=1e-6)

    def test_rejects_the_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            self.gc.mount_jacobian(np.zeros(8), is_left=True)


class ArmCommandTest(unittest.TestCase):
    """The arm shapes its command, then backs off to the spring caps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.robot = Axol(AxolConfig())

    def _arm_at(self, arm, measured: np.ndarray):
        sent: list[list[tuple[float, ...]]] = []
        arm._command_sink = sent.append
        arm.resolve_joint_offsets = AsyncMock()
        arm._joint_offsets = np.zeros(8, dtype=np.float32)
        arm._last_q_commanded = None
        for i, j in enumerate(ARM_JOINTS):
            arm.motors[j]._position = float(measured[i])
        if Joint.GRIPPER in arm.motors:
            arm.motors[Joint.GRIPPER]._position = 0.0
        arm._unverified_zeros = set()
        arm._unresolved_offsets = set()
        arm.set_spring_caps(None)
        arm.set_squeeze(None)
        return sent

    def test_shaped_command_is_measured_plus_the_shaped_torque_over_kp(self) -> None:
        arm = self.robot.left
        measured = np.array([0.1, -0.3, 0.05, 0.6, 0.0, 0.1, -0.1, 0.0], np.float32)
        sent = self._arm_at(arm, measured)
        kp = arm._kp_vector
        gc = arm._gravity_comp
        _p, rot, jac = gc.mount_jacobian(measured[:7].astype(np.float64), is_left=True)
        normal = np.array([0.0, -1.0, 0.0])
        tool = parcel_tool(141.5)
        spec = SqueezeSpec(normal, tuple(tool.contacts("flush")), 8.0)
        arm.set_squeeze(spec)
        # Jog 2 cm into the box: translation run-ahead of the mount.
        dq = np.linalg.pinv(jac) @ np.concatenate([0.02 * normal, np.zeros(3)])
        target = measured.copy()
        target[:7] += dq.astype(np.float32)
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        expected = shape_squeeze(
            kp * dq,
            kp,
            jac,
            rot,
            normal,
            orient_contacts(spec.contacts, rot, normal),
            {},
            8.0,
        )
        self.assertEqual(expected.force, 8.0)  # 2 cm is well past the cap
        np.testing.assert_allclose(
            sent_arm, measured[:7] + expected.tau / kp, atol=2e-6
        )
        self.assertAlmostEqual(arm.squeeze_force, 8.0)
        # The tip is loaded as much as the face in the arm's stiffness model.
        pts = orient_contacts(spec.contacts, rot, normal)
        split = _contact_split(jac, kp, rot, normal, pts, expected.tau / kp)
        np.testing.assert_allclose(split, [4.0, 4.0], atol=1e-6)
        # Off again: the raw target goes out.
        arm.set_squeeze(None)
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)
        self.assertEqual(arm.squeeze_force, 0.0)

    def test_spring_caps_still_back_off_underneath(self) -> None:
        arm = self.robot.left
        measured = np.array([0.1, -0.3, 0.05, 0.6, 0.0, 0.1, -0.1, 0.0], np.float32)
        sent = self._arm_at(arm, measured)
        normal = np.array([0.0, -1.0, 0.0])
        arm.set_squeeze(
            SqueezeSpec(normal, tuple(URDF_TOOL.contacts("straight")), 50.0)
        )
        arm.set_spring_caps({Joint.SHOULDER_2: 1.0})
        gc = arm._gravity_comp
        _p, _rot, jac = gc.mount_jacobian(measured[:7].astype(np.float64), is_left=True)
        dq = np.linalg.pinv(jac) @ np.concatenate([0.03 * normal, np.zeros(3)])
        target = measured.copy()
        target[:7] += dq.astype(np.float32)
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        kp_s2 = arm._arm_config.shoulder_2.kp
        # The even-split limit already keeps shoulder_2 within 1 Nm; either
        # way the wire never carries more than the cap on it.
        self.assertLessEqual(abs(sent_arm[1] - measured[1]) * kp_s2, 1.0 + 1e-6)
        self.assertEqual(sent[-1][1][-1], 1.0)  # the cap still rides the tuple

    def test_no_measured_positions_sends_the_command_unshaped(self) -> None:
        arm = self.robot.left
        measured = np.array([0.1, -0.3, 0.05, 0.6, 0.0, 0.1, -0.1, 0.0], np.float32)
        sent = self._arm_at(arm, measured)
        arm.set_squeeze(
            SqueezeSpec(np.array([0.0, -1.0, 0.0]), tuple(URDF_TOOL.contacts("flush")))
        )
        for motor in arm.motors.values():
            motor._position = None
        target = measured + np.float32(0.01)
        target[7] = 0.0
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)


class RobotPairTest(unittest.TestCase):
    """``Axol.set_squeeze`` derives each arm's inward normal from the measured pair."""

    def test_normals_point_at_each_other(self) -> None:
        robot = Axol(AxolConfig())
        for arm in (robot.left, robot.right):
            arm._command_sink = lambda cmds: None
            arm.resolve_joint_offsets = AsyncMock()
            arm._joint_offsets = np.zeros(8, dtype=np.float32)
            arm._unverified_zeros = set()
            arm._unresolved_offsets = set()
            for motor in arm.motors.values():
                motor._position = 0.0
        tool = parcel_tool(141.5)
        robot.set_squeeze(tool.contacts("flush"), 8.0)
        rest = np.zeros(8, np.float32)
        asyncio.run(robot.motion_control(rest, rest))
        spec_l, spec_r = robot.left._squeeze, robot.right._squeeze
        self.assertIsNotNone(spec_l)
        self.assertIsNotNone(spec_r)
        # Left arm sits at +y: its box is toward -y, and the right's toward +y.
        self.assertLess(spec_l.normal[1], -0.9)
        self.assertGreater(spec_r.normal[1], 0.9)
        np.testing.assert_allclose(spec_l.normal, -spec_r.normal)
        self.assertEqual(spec_l.force_cap, 8.0)
        self.assertEqual(len(spec_l.contacts), 2)
        # Off clears both.
        robot.set_squeeze(None)
        self.assertIsNone(robot.left._squeeze)
        self.assertIsNone(robot.right._squeeze)
        self.assertEqual(robot.squeeze_forces, (0.0, 0.0))

    def test_needs_a_contact(self) -> None:
        robot = Axol(AxolConfig())
        with self.assertRaises(ValueError):
            robot.set_squeeze([], 8.0)

    def test_rt_robot_refreshes_the_specs_too(self) -> None:
        """The hardware path (RtAxol) commands the arms directly, bypassing
        Axol.motion_control — it must still hand each arm its spec, or the
        shaping silently never runs on the robot."""
        from almond_axol.rt.robot import RtAxol

        robot = Axol(AxolConfig())
        sent: list[tuple] = []
        for arm in (robot.left, robot.right):
            arm._command_sink = sent.append
            arm._joint_offsets = np.zeros(8, dtype=np.float32)
            arm._unverified_zeros = set()
            arm._unresolved_offsets = set()
            for motor in arm.motors.values():
                motor._position = 0.0
        rt = object.__new__(RtAxol)
        rt._robot = robot
        rt._link = SimpleNamespace(limp=None)
        rt._limp_announced = False
        robot.set_squeeze(parcel_tool(141.5).contacts("flush"), 8.0)
        rest = np.zeros(8, np.float32)
        with self.assertLogs("almond_axol.robot.axol", level="INFO") as logs:
            asyncio.run(rt.motion_control(rest, rest))
            # Press in: both commands run ahead toward the other arm (at
            # q = 0 shoulder_2 swings either mount toward -y).
            left_in, right_in = rest.copy(), rest.copy()
            s2 = ARM_JOINTS.index(Joint.SHOULDER_2)
            left_in[s2] = 0.05
            right_in[s2] = -0.05
            asyncio.run(rt.motion_control(left_in, right_in))
            self.assertGreater(max(robot.squeeze_forces), 0.5)
            asyncio.run(rt.motion_control(left_in, right_in))  # logs last tick's
        self.assertIsNotNone(robot.left._squeeze)
        self.assertIsNotNone(robot.right._squeeze)
        self.assertLess(robot.left._squeeze.normal[1], -0.9)
        self.assertEqual(len(sent), 6)
        # And the arms report what they press with in the log so a
        # deployment can be checked without a force gauge.
        self.assertTrue(
            any("box squeeze:" in line for line in logs.output), logs.output
        )


class ToolContactsTest(unittest.TestCase):
    def test_parcel_flush_is_face_and_tip(self) -> None:
        tool = parcel_tool(141.5)
        pts = tool.contacts("flush")
        self.assertEqual(len(pts), 2)
        np.testing.assert_allclose(pts[0], tool.foot(1.0))
        np.testing.assert_allclose(pts[1], [PARCEL_TIP_IN_M, 0.0, -PARCEL_TIP_FWD_M])
        self.assertAlmostEqual(PARCEL_TIP_FWD_M, 0.1385)
        # The mirrored side.
        np.testing.assert_allclose(
            tool.contacts("flush", -1.0)[1], [-PARCEL_TIP_IN_M, 0.0, -PARCEL_TIP_FWD_M]
        )

    def test_parcel_straight_is_the_blade_along_the_box(self) -> None:
        pts = parcel_tool(141.5).contacts("straight")
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pts[1], [0.0, 0.0, -PARCEL_TIP_FWD_M])

    def test_urdf_tool_touches_at_the_mount(self) -> None:
        for grasp in ("flush", "straight"):
            pts = URDF_TOOL.contacts(grasp)
            self.assertEqual(len(pts), 1)
            np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])


class CoreDecisionTest(unittest.TestCase):
    def test_default_force_is_eight_newtons(self) -> None:
        self.assertEqual(VRTeleopConfig().box_squeeze_force, 8.0)

    def test_shaping_follows_box_mode_grasp_and_tool(self) -> None:
        core = _core(box_mode=True, box_tool="parcel", box_grasp="flush")
        contacts, force = core.squeeze()
        self.assertEqual(force, 8.0)
        self.assertEqual(len(contacts), 2)
        np.testing.assert_allclose(contacts[1], parcel_tool(141.5).tip(1.0))
        core.set_live("box_grasp", "straight")
        core._apply_live_requests()
        contacts, _force = core.squeeze()
        np.testing.assert_allclose(contacts[0], [0.0, 0.0, 0.0])
        core.set_live("box_tool", "urdf")
        core._apply_live_requests()
        contacts, _force = core.squeeze()
        self.assertEqual(len(contacts), 1)
        core.set_live("box_squeeze_force", 12)
        core._apply_live_requests()
        self.assertEqual(core.squeeze()[1], 12.0)

    def test_off_outside_box_mode_during_reset_or_at_zero(self) -> None:
        self.assertIsNone(_core().squeeze())
        core = _core(box_mode=True)
        core.request_reset()
        self.assertIsNone(core.squeeze())
        self.assertIsNone(_core(box_mode=True, box_squeeze_force=0.0).squeeze())


class LiveSettingTest(unittest.TestCase):
    def test_published_only_where_the_robot_can_shape(self) -> None:
        core = _core(box_mode=True)
        hardware = LiveSettings(
            core,
            SimpleNamespace(
                set_spring_caps=lambda c: None, set_squeeze=lambda c, f: None
            ),
            lambda s: None,
        )
        sim = LiveSettings(core, object(), lambda s: None)
        self.assertIn("box_squeeze_force", {d["key"] for d in hardware.schema()})
        self.assertEqual(hardware.values()["box_squeeze_force"], 8.0)
        self.assertNotIn("box_squeeze_force", {d["key"] for d in sim.schema()})
        with self.assertRaises(ValueError):
            sim.apply("box_squeeze_force", 3.0)
        hardware.apply("box_squeeze_force", 5.0)
        core._apply_live_requests()
        self.assertEqual(core.squeeze()[1], 5.0)
        with self.assertRaises(ValueError):
            hardware.apply("box_squeeze_force", -1)


class UnitsSanityTest(unittest.TestCase):
    def test_even_split_moment_is_small_and_wrist_sized(self) -> None:
        # Half the force on a 13.4 cm lever at 8 N: ~0.5 Nm, well inside the
        # wrists' 5 Nm caps — the shaping never fights the configured limits.
        self.assertLess(0.5 * 8.0 * PARCEL_TIP_FWD_M, 1.0)
        self.assertAlmostEqual(math.degrees(parcel_tool(141.5).flush_tilt), 38.5)


if __name__ == "__main__":
    unittest.main()
