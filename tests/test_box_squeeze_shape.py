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
back-off, the robot deriving each arm's normal from the measured pair and
shaping only the pair's *common* squeeze (a carry is not a clamp), the
tool geometry's contact points, the core's decision, the live setting and
the teleop loop handing the shaping to the robot on change.
"""

from __future__ import annotations

import asyncio
import logging
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.robot.squeeze import SqueezeSpec, orient_contacts, shape_squeeze
from almond_axol.teleop.box import (
    PARCEL_FACE_HEIGHT_M,
    PARCEL_TIP_FWD_M,
    PARCEL_TIP_IN_M,
    URDF_TOOL,
    parcel_tool,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LiveSettings


def _core(**overrides) -> VRTeleopCore:
    # The shaping is opt-in (off by default while box mode's alignment is
    # settled on hardware); these tests exercise it.
    overrides.setdefault("box_squeeze_force", 8.0)
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
        self.assertGreater(np.ptp(before), abs(before.sum()))
        res = shape_squeeze(self.kp * dq, self.kp, self.jac, self.rot, self.n, self.pts)
        after = _contact_split(
            self.jac, self.kp, self.rot, self.n, self.pts, res.tau / self.kp
        )
        np.testing.assert_allclose(after, np.full(len(after), after.mean()), atol=1e-6)
        self.assertAlmostEqual(after.sum(), before.sum(), places=6)
        self.assertGreater(res.estimate, 0.0)
        self.assertEqual(res.force, res.estimate)  # no cap in the way
        self.assertEqual(res.limit, float("inf"))

    def test_the_rest_of_the_run_ahead_is_kept(self) -> None:
        """Servo lag / payload: a run-ahead orthogonal (in the compliance
        metric) to what the contacts can push back with passes through."""
        r = self.pts @ self.rot.T
        wr = np.hstack((np.tile(self.n, (len(r), 1)), np.cross(r, self.n)))
        basis = self.jac.T @ wr.T
        rng = np.random.default_rng(3)
        rest = rng.normal(size=7)
        # Remove its component along the contact basis in the 1/kp metric.
        d = 1.0 / self.kp
        coef = np.linalg.solve(basis.T @ (d[:, None] * basis), basis.T @ (d * rest))
        rest -= basis @ coef
        # 6 N total, unevenly: the root's top corner pressing, its bottom
        # corner in tension (the face rolled) and the tip lightly loaded.
        tau = rest + basis @ np.array([7.0, -3.0, 2.0])
        res = shape_squeeze(tau, self.kp, self.jac, self.rot, self.n, self.pts)
        self.assertAlmostEqual(res.estimate, 6.0, places=6)
        np.testing.assert_allclose(res.contact_forces, [7.0, -3.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(
            res.tau, rest + basis @ np.array([2.0, 2.0, 2.0]), atol=1e-6
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
        np.testing.assert_allclose(split, [8.0 / 3.0] * 3, atol=1e-6)

    def test_spring_caps_bound_the_force_through_the_even_split(self) -> None:
        dq = self._translation_run_ahead(0.05)
        r = self.pts @ self.rot.T
        wr = np.hstack((np.tile(self.n, (len(r), 1)), np.cross(r, self.n)))
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

    def test_shared_caps_the_clamp_and_passes_the_carry(self) -> None:
        """``shared`` (the pair's mean inward force) is the clamp: capped.
        What this arm's estimate differs from it by is the carry: through in
        full, either sign, so the arm moving away from its box leads."""
        r = self.pts @ self.rot.T
        wr = np.hstack((np.tile(self.n, (len(r), 1)), np.cross(r, self.n)))
        basis = self.jac.T @ wr.T
        even = basis.mean(axis=1)
        fit = np.array([6.0, 4.0, 2.0])
        tau = basis @ fit  # 12 N, all through the fit
        # Pair squeeze 3 N, this arm 12: it is the trailing arm of a carry,
        # pushing 9 N more than the clamp. Clamp under the cap: 3 + 9 = 12,
        # now evenly split.
        res = shape_squeeze(
            tau, self.kp, self.jac, self.rot, self.n, self.pts, {}, 8.0, shared=3.0
        )
        self.assertAlmostEqual(res.estimate, 12.0, places=6)
        self.assertEqual(res.force, 3.0)
        np.testing.assert_allclose(res.tau, 12.0 * even, atol=1e-6)
        # Pair squeeze 11 N: the clamp is capped to 8, the 1 N carry kept.
        res = shape_squeeze(
            tau, self.kp, self.jac, self.rot, self.n, self.pts, {}, 8.0, shared=11.0
        )
        self.assertEqual(res.force, 8.0)
        np.testing.assert_allclose(res.tau, 9.0 * even, atol=1e-6)
        # Pair squeeze 50 N while this arm estimates 12: it is the *leading*
        # arm of a carry (38 N less than the clamp). Capped to 8 the clamp
        # leaves it pulling away with 30 N — and the other arm, 88 → 46, so
        # the pair's net push (2 × 8 + their difference) is what a rigid
        # pair's servo lag would give.
        res = shape_squeeze(
            tau, self.kp, self.jac, self.rot, self.n, self.pts, {}, 8.0, shared=50.0
        )
        self.assertEqual(res.force, 8.0)
        np.testing.assert_allclose(res.tau, -30.0 * even, atol=1e-6)
        # Nothing in common (a pure carry, or the pair opening): untouched.
        for shared in (0.0, -2.0):
            res = shape_squeeze(
                tau,
                self.kp,
                self.jac,
                self.rot,
                self.n,
                self.pts,
                {},
                8.0,
                shared=shared,
            )
            self.assertEqual(res.force, 0.0)
            np.testing.assert_array_equal(res.tau, tau)
        # A pair squeeze inside the ramp fades the redistribution in.
        res = shape_squeeze(
            tau, self.kp, self.jac, self.rot, self.n, self.pts, {}, 8.0, shared=1.0
        )
        self.assertEqual(res.force, 1.0)
        np.testing.assert_allclose(res.tau, 0.5 * tau + 6.0 * even, atol=1e-6)

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
        cls.robot = AxolHardware(AxolConfig())

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
        # Without the pair's word on how much of this is a squeeze, the
        # command goes out as it is.
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)
        self.assertEqual(arm.squeeze_force, 0.0)
        # The pair's step (AxolHardware.refresh_squeeze): estimate, then
        # share — here the other arm presses just as hard.
        estimate = arm.squeeze_estimate(target)
        self.assertGreater(estimate, 8.0)
        arm.set_squeeze_shared(estimate)
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
        np.testing.assert_allclose(split, [8.0 / 3.0] * 3, atol=1e-6)
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
        arm.set_squeeze_shared(arm.squeeze_estimate(target))
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
        self.assertIsNone(arm.squeeze_estimate(target))
        arm.set_squeeze_shared(5.0)
        asyncio.run(arm.motion_control(target))
        sent_arm = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)


class RobotPairTest(unittest.TestCase):
    """``Axol.set_squeeze`` derives each arm's inward normal from the measured pair."""

    def test_normals_point_at_each_other(self) -> None:
        robot = AxolHardware(AxolConfig())
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
        self.assertEqual(len(spec_l.contacts), 3)
        # Off clears both.
        robot.set_squeeze(None)
        self.assertIsNone(robot.left._squeeze)
        self.assertIsNone(robot.right._squeeze)
        self.assertEqual(robot.squeeze_forces, (0.0, 0.0))

    def test_needs_a_contact(self) -> None:
        robot = AxolHardware(AxolConfig())
        with self.assertRaises(ValueError):
            robot.set_squeeze([], 8.0)

    def _pair(self):
        robot = AxolHardware(AxolConfig())
        sent: dict[bool, list] = {True: [], False: []}
        for arm in (robot.left, robot.right):
            arm._command_sink = sent[arm._is_left].append
            arm.resolve_joint_offsets = AsyncMock()
            arm._joint_offsets = np.zeros(8, dtype=np.float32)
            arm._unverified_zeros = set()
            arm._unresolved_offsets = set()
            for motor in arm.motors.values():
                motor._position = 0.0
        robot.set_squeeze(parcel_tool(141.5).contacts("straight"), 8.0)
        rest = np.zeros(8, np.float32)
        asyncio.run(robot.motion_control(rest, rest))  # specs from the pair
        gc = robot._gravity_comp
        q0 = np.zeros(7)
        jac_l = gc.mount_jacobian(q0, is_left=True)[2]
        jac_r = gc.mount_jacobian(q0, is_left=False)[2]

        def translate(jac, delta: np.ndarray) -> np.ndarray:
            target = rest.copy()
            target[:7] += np.linalg.pinv(jac) @ np.concatenate([delta, np.zeros(3)])
            return target

        return robot, sent, jac_l, jac_r, translate

    @staticmethod
    def _settle(robot) -> None:
        """Forget the shaping low-pass's history: the next command applies
        its correction in full, as after a pause (see ``_smoothed_squeeze``)."""
        for arm in (robot.left, robot.right):
            arm._squeeze_corr = None
            arm._squeeze_corr_t = None

    def test_a_carry_is_not_a_squeeze(self) -> None:
        """The pair moving sideways loads one arm's normal exactly like a
        clamp and unloads the other's; shaping or capping it would hold the
        leading arm back and skew the grippers — the lag the operator saw.
        Only the arms' *common* inward run-ahead is shaped."""
        robot, sent, jac_l, jac_r, translate = self._pair()
        normal_l = robot.left._squeeze.normal  # toward -y
        # Both grippers 2 cm along the left's inward normal: a carry, well
        # past what the 8 N cap would allow if it were read as a squeeze.
        left = translate(jac_l, 0.02 * normal_l)
        right = translate(jac_r, 0.02 * normal_l)
        asyncio.run(robot.motion_control(left, right))
        est_l = robot.left.squeeze_estimate(left)
        est_r = robot.right.squeeze_estimate(right)
        self.assertGreater(est_l, 8.0)
        self.assertAlmostEqual(est_r, -est_l, places=6)  # equal and opposite
        self.assertLess(max(robot.squeeze_forces), 1e-6)
        for arm, target in ((True, left), (False, right)):
            sent_arm = np.array([c[0] for c in sent[arm][-1][:7]])
            np.testing.assert_allclose(sent_arm, target[:7], atol=1e-6)

    def test_a_clamp_is_shaped_on_both_and_a_carried_clamp_keeps_moving(self) -> None:
        robot, sent, jac_l, jac_r, translate = self._pair()
        normal_l = robot.left._squeeze.normal
        # Width jogged 2 cm in on each side: both press, both capped.
        left = translate(jac_l, 0.02 * normal_l)
        right = translate(jac_r, -0.02 * normal_l)
        self._settle(robot)  # (the tests' commands are microseconds apart)
        asyncio.run(robot.motion_control(left, right))
        self.assertEqual(robot.squeeze_forces, (8.0, 8.0))
        run_l = np.array([c[0] for c in sent[True][-1][:7]])
        run_r = np.array([c[0] for c in sent[False][-1][:7]])
        # Now the same clamp carried 1 cm toward the left's side: the left
        # (trailing) arm's inward run-ahead grows, the right's shrinks by the
        # same amount, so the pair's squeeze — their mean — is unchanged and
        # still capped, and the carry passes on *both* arms: the left pushes
        # by it, the right pulls away by it, and the pair translates as one
        # under position control with the clamp force-controlled between.
        left_c = translate(jac_l, 0.03 * normal_l)
        right_c = translate(jac_r, -0.01 * normal_l)
        self._settle(robot)
        asyncio.run(robot.motion_control(left_c, right_c))
        est_l = robot.left.squeeze_estimate(left_c)
        est_r = robot.right.squeeze_estimate(right_c)
        self.assertGreater(est_l, est_r)
        self.assertGreater(est_r, 0.0)
        self.assertEqual(robot.squeeze_forces, (8.0, 8.0))
        moved_l = np.array([c[0] for c in sent[True][-1][:7]])
        moved_r = np.array([c[0] for c in sent[False][-1][:7]])
        along_l = float((jac_l @ (moved_l - run_l))[:3] @ normal_l)
        along_r = float((jac_r @ (moved_r - run_r))[:3] @ normal_l)
        # Both mounts' commands moved toward the left's box side (the
        # direction of the carry) relative to the plain clamp — the right
        # leading, not held at the cap against its measured pose.
        self.assertGreater(along_l, 0.004)
        self.assertGreater(along_r, 0.004)
        self.assertAlmostEqual(along_l, along_r, delta=0.004)


class SmoothingTest(unittest.TestCase):
    """The shaped command follows the measured pose along the contact
    directions — so measured jitter went straight back out and the clamp
    buzzed. The correction is low-passed (``SQUEEZE_SMOOTH_S``)."""

    def _arm(self):
        robot = AxolHardware(AxolConfig())
        arm = robot.left
        sent: list = []
        arm._command_sink = sent.append
        arm.resolve_joint_offsets = AsyncMock()
        arm._joint_offsets = np.zeros(8, dtype=np.float32)
        arm._unverified_zeros = set()
        arm._unresolved_offsets = set()
        measured = np.array([0.1, -0.3, 0.05, 0.6, 0.0, 0.1, -0.1, 0.0], np.float32)
        for i, j in enumerate(ARM_JOINTS):
            arm.motors[j]._position = float(measured[i])
        arm.motors[Joint.GRIPPER]._position = 0.0
        normal = np.array([0.0, -1.0, 0.0])
        arm.set_squeeze(
            SqueezeSpec(normal, tuple(parcel_tool(141.5).contacts("straight")), 8.0)
        )
        jac = arm._gravity_comp.mount_jacobian(
            measured[:7].astype(np.float64), is_left=True
        )[2]
        target = measured.copy()
        target[:7] += (
            np.linalg.pinv(jac) @ np.concatenate([0.02 * normal, np.zeros(3)])
        ).astype(np.float32)
        return arm, sent, measured, target

    def _send(self, arm, target, sent) -> np.ndarray:
        arm.set_squeeze_shared(arm.squeeze_estimate(target))
        asyncio.run(arm.motion_control(target))
        return np.array([c[0] for c in sent[-1][:7]])

    def test_measured_jitter_is_filtered_out_of_the_command(self) -> None:
        arm, sent, measured, target = self._arm()
        first = self._send(arm, target, sent)  # no history: the full correction
        self.assertGreater(np.abs(first - target[:7]).max(), 1e-4)
        # The arm twitches by 0.5° on shoulder_2 between two commands 1/120 s
        # apart: unfiltered, the whole twitch would ride the command out.
        s2 = ARM_JOINTS.index(Joint.SHOULDER_2)
        arm.motors[Joint.SHOULDER_2]._position = float(measured[s2] + math.radians(0.5))
        arm._squeeze_corr_t -= 1.0 / 120.0
        with patch(
            "almond_axol.robot.axol.time.monotonic",
            return_value=arm._squeeze_corr_t + 1.0 / 120.0,
        ):
            second = self._send(arm, target, sent)
        arm.motors[Joint.SHOULDER_2]._position = float(measured[s2])
        # The correction moved only a fraction (1 - e^(-dt/tau) ≈ 15 %) of
        # the way toward the twitched one.
        moved = np.abs(second - first).max()
        self.assertLess(moved, 0.3 * math.radians(0.5))
        self.assertGreater(moved, 0.05 * math.radians(0.5))

    def test_a_released_clamp_eases_out(self) -> None:
        arm, sent, measured, target = self._arm()
        shaped = self._send(arm, target, sent)
        # The pair now says there is nothing in common: the correction
        # decays instead of stepping off.
        arm.set_squeeze_shared(0.0)
        asyncio.run(arm.motion_control(target))
        eased = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(eased, shaped, atol=1e-3)
        self.assertEqual(arm.squeeze_force, 0.0)
        # After a long gap the history is stale and nothing is applied.
        arm._squeeze_corr_t -= 1.0
        asyncio.run(arm.motion_control(target))
        plain = np.array([c[0] for c in sent[-1][:7]])
        np.testing.assert_allclose(plain, target[:7], atol=1e-6)
        self.assertIsNone(arm._squeeze_corr)


class ToolContactsTest(unittest.TestCase):
    def test_parcel_flush_is_face_corners_and_tip(self) -> None:
        tool = parcel_tool(141.5)
        pts = tool.contacts("flush")
        self.assertEqual(len(pts), 3)
        half = np.array([0.0, PARCEL_FACE_HEIGHT_M / 2, 0.0])
        np.testing.assert_allclose(pts[0], tool.foot(1.0) + half, atol=1e-7)
        np.testing.assert_allclose(pts[1], tool.foot(1.0) - half, atol=1e-7)
        np.testing.assert_allclose(pts[2], [PARCEL_TIP_IN_M, 0.0, -PARCEL_TIP_FWD_M])
        self.assertAlmostEqual(PARCEL_TIP_FWD_M, 0.1385)
        self.assertAlmostEqual(PARCEL_FACE_HEIGHT_M, 0.060)
        # The mirrored side.
        np.testing.assert_allclose(
            tool.contacts("flush", -1.0)[2], [-PARCEL_TIP_IN_M, 0.0, -PARCEL_TIP_FWD_M]
        )

    def test_parcel_straight_is_the_blade_along_the_box(self) -> None:
        pts = parcel_tool(141.5).contacts("straight")
        # The root's top and bottom corners (the blade is 60 mm tall there,
        # centred on the mount) and the tip.
        np.testing.assert_allclose(pts[0], [0.0, 0.03, 0.0])
        np.testing.assert_allclose(pts[1], [0.0, -0.03, 0.0])
        np.testing.assert_allclose(pts[2], [0.0, 0.0, -PARCEL_TIP_FWD_M])

    def test_the_face_corners_take_the_roll_out_of_the_clamp(self) -> None:
        """With the root as a single centre-line point the shaping said
        nothing about the roll about the fingers, and the face pressed along
        its top edge (thumb and index pinching, the pinky off the box).
        With the corners it does: the even split has no roll moment, so a
        rolled run-ahead is straightened."""
        cfg = AxolConfig()
        gc = GravityCompensator(cfg)
        q = np.array([0.4, -0.15, 0.1, 1.2, 0.0, 0.3, 0.0])
        _p, rot, jac = gc.mount_jacobian(q, is_left=True)
        kp = np.array([float(getattr(cfg.left, j.value).kp) for j in ARM_JOINTS])
        n = np.array([0.0, -1.0, 0.0])
        pts = orient_contacts(parcel_tool(141.5).contacts("straight"), rot, n)
        # A squeeze plus a roll about the fingers (the mount's Z, in base
        # frame the third column of the rotation): the top corner digs in.
        fingers = rot[:, 2]
        dq = np.linalg.pinv(jac) @ np.concatenate([0.01 * n, 0.05 * fingers])
        before = _contact_split(jac, kp, rot, n, pts, dq)
        self.assertGreater(abs(before[0] - before[1]), 0.5 * abs(before.sum()))
        res = shape_squeeze(kp * dq, kp, jac, rot, n, pts)
        after = _contact_split(jac, kp, rot, n, pts, res.tau / kp)
        self.assertAlmostEqual(after[0], after[1], places=6)  # no roll
        self.assertAlmostEqual(after[1], after[2], places=6)  # no pitch

    def test_urdf_tool_touches_at_the_mount(self) -> None:
        for grasp in ("flush", "straight"):
            pts = URDF_TOOL.contacts(grasp)
            self.assertEqual(len(pts), 1)
            np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])


class CoreDecisionTest(unittest.TestCase):
    def test_default_is_off(self) -> None:
        # Off until the pair's alignment and a flat grasp are settled on
        # hardware without it: box mode then sends the IK's commands as
        # they are, and the arms press with their plain springs.
        self.assertEqual(VRTeleopConfig().box_squeeze_force, 0.0)
        core = VRTeleopCore(
            VRTeleopConfig(box_mode=True),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )
        self.assertIsNone(core.squeeze())
        self.assertIsNone(core.spring_caps())

    def test_shaping_follows_box_mode_grasp_and_tool(self) -> None:
        core = _core(box_mode=True, box_tool="parcel", box_grasp="flush")
        contacts, force = core.squeeze()
        self.assertEqual(force, 8.0)
        self.assertEqual(len(contacts), 3)
        np.testing.assert_allclose(contacts[2], parcel_tool(141.5).tip(1.0))
        core.set_live("box_grasp", "straight")
        core._apply_live_requests()
        contacts, _force = core.squeeze()
        np.testing.assert_allclose(contacts[0], [0.0, 0.03, 0.0])
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
