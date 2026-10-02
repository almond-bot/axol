"""Box mode's squeeze lean: the clamp force put through the tool's contacts.

An impedance arm pressing its gripper into a box exerts, at the gripper
mount, the wrench its springs make of the run-ahead. A plain lateral
run-ahead (the width jogged in) is a force at the mount plus the moment it
takes to hold the mount's orientation against the arm's stiffness coupling
— and the parcel gripper touches the box at its blade's root beside the
wrist and its tip 13 cm on, so that moment loads them unevenly: the pinch.
``almond_axol.teleop.box.squeeze_lean`` gives the target offset (mostly an
inward yaw) that turns the run-ahead into a pure force through the
contacts' centroid, from the arm's Jacobian and stiffness; the IK worker
adds it to the gripper targets in proportion to the measured clamp depth.

Covered here: the lean math with the left arm's real Jacobian (the wrench
it produces goes through the centroid so the contacts share the force
evenly, where a plain jog does not; its size and direction; the force cap
as a pullback), the MuJoCo mount Jacobian (against finite differences and
the URDF's fixed gripper joint), the worker applying the lean to its
targets from a measured depth (and not to a carry, a blend or the sim),
the tool geometry's contact points, the measured-arm readout, the live
setting and the core forwarding the measurement to the worker.
"""

from __future__ import annotations

import logging
import math
import types
import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.teleop.box import (
    PARCEL_FACE_HEIGHT_M,
    PARCEL_TIP_FWD_M,
    PARCEL_TIP_IN_M,
    URDF_TOOL,
    BoxState,
    ideal_gripper_poses,
    parcel_tool,
    rodrigues,
    side_clamp_rotation,
    squeeze_lean,
    toe_out,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore, measured_arms
from almond_axol.teleop.live import LiveSettings
from almond_axol.teleop.worker import _LEAN_TAU_S, IKWorker

_CFG = AxolConfig()
_GC = GravityCompensator(_CFG)
_KP_L = np.array([float(getattr(_CFG.left, j.value).kp) for j in ARM_JOINTS])
_KP_R = np.array([float(getattr(_CFG.right, j.value).kp) for j in ARM_JOINTS])
# Elbow bent, hand ahead of the shoulder at about box height.
_Q_BOX_L = np.array([0.4, -0.15, 0.1, 1.2, 0.0, 0.3, 0.0])
_Q_BOX_R = np.array([0.4, 0.15, -0.1, 1.2, 0.0, -0.3, 0.0])


def _wrench(jac, kp, delta):
    """The wrench (force, moment about the mount) a run-ahead ``delta`` produces."""
    compliance = (jac / kp) @ jac.T
    return np.linalg.solve(compliance, delta)


def _contact_split(wrench, rot, normal, pts):
    """Per-contact inward forces that reproduce ``wrench``, and the residual."""
    r = np.asarray(pts) @ rot.T
    basis = np.hstack((np.tile(normal, (len(r), 1)), np.cross(r, normal))).T
    forces, *_ = np.linalg.lstsq(basis, wrench, rcond=None)
    return forces, wrench - basis @ forces


class LeanMathTest(unittest.TestCase):
    """The lean with the left arm's real Jacobian at a box-carrying pose."""

    @classmethod
    def setUpClass(cls) -> None:
        _p, cls.rot, cls.jac = _GC.mount_jacobian(_Q_BOX_L, is_left=True)
        cls.n = np.array([0.0, -1.0, 0.0])  # left gripper: the box is at -y
        cls.pts = np.asarray(parcel_tool(141.5).contacts("straight"))

    def _delta(self, lean, depth):
        return np.concatenate([depth * self.n + lean.translation, lean.rotation])

    def test_a_plain_jog_loads_the_contacts_unevenly(self) -> None:
        w = _wrench(self.jac, _KP_L, np.concatenate([0.01 * self.n, np.zeros(3)]))
        r_c = self.pts.mean(axis=0) @ self.rot.T
        about_centroid = w[3:] - np.cross(r_c, w[:3])
        # The moment it takes to hold the mount's orientation is of the
        # order of a newton-metre per centimetre — carried by the contacts
        # loading unevenly, since a force through their centroid has none.
        self.assertGreater(np.linalg.norm(about_centroid), 0.5)
        forces, _res = _contact_split(w, self.rot, self.n, self.pts)
        self.assertGreater(forces.max() - forces.min(), abs(forces.sum()))

    def test_the_lean_puts_the_force_through_the_centroid(self) -> None:
        depth = 0.01
        lean = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, depth)
        w = _wrench(self.jac, _KP_L, self._delta(lean, depth))
        # A pure inward force ...
        np.testing.assert_allclose(w[:3], lean.force * self.n, atol=1e-6)
        # ... whose moment about the mount is exactly the centroid's lever.
        r_c = self.pts.mean(axis=0) @ self.rot.T
        np.testing.assert_allclose(w[3:], np.cross(r_c, lean.force * self.n), atol=1e-6)
        # So the three contacts share it evenly with nothing left over.
        forces, res = _contact_split(w, self.rot, self.n, self.pts)
        np.testing.assert_allclose(forces, np.full(3, lean.force / 3), atol=1e-6)
        self.assertLess(np.linalg.norm(res), 1e-6)

    def test_size_and_direction(self) -> None:
        depth = 0.01
        lean = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, depth)
        # ~6 N per centimetre along the lean; a plain jog is stiffer.
        self.assertGreater(lean.force, 4.0)
        self.assertLess(lean.force, 9.0)
        self.assertAlmostEqual(lean.force, lean.stiffness * depth)
        self.assertEqual(lean.depth, depth)
        self.assertEqual(lean.pullback, 0.0)
        # An inward yaw of about a degree per centimetre, leaning the
        # contacts' centroid (mostly the far tip) *into* the box.
        angle = math.degrees(np.linalg.norm(lean.rotation))
        self.assertGreater(angle, 0.5)
        self.assertLess(angle, 3.0)
        r_c = self.pts.mean(axis=0) @ self.rot.T
        self.assertGreater(float(np.cross(lean.rotation, r_c) @ self.n), 0.0)
        # At this pose the fingers point roughly +x and the box is at -y, so
        # that is a turn about -z (a negative angle about up).
        self.assertLess(math.degrees(lean.rotation[2]), -0.5)
        self.assertLess(abs(math.degrees(lean.rotation[0])), 1.0)  # a touch of roll
        # And a millimetre or two of translation to go with it, none of it
        # along the normal (the depth is the operator's).
        self.assertLess(np.linalg.norm(lean.translation), 0.004)
        self.assertAlmostEqual(float(lean.translation @ self.n), 0.0, places=9)

    def test_the_right_arm_leans_the_other_way(self) -> None:
        _p, rot, jac = _GC.mount_jacobian(_Q_BOX_R, is_left=False)
        n = np.array([0.0, 1.0, 0.0])  # right gripper: the box is at +y
        pts = np.asarray(parcel_tool(141.5).contacts("straight", -1.0))
        lean = squeeze_lean(jac, _KP_R, rot, n, pts, 0.01)
        r_c = pts.mean(axis=0) @ rot.T
        self.assertGreater(float(np.cross(lean.rotation, r_c) @ n), 0.0)
        w = _wrench(
            jac, _KP_R, np.concatenate([0.01 * n + lean.translation, lean.rotation])
        )
        forces, _res = _contact_split(w, rot, n, pts)
        np.testing.assert_allclose(forces, np.full(3, lean.force / 3), atol=1e-6)

    def test_proportional_to_depth(self) -> None:
        a = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, 0.01)
        b = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, 0.02)
        np.testing.assert_allclose(b.rotation, 2 * a.rotation)
        np.testing.assert_allclose(b.translation, 2 * a.translation)
        self.assertAlmostEqual(b.force, 2 * a.force)

    def test_force_cap_pulls_the_depth_back(self) -> None:
        free = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, 0.03)
        self.assertGreater(free.force, 8.0)
        capped = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, 0.03, 8.0)
        self.assertEqual(capped.force, 8.0)
        self.assertAlmostEqual(capped.depth, 8.0 / capped.stiffness)
        self.assertAlmostEqual(capped.pullback, capped.depth - 0.03)
        self.assertLess(capped.pullback, 0.0)
        # The lean itself is the cap's, not the operator's depth's.
        at_cap = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, capped.depth)
        np.testing.assert_allclose(capped.rotation, at_cap.rotation, atol=1e-9)
        # A cap above the force changes nothing.
        loose = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, 0.01, 40.0)
        self.assertEqual(loose.pullback, 0.0)

    def test_nothing_without_a_depth_or_a_model(self) -> None:
        for depth in (0.0, -0.01):
            lean = squeeze_lean(self.jac, _KP_L, self.rot, self.n, self.pts, depth)
            self.assertEqual(lean.force, 0.0)
            np.testing.assert_array_equal(lean.rotation, 0.0)
        self.assertEqual(
            squeeze_lean(
                self.jac, _KP_L, self.rot, self.n, np.zeros((0, 3)), 0.01
            ).force,
            0.0,
        )
        self.assertEqual(
            squeeze_lean(self.jac, np.zeros(7), self.rot, self.n, self.pts, 0.01).force,
            0.0,
        )

    def test_single_contact_at_the_mount_is_a_pure_force(self) -> None:
        pts = np.asarray(URDF_TOOL.contacts("straight"))
        lean = squeeze_lean(self.jac, _KP_L, self.rot, self.n, pts, 0.01)
        w = _wrench(self.jac, _KP_L, self._delta(lean, 0.01))
        np.testing.assert_allclose(w[:3], lean.force * self.n, atol=1e-6)
        np.testing.assert_allclose(w[3:], 0.0, atol=1e-6)


class MountJacobianTest(unittest.TestCase):
    def test_mount_is_the_urdf_gripper_link(self) -> None:
        # The fixed gripper joint on each wrist: 28.5 mm across (mirrored per
        # side) and 32.3 mm along the fingers, no rotation.
        for is_left, x in ((True, -0.0285), (False, 0.0285)):
            offset, r_off = _GC._mount_offset[is_left]
            np.testing.assert_allclose(offset, [x, 0.0, -0.032251], atol=1e-6)
            np.testing.assert_allclose(r_off, np.eye(3), atol=1e-9)

    def test_jacobian_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(1)
        for is_left in (True, False):
            q = rng.uniform(-0.8, 0.8, len(ARM_JOINTS))
            _p, rot, jac = _GC.mount_jacobian(q, is_left=is_left)
            self.assertEqual(jac.shape, (6, 7))
            np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
            eps = 1e-6
            fd = np.zeros((6, 7))
            for i in range(7):
                dq = np.zeros(7)
                dq[i] = eps
                p1, r1, _ = _GC.mount_jacobian(q + dq, is_left=is_left)
                p0, r0, _ = _GC.mount_jacobian(q - dq, is_left=is_left)
                fd[:3, i] = (p1 - p0) / (2 * eps)
                skew = ((r1 - r0) / (2 * eps)) @ rot.T
                fd[3:, i] = (skew[2, 1], skew[0, 2], skew[1, 0])
            np.testing.assert_allclose(jac, fd, atol=1e-6)

    def test_rejects_the_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            _GC.mount_jacobian(np.zeros(8), is_left=True)


class _ModelSolver:
    """Solver stub whose FK is the MuJoCo model's mount frame (16-joint layout)."""

    left_indices = list(range(0, 7))
    right_indices = list(range(8, 15))

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        pl, rl, _ = _GC.mount_jacobian(q[self.left_indices], is_left=True)
        pr, rr, _ = _GC.mount_jacobian(q[self.right_indices], is_left=False)
        return (pl.astype(np.float32), rl.astype(np.float32)), (
            pr.astype(np.float32),
            rr.astype(np.float32),
        )


def _q_full() -> np.ndarray:
    q = np.zeros(16, np.float32)
    q[0:7] = _Q_BOX_L
    q[8:15] = _Q_BOX_R
    return q


class WorkerLeanTest(unittest.TestCase):
    """The worker leaning its box targets from the measured clamp depth."""

    def _worker(self, **cfg) -> IKWorker:
        w = object.__new__(IKWorker)
        w._config = types.SimpleNamespace(
            **{
                "box_tool": "parcel",
                "box_tool_open_deg": 141.5,
                "box_grasp": "straight",
                "box_squeeze_lean": 1.0,
                "box_squeeze_force": 0.0,
                **cfg,
            }
        )
        w._solver = _ModelSolver()
        w._lean_model = _GC
        w._measured, w._measured_t = None, 0.0
        w._lean_depth, w._lean_t, w._lean_force = 0.0, None, 0.0
        w._lean_trim = {"left": 0.0, "right": 0.0}
        w._trim_prev = None
        w._trim_moving_t = -math.inf
        w._clamp_log_t = -math.inf
        return w

    def _box(self) -> BoxState:
        # A level pair, lateral axis +y (the left gripper at +y): the left
        # gripper's inward normal is -y, the right's +y. Faces +1 / -1.
        return BoxState(
            center=np.zeros(3, np.float32),
            rot=np.eye(3, dtype=np.float32),
            width=0.3,
            face={"left": 1.0, "right": -1.0},
            tilt=0.0,
            align_start={},
            align_t0=0.0,
            align_duration=0.0,
            tool=parcel_tool(141.5),
        )

    def _targets(self, q, depth_l: float, depth_r: float):
        """Raw targets: the arms' current mounts pushed into the box."""
        (pl, rl), (pr, rr) = _ModelSolver().fk(q)
        return {
            "left": (pl + np.array([0, -depth_l, 0], np.float32), rl),
            "right": (pr + np.array([0, depth_r, 0], np.float32), rr),
        }

    def _settle(self, w, box, targets, q, seconds: float = 2.0, cap=None):
        """Run the lean at 120 Hz until the depth filter has settled."""
        out = targets
        n = int(seconds * 120)
        for i in range(n):
            w._measured_t = 100.0 + i / 120.0  # a fresh report every frame
            out = w._squeeze_lean(box, targets, q, 100.0 + i / 120.0)
        return out

    def test_no_measurement_no_model_or_blending_leaves_targets_alone(self) -> None:
        q = _q_full()
        box = self._box()
        targets = self._targets(q, 0.01, 0.01)
        w = self._worker()
        self.assertIs(w._squeeze_lean(box, targets, q, 100.0), targets)
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        w._measured_t = 100.0
        w._lean_model = None
        self.assertIs(w._squeeze_lean(box, targets, q, 100.0), targets)
        w._lean_model = _GC
        box.align_duration = 1.0  # still blending in
        self.assertIs(w._squeeze_lean(box, targets, q, 100.0), targets)
        self.assertEqual(w.squeeze_force, 0.0)
        # Both knobs off: nothing to do either.
        off = self._worker(box_squeeze_lean=0.0)
        off.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        off._measured_t = 100.0
        self.assertIs(off._squeeze_lean(self._box(), targets, q, 100.0), targets)

    def test_no_depth_no_lean(self) -> None:
        q = _q_full()
        w = self._worker()
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        # Targets exactly where the arms are: not pressing.
        targets = self._targets(q, 0.0, 0.0)
        out = self._settle(w, self._box(), targets, q, seconds=0.5)
        np.testing.assert_allclose(out["left"][0], targets["left"][0], atol=1e-7)
        np.testing.assert_allclose(out["left"][1], targets["left"][1], atol=1e-7)
        self.assertLess(w.squeeze_force, 1e-6)
        # Pulled *off* the box: not pressing either.
        out = self._settle(w, self._box(), self._targets(q, -0.01, -0.01), q, 0.5)
        self.assertLess(w.squeeze_force, 1e-6)

    def test_a_clamp_leans_both_grippers_in(self) -> None:
        q = _q_full()
        w = self._worker()
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        box = self._box()
        targets = self._targets(q, 0.01, 0.01)  # 1 cm past the box on each side
        out = self._settle(w, box, targets, q)
        # ~6 N a side.
        self.assertGreater(w.squeeze_force, 4.0)
        self.assertLess(w.squeeze_force, 9.0)
        tool = parcel_tool(141.5)
        for side, normal, face in (
            ("left", [0.0, -1.0, 0.0], 1.0),
            ("right", [0.0, 1.0, 0.0], -1.0),
        ):
            pos, rot = out[side]
            pos0, rot0 = targets[side]
            rel = rot @ rot0.T  # the added rotation, world frame
            angle = math.degrees(math.acos(min(1.0, (np.trace(rel) - 1) / 2)))
            self.assertGreater(angle, 0.5)
            self.assertLess(angle, 3.0)
            # An inward lean: the rotation carries each gripper's contact
            # centroid (mostly its far tip) toward its box side.
            axis = np.array(
                [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
            )
            centroid = np.asarray(tool.contacts("straight", face)).mean(axis=0)
            r_c = rot0.astype(np.float64) @ centroid
            self.assertGreater(float(np.cross(axis, r_c) @ np.asarray(normal)), 0.0)
            # A millimetre or two of translation, none along the normal
            # (no cap: the depth stays the operator's).
            shift = pos.astype(np.float64) - pos0.astype(np.float64)
            self.assertLess(np.linalg.norm(shift), 0.004)
            self.assertAlmostEqual(float(shift[1]), 0.0, places=6)

    def test_the_lean_scale(self) -> None:
        q = _q_full()
        box = self._box()
        targets = self._targets(q, 0.01, 0.01)
        angles = []
        for scale in (1.0, 2.0):
            w = self._worker(box_squeeze_lean=scale)
            w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
            out = self._settle(w, box, targets, q)
            rel = out["left"][1] @ targets["left"][1].T
            angles.append(math.acos(min(1.0, (np.trace(rel) - 1) / 2)))
        self.assertAlmostEqual(angles[1], 2 * angles[0], places=4)

    def test_a_carry_is_not_a_clamp(self) -> None:
        # The pair moving toward -y: the left arm lags *into* its box side,
        # the right arm lags away from its, by the same amount. The mean
        # depth is zero, so nothing leans and nothing pulls back.
        q = _q_full()
        w = self._worker(box_squeeze_force=8.0)
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        targets = self._targets(q, 0.02, -0.02)
        out = self._settle(w, self._box(), targets, q)
        self.assertEqual(w.squeeze_force, 0.0)
        np.testing.assert_allclose(out["left"][0], targets["left"][0], atol=1e-7)
        np.testing.assert_allclose(out["right"][1], targets["right"][1], atol=1e-7)
        # A carried clamp: 1 cm of clamp plus the same lag — the mean is the
        # clamp, and both arms lean for exactly that.
        w2 = self._worker()
        w2.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        self._settle(w2, self._box(), self._targets(q, 0.03, -0.01), q)
        w3 = self._worker()
        w3.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        self._settle(w3, self._box(), self._targets(q, 0.01, 0.01), q)
        self.assertAlmostEqual(w2.squeeze_force, w3.squeeze_force, places=3)

    def test_the_force_cap_holds_the_width_on_both_arms(self) -> None:
        q = _q_full()
        w = self._worker(box_squeeze_force=4.0)
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        targets = self._targets(q, 0.03, 0.03)  # 3 cm in: ~18 N without a cap
        out = self._settle(w, self._box(), targets, q)
        self.assertAlmostEqual(w.squeeze_force, 4.0, places=6)
        # Each target is pulled back out along its own normal by the same
        # amount: the width opens symmetrically, the pair's centre stays.
        back_l = float(out["left"][0][1] - targets["left"][0][1])  # +y = out
        back_r = float(targets["right"][0][1] - out["right"][0][1])
        self.assertGreater(back_l, 0.015)
        self.assertLess(back_l, 0.03)
        self.assertAlmostEqual(back_l, back_r, places=3)
        # And with the lean scaled off the cap still holds — it is not the
        # lean's to switch off.
        w0 = self._worker(box_squeeze_lean=0.0, box_squeeze_force=4.0)
        w0.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        out0 = self._settle(w0, self._box(), targets, q)
        self.assertAlmostEqual(w0.squeeze_force, 4.0, places=6)
        np.testing.assert_allclose(out0["left"][1], targets["left"][1], atol=1e-7)
        self.assertGreater(float(out0["left"][0][1] - targets["left"][0][1]), 0.015)

    def test_the_depth_is_low_passed(self) -> None:
        q = _q_full()
        w = self._worker()
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        box = self._box()
        # First sample seeds the filter; a step in the raw depth is then
        # followed with the time constant.
        w._measured_t = 100.0
        w._squeeze_lean(box, self._targets(q, 0.0, 0.0), q, 100.0)
        step = self._targets(q, 0.01, 0.01)
        w._measured_t = 100.0 + _LEAN_TAU_S
        w._squeeze_lean(box, step, q, 100.0 + _LEAN_TAU_S)
        self.assertAlmostEqual(w._lean_depth, 0.01 * (1 - math.exp(-1)), places=6)
        # A long gap restarts it at the raw value.
        w._measured_t = 200.0
        w._squeeze_lean(box, step, q, 200.0)
        self.assertAlmostEqual(w._lean_depth, 0.01, places=6)

    def test_a_stale_measurement_is_no_measurement(self) -> None:
        q = _q_full()
        w = self._worker()
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        targets = self._targets(q, 0.01, 0.01)
        self._settle(w, self._box(), targets, q)
        self.assertGreater(w.squeeze_force, 4.0)
        # The core stops reporting (no reading from the arms): the lean
        # lets go rather than working from an old measurement.
        w._measured_t = 100.0
        out = w._squeeze_lean(self._box(), targets, q, 101.0)
        self.assertIs(out, targets)
        self.assertEqual(w.squeeze_force, 0.0)

    def test_pair_status_reports_the_force(self) -> None:
        q = _q_full()
        w = self._worker()
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        self._settle(w, self._box(), self._targets(q, 0.01, 0.01), q)
        self.assertGreater(w.squeeze_force, 4.0)
        self.assertEqual(round(w.squeeze_force, 1), round(w._lean_force, 1))


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

    def test_urdf_tool_touches_at_the_mount(self) -> None:
        for grasp in ("flush", "straight"):
            pts = URDF_TOOL.contacts(grasp)
            self.assertEqual(len(pts), 1)
            np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0])

    def test_flush_tilt(self) -> None:
        self.assertAlmostEqual(math.degrees(parcel_tool(141.5).flush_tilt), 38.5)


class _Arm:
    def __init__(self, positions, kp=_KP_L, fail=False):
        self._positions = positions
        self.kp = kp
        self._fail = fail

    @property
    def positions(self):
        if self._fail:
            raise RuntimeError("no feedback yet")
        return self._positions


class MeasuredArmsTest(unittest.TestCase):
    def test_reads_both_arms(self) -> None:
        robot = SimpleNamespace(
            left=_Arm(np.arange(8, dtype=np.float32)),
            right=_Arm(np.arange(8, 16, dtype=np.float32), kp=_KP_R),
        )
        pos_l, pos_r, kp_l, kp_r = measured_arms(robot)
        np.testing.assert_array_equal(pos_l, np.arange(8))
        np.testing.assert_array_equal(pos_r, np.arange(8, 16))
        np.testing.assert_array_equal(kp_l, _KP_L)
        np.testing.assert_array_equal(kp_r, _KP_R)

    def test_none_without_arms_or_a_reading(self) -> None:
        self.assertIsNone(measured_arms(object()))  # the sim
        self.assertIsNone(measured_arms(None))
        one = SimpleNamespace(left=_Arm(np.zeros(8, np.float32)), right=None)
        self.assertIsNone(measured_arms(one))  # a bench arm
        failing = SimpleNamespace(
            left=_Arm(np.zeros(8, np.float32), fail=True),
            right=_Arm(np.zeros(8, np.float32)),
        )
        self.assertIsNone(measured_arms(failing))
        nan = SimpleNamespace(
            left=_Arm(np.full(8, np.nan, np.float32)),
            right=_Arm(np.zeros(8, np.float32)),
        )
        self.assertIsNone(measured_arms(nan))


def _core(**overrides) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**overrides),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


class ConfigAndLiveTest(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = VRTeleopConfig()
        self.assertEqual(cfg.box_squeeze_lean, 1.0)  # the model's lean, on
        self.assertEqual(cfg.box_squeeze_force, 0.0)  # no force cap
        self.assertEqual(cfg.box_squeeze_torque, 0.0)  # no torque cap

    def test_forwarded_to_the_worker(self) -> None:
        core = _core(box_mode=True)
        for key in ("box_squeeze_lean", "box_squeeze_force"):
            self.assertIn(key, core._LIVE_WORKER_FIELDS)
        core.set_live("box_squeeze_lean", 1.5)
        core.set_live("box_squeeze_force", 8)
        core._apply_live_requests()
        self.assertEqual(core.config.box_squeeze_lean, 1.5)
        self.assertEqual(core.config.box_squeeze_force, 8.0)
        self.assertIn(("box_squeeze_lean", 1.5), core._worker_updates)
        self.assertIn(("box_squeeze_force", 8.0), core._worker_updates)

    def test_live_settings_published_only_with_measured_arms(self) -> None:
        core = _core(box_mode=True)
        hardware = LiveSettings(
            core,
            SimpleNamespace(left=_Arm(np.zeros(8)), right=_Arm(np.zeros(8))),
            lambda s: None,
        )
        sim = LiveSettings(core, object(), lambda s: None)
        keys = {d["key"] for d in hardware.schema()}
        self.assertIn("box_squeeze_lean", keys)
        self.assertIn("box_squeeze_force", keys)
        self.assertEqual(hardware.values()["box_squeeze_lean"], 1.0)
        sim_keys = {d["key"] for d in sim.schema()}
        self.assertNotIn("box_squeeze_lean", sim_keys)
        self.assertNotIn("box_squeeze_force", sim_keys)
        with self.assertRaises(ValueError):
            sim.apply("box_squeeze_lean", 2.0)
        hardware.apply("box_squeeze_lean", 0.5)
        core._apply_live_requests()
        self.assertEqual(core.config.box_squeeze_lean, 0.5)
        with self.assertRaises(ValueError):
            hardware.apply("box_squeeze_force", -1)


class _Conn:
    """Pipe stub for run_ik_loop: records sends, answers each frame once."""

    def __init__(self, q):
        self.sent: list = []
        self._q = q

    def send(self, msg):
        self.sent.append(msg)

    def poll(self, _timeout=None):
        return True

    def recv(self):
        return (self._q.copy(), None)


def _frame(l_lock: bool, r_lock: bool):
    from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=VRPose(position=zero, quaternion=identity),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=l_lock,
        r_lock=r_lock,
    )


def _meas_messages(sent: list) -> list:
    return [m for m in sent if isinstance(m, tuple) and m and m[0] == "meas"]


class ForwardingTest(unittest.TestCase):
    def _run(self, core, frame_fn, measured, n_frames: int) -> list:
        import threading

        conn = _Conn(np.zeros(16, np.float32))
        stop = threading.Event()
        frames = {"n": 0}

        def get_frame():
            frames["n"] += 1
            if frames["n"] > n_frames:
                stop.set()
            return frame_fn()  # a fresh frame each time

        core.run_ik_loop(
            conn,
            get_frame,
            stop,
            lambda: True,
            lambda _t: None,
            get_measured=lambda: measured,
        )
        return conn.sent

    def test_measurement_is_sent_ahead_of_box_mode_frames(self) -> None:
        from almond_axol.vr.models import VRFrame

        core = _core(box_mode=True, ik_frequency=1000.0)
        core.set_target(np.zeros(16, np.float32))
        # Right grip rising edge: the right hand leads the pair.
        core.update_engage(_frame(False, False))
        core.update_engage(_frame(False, True))
        self.assertTrue(core.teleop_enabled)
        measured = (np.zeros(8), np.ones(8), _KP_L, _KP_R)
        sent = self._run(core, lambda: _frame(False, True), measured, 6)
        meas = _meas_messages(sent)
        self.assertTrue(meas, sent)
        self.assertEqual(len(meas[0]), 5)
        np.testing.assert_array_equal(meas[0][1], np.zeros(8))
        np.testing.assert_array_equal(meas[0][2], np.ones(8))
        # Each measurement is immediately followed by its frame.
        for i, m in enumerate(sent[:-1]):
            if isinstance(m, tuple) and m and m[0] == "meas":
                self.assertIsInstance(sent[i + 1], VRFrame)

    def test_nothing_is_sent_outside_box_mode(self) -> None:
        core = _core(ik_frequency=1000.0)
        core.set_target(np.zeros(16, np.float32))
        core.update_engage(_frame(False, False))
        core.update_engage(_frame(True, True))
        measured = (np.zeros(8), np.zeros(8), _KP_L, _KP_R)
        sent = self._run(core, lambda: _frame(True, True), measured, 4)
        self.assertTrue(sent)
        self.assertFalse(_meas_messages(sent))


class _YawedSolver:
    """Solver stub: FK is a level pair whose grippers sit yawed about up.

    ``yaw`` is per side (rad, right-handed about +z); the mounts are at the
    pair's slots so a target pushed inward reads as depth.
    """

    left_indices = list(range(0, 7))
    right_indices = list(range(8, 15))

    def __init__(self, yaw: dict[str, float]) -> None:
        self.yaw = yaw

    @staticmethod
    def ideal() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        rot = np.eye(3, dtype=np.float32)
        return ideal_gripper_poses(
            np.array((0.4, 0.0, 0.3), np.float32),
            rot,
            0.3,
            {
                "left": side_clamp_rotation(1.0, 1.0, 0.0),
                "right": side_clamp_rotation(-1.0, 1.0, 0.0),
            },
        )

    def fk(self, q):
        del q
        up = np.array((0.0, 0.0, 1.0))
        out = {}
        for side, (pos, rot) in self.ideal().items():
            out[side] = (pos, (rodrigues(up, self.yaw[side]) @ rot).astype(np.float32))
        return out["left"], out["right"]


class WorkerTrimTest(unittest.TestCase):
    """The squeeze trim: measured toe-out integrated into inward yaw."""

    def _worker(self, solver, **cfg) -> IKWorker:
        w = object.__new__(IKWorker)
        w._config = types.SimpleNamespace(
            **{
                "box_tool": "urdf",
                "box_grasp": "straight",
                "box_squeeze_lean": 0.0,
                "box_squeeze_trim": 5.0,
                "box_squeeze_force": 0.0,
                **cfg,
            }
        )
        w._solver = solver
        w._lean_model = None
        w._measured, w._measured_t = None, 0.0
        w._lean_depth, w._lean_t, w._lean_force = 0.0, None, 0.0
        w._lean_trim = {"left": 0.0, "right": 0.0}
        w._trim_prev = None
        w._trim_moving_t = -math.inf
        w._clamp_log_t = -math.inf
        q = np.zeros(16, np.float32)
        w.note_measured(q[0:8], q[8:16], _KP_L, _KP_R)
        return w

    def _box(self) -> BoxState:
        return BoxState(
            center=np.zeros(3, np.float32),
            rot=np.eye(3, dtype=np.float32),
            width=0.3,
            face={"left": 1.0, "right": 1.0},
            tilt=0.0,
            align_start={},
            align_t0=0.0,
            align_duration=0.0,
            tool=URDF_TOOL,
        )

    @staticmethod
    def _targets(depth: float):
        """The ideal slots pushed ``depth`` into the box (the left is at +y)."""
        ideal = _YawedSolver.ideal()
        return {
            "left": (
                ideal["left"][0] + np.array([0, -depth, 0], np.float32),
                ideal["left"][1],
            ),
            "right": (
                ideal["right"][0] + np.array([0, depth, 0], np.float32),
                ideal["right"][1],
            ),
        }

    def _run(self, w, targets, seconds: float, t0: float = 100.0):
        out = targets
        n = int(seconds * 120)
        for i in range(n):
            t = t0 + i / 120.0
            w._measured_t = t
            out = w._squeeze_lean(self._box(), targets, np.zeros(16, np.float32), t)
        return out, t0 + n / 120.0

    def test_toe_out_integrates_into_inward_yaw_and_is_bounded(self) -> None:
        # Tips 2° off the box on both sides (left tip swings away with +yaw).
        w = self._worker(
            _YawedSolver({"left": math.radians(2.0), "right": -math.radians(2.0)})
        )
        targets = self._targets(0.01)
        out, t = self._run(w, targets, 1.0)
        # 1°/s per degree of error once the pair has been still 0.3 s, so
        # ~1.4° after a second.
        self.assertGreater(w.squeeze_trim_deg, 1.2)
        self.assertLess(w.squeeze_trim_deg, 2.0)
        # The targets are yawed tip-inward by the trim: relative to the
        # ideal they read as *negative* toe-out of that size.
        normals = {
            "left": -np.array([0.0, 1.0, 0.0]),
            "right": np.array([0.0, 1.0, 0.0]),
        }
        up = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(
            math.degrees(toe_out(targets, out, normals, up)),
            -w.squeeze_trim_deg,
            places=4,
        )
        # Positions untouched; the trim is a yaw.
        for side in ("left", "right"):
            np.testing.assert_array_equal(out[side][0], targets[side][0])
        # Kept on: bounded by box_squeeze_trim.
        out, t = self._run(w, targets, 6.0, t0=t)
        self.assertAlmostEqual(w.squeeze_trim_deg, 5.0, places=6)

    def test_the_trim_bleeds_away_once_the_grippers_stop_pressing(self) -> None:
        w = self._worker(
            _YawedSolver({"left": math.radians(2.0), "right": -math.radians(2.0)})
        )
        _out, t = self._run(w, self._targets(0.01), 3.0)
        self.assertGreater(w.squeeze_trim_deg, 2.5)
        # Width jogged back out: the targets sit at (in fact behind) the
        # measured mounts, no depth, and the trim decays (1 s).
        out, t = self._run(w, self._targets(-0.01), 3.0, t0=t)
        self.assertLess(w.squeeze_trim_deg, 0.3)
        self.assertEqual(w.squeeze_force, 0.0)

    def test_the_trims_are_per_arm(self) -> None:
        # Only the right tip is off the box (its blade gives more): only
        # the right target is trimmed.
        w = self._worker(_YawedSolver({"left": 0.0, "right": -math.radians(2.0)}))
        targets = self._targets(0.01)
        out, _t = self._run(w, targets, 1.0)
        trims = w.squeeze_trims_deg
        self.assertAlmostEqual(trims["left"], 0.0, places=9)
        self.assertGreater(trims["right"], 1.2)
        np.testing.assert_array_equal(out["left"][1], targets["left"][1])
        self.assertAlmostEqual(w.squeeze_trim_deg, 0.5 * trims["right"], places=9)

    def test_a_moving_pair_holds_the_trims(self) -> None:
        # A turn of the pair lags both arms the same way about up — toe-out
        # on one side, toe-in on the other. Per-arm trims can't average
        # that away, so they only integrate once the pair has been still.
        w = self._worker(
            _YawedSolver({"left": math.radians(3.0), "right": math.radians(3.0)})
        )
        still = self._targets(0.01)
        up = np.array([0.0, 0.0, 1.0])
        n = int(2.0 * 120)
        for i in range(n):
            t = 100.0 + i / 120.0
            w._measured_t = t
            # The commanded pair turns 20°/s: nothing integrates.
            turning = {
                side: (
                    pos,
                    (rodrigues(up, math.radians(20.0) * (i / 120.0)) @ rot).astype(
                        np.float32
                    ),
                )
                for side, (pos, rot) in still.items()
            }
            w._squeeze_lean(self._box(), turning, np.zeros(16, np.float32), t)
        self.assertEqual(w.squeeze_trims_deg, {"left": 0.0, "right": 0.0})
        # Still again: the first 0.3 s are the lag settling, then the trims
        # follow whatever toe-out is left (here the stub's, per side).
        _out, _t = self._run(w, still, 1.3, t0=100.0 + n / 120.0)
        trims = w.squeeze_trims_deg
        self.assertGreater(trims["left"], 0.5)  # a +yaw swings the left tip out
        self.assertLess(trims["right"], -0.5)  # and the right tip in

    def test_off_at_zero(self) -> None:
        w = self._worker(
            _YawedSolver({"left": math.radians(2.0), "right": -math.radians(2.0)}),
            box_squeeze_trim=0.0,
        )
        targets = self._targets(0.01)
        out, _t = self._run(w, targets, 1.0)
        self.assertIs(out, targets)
        self.assertEqual(w.squeeze_trim_deg, 0.0)

    def test_a_pair_still_blending_resets_the_trim(self) -> None:
        w = self._worker(
            _YawedSolver({"left": math.radians(2.0), "right": -math.radians(2.0)})
        )
        self._run(w, self._targets(0.01), 1.0)
        self.assertGreater(w.squeeze_trim_deg, 1.0)
        box = self._box()
        box.align_duration = 1.0
        targets = self._targets(0.01)
        self.assertIs(
            w._squeeze_lean(box, targets, np.zeros(16, np.float32), 200.0), targets
        )
        self.assertEqual(w.squeeze_trim_deg, 0.0)


if __name__ == "__main__":
    unittest.main()
