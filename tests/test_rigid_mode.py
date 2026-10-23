"""Rigid mode: the stock gripper's pair mode.

The operator sets each gripper's pose and stroke in ordinary teleop; rigid
mode then locks the two together exactly as they are and one controller
moves the pair as a rigid body in all six degrees of freedom. There is no
alignment blend, width or grasp control (box mode's, the parcel gripper's),
the strokes are held, and the sticks only jog the elbows.
"""

from __future__ import annotations

import logging
import types
import unittest

import numpy as np

from almond_axol.teleop.box import rigid_targets, rodrigues, snap_rigid
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.worker import IKWorker, _relative_target_np
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

_UP = np.array((0.0, 0.0, 1.0))
# The VR world's up (+y) in the controller frame _vr_to_flu_np produces.
_CTRL_UP = np.array((0.0, 1.0, 0.0))


def _rot(axis, angle: float) -> np.ndarray:
    return rodrigues(np.asarray(axis, dtype=np.float64), angle).astype(np.float64)


def _relative(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Pose of ``b`` in the frame of ``a``."""
    pa, ra = np.asarray(a[0], np.float64), np.asarray(a[1], np.float64)
    pb, rb = np.asarray(b[0], np.float64), np.asarray(b[1], np.float64)
    return ra.T @ (pb - pa), ra.T @ rb


# Two arbitrary, unaligned gripper poses — rigid mode keeps them as they are.
_LEFT = (
    np.array((0.42, 0.18, 0.31), np.float32),
    (_rot((0, 0, 1), 0.5) @ _rot((1, 0, 0), 0.7)).astype(np.float32),
)
_RIGHT = (
    np.array((0.35, -0.22, 0.24), np.float32),
    (_rot((0, 1, 0), -0.4) @ _rot((1, 0, 0), -0.3)).astype(np.float32),
)


class RigidGeometryTest(unittest.TestCase):
    def test_identity_returns_the_snap_poses(self) -> None:
        state = snap_rigid(_LEFT, _RIGHT)
        out = rigid_targets(state, np.zeros(3), np.eye(3))
        for side, pose in (("left", _LEFT), ("right", _RIGHT)):
            np.testing.assert_allclose(out[side][0], pose[0], atol=1e-6)
            np.testing.assert_allclose(out[side][1], pose[1], atol=1e-6)

    def test_the_pair_moves_as_one_rigid_body_about_its_centre(self) -> None:
        state = snap_rigid(_LEFT, _RIGHT)
        turn = _rot((0.3, -0.5, 0.8) / np.linalg.norm((0.3, -0.5, 0.8)), 0.9)
        shift = np.array((0.05, -0.02, 0.1))
        out = rigid_targets(state, shift, turn)
        # The grippers keep their pose relative to each other ...
        p0, r0 = _relative(_LEFT, _RIGHT)
        p1, r1 = _relative(out["left"], out["right"])
        np.testing.assert_allclose(p1, p0, atol=1e-5)
        np.testing.assert_allclose(r1, r0, atol=1e-5)
        # ... each turned by the full rotation, about the shifted centre.
        center = 0.5 * (_LEFT[0] + _RIGHT[0])
        mid = 0.5 * (out["left"][0] + out["right"][0])
        np.testing.assert_allclose(mid, center + shift, atol=1e-6)
        np.testing.assert_allclose(out["left"][1], turn @ _LEFT[1], atol=1e-5)


class _Solver:
    """Fixed FK at the two unaligned poses; ``ik`` records its targets."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def set_posture_pose(self, q: np.ndarray) -> None:
        del q

    def fk(self, q: np.ndarray):
        del q
        return _LEFT, _RIGHT

    def ik(self, q: np.ndarray, **kwargs: object) -> np.ndarray:
        self.calls.append(kwargs)
        return np.asarray(q, dtype=np.float32).copy()


def _frame(**sticks: float | bool) -> VRFrame:
    identity = VRQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=VRPose(position=zero, quaternion=identity),
        r_ee=VRPose(position=zero, quaternion=identity),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=True,
        r_lock=True,
        **sticks,
    )


def _worker() -> IKWorker:
    worker = object.__new__(IKWorker)
    worker._config = types.SimpleNamespace(
        ik_frequency=120.0,
        position_multiplier=1.0,
        rotation_multiplier=1.0,
        gripper="parallel",
        box_width_speed=0.1,
        box_elbow_out=30.0,
        box_elbow_speed=30.0,
        box_elbow_weight=0.0,
        box_grasp="straight",
    )
    worker._solver = _Solver()
    worker._rec = None
    worker._active = {"left": False, "right": False}
    worker._hold_fk = {}
    worker._hold_elbow_fk = {}
    worker._ramp = {}
    worker._box = None
    worker._rigid = None
    worker._box_leader = None
    worker._box_snap = None
    worker._freeze_since = {}
    worker._freeze_targets = {}
    worker._last_solve_t = None
    return worker


_IDENTITY = (np.zeros(3, np.float32), np.eye(3, dtype=np.float32))


def _step(worker: IKWorker, leader: str, lead_ctrl=_IDENTITY, **sticks):
    frame = _frame(**sticks)
    frame.box_leader = leader
    ctrl = {"left": _IDENTITY, "right": _IDENTITY}
    ctrl[leader] = (
        np.asarray(lead_ctrl[0], np.float32),
        np.asarray(lead_ctrl[1], np.float32),
    )
    n = len(worker._solver.calls)
    worker._step_rigid(
        frame, np.zeros(14, np.float32), ctrl, ctrl["left"][0], ctrl["right"][0]
    )
    if len(worker._solver.calls) == n:
        return None
    call = worker._solver.calls[-1]
    return {"left": call["left_pose"], "right": call["right_pose"]}


class RigidWorkerTest(unittest.TestCase):
    def test_the_snap_moves_nothing(self) -> None:
        worker = _worker()
        self.assertIsNone(_step(worker, "right"))  # snap frame holds
        targets = _step(worker, "right")
        for side, pose in (("left", _LEFT), ("right", _RIGHT)):
            np.testing.assert_allclose(targets[side][0], pose[0], atol=1e-6)
            np.testing.assert_allclose(targets[side][1], pose[1], atol=1e-6)

    def test_the_leader_moves_the_pair_in_all_six_dof(self) -> None:
        worker = _worker()
        _step(worker, "right")
        # A pitch-and-roll the box mode would drop, plus a move.
        ctrl_rot = _rot((1, 0, 0), 0.4) @ _rot((0, 0, 1), -0.3) @ _rot(_CTRL_UP, 0.2)
        ctrl_pos = np.array((0.03, -0.04, 0.06))
        targets = _step(worker, "right", (ctrl_pos, ctrl_rot))
        p0, r0 = _relative(_LEFT, _RIGHT)
        p1, r1 = _relative(targets["left"], targets["right"])
        np.testing.assert_allclose(p1, p0, atol=1e-5)
        np.testing.assert_allclose(r1, r0, atol=1e-5)
        # The pair turns by exactly the leader's clutch-mapped rotation ...
        lead_pos, lead_rot = _relative_target_np(
            ctrl_pos, ctrl_rot, *_IDENTITY, *_RIGHT
        )
        np.testing.assert_allclose(targets["right"][1], lead_rot, atol=1e-5)
        # ... about the centre, carried by the leader gripper's translation.
        mid = 0.5 * (targets["left"][0] + targets["right"][0])
        center = 0.5 * (_LEFT[0] + _RIGHT[0])
        np.testing.assert_allclose(mid, center + (lead_pos - _RIGHT[0]), atol=1e-5)
        self.assertGreater(np.linalg.norm(lead_pos - _RIGHT[0]), 0.05)

    def test_sticks_jog_the_elbows_only(self) -> None:
        worker = _worker()
        _step(worker, "left")
        _step(worker, "left")  # starts the stick clock
        before = _step(worker, "left")
        worker._rigid.stick_t -= 0.05
        after = _step(worker, "left", l_stick_x=1.0)
        for side in ("left", "right"):
            np.testing.assert_allclose(after[side][0], before[side][0], atol=1e-6)
        self.assertEqual(worker._config.box_elbow_out, 30.0)
        worker._rigid.stick_t -= 0.05
        _step(worker, "left", r_stick_y=-1.0)  # forward: elbows out
        self.assertGreater(worker._config.box_elbow_out, 30.0)

    def test_a_handover_keeps_the_pair_where_the_leader_left_it(self) -> None:
        worker = _worker()
        _step(worker, "right")
        moved = (np.array((0.02, 0.05, -0.03)), _rot((1, 0, 0), 0.3))
        led = _step(worker, "right", moved)
        # The left hand takes over from wherever it is: nothing jumps.
        self.assertIsNone(_step(worker, "left", moved))
        held = _step(worker, "left", moved)
        for side in ("left", "right"):
            np.testing.assert_allclose(held[side][0], led[side][0], atol=1e-6)
            np.testing.assert_allclose(held[side][1], led[side][1], atol=1e-6)

    def test_the_parcel_gripper_dispatches_to_box_mode(self) -> None:
        worker = _worker()
        self.assertFalse(worker._is_parcel())
        worker._config.gripper = "parcel"
        self.assertTrue(worker._is_parcel())

    def test_pair_status_has_no_alignment_cue(self) -> None:
        status = _worker().pair_status(np.zeros(14))
        self.assertFalse(status["aligned"])
        self.assertEqual(status["squeeze"], 0.0)


def _core(**overrides) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**overrides),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )


def _grip_frame(l_lock: bool, r_lock: bool, l_grip: float, r_grip: float):
    frame = _frame(l_grip=l_grip, r_grip=r_grip)
    frame.l_lock = l_lock
    frame.r_lock = r_lock
    return frame


class RigidCoreTest(unittest.TestCase):
    def _engaged_two_arms(self, l_grip: float, r_grip: float) -> VRTeleopCore:
        core = _core(gripper="parallel")
        core.update_engage(_grip_frame(False, False, l_grip, r_grip))
        core.update_engage(_grip_frame(True, True, l_grip, r_grip))
        self.assertTrue(core.left_enabled and core.right_enabled)
        self.assertEqual((core.l_grip, core.r_grip), (l_grip, r_grip))
        return core

    def test_switching_on_engages_the_pair_and_holds_the_strokes(self) -> None:
        core = self._engaged_two_arms(0.2, 0.7)
        core.set_rigid_mode(True)
        core.update_engage(_grip_frame(False, False, 1.0, 1.0))
        self.assertTrue(core.box_mode)
        self.assertEqual(core._box_leader, "right")  # box_lead_hand default
        self.assertTrue(core.teleop_enabled)
        # Triggers no longer move the grippers.
        core.update_engage(_grip_frame(False, False, 0.9, 0.0))
        self.assertEqual((core.l_grip, core.r_grip), (0.2, 0.7))
        # Box mode's caps are box mode's alone.
        self.assertIsNone(core.spring_caps())
        # The other grip takes the lead over, strokes still held.
        core.update_engage(_grip_frame(True, False, 0.9, 0.0))
        self.assertEqual(core._box_leader, "left")
        self.assertEqual((core.l_grip, core.r_grip), (0.2, 0.7))

    def test_after_rigid_mode_a_trigger_must_reach_the_held_stroke(self) -> None:
        core = self._engaged_two_arms(0.2, 0.7)
        core.set_rigid_mode(True)
        core.update_engage(_grip_frame(False, False, 1.0, 1.0))
        core.set_rigid_mode(False)
        # Both arms engage at once; relaxed triggers don't drop the object.
        core.update_engage(_grip_frame(False, False, 1.0, 1.0))
        self.assertFalse(core.box_mode)
        self.assertTrue(core.left_enabled and core.right_enabled)
        self.assertEqual((core.l_grip, core.r_grip), (0.2, 0.7))
        core.update_engage(_grip_frame(False, False, 0.6, 0.9))
        self.assertEqual((core.l_grip, core.r_grip), (0.2, 0.7))
        # Left squeezes past its held stroke, right comes within tolerance.
        core.update_engage(_grip_frame(False, False, 0.1, 0.72))
        self.assertEqual((core.l_grip, core.r_grip), (0.1, 0.72))
        # From here the triggers drive the grippers as usual.
        core.update_engage(_grip_frame(False, False, 0.5, 1.0))
        self.assertEqual((core.l_grip, core.r_grip), (0.5, 1.0))

    def test_parcel_box_mode_keeps_its_caps_and_triggers(self) -> None:
        core = _core(gripper="parcel", box_mode=True)
        self.assertIsNotNone(core.spring_caps())
        core.update_engage(_grip_frame(False, False, 1.0, 1.0))
        core.update_engage(_grip_frame(False, True, 0.4, 0.3))
        self.assertEqual((core.l_grip, core.r_grip), (0.4, 0.3))


if __name__ == "__main__":
    unittest.main()
