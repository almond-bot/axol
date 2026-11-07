"""Plain teleop's stow: A with the parcel gripper, outside box mode.

The worker carries the two grippers in as a rigid pair (``_pair_frame`` /
``_pair_targets``) so a box squeezed between them keeps its squeeze; the
core starts it on an A press, holds both grips, cancels it on a grip click
and freezes both arms when the worker reports it done.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.worker import _pair_frame, _pair_targets


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


class PairFrameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.left = (np.array((0.4, 0.16, 0.8)), _rot("z", -20.0) @ _rot("y", 10.0))
        self.right = (np.array((0.45, -0.13, 0.75)), _rot("z", 25.0))

    def test_frame_is_centred_with_y_from_right_to_left(self) -> None:
        center, rot = _pair_frame(self.left, self.right)
        np.testing.assert_allclose(center, (0.425, 0.015, 0.775))
        y = self.left[0] - self.right[0]
        np.testing.assert_allclose(rot[:, 1], y / np.linalg.norm(y), atol=1e-12)
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)
        self.assertGreater(float(rot[2, 2]), 0.9)  # z as near up as it can be

    def test_moving_the_frame_keeps_the_grippers_relative_poses(self) -> None:
        center, rot = _pair_frame(self.left, self.right)
        rel = {
            side: (rot.T @ (pose[0] - center), rot.T @ pose[1])
            for side, pose in (("left", self.left), ("right", self.right))
        }
        same = _pair_targets((center, rot), rel)
        for side, pose in (("left", self.left), ("right", self.right)):
            np.testing.assert_allclose(same[side][0], pose[0], atol=1e-6)
            np.testing.assert_allclose(same[side][1], pose[1], atol=1e-6)
        moved = _pair_targets((np.array((0.2, 0.0, 0.6)), np.eye(3)), rel)
        before = self.right[1].T @ self.left[1], self.left[0] - self.right[0]
        lr, rr = moved["left"][1], moved["right"][1]
        np.testing.assert_allclose(rr.T @ lr, before[0], atol=1e-6)
        self.assertAlmostEqual(
            float(np.linalg.norm(moved["left"][0] - moved["right"][0])),
            float(np.linalg.norm(before[1])),
            places=6,
        )


def _core(**overrides) -> VRTeleopCore:
    core = VRTeleopCore(
        VRTeleopConfig(**{"gripper": "parcel", **overrides}),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
    )
    core.q = np.zeros(16, dtype=np.float32)
    return core


def _frame(*, stow=False, l_lock=False, r_lock=False, l_grip=0.3, r_grip=0.8):
    return SimpleNamespace(
        stow=stow, l_lock=l_lock, r_lock=r_lock, l_grip=l_grip, r_grip=r_grip
    )


class CoreTest(unittest.TestCase):
    def test_a_starts_it_and_the_grips_hold(self) -> None:
        core = _core()
        core.left_enabled = core.right_enabled = True
        core.l_grip, core.r_grip = 0.4, 0.6
        self.assertTrue(core._update_arm_stow(_frame(stow=True)))
        self.assertTrue(core._arm_stowing and core._stowing and core._stow_pulse)
        # While it runs it takes every frame: the triggers don't reach the grips.
        self.assertTrue(core._update_arm_stow(_frame(l_grip=0.0, r_grip=0.0)))
        self.assertEqual((core.l_grip, core.r_grip), (0.4, 0.6))

    def test_only_with_the_parcel_gripper(self) -> None:
        core = _core(gripper="parallel")
        self.assertFalse(core._update_arm_stow(_frame(stow=True)))
        self.assertFalse(core._arm_stowing)

    def test_a_grip_click_cancels_it_and_freezes_both_arms(self) -> None:
        core = _core()
        core.left_enabled = core.right_enabled = True
        core._update_arm_stow(_frame(stow=True))
        core._update_arm_stow(_frame(r_lock=True))
        self.assertFalse(core._arm_stowing or core._stowing)
        self.assertFalse(core.left_enabled or core.right_enabled)

    def test_done_freezes_both_arms(self) -> None:
        core = _core()
        core.right_enabled = True
        core._update_arm_stow(_frame(stow=True))
        core._finish_stow({"stow": "move"})
        self.assertFalse(core._stow_pulse)
        self.assertTrue(core._arm_stowing)
        core._finish_stow({"stow": "done"})
        self.assertFalse(core._arm_stowing or core._stowing)
        self.assertFalse(core.left_enabled or core.right_enabled)


if __name__ == "__main__":
    unittest.main()
