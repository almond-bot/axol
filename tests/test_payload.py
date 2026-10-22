"""Runtime payload in the gravity model (``Axol.set_payload``).

A payload is a point mass lumped into the ``wrist_3`` body with its CoM given
in the gripper link frame. Covers the model update (and its exact undo),
the gripper-frame transform, validation, and the robot-level delegation.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from almond_axol.constants import GRIPPER_TIP_OFFSET
from almond_axol.robot import Axol, Sim
from almond_axol.robot import axol as axol_module
from almond_axol.robot.axol import AxolHardware
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import (
    MAX_PAYLOAD_KG,
    GravityCompensator,
    payload_sides,
    validate_payload,
)

# Elbow bent forward: every joint up the chain carries a horizontal lever.
_BENT = np.array([0.3, 0.2, 0.0, -1.2, 0.1, 0.2, 0.0], dtype=np.float32)


class GravityPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AxolConfig()
        self.gc = GravityCompensator(self.config)

    def test_zero_payload_restores_the_configured_model_exactly(self) -> None:
        for is_left in (True, False):
            with self.subTest(is_left=is_left):
                before = self.gc.gravity_arm(_BENT, is_left=is_left)
                self.gc.set_payload(0.8, GRIPPER_TIP_OFFSET, is_left=is_left)
                loaded = self.gc.gravity_arm(_BENT, is_left=is_left)
                self.assertGreater(float(np.max(np.abs(loaded - before))), 0.5)
                self.gc.set_payload(0.0, is_left=is_left)
                np.testing.assert_array_equal(
                    self.gc.gravity_arm(_BENT, is_left=is_left), before
                )

    def test_payload_only_loads_its_own_arm(self) -> None:
        right_before = self.gc.gravity_arm(_BENT, is_left=False)
        self.gc.set_payload(1.0, GRIPPER_TIP_OFFSET, is_left=True)
        np.testing.assert_array_equal(
            self.gc.gravity_arm(_BENT, is_left=False), right_before
        )

    def test_a_new_payload_replaces_the_old_one(self) -> None:
        self.gc.set_payload(0.5, GRIPPER_TIP_OFFSET, is_left=True)
        once = self.gc.gravity_arm(_BENT, is_left=True)
        self.gc.set_payload(2.0, (0.0, 0.0, 0.0), is_left=True)
        self.gc.set_payload(0.5, GRIPPER_TIP_OFFSET, is_left=True)
        np.testing.assert_allclose(
            self.gc.gravity_arm(_BENT, is_left=True), once, atol=1e-6
        )
        mass, com = self.gc.payload(is_left=True)
        self.assertEqual(mass, 0.5)
        np.testing.assert_allclose(com, GRIPPER_TIP_OFFSET)

    def test_gripper_frame_com_matches_a_heavier_configured_link(self) -> None:
        # A payload placed exactly at wrist_3's own CoM is the same physics as
        # a heavier wrist_3 link — which only holds if the gripper-frame CoM
        # is mapped into the wrist_3 body frame correctly.
        extra = 0.6
        for is_left in (True, False):
            with self.subTest(is_left=is_left):
                arm = self.config.left if is_left else self.config.right
                rot, pos = self.gc._mount[is_left]
                link_com = np.asarray(arm.wrist_3.com, dtype=np.float64)
                com_grip = rot.T @ (link_com - pos)
                self.gc.set_payload(extra, com_grip, is_left=is_left)

                heavier = replace(
                    arm, wrist_3=replace(arm.wrist_3, mass=arm.wrist_3.mass + extra)
                )
                cfg = replace(
                    self.config,
                    **({"left": heavier} if is_left else {"right": heavier}),
                )
                reference = GravityCompensator(cfg)
                np.testing.assert_allclose(
                    self.gc.gravity_arm(_BENT, is_left=is_left),
                    reference.gravity_arm(_BENT, is_left=is_left),
                    atol=1e-5,
                )

    def test_elbow_holds_the_tip_payload_moment(self) -> None:
        # Elbow bent 90° with the upper arm hanging: the forearm is
        # horizontal, so 1 kg at the fingertips loads the elbow by
        # g · (forearm + wrist + tip reach) — 0.2-0.6 m on this arm.
        q = np.zeros(7, dtype=np.float32)
        q[3] = -np.pi / 2
        before = self.gc.gravity_arm(q, is_left=True)
        self.gc.set_payload(1.0, GRIPPER_TIP_OFFSET, is_left=True)
        delta = self.gc.gravity_arm(q, is_left=True) - before
        self.assertGreater(abs(float(delta[3])), 9.81 * 0.2)
        self.assertLess(abs(float(delta[3])), 9.81 * 0.6)


class PayloadValidationTest(unittest.TestCase):
    def test_accepts_numpy_scalars_and_sequences(self) -> None:
        mass, com = validate_payload(np.float32(0.5), np.array([0.0, 0.0, -0.1]))
        self.assertEqual(mass, 0.5)
        np.testing.assert_allclose(com, [0.0, 0.0, -0.1])

    def test_rejects_bad_masses(self) -> None:
        for bad in (-0.1, MAX_PAYLOAD_KG + 0.1, 500, float("nan"), True, "1"):
            with self.subTest(mass=bad), self.assertRaises(ValueError):
                validate_payload(bad, (0.0, 0.0, 0.0))  # type: ignore[arg-type]

    def test_rejects_bad_coms(self) -> None:
        for bad in ((0.0, 0.0), (0.0, 0.0, float("inf")), (0.0, 0.0, -2.0)):
            with self.subTest(com=bad), self.assertRaises(ValueError):
                validate_payload(0.5, bad)

    def test_sides(self) -> None:
        self.assertEqual(payload_sides("left"), (True,))
        self.assertEqual(payload_sides("right"), (False,))
        self.assertEqual(payload_sides("both"), (True, False))
        with self.assertRaises(ValueError):
            payload_sides("middle")


class RobotPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(patch.object(axol_module, "CanBus"))
        self.enterContext(patch("almond_axol.rt.robot.RtLink"))

    def test_axol_delegates_to_the_shared_gravity_model(self) -> None:
        robot = Axol(AxolConfig(), left_channel="can0", right_channel="can1")
        robot.set_payload("left", 0.4, GRIPPER_TIP_OFFSET)
        self.assertEqual(robot.payload("left")[0], 0.4)
        self.assertEqual(robot.payload("right")[0], 0.0)
        # The arms evaluate gravity through the same compensator.
        assert robot.left is not None
        self.assertIs(robot.left._gravity_comp, robot._robot._gravity_comp)

        robot.set_payload("both", 0.2)
        self.assertEqual(robot.payload("left")[0], 0.2)
        self.assertEqual(robot.payload("right")[0], 0.2)
        with self.assertRaises(ValueError):
            robot.payload("both")

    def test_invalid_payload_changes_nothing(self) -> None:
        hardware = AxolHardware(AxolConfig(), left_channel="can0", right_channel=None)
        hardware.set_payload("left", 0.3)
        with self.assertRaises(ValueError):
            hardware.set_payload("left", 300.0)  # grams by mistake
        self.assertEqual(hardware.payload("left")[0], 0.3)

    def test_sim_validates_like_the_robot(self) -> None:
        sim = Sim()
        sim.set_payload("right", 0.5, GRIPPER_TIP_OFFSET)
        with self.assertRaises(ValueError):
            sim.set_payload("right", -1.0)
        with self.assertRaises(ValueError):
            sim.set_payload("up", 0.5)


if __name__ == "__main__":
    unittest.main()
