"""IK places the Jelly deck by the live lift height.

IK freezes the lift fully raised and solves in the torso frame, which the
lift carries with the arms; lowering it brings the deck (and the lower
column) up toward them. The solver moves those links by the height the lift
reports, and the teleop core forwards that height to the IK worker.
"""

from __future__ import annotations

import logging
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from almond_axol.constants import torso_links, upper_arm_guard_links
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.model import lift_link_shift, shared_robot
from almond_axol.kinematics.solver import KinematicsSolver, body_collision_distance
from almond_axol.robot.jelly import Jelly
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore

_DECK = tuple(f"deck_{i}" for i in range(3))


class LiftShiftTest(unittest.TestCase):
    def test_jelly_lift_leaves_the_deck_and_lower_column_behind(self) -> None:
        shift, travel = lift_link_shift("mobile", gripper="parallel")
        self.assertAlmostEqual(travel, 0.34)
        names = list(shared_robot("mobile", gripper="parallel").links.names)
        row = {n: shift[names.index(n)] for n in names}
        # Two equal stages: the deck rises twice as fast as the middle one.
        for link in ("jelly", *_DECK):
            np.testing.assert_allclose(row[link], [0, 0, 2], atol=1e-6)
        np.testing.assert_allclose(row["lift_stage"], [0, 0, 1], atol=1e-6)
        # The top section, plate, shoulder bar and arms ride the lift.
        for link in ("base", "lift_plate", "s1", "head", "left_e1", "right_gripper"):
            np.testing.assert_allclose(row[link], 0, atol=1e-6)

    def test_versions_without_a_driven_lift_have_none(self) -> None:
        shift, travel = lift_link_shift("owl")
        self.assertEqual(travel, 0.0)
        self.assertFalse(np.any(shift))

    def test_the_deck_is_collided_but_not_hard_stopped(self) -> None:
        self.assertTrue(set(_DECK) <= set(torso_links("mobile")))
        self.assertFalse(set(_DECK) & set(upper_arm_guard_links("mobile")))


class SolverLiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = KinematicsSolver(
            KinematicsConfig(robot_model="mobile", gripper="parallel")
        )
        rc = cls.solver.robot_coll
        cls.deck = [
            k
            for k, (i, j) in enumerate(zip(rc.active_idx_i, rc.active_idx_j))
            if {rc.link_names[int(i)], rc.link_names[int(j)]} & set(_DECK)
        ]

    def tearDown(self) -> None:
        self.solver.set_lift(1.0)

    def _deck_clearance(self, q: np.ndarray) -> float:
        s = self.solver
        d = body_collision_distance(
            s.robot, s.robot_coll, jnp.asarray(s.to_pyroki_order(q)), s.link_shift
        )
        return float(np.asarray(d)[self.deck].min())

    def test_raised_is_the_frozen_model(self) -> None:
        s = self.solver
        q = jnp.zeros(s.robot.joints.num_actuated_joints)
        np.testing.assert_allclose(
            body_collision_distance(s.robot, s.robot_coll, q, s.link_shift),
            s.robot_coll.compute_self_collision_distance(s.robot, q),
            atol=1e-6,
        )
        self.assertEqual(s.lift_lowered, 0.0)

    def test_lowering_brings_the_deck_to_the_hanging_arms(self) -> None:
        home = np.zeros(14, np.float32)
        self.assertGreater(self._deck_clearance(home), 0.15)
        self.solver.set_lift(0.5)
        self.assertAlmostEqual(self.solver.lift_lowered, 0.17, places=6)
        # Hanging straight down with the lift half lowered, the hands are in
        # the deck (the physics hulls agree).
        self.assertLess(self._deck_clearance(home), 0.0)

    def test_ik_keeps_the_hands_off_the_risen_deck(self) -> None:
        s = self.solver
        q = np.zeros(14, np.float32)
        (lp, lr), (rp, rr) = s.fk(q)
        targets = dict(
            left_pose=(lp + [0.3, 0.0, -0.05], lr),
            right_pose=(rp + [0.3, 0.0, -0.05], rr),
        )
        for _ in range(150):
            q = s.ik(q, **targets)
        s.set_lift(0.5)
        # The lift came down with the hands just over the deck: now in it.
        self.assertLess(self._deck_clearance(q), 0.0)
        for _ in range(150):
            q = s.ik(q, **targets)
        # The soft cost lifts them back out, holding the same target.
        self.assertGreater(self._deck_clearance(q), 0.005)


class JellyLiftRaisedTest(unittest.TestCase):
    def _jelly(self, lift) -> Jelly:
        jelly = Jelly.__new__(Jelly)
        jelly._lift = lift
        return jelly

    def test_reads_the_homed_height(self) -> None:
        lift = SimpleNamespace(height_percent=40.0, status_is_fresh=lambda _s: True)
        self.assertAlmostEqual(self._jelly(lift).lift_raised, 0.4)

    def test_unknown_without_a_fresh_homed_reading(self) -> None:
        self.assertIsNone(self._jelly(None).lift_raised)
        stale = SimpleNamespace(height_percent=40.0, status_is_fresh=lambda _s: False)
        self.assertIsNone(self._jelly(stale).lift_raised)
        unhomed = SimpleNamespace(height_percent=None, status_is_fresh=lambda _s: True)
        self.assertIsNone(self._jelly(unhomed).lift_raised)


class _Conn:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, msg) -> None:
        self.sent.append(msg)

    def poll(self, _timeout=None) -> bool:
        return True

    def recv(self):
        return (np.zeros(16, np.float32), None)


class ForwardLiftTest(unittest.TestCase):
    def test_sends_the_height_when_it_moves(self) -> None:
        core = VRTeleopCore(
            VRTeleopConfig(ik_frequency=1000.0),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )
        core.set_target(np.zeros(16, np.float32))
        heights = iter([None, 1.0, 1.0, 0.9995, 0.5, None, 0.5, 0.25])
        stop = threading.Event()

        def get_lift():
            try:
                return next(heights)
            except StopIteration:
                stop.set()
                return 0.25

        conn = _Conn()
        core.run_ik_loop(
            conn,
            MagicMock(return_value=None),
            stop,
            lambda: True,
            lambda _t: None,
            get_lift=get_lift,
        )
        lifts = [m[1] for m in conn.sent if isinstance(m, tuple) and m[0] == "lift"]
        # Unknown heights send nothing; sub-permille moves are dropped.
        self.assertEqual(lifts, [1.0, 0.5, 0.25])


if __name__ == "__main__":
    unittest.main()
