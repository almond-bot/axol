"""Quest controller haptics: collision proximity and push force per arm.

The IK solvers report how close each arm is to the arm/base collision
boundary (``collision_proximity``), the worker returns it with each solve,
and ``VRTeleopCore`` combines it with the low-passed torque residuals into
the ``{"type": "haptics"}`` push the headset plays.
"""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.solver import KinematicsSolver
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.teleop import core as core_module
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.held_load import HeldLoad, load_regressor
from almond_axol.teleop.live import LiveSettings


def _core(sent: list | None = None, **config) -> VRTeleopCore:
    """A core with both cues on (the body ticks are off by default)."""
    return VRTeleopCore(
        VRTeleopConfig(**{"haptic_near": True, **config}),
        logging.getLogger("test"),
        broadcast_tracking=lambda _enabled: None,
        broadcast_json=(sent.append if sent is not None else None),
    )


class _Robot:
    """Torque residuals of a constant push on the left elbow."""

    def __init__(self, left_elbow: float = 0.0) -> None:
        self.left_elbow = left_elbow

    def torque_residuals(self):
        left = np.zeros(7)
        left[3] = self.left_elbow
        return left, np.zeros(7)


def _feed(core: VRTeleopCore, robot: object, seconds: float, hz: float = 250.0):
    """Run the control-loop hook for ``seconds`` of simulated time.

    The clock carries on from the previous call on the same core. The patched
    clock returns the current simulated time rather than consuming a sequence,
    so stray calls from other threads can't exhaust it.
    """
    n = int(seconds * hz)
    t0 = getattr(core, "_test_clock", 100.0)
    now = [t0]
    with patch.object(core_module.time, "perf_counter", side_effect=lambda: now[0]):
        for i in range(n):
            now[0] = t0 + i / hz
            core.maybe_broadcast_haptics(robot)
    core._test_clock = t0 + n / hz


class HapticLevelsTest(unittest.TestCase):
    def test_disengaged_arms_are_silent(self) -> None:
        core = _core()
        core.collision_near = {"left": 1.0, "right": 1.0}
        _feed(core, _Robot(left_elbow=20.0), 0.5)
        self.assertEqual(
            core.haptic_levels(),
            {"left": {"near": 0.0, "force": 0.0}, "right": {"near": 0.0, "force": 0.0}},
        )

    def test_near_and_force_per_arm(self) -> None:
        core = _core(haptic_torque=6.0)
        core.left_enabled = core.right_enabled = True
        core.collision_near = {"left": 0.1, "right": 1.0}
        # 10.5 Nm sustained: 4.5 past the 6 Nm start of a 9 Nm ramp.
        _feed(core, _Robot(left_elbow=10.5), 1.0)
        levels = core.haptic_levels()
        self.assertAlmostEqual(levels["left"]["force"], 0.5, places=2)
        self.assertEqual(levels["right"]["force"], 0.0)
        # A resting pose a little inside a pair's onset reads as clear.
        self.assertEqual(levels["left"]["near"], 0.0)
        self.assertEqual(levels["right"]["near"], 1.0)

    def test_a_short_spike_does_not_rumble(self) -> None:
        core = _core(haptic_torque=6.0)
        core.left_enabled = True
        _feed(core, _Robot(), 0.2)
        _feed(core, _Robot(left_elbow=12.0), 0.01)
        self.assertEqual(core.haptic_levels()["left"]["force"], 0.0)

    def test_nan_torque_is_ignored(self) -> None:
        core = _core()
        core.left_enabled = True
        robot = SimpleNamespace(torque_residuals=lambda: (np.full(7, np.nan), None))
        _feed(core, robot, 0.2)
        self.assertEqual(core.haptic_levels()["left"]["force"], 0.0)

    def test_off_zero_torque_and_reset_silence_it(self) -> None:
        for config, resetting in (
            ({"haptics": False}, False),
            ({}, True),
        ):
            with self.subTest(config=config, resetting=resetting):
                core = _core(**config)
                core.left_enabled = True
                core.collision_near = {"left": 1.0, "right": 0.0}
                _feed(core, _Robot(left_elbow=20.0), 0.5)
                with patch.object(
                    VRTeleopCore,
                    "is_resetting",
                    new_callable=unittest.mock.PropertyMock,
                    return_value=resetting,
                ):
                    self.assertEqual(core.haptic_levels()["left"]["near"], 0.0)
                    self.assertEqual(core.haptic_levels()["left"]["force"], 0.0)
        core = _core(haptic_torque=0.0)
        core.left_enabled = True
        core.collision_near = {"left": 1.0, "right": 0.0}
        _feed(core, _Robot(left_elbow=20.0), 0.5)
        self.assertEqual(core.haptic_levels()["left"], {"near": 1.0, "force": 0.0})

    def test_body_ticks_are_off_by_default(self) -> None:
        self.assertFalse(VRTeleopConfig().haptic_near)
        self.assertTrue(VRTeleopConfig().haptics)
        core = _core(haptic_near=False)
        core.left_enabled = True
        core.collision_near = {"left": 1.0, "right": 0.0}
        _feed(core, _Robot(left_elbow=20.0), 0.5)
        self.assertEqual(core.haptic_levels()["left"]["near"], 0.0)
        self.assertGreater(core.haptic_levels()["left"]["force"], 0.0)

    def test_box_mode_plays_on_the_leading_controller(self) -> None:
        core = _core()
        core.left_enabled = core.right_enabled = True
        core.box_mode = True
        core._box_leader = "right"
        core.collision_near = {"left": 1.0, "right": 0.0}
        levels = core.haptic_levels()
        self.assertEqual(levels["right"]["near"], 1.0)
        self.assertEqual(levels["left"]["near"], 0.0)


class HapticsBroadcastTest(unittest.TestCase):
    def test_pushes_while_playing_then_one_stop(self) -> None:
        sent: list = []
        core = _core(sent)
        _feed(core, _Robot(), 0.5)
        self.assertEqual(sent, [])
        core.left_enabled = True
        core.collision_near = {"left": 1.0, "right": 0.0}
        _feed(core, _Robot(), 0.5)
        self.assertGreater(len(sent), 10)
        self.assertLessEqual(len(sent), 16)
        self.assertEqual(sent[-1]["type"], "haptics")
        self.assertEqual(sent[-1]["value"]["left"]["near"], 1.0)
        sent.clear()
        core.left_enabled = False
        _feed(core, _Robot(), 0.5)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["value"]["left"], {"near": 0.0, "force": 0.0})

    def test_worker_reply_carries_proximity(self) -> None:
        core = _core()
        q = np.zeros(14)
        self.assertIs(core._unpack_solution((q, None, {"left": 0.5})), q)
        self.assertEqual(core.collision_near, {"left": 0.5})
        # A reply between proximity reports keeps the last one.
        core._unpack_solution((q, None, None))
        self.assertEqual(core.collision_near, {"left": 0.5})
        self.assertIs(core._unpack_solution((q, None)), q)


class LiveSettingTest(unittest.TestCase):
    def test_push_torque_is_hardware_only(self) -> None:
        core = _core()
        hardware = LiveSettings(core, _Robot(), lambda s: None)
        sim = LiveSettings(core, object(), lambda s: None)
        self.assertIn("haptic_torque", {d["key"] for d in hardware.schema()})
        self.assertNotIn("haptic_torque", {d["key"] for d in sim.schema()})
        self.assertIn("haptics", {d["key"] for d in sim.schema()})
        self.assertIn("haptic_near", {d["key"] for d in sim.schema()})

    def test_apply_routes_to_the_core(self) -> None:
        core = _core()
        live = LiveSettings(core, _Robot(), lambda s: None)
        live.apply("haptics", "toggle")
        live.apply("haptic_torque", 9.0)
        live.apply("haptic_near", "toggle")
        core._apply_live_requests()
        self.assertFalse(core.config.haptic_near)
        self.assertFalse(core.config.haptics)
        self.assertEqual(core.config.haptic_torque, 9.0)


# An arm reaching forward, elbow bent: a carrying pose.
_CARRY_Q = np.array([0.4, 0.3, 0.0, 1.2, 0.0, 0.3, 0.0])
_GRAVITY = GravityCompensator()


def _payload_torque(q: np.ndarray, mass: float, com=(0.0, 0.0, -0.12)) -> np.ndarray:
    """What holding ``mass`` at ``com`` (gripper frame) costs the left arm."""
    base = _GRAVITY.gravity_arm(q, is_left=True).astype(np.float64)
    _GRAVITY.set_payload(mass, com, is_left=True)
    try:
        return _GRAVITY.gravity_arm(q, is_left=True) - base
    finally:
        _GRAVITY.set_payload(0.0, is_left=True)


def _tip_force_torque(q: np.ndarray, force) -> np.ndarray:
    """What a force (N, base frame) on the left gripper mount costs the arm."""
    _, _, jac = _GRAVITY.mount_jacobian(q, is_left=True)
    return -(jac[:3].T @ np.asarray(force, dtype=np.float64))


class HeldLoadTest(unittest.TestCase):
    def _regressor(self, q, *, squeeze=False):
        _, rot, jac = _GRAVITY.mount_jacobian(q, is_left=True)
        return load_regressor(jac, rot, squeeze=squeeze)

    def test_regressor_matches_the_gravity_model_payload(self) -> None:
        a = self._regressor(_CARRY_Q)
        com = np.array([0.02, -0.01, -0.12])
        np.testing.assert_allclose(
            a @ np.r_[1.5, 1.5 * com], _payload_torque(_CARRY_Q, 1.5, com), atol=1e-4
        )

    def test_a_held_weight_is_explained_and_a_push_is_not(self) -> None:
        held = HeldLoad(squeeze=False)
        a = self._regressor(_CARRY_Q)
        weight = _payload_torque(_CARRY_Q, 2.0)
        for _ in range(50):
            explained = held.explain(a, weight, tol=3.0, learn=0.1)
        np.testing.assert_allclose(explained, weight, atol=1e-3)
        self.assertAlmostEqual(held.mass, 2.0, places=2)
        # Pushing sideways or up from below: the push stays in the cue and
        # the held weight is still explained underneath it.
        for force in ((0.0, 25.0, 0.0), (0.0, 0.0, 40.0)):
            push = _tip_force_torque(_CARRY_Q, force)
            explained = held.explain(a, weight + push, tol=3.0, learn=0.1)
            np.testing.assert_allclose(explained, weight, atol=0.05)
        self.assertAlmostEqual(held.mass, 2.0, places=2)

    def test_the_weight_follows_the_arm_to_a_new_pose(self) -> None:
        held = HeldLoad(squeeze=False)
        held.explain(
            self._regressor(_CARRY_Q),
            _payload_torque(_CARRY_Q, 2.0),
            tol=3.0,
            learn=1.0,
        )
        lowered = _CARRY_Q + np.array([0.6, 0.0, 0.5, -0.8, 0.4, 0.2, 0.3])
        push = _tip_force_torque(lowered, (0.0, 25.0, 0.0))
        explained = held.explain(
            self._regressor(lowered),
            _payload_torque(lowered, 2.0) + push,
            tol=3.0,
            learn=0.0,
        )
        np.testing.assert_allclose(explained, _payload_torque(lowered, 2.0), atol=0.05)

    def test_box_squeeze_is_fit_apart_from_the_weight_and_left_in(self) -> None:
        _, rot, _ = _GRAVITY.mount_jacobian(_CARRY_Q, is_left=True)
        a = self._regressor(_CARRY_Q, squeeze=True)
        held = HeldLoad(squeeze=True)
        squeeze = _tip_force_torque(_CARRY_Q, 12.0 * rot[:, 0])
        weight = _payload_torque(_CARRY_Q, 1.0)
        # Only the box's weight is explained; the clamp stays in the cue.
        np.testing.assert_allclose(
            held.explain(a, squeeze + weight, tol=3.0, learn=1.0), weight, atol=1e-3
        )
        self.assertAlmostEqual(held.mass, 1.0, places=2)
        poke = _tip_force_torque(_CARRY_Q, -30.0 * rot[:, 2])
        np.testing.assert_allclose(
            held.explain(a, squeeze + weight + poke, tol=3.0, learn=0.0),
            weight,
            atol=0.05,
        )


class _Arm:
    def __init__(self, q: np.ndarray) -> None:
        self.positions = np.r_[q, 0.0]


class _HoldingRobot:
    """A left arm at the carrying pose, holding ``mass`` plus a ``push``."""

    def __init__(self, mass: float, push=(0.0, 0.0, 0.0)) -> None:
        self.left = _Arm(_CARRY_Q)
        self.right = _Arm(np.zeros(7))
        self.residual = _payload_torque(_CARRY_Q, mass) + _tip_force_torque(
            _CARRY_Q, push
        )

    def torque_residuals(self):
        return self.residual, np.zeros(7)


class HeldLoadHapticsTest(unittest.TestCase):
    def _force(self, robot, *, grip: float, **config) -> float:
        core = _core(haptic_torque=4.0, **config)
        core.left_enabled = True
        core.l_grip = grip
        _feed(core, robot, 1.0)
        return core.haptic_levels()["left"]["force"]

    def test_a_held_object_does_not_rumble(self) -> None:
        heavy = _HoldingRobot(mass=2.0)
        self.assertGreater(self._force(heavy, grip=1.0), 0.0)
        self.assertEqual(self._force(heavy, grip=0.2), 0.0)
        self.assertGreater(self._force(heavy, grip=0.2, haptic_held_load=False), 0.0)

    def test_pushing_while_holding_still_rumbles(self) -> None:
        core = _core(haptic_torque=4.0)
        core.left_enabled = True
        core.l_grip = 0.2
        robot = _HoldingRobot(mass=2.0)
        _feed(core, robot, 1.0)
        self.assertEqual(core.haptic_levels()["left"]["force"], 0.0)
        robot.residual = robot.residual + _tip_force_torque(_CARRY_Q, (0, 30.0, 0))
        _feed(core, robot, 0.5)
        self.assertGreater(core.haptic_levels()["left"]["force"], 0.3)
        self.assertAlmostEqual(core.held_loads["left"].mass, 2.0, places=1)
        # Opening the gripper forgets the load.
        core.l_grip = 1.0
        _feed(core, robot, 0.1)
        self.assertNotIn("left", core.held_loads)

    def _box_force(self, robot, **config) -> float:
        core = _core(gripper="parcel", box_mode=True, **config)
        core.left_enabled = core.right_enabled = True
        core._box_leader = "left"
        core.l_grip = 1.0
        _feed(core, robot, 1.0)
        self.assertTrue(core.held_loads["left"].squeeze)
        return core.haptic_levels()["left"]["force"]

    def test_box_mode_carry_is_quiet_and_a_hard_squeeze_rumbles(self) -> None:
        _, rot, _ = _GRAVITY.mount_jacobian(_CARRY_Q, is_left=True)
        # A heavy box held with a light clamp: quiet.
        light = _HoldingRobot(mass=2.0, push=2.0 * rot[:, 0])
        self.assertEqual(self._box_force(light, haptic_torque=4.0), 0.0)
        # Jogging the width in past the box: the clamp rumbles.
        hard = _HoldingRobot(mass=2.0, push=25.0 * rot[:, 0])
        self.assertGreater(self._box_force(hard, haptic_torque=4.0), 0.3)

    def test_a_wrist_at_its_box_mode_cap_rumbles_full(self) -> None:
        robot = _HoldingRobot(mass=0.0)
        robot.residual = np.zeros(7)
        robot.residual[5] = 5.0  # wrist_2 held at box_wrist_torque
        self.assertEqual(
            self._box_force(robot, haptic_torque=6.0, box_wrist_torque=5.0), 1.0
        )
        robot.residual[5] = 3.0  # below 80% of the cap
        self.assertEqual(
            self._box_force(robot, haptic_torque=6.0, box_wrist_torque=5.0), 0.0
        )


class SolverProximityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = KinematicsSolver(KinematicsConfig())
        config = VRTeleopConfig()
        cls.rest = np.concatenate([config.rest_pose_left, config.rest_pose_right])

    def _blocked(self, solver) -> np.ndarray:
        """Drive the left gripper target into the body, from rest."""
        (lp, lr), right = solver.fk(self.rest)
        target = np.array([lp[0], 0.0, lp[2]], dtype=np.float32)
        q = self.rest.astype(np.float32)
        for _ in range(60):
            q = solver.ik(q, left_pose=(target, lr), right_pose=right)
        return q

    def test_clear_at_home_and_rest(self) -> None:
        # Teleop's IK worker sets the rest pose as the reference.
        solver = KinematicsSolver(KinematicsConfig())
        solver.set_proximity_reference(self.rest)
        for q in (np.zeros(14, np.float32), self.rest):
            self.assertEqual(solver.collision_proximity(q), {"left": 0.0, "right": 0.0})

    def test_full_when_ik_holds_the_arm_off_the_body(self) -> None:
        levels = self.solver.collision_proximity(self._blocked(self.solver))
        self.assertEqual(levels["left"], 1.0)
        self.assertLess(levels["right"], 0.1)

    def test_rest_reads_clear_on_a_mount_close_to_the_arms(self) -> None:
        # The Owl Mount's foot plate sits inside the lead at the default rest
        # pose.
        solver = KinematicsSolver(KinematicsConfig(robot_model="owl"))
        self.assertGreater(solver.collision_proximity(self.rest)["left"], 0.0)
        solver.set_proximity_reference(self.rest)
        self.assertEqual(
            solver.collision_proximity(self.rest), {"left": 0.0, "right": 0.0}
        )
        self.assertGreater(
            solver.collision_proximity(self._blocked(solver))["left"], 0.9
        )

    def test_none_without_collision_detection(self) -> None:
        solver = KinematicsSolver(KinematicsConfig(self_collision=False))
        self.assertIsNone(solver.collision_proximity(self.rest))

    def test_mink_backend(self) -> None:
        try:
            from almond_axol.kinematics.mink_backend import MinkKinematicsSolver
        except ImportError:
            self.skipTest("mink not installed")
        solver = MinkKinematicsSolver(KinematicsConfig(backend="mink"))
        self.assertEqual(
            solver.collision_proximity(np.zeros(14, np.float32)),
            {"left": 0.0, "right": 0.0},
        )
        solver.set_proximity_reference(self.rest)
        self.assertEqual(
            solver.collision_proximity(self.rest), {"left": 0.0, "right": 0.0}
        )
        levels = solver.collision_proximity(self._blocked(solver))
        self.assertEqual(levels["left"], 1.0)
        self.assertLess(levels["right"], 1.0)


if __name__ == "__main__":
    unittest.main()
