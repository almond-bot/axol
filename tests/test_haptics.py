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
from almond_axol.teleop import core as core_module
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.teleop.live import LiveSettings


def _core(sent: list | None = None, **config) -> VRTeleopCore:
    return VRTeleopCore(
        VRTeleopConfig(**config),
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

    def test_apply_routes_to_the_core(self) -> None:
        core = _core()
        live = LiveSettings(core, _Robot(), lambda s: None)
        live.apply("haptics", "toggle")
        live.apply("haptic_torque", 9.0)
        core._apply_live_requests()
        self.assertFalse(core.config.haptics)
        self.assertEqual(core.config.haptic_torque, 9.0)


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
        for q in (np.zeros(14, np.float32), self.rest):
            self.assertEqual(
                self.solver.collision_proximity(q), {"left": 0.0, "right": 0.0}
            )

    def test_full_when_ik_holds_the_arm_off_the_body(self) -> None:
        levels = self.solver.collision_proximity(self._blocked(self.solver))
        self.assertEqual(levels["left"], 1.0)
        self.assertLess(levels["right"], 0.1)

    def test_rest_reads_clear_on_a_mount_close_to_the_arms(self) -> None:
        # The Owl Mount's feet sit inside the lead at the default rest pose.
        solver = KinematicsSolver(KinematicsConfig(robot_model="owl"))
        self.assertEqual(solver.collision_proximity(self.rest)["left"], 1.0)
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
