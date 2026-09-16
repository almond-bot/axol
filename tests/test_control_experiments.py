"""The opt-in tracking experiments (``AxolConfig.experiments``).

Every knob defaults to the production control law; each is reachable as a
``--axol.experiments.<name>`` flag; the values ride the realtime core's
config as ``exp`` lines; and the Python math the core's ``filter.rs`` is
golden-tested against behaves as its docstrings promise.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import fields

import numpy as np

from almond_axol.constants import ARM_JOINTS, Joint
from almond_axol.robot.config import AxolConfig, ControlExperiments
from almond_axol.robot.control import (
    FRICTION_FF_K_MAX,
    ErrorIntegrator,
    SlewLimiter,
    compute_friction,
    coulomb_unit,
    stiction_compensation,
)

DT = 1.0 / 240.0


class ExperimentsConfigTest(unittest.TestCase):
    def test_defaults_are_the_production_law(self) -> None:
        exp = ControlExperiments()
        self.assertTrue(exp.is_default())
        self.assertEqual(exp.friction_k_max, FRICTION_FF_K_MAX)
        self.assertEqual(exp.friction_slew, 0.0)
        self.assertEqual(exp.stiction_gain, 0.0)
        self.assertEqual(exp.integrator_hz, 0.0)
        self.assertFalse(exp.tracker_wire_vel)
        self.assertFalse(exp.tracker_accel_ff)
        self.assertTrue(AxolConfig().experiments.is_default())
        self.assertFalse(ControlExperiments(integrator_hz=0.3).is_default())

    def test_config_lines_declare_every_field(self) -> None:
        exp = ControlExperiments(friction_slew=30.0, tracker_wire_vel=True)
        lines = exp.config_lines()
        self.assertEqual(len(lines), len(fields(ControlExperiments)))
        names = [line.split()[1] for line in lines]
        self.assertEqual(names, [f.name for f in fields(ControlExperiments)])
        self.assertIn("exp friction_slew 30.0", lines)
        # Booleans go over the wire as 0/1 (the core parses every value as
        # a float).
        self.assertIn("exp tracker_wire_vel 1", lines)
        self.assertIn("exp tracker_accel_ff 0", lines)
        for line in lines:
            float(line.split()[2])

    def test_flags_reach_the_config(self) -> None:
        from almond_axol.cli.config import TeleopCmdConfig, parse

        cfg = parse(
            TeleopCmdConfig,
            [
                "--axol.experiments.friction_k_max",
                "400",
                "--axol.experiments.friction_slew",
                "30",
                "--axol.experiments.stiction_gain",
                "0.6",
                "--axol.experiments.integrator_hz",
                "0.3",
                "--axol.experiments.tracker_wire_vel",
                "true",
            ],
            settings_op=None,
        )
        exp = cfg.axol.experiments
        self.assertEqual(exp.friction_k_max, 400.0)
        self.assertEqual(exp.friction_slew, 30.0)
        self.assertEqual(exp.stiction_gain, 0.6)
        self.assertEqual(exp.integrator_hz, 0.3)
        self.assertTrue(exp.tracker_wire_vel)
        # Untouched fields keep their defaults; the per-joint config is
        # unaffected by the experiments block.
        self.assertFalse(exp.tracker_accel_ff)
        self.assertEqual(cfg.axol.left.elbow.kp, AxolConfig().left.elbow.kp)

    def test_settings_file_reaches_the_config(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from almond_axol.cli.config import TeleopCmdConfig, parse

        # The robot's shared ``~/.almond/settings.json`` has the shape of the
        # config dataclasses, so an ``experiments`` block persists a chosen
        # set per robot (``--settings_path`` reads an alternate file).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "axol": {
                            "experiments": {
                                "integrator_hz": 0.3,
                                "tracker_wire_vel": True,
                            }
                        }
                    }
                )
            )
            cfg = parse(
                TeleopCmdConfig, ["--settings_path", str(path)], settings_op="teleop"
            )
        exp = cfg.axol.experiments
        self.assertEqual(exp.integrator_hz, 0.3)
        self.assertTrue(exp.tracker_wire_vel)
        self.assertEqual(exp.friction_slew, 0.0)


class FrictionCapTest(unittest.TestCase):
    def test_default_cap_is_unchanged(self) -> None:
        for v in (-1.5, -0.2, -0.01, 0.0, 0.01, 0.2, 1.5):
            self.assertEqual(
                compute_friction(v, 0.6, 250.0, 0.15, 0.02),
                compute_friction(v, 0.6, 250.0, 0.15, 0.02, FRICTION_FF_K_MAX),
            )

    def test_raised_cap_saturates_at_lower_speed(self) -> None:
        # At the production cap the Coulomb term is ~20% of fc at 0.02 rad/s
        # and ~46% at 0.05 (the slow-motion deadband); at a cap of 400 it
        # is ~66% and ~96%.
        self.assertAlmostEqual(coulomb_unit(0.02, 800.0), math.tanh(0.2), places=12)
        self.assertAlmostEqual(
            coulomb_unit(0.02, 800.0, 400.0), math.tanh(0.8), places=12
        )
        self.assertGreater(coulomb_unit(0.05, 800.0, 400.0), 0.95)
        # A fitted k below the cap is used as-is.
        self.assertEqual(coulomb_unit(0.02, 50.0, 400.0), math.tanh(0.1))


class SlewLimiterTest(unittest.TestCase):
    def test_rate_limits_and_passes_through_when_off(self) -> None:
        sl = SlewLimiter(2)
        self.assertEqual(sl.update([1.0, -1.0], 30.0, DT), [1.0, -1.0])  # adopts
        out = sl.update([-1.0, 1.0], 30.0, DT)
        step = 30.0 * DT
        self.assertAlmostEqual(out[0], 1.0 - step)
        self.assertAlmostEqual(out[1], -1.0 + step)
        # Small moves within the budget pass unchanged.
        y = sl.update([out[0] + 0.01, out[1]], 30.0, DT)
        self.assertAlmostEqual(y[0], out[0] + 0.01)
        # rate 0 is a pass-through.
        self.assertEqual(sl.update([5.0, 5.0], 0.0, DT), [5.0, 5.0])
        sl.reset()
        self.assertEqual(sl.update([-3.0, 2.0], 30.0, DT), [-3.0, 2.0])


class StictionTest(unittest.TestCase):
    def test_pushes_toward_target_and_saturates(self) -> None:
        scale = math.radians(0.1)
        self.assertAlmostEqual(
            stiction_compensation(0.01, 0.0, 1.3, 0.6, scale),
            0.6 * 1.3 * math.tanh(0.01 / scale),
        )
        self.assertLess(stiction_compensation(-0.01, 0.0, 1.3, 0.6, scale), 0.0)
        self.assertEqual(stiction_compensation(0.0, 0.0, 1.3, 0.6, scale), 0.0)
        # Never more than gain·fc on its own.
        self.assertLessEqual(
            abs(stiction_compensation(1.0, 0.0, 1.3, 0.6, scale)), 0.6 * 1.3
        )

    def test_fades_as_the_velocity_feedforward_saturates(self) -> None:
        scale = math.radians(0.1)
        stuck = stiction_compensation(0.01, 0.0, 1.3, 0.6, scale)
        moving = stiction_compensation(0.01, coulomb_unit(0.05, 800.0), 1.3, 0.6, scale)
        fast = stiction_compensation(0.01, coulomb_unit(0.5, 800.0), 1.3, 0.6, scale)
        self.assertGreater(stuck, moving)
        self.assertGreater(moving, fast)
        self.assertLess(fast, 1e-3)

    def test_off_by_default(self) -> None:
        self.assertEqual(stiction_compensation(0.01, 0.0, 1.3, 0.0, 1e-3), 0.0)
        self.assertEqual(stiction_compensation(0.01, 0.0, 0.0, 0.6, 1e-3), 0.0)


class ErrorIntegratorTest(unittest.TestCase):
    def test_winds_clamps_freezes_and_resets(self) -> None:
        ki = 250.0 * 2 * math.pi * 0.3  # kp=250 at a 0.3 Hz crossover
        it = ErrorIntegrator(1)
        freeze = math.radians(2.0)
        out = it.update([0.004], [ki], [2.6], freeze, DT)[0]
        self.assertAlmostEqual(out, ki * 0.004 * DT)
        # A move-sized error holds the state rather than winding up.
        held = it.update([0.06], [ki], [2.6], freeze, DT)[0]
        self.assertEqual(held, out)
        # No fresh feedback (dt=0) also holds.
        self.assertEqual(it.update([0.004], [ki], [2.6], freeze, 0.0)[0], out)
        # Anti-windup clamp on the output.
        for _ in range(10_000):
            it.update([0.01], [ki], [0.5], freeze, DT)
        self.assertEqual(it.update([0.01], [ki], [0.5], freeze, DT)[0], 0.5)
        # Reversed error unwinds; reset zeroes; ki=0 zeroes.
        self.assertLess(it.update([-0.01], [ki], [0.5], freeze, DT)[0], 0.5)
        it.reset()
        self.assertEqual(it.update([0.0], [ki], [0.5], freeze, DT)[0], 0.0)
        it.update([0.01], [ki], [0.5], freeze, DT)
        self.assertEqual(it.update([0.01], [0.0], [0.5], freeze, DT)[0], 0.0)

    def test_crossover_means_integral_equals_proportional(self) -> None:
        # At the crossover frequency the integral of a sinusoidal error has
        # the same amplitude as kp·e: |ki/(jω)| = kp at ω = 2π·f.
        kp, f_hz = 250.0, 0.3
        ki = kp * 2 * math.pi * f_hz
        self.assertAlmostEqual(ki / (2 * math.pi * f_hz), kp)


class ClassicPathTest(unittest.IsolatedAsyncioTestCase):
    """The classic (non-core) motion_control path applies the experiments.

    With the defaults the shipped torque is the old law; with an integrator
    on, a held position error steadily pulls the feedforward toward the
    target. Uses the bench wrist kit so no fixed-stop zero verification is
    involved (the same fixture as ``test_rom_partial_arm``).
    """

    def _arm(self, config: AxolConfig):
        from unittest.mock import patch

        from almond_axol.motor import ControlMode
        from almond_axol.robot.axol import AxolHardware
        from tests.test_rom_partial_arm import WRIST_KIT, _FakeBus, _FakeDriver

        with (
            patch("almond_axol.robot.axol.CanBus", _FakeBus),
            patch(
                "almond_axol.motor.motor.make_driver",
                side_effect=lambda *_a, **_k: _FakeDriver(),
            ),
        ):
            hw = AxolHardware(
                config,
                left_channel="can0",
                right_channel=None,
                left_joints=set(WRIST_KIT),
            )
        arm = hw.left
        assert arm is not None
        arm._unresolved_offsets.clear()
        arm._joint_offsets[:] = 0.0
        for joint, motor in arm.motors.items():
            # Feedback cache: every joint reads exactly the rest pose, so a
            # non-zero target is a held position error.
            motor._position = 0.0
            motor._torque = 0.0
            motor._feedback_ts = 1.0
            motor.mode = (
                ControlMode.POSITION_FORCE
                if joint == Joint.GRIPPER
                else ControlMode.IMPEDANCE
            )
        return arm

    async def _drive(self, arm, q, cycles: int) -> list[tuple[float, ...]]:
        import asyncio

        for _ in range(cycles):
            await arm.motion_control(q)
            await asyncio.sleep(0.002)
        return arm.motors[Joint.WRIST_2]._driver.impedance

    async def test_defaults_match_the_production_law(self) -> None:
        q = np.zeros(8, dtype=np.float32)
        q[ARM_JOINTS.index(Joint.WRIST_2)] = 0.004
        arm = self._arm(AxolConfig())
        sent = await self._drive(arm, q, 3)
        # Gravity at the commanded pose plus the velocity friction FF (≈ fo
        # at ~zero velocity); neither position-error term is active.
        jc = arm._arm_config.wrist_2
        p_des, v_des, kp, kd, t_ff = sent[-1]
        self.assertAlmostEqual(p_des, 0.004, places=5)
        self.assertEqual((kp, kd), (jc.kp, jc.kd))
        self.assertAlmostEqual(
            t_ff,
            compute_friction(
                v_des, jc.friction.fc, jc.friction.k, jc.friction.fv, jc.friction.fo
            )
            + arm._gravity_comp.gravity_arm(q[:7], is_left=True)[
                ARM_JOINTS.index(Joint.WRIST_2)
            ],
            places=6,
        )

    async def test_integrator_pulls_toward_a_held_error(self) -> None:
        q = np.zeros(8, dtype=np.float32)
        q[ARM_JOINTS.index(Joint.WRIST_2)] = 0.004
        base = await self._drive(self._arm(AxolConfig()), q, 8)
        integ = await self._drive(
            self._arm(AxolConfig(experiments=ControlExperiments(integrator_hz=0.3))),
            q,
            8,
        )
        # Same target/gains; the torque grows with the integrator on, and
        # keeps growing while the error is held.
        self.assertEqual(base[-1][:4], integ[-1][:4])
        self.assertGreater(integ[-1][4], base[-1][4])
        self.assertGreater(integ[-1][4] - base[-1][4], integ[3][4] - base[3][4])

    async def test_stiction_acts_on_the_error_sign(self) -> None:
        exp = ControlExperiments(stiction_gain=0.6)
        q = np.zeros(8, dtype=np.float32)
        i = ARM_JOINTS.index(Joint.WRIST_2)
        q[i] = 0.01
        plus = (await self._drive(self._arm(AxolConfig(experiments=exp)), q, 2))[-1][4]
        base = (await self._drive(self._arm(AxolConfig()), q, 2))[-1][4]
        q[i] = -0.01
        minus = (await self._drive(self._arm(AxolConfig(experiments=exp)), q, 2))[-1][4]
        base_minus = (await self._drive(self._arm(AxolConfig()), q, 2))[-1][4]
        fc = AxolConfig().left.wrist_2.friction.fc
        self.assertAlmostEqual(plus - base, 0.6 * fc, places=3)
        self.assertAlmostEqual(minus - base_minus, -0.6 * fc, places=3)


if __name__ == "__main__":
    unittest.main()
