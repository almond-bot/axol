"""Stiction compensation: the control math, its config plumbing, and the
realtime-core config line that carries it.

The Rust core pins ``filter::stiction`` / ``filter::coulomb_unit`` to the
vectors in ``stiction_matches_python`` (rust/axol-rt/src/filter.rs); the
golden table here is the same one, so a change to either side breaks both.
"""

from __future__ import annotations

import math
import unittest

from almond_axol.robot.config import (
    ArmConfig,
    AxolConfig,
    JointConfig,
    _calibrated_joint,
)
from almond_axol.robot.control import (
    DITHER_PHASE_STAGGER,
    STICTION_FADE_VEL,
    TorqueDither,
    compute_friction,
    coulomb_unit,
    dither_step,
    stiction_amplitude,
    stiction_compensation,
    stribeck_amplitude,
    stribeck_excess,
)

_SCALE = math.radians(0.1)

# (err rad, v_meas rad/s, want Nm) for amp = 0.36 (fc = 0.6, gain = 0.6).
_GOLDEN = [
    (-0.02, 0.0, -0.3599999999198255),
    (-0.002, 0.0, -0.29390273095808195),
    (-0.0005, 0.0, -0.10040068111546155),
    (0.0, 0.0, 0.0),
    (0.0005, 0.0, 0.10040068111546155),
    (0.002, 0.0, 0.29390273095808195),
    (0.02, 0.0, 0.3599999999198255),
    (0.02, 0.022, 0.1349638509268763),
    (0.02, -0.044, 0.03638171680139523),
    (0.02, 0.15, 3.268646545848154e-05),
]

# Dither at 1.5 Nm / 60 Hz stepped at 240 Hz: first four outputs of slot 0
# and slot 1 (one golden angle further round).
_DITHER_GOLDEN = {
    0: [1.5, 8.498308346471969e-16, -1.5, -1.6996616692943939e-15],
    1: [
        -1.1060533171174791,
        -1.0132354413922864,
        1.106053317117479,
        1.0132354413922875,
    ],
}


class StictionMathTest(unittest.TestCase):
    def test_golden_vectors_shared_with_the_core(self) -> None:
        for err, v, want in _GOLDEN:
            amp = stiction_amplitude(0.6, 0.6, 0.0, 0.0)
            got = stiction_compensation(err, v, amp, _SCALE)
            self.assertAlmostEqual(got, want, places=12, msg=f"err={err} v={v}")

    def test_dither_golden_vectors_shared_with_the_core(self) -> None:
        self.assertAlmostEqual(DITHER_PHASE_STAGGER, 2.399963229728653, places=15)
        for slot, want in _DITHER_GOLDEN.items():
            phase = slot * DITHER_PHASE_STAGGER
            for k, w in enumerate(want):
                phase, torque = dither_step(phase, 1.5, 60.0, 1.0 / 240.0)
                self.assertAlmostEqual(torque, w, places=12, msg=f"slot {slot} k {k}")
        # Off is exactly off, and does not run the phase.
        self.assertEqual(dither_step(1.0, 0.0, 60.0, 0.01), (1.0, 0.0))
        dither = TorqueDither(2)
        self.assertEqual(dither.update([0.0, 0.0], [60.0, 60.0]), [0.0, 0.0])

    def test_zero_gain_is_exactly_the_production_law(self) -> None:
        self.assertEqual(stiction_amplitude(0.6, 0.0, 0.0, 12.0), 0.0)
        self.assertEqual(stiction_compensation(0.02, 0.0, 0.0, _SCALE), 0.0)

    def test_amplitude_follows_the_gravity_load(self) -> None:
        # right shoulder_1: 0.39 Nm at rest, ~2.8 Nm under 12 Nm of gravity.
        self.assertAlmostEqual(stiction_amplitude(1.3, 0.3, 0.2, 0.0), 0.39)
        self.assertAlmostEqual(stiction_amplitude(1.3, 0.3, 0.2, -12.0), 2.79)
        self.assertAlmostEqual(stiction_amplitude(1.3, 0.3, 0.2, 12.0), 2.79)

    def test_pushes_toward_the_target_and_saturates_at_gain_fc(self) -> None:
        amp = stiction_amplitude(1.3, 0.6, 0.0, 0.0)
        big = stiction_compensation(0.1, 0.0, amp, _SCALE)
        self.assertAlmostEqual(big, 0.6 * 1.3, places=6)
        self.assertLess(stiction_compensation(-0.1, 0.0, amp, _SCALE), 0.0)
        self.assertEqual(stiction_compensation(0.1, 0.0, 0.0, _SCALE), 0.0)
        # Within the scale the push is proportional: a stiffer-than-kp spring.
        small = stiction_compensation(_SCALE / 10, 0.0, amp, _SCALE)
        self.assertAlmostEqual(small, 0.6 * 1.3 * math.tanh(0.1), places=9)

    def test_fades_on_measured_velocity_not_commanded(self) -> None:
        # The push is for a *stuck* joint: full at rest, gone once the joint
        # actually slides — whatever the command is doing.
        amp = stiction_amplitude(1.3, 0.6, 0.0, 0.0)
        stuck = stiction_compensation(0.02, 0.0, amp, _SCALE)
        one_lsb = stiction_compensation(0.02, 0.022, amp, _SCALE)
        sliding = stiction_compensation(0.02, 0.15, amp, _SCALE)
        self.assertAlmostEqual(stuck, 0.78, places=6)
        self.assertLess(one_lsb, 0.5 * stuck)
        self.assertLess(sliding, 0.001)
        self.assertEqual(STICTION_FADE_VEL, 0.03)
        total = compute_friction(0.3, 1.3, 250.0, 0.0, 0.0) + sliding
        self.assertLessEqual(total, (1 + 0.6) * 1.3 + 1e-9)

    def test_coulomb_unit_matches_compute_friction(self) -> None:
        for v in (-1.0, -0.05, 0.0, 0.02, 0.4):
            self.assertAlmostEqual(
                compute_friction(v, 0.6, 250.0, 0.15, 0.02),
                0.6 * coulomb_unit(v, 250.0) + 0.15 * v + 0.02,
                places=12,
            )


# (v_meas rad/s, want) for amp = 1, v_s = 0.1, v0 = 0.02.
_STRIBECK_GOLDEN = [
    (-0.3, -0.00012340980408665668),
    (-0.1, -0.36784603928630505),
    (-0.05, -0.7683759879897785),
    (-0.02, -0.7317316219624262),
    (0.0, 0.0),
    (0.01, 0.4575190147179108),
    (0.02, 0.7317316219624262),
    (0.05, 0.7683759879897785),
    (0.1, 0.36784603928630505),
    (0.2, 0.018315638813231488),
]


class StribeckTest(unittest.TestCase):
    def test_golden_vectors_shared_with_the_core(self) -> None:
        for v, want in _STRIBECK_GOLDEN:
            self.assertAlmostEqual(
                stribeck_excess(v, 1.0, 0.1), want, places=12, msg=f"v={v}"
            )

    def test_zero_at_rest_and_off_by_default(self) -> None:
        self.assertEqual(stribeck_excess(0.0, 1.0, 0.1), 0.0)
        self.assertEqual(stribeck_excess(0.05, 0.0, 0.1), 0.0)
        self.assertEqual(stribeck_amplitude(0.0, 0.3, 0.1, 12.0), 0.0)

    def test_has_the_measured_shape(self) -> None:
        # Excess peaks in the creep band and is gone by 3·v_s; the slope
        # between 0.05 and 0.2 rad/s is negative, like the measured curve.
        amp = stribeck_amplitude(1.0, 0.3, 0.1, 12.0)  # 1.5 Nm under 12 Nm of load
        self.assertAlmostEqual(amp, 1.5)
        creep = stribeck_excess(0.05, amp, 0.1)
        fast = stribeck_excess(0.2, amp, 0.1)
        self.assertGreater(creep, 1.0)
        self.assertLess(fast, 0.05)
        self.assertLess(stribeck_excess(-0.05, amp, 0.1), 0.0)


class LoadFrictionTest(unittest.TestCase):
    def test_fl_defaults_to_zero_and_loads_from_calibration(self) -> None:
        from almond_axol.robot.config import FrictionParams

        self.assertEqual(FrictionParams(fc=0.6, k=100.0, fv=0.0, fo=0.0).fl, 0.0)
        out = _calibrated_joint(
            ArmConfig().shoulder_1,
            {"friction": {"fc": 0.6, "k": 100.0, "fv": 0.0, "fo": 0.0, "fl": 0.08}},
        )
        self.assertEqual((out.friction.fc, out.friction.fl), (0.6, 0.08))
        # An older calibration file without fl still loads.
        out = _calibrated_joint(
            ArmConfig().shoulder_1,
            {"friction": {"fc": 0.6, "k": 100.0, "fv": 0.0, "fo": 0.0}},
        )
        self.assertEqual(out.friction.fl, 0.0)


class StictionConfigTest(unittest.TestCase):
    def test_defaults_are_off_on_every_joint(self) -> None:
        arm = ArmConfig()
        for name in (
            "shoulder_1",
            "shoulder_2",
            "shoulder_3",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ):
            jc: JointConfig = getattr(arm, name)
            self.assertEqual(jc.stiction_gain, 0.0, name)
            self.assertEqual(jc.stiction_err_deg, 0.1, name)
            self.assertEqual(jc.dither_nm, 0.0, name)
            self.assertEqual(jc.wire_mode, "mit", name)
            self.assertEqual(jc.stribeck_gain, 0.0, name)

    def test_calibration_file_can_set_the_fields(self) -> None:
        base = ArmConfig().shoulder_1
        out = _calibrated_joint(base, {"stiction_gain": 0.5, "stiction_err_deg": 0.2})
        self.assertEqual((out.stiction_gain, out.stiction_err_deg), (0.5, 0.2))
        untouched = _calibrated_joint(base, {"kp": 200.0})
        self.assertEqual(untouched.stiction_gain, 0.0)

    def test_resolved_config_keeps_the_fields(self) -> None:
        cfg = AxolConfig()
        cfg.right.shoulder_1.stiction_gain = 0.6
        resolved = cfg.resolved()
        self.assertEqual(resolved.right.shoulder_1.stiction_gain, 0.6)
        self.assertEqual(resolved.left.shoulder_1.stiction_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
