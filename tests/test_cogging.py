"""Cogging ("osc") cancellation: the Fourier fit from a friction sweep, its
calibration-file round trip, and the motor-frame series the core evaluates."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.robot.calibration import (
    clean_cogging,
    load_calibration,
    update_joint_calibration,
)
from almond_axol.robot.config import (
    CoggingModel,
    FrictionParams,
    JointConfig,
    _calibrated_joint,
)
from almond_axol.tuning.cogging import angle_highpass, fit_cogging, prediction_r


def _sweep(model: CoggingModel, seed: int, noise: float = 0.2):
    """A bidirectional constant-speed sweep over -50..-20°: gravity (a slow
    sine), Coulomb friction flipping with direction, the ripple, noise."""
    rng = np.random.default_rng(seed)
    q_fwd = np.radians(np.linspace(-50, -20, 3000))
    q = np.concatenate([q_fwd, q_fwd[::-1]])
    direction = np.array(["+"] * len(q_fwd) + ["-"] * len(q_fwd))
    gravity = -14.0 * np.cos(q)
    friction = np.where(direction == "+", 1.3, -1.3)
    tau = gravity + friction + model.torque(q) + noise * rng.standard_normal(len(q))
    return q, tau, direction


class CoggingModelTest(unittest.TestCase):
    MODEL = CoggingModel(3.62, ((1, 0.03, -0.02), (2, 0.45, -0.25), (4, 0.05, 0.1)))

    def test_motor_terms_reproduce_the_joint_series_at_joint_equals_motor_plus_offset(
        self,
    ) -> None:
        for offset in (0.0, 0.37, -1.2):
            terms = self.MODEL.motor_terms(offset, gain=0.8)
            for q_motor in np.linspace(-1.0, 1.0, 7):
                got = sum(
                    a * math.cos(w * q_motor) + b * math.sin(w * q_motor)
                    for w, a, b in terms
                )
                self.assertAlmostEqual(
                    got, 0.8 * float(self.MODEL.torque(q_motor + offset)), places=10
                )

    def test_dict_round_trip_and_calibration_overlay(self) -> None:
        self.assertEqual(CoggingModel.from_dict(self.MODEL.as_dict()), self.MODEL)
        jc = JointConfig(
            kp=1.0,
            kd=0.1,
            friction=FrictionParams(fc=0.0, k=0.0, fv=0.0, fo=0.0),
            mass=1.0,
            com=(0.0, 0.0, 0.0),
        )
        got = _calibrated_joint(jc, {"cogging": self.MODEL.as_dict()})
        self.assertEqual(got.cogging, self.MODEL)
        self.assertEqual(got.cogging_gain, 1.0)


class FitTest(unittest.TestCase):
    TRUE = CoggingModel(3.62, ((1, 0.02, 0.0), (2, 0.5, -0.2), (4, 0.0, 0.1)))

    def test_the_fit_recovers_the_ripple_through_gravity_and_friction(self) -> None:
        fit = fit_cogging(*_sweep(self.TRUE, 0))
        for (k, a, b), (k0, a0, b0) in zip(fit.model.harmonics, self.TRUE.harmonics):
            self.assertEqual(k, k0)
            self.assertAlmostEqual(a, a0, delta=0.03)
            self.assertAlmostEqual(b, b0, delta=0.03)
        self.assertGreater(fit.r2, 0.7)
        # Another pass (other noise) is predicted out of sample.
        self.assertGreater(prediction_r(fit.model, *_sweep(self.TRUE, 1)), 0.8)

    def test_pure_noise_predicts_nothing(self) -> None:
        empty = CoggingModel(3.62, ((1, 0.0, 0.0),))
        fit = fit_cogging(*_sweep(empty, 2))
        self.assertLess(fit.r2, 0.05)
        self.assertLess(abs(prediction_r(fit.model, *_sweep(empty, 3))), 0.1)

    def test_too_little_travel_is_refused(self) -> None:
        q = np.radians(np.linspace(0, 5, 200))
        with self.assertRaisesRegex(ValueError, "too little travel"):
            fit_cogging(q, np.zeros_like(q), np.array(["+"] * len(q)))
        self.assertEqual(len(angle_highpass(np.array([1.0]), np.array([1.0]))[0]), 0)


class CalibrationFileTest(unittest.TestCase):
    def test_a_saved_series_loads_back_and_a_bad_one_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.json"
            update_joint_calibration(
                "right",
                "shoulder_1",
                cogging=FitTest.TRUE.as_dict(),
                hub_serial="hub1",
                path=path,
            )
            got = load_calibration(path, expected_hub_serial="hub1")
            self.assertEqual(
                CoggingModel.from_dict(got["right"]["shoulder_1"]["cogging"]),
                FitTest.TRUE,
            )
            raw = json.loads(path.read_text())
            raw["right"]["elbow"] = {"cogging": {"period_deg": -1, "harmonics": []}}
            path.write_text(json.dumps(raw))
            got = load_calibration(path, expected_hub_serial="hub1")
            self.assertNotIn("elbow", got["right"])
            with self.assertRaises(ValueError):
                update_joint_calibration(
                    "right",
                    "elbow",
                    cogging={"period_deg": 3.6, "harmonics": [[0, 1, 1]]},
                    hub_serial="hub1",
                    path=path,
                )

    def test_clean_cogging_rejects_malformed_series(self) -> None:
        good = {"period_deg": 3.62, "harmonics": [[1, 0.1, 0.2], [2.0, -0.3, 0.0]]}
        self.assertEqual(
            clean_cogging(good),
            {"period_deg": 3.62, "harmonics": [[1, 0.1, 0.2], [2, -0.3, 0.0]]},
        )
        for bad in (
            None,
            {"period_deg": 0, "harmonics": [[1, 0, 0]]},
            {"period_deg": 3.6, "harmonics": []},
            {"period_deg": 3.6, "harmonics": [[1.5, 0, 0]]},
            {"period_deg": 3.6, "harmonics": [[1, float("nan"), 0]]},
            {"period_deg": 3.6, "harmonics": [[float("inf"), 0, 0]]},
            {"period_deg": 3.6, "harmonics": [[1, 0]]},
        ):
            with self.subTest(bad=bad):
                self.assertIsNone(clean_cogging(bad))


if __name__ == "__main__":
    unittest.main()
