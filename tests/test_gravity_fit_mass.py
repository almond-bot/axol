"""The gravity fitter must be able to say "this link is heavier", not only
"this link's CoM is elsewhere".

Bench, right elbow: the link weighs 0.25 kg in the model and its sweep demanded
a 93 mm CoM shift, which the plausibility cap rejected -- 0.23 Nm of torque the
light link can only produce with an absurd lever, i.e. ~78 g of unmodelled
mass at the hand. shoulder_2's fit was then rejected downstream of it.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.cli.tune import gravity as G
from almond_axol.constants import Joint
from almond_axol.robot import calibration as C
from almond_axol.robot.config import AxolConfig


def _sweep(joint: Joint, cfg: AxolConfig, lo: float, hi: float, n: int = 40):
    q = np.linspace(lo, hi, n)
    tau = G._model_torques(cfg, joint, False, q, {})
    return q, tau


def _lever_and_perp(cfg: AxolConfig, joint: Joint, q: np.ndarray):
    """In-plane lever direction and a unit vector across it, found from the
    torque sensitivity columns so the test is not hard-coded to one joint's
    axis (the elbow rotates about link-frame x: com_x is unobservable)."""
    com0 = np.array(getattr(cfg.right, joint.value).com)
    sens = []
    for ax in range(3):
        s = np.zeros(3)
        s[ax] = 0.005
        hi = G._model_torques(
            G._with_com(cfg, False, joint, tuple(com0 + s)), joint, False, q, {}
        )
        lo = G._model_torques(
            G._with_com(cfg, False, joint, tuple(com0 - s)), joint, False, q, {}
        )
        sens.append(np.linalg.norm(hi - lo))
    observable = np.array(sens) > 1e-9
    u = np.where(observable, com0, 0.0)
    u = u / np.linalg.norm(u)
    axis = np.eye(3)[int(np.argmin(sens))]
    perp = np.cross(axis, u)
    return com0, u, perp / np.linalg.norm(perp)


class MassColumnTest(unittest.TestCase):
    def test_a_heavier_link_is_fitted_as_mass_not_as_a_lever(self) -> None:
        cfg = AxolConfig()
        heavier = G._with_mass(cfg, False, Joint.ELBOW, cfg.right.elbow.mass * 1.08)
        q, tau = _sweep(Joint.ELBOW, heavier, math.radians(-147), math.radians(-3))
        fit = G.fit_com(q, tau, Joint.ELBOW, False, {})
        self.assertIsNotNone(fit)
        com_fit, offset, _, tau_after, mass_fit = fit
        self.assertAlmostEqual(mass_fit / cfg.right.elbow.mass, 1.08, delta=0.01)
        shift = np.linalg.norm(np.array(com_fit) - np.array(cfg.right.elbow.com))
        self.assertLess(shift, 0.003)  # not the 93 mm the CoM-only fit produced
        self.assertLess(float(np.abs(tau - tau_after - offset).max()), 0.01)

    def test_a_com_moved_across_the_lever_is_still_fitted_as_com(self) -> None:
        """Across the lever is the one direction a single sweep can tell apart
        from mass, so it must land on the CoM. The step also lengthens the
        lever a little (sqrt(r^2 + d^2)); that along-lever part is booked as
        mass by design, so the mass expectation is the analytic ratio."""
        cfg = AxolConfig()
        q0 = np.linspace(math.radians(-147), math.radians(-3), 40)
        com0, u, perp = _lever_and_perp(cfg, Joint.ELBOW, q0)
        step = 0.02 * perp
        moved = G._with_com(cfg, False, Joint.ELBOW, tuple(com0 + step))
        q, tau = _sweep(Joint.ELBOW, moved, math.radians(-147), math.radians(-3))
        com_fit, _, _, _, mass_fit = G.fit_com(q, tau, Joint.ELBOW, False, {})
        got = np.array(com_fit) - com0
        # Direction across the lever is recovered on the CoM...
        self.assertAlmostEqual(float(got @ perp), 0.02, delta=0.003)
        self.assertLess(abs(float(got @ u)), 0.003)
        # ...and the lever's growth in magnitude is recovered as mass.
        r_plane = np.linalg.norm(np.where(np.abs(u) > 1e-9, com0, 0.0))
        expected = math.hypot(r_plane, 0.02) / r_plane
        self.assertAlmostEqual(mass_fit / cfg.right.elbow.mass, expected, delta=0.015)

    def test_a_com_moved_along_the_lever_is_attributed_to_mass(self) -> None:
        """Along the lever, mass and CoM are indistinguishable; by design the
        fitter books it as mass (a light link does not move its CoM 93 mm, a
        harness downstream does add mass)."""
        cfg = AxolConfig()
        q0 = np.linspace(math.radians(-147), math.radians(-3), 40)
        com0, u, _ = _lever_and_perp(cfg, Joint.ELBOW, q0)
        moved = G._with_com(cfg, False, Joint.ELBOW, tuple(com0 + 0.02 * u))
        q, tau = _sweep(Joint.ELBOW, moved, math.radians(-147), math.radians(-3))
        com_fit, offset, _, tau_after, mass_fit = G.fit_com(
            q, tau, Joint.ELBOW, False, {}
        )
        self.assertLess(np.linalg.norm(np.array(com_fit) - com0), 0.003)
        r_plane = np.linalg.norm(np.where(np.abs(u) > 1e-9, com0, 0.0))
        self.assertAlmostEqual(
            mass_fit / cfg.right.elbow.mass, 1.0 + 0.02 / r_plane, delta=0.02
        )
        self.assertLess(float(np.abs(tau - tau_after - offset).max()), 0.01)

    def test_an_implausible_mass_is_refused_with_a_message_that_says_mass(self) -> None:
        cfg = AxolConfig()
        heavy = G._with_mass(cfg, False, Joint.ELBOW, cfg.right.elbow.mass * 3.0)
        q, tau = _sweep(Joint.ELBOW, heavy, math.radians(-147), math.radians(-3))
        with self.assertRaisesRegex(RuntimeError, "mass"):
            G.fit_com(q, tau, Joint.ELBOW, False, {})


class MassPersistenceTest(unittest.TestCase):
    def test_mass_round_trips_through_the_calibration_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calibration.json"
            C.update_joint_calibration(
                "right",
                "elbow",
                com=(0.02, 0.0, -0.07),
                mass=0.27,
                hub_serial="TEST",
                path=p,
            )
            raw = json.loads(p.read_text())
            self.assertEqual(raw["right"]["elbow"]["mass"], 0.27)
            loaded = C.load_calibration(p, expected_hub_serial="TEST")
            self.assertEqual(loaded["right"]["elbow"]["mass"], 0.27)

    def test_overlay_applies_the_fitted_mass(self) -> None:
        from almond_axol.robot.config import _calibrated_joint

        cfg = AxolConfig()
        jc = _calibrated_joint(
            cfg.right.elbow, {"mass": 0.27, "com": [0.02, 0.0, -0.07]}
        )
        self.assertEqual(jc.mass, 0.27)
        self.assertEqual(jc.com, (0.02, 0.0, -0.07))


if __name__ == "__main__":
    unittest.main()
