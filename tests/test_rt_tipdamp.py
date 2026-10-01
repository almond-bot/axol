"""The realtime core's tip damper, Python side: the arm's product-of-
exponentials FK against the solver, the config lines the core parses, and
the rt robot sending them only once the joint offsets are resolved."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from almond_axol.robot.config import AxolConfig, TipDampConfig
from almond_axol.rt.tipdamp import (
    REFERENCES,
    chain_height,
    check_chain,
    config_lines,
    poe_chain,
)
from almond_axol.tuning.tracking_model import TrackingModel

from .test_controller_option import _hardware


def _chain() -> dict[str, np.ndarray]:
    w = np.tile([0.0, -1.0, 0.0], (7, 1))
    m = np.eye(4)
    m[0, 3] = 0.5
    return {"w": w, "r": np.zeros((7, 3)), "m": m}


class ChainTest(unittest.TestCase):
    def test_the_product_of_exponentials_matches_the_solver(self) -> None:
        from almond_axol.kinematics.solver import KinematicsSolver

        solver = KinematicsSolver()
        for side in ("left", "right"):
            chain = poe_chain(solver, side)
            self.assertLess(check_chain(solver, side, chain, n=10), 2e-5, side)

    def test_a_pitch_chain_lifts_the_mount(self) -> None:
        # Every axis along −y through the origin, mount 0.5 m out along x:
        # +q raises it (right-hand rule about −y).
        self.assertAlmostEqual(chain_height(_chain(), np.zeros(7)), 0.0)
        q = np.zeros(7)
        q[0] = 0.1
        self.assertAlmostEqual(chain_height(_chain(), q), 0.5 * math.sin(0.1))


class ConfigLinesTest(unittest.TestCase):
    def _cfg(self, **kw) -> TipDampConfig:
        return TipDampConfig(
            gain=120.0,
            joints={"shoulder_1": 1.0, "elbow": 0.6},
            joint_lp={"elbow": 6.0},
            reference="model",
            imu_port=47811,
            tracking_models={
                "shoulder_1": TrackingModel(1.0, 14.8, 0.6, 36.2, 0.0036, 0.3, 6, 0.07),
                "shoulder_2": TrackingModel(
                    1.0, 17.2, 0.55, math.inf, 0.007, 0.3, 6, 0.1
                ),
            },
            **kw,
        )

    def test_the_lines_carry_what_the_core_parses(self) -> None:
        lines = config_lines(1, "canR", self._cfg(), _chain(), np.arange(7) * 0.1)
        kinds = [ln.split()[0] for ln in lines]
        self.assertEqual(kinds, ["tipkin", "tipoff", "tipmodel", "tipmodel", "tipdamp"])
        kin = lines[0].split()
        self.assertEqual(len(kin), 3 + 58)
        self.assertEqual(kin[1:3], ["1", "canR"])
        self.assertEqual(
            [float(x) for x in lines[1].split()[3:]], list(np.arange(7) * 0.1)
        )
        s2 = lines[3].split()
        self.assertEqual(s2[3], "1")  # shoulder_2's arm slot
        self.assertEqual(float(s2[6]), 0.0)  # no zero → 0
        damp = lines[4].split()
        self.assertEqual(len(damp), 19 + 3 * 2)
        self.assertEqual(damp[3], "47811")
        self.assertEqual(int(damp[11]), REFERENCES["model"])
        self.assertEqual(float(damp[17]), 1.4)  # trip_hf_acc
        self.assertEqual(damp[18], "2")
        self.assertEqual(damp[19:], ["0", "1.0", "0.0", "3", "0.6", "6.0"])

    def test_an_unresolved_offset_and_a_bad_joint_are_refused(self) -> None:
        offsets = np.zeros(7)
        offsets[4] = math.nan
        with self.assertRaises(ValueError):
            config_lines(0, "canL", self._cfg(), _chain(), offsets)
        with self.assertRaises(ValueError):
            TipDampConfig(gain=1.0, joints={"knee": 1.0})
        with self.assertRaises(ValueError):
            TipDampConfig(gain=1.0, joints={"elbow": 1.0}, reference="imu")


class RobotConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("almond_axol.rt.link.find_binary", return_value="/fake/axol-rt")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tip_lines_go_on_the_resolved_configure_only(self) -> None:
        from almond_axol.rt import Axol

        config = AxolConfig()
        config.left.tip_damp = TipDampConfig(
            gain=100.0, joints={"shoulder_1": 1.0}, imu_port=47810, chain=_chain()
        )
        rt = Axol._wrap(_hardware(config))
        first = rt._config_text()
        self.assertNotIn("tipdamp", first)
        for arm in (rt._robot.left,):
            arm._joint_offsets[:] = 0.25
        second = rt._config_text(cogging=True).splitlines()
        tips = [ln for ln in second if ln.startswith("tip")]
        self.assertEqual(
            [ln.split()[0] for ln in tips], ["tipkin", "tipoff", "tipdamp"]
        )
        self.assertTrue(all(ln.split()[1] == "0" for ln in tips))
        self.assertEqual(tips[1].split()[3:], ["0.25"] * 7)


if __name__ == "__main__":
    unittest.main()
