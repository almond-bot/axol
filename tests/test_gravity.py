from __future__ import annotations

import re
import unittest

import mujoco
import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.gravity import GravityCompensator


class MujocoPinTest(unittest.TestCase):
    """gravity.py targets the MuJoCo 3.8/3.9 API (``mj_fullM(m, dst, qM)``).

    3.10 changed that signature and 3.11 removed ``mjData.qM``; the
    ``mujoco<3.10`` bound in pyproject.toml is what keeps tool/pip installs
    (which re-resolve instead of reading uv.lock) on a working version.
    """

    def test_installed_mujoco_is_within_the_supported_line(self) -> None:
        major, minor = (
            int(p) for p in re.match(r"(\d+)\.(\d+)", mujoco.__version__).groups()
        )
        self.assertEqual(major, 3)
        self.assertGreaterEqual(minor, 8)
        self.assertLess(minor, 10, "bump gravity.py's mj_fullM call with the pin")


class InertiaTest(unittest.TestCase):
    def test_dense_inertia_is_symmetric_positive_definite(self) -> None:
        comp = GravityCompensator()
        q = np.linspace(-0.4, 0.6, len(ARM_JOINTS))
        gravity, inertia = comp.gravity_and_inertia_arm(q, is_left=True)

        self.assertEqual(gravity.shape, (len(ARM_JOINTS),))
        self.assertEqual(inertia.shape, (len(ARM_JOINTS),))
        self.assertTrue(np.all(inertia > 0.0))
        m = comp._m_full
        self.assertTrue(np.allclose(m, m.T, atol=1e-9))
        self.assertTrue(np.all(np.linalg.eigvalsh(m) > 0.0))
        self.assertTrue(np.all(np.isfinite(gravity)))


if __name__ == "__main__":
    unittest.main()
