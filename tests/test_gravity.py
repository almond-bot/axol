from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.gravity import GravityCompensator


class MassMatrixApiTest(unittest.TestCase):
    def test_mass_matrix_api_selection_covers_qm_removal(self) -> None:
        for version, uses_data in (
            ("3.8.1", False),
            ("3.9.0", False),
            ("3.10.0", True),
            ("3.11.0", True),
        ):
            with self.subTest(version=version):
                with mock.patch.object(mujoco, "__version__", version):
                    comp = GravityCompensator()
                sparse = object()
                # The newer API must not access the removed qM attribute.
                comp._data = (
                    SimpleNamespace() if uses_data else SimpleNamespace(qM=sparse)
                )
                gravity = np.arange(7, dtype=np.float32)
                with (
                    mock.patch.object(comp, "gravity_arm", return_value=gravity),
                    mock.patch.object(mujoco, "mj_fullM") as full,
                ):
                    actual_gravity, _ = comp.gravity_and_inertia_arm(
                        np.zeros(7), is_left=True
                    )
                args = full.call_args.args
                self.assertIs(args[0], comp._model)
                self.assertIs(args[1], comp._data if uses_data else comp._m_full)
                self.assertIs(args[2], comp._m_full if uses_data else sparse)
                self.assertIs(actual_gravity, gravity)


class InertiaTest(unittest.TestCase):
    def test_dense_mass_matrix_matches_mujoco_operator_for_both_arms(self) -> None:
        comp = GravityCompensator()
        fixtures = (
            np.zeros(7),
            np.array([-0.8708, 0, 0, 1.395, 0, 0, 0.3442]),
            np.linspace(-0.4, 0.6, 7),
        )
        for is_left in (True, False):
            for q in fixtures:
                with self.subTest(is_left=is_left, q=q.tolist()):
                    gravity, inertia = comp.gravity_and_inertia_arm(q, is_left=is_left)
                    np.testing.assert_allclose(
                        gravity, comp.gravity_arm(q, is_left=is_left), atol=1e-7
                    )
                    indices = comp._left_dof_idx if is_left else comp._right_dof_idx
                    for joint, dof in enumerate(indices):
                        basis = np.zeros(comp._model.nv)
                        basis[dof] = 1.0
                        product = np.zeros_like(basis)
                        mujoco.mj_mulM(comp._model, comp._data, product, basis)
                        np.testing.assert_allclose(
                            comp._m_full[:, dof], product, rtol=1e-10, atol=1e-12
                        )
                        self.assertAlmostEqual(inertia[joint], product[dof], places=7)

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
