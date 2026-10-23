"""Exported IK programs (almond_axol.kinematics.aot): reuse without re-tracing.

A graph shape is exported on first use; a fresh wrapper (a new process, in
effect) loads that export instead of tracing, gives the same answers, and
never reuses a program traced for different static arguments, input shapes
or robot model.
"""

from __future__ import annotations

import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from almond_axol.kinematics import aot


@functools.partial(jax.jit, static_argnames=("iterations",))
def _toy(a, b, scale, iterations, extra=None):
    # A Cholesky solve, like the IK's dense_cholesky step (LAPACK kernels).
    m = a @ a.T + jnp.eye(a.shape[0])
    x = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(m), b)
    for _ in range(iterations):
        x = x * scale
    return x if extra is None else x + extra


class ExportedJitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        patcher = patch.object(aot, "EXPORT_DIR", Path(self._dir.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._dir.cleanup)
        rng = np.random.default_rng(0)
        self.a = jnp.asarray(rng.normal(size=(3, 3)), jnp.float32)
        self.b = jnp.asarray(rng.normal(size=3), jnp.float32)

    def _exports(self) -> list[str]:
        return sorted(p.name for p in Path(self._dir.name).glob("*.jaxexport"))

    def test_exports_once_then_reuses_with_the_same_answer(self) -> None:
        want = _toy(self.a, self.b, 0.5, iterations=2)
        first = aot.ExportedJit(_toy, "toy", static_argnames=("iterations",))
        np.testing.assert_allclose(
            first(self.a, self.b, 0.5, iterations=2), want, rtol=1e-6
        )
        self.assertEqual(len(self._exports()), 1)

        # A new wrapper — a new process — loads the export; tracing it again
        # would mean the export is not being used.
        second = aot.ExportedJit(_toy, "toy", static_argnames=("iterations",))
        with patch.object(
            aot.ExportedJit, "_plain", side_effect=AssertionError("traced again")
        ):
            got = second(self.a, self.b, 0.5, iterations=2)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        self.assertEqual(len(self._exports()), 1)

    def test_a_different_program_never_reuses_an_export(self) -> None:
        f = aot.ExportedJit(_toy, "toy", static_argnames=("iterations",))
        f(self.a, self.b, 0.5, iterations=2)
        f(self.a, self.b, 0.5, iterations=3)  # another static argument
        f(self.a, self.b, 0.5, iterations=2, extra=self.b)  # another structure
        f(self.a[:2, :2], self.b[:2], 0.5, iterations=2)  # other shapes
        f(self.a, self.b, 0.5, iterations=2, _fingerprint="another robot")
        self.assertEqual(len(self._exports()), 5)

    def test_turned_off_it_just_calls_the_function(self) -> None:
        with patch.dict("os.environ", {"ALMOND_IK_EXPORTS": "0"}):
            f = aot.ExportedJit(_toy, "toy", static_argnames=("iterations",))
            f(self.a, self.b, 0.5, iterations=2)
        self.assertEqual(self._exports(), [])


if __name__ == "__main__":
    unittest.main()
