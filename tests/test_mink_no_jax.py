"""Exercise public Mink imports in a process that cannot load the JAX stack.

CI also runs this file in an installation without any JAX distributions. The
import blocker makes the same checks meaningful in a developer's full env.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BLOCK_JAX = """
import importlib.abc
import sys

BLOCKED = {"jax", "jaxlib", "jaxlie", "jaxls", "pyroki", "jax_dataclasses"}

class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"JAX is unavailable: {fullname}", name=fullname)

sys.meta_path.insert(0, NoJax())
"""


def _run_without_jax(source: str) -> None:
    code = _BLOCK_JAX + textwrap.dedent(source)
    code += "\nassert not (BLOCKED & {n.split('.')[0] for n in sys.modules})\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mink_ci_environment_has_no_jax_distributions() -> None:
    if os.environ.get("AXOL_EXPECT_NO_JAX") != "1":
        pytest.skip("dedicated Mink-only installation check")
    for package in ("jax", "jaxlib", "jaxlie", "almond-jaxls", "almond-pyroki"):
        with pytest.raises(importlib.metadata.PackageNotFoundError):
            importlib.metadata.distribution(package)


@pytest.mark.parametrize(
    "module",
    [
        "almond_axol.cli",
        "almond_axol.cli.teleop",
        "almond_axol.cli.collect_data",
        "almond_axol.cli.collect_dagger",
        "almond_axol.cli.run_policy",
        "almond_axol.teleop.worker",
    ],
)
def test_public_command_imports_without_jax(module: str) -> None:
    _run_without_jax(f"import importlib\nimportlib.import_module({module!r})\n")


def test_mink_kinematics_and_cartesian_observations_without_jax() -> None:
    _run_without_jax(
        """
        import numpy as np
        from almond_axol.kinematics import KinematicsConfig, Pose
        from almond_axol.kinematics.mink_backend import MinkKinematicsSolver
        from almond_axol.kinematics.mujoco_fk import AxolForwardKinematics

        solver = MinkKinematicsSolver(KinematicsConfig(backend="mink", elbow_weight=0))
        q = np.zeros(14, dtype=np.float32)
        left, right = solver.fk(q)
        assert left[0].shape == right[0].shape == (3,)
        assert left[1].shape == right[1].shape == (3, 3)
        observation_fk = AxolForwardKinematics()
        poses = observation_fk.ee_poses(q[:7], q[7:])
        assert all(pose.shape == (6,) and np.isfinite(pose).all() for pose in poses)
        """
    )
