"""Real Mink stream/FK parity against recorded reference solver output.

The fixture was generated from the independent reference source by
tests/tools/mink_ik_reference.py. These tests need no user cache, robot,
camera, or JAX runtime. Source/asset hashes prevent a self-consistent model
change from silently changing the model's coordinate frame.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest

from almond_axol.policy import mink_ik
from almond_axol.policy.mink_ik import MinkIK, MinkIKConfig, pose6_to_pos_rot_np
from tests.tools.mink_ik_reference import (
    COMMITTED_FIXTURES,
    load_fixture,
    runtime_platform,
)

FIXTURES = COMMITTED_FIXTURES


@pytest.fixture(scope="module")
def golden():
    native = os.environ.get("AXOL_MINK_REFERENCE_FIXTURE")
    arrays, metadata = load_fixture(
        Path(native) if native else FIXTURES, require_native=bool(native)
    )
    if not native and metadata["runtime_platform"] != runtime_platform():
        pytest.fail(
            "The committed Mink expectations were generated on "
            f"{metadata['runtime_platform']}. Generate a native fixture with "
            "tests/tools/mink_ik_reference.py --replay-fixture and set "
            "AXOL_MINK_REFERENCE_FIXTURE to its directory; see mink_ik/PROVENANCE.md."
        )
    return arrays


@pytest.fixture(scope="module")
def provenance():
    return json.loads((FIXTURES / "provenance.json").read_text())


def test_fixture_and_copied_solver_assets_match_reference_provenance(provenance):
    assert provenance["reference_commit"] == "b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b"
    assert (
        hashlib.sha256((FIXTURES / "stream.npz").read_bytes()).hexdigest()
        == provenance["fixture_sha256"]
    )
    package = Path(mink_ik.__file__).parent
    for name, digest in provenance["production_source_sha256"].items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name


def test_solver_config_and_numerical_runtime_match_reference(provenance):
    assert asdict(MinkIKConfig()) == provenance["config"]
    assert mujoco.__version__ == "3.11.0"
    assert importlib.metadata.version("mink") == "1.2.0"


def test_real_solver_matches_reference_stream_far_targets_and_reset_exactly(golden):
    solver = MinkIK()
    rest = golden["rest14"]
    solver.set_rest_posture(rest)
    solver.reset_tracking_state()
    joints = rest.copy()
    assert len(golden["expected_joints"]) == 360
    for tick in range(len(golden["expected_joints"])):
        if golden["reset_before"][tick]:
            joints = rest.copy()
            solver.reset_tracking_state()
        previous = joints.copy()
        joints = solver.solve(
            joints,
            (golden["left_pos"][tick], golden["left_rot"][tick]),
            (golden["right_pos"][tick], golden["right_rot"][tick]),
        )
        np.testing.assert_array_equal(
            joints, golden["expected_joints"][tick], err_msg=f"tick {tick}"
        )
        assert solver.fail_count == golden["expected_fail_count"][tick]
        assert np.max(np.abs(joints - previous)) <= solver.per_call_step_bound + 1e-7


def test_world_frame_stream_matches_independent_reference_exactly(golden):
    from almond_axol.policy.mink_ik import pose6_world_to_model

    solver = MinkIK()
    rest = golden["rest14"]
    solver.set_rest_posture(rest)
    solver.reset_tracking_state()
    joints = rest.copy()
    for tick, poses in enumerate(golden["current_wire_pose6"]):
        if golden["reset_before"][tick]:
            joints = rest.copy()
            solver.reset_tracking_state()
        targets = [pose6_world_to_model(pose) for pose in poses]
        for side, (position, rotation) in enumerate(targets):
            np.testing.assert_array_equal(
                position, golden["current_wire_positions"][tick, side]
            )
            np.testing.assert_array_equal(
                rotation, golden["current_wire_rotations"][tick, side]
            )
        joints = solver.solve(joints, *targets)
        np.testing.assert_array_equal(
            joints,
            golden["current_wire_joints"][tick],
            err_msg=f"world-frame tick {tick}",
        )
    assert solver.fail_count == 0


def test_native_model_fk_matches_independent_model_frame(golden):
    solver = MinkIK()
    assert solver.model.nq == 18
    assert solver.tracker._model is solver.model
    assert (
        mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_tcp") >= 0
    )
    np.testing.assert_allclose(
        solver.shoulder_positions["left"], [0.13, 0.0, 0.86], atol=1e-6
    )
    np.testing.assert_allclose(
        solver.shoulder_positions["right"], [-0.13, 0.0, 0.86], atol=1e-6
    )
    for sample, joints in enumerate(golden["fk_joints"]):
        for side, (position, rotation) in enumerate(solver.fk(joints)):
            np.testing.assert_allclose(
                position, golden["fk_positions"][sample, side], rtol=0, atol=1e-5
            )
            np.testing.assert_allclose(
                rotation, golden["fk_rotations"][sample, side], rtol=0, atol=1e-4
            )


def test_pose_conversion_matches_reference_at_zero_pi_and_seeded_rotations(golden):
    for sample, pose in enumerate(golden["pose6"]):
        position, rotation = pose6_to_pos_rot_np(pose)
        np.testing.assert_array_equal(position, golden["pose_positions"][sample])
        np.testing.assert_array_equal(rotation, golden["pose_rotations"][sample])
        assert position.dtype == np.float32
        assert rotation.dtype == np.float32


def test_alternate_model_is_rejected_before_construction(tmp_path):
    other = tmp_path / "different-model.urdf"
    other.write_text(mink_ik.PINNED_URDF.read_text())
    with pytest.raises(ValueError, match="requires the bundled model"):
        MinkIK(urdf_path=other)


def test_import_and_real_warmup_work_without_jax_or_hardware_modules():
    code = """
import importlib.abc
import sys
# qpsolvers probes all optional backends, including JAX, when available.
# Simulate a base installation without those optional/hardware dependencies.
prefixes = ('jax', 'jaxlib', 'jaxlie', 'pyroki', 'pyzed', 'almond_axol.rt', 'almond_axol.lerobot')
class UnavailableOptionalDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in prefixes):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, UnavailableOptionalDependency())
from almond_axol.policy.mink_ik import MinkIK
solver = MinkIK()
assert solver.fail_count == 0
for prefix in prefixes:
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
