"""Native parity fixtures must remain bound to the independent reference baseline."""

import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest

from tests.tools.mink_ik_reference import (
    COMMITTED_FIXTURES,
    load_fixture,
    load_reference,
    runtime_platform,
    validate_shared_name_tables,
)


def write_fixture(path, arrays, metadata):
    path.mkdir(exist_ok=True)
    archive = path / "stream.npz"
    np.savez_compressed(archive, **arrays)
    metadata["fixture_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (path / "provenance.json").write_text(json.dumps(metadata))


@pytest.fixture
def native_fixture(tmp_path):
    arrays, metadata = load_fixture(COMMITTED_FIXTURES)
    # These guard tests do not claim numerical native parity. Only the explicit
    # generator runs the verified independent solver to establish that result.
    metadata["input_fixture_sha256"] = metadata["fixture_sha256"]
    metadata["runtime_platform"] = runtime_platform()
    write_fixture(tmp_path, arrays, metadata)
    return tmp_path


def test_native_fixture_accepts_exact_inputs_and_matching_provenance(native_fixture):
    arrays, _ = load_fixture(native_fixture, require_native=True)
    assert arrays["expected_joints"].shape == (360, 14)


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    [
        ("reference_commit", "0" * 40, "reference commit"),
        ("fixture_sha256", "0" * 64, "archive hash"),
        ("input_fixture_sha256", "0" * 64, "committed inputs"),
        ("source_sha256", {}, "source_sha256"),
        ("production_source_sha256", {}, "production_source_sha256"),
        ("reference_archive_sha256", "0" * 64, "reference_archive_sha256"),
        ("config", {}, "config"),
        ("stream", {}, "stream"),
        ("runtime_versions", {}, "runtime_versions"),
        ("runtime_platform", {"system": "other", "machine": "other"}, "platform"),
    ],
)
def test_native_fixture_rejects_provenance_drift(
    native_fixture, field, changed, message
):
    path = native_fixture / "provenance.json"
    metadata = json.loads(path.read_text())
    metadata[field] = changed
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        load_fixture(native_fixture, require_native=True)


@pytest.mark.parametrize(
    "changed_input", ["left_pos", "reset_before", "fk_joints", "current_wire_pose6"]
)
def test_native_fixture_rejects_changed_inputs_even_with_valid_hash(
    native_fixture, changed_input
):
    arrays, metadata = load_fixture(native_fixture)
    if arrays[changed_input].dtype == bool:
        arrays[changed_input][0] = not arrays[changed_input][0]
    else:
        arrays[changed_input].flat[0] += 0.125
    write_fixture(native_fixture, arrays, metadata)
    with pytest.raises(ValueError, match=f"changed frozen input {changed_input}"):
        load_fixture(native_fixture, require_native=True)


def test_native_fixture_rejects_changed_input_dtype(native_fixture):
    arrays, metadata = load_fixture(native_fixture)
    arrays["pose6"] = arrays["pose6"].astype(np.float64)
    write_fixture(native_fixture, arrays, metadata)
    with pytest.raises(ValueError, match="changed frozen input pose6"):
        load_fixture(native_fixture, require_native=True)


def test_reference_sources_are_verified_before_import(tmp_path):
    _, metadata = load_fixture(COMMITTED_FIXTURES)
    for name in metadata["source_sha256"]:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("raise AssertionError('must not import unverified source')\n")
    with (
        patch("tests.tools.mink_ik_reference.importlib.import_module") as import_module,
        pytest.raises(ValueError, match="source/assets differ"),
    ):
        load_reference(tmp_path)
    import_module.assert_not_called()


@pytest.mark.parametrize("table", ["urdf_arm_joint_names", "urdf_body_name"])
def test_shared_sdk_name_drift_cannot_move_both_solvers(table):
    with (
        patch(f"almond_axol.constants.{table}", return_value="changed"),
        pytest.raises(ValueError, match="name table differs"),
    ):
        validate_shared_name_tables()
