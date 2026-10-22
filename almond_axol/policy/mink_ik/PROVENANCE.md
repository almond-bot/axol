# Mink IK

This package provides Axol tracking with Mink, MuJoCo and DAQP. It exposes
`MinkIK`, `MinkIKConfig`, `pose6_to_pos_rot_np` and `pose6_world_to_model`.
The solver owns mutable MuJoCo data and belongs to one control thread.

The supported numerical runtime uses `mujoco==3.11.0`, `mink==1.2.0` and
DAQP through qpsolvers. The reference fixture records the exact NumPy,
DAQP and qpsolvers versions used to generate its expectations. Different
MuJoCo versions can produce different collision constraints even when the
same model imports successfully.

Mink execution does not require JAX, cameras or CAN. qpsolvers may probe an
installed optional backend during discovery; the Mink-only CI environment
verifies imports, warmup and execution with JAX packages unavailable.

## Model and attribution

The bundled `assets/axol_mink.urdf` and meshes define the model used for FK,
reach limits and collision constraints. Its root frame differs from Axol
world coordinates by a 90-degree yaw. `pose6_world_to_model` converts
world-frame Cartesian actions into model-frame targets after NumPy pose
decoding. The teleop adapter applies the inverse transform to FK results.
See `assets/README.md` for the asset convention and mesh license.

The axis-angle conversion retains the float32 coercion and expression
order required by the numerical reference. Its third-party copyright,
source attribution and Apache 2.0 license are retained in
`NOTICE-Xiaomi-Robotics-1.txt` and `LICENSE-Xiaomi-Robotics-1.txt`.

## Numerical reference

`tests/data/mink_ik_reference/provenance.json` records independent reference
source hashes, production source hashes, model assets and runtime versions.
`stream.npz` contains a 30 Hz two-arm target stream, unreachable targets,
tracking resets, FK samples and pose-conversion inputs. The solver must
match all 360 recurrent joint solutions exactly on the same architecture;
FK checks retain their explicit geometric tolerances.

`reference-source.tar.gz` contains an independent numerical implementation
under `mink_reference/`, with its complete model assets. The generator in
`tests/tools/mink_ik_reference.py` verifies those source and asset hashes
before importing them. The reference is not regenerated from the production
implementation, downloaded at test time or replaced by a fallback solver.

MuJoCo, DAQP and NumPy do not promise identical floating-point results
across CPU architectures. CI therefore replays the committed inputs through
the independent implementation on the runner, then compares the production
solver with those native expectations. It validates the input arrays and
dtypes, source and asset hashes, configuration, dependencies and platform.
`AXOL_MINK_REFERENCE_FIXTURE` selects the generated expectations for pytest.
CI publishes that fixture and its provenance as an artifact.

To generate native expectations, run from the Axol root using destinations
outside this repository:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export AXOL_MINK_REFERENCE_FIXTURE=/tmp/axol-mink-native
mkdir -p /tmp/axol-mink-reference
tar -xzf tests/data/mink_ik_reference/reference-source.tar.gz \
  -C /tmp/axol-mink-reference
PYTHONPATH=. .venv/bin/python tests/tools/mink_ik_reference.py \
  --reference-root /tmp/axol-mink-reference/mink_reference \
  --replay-fixture --write-fixture "$AXOL_MINK_REFERENCE_FIXTURE"
.venv/bin/python -m pytest tests/test_mink_ik_parity.py
```

Linux x86_64 tests can use the committed fixture directly. Other platforms
require an explicit native fixture; incompatible or missing provenance
fails instead of weakening or skipping the comparison.

## Frame conversion and rounding

Exact reference replay compares the same decoded targets. It does not
promise bitwise-identical joint trajectories when poses are first encoded
in different coordinate frames. Float32 axis-angle conversion introduces
small rotation differences, and collision-constrained iterations can
amplify those differences. Keep the world-frame convention and numerical
precision consistent across the policy interface. These numerical checks
complement physical tracking tests; they do not model the robot plant or
actuator filtering.
