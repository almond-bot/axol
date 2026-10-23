# Independent Mink numerical reference

`reference-source.tar.gz` contains a frozen Axol numerical implementation used
only to reproduce reference expectations on each CPU architecture. Its source
revision is recorded as `reference_commit` in `provenance.json`.

The archive contains six numerical modules under `mink_reference/`, the pinned
model and meshes, and their licenses. Documentation and package paths have been
normalized, the model filename is `assets/axol_mink.urdf`, and the conversion
helper is reduced to its original `aa2rotm` function. These are source-packaging
changes: numerical expressions and model geometry are preserved. The reference
was prepared independently of the production implementation being tested.

Archive SHA-256:
`a3a5095000f6be84db92efc880cbe7f723f4cc0d3710bf145cfac0c1bada2642`.

`provenance.json` pins the reference archive and every reference source/asset
hash. Production source hashes are recorded separately because its documentation
and public API names differ. Before importing the reference under an isolated
namespace, the replay tool verifies its hashes, numerical dependency versions,
and shared joint/body naming tables. No external source checkout is required.

`stream.npz` retains the original numerical arrays byte-for-byte: 360 tracking
steps including distant targets and a reset, independent FK samples, NumPy pose
conversions, and public-world pose vectors. Native replay uses those exact input
arrays and obtains all expected outputs from the independent reference. The
world-frame vectors exercise decoding and the fixed model-root transform; they
make no claim of application-specific protocol equivalence.

To generate native expectations from the committed inputs:

```bash
mkdir -p /tmp/mink-reference-source
tar -xzf tests/data/mink_ik_reference/reference-source.tar.gz \
  -C /tmp/mink-reference-source
PYTHONPATH=. python tests/tools/mink_ik_reference.py \
  --reference-root /tmp/mink-reference-source/mink_reference \
  --replay-fixture --write-fixture /tmp/mink-native
AXOL_MINK_REFERENCE_FIXTURE=/tmp/mink-native python -m pytest \
  tests/test_mink_ik_parity.py
```

The reference solver and URDF retain the repository's MIT license
(Copyright 2026 Almond AI, Inc.). The axis-angle conversion helper retains
Copyright 2026 Xiaomi Corporation and its Apache-2.0 license, bundled as
`LICENSE-Xiaomi-Robotics-1.txt`. The gripper meshes retain
`assets/meshes/LICENSE-yam_linear4310.txt` and their original attribution.

Archive member names are sorted; file ownership, timestamps and the gzip
filename are zeroed. Do not regenerate the reference from the production
controller: a numerical reference update requires a separate source revision
and explicit parity review.
