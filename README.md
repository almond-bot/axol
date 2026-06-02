# Axol

<img src="assets/axol.png" width="400" alt="Axol dual-arm robot" />

Command-line interface and Python SDK for the Almond Axol dual-arm robot. CLI invoked as `axol <command> [flags]`.

The full documentation is authored as [Mintlify](https://mintlify.com) MDX under [`docs/`](docs/) and rendered as a docs site. The pages below link to the MDX sources.

**New here?** See the [Teleoperation quickstart](docs/quickstart/teleop.mdx) to go from installation to a live teleoperation session.

## Requirements

- **Linux**
- **Python 3.13+**
- **(Optional) NVIDIA Jetson** — if ZED cameras are used.

## Installation

Install the package using [`uv`](https://docs.astral.sh/uv/). `pyroki` and `lerobot` are sourced from Git and are resolved automatically:

```bash
uv sync
```

Then activate the virtual environment so the `axol` CLI is on your path (or prefix every command with `uv run`):

```bash
source .venv/bin/activate
```

Install optional dependency groups as needed:

| Extra | Contents | When to use |
|---|---|---|
| `lerobot` | LeRobot (from GitHub) | `collect-data`, `run-policy` |
| `sim` | viser | `teleop --sim` |
| `cuda` | JAX with CUDA 13 support | GPU-accelerated JAX (IK solver used by `teleop`); note that CPU is usually faster for the JAX IK solver |
| `dev` | OpenCV (headless) | Development / debugging |

```bash
uv sync --extra lerobot --extra sim        # teleoperation + data collection
uv sync --extra lerobot --extra cuda       # policy execution on GPU
uv sync --extra lerobot --extra sim --extra cuda   # everything
```

The ZED Python bindings (`pyzed`) are not on PyPI and must be installed separately after the ZED SDK is installed:

```bash
axol zed.install
```

Before using any motor or robot commands, initialize the CAN hardware:

```bash
axol can.setup
```

See [`docs/installation.mdx`](docs/installation.mdx) for the full installation guide.

## Sitemap

### Get Started

- [Overview](docs/index.mdx)
- [Installation](docs/installation.mdx)

### Quickstart

- [Teleoperation](docs/quickstart/teleop.mdx)
- [Data Collection](docs/quickstart/data-collection.mdx) — two-machine workflow (main host + ZED box)
- [Policy Inference](docs/quickstart/inference.mdx) — two-machine workflow (main host + ZED box)

### CLI Reference

- [`can.setup`](docs/cli/can-setup.mdx)
- [`can.enable`](docs/cli/can-enable.mdx)
- [`motor.info`](docs/cli/motor-info.mdx)
- [`motor.set-can-id`](docs/cli/motor-set-can-id.mdx)
- [`motor.set-zero-pos`](docs/cli/motor-set-zero-pos.mdx)
- [`teleop`](docs/cli/teleop.mdx)
- [`collect-data`](docs/cli/collect-data.mdx)
- [`run-policy`](docs/cli/run-policy.mdx)
- [`zed.stream`](docs/cli/zed-stream.mdx)
- [`zed.install`](docs/cli/zed-install.mdx)
- [`zed.sync-clocks`](docs/cli/zed-sync-clocks.mdx)
- [`tune.pid`](docs/cli/tune-pid.mdx)
- [`tune.friction`](docs/cli/tune-friction.mdx)
- [`tune.repeatability`](docs/cli/tune-repeatability.mdx)
- [`gravity-comp`](docs/cli/gravity-comp.mdx)

### Python API

- [Core Concepts](docs/api/concepts.mdx)
- [`almond_axol.robot`](docs/api/robot.mdx) — `Axol`, `Sim`, configuration, gravity compensation
- [`almond_axol.kinematics`](docs/api/kinematics.mdx)
- [`almond_axol.teleop`](docs/api/teleop.mdx)
- [`almond_axol.vr`](docs/api/vr.mdx)
- [`almond_axol.zed`](docs/api/zed.mdx)
- [`almond_axol.motor`](docs/api/motor.mdx)
- [`almond_axol.lerobot`](docs/api/lerobot.mdx)

## Previewing the docs

The docs are a Mintlify project rooted at [`docs/`](docs/). To preview locally:

```bash
npm i -g mint
cd docs && mint dev
```
