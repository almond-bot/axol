# Almond Axol SDK

<img src="assets/axol.png" width="400" alt="Axol dual-arm robot" />

Command-line interface and Python SDK for the Almond Axol dual-arm robot. CLI invoked as `axol <command> [flags]`.

The browser front-ends live under [`web/`](web/): a **VR teleoperation interface** (WebXR, hosted at [axol.almond.bot](https://axol.almond.bot)) and a **web control panel** that drives the robot from a browser via `axol serve`. See [`web/README.md`](web/README.md) for the front-end details.

The full documentation is hosted at [docs.almond.bot](https://docs.almond.bot). The sources live under [`docs/`](docs/), and the pages below link to them.

**New here?** See the [Teleoperation quickstart](https://docs.almond.bot/quickstart/teleop) to go from installation to a live teleoperation session, or the [Web Control Panel guide](https://docs.almond.bot/guides/control-panel) to drive Axol from a browser.

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

To drive Axol from a browser instead of the terminal, build the web UI once (it's served by `axol serve`):

```bash
cd web
npm install
npm run build --workspace=packages/axol-vr-client   # client package first
npm run build --workspace=app                        # → web/app/dist
```

See the [installation guide](https://docs.almond.bot/installation) for the full walkthrough.

## Sitemap

### Get Started

- [Overview](https://docs.almond.bot)
- [Installation](https://docs.almond.bot/installation)

### Quickstart

- [Teleoperation](https://docs.almond.bot/quickstart/teleop)
- [Data Collection](https://docs.almond.bot/quickstart/data-collection) — two-machine workflow (main host + ZED box)
- [Policy Inference](https://docs.almond.bot/quickstart/inference) — two-machine workflow (main host + ZED box)

### Web Interfaces

- [Web Control Panel](https://docs.almond.bot/guides/control-panel) — drive the robot from a browser via `axol serve`
- [VR Interface](https://docs.almond.bot/guides/vr-interface) — the in-repo WebXR teleop app (`web/`)

### CLI Reference

- [Command configuration](https://docs.almond.bot/cli/configuration) — draccus config model for `teleop`, `gravity-comp`, `collect-data`, `run-policy`
- [`serve`](https://docs.almond.bot/cli/serve) — web control panel + API server
- [`can.setup`](https://docs.almond.bot/cli/can-setup)
- [`can.enable`](https://docs.almond.bot/cli/can-enable)
- [`motor.info`](https://docs.almond.bot/cli/motor-info)
- [`motor.set-can-id`](https://docs.almond.bot/cli/motor-set-can-id)
- [`motor.set-zero-pos`](https://docs.almond.bot/cli/motor-set-zero-pos)
- [`teleop`](https://docs.almond.bot/cli/teleop)
- [`collect-data`](https://docs.almond.bot/cli/collect-data)
- [`run-policy`](https://docs.almond.bot/cli/run-policy)
- [`zed.stream`](https://docs.almond.bot/cli/zed-stream)
- [`zed.install`](https://docs.almond.bot/cli/zed-install)
- [`zed.sync-clocks`](https://docs.almond.bot/cli/zed-sync-clocks)
- [`tune.pid`](https://docs.almond.bot/cli/tune-pid)
- [`tune.friction`](https://docs.almond.bot/cli/tune-friction)
- [`tune.repeatability`](https://docs.almond.bot/cli/tune-repeatability)
- [`gravity-comp`](https://docs.almond.bot/cli/gravity-comp)

### Python API

- [Core Concepts](https://docs.almond.bot/api/concepts)
- [`almond_axol.robot`](https://docs.almond.bot/api/robot) — `Axol`, `Sim`, configuration, gravity compensation
- [`almond_axol.kinematics`](https://docs.almond.bot/api/kinematics)
- [`almond_axol.teleop`](https://docs.almond.bot/api/teleop)
- [`almond_axol.vr`](https://docs.almond.bot/api/vr)
- [`almond_axol.zed`](https://docs.almond.bot/api/zed)
- [`almond_axol.motor`](https://docs.almond.bot/api/motor)
- [`almond_axol.lerobot`](https://docs.almond.bot/api/lerobot)
