# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Almond Axol is a Python CLI + SDK for the Almond Axol dual-arm robot. Since no physical robot hardware is available in the cloud VM, all development and testing uses the **sim** mode (`--sim`), which renders the robot in a browser via viser.

### Running the application

- **Sim teleop** (the primary way to exercise the app without hardware): `uv run axol teleop --sim`
 - Opens a viser 3D viewer at `http://localhost:8002` and a VR WebSocket server on port 8000.
 - With no VR headset connected the arms just hold the rest pose. To actually drive them, either use the `Sim` SDK directly (`sim.motion_control(left=..., right=...)`, see the `Sim` docstring in `almond_axol/robot/sim.py`) or connect a WebSocket client to `wss://localhost:8000/ws` (self-signed cert — disable TLS verification) and stream `VRFrame` JSON with both `l_lock`/`r_lock` true to engage tracking.
 - The viser server persists engage/IK state across teleop restarts only within one process; if a WebSocket client leaves tracking engaged and reconnects, restart the `teleop` process for a clean engage.

### Web front-end (second product)

The browser UIs live under `web/` (a Vite + React monorepo: the WebXR `/vr` teleop app and the `/control` panel served by `axol serve`). Node 22 is available; `web/` is **not** covered by `uv sync`. Standard install/build/dev commands are in `web/README.md` (build the `packages/axol-vr-client` workspace before `app`).

### Linting

- `ruff check .` and `ruff format --check .` — ruff is not a project dependency; it's pinned in `.pre-commit-config.yaml` (see the `rev:` field). Easiest: `uv tool install pre-commit && pre-commit run --all-files`, which uses the pinned version automatically. Or install ruff directly at the same version, e.g. `uv tool install ruff@0.9.7`.

### Testing

- No automated test suite exists in this repository. Validate changes by importing the package and exercising the `Sim`-based code paths.

### Dependency extras

| Extra | Purpose |
|-------|---------|
| `sim` | viser (browser 3D visualizer) — needed for sim mode |
| `lerobot` | LeRobot data collection/policy — requires hardware + ZED cameras |

For cloud development: `uv sync --extra sim` is sufficient.

### Gotchas

- Python 3.13+ is required (`.python-version` pins `3.13`). The VM ships with 3.12; use `uv python install 3.13` if needed.
- The `uv` package manager must be on PATH (`$HOME/.local/bin`).
- Hardware-dependent commands (`can.setup`, `motor.*`, `gravity-comp`, `tune.*`, `zed.*`, `collect-data`, `run-policy`) will fail without physical robot/CAN bus — this is expected.
