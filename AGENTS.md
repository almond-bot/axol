# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Almond Axol is a Python CLI + SDK for the Almond Axol dual-arm robot. Since no physical robot hardware is available in the cloud VM, all development and testing uses the **sim** mode (`--robot sim`), which renders the robot in a browser via viser.

### Running the application

- **Sim teleop** (the primary way to exercise the app without hardware): `uv run axol teleop --robot sim`
  - Opens a viser 3D viewer at `http://localhost:8080` and a VR WebSocket server on port 8000.
- The `Sim` class does not implement `__aenter__`/`__aexit__`; call `await sim.enable()` / `await sim.disable()` directly (despite the README showing `async with Sim()`).

### Linting

- `ruff check .` and `ruff format --check .` — ruff is configured in `.pre-commit-config.yaml` (v0.9.7), not as a project dependency. Install via `uv tool install ruff@0.9.7`.

### Testing

- No automated test suite exists in this repository. Validate changes by importing the package and exercising the `Sim`-based code paths.

### Dependency extras

| Extra | Purpose |
|-------|---------|
| `sim` | viser (browser 3D visualizer) — needed for sim mode |
| `dev` | opencv-python-headless — dev/debugging |
| `lerobot` | LeRobot data collection/policy — requires hardware + ZED cameras |
| `cuda` | JAX with CUDA — requires GPU |

For cloud development: `uv sync --extra sim --extra dev` is sufficient.

### Gotchas

- Python 3.13+ is required (`.python-version` pins `3.13`). The VM ships with 3.12; use `uv python install 3.13` if needed.
- The `uv` package manager must be on PATH (`$HOME/.local/bin`).
- Hardware-dependent commands (`can.setup`, `motor.*`, `gravity-comp`, `tune.*`, `zed.*`, `collect-data`, `run-policy`) will fail without physical robot/CAN bus — this is expected.
